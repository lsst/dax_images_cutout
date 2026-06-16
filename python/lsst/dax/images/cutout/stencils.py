# This file is part of dax_images_cutout.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

__all__ = (
    "MaskBackend",
    "PixelStencil",
    "SkyCircle",
    "SkyPolygon",
    "SkyStencil",
    "StencilNotContainedError",
)

import enum
import struct
from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableMapping
from hashlib import blake2b
from typing import Any

import astropy.coordinates
import astropy.units as u
import numpy as np
import starlink.Ast as Ast
from astropy.coordinates import SkyCoord

import lsst.sphgeom
from lsst.afw.geom import SkyWcs
from lsst.afw.image import Mask
from lsst.daf.base import PropertyList
from lsst.geom import Angle, Box2I, SpherePoint, radians
from lsst.images import Box, GeneralFrame, NoOverlapError, SkyProjection
from lsst.images.utils import round_half_down, round_half_up

# Generic pixel coordinate frame used when converting a legacy ``SkyWcs``
# into a ``SkyProjection``.  The frame identity only affects AST domain
# labels; the unit must be pixels so ``SkyProjection`` accepts the transform.
_PIXEL_FRAME = GeneralFrame(unit=u.pix)


class MaskBackend(enum.Enum):
    """Selects the algorithm used to rasterize a stencil onto pixels."""

    AST = enum.auto()
    """Mask using starlink-pyast ``Region.mask`` on the true sky region."""

    SPHGEOM = enum.auto()
    """Mask by testing pixel centers against the ``lsst.sphgeom`` region."""


def _as_projection(wcs: SkyWcs | SkyProjection) -> SkyProjection:
    """Normalize a WCS argument to a `lsst.images.SkyProjection`."""
    if isinstance(wcs, SkyProjection):
        return wcs
    if isinstance(wcs, SkyWcs):
        return SkyProjection.from_legacy(wcs, _PIXEL_FRAME)
    raise TypeError(f"Expected SkyWcs or SkyProjection; got {type(wcs).__name__}.")


def _as_box(bbox: Box2I | Box) -> Box:
    """Normalize a bounding-box argument to a `lsst.images.Box`."""
    if isinstance(bbox, Box):
        return bbox
    if isinstance(bbox, Box2I):
        return Box.from_legacy(bbox)
    raise TypeError(f"Expected Box2I or lsst.images.Box; got {type(bbox).__name__}.")


def _round_box_from_bounds(x_min: float, x_max: float, y_min: float, y_max: float) -> Box:
    """Build an integer pixel `Box` from continuous coordinate bounds.

    Uses the same rounding convention as `lsst.images.Region.bbox`, so that
    pixels whose centers lie within the bounds are included.
    """
    return Box.factory[
        round_half_up(y_min) : round_half_down(y_max) + 1,
        round_half_up(x_min) : round_half_down(x_max) + 1,
    ]


class _AstLineSource:
    """Feeds AST-native text lines to a `starlink.Ast.Channel`."""

    def __init__(self, text: str) -> None:
        self._lines = text.splitlines()

    def astsource(self) -> str | None:
        return self._lines.pop(0) if self._lines else None


def _starlink_sky_to_pixel(projection: SkyProjection) -> Ast.Mapping:
    """Return the sky->pixel mapping of ``projection`` as a starlink-pyast
    Mapping.

    ``lsst.images`` may wrap AST with either astshim or starlink-pyast
    depending on the runtime environment.  The transform is serialized to
    AST's native text form via the public `~lsst.images.Transform.show`
    method and re-read with starlink-pyast, so that all region masking
    happens in starlink-pyast regardless of which wrapper ``lsst.images``
    uses internally.
    """
    return Ast.Channel(_AstLineSource(projection.sky_to_pixel_transform.show())).read()


class StencilNotContainedError(RuntimeError):
    """Exception that may be raised when a stencil is not with a desired
    bounding box.
    """


class PixelStencil(ABC):
    """An image cutout stencil defined in pixel coordinates."""

    @property
    @abstractmethod
    def bbox(self) -> Box:
        """Bounding box of this stencil, as a `lsst.images.Box`."""
        raise NotImplementedError()

    @abstractmethod
    def set_mask(self, mask: Mask, bits: int) -> None:
        """Set mask planes for pixels whose centers the stencil covers.

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Mask to modify in-place.
        bits : `int`
            Integer bitmask to bitwise-OR into pixels covered by the stencil.
        """
        raise NotImplementedError()


def _apply_boolean_to_mask(mask: Mask, box: Box, covered: np.ndarray, bits: int) -> None:
    """Bitwise-OR ``bits`` into ``mask`` for the `True` entries of ``covered``.

    ``covered`` must be a boolean array with shape ``box.shape``
    (``(ny, nx)``).
    """
    submask = mask[box.to_legacy()]
    submask.array[:, :] |= (bits * covered).astype(submask.array.dtype)


class _AstPixelRegion(PixelStencil):
    """Pixel-coordinate stencil backed by a starlink-pyast sky `Region`.

    The mask is computed by rasterizing the sky region directly through the
    sky-to-pixel mapping, so the true (great-circle) region boundary is tested
    at each pixel center without linearizing it into pixel-space chords.

    Parameters
    ----------
    sky_region : `starlink.Ast.Region`
        The stencil region expressed in an ICRS sky frame.
    sky_to_pixel : `starlink.Ast.Mapping`
        Mapping whose forward direction transforms sky coordinates to pixels,
        as required by ``Region.mask`` (region frame to grid).
    bbox : `lsst.images.Box`
        Bounding box the stencil is restricted to.
    """

    def __init__(self, sky_region: Ast.Region, sky_to_pixel: Ast.Mapping, bbox: Box) -> None:
        self._sky_region = sky_region
        self._sky_to_pixel = sky_to_pixel
        self._bbox = bbox

    @property
    def bbox(self) -> Box:
        # Docstring inherited.
        return self._bbox

    def set_mask(self, mask: Mask, bits: int) -> None:
        # Docstring inherited.
        scratch = np.zeros(self._bbox.shape, dtype=np.int64)
        self._sky_region.mask(
            self._sky_to_pixel,
            1,
            [self._bbox.x.min, self._bbox.y.min],
            [self._bbox.x.max, self._bbox.y.max],
            scratch,
            1,
        )
        _apply_boolean_to_mask(mask, self._bbox, scratch != 0, bits)


class _SphgeomPixelRegion(PixelStencil):
    """Pixel-coordinate stencil that tests pixel centers against a sphgeom
    region.

    Parameters
    ----------
    region : `lsst.sphgeom.Region`
        Sky region to test pixel centers against.
    projection : `lsst.images.SkyProjection`
        Mapping used to convert pixel centers to sky coordinates.
    bbox : `lsst.images.Box`
        Bounding box the stencil is restricted to.
    """

    def __init__(self, region: lsst.sphgeom.Region, projection: SkyProjection, bbox: Box) -> None:
        self._region = region
        self._projection = projection
        self._bbox = bbox

    @property
    def bbox(self) -> Box:
        # Docstring inherited.
        return self._bbox

    def set_mask(self, mask: Mask, bits: int) -> None:
        # Docstring inherited.
        grid = self._bbox.meshgrid()
        sky = self._projection.pixel_to_sky(x=grid.x.ravel(), y=grid.y.ravel())
        covered = self._region.contains(sky.ra.radian, sky.dec.radian).reshape(self._bbox.shape)
        _apply_boolean_to_mask(mask, self._bbox, covered, bits)


class SkyStencil(ABC):
    """An image cutout stencil defined in sky (ICRS) coordinates."""

    _clip: bool

    def to_pixels(
        self,
        wcs: SkyWcs | SkyProjection,
        bbox: Box2I | Box,
        *,
        backend: MaskBackend = MaskBackend.AST,
    ) -> PixelStencil:
        """Transform to a pixel-coordinate stencil.

        Parameters
        ----------
        wcs : `lsst.afw.geom.SkyWcs` or `lsst.images.SkyProjection`
            Mapping from sky coordinates to pixel coordinates.  A legacy
            ``SkyWcs`` is converted to a `SkyProjection` internally.
        bbox : `lsst.geom.Box2I` or `lsst.images.Box`
            Bounds that the returned stencil must lie within.
        backend : `MaskBackend`, optional
            Algorithm used to rasterize the stencil.  Defaults to
            `MaskBackend.AST`.

        Returns
        -------
        pixels : `PixelStencil`
            Pixel-coordinate stencil object.  `PixelStencil.bbox` is guaranteed
            to be contained by the given ``bbox``.

        Raises
        ------
        StencilNotContainedError
            Raised when ``clip`` is `False` and the pixel-coordinate stencil
            does not lie within ``bbox``.
        """
        projection = _as_projection(wcs)
        box = _as_box(bbox)
        tight = self._pixel_bbox(projection)
        final = self._resolve_box(tight, box)
        if backend is MaskBackend.AST:
            return _AstPixelRegion(self._ast_sky_region(), _starlink_sky_to_pixel(projection), final)
        if backend is MaskBackend.SPHGEOM:
            return _SphgeomPixelRegion(self.region, projection, final)
        raise ValueError(f"Unknown mask backend: {backend!r}.")

    def _pixel_bbox(self, projection: SkyProjection) -> Box:
        """Compute the tight pixel bounding box of this stencil.

        The boundary is sampled on the sky and transformed to pixels, so both
        mask backends share an identical bounding box.
        """
        xy = projection.sky_to_pixel(self._boundary_skycoord())
        return _round_box_from_bounds(
            float(np.min(xy.x)), float(np.max(xy.x)), float(np.min(xy.y)), float(np.max(xy.y))
        )

    def _resolve_box(self, tight: Box, box: Box) -> Box:
        """Clip ``tight`` to ``box`` or raise if not contained.

        Honors the stencil's ``clip`` flag: when clipping, returns the
        intersection (raising `StencilNotContainedError` when disjoint); when
        not clipping, returns ``tight`` only if ``box`` contains it.
        """
        if self._clip:
            try:
                return tight.intersection(box)
            except NoOverlapError:
                raise StencilNotContainedError(f"{self} does not overlap {box}.") from None
        if not box.contains(tight):
            raise StencilNotContainedError(f"{self} has pixel bbox {tight}, which is not within {box}.")
        return tight

    @abstractmethod
    def _ast_sky_region(self) -> Ast.Region:
        """Return a starlink-pyast `Region` in an ICRS sky frame."""
        raise NotImplementedError()

    @abstractmethod
    def _boundary_skycoord(self) -> SkyCoord:
        """Return sky coordinates sampling the stencil boundary.

        Used by the sphgeom backend to size the pixel bounding box.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def region(self) -> lsst.sphgeom.Region:
        """A `lsst.sphgeom.Region` that bounds this stencil on the sky."""
        raise NotImplementedError()

    @abstractmethod
    def to_fits_metadata(self, metadata: PropertyList | MutableMapping[str, Any]) -> None:
        """Write FITS header entries that describe the stencil."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def fingerprint(self) -> bytes:
        """A 16-byte blob that is unique to this stencil."""
        raise NotImplementedError()


class SkyCircle(SkyStencil):
    """A sky-coordinate circular stencil.

    Parameters
    ----------
    center : `SpherePoint`
        The center of the circle, in ICRS (ra, dec).
    radius : `Angle`
        Radius of the circle.
    clip : `bool`, optional
        If `True` (`False` is default), clip pixel stencils returned by
        `to_pixels` instead of raising `StencilNotContainedError`.

    """

    #: Number of points used to sample the circle boundary when sizing the
    #: pixel bounding box for the sphgeom backend.
    BOUNDARY_SAMPLES = 64

    def __init__(self, center: SpherePoint, radius: Angle, clip: bool = False):
        self._center = center
        self._radius = radius
        self._clip = clip

    def __repr__(self) -> str:
        return f"SkyCircle({self._center!r}, {self._radius!r}, clip={self._clip!r})"

    @classmethod
    def from_astropy(
        cls, center: astropy.coordinates.SkyCoord, radius: astropy.coordinates.Angle, clip: bool = False
    ) -> SkyCircle:
        """Construct from `astropy.coordinates` arguments.

        Parameters
        ----------
        center : `astropy.coordinates.SkyCoord`
            The center of the circle, in ICRS (ra, dec).
        radius : `Angle`
            Radius of the circle.
        clip : `bool`, optional
            If `True` (`False` is default), clip pixel stencils returned by
            `to_pixels` instead of raising `StencilNotContainedError`.

        Returns
        -------
        stencil : `SkyCircle`
            Circular stencil.

        Raises
        ------
        ValueError
            Raised if ``center`` or ``radius`` is not scalar-valued.
        """
        return cls(
            center=_spherepoint_from_astropy(center),
            radius=_angle_from_astropy(radius),
            clip=clip,
        )

    @classmethod
    def from_sphgeom(cls, circle: lsst.sphgeom.Circle, clip: bool = False) -> SkyCircle:
        """Construct from a `lsst.sphgeom.Circle` instance."""
        return cls(SpherePoint(circle.getCenter()), Angle(circle.getOpeningAngle()), clip=clip)

    def to_polygon(self, n_vertices: int = 16) -> SkyPolygon:
        """Return a polygon sky stencil that approximates this circle.

        Parameters
        ----------
        n_vertices : `int`, optional
            Number of polygon vertices in the approximation.

        Returns
        -------
        polygon : `SkyPolygon`
            Polygon approximation.

        Notes
        -----
        For large circles and/or highly nonlinear projections, this polygon
        approximation can be mapped much more accurately to pixel coordinates.
        """
        factor = (2 * np.pi / n_vertices) * radians
        return SkyPolygon(
            (self._center.offset(b * factor, self._radius) for b in range(n_vertices)), clip=self._clip
        )

    def _ast_sky_region(self) -> Ast.Region:
        # Docstring inherited.
        return Ast.Circle(
            Ast.SkyFrame("System=ICRS"),
            1,
            [self._center.getRa().asRadians(), self._center.getDec().asRadians()],
            [self._radius.asRadians()],
        )

    def _boundary_skycoord(self) -> SkyCoord:
        # Docstring inherited.
        factor = (2 * np.pi / self.BOUNDARY_SAMPLES) * radians
        points = [self._center.offset(b * factor, self._radius) for b in range(self.BOUNDARY_SAMPLES)]
        return SkyCoord(
            ra=[p.getRa().asRadians() for p in points] * u.rad,
            dec=[p.getDec().asRadians() for p in points] * u.rad,
            frame="icrs",
        )

    @property
    def region(self) -> lsst.sphgeom.Region:
        # Docstring inherited.
        return lsst.sphgeom.Circle(self._center.getVector(), self._radius)

    def to_fits_metadata(self, metadata: PropertyList | MutableMapping[str, Any]) -> None:
        # Docstring inherited.
        metadata["ST_TYPE"] = "CIRCLE"
        metadata["ST_RA"] = self._center.getRa().asDegrees()
        metadata["ST_DEC"] = self._center.getDec().asDegrees()
        metadata["ST_RAD"] = self._radius.asDegrees()

    @property
    def fingerprint(self) -> bytes:
        # Docstring inherited.
        hasher = blake2b(digest_size=16)
        hasher.update(b"CIRCLE")
        hasher.update(struct.pack("!d", self._center.getRa().asRadians()))
        hasher.update(struct.pack("!d", self._center.getDec().asRadians()))
        hasher.update(struct.pack("!d", self._radius.asRadians()))
        return hasher.digest()


class SkyPolygon(SkyStencil):
    """A sky-coordinate stencil in the shape of a great-circle polygon.

    Parameters
    ----------
    vertices : `Iterable` [ `SpherePoint` ]
        Vertices of the polygon, CCW when looking out from the origin.
        Implicitly closed (the first vertex should not be duplicated as the
        last).
    clip : `bool`, optional
        If `True` (`False` is default), clip pixel stencils returned by
        `to_pixels` instead of raising `StencilNotContainedError`.

    Notes
    -----
    Vertex orientation is not checked at construction, and incorrect
    orientation may result in unspecified failures in `to_pixels`.
    """

    def __init__(self, vertices: Iterable[SpherePoint], clip: bool = False):
        self._vertices = tuple(vertices)
        self._clip = clip

    @classmethod
    def from_astropy(cls, vertices: astropy.coordinates.SkyCoord, clip: bool = False) -> SkyPolygon:
        """Construct from `astropy.coordinates` arguments.

        Parameters
        ----------
        vertices : `astropy.coordinates.SkyCoord`
            Array of vertices.  CCW when looking out from the origin.
            Implicitly closed (the first vertex should not be duplicated as the
            last).
        clip : `bool`, optional
            If `True` (`False` is default), clip pixel stencils returned by
            `to_pixels` instead of raising `StencilNotContainedError`.

        Returns
        -------
        stencil : `SkyPolygon`
            Polygon stencil.
        """
        return cls((_spherepoint_from_astropy(v) for v in vertices), clip=clip)

    def _ast_sky_region(self) -> Ast.Region:
        # Docstring inherited.
        sky_frame = Ast.SkyFrame("System=ICRS")
        ra = [v.getRa().asRadians() for v in self._vertices]
        dec = [v.getDec().asRadians() for v in self._vertices]
        centroid = lsst.sphgeom.LonLat(self.region.getCentroid())
        probe = [centroid.getLon().asRadians(), centroid.getLat().asRadians()]
        polygon = Ast.Polygon(sky_frame, np.array([ra, dec]))
        # AST's bounded interior depends on vertex winding: with the wrong
        # winding the polygon represents its own complement.  ``negate`` flips
        # ``pointinregion`` but not the ``mask`` polarity, so reverse the
        # vertices instead to obtain a region whose interior is the polygon.
        if not polygon.pointinregion(probe):
            polygon = Ast.Polygon(sky_frame, np.array([ra[::-1], dec[::-1]]))
        return polygon

    def _boundary_skycoord(self) -> SkyCoord:
        # Docstring inherited.
        return SkyCoord(
            ra=[v.getRa().asRadians() for v in self._vertices] * u.rad,
            dec=[v.getDec().asRadians() for v in self._vertices] * u.rad,
            frame="icrs",
        )

    @property
    def region(self) -> lsst.sphgeom.Region:
        # Docstring inherited.
        return lsst.sphgeom.ConvexPolygon([v.getVector() for v in self._vertices])

    def to_fits_metadata(self, metadata: PropertyList | MutableMapping[str, Any]) -> None:
        # Docstring inherited.
        metadata["ST_TYPE"] = "POLYGON"
        if len(self._vertices) > 100:
            raise NotImplementedError(
                "TODO: FITS limitations make it difficult to serialize big stencils to the header."
            )
        for n, v in enumerate(self._vertices):
            metadata[f"ST_RA{n:02d}"] = v.getRa().asDegrees()
            metadata[f"ST_DEC{n:02d}"] = v.getDec().asDegrees()

    @property
    def fingerprint(self) -> bytes:
        # Docstring inherited.
        hasher = blake2b(digest_size=16)
        hasher.update(b"POLYGON")
        for v in self._vertices:
            hasher.update(struct.pack("!dd", v.getRa().asRadians(), v.getDec().asRadians()))
        return hasher.digest()


def _angle_from_astropy(angle: astropy.coordinates.Angle) -> Angle:
    """Convert an `astropy.coordinates.Angle` to a `lsst.geom.Angle`.

    Parameters
    ----------
    angle : `astropy.coordinates.Angle`
        Astropy Angle to convert.  Must be a scalar.

    Returns
    -------
    angle : `lsst.geom.Angle`
        LSST angle.
    """
    if not angle.isscalar:
        raise ValueError("Only scalar angles are supported.")
    return Angle(angle.rad * radians)


def _spherepoint_from_astropy(skycoord: astropy.coordinates.SkyCoord) -> SpherePoint:
    """Convert an `astropy.coordinates.SkyCoord` to a `lsst.geom.SpherePoint`.

    Parameters
    ----------
    skycoord : `astropy.coordinates.SkyCoord`
        Astropy coordinates to convert.  Must be a scalar.

    Returns
    -------
    angle : `lsst.geom.SpherePoint`
        LSST spherical point.
    """
    if not skycoord.isscalar:
        raise ValueError("Only scalar coordinates are supported.")
    icrs = skycoord.transform_to("icrs")
    return SpherePoint(icrs.ra.rad * radians, icrs.dec.rad * radians)
