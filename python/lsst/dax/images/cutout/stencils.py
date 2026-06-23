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
from collections.abc import Iterable
from hashlib import blake2b

import astropy.coordinates
import astropy.io.fits
import astropy.units as u
import numpy as np
import starlink.Ast as Ast
from astropy.coordinates import SkyCoord

import lsst.sphgeom
from lsst.images import Box, Mask, NoOverlapError, SkyProjection
from lsst.sphgeom import Angle, LonLat, UnitVector3d


class MaskBackend(enum.Enum):
    """Selects the algorithm used to rasterize a stencil onto pixels."""

    AST = enum.auto()
    """Mask using starlink-pyast ``Region.mask`` on the true sky region."""

    SPHGEOM = enum.auto()
    """Mask by testing pixel centers against the ``lsst.sphgeom`` region."""


def _skycoord_from_lonlat(lonlat: LonLat) -> SkyCoord:
    """Return an ICRS `astropy.coordinates.SkyCoord` for a `sphgeom.LonLat`."""
    return SkyCoord(
        ra=lonlat.getLon().asRadians() * u.rad,
        dec=lonlat.getLat().asRadians() * u.rad,
        frame="icrs",
    )


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
    def _coverage(self) -> np.ndarray:
        """Boolean array over `bbox`, `True` for pixels the stencil covers.

        The array has shape ``bbox.shape`` (``(ny, nx)``).
        """
        raise NotImplementedError()

    def set_mask(self, mask: Mask, plane: str, *, covered: bool = True) -> None:
        """Set a mask plane for pixels inside or outside the stencil.

        Parameters
        ----------
        mask : `lsst.images.Mask`
            Mask to modify in-place.  Its schema must already define ``plane``
            and its bounding box must contain `bbox`.
        plane : `str`
            Name of the mask plane to set.
        covered : `bool`, optional
            If `True` (default), set ``plane`` where the stencil covers a pixel
            center.  If `False`, set ``plane`` where the stencil does *not*
            cover a pixel, including the region of ``mask`` that lies outside
            `bbox`.
        """
        coverage = self._coverage()
        if not covered:
            coverage = np.logical_not(coverage)
        # Pixels outside the stencil's bounding box are never covered, so they
        # take the value assigned to uncovered pixels.
        full = np.full(mask.bbox.shape, not covered, dtype=bool)
        y_off = self.bbox.y.min - mask.bbox.y.min
        x_off = self.bbox.x.min - mask.bbox.x.min
        full[y_off : y_off + self.bbox.shape.y, x_off : x_off + self.bbox.shape.x] = coverage
        mask.set(plane, full)


class _AstPixelRegion(PixelStencil):
    """Pixel-coordinate stencil backed by a starlink-pyast sky `Region`.

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

    def _coverage(self) -> np.ndarray:
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
        return scratch != 0


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

    def _coverage(self) -> np.ndarray:
        # Docstring inherited.
        grid = self._bbox.meshgrid()
        sky = self._projection.pixel_to_sky(x=grid.x.ravel(), y=grid.y.ravel())
        return self._region.contains(sky.ra.radian, sky.dec.radian).reshape(self._bbox.shape)


class SkyStencil(ABC):
    """An image cutout stencil defined in sky (ICRS) coordinates."""

    _clip: bool

    def to_pixels(
        self,
        projection: SkyProjection,
        bbox: Box,
        *,
        backend: MaskBackend = MaskBackend.AST,
    ) -> PixelStencil:
        """Transform to a pixel-coordinate stencil.

        Parameters
        ----------
        projection : `lsst.images.SkyProjection`
            Mapping from sky coordinates to pixel coordinates.
        bbox : `lsst.images.Box`
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
        tight = self._pixel_bbox(projection)
        final = self._resolve_box(tight, bbox)
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
        return Box.from_float_bounds(
            x_min=float(np.min(xy.x)),
            x_max=float(np.max(xy.x)),
            y_min=float(np.min(xy.y)),
            y_max=float(np.max(xy.y)),
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

        Used to size the pixel bounding box.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def region(self) -> lsst.sphgeom.Region:
        """A `lsst.sphgeom.Region` that bounds this stencil on the sky."""
        raise NotImplementedError()

    @abstractmethod
    def to_fits_metadata(self) -> astropy.io.fits.Header:
        """Return FITS header cards that describe the stencil.

        The cards carry per-keyword comments and are merged into the cutout
        provenance header.
        """
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
    center : `lsst.sphgeom.LonLat`
        The center of the circle, in ICRS (longitude, latitude).
    radius : `lsst.sphgeom.Angle`
        Radius of the circle.
    clip : `bool`, optional
        If `True` (`False` is default), clip pixel stencils returned by
        `to_pixels` instead of raising `StencilNotContainedError`.
    """

    #: Number of points used to sample the circle boundary when sizing the
    #: pixel bounding box.
    BOUNDARY_SAMPLES = 64

    def __init__(self, center: LonLat, radius: Angle, clip: bool = False):
        self._center = center
        self._radius = radius
        self._clip = clip

    def __repr__(self) -> str:
        return (
            f"SkyCircle(LonLat.fromRadians({self._center.getLon().asRadians()!r}, "
            f"{self._center.getLat().asRadians()!r}), "
            f"Angle({self._radius.asRadians()!r}), clip={self._clip!r})"
        )

    @classmethod
    def from_astropy(
        cls, center: astropy.coordinates.SkyCoord, radius: astropy.coordinates.Angle, clip: bool = False
    ) -> SkyCircle:
        """Construct from `astropy.coordinates` arguments.

        Parameters
        ----------
        center : `astropy.coordinates.SkyCoord`
            The center of the circle, in ICRS (ra, dec).  Must be scalar.
        radius : `astropy.coordinates.Angle`
            Radius of the circle.  Must be scalar.
        clip : `bool`, optional
            If `True` (`False` is default), clip pixel stencils returned by
            `to_pixels` instead of raising `StencilNotContainedError`.

        Returns
        -------
        stencil : `SkyCircle`
            Circular stencil.
        """
        return cls(
            center=_lonlat_from_astropy(center),
            radius=_angle_from_astropy(radius),
            clip=clip,
        )

    @classmethod
    def from_sphgeom(cls, circle: lsst.sphgeom.Circle, clip: bool = False) -> SkyCircle:
        """Construct from a `lsst.sphgeom.Circle` instance."""
        return cls(LonLat(circle.getCenter()), circle.getOpeningAngle(), clip=clip)

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
        This helper is retained for callers that want a polygon approximation;
        it is no longer used by `to_pixels`, which masks the true circle.
        """
        center = _skycoord_from_lonlat(self._center)
        position_angle = (np.arange(n_vertices) / n_vertices * 2.0 * np.pi) * u.rad
        radius = astropy.coordinates.Angle(self._radius.asRadians() * u.rad)
        points = center.directional_offset_by(position_angle, radius)
        vertices = [LonLat.fromRadians(float(p.ra.rad), float(p.dec.rad)) for p in points]
        return SkyPolygon(vertices, clip=self._clip)

    def _ast_sky_region(self) -> Ast.Region:
        # Docstring inherited.
        return Ast.Circle(
            Ast.SkyFrame("System=ICRS"),
            1,
            [self._center.getLon().asRadians(), self._center.getLat().asRadians()],
            [self._radius.asRadians()],
        )

    def _boundary_skycoord(self) -> SkyCoord:
        # Docstring inherited.
        center = _skycoord_from_lonlat(self._center)
        position_angle = (np.arange(self.BOUNDARY_SAMPLES) / self.BOUNDARY_SAMPLES * 2.0 * np.pi) * u.rad
        radius = astropy.coordinates.Angle(self._radius.asRadians() * u.rad)
        return center.directional_offset_by(position_angle, radius)

    @property
    def region(self) -> lsst.sphgeom.Region:
        # Docstring inherited.
        return lsst.sphgeom.Circle(UnitVector3d(self._center), self._radius)

    def to_fits_metadata(self) -> astropy.io.fits.Header:
        # Docstring inherited.
        header = astropy.io.fits.Header()
        header.set("ST_TYPE", "CIRCLE", "Type of stencil used to create this cutout")
        header.set("ST_RA", self._center.getLon().asDegrees(), "[deg] Circle center Right Ascension")
        header.set("ST_DEC", self._center.getLat().asDegrees(), "[deg] Circle center Declination")
        header.set("ST_RAD", self._radius.asDegrees(), "[deg] Circle radius")
        return header

    @property
    def fingerprint(self) -> bytes:
        # Docstring inherited.
        hasher = blake2b(digest_size=16)
        hasher.update(b"CIRCLE")
        hasher.update(struct.pack("!d", self._center.getLon().asRadians()))
        hasher.update(struct.pack("!d", self._center.getLat().asRadians()))
        hasher.update(struct.pack("!d", self._radius.asRadians()))
        return hasher.digest()


class SkyPolygon(SkyStencil):
    """A sky-coordinate stencil in the shape of a great-circle polygon.

    Parameters
    ----------
    vertices : `Iterable` [ `lsst.sphgeom.LonLat` ]
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

    def __init__(self, vertices: Iterable[LonLat], clip: bool = False):
        self._vertices = tuple(vertices)
        self._clip = clip

    @classmethod
    def from_astropy(cls, vertices: astropy.coordinates.SkyCoord, clip: bool = False) -> SkyPolygon:
        """Construct from an array-valued `astropy.coordinates.SkyCoord`.

        Parameters
        ----------
        vertices : `astropy.coordinates.SkyCoord`
            Array of vertices, CCW when looking out from the origin.
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
        return cls((_lonlat_from_astropy(v) for v in vertices), clip=clip)

    def _ast_sky_region(self) -> Ast.Region:
        # Docstring inherited.
        sky_frame = Ast.SkyFrame("System=ICRS")
        ra = [v.getLon().asRadians() for v in self._vertices]
        dec = [v.getLat().asRadians() for v in self._vertices]
        centroid = lsst.sphgeom.LonLat(self.region.getCentroid())
        probe = [centroid.getLon().asRadians(), centroid.getLat().asRadians()]
        polygon = Ast.Polygon(sky_frame, np.array([ra, dec]))
        # AST's bounded interior depends on vertex winding: with the wrong
        # winding the polygon represents its own complement.  ``negate`` flips
        # ``pointinregion`` but not the ``mask`` polarity, so reverse the
        # vertices instead to obtain a region whose interior is the polygon.
        if not polygon.pointinregion(probe):
            polygon = Ast.Polygon(sky_frame, np.array([ra[::-1], dec[::-1]]))
        # ``Region.mask`` rasterizes by simplifying the region into the pixel
        # frame.  By default AST re-fits the polygon to straight-edged
        # pixel-space vertices, discarding the great-circle curvature of the
        # edges whenever the sky-to-pixel projection is non-gnomonic.
        # ``SimpVertices=0`` makes AST keep the curved edges unless they match
        # the straight approximation to within the region's uncertainty.
        polygon.set("SimpVertices=0")
        return polygon

    def _boundary_skycoord(self) -> SkyCoord:
        # Docstring inherited.
        return SkyCoord(
            ra=[v.getLon().asRadians() for v in self._vertices] * u.rad,
            dec=[v.getLat().asRadians() for v in self._vertices] * u.rad,
            frame="icrs",
        )

    @property
    def region(self) -> lsst.sphgeom.Region:
        # Docstring inherited.
        return lsst.sphgeom.ConvexPolygon([UnitVector3d(v) for v in self._vertices])

    def to_fits_metadata(self) -> astropy.io.fits.Header:
        # Docstring inherited.
        header = astropy.io.fits.Header()
        header.set("ST_TYPE", "POLYGON", "Type of stencil used to create this cutout")
        if len(self._vertices) > 100:
            raise NotImplementedError(
                "TODO: FITS limitations make it difficult to serialize big stencils to the header."
            )
        for n, v in enumerate(self._vertices):
            header.set(f"ST_RA{n:02d}", v.getLon().asDegrees(), f"[deg] Vertex {n} Right Ascension")
            header.set(f"ST_DEC{n:02d}", v.getLat().asDegrees(), f"[deg] Vertex {n} Declination")
        return header

    @property
    def fingerprint(self) -> bytes:
        # Docstring inherited.
        hasher = blake2b(digest_size=16)
        hasher.update(b"POLYGON")
        for v in self._vertices:
            hasher.update(struct.pack("!dd", v.getLon().asRadians(), v.getLat().asRadians()))
        return hasher.digest()


def _angle_from_astropy(angle: astropy.coordinates.Angle) -> Angle:
    """Convert an `astropy.coordinates.Angle` to a `lsst.sphgeom.Angle`.

    Parameters
    ----------
    angle : `astropy.coordinates.Angle`
        Astropy Angle to convert.  Must be a scalar.

    Returns
    -------
    angle : `lsst.sphgeom.Angle`
        Equivalent sphgeom angle.
    """
    if not angle.isscalar:
        raise ValueError("Only scalar angles are supported.")
    return Angle(angle.to_value(u.rad))


def _lonlat_from_astropy(skycoord: astropy.coordinates.SkyCoord) -> LonLat:
    """Convert an `astropy.coordinates.SkyCoord` to a `lsst.sphgeom.LonLat`.

    Parameters
    ----------
    skycoord : `astropy.coordinates.SkyCoord`
        Astropy coordinates to convert.  Must be a scalar.

    Returns
    -------
    lonlat : `lsst.sphgeom.LonLat`
        Equivalent spherical point.
    """
    if not skycoord.isscalar:
        raise ValueError("Only scalar coordinates are supported.")
    icrs = skycoord.transform_to("icrs")
    return LonLat.fromRadians(float(icrs.ra.rad), float(icrs.dec.rad))
