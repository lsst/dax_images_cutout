# This file is part of image_cutout_backend.
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
    "PixelPolygon",
    "PixelStencil",
    "SkyCircle",
    "SkyPolygon",
    "SkyStencil",
    "SpanSetStencil",
    "StencilNotContainedError",
)

import struct
from abc import ABC, abstractmethod
from hashlib import blake2b
from typing import Iterable, Optional

import astropy.coordinates
import lsst.afw.detection
import lsst.afw.geom.ellipses
import lsst.afw.geom.polygon
import lsst.sphgeom
import numpy as np
from lsst.afw.geom import SkyWcs, SpanSet, linearizeTransform, makeCdMatrix, makeSkyWcs, makeWcsPairTransform
from lsst.afw.image import Mask
from lsst.daf.base import PropertyList
from lsst.geom import Angle, Box2I, Point2D, SpherePoint, radians


class StencilNotContainedError(RuntimeError):
    """Exception that may be raised when a stencil is not with a desired
    bounding box.
    """


class PixelStencil(ABC):
    """An image cutout stencil defined in pixel coordinates."""

    @property
    @abstractmethod
    def bbox(self) -> Box2I:
        """Bounding box of this stencil in pixel coordinates."""
        raise NotImplementedError()

    @abstractmethod
    def set_mask(self, mask: Mask, bits: int) -> None:
        """Set mask planes for the pixels covered by this stencil.

        Parameters
        ----------
        mask : `Mask`
            Mask to modify in-place.
        bits : `int`
            Integer bitmask to bitwise-OR into pixels covered by the stencil.

        Notes
        -----
        The what "pixels covered by" means in detail is implementation-defined,
        at least at present.
        """
        raise NotImplementedError()


class SpanSetStencil(PixelStencil):
    """A pixel-coordinate stencil backed by `lsst.afw.geom.SpanSet`.

    Parameters
    ----------
    spans : `SpanSet`
        Data structure containing ``(y, x0, x1)`` spans that define
        the stencil area.
    """

    def __init__(self, spans: SpanSet):
        self.spans = spans

    @property
    def bbox(self) -> Box2I:
        # Docstring inherited.
        return self.spans.getBBox()

    def set_mask(self, mask: Mask, bits: int) -> None:
        # Docstring inherited.
        self.spans.setMask(mask, bits)


class PixelPolygon(PixelStencil):
    """A pixel-coordinate stencil backed by a polygon.

    Parameters
    ----------
    polygon : `lsst.afw.geom.polygon.Polygon`
        Backing polygon.
    bbox : `Box2I`, optional
        If provided (`None` is default), a pixel bounding box to clip
        the polygon to.

    Notes
    -----
    This class is backed by a polygon with straight-line edges in the image
    pixel coordinate system; this corresponds exactly to a great-circle polygon
    on the sky only for gnomonic projections.

    The `set_mask` implementation for this class currently masks pixels that
    are more than 50% covered by the polygon, which may not be the same as
    pixels whose centers are contained by the polygon when a vertex lies within
    a pixel.  This may change in the future.
    """

    def __init__(self, polygon: lsst.afw.geom.polygon.Polygon, bbox: Optional[Box2I] = None):
        self._polygon = polygon
        self._bbox = Box2I(polygon.getBBox())
        if bbox is not None:
            self._bbox.clip(bbox)

    @property
    def bbox(self) -> Box2I:
        # Docstring inherited.
        return self._bbox

    def set_mask(self, mask: Mask, bits: int) -> None:
        # Docstring inherited.
        image = self._polygon.createImage(self.bbox)
        submask = mask[self.bbox]
        submask.array[:, :] |= bits * (image.array > 0.5)


class SkyStencil(ABC):
    """An image cutout stencil defined in sky (ICRS) coordinates."""

    @abstractmethod
    def to_pixels(self, wcs: SkyWcs, bbox: Box2I) -> PixelStencil:
        """Transform to a pixel-coordinate set of spans.

        Parameters
        ----------
        wcs : `SkyWcs`
            Mapping from sky coordinates to pixel coordinates.
        bbox : `Box2I`
            Bounds that the returned stencil must lie within.

        Returns
        -------
        pixels : `PixelStencil`
            Pixel-coordinate stencil object.  `PixelStencil.bbox` is guaranteed
            to be contained by the given ``bbox`` for this object.

        Raises
        ------
        StencilNotContainedError
            May be raised if the pixel-coordinate stencil does not lie within
            ``bbox``.  Implementations may also clip instead.

        Notes
        -----
        This operation may be an approximation; see concrete class
        documentation for additional information.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def region(self) -> lsst.sphgeom.Region:
        """A `lsst.sphgeom.Region` that bounds this stencil on the sky."""
        raise NotImplementedError()

    @abstractmethod
    def to_fits_metadata(self, metadata: PropertyList) -> None:
        """Write FITS header entries that describe the stencil.

        Parameters
        ----------
        metadata : `PropertyList`
            Metadata, to be modified in place.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def fingerprint(self) -> bytes:
        """A 16-byte blob that is unique to this stencil.

        This may be a hash or a reversible encoding.
        """
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

    Notes
    -----
    The `to_pixels` implementation for this class transforms the circle
    to an ellipse via a local linear approximation to the WCS.  This works well
    for small circles on the scale of a few arcseconds, and should usually be
    fine for circles on the scale of a few arcminutes, as long as the WCS is
    not highly nonlinear.  For larger circles, it may be more accurate to
    first call `to_polygon` and pixelize the result.
    """

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
        )

    @classmethod
    def from_sphgeom(cls, circle: lsst.sphgeom.Circle, clip: bool = False) -> SkyCircle:
        """Construct from a `lsst.sphgeom.Circle` instance."""
        return cls(SpherePoint(circle.getCenter()), Angle(circle.getOpeningAngle()))

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
        approximation be mapped much more accurately to pixel coordinates.
        """
        factor = (2 * np.pi / n_vertices) * radians
        return SkyPolygon(self._center.offset(b * factor, self._radius) for b in range(n_vertices))

    def to_pixels(self, wcs: SkyWcs, bbox: Box2I) -> PixelStencil:
        # Docstring inherited.
        transform = makeWcsPairTransform(_make_local_gnomonic_wcs(self._center), wcs)
        affine = linearizeTransform(transform, Point2D(0.0, 0.0))
        sky_ellipse_core = lsst.afw.geom.ellipses.Axes(
            self._radius.asRadians(), self._radius.asRadians(), 0.0
        )
        pixel_ellipse_core = sky_ellipse_core.transform(affine.getLinear())
        spans = SpanSet.fromShape(
            lsst.afw.geom.ellipses.Ellipse(pixel_ellipse_core, wcs.skyToPixel(self._center))
        )
        if not bbox.contains(spans.getBBox()):
            if self._clip:
                spans = spans.clippedTo(bbox)
            else:
                raise StencilNotContainedError(
                    f"{self} has bbox {spans.getBBox()} in pixel coordinates, which is not within {bbox}."
                )
        return SpanSetStencil(spans)

    @property
    def region(self) -> lsst.sphgeom.Region:
        # Docstring inherited.
        return lsst.sphgeom.Circle(self._center.getVector(), self._radius)

    def to_fits_metadata(self, metadata: PropertyList) -> None:
        # Docstring inherited.
        metadata.set("ST_TYPE", "CIRCLE", "Type of stencil used to create this cutout.")
        metadata.set("ST_RA", self._center.getRa().asDegrees(), "Circle center right ascension in degrees.")
        metadata.set("ST_DEC", self._center.getDec().asDegrees(), "Circle center declination in degrees.")
        metadata.set("ST_RAD", self._radius.asDegrees(), "Circle radius in degrees.")

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
        return cls(_spherepoint_from_astropy(v) for v in vertices)

    def to_pixels(self, wcs: SkyWcs, bbox: Box2I) -> PixelStencil:
        # Docstring inherited.
        pixel_vertices = wcs.skyToPixel(self._vertices)
        pixel_vertices.append(pixel_vertices[0])  # afw.polygon expects to be explicitly closed
        # Input sky coordinates should be CCW looking out, and afw.polygon
        # expects pixel coordinates to be CW.  First question is whether the
        # WCS has a parity flip, which is frustratingly not something we can
        # ask it (or the underlying AST object) directly, so we compute the
        # determinant of a linear approximation.  It doesn't matter where,
        # since the polarity can't actually change with position
        affine = wcs.linearizeSkyToPixel(self._vertices[0], radians)
        if affine.getLinear().computeDeterminant() > 1:
            # No parity flip, so we have to reverse the vertices ourselves.
            pixel_vertices = reversed(pixel_vertices)
        result = PixelPolygon(
            lsst.afw.geom.polygon.Polygon(pixel_vertices), bbox=(bbox if self._clip else None)
        )
        if not self._clip and not bbox.contains(result.bbox):
            raise StencilNotContainedError(
                f"{self} has bbox {result.bbox} in pixel coordinates, which is not within {bbox}."
            )
        return result

    @property
    def region(self) -> lsst.sphgeom.Region:
        # Docstring inherited.
        return lsst.sphgeom.ConvexPolygon([v.getVector() for v in self._vertices])

    def to_fits_metadata(self, metadata: PropertyList) -> None:
        # Docstring inherited.
        metadata.set("ST_TYPE", "POLYGON", "Type of stencil used to create this cutout.")
        if len(self._vertices) > 100:
            raise NotImplementedError(
                "TODO: FITS limitations make it difficult to serialize big stencils to the header."
            )
        for n, v in enumerate(self._vertices):
            metadata.set(f"ST_RA{n:02d}", v.getRa().asDegrees(), f"Vertex {n} right ascension in degrees.")
            metadata.set(f"ST_DEC{n:02d}", v.getDec().asDegrees(), f"Vertex {n} declination in degrees.")

    @property
    def fingerprint(self) -> bytes:
        # Docstring inherited.
        hasher = blake2b(digest_size=16)
        hasher.update(b"POLYGON")
        for v in self._vertices:
            hasher.update(struct.pack("!d!d", v.getRa().asRadians(), v.getDecx().asRadians()))
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


def _make_local_gnomonic_wcs(position: SpherePoint) -> SkyWcs:
    """Construct a WCS that represents a local gnomonic transform at a point.

    Parameters
    ----------
    position : `SpherePoint`
        Sky coordinate of the center of the projection.

    Returns
    -------
    wcs : `SkyWcs`
        A gnomonic (TAN) WCS with ``CRVAL=position``, ``CRPIX=(0,0)``, a pixel
        scale of ``1/rad``, aligned such that ``(x, y)`` correspond to
        ``(ra, dec)``.
    """
    return makeSkyWcs(Point2D(0.0, 0.0), position, makeCdMatrix(1.0 * radians))
