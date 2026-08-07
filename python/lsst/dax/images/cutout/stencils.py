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

"""Stencils that rasterize sky-defined shapes onto pixel grids.

A `SkyStencil` (`SkyCircle` or `SkyPolygon`) describes a shape on the sky,
such as a circle around a point of interest.  `SkyStencil.to_pixels` converts
it to a `PixelStencil` with a pixel-coordinate bounding `~lsst.images.Box`,
which can then be used to cut out a subimage or set a `~lsst.images.Mask`
plane for the covered pixels.
"""

from __future__ import annotations

__all__ = (
    "PixelStencil",
    "SkyCircle",
    "SkyPolygon",
    "SkyStencil",
    "StencilNotContainedError",
)

import struct
from abc import ABC, abstractmethod
from hashlib import blake2b

import astropy.coordinates
import astropy.io.fits
import astropy.units as u
import numpy as np
import starlink.Ast as Ast
from astropy.coordinates import SkyCoord

from lsst.images import Box, Mask, NoOverlapError, SkyProjection


def _starlink_sky_to_pixel(projection: SkyProjection) -> Ast.Mapping:
    """Return the sky->pixel mapping of ``projection`` as a starlink-pyast
    Mapping.

    This package may wrap AST with either astshim or starlink-pyast depending
    on the runtime environment.  The transform is serialized to AST's native
    text form via the public `~lsst.images.Transform.show` method and re-read
    with starlink-pyast (whose ``Channel`` accepts a sequence of lines as its
    source), so that all region masking happens in starlink-pyast regardless
    of which wrapper is used internally.
    """
    return Ast.Channel(projection.sky_to_pixel_transform.show().splitlines()).read()


def _region_pixel_bbox(sky_region: Ast.Region, sky_to_pixel: Ast.Mapping) -> Box:
    """Return the tight pixel bounding box of a sky `Region`.

    The sky region is mapped into the pixel frame and its bounds are read from
    that mapped region, so the box follows the true great-circle edges of the
    region rather than the straight chords between its projected vertices.
    Because the same ``sky_region`` and ``sky_to_pixel`` are used to rasterize
    the coverage, the box and the mask cannot disagree about the region's
    extent.
    """
    pixel_region = sky_region.mapregion(sky_to_pixel, Ast.Frame(2))
    lbnd, ubnd = pixel_region.getregionbounds()
    return Box.from_float_bounds(
        x_min=float(lbnd[0]),
        x_max=float(ubnd[0]),
        y_min=float(lbnd[1]),
        y_max=float(ubnd[1]),
    )


def _interior_probe(vertices: SkyCoord) -> tuple[float, float]:
    """Return an interior point of a convex spherical polygon.

    Parameters
    ----------
    vertices
        Corners of a convex polygon, as an array-valued ICRS `SkyCoord`.

    Returns
    -------
    lon_rad : `float`
        ICRS longitude of the interior point, in radians.
    lat_rad : `float`
        ICRS latitude of the interior point, in radians.

    Notes
    -----
    The point is the normalized mean of the vertex unit vectors.  For a convex
    polygon this always lies strictly inside, which is what the vertex-winding
    correction in `SkyPolygon` requires.
    """
    xyz = vertices.cartesian.xyz.value
    mean = xyz.mean(axis=1)
    mean /= np.linalg.norm(mean)
    lon = float(np.arctan2(mean[1], mean[0]))
    lat = float(np.arcsin(mean[2]))
    return lon, lat


def _is_convex(vertices: SkyCoord) -> bool:
    """Return whether spherical polygon ``vertices`` is convex.

    Each edge ``(i, j)`` spans a great circle whose plane has normal
    ``vᵢ × vⱼ``.  The polygon is convex when, for every edge, all the other
    vertices lie on a single side of that plane.  The test is independent of
    vertex winding: it accepts a consistent orientation in either direction and
    rejects concave or self-intersecting outlines.
    """
    xyz = vertices.cartesian.xyz.value  # (3, n)
    n = xyz.shape[1]
    if n < 3:
        return False
    # Signed distance of every vertex to every edge great-circle plane.
    normals = np.stack([np.cross(xyz[:, i], xyz[:, (i + 1) % n]) for i in range(n)])  # (n, 3)
    distances = normals @ xyz  # (n_edges, n_vertices)
    # Tolerance relative to the largest distance, so nearly-collinear vertices
    # are treated as on-plane rather than as a spurious reflex angle.
    tol = 1e-9 * np.max(np.abs(distances))
    for i in range(n):
        # The edge's own endpoints lie on its plane by construction; only the
        # remaining vertices decide which side the edge bounds.  For a convex
        # polygon they never straddle it.
        others = distances[i, [k for k in range(n) if k != i and k != (i + 1) % n]]
        if np.any(others > tol) and np.any(others < -tol):
            return False
    return True


class StencilNotContainedError(RuntimeError):
    """Exception that may be raised when a stencil is not with a desired
    bounding box.
    """


class PixelStencil(ABC):
    """An image cutout stencil defined in pixel coordinates."""

    @property
    @abstractmethod
    def bbox(self) -> Box:
        """Bounding box of this stencil, as a `~lsst.images.Box`."""
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
        mask
            Mask to modify in-place.  Its schema must already define ``plane``
            and its bounding box must contain `bbox`.
        plane
            Name of the mask plane to set.
        covered
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
    sky_region
        The stencil region expressed in an ICRS sky frame, as a
        `starlink.Ast.Region`.
    sky_to_pixel
        Mapping whose forward direction transforms sky coordinates to pixels,
        as required by ``Region.mask`` (region frame to grid).
    bbox
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


class SkyStencil(ABC):
    """An image cutout stencil defined in sky (ICRS) coordinates."""

    _clip: bool

    def to_pixels(
        self,
        projection: SkyProjection,
        bbox: Box,
    ) -> PixelStencil:
        """Transform to a pixel-coordinate stencil.

        Parameters
        ----------
        projection
            Mapping from sky coordinates to pixel coordinates.
        bbox
            Bounds that the returned stencil must lie within.

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
        sky_region = self._ast_sky_region()
        sky_to_pixel = _starlink_sky_to_pixel(projection)
        tight = _region_pixel_bbox(sky_region, sky_to_pixel)
        final = self._resolve_box(tight, bbox)
        return _AstPixelRegion(sky_region, sky_to_pixel, final)

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
    center
        The center of the circle, as a scalar `astropy.coordinates.SkyCoord`
        in any frame.  It is converted to ICRS on construction.
    radius
        Radius of the circle, as a scalar `astropy.coordinates.Angle`.
    clip
        If `True` (`False` is default), clip pixel stencils returned by
        `to_pixels` instead of raising `StencilNotContainedError`.
    """

    def __init__(
        self,
        center: astropy.coordinates.SkyCoord,
        radius: astropy.coordinates.Angle,
        clip: bool = False,
    ) -> None:
        if not center.isscalar:
            raise ValueError("SkyCircle center must be a scalar SkyCoord.")
        if not radius.isscalar:
            raise ValueError("SkyCircle radius must be a scalar Angle.")
        # transform_to always returns a new object, so this both normalizes to
        # ICRS and isolates us from later mutation of the caller's SkyCoord.
        self._center = center.transform_to("icrs")
        self._radius = radius
        self._clip = clip

    def __repr__(self) -> str:
        return (
            f"SkyCircle(SkyCoord(ra={float(self._center.ra.deg)!r}, dec={float(self._center.dec.deg)!r}, "
            f"unit='deg', frame='icrs'), Angle({float(self._radius.to_value(u.rad))!r}, unit='rad'), "
            f"clip={self._clip!r})"
        )

    def to_polygon(self, n_vertices: int = 16) -> SkyPolygon:
        """Return a polygon sky stencil that approximates this circle.

        Parameters
        ----------
        n_vertices
            Number of polygon vertices in the approximation.

        Returns
        -------
        polygon : `SkyPolygon`
            Polygon approximation.

        Notes
        -----
        This helper is retained for callers that want a polygon approximation;
        it is not used by `to_pixels`, which masks the true circle.
        """
        position_angle = (np.arange(n_vertices) / n_vertices * 2.0 * np.pi) * u.rad
        points = self._center.directional_offset_by(position_angle, self._radius)
        return SkyPolygon(points, clip=self._clip)

    def _ast_sky_region(self) -> Ast.Region:
        # Docstring inherited.
        return Ast.Circle(
            Ast.SkyFrame("System=ICRS"),
            1,
            [self._center.ra.rad, self._center.dec.rad],
            [self._radius.to_value(u.rad)],
        )

    def to_fits_metadata(self) -> astropy.io.fits.Header:
        # Docstring inherited.
        header = astropy.io.fits.Header()
        header.set("ST_TYPE", "CIRCLE", "Type of stencil used to create this cutout")
        header.set("ST_RA", self._center.ra.deg, "[deg] Circle center Right Ascension")
        header.set("ST_DEC", self._center.dec.deg, "[deg] Circle center Declination")
        header.set("ST_RAD", self._radius.to_value(u.deg), "[deg] Circle radius")
        return header

    @property
    def fingerprint(self) -> bytes:
        # Docstring inherited.
        hasher = blake2b(digest_size=16)
        hasher.update(b"CIRCLE")
        hasher.update(struct.pack("!d", self._center.ra.rad))
        hasher.update(struct.pack("!d", self._center.dec.rad))
        hasher.update(struct.pack("!d", self._radius.to_value(u.rad)))
        return hasher.digest()


class SkyPolygon(SkyStencil):
    """A sky-coordinate stencil in the shape of a great-circle polygon.

    Parameters
    ----------
    vertices
        Vertices of a convex polygon, as an array-valued
        `astropy.coordinates.SkyCoord` in any frame (converted to ICRS on
        construction).  Implicitly closed (the first vertex should not be
        duplicated as the last).  Either winding is accepted; the interior is
        taken to be the smaller of the two regions the vertices bound.

    clip
        If `True` (`False` is default), clip pixel stencils returned by
        `to_pixels` instead of raising `StencilNotContainedError`.

    Raises
    ------
    ValueError
        Raised if fewer than three vertices are given or if the vertices do
        not describe a convex polygon.

    Notes
    -----
    Only convex polygons are supported.  Convexity is checked at construction
    so that the interior can be identified unambiguously; concave or
    self-intersecting outlines are rejected.
    """

    def __init__(self, vertices: astropy.coordinates.SkyCoord, clip: bool = False) -> None:
        if vertices.isscalar:
            raise ValueError("SkyPolygon vertices must be an array-valued SkyCoord.")
        # transform_to always returns a new object, so this both normalizes to
        # ICRS and isolates us from later mutation of the caller's SkyCoord.
        vertices = vertices.transform_to("icrs")
        if not _is_convex(vertices):
            raise ValueError("SkyPolygon vertices must describe a convex polygon.")
        self._vertices = vertices
        self._clip = clip

    def _ast_sky_region(self) -> Ast.Region:
        # Docstring inherited.
        sky_frame = Ast.SkyFrame("System=ICRS")
        ra = self._vertices.ra.rad
        dec = self._vertices.dec.rad
        # The vertex mean is a guaranteed-interior probe because the vertices
        # are validated convex at construction.
        probe = list(_interior_probe(self._vertices))
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

    def to_fits_metadata(self) -> astropy.io.fits.Header:
        # Docstring inherited.
        header = astropy.io.fits.Header()
        header.set("ST_TYPE", "POLYGON", "Type of stencil used to create this cutout")
        if len(self._vertices) > 100:
            raise NotImplementedError(
                "TODO: FITS limitations make it difficult to serialize big stencils to the header."
            )
        for n, v in enumerate(self._vertices):
            header.set(f"ST_RA{n:02d}", v.ra.deg, f"[deg] Vertex {n} Right Ascension")
            header.set(f"ST_DEC{n:02d}", v.dec.deg, f"[deg] Vertex {n} Declination")
        return header

    @property
    def fingerprint(self) -> bytes:
        # Docstring inherited.
        hasher = blake2b(digest_size=16)
        hasher.update(b"POLYGON")
        for v in self._vertices:
            hasher.update(struct.pack("!dd", v.ra.rad, v.dec.rad))
        return hasher.digest()
