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

import unittest

import astropy.io.fits
import astropy.units as u
import astropy.wcs
import numpy as np
from astropy.coordinates import Angle, SkyCoord

from lsst.dax.images.cutout.stencils import (
    SkyCircle,
    SkyPolygon,
    SkyStencil,
    StencilNotContainedError,
)
from lsst.images import Box, GeneralFrame, Mask, MaskPlane, MaskSchema, SkyProjection

# Bounding box for the cutout tests, in [y, x] (stop exclusive).  Slightly
# bigger in x to catch x<->y transposition bugs.
TEST_BOX = Box.factory[-13:28, -16:27]

# Center used by most of the stencils in these tests.
CENTER = SkyCoord(ra=12.0 * u.deg, dec=13.0 * u.deg, frame="icrs")


def _arcsec(value: float) -> Angle:
    """Return an `astropy.coordinates.Angle` of ``value`` arcseconds."""
    return Angle(value * u.arcsec)


def _make_wcs() -> astropy.wcs.WCS:
    """Build a gnomonic FITS WCS with 0.1 arcsec pixels at (12, 13) deg.

    The reference pixel is placed at pixel (5, 7) so the stencils land at an
    arbitrary nonzero offset within `TEST_BOX`.
    """
    wcs = astropy.wcs.WCS(naxis=2)
    # FITS CRPIX is 1-based, so 0-based pixel (5, 7) is CRPIX (6, 8).
    wcs.wcs.crpix = [6.0, 8.0]
    wcs.wcs.crval = [12.0, 13.0]
    scale = 0.1 / 3600.0
    wcs.wcs.cd = [[-scale, 0.0], [0.0, scale]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _make_car_wcs() -> astropy.wcs.WCS:
    """Build a plate-carree (CAR) WCS referenced on the equator.

    CAR is non-gnomonic, so great circles do not map to straight lines in
    pixel space.  The equatorial reference keeps pixel coordinates a simple
    (lon, lat) grid; a polygon placed at high declination then has edges that
    bow well away from the straight chords joining its projected vertices.
    """
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [0.0, 0.0]
    scale = 0.02
    wcs.wcs.cd = [[-scale, 0.0], [0.0, scale]]
    wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    return wcs


def _sky_contains(sky_stencil: SkyStencil, sky: SkyCoord) -> np.ndarray:
    """Return a boolean array, `True` where ``sky`` points are inside the
    stencil, computed independently of the code under test.

    Circles use the great-circle separation from the center.  Convex polygons
    use a spherical half-space test on unit vectors: for each edge ``(i, j)``
    taken in vertex order the value ``sky . (v_i x v_j)`` is computed, and a
    point is considered a consistent-sign candidate when those values share one
    sign across all edges (so the half-space test is independent of vertex
    winding).  Boundary points (zero) count as inside.  The consistent-sign
    condition is satisfied both inside the polygon and inside its antipodal
    reflection; an additional interior-hemisphere check (point must lie in the
    same hemisphere as the normalized mean of the vertices) excludes the
    antipode.
    """
    if isinstance(sky_stencil, SkyCircle):
        return np.asarray(sky_stencil._center.separation(sky) <= sky_stencil._radius)
    assert isinstance(sky_stencil, SkyPolygon)
    verts = sky_stencil._vertices.cartesian.xyz.value  # (3, nverts)
    points = sky.cartesian.xyz.value.reshape(3, -1)  # (3, npts)
    nverts = verts.shape[1]
    orientations = np.array(
        [np.cross(verts[:, i], verts[:, (i + 1) % nverts]) @ points for i in range(nverts)]
    )  # (nverts, npts)
    consistent_sign = np.all(orientations >= 0.0, axis=0) | np.all(orientations <= 0.0, axis=0)
    # The consistent-sign test alone is satisfied inside the polygon and inside
    # its antipodal reflection; require the point to lie in the same hemisphere
    # as the polygon interior to keep only the polygon itself.
    interior = verts.mean(axis=1)
    interior /= np.linalg.norm(interior)
    return consistent_sign & (interior @ points > 0.0)


def _brute_force_stencil_array(sky_stencil: SkyStencil, wcs: astropy.wcs.WCS, box: Box) -> np.ndarray:
    """Make a boolean ``(ny, nx)`` array, `True` where a center is inside.

    The pixel grid is transformed to the sky with the FITS WCS (independent of
    the `~lsst.images.SkyProjection` under test) and tested against the stencil
    with an independent spherical-geometry implementation.
    """
    grid = box.meshgrid()
    sky = wcs.pixel_to_world(grid.x.ravel(), grid.y.ravel())
    return _sky_contains(sky_stencil, sky).reshape(box.shape)


def _cartesian_pixel_coverage(polygon: SkyPolygon, wcs: astropy.wcs.WCS, box: Box) -> np.ndarray:
    """Make a boolean ``(ny, nx)`` array for the polygon with straight edges.

    The vertices are projected to pixels and joined by straight chords (a
    convex point-in-polygon test).  This models the cartesian rasterization
    that ignores great-circle curvature, so it can be compared against the
    true spherical coverage to show the curved scenario is non-trivial.
    """
    vertices = polygon._vertices
    vx, vy = wcs.world_to_pixel_values(vertices.ra.deg, vertices.dec.deg)
    grid = box.meshgrid()
    px = grid.x.ravel().astype(float)
    py = grid.y.ravel().astype(float)
    n = len(vx)
    cross = np.array(
        [
            (vx[(i + 1) % n] - vx[i]) * (py - vy[i]) - (vy[(i + 1) % n] - vy[i]) * (px - vx[i])
            for i in range(n)
        ]
    )
    inside = np.all(cross >= 0.0, axis=0) | np.all(cross <= 0.0, axis=0)
    return inside.reshape(box.shape)


def _check_to_pixel(
    test_case: unittest.TestCase,
    sky_stencil: SkyStencil,
    wcs: astropy.wcs.WCS,
    *,
    box: Box = TEST_BOX,
    expected_bbox: Box | None = None,
    max_missing: int = 0,
    max_extra: int = 0,
    plot: bool = False,
) -> None:
    """Check a `SkyStencil.to_pixels` result against brute force.

    ``box`` is the reference bounding box passed to `to_pixels`; when it does
    not fully contain the stencil the result is clipped to it.  Brute force is
    evaluated over ``box`` too, which yields the correct expected coverage for
    a clipped stencil: a pixel that is inside the region and inside ``box`` is
    necessarily inside the clipped bounding box, since the region is contained
    by its own tight bounding box.  ``expected_bbox``, if given, is asserted to
    equal the clipped result bounding box.
    """
    projection = SkyProjection.from_fits_wcs(wcs, GeneralFrame(unit=u.pix))
    pixel_stencil = sky_stencil.to_pixels(projection, box)
    test_case.assertTrue(box.contains(pixel_stencil.bbox))
    if expected_bbox is not None:
        test_case.assertEqual(pixel_stencil.bbox, expected_bbox)
    mask = Mask(schema=MaskSchema([MaskPlane("STENCIL", "stencil coverage")]), bbox=box)
    pixel_stencil.set_mask(mask, "STENCIL")
    got = mask.get("STENCIL")
    check_array = _brute_force_stencil_array(sky_stencil, wcs, box)
    missing = np.logical_and(check_array, np.logical_not(got))
    extra = np.logical_and(got, np.logical_not(check_array))
    if plot:
        from matplotlib import pyplot

        display_array = np.zeros((box.shape.y, box.shape.x, 3), dtype=np.uint8)
        display_array[:, :, 0] = 255 * check_array
        display_array[:, :, 1] = 255 * got
        pyplot.imshow(display_array, origin="lower", interpolation="nearest")
        pyplot.title("red=check, green=SkyStencil.to_pixel, yellow=both")
        pyplot.show()
    test_case.assertLessEqual(int(missing.sum()), max_missing)
    test_case.assertLessEqual(int(extra.sum()), max_extra)


class SkyCircleTestCase(unittest.TestCase):
    """Tests for `SkyCircle`."""

    def setUp(self) -> None:
        self.instance = SkyCircle(CENTER, _arcsec(1.0))

    def test_repr(self) -> None:
        """A SkyCircle can be reconstructed by evaluating its repr."""
        self.assertEqual(eval(repr(self.instance)).fingerprint, self.instance.fingerprint)

    def test_to_pixel(self) -> None:
        """Check circle rasterization against brute force."""
        _check_to_pixel(self, self.instance, _make_wcs(), max_missing=2, max_extra=2)

    def test_to_polygon_vertices_on_circle(self) -> None:
        """Polygon approximation vertices are one radius from the circle
        center.
        """
        polygon_stencil = self.instance.to_polygon()
        separations = CENTER.separation(polygon_stencil._vertices)
        np.testing.assert_allclose(separations.to_value(u.arcsec), 1.0, rtol=1e-6)

    def test_to_polygon_to_pixel(self) -> None:
        """Check polygon rasterization against brute force."""
        polygon_stencil = self.instance.to_polygon()
        _check_to_pixel(self, polygon_stencil, _make_wcs(), max_missing=6, max_extra=6)

    def test_ast_sky_region_contains_center(self) -> None:
        """The AST region for a circle contains the circle's center."""
        region = self.instance._ast_sky_region()
        self.assertTrue(region.pointinregion([CENTER.ra.rad, CENTER.dec.rad]))


class SkyPolygonTestCase(unittest.TestCase):
    """Tests for `SkyPolygon` construction and orientation handling."""

    def test_rejects_non_convex(self) -> None:
        """A concave polygon is rejected at construction.

        The winding correction relies on the polygon being convex, so a
        concave or self-intersecting outline must be rejected rather than
        silently rasterized as some other region.
        """
        chevron = SkyCoord(
            ra=[-6.0, 6.0, 6.0, 0.0, -6.0] * u.deg,
            dec=[-6.0, -6.0, 6.0, -4.0, 6.0] * u.deg,
            frame="icrs",
        )
        with self.assertRaisesRegex(ValueError, "convex"):
            SkyPolygon(chevron)

    def test_accepts_either_winding(self) -> None:
        """Convex polygons are accepted regardless of vertex winding."""
        ccw = SkyCoord(ra=[-4.0, 4.0, 0.0] * u.deg, dec=[70.0, 70.0, 66.0] * u.deg, frame="icrs")
        cw = SkyCoord(ra=[0.0, 4.0, -4.0] * u.deg, dec=[66.0, 70.0, 70.0] * u.deg, frame="icrs")
        # Neither construction raises.
        SkyPolygon(ccw)
        SkyPolygon(cw)

    def test_ast_sky_region_contains_center(self) -> None:
        """The AST region for a polygon contains an independently-derived
        interior point.

        Uses the center of a circle whose polygon approximation is the region:
        the center is manifestly interior and is not the probe point the
        winding correction uses, so a broken winding correction is actually
        caught.
        """
        polygon = SkyCircle(CENTER, _arcsec(2.0)).to_polygon(n_vertices=8)
        region = polygon._ast_sky_region()
        self.assertTrue(region.pointinregion([CENTER.ra.rad, CENTER.dec.rad]))


class SetMaskTestCase(unittest.TestCase):
    """Tests for `PixelStencil.set_mask`."""

    def test_covered_false_marks_outside(self) -> None:
        """``set_mask(covered=False)`` flags exactly the pixels the stencil
        does not cover, including the region of the mask outside the stencil's
        bounding box.
        """
        projection = SkyProjection.from_fits_wcs(_make_wcs(), GeneralFrame(unit=u.pix))
        circle = SkyCircle(CENTER, _arcsec(1.0))
        pixel_stencil = circle.to_pixels(projection, TEST_BOX)

        inside = Mask(schema=MaskSchema([MaskPlane("STENCIL", "stencil coverage")]), bbox=TEST_BOX)
        pixel_stencil.set_mask(inside, "STENCIL")

        outside = Mask(schema=MaskSchema([MaskPlane("STENCIL", "stencil coverage")]), bbox=TEST_BOX)
        pixel_stencil.set_mask(outside, "STENCIL", covered=False)

        inside_arr = inside.get("STENCIL")
        outside_arr = outside.get("STENCIL")
        # The two planes partition the mask: every pixel is flagged in exactly
        # one of them.
        np.testing.assert_array_equal(outside_arr, np.logical_not(inside_arr))
        # The stencil covers some pixels but not the whole box, so neither
        # plane is empty.
        self.assertTrue(inside_arr.any())
        self.assertTrue(outside_arr.any())


class GreatCircleCurvatureTestCase(unittest.TestCase):
    """Polygon stencils whose great-circle edges curve in pixel space.

    The other tests use a gnomonic (TAN) projection, which maps great circles
    to exactly straight lines and so cannot exercise edge curvature.  These
    tests use a plate-carree (CAR) projection referenced on the equator with a
    polygon at high declination, where the great-circle edges bow well away
    from the straight pixel-space chords joining the projected vertices.
    Rasterization must follow the true great circle rather than the chord.

    The vertices land exactly on pixel centers in this geometry (lon ``+/-4``
    and ``0`` degrees, dec ``70`` and ``66`` degrees map to integer pixels at
    this reference and scale), so the handful of pixels of residual
    disagreement allowed by the tolerances below are the vertex pixels
    themselves: their centers sit exactly on the polygon boundary, where
    containment is a tie that the edge test resolves differently for boundary
    vertices.  Vertices at generic sub-pixel positions would typically agree
    exactly.
    """

    def setUp(self) -> None:
        self.wcs = _make_car_wcs()
        self.projection = SkyProjection.from_fits_wcs(self.wcs, GeneralFrame(unit=u.pix))
        self.polygon = SkyPolygon(
            SkyCoord(ra=[-4.0, 4.0, 0.0] * u.deg, dec=[70.0, 70.0, 66.0] * u.deg, frame="icrs")
        )
        # The tight pixel box comes from the region mapped into pixels, so it
        # follows the curved edges; a generous reference box leaves it
        # unclipped.
        self.box = self.polygon.to_pixels(self.projection, Box.factory[-10000:10000, -10000:10000]).bbox

    def test_curved_edge_extends_bbox(self) -> None:
        """The tight pixel bbox covers curved edges, not just projected
        vertices.

        In this CAR scenario the great-circle top edge bows to y~=3502.24 in
        pixels while the two top vertices both project to y=3500.  A bbox
        sized from the vertices alone would stop at y=3500 and silently drop
        the rows the mask should cover; the region-derived bbox must reach
        the curved apex.
        """
        vertices_xy = self.projection.sky_to_pixel(self.polygon._vertices)
        vertex_y_max = float(np.max(vertices_xy.y))
        # The apex of the bowed edge, sampled densely along the great circle.
        top_edge = self.polygon._vertices[0].directional_offset_by(
            self.polygon._vertices[0].position_angle(self.polygon._vertices[1]),
            self.polygon._vertices[0].separation(self.polygon._vertices[1]) * np.linspace(0.0, 1.0, 64),
        )
        apex_y = float(np.max(self.projection.sky_to_pixel(top_edge).y))
        self.assertGreater(apex_y, vertex_y_max + 1.0)
        # bbox.y.max is inclusive; the box must reach the curved apex row.
        self.assertGreaterEqual(self.box.y.max, int(np.floor(apex_y)))

    def test_scenario_exercises_curvature(self) -> None:
        """The true spherical coverage differs substantially from a straight-
        edged pixel-space approximation, so the rasterization checks are a
        meaningful test of great-circle handling rather than vacuously true.
        """
        truth = _brute_force_stencil_array(self.polygon, self.wcs, self.box)
        cartesian = _cartesian_pixel_coverage(self.polygon, self.wcs, self.box)
        self.assertGreater(int(np.sum(truth != cartesian)), 500)

    def test_rasterization_follows_great_circle(self) -> None:
        """Rasterization follows the true great-circle edges.

        Only the three vertex pixels may disagree (see the class docstring); a
        larger count would mean the edges were rasterized as straight pixel
        chords.
        """
        pixel_stencil = self.polygon.to_pixels(self.projection, self.box)
        mask = Mask(schema=MaskSchema([MaskPlane("STENCIL", "stencil coverage")]), bbox=self.box)
        pixel_stencil.set_mask(mask, "STENCIL")
        got = mask.get("STENCIL")
        truth = _brute_force_stencil_array(self.polygon, self.wcs, self.box)
        self.assertLessEqual(int(np.sum(got != truth)), 3)


class StencilContainmentTestCase(unittest.TestCase):
    """Clipping and raising when a stencil only partially overlaps, or does not
    overlap at all, the reference bounding box passed to `to_pixels`.

    The 1 arcsec circle used throughout has the fixed tight pixel bounding box
    ``Box.factory[-3:18, -5:16]`` under `_make_wcs`, so the reference boxes
    below produce exactly predictable intersections.
    """

    # Reference boxes relative to the circle's tight pixel bbox
    # [y=-3:18, x=-5:16].
    TIGHT_BOX = Box.factory[-3:18, -5:16]
    PARTIAL_BOX = Box.factory[5:30, 5:30]
    PARTIAL_CLIPPED = Box.factory[5:18, 5:16]
    INSIDE_STENCIL_BOX = Box.factory[12:18, 12:16]
    TOUCHING_BOX = Box.factory[-3:18, 16:30]
    DISJOINT_BOX = Box.factory[100:120, 100:120]

    def setUp(self) -> None:
        self.wcs = _make_wcs()
        self.projection = SkyProjection.from_fits_wcs(self.wcs, GeneralFrame(unit=u.pix))

    def _circle(self, *, clip: bool) -> SkyCircle:
        return SkyCircle(CENTER, _arcsec(1.0), clip=clip)

    def test_clip_false_raises_on_partial_overlap(self) -> None:
        with self.assertRaises(StencilNotContainedError):
            self._circle(clip=False).to_pixels(self.projection, self.PARTIAL_BOX)

    def test_clip_false_raises_when_box_inside_stencil(self) -> None:
        with self.assertRaises(StencilNotContainedError):
            self._circle(clip=False).to_pixels(self.projection, self.INSIDE_STENCIL_BOX)

    def test_clip_false_raises_when_disjoint(self) -> None:
        with self.assertRaises(StencilNotContainedError):
            self._circle(clip=False).to_pixels(self.projection, self.DISJOINT_BOX)

    def test_clip_true_raises_when_touching(self) -> None:
        # The box starts one pixel beyond the tight bbox's max x, so the two
        # share no pixel and clipping cannot produce an overlap.
        with self.assertRaises(StencilNotContainedError):
            self._circle(clip=True).to_pixels(self.projection, self.TOUCHING_BOX)

    def test_clip_true_raises_when_disjoint(self) -> None:
        with self.assertRaises(StencilNotContainedError):
            self._circle(clip=True).to_pixels(self.projection, self.DISJOINT_BOX)

    def test_clip_true_unchanged_when_contained(self) -> None:
        # A fully contained stencil keeps its tight bbox even when clipping.
        _check_to_pixel(
            self,
            self._circle(clip=True),
            self.wcs,
            box=TEST_BOX,
            expected_bbox=self.TIGHT_BOX,
            max_missing=2,
            max_extra=2,
        )

    def test_clip_true_clips_to_intersection_on_partial_overlap(self) -> None:
        _check_to_pixel(
            self,
            self._circle(clip=True),
            self.wcs,
            box=self.PARTIAL_BOX,
            expected_bbox=self.PARTIAL_CLIPPED,
            max_missing=2,
            max_extra=2,
        )

    def test_clip_true_clips_to_box_when_box_inside_stencil(self) -> None:
        _check_to_pixel(
            self,
            self._circle(clip=True),
            self.wcs,
            box=self.INSIDE_STENCIL_BOX,
            expected_bbox=self.INSIDE_STENCIL_BOX,
            max_missing=2,
            max_extra=2,
        )


class StencilFitsMetadataTestCase(unittest.TestCase):
    """`SkyStencil.to_fits_metadata` returns an `astropy.io.fits.Header` whose
    cards carry the descriptive comments.
    """

    def test_circle(self) -> None:
        circle = SkyCircle(CENTER, _arcsec(1.0))
        header = circle.to_fits_metadata()
        self.assertIsInstance(header, astropy.io.fits.Header)
        self.assertEqual(header["ST_TYPE"], "CIRCLE")
        self.assertEqual(header.comments["ST_TYPE"], "Type of stencil used to create this cutout")
        self.assertAlmostEqual(header["ST_RA"], 12.0)
        self.assertAlmostEqual(header["ST_DEC"], 13.0)
        self.assertAlmostEqual(header["ST_RAD"], (1.0 * u.arcsec).to_value(u.deg))
        self.assertEqual(header.comments["ST_RAD"], "[deg] Circle radius")

    def test_polygon(self) -> None:
        polygon = SkyCircle(CENTER, _arcsec(2.0)).to_polygon(n_vertices=4)
        header = polygon.to_fits_metadata()
        self.assertIsInstance(header, astropy.io.fits.Header)
        self.assertEqual(header["ST_TYPE"], "POLYGON")
        self.assertEqual(header.comments["ST_TYPE"], "Type of stencil used to create this cutout")
        self.assertIn("ST_RA00", header)
        self.assertIn("ST_DEC00", header)
        self.assertEqual(header.comments["ST_RA00"], "[deg] Vertex 0 Right Ascension")
        self.assertEqual(header.comments["ST_DEC00"], "[deg] Vertex 0 Declination")


class NonIcrsInputTestCase(unittest.TestCase):
    """Non-ICRS inputs are converted to ICRS, and the caller's coordinates are
    not mutated.
    """

    def test_circle(self) -> None:
        galactic_center = SkyCoord(l=120.0 * u.deg, b=30.0 * u.deg, frame="galactic")
        expected = galactic_center.icrs
        circle = SkyCircle(galactic_center, _arcsec(1.0))
        header = circle.to_fits_metadata()
        self.assertAlmostEqual(header["ST_RA"], expected.ra.deg)
        self.assertAlmostEqual(header["ST_DEC"], expected.dec.deg)
        self.assertEqual(galactic_center.frame.name, "galactic")

    def test_polygon(self) -> None:
        galactic_vertices = SkyCoord(
            l=[120.0, 121.0, 120.5] * u.deg, b=[30.0, 30.0, 31.0] * u.deg, frame="galactic"
        )
        expected = galactic_vertices.icrs
        polygon = SkyPolygon(galactic_vertices)
        header = polygon.to_fits_metadata()
        self.assertAlmostEqual(header["ST_RA00"], expected[0].ra.deg)
        self.assertAlmostEqual(header["ST_DEC00"], expected[0].dec.deg)
        self.assertEqual(galactic_vertices.frame.name, "galactic")


if __name__ == "__main__":
    unittest.main()
