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

import astropy.coordinates
import astropy.io.fits
import astropy.units as u
import astropy.wcs
import numpy as np

import lsst.sphgeom
from lsst.dax.images.cutout.stencils import (
    MaskBackend,
    SkyCircle,
    SkyStencil,
    StencilNotContainedError,
)
from lsst.images import Box, GeneralFrame, Mask, MaskPlane, MaskSchema, SkyProjection
from lsst.sphgeom import Angle, LonLat, UnitVector3d  # noqa: F401  (used by eval(repr))

# Bounding box for the cutout tests, in [y, x] (stop exclusive).  Slightly
# bigger in x to catch x<->y transposition bugs.
TEST_BOX = Box.factory[-13:28, -16:27]


def _arcsec(value: float) -> Angle:
    """Return a `lsst.sphgeom.Angle` for ``value`` arcseconds."""
    return Angle((value * u.arcsec).to_value(u.rad))


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


class ModuleHelpersTestCase(unittest.TestCase):
    """Tests for module-level helpers that survive the rewrite."""

    def test_mask_backend_members(self) -> None:
        self.assertEqual({b.name for b in MaskBackend}, {"AST", "SPHGEOM"})


class SkyCircleTestCase(unittest.TestCase):
    """Tests for `SkyCircle`."""

    def setUp(self) -> None:
        self.center = LonLat.fromDegrees(12.0, 13.0)
        self.instance = SkyCircle(self.center, _arcsec(1.0))

    def test_from_astropy(self) -> None:
        other = SkyCircle.from_astropy(
            astropy.coordinates.SkyCoord(
                frame="icrs", ra=12.0 * astropy.units.deg, dec=13.0 * astropy.units.deg
            ),
            astropy.coordinates.Angle(1.0 * astropy.units.arcsec),
        )
        self.assertEqual(self.instance.region, other.region)

    def test_repr(self) -> None:
        self.assertEqual(eval(repr(self.instance)).region, self.instance.region)

    def test_to_pixel(self) -> None:
        _check_to_pixel(self, self.instance, _make_wcs(), backend=MaskBackend.AST, max_missing=2, max_extra=2)

    def test_to_polygon(self) -> None:
        polygon_stencil = self.instance.to_polygon()
        self.assertNotEqual(
            self.instance.region.relate(polygon_stencil.region.getBoundingCircle()), lsst.sphgeom.DISJOINT
        )
        _check_to_pixel(
            self, polygon_stencil, _make_wcs(), backend=MaskBackend.AST, max_missing=6, max_extra=6
        )

    def test_ast_sky_region_circle_contains_center(self) -> None:
        region = self.instance._ast_sky_region()
        self.assertTrue(
            region.pointinregion([self.center.getLon().asRadians(), self.center.getLat().asRadians()])
        )

    def test_to_pixel_sphgeom(self) -> None:
        _check_to_pixel(
            self, self.instance, _make_wcs(), backend=MaskBackend.SPHGEOM, max_missing=0, max_extra=0
        )

    def test_to_pixel_sphgeom_polygon(self) -> None:
        polygon_stencil = self.instance.to_polygon()
        _check_to_pixel(
            self, polygon_stencil, _make_wcs(), backend=MaskBackend.SPHGEOM, max_missing=0, max_extra=0
        )


class SkyPolygonTestCase(unittest.TestCase):
    """Tests for `SkyPolygon` orientation handling."""

    def setUp(self) -> None:
        self.instance = SkyCircle(LonLat.fromDegrees(12.0, 13.0), _arcsec(2.0)).to_polygon(n_vertices=8)

    def test_ast_sky_region_polygon_contains_centroid(self) -> None:
        region = self.instance._ast_sky_region()
        lonlat = lsst.sphgeom.LonLat(self.instance.region.getCentroid())
        self.assertTrue(region.pointinregion([lonlat.getLon().asRadians(), lonlat.getLat().asRadians()]))


class BackendComparisonTestCase(unittest.TestCase):
    """Assert the AST and sphgeom backends agree on bbox and masked pixels."""

    def setUp(self) -> None:
        self.center = LonLat.fromDegrees(12.0, 13.0)
        self.projection = SkyProjection.from_fits_wcs(_make_wcs(), GeneralFrame(unit=u.pix))
        self.box = TEST_BOX

    def _masked_array(self, stencil: SkyStencil, backend: MaskBackend) -> tuple[np.ndarray, Box]:
        pixel_stencil = stencil.to_pixels(self.projection, self.box, backend=backend)
        mask = Mask(schema=MaskSchema([MaskPlane("STENCIL", "stencil coverage")]), bbox=self.box)
        pixel_stencil.set_mask(mask, "STENCIL")
        return mask.get("STENCIL"), pixel_stencil.bbox

    def test_backends_agree_circle(self) -> None:
        circle = SkyCircle(self.center, _arcsec(1.0))
        ast_mask, ast_box = self._masked_array(circle, MaskBackend.AST)
        sph_mask, sph_box = self._masked_array(circle, MaskBackend.SPHGEOM)
        self.assertEqual(ast_box, sph_box)
        self.assertEqual(int(np.sum(ast_mask != sph_mask)), 0)

    def test_backends_agree_polygon(self) -> None:
        polygon = SkyCircle(self.center, _arcsec(1.0)).to_polygon()
        ast_mask, ast_box = self._masked_array(polygon, MaskBackend.AST)
        sph_mask, sph_box = self._masked_array(polygon, MaskBackend.SPHGEOM)
        self.assertEqual(ast_box, sph_box)
        self.assertLessEqual(int(np.sum(ast_mask != sph_mask)), 12)

    def test_set_mask_covered_false_marks_outside(self) -> None:
        """``set_mask(covered=False)`` flags exactly the pixels the stencil
        does not cover, including the region of the mask outside the stencil's
        bounding box.
        """
        circle = SkyCircle(self.center, _arcsec(1.0))
        pixel_stencil = circle.to_pixels(self.projection, self.box)

        inside = Mask(schema=MaskSchema([MaskPlane("STENCIL", "stencil coverage")]), bbox=self.box)
        pixel_stencil.set_mask(inside, "STENCIL")

        outside = Mask(schema=MaskSchema([MaskPlane("STENCIL", "stencil coverage")]), bbox=self.box)
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


def _brute_force_stencil_array(sky_stencil: SkyStencil, wcs: astropy.wcs.WCS, box: Box) -> np.ndarray:
    """Make a boolean ``(ny, nx)`` array, `True` where a center is inside.

    The pixel grid is transformed to the sky with the FITS WCS (independent of
    the `SkyProjection` under test) and tested against the stencil's sphgeom
    region.
    """
    grid = box.meshgrid()
    sky = wcs.pixel_to_world(grid.x.ravel(), grid.y.ravel())
    contained = sky_stencil.region.contains(sky.ra.rad, sky.dec.rad)
    return contained.reshape(box.shape)


def _check_to_pixel(
    test_case: unittest.TestCase,
    sky_stencil: SkyStencil,
    wcs: astropy.wcs.WCS,
    *,
    box: Box = TEST_BOX,
    expected_bbox: Box | None = None,
    backend: MaskBackend = MaskBackend.AST,
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
    pixel_stencil = sky_stencil.to_pixels(projection, box, backend=backend)
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


class StencilContainmentTestCase(unittest.TestCase):
    """Clipping and raising when a stencil only partially overlaps, or does not
    overlap at all, the reference bounding box passed to `to_pixels`.

    The 1 arcsec circle used throughout has the fixed tight pixel bounding box
    ``Box.factory[-3:18, -5:16]`` under `_make_wcs`, so the reference boxes
    below produce exactly predictable intersections.
    """

    # Reference boxes relative to the circle's tight pixel bbox
    # [y=-3:18, x=-5:16].
    PARTIAL_BOX = Box.factory[5:30, 5:30]
    PARTIAL_CLIPPED = Box.factory[5:18, 5:16]
    INSIDE_STENCIL_BOX = Box.factory[12:18, 12:16]
    TOUCHING_BOX = Box.factory[-3:18, 16:30]
    DISJOINT_BOX = Box.factory[100:120, 100:120]

    # Per-backend rasterization tolerance, matching the existing circle tests.
    BACKEND_TOLERANCE = {MaskBackend.AST: 2, MaskBackend.SPHGEOM: 0}

    def setUp(self) -> None:
        self.center = LonLat.fromDegrees(12.0, 13.0)
        self.wcs = _make_wcs()
        self.projection = SkyProjection.from_fits_wcs(self.wcs, GeneralFrame(unit=u.pix))

    def _circle(self, *, clip: bool) -> SkyCircle:
        return SkyCircle(self.center, _arcsec(1.0), clip=clip)

    # Box resolution happens in `to_pixels` before any mask backend is
    # selected, so the raising behavior is backend-independent; the default
    # backend is sufficient for the raising tests below.

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
        for backend, tolerance in self.BACKEND_TOLERANCE.items():
            with self.subTest(backend=str(backend)):
                _check_to_pixel(
                    self,
                    self._circle(clip=True),
                    self.wcs,
                    box=TEST_BOX,
                    expected_bbox=Box.factory[-3:18, -5:16],
                    backend=backend,
                    max_missing=tolerance,
                    max_extra=tolerance,
                )

    def test_clip_true_clips_to_intersection_on_partial_overlap(self) -> None:
        for backend, tolerance in self.BACKEND_TOLERANCE.items():
            with self.subTest(backend=str(backend)):
                _check_to_pixel(
                    self,
                    self._circle(clip=True),
                    self.wcs,
                    box=self.PARTIAL_BOX,
                    expected_bbox=self.PARTIAL_CLIPPED,
                    backend=backend,
                    max_missing=tolerance,
                    max_extra=tolerance,
                )

    def test_clip_true_clips_to_box_when_box_inside_stencil(self) -> None:
        for backend, tolerance in self.BACKEND_TOLERANCE.items():
            with self.subTest(backend=str(backend)):
                _check_to_pixel(
                    self,
                    self._circle(clip=True),
                    self.wcs,
                    box=self.INSIDE_STENCIL_BOX,
                    expected_bbox=self.INSIDE_STENCIL_BOX,
                    backend=backend,
                    max_missing=tolerance,
                    max_extra=tolerance,
                )


class StencilFitsMetadataTestCase(unittest.TestCase):
    """`SkyStencil.to_fits_metadata` returns an `astropy.io.fits.Header` whose
    cards carry the descriptive comments.
    """

    def test_circle(self) -> None:
        circle = SkyCircle(LonLat.fromDegrees(12.0, 13.0), _arcsec(1.0))
        header = circle.to_fits_metadata()
        self.assertIsInstance(header, astropy.io.fits.Header)
        self.assertEqual(header["ST_TYPE"], "CIRCLE")
        self.assertEqual(header.comments["ST_TYPE"], "Type of stencil used to create this cutout")
        self.assertAlmostEqual(header["ST_RA"], 12.0)
        self.assertAlmostEqual(header["ST_DEC"], 13.0)
        self.assertAlmostEqual(header["ST_RAD"], (1.0 * u.arcsec).to_value(u.deg))
        self.assertEqual(header.comments["ST_RAD"], "[deg] Circle radius")

    def test_polygon(self) -> None:
        polygon = SkyCircle(LonLat.fromDegrees(12.0, 13.0), _arcsec(2.0)).to_polygon(n_vertices=4)
        header = polygon.to_fits_metadata()
        self.assertIsInstance(header, astropy.io.fits.Header)
        self.assertEqual(header["ST_TYPE"], "POLYGON")
        self.assertEqual(header.comments["ST_TYPE"], "Type of stencil used to create this cutout")
        self.assertIn("ST_RA00", header)
        self.assertIn("ST_DEC00", header)
        self.assertEqual(header.comments["ST_RA00"], "[deg] Vertex 0 Right Ascension")
        self.assertEqual(header.comments["ST_DEC00"], "[deg] Vertex 0 Declination")


if __name__ == "__main__":
    unittest.main()
