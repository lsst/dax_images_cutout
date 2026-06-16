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
import astropy.units as u
import numpy as np

import lsst.sphgeom
from lsst.afw.geom import SkyWcs, makeCdMatrix, makeSkyWcs
from lsst.dax.images.cutout.stencils import (
    MaskBackend,
    SkyCircle,
    SkyStencil,
    _round_box_from_bounds,
)
from lsst.geom import Box2I, Point2D, Point2I, SpherePoint, arcseconds, degrees
from lsst.images import Box, GeneralFrame, Mask, MaskPlane, MaskSchema, SkyProjection
from lsst.sphgeom import Angle, LonLat, UnitVector3d  # noqa: F401  (used by eval(repr))


def _arcsec(value: float) -> Angle:
    """Return a `lsst.sphgeom.Angle` for ``value`` arcseconds."""
    return Angle((value * u.arcsec).to_value(u.rad))


class ModuleHelpersTestCase(unittest.TestCase):
    """Tests for module-level helpers that survive the rewrite."""

    def test_round_box_from_bounds(self) -> None:
        # x in [4.6, 9.4], y in [2.6, 5.4] -> box [3:6, 5:10] in [y, x].
        box = _round_box_from_bounds(4.6, 9.4, 2.6, 5.4)
        self.assertEqual(box, Box.factory[3:6, 5:10])

    def test_mask_backend_members(self) -> None:
        self.assertEqual({b.name for b in MaskBackend}, {"AST", "SPHGEOM"})


class SkyCircleTestCase(unittest.TestCase):
    """Tests for `SkyCircle`."""

    def setUp(self) -> None:
        self.center = LonLat.fromDegrees(12.0, 13.0)
        self.instance = SkyCircle(self.center, _arcsec(1.0))
        # An afw WCS at the same center, used for the brute-force reference and
        # to build a SkyProjection input.
        self.wcs_center = SpherePoint(12.0, 13.0, degrees)

    def _wcs(self) -> SkyWcs:
        return makeSkyWcs(Point2D(5.0, 7.0), self.wcs_center, makeCdMatrix(0.1 * arcseconds))

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
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        _check_to_pixel(
            self, self.instance, self._wcs(), bbox, backend=MaskBackend.AST, max_missing=2, max_extra=2
        )

    def test_to_polygon(self) -> None:
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        polygon_stencil = self.instance.to_polygon()
        self.assertNotEqual(
            self.instance.region.relate(polygon_stencil.region.getBoundingCircle()), lsst.sphgeom.DISJOINT
        )
        _check_to_pixel(
            self, polygon_stencil, self._wcs(), bbox, backend=MaskBackend.AST, max_missing=6, max_extra=6
        )

    def test_ast_sky_region_circle_contains_center(self) -> None:
        region = self.instance._ast_sky_region()
        self.assertTrue(
            region.pointinregion([self.center.getLon().asRadians(), self.center.getLat().asRadians()])
        )

    def test_to_pixel_sphgeom(self) -> None:
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        _check_to_pixel(
            self, self.instance, self._wcs(), bbox, backend=MaskBackend.SPHGEOM, max_missing=0, max_extra=0
        )

    def test_to_pixel_sphgeom_polygon(self) -> None:
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        polygon_stencil = self.instance.to_polygon()
        _check_to_pixel(
            self, polygon_stencil, self._wcs(), bbox, backend=MaskBackend.SPHGEOM, max_missing=0, max_extra=0
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
        wcs = makeSkyWcs(Point2D(5.0, 7.0), SpherePoint(12.0, 13.0, degrees), makeCdMatrix(0.1 * arcseconds))
        self.projection = SkyProjection.from_legacy(wcs, GeneralFrame(unit=u.pix))
        self.box = Box.from_legacy(Box2I(Point2I(-16, -13), Point2I(26, 27)))

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


def _brute_force_stencil_array(sky_stencil: SkyStencil, wcs: SkyWcs, bbox: Box2I) -> np.ndarray:
    """Make a boolean ``(ny, nx)`` array, `True` where a center is inside."""
    x1 = np.arange(bbox.getBeginX(), bbox.getEndX())
    y1 = np.arange(bbox.getBeginY(), bbox.getEndY())
    x2, y2 = np.meshgrid(x1, y1)
    pixels = np.zeros((2, bbox.getArea()), dtype=float)
    pixels[0, :] = x2.flatten()
    pixels[1, :] = y2.flatten()
    sky = wcs.getTransform().getMapping().applyForward(pixels)
    contained = sky_stencil.region.contains(sky[0, :], sky[1, :])
    return contained.reshape(bbox.getHeight(), bbox.getWidth())


def _check_to_pixel(
    test_case: unittest.TestCase,
    sky_stencil: SkyStencil,
    wcs: SkyWcs,
    bbox: Box2I,
    *,
    backend: MaskBackend = MaskBackend.AST,
    max_missing: int = 0,
    max_extra: int = 0,
    plot: bool = False,
) -> None:
    """Check a `SkyStencil.to_pixels` result against brute force."""
    projection = SkyProjection.from_legacy(wcs, GeneralFrame(unit=u.pix))
    box = Box.from_legacy(bbox)
    pixel_stencil = sky_stencil.to_pixels(projection, box, backend=backend)
    test_case.assertTrue(box.contains(pixel_stencil.bbox))
    mask = Mask(schema=MaskSchema([MaskPlane("STENCIL", "stencil coverage")]), bbox=box)
    pixel_stencil.set_mask(mask, "STENCIL")
    got = mask.get("STENCIL")
    check_array = _brute_force_stencil_array(sky_stencil, wcs, bbox)
    missing = np.logical_and(check_array, np.logical_not(got))
    extra = np.logical_and(got, np.logical_not(check_array))
    if plot:
        from matplotlib import pyplot

        display_array = np.zeros((bbox.getHeight(), bbox.getWidth(), 3), dtype=np.uint8)
        display_array[:, :, 0] = 255 * check_array
        display_array[:, :, 1] = 255 * got
        pyplot.imshow(display_array, origin="lower", interpolation="nearest")
        pyplot.title("red=check, green=SkyStencil.to_pixel, yellow=both")
        pyplot.show()
    test_case.assertLessEqual(int(missing.sum()), max_missing)
    test_case.assertLessEqual(int(extra.sum()), max_extra)


if __name__ == "__main__":
    unittest.main()
