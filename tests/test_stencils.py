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
import astropy.units
import lsst.sphgeom
import numpy as np
from lsst.afw.geom import SkyWcs, makeCdMatrix, makeSkyWcs
from lsst.afw.image import Mask
from lsst.geom import Angle, Box2I, Point2D, Point2I, SpherePoint, arcseconds, degrees
from lsst.dax.images.cutout.stencils import SkyCircle, SkyStencil


class SkyCircleTestCase(unittest.TestCase):
    """Tests for `SkyCircle."""

    def setUp(self) -> None:
        self.center = SpherePoint(12.0, 13.0, degrees)
        self.instance = SkyCircle(self.center, Angle(1.0, arcseconds))

    def test_from_astropy(self) -> None:
        """Test that the from_astropy factory function is equivalent to
        passing the same values to the constructor, by comparing `region`
        result.

        This is a pretty circular test, but it's at least runs a lot of code
        to make sure it's not completely broken.
        """
        other = SkyCircle.from_astropy(
            astropy.coordinates.SkyCoord(
                frame="icrs", ra=12.0 * astropy.units.deg, dec=13.0 * astropy.units.deg
            ),
            astropy.coordinates.Angle(1.0 * astropy.units.arcsec),
        )
        self.assertEqual(self.instance.region, other.region)

    def test_repr(self) -> None:
        """Test that eval(repr(...)) round-trips."""
        self.assertEqual(eval(repr(self.instance)).region, self.instance.region)

    def test_to_pixel(self) -> None:
        """Test `SkyCircle.to_pixel`, implicitly testing the
        `PixelStencil` implementation it returns.
        """
        # WCS is gnomonic, with projection point at the circle center and
        # 0.1" pixels, so 1" radius circle will have be about 20 pixels across
        # the diameter.  Make the offset arbitray but nonzero.
        wcs = makeSkyWcs(Point2D(5.0, 7.0), self.center, makeCdMatrix(0.1 * arcseconds))
        # Bounding box should be roughly twice the size of the circle.
        # Make it slightly bigger in x to catch x<->y transposition bugs.
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        # Check to_pixel(...).set_mask(...) against a brute-force
        # implementation.  The max_missing threshold (and implicitly, the
        # max_extra=0 threshold) was set after inspection with plot=True to
        # prevent accidental regressions.
        _check_to_pixel(self, self.instance, wcs, bbox, max_missing=12)

    def test_to_polygon(self) -> None:
        """Test conversion to a `SkyPolygon`.

        This provides a lot of test coverage for `SkyPolygon` as well,
        but it doesn't check the non-convex case (hard, since sphgeom doesn't
        have a non-convex polygon), and it doesn't check conversion from
        astropy or the documented orientation conventions.
        """
        # Same WCS and bbox as used in test_to_pixel.
        wcs = makeSkyWcs(Point2D(5.0, 7.0), self.center, makeCdMatrix(0.1 * arcseconds))
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        # Convert to polygon.
        polygon_stencil = self.instance.to_polygon()
        # Make sure sphgeom regions at least aren't disjoint; in exact math
        # the true circle would contain the polygon, but round-off error makes
        # that not guaranteed with floats.
        self.assertNotEqual(
            self.instance.region.relate(polygon_stencil.region.getBoundingCircle()), lsst.sphgeom.DISJOINT
        )
        # Check the polygon's to_pixel implementation.  The max_missing
        # threshold (and implicitly, the max_extra=0 threshold) was set after
        # inspection with plot=True to prevent accidental regressions.
        _check_to_pixel(self, polygon_stencil, wcs, bbox, max_missing=4)


def _brute_force_stencil_array(sky_stencil: SkyStencil, wcs: SkyWcs, bbox: Box2I) -> np.ndarray:
    """Create a boolean Numpy array of a `SkyStencil` by transforming and
    checking every pixel within it.

    Parameters
    ----------
    sky_stencil : `SkyStencil`
        Stencil to create an image of.
    wcs : `SkyWcs`
        WCS that transforms sky coordinates to pixel coordinates.
    bbox : `Box2I`
        Bounding box that must contain the pixel-coordinate stencil.

    Returns
    -------
    array : `np.ndarray`
        2-d boolean array of shape ``(bbox.getWidth(), bbox.getHeight())``.
        `True` pixels are those whose WCS-transformed centers are within
        ``sky_stencil``.
    """
    # Make Numpy arrays of the pixel coordinates.
    x1 = np.arange(bbox.getBeginX(), bbox.getEndX())
    y1 = np.arange(bbox.getBeginY(), bbox.getEndY())
    x2, y2 = np.meshgrid(x1, y1)
    # Stuff those into one (2, nPoints) array so we can pass it to the AST
    # mapping inside a WCS.
    pixels = np.zeros((2, bbox.getArea()), dtype=float)
    pixels[0, :] = x2.flatten()
    pixels[1, :] = y2.flatten()
    # Transform all those points to sky coordinates, yielding another
    # (2, nPoint) array.
    sky = wcs.getTransform().getMapping().applyForward(pixels)
    # Test which of those points are inside the sky stencil's sphgeom region,
    # and reshape back to image coordinates.
    contained = sky_stencil.region.contains(sky[0, :], sky[1, :])
    # Reshape the boolean 'contained' array back to an image and return it.
    return contained.reshape(bbox.getHeight(), bbox.getWidth())


def _check_to_pixel(
    test_case: unittest.TestCase,
    sky_stencil: SkyStencil,
    wcs: SkyWcs,
    bbox: Box2I,
    *,
    max_missing: int = 0,
    max_extra: int = 0,
    plot: bool = False,
) -> None:
    """Test helper function that checks a `SkyStencil.to_pixel` implementation.

    Parameters
    ----------
    test_case : `unittest.TestCase`
        Test case object that provides assertion methods.
    sky_stencil : `SkyStencil`
        Stencil to test.
    wcs : `SkyWcs`
        WCS that transforms sky coordinates to pixel coordinates.
    bbox : `Box2I`
        Bounding box that must contain the pixel-coordinate stencil.
    max_missing : `int`, optional
        The number of WCS-transformed pixels that are in the stencil but not
        masked by `PixelStencil.set_mask` must not exceed this number (defaults
        to 0).
    max_extra : `int`, optional
        The number of WCS-transformed pixels that are not in the stencil but
        are masked by `PixelStencil.set_mask` must not exceed this number
        (defaults to 0).
    plot : `bool`, optional
        If `True` (`False` is default), create an interactive matplotlib image
        of the comparison for human inspection.  This should never be set
        except when debugging or actively developing a new test, and even then
        it probably requires that the test not be run by pytest.
    """
    pixel_stencil = sky_stencil.to_pixels(wcs, bbox)
    test_case.assertTrue(bbox.contains(pixel_stencil.bbox))
    mask = Mask(bbox)
    mask.addMaskPlane("STENCIL")
    bits = mask.getPlaneBitMask("STENCIL")
    pixel_stencil.set_mask(mask, bits)
    check_array = _brute_force_stencil_array(sky_stencil, wcs, bbox)
    missing = np.logical_and(check_array, np.logical_not(mask.array & bits))
    extra = np.logical_and(mask.array & bits, np.logical_not(check_array))
    if plot:
        from matplotlib import pyplot

        display_array = np.zeros((bbox.getHeight(), bbox.getWidth(), 3), dtype=np.uint8)
        display_array[:, :, 0] = 255 * check_array
        display_array[:, :, 1] = 255 * (mask.array & bits).astype(bool)
        pyplot.imshow(display_array, origin="lower", interpolation="nearest")
        pyplot.title("red=check, green=SkyStencil.to_pixel, yellow=both")
        pyplot.show()
    test_case.assertLessEqual(sum(missing.flatten()), max_missing)
    test_case.assertLessEqual(sum(extra.flatten()), max_extra)


if __name__ == "__main__":
    unittest.main()
