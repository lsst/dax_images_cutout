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
import numpy as np

from lsst.dax.images.cutout._fits_projection import projection_and_bbox_from_fits_header
from lsst.images import Box, SkyProjection

try:
    from lsst.afw.geom import getImageXY0FromMetadata, makeSkyWcs
    from lsst.daf.base import PropertyList
    from lsst.geom import Box2I, Extent2I

    HAVE_AFW = True
except ImportError:
    HAVE_AFW = False


def _make_header() -> astropy.io.fits.Header:
    """Make a primary gnomonic sky WCS plus an 'A' alternate WCS for XY0."""
    header = astropy.io.fits.Header()
    cards = {
        "WCSAXES": 2,
        "CRPIX1": 10.0,
        "CRPIX2": 20.0,
        "CRVAL1": 12.0,
        "CRVAL2": 13.0,
        "CD1_1": -2.78e-05,
        "CD2_2": 2.78e-05,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1A": 1.0,
        "CRPIX2A": 1.0,
        "CRVAL1A": 100.0,
        "CRVAL2A": 200.0,
        "CD1_1A": 1.0,
        "CD2_2A": 1.0,
        "CTYPE1A": "LINEAR",
        "CTYPE2A": "LINEAR",
    }
    for key, value in cards.items():
        header[key] = value
    return header


class FitsProjectionTestCase(unittest.TestCase):
    """Tests for the FITS-header projection helper (no afw required)."""

    def setUp(self) -> None:
        self.header = _make_header()
        # (ny, nx) as returned by astropy ``hdu.shape``.
        self.shape = (64, 48)

    def test_bbox_origin_and_size(self) -> None:
        _, bbox = projection_and_bbox_from_fits_header(self.header, self.shape)
        self.assertIsInstance(bbox, Box)
        # The "A" WCS has CRPIX*A = 1 and CRVAL*A = (100, 200), so grid (1, 1)
        # maps to the parent origin (100, 200); the size comes from ``shape``.
        self.assertEqual(bbox.x.start, 100)
        self.assertEqual(bbox.y.start, 200)
        self.assertEqual(bbox.x.size, self.shape[1])
        self.assertEqual(bbox.y.size, self.shape[0])

    def test_reference_pixel_maps_to_crval(self) -> None:
        projection, bbox = projection_and_bbox_from_fits_header(self.header, self.shape)
        self.assertIsInstance(projection, SkyProjection)
        # The projection is in parent pixel coordinates; the primary reference
        # pixel CRPIX (1-based grid) sits at parent CRPIX - 1 + origin and must
        # map to CRVAL.
        ref_x = self.header["CRPIX1"] - 1 + bbox.x.start
        ref_y = self.header["CRPIX2"] - 1 + bbox.y.start
        sky = projection.pixel_to_sky(x=np.array([ref_x]), y=np.array([ref_y]))
        self.assertAlmostEqual(float(sky.ra.deg[0]), self.header["CRVAL1"], places=6)
        self.assertAlmostEqual(float(sky.dec.deg[0]), self.header["CRVAL2"], places=6)


@unittest.skipUnless(HAVE_AFW, "lsst.afw/lsst.geom not available")
class FitsProjectionAfwReferenceTestCase(unittest.TestCase):
    """Cross-check the helper against the legacy afw implementation."""

    def setUp(self) -> None:
        self.header = _make_header()
        self.shape = (64, 48)

    def test_projection_matches_afw(self) -> None:
        projection, bbox = projection_and_bbox_from_fits_header(self.header, self.shape)
        pl = PropertyList()
        pl.update(self.header)
        wcs = makeSkyWcs(pl)
        xs = np.array([bbox.x.start + 5.0, bbox.x.start + 30.0])
        ys = np.array([bbox.y.start + 7.0, bbox.y.start + 25.0])
        sky = projection.pixel_to_sky(x=xs, y=ys)
        for i in range(len(xs)):
            reference = wcs.pixelToSky(float(xs[i]), float(ys[i]))
            self.assertAlmostEqual(float(sky.ra.deg[i]), reference.getRa().asDegrees(), places=6)
            self.assertAlmostEqual(float(sky.dec.deg[i]), reference.getDec().asDegrees(), places=6)

    def test_bbox_matches_afw_reference(self) -> None:
        _, bbox = projection_and_bbox_from_fits_header(self.header, self.shape)
        # afw reference for the parent bbox (XY0 from 'A' WCS + dimensions).
        pl = PropertyList()
        pl.update(self.header)
        xy0 = getImageXY0FromMetadata(pl, "A", strip=False)
        dimensions = Extent2I(self.shape[1], self.shape[0])
        self.assertEqual(bbox, Box.from_legacy(Box2I(xy0, dimensions)))


if __name__ == "__main__":
    unittest.main()
