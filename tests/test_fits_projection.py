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

from lsst.afw.geom import getImageXY0FromMetadata
from lsst.daf.base import PropertyList
from lsst.dax.images.cutout._fits_projection import projection_and_bbox_from_fits_header
from lsst.geom import Box2I, Extent2I
from lsst.images import Box, SkyProjection


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
    """Tests for the FITS-header projection helper."""

    def setUp(self) -> None:
        self.header = _make_header()
        # (ny, nx) as returned by astropy ``hdu.shape``.
        self.shape = (64, 48)

    def test_projection_round_trips_reference_pixel(self) -> None:
        projection, _ = projection_and_bbox_from_fits_header(self.header, self.shape)
        self.assertIsInstance(projection, SkyProjection)
        sky = projection.pixel_to_sky(x=np.array([10.0]), y=np.array([20.0]))
        # The reference pixel CRPIX maps to CRVAL.
        self.assertAlmostEqual(float(sky.ra.deg[0]), 12.0, places=6)
        self.assertAlmostEqual(float(sky.dec.deg[0]), 13.0, places=6)

    def test_bbox_matches_afw_reference(self) -> None:
        _, bbox = projection_and_bbox_from_fits_header(self.header, self.shape)
        self.assertIsInstance(bbox, Box)
        # afw reference for the parent bbox (XY0 from 'A' WCS + dimensions).
        pl = PropertyList()
        pl.update(self.header)
        xy0 = getImageXY0FromMetadata(pl, "A", strip=False)
        dimensions = Extent2I(self.shape[1], self.shape[0])
        self.assertEqual(bbox, Box.from_legacy(Box2I(xy0, dimensions)))


if __name__ == "__main__":
    unittest.main()
