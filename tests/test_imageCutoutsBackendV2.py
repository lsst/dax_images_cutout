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

import os.path
import tempfile
import unittest

import astropy.io.fits
import astropy.units as u

import lsst.images
import lsst.sphgeom
import lsst.utils.tests
from lsst.daf.butler import Butler
from lsst.dax.images.cutout import CutoutMode, ImageCutoutFactory, projection_finders, stencils

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class TestImageCutoutsBackendV2(lsst.utils.tests.TestCase):
    """Tests for ImageCutoutsBackend using lsst.images data models."""

    def setUp(self):
        collection = "LSSTCam/runs/DRP/DP2/v30_0_8/DM-55060/deep_coadd_rewrite/20260612T200929Z"
        self.butler = Butler.from_config(os.path.join(TESTDIR, "butler"), collections=collection)
        self.enterContext(self.butler)

        # Try: RA = 0:01:01.7  Dec = -3:02:13
        point = lsst.sphgeom.LonLat.fromDegrees(0.25708, -3.03694)
        radius = lsst.sphgeom.Angle((3 * u.arcsec).to_value(u.rad))
        self.stencil = stencils.SkyCircle(point, radius)

        self.projectionFinders = (
            projection_finders.ReadComponents(),
            projection_finders.ReadComponentsAstropyFits(),
        )

        self.dataId = {"patch": 76, "tract": 5428, "band": "g", "skymap": "lsst_cells_v2"}

    def test_extract_ref(self):
        """Test that extract_ref produces a reasonable cutout."""
        dataRef = self.butler.registry.findDataset("deep_coadd", dataId=self.dataId)

        with tempfile.TemporaryDirectory() as tempdir:
            for cutout_mode in CutoutMode:
                # Try each available projection finder at least once.
                proj_finder = self.projectionFinders[cutout_mode.value % len(self.projectionFinders)]
                cutoutBackend = ImageCutoutFactory(self.butler, proj_finder, tempdir)
                result = cutoutBackend.extract_ref(self.stencil, dataRef, cutout_mode=cutout_mode)
                match result.cutout:
                    case lsst.images.MaskedImage():
                        box = result.cutout.bbox
                        array = result.cutout.image.array
                    case lsst.images.Image():
                        array = result.cutout.array
                        box = result.cutout.bbox
                    case astropy.io.fits.HDUList():
                        hdu = result.cutout[1]
                        array = hdu.data
                        # Only checks the shape.
                        box = lsst.images.Box.factory[1 : hdu.shape[0] + 1, 1 : hdu.shape[1] + 1]
                    case _:
                        raise RuntimeError(f"Unexpected cutout type: {type(result.cutout)}")

                with self.subTest(cutout_mode=cutout_mode):
                    self.assertEqual(box.x.size, 29)
                    self.assertEqual(box.y.size, 28)

                    # We are reading these values from the fuzzed data that
                    # has no scientific content but we can test that each
                    # cutout is returning the same value. These numbers have
                    # not been validated by external tooling and will change
                    # on redoing the fuzzing.
                    self.assertFloatsAlmostEqual(array[14, 15], 2.788926362991333)
                    self.assertFloatsAlmostEqual(array[1, 1], 7.183607578277588)
                    self.assertFloatsAlmostEqual(array[27, 28], -11.042882919311523)

                    output = cutoutBackend.process_ref(self.stencil, dataRef, cutout_mode=cutout_mode)
                    self.assertTrue(output.exists())

                    output = cutoutBackend.process_uuid(self.stencil, dataRef.id, cutout_mode=cutout_mode)
                    self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
