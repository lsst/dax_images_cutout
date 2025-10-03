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

import lsst.afw.image
import lsst.daf.butler
import lsst.geom
import lsst.resources
import lsst.utils.tests
from lsst.dax.images.cutout import CutoutMode, ImageCutoutFactory, projection_finders, stencils


class TestImageCutoutsBackend(lsst.utils.tests.TestCase):
    """Tests for ImageCutoutsBackend."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.data_dir = lsst.utils.getPackageDir("testdata_image_cutouts")
        except LookupError:
            raise unittest.SkipTest("testdata_image_cutouts not setup.") from None

    def setUp(self):
        collection = "2.2i/runs/test-med-1/w_2022_03/DM-33223/20220118T193330Z"
        self.butler = lsst.daf.butler.Butler(os.path.join(self.data_dir, "repo"), collections=collection)

        # Centered on a galaxy
        point = lsst.geom.SpherePoint(56.6400770 * lsst.geom.degrees, -36.4492250 * lsst.geom.degrees)
        radius = 10 * lsst.geom.arcseconds
        self.stencil = stencils.SkyCircle(point, radius)

        self.projectionFinders = (
            projection_finders.ReadComponents(),
            projection_finders.ReadComponentsAstropyFits(),
        )

        self.dataId = {"patch": 24, "tract": 3828, "band": "r", "skymap": "DC2"}

    def test_extract_ref(self):
        """Test that extract_ref produces a reasonable cutout."""
        dataRef = self.butler.registry.findDataset("deepCoadd_calexp", dataId=self.dataId)

        with tempfile.TemporaryDirectory() as tempdir:
            for cutout_mode in CutoutMode:
                # Try each available projection finder at least once.
                proj_finder = self.projectionFinders[cutout_mode.value % len(self.projectionFinders)]
                cutoutBackend = ImageCutoutFactory(self.butler, proj_finder, tempdir)
                result = cutoutBackend.extract_ref(self.stencil, dataRef, cutout_mode=cutout_mode)
                # The galaxy should be near the center of the image.
                match result.cutout:
                    case lsst.afw.image.Exposure() | lsst.afw.image.MaskedImage():
                        box = result.cutout.getBBox()
                        array = result.cutout.image.array
                    case lsst.afw.image.Image():
                        array = result.cutout.array
                        box = result.cutout.getBBox()
                    case astropy.io.fits.HDUList():
                        hdu = result.cutout[1]
                        array = hdu.data
                        # Only checks the shape.
                        box = lsst.geom.Box2I(lsst.geom.Point2I([1, 1]), lsst.geom.Extent2I(hdu.shape))
                    case _:
                        raise RuntimeError(f"Unexpected cutout type: {type(result.cutout)}")
                self.assertEqual(box.width, 101)
                self.assertEqual(box.height, 101)
                self.assertFloatsAlmostEqual(array[50, 49], 2.083247661590576)
                self.assertFloatsAlmostEqual(array[1, 1], -0.14575983583927155)
                self.assertFloatsAlmostEqual(array[100, 100], 0.08674515783786774)

                output = cutoutBackend.process_ref(self.stencil, dataRef, cutout_mode=cutout_mode)
                self.assertTrue(output.exists())

                output = cutoutBackend.process_uuid(self.stencil, dataRef.id, cutout_mode=cutout_mode)
                self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
