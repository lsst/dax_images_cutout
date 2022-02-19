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

import os.path
import tempfile
import unittest

import numpy as np
import PIL

import lsst.utils.tests
from lsst.image_cutout_backend import ImageCutoutBackend, RgbImageCutout, projection_finders, stencils


class TestRgbImageCutouts(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.data_dir = lsst.utils.getPackageDir("testdata_image_cutouts")
        except LookupError:
            raise unittest.SkipTest("testdata_image_cutouts not setup.")

    def setUp(self):
        collection = "2.2i/runs/test-med-1/w_2022_03/DM-33223/20220118T193330Z"
        self.butler = lsst.daf.butler.Butler(os.path.join(self.data_dir, "repo"), collections=collection)

        # Centered on a galaxy
        point = lsst.geom.SpherePoint(56.6400770 * lsst.geom.degrees, -36.4492250 * lsst.geom.degrees)
        radius = 10 * lsst.geom.arcseconds
        self.stencil = stencils.SkyCircle(point, radius)

        self.projectionFinder = projection_finders.ReadComponents()

        dataId = dict(patch=24, tract=3828, skymap="DC2")
        dataId['band'] = 'g'
        self.dataRefB = self.butler.registry.findDataset("deepCoadd_calexp", dataId=dataId)
        dataId['band'] = 'r'
        self.dataRefG = self.butler.registry.findDataset("deepCoadd_calexp", dataId=dataId)
        dataId['band'] = 'i'
        self.dataRefR = self.butler.registry.findDataset("deepCoadd_calexp", dataId=dataId)

    def test_extract_ref(self):
        """Extract images with g->B, r->G, i->R."""
        with tempfile.TemporaryDirectory() as tempdir:
            backend = ImageCutoutBackend(self.butler, self.projectionFinder, tempdir)
            cutoutBackend = RgbImageCutout(backend)
            result = cutoutBackend.extract_ref(self.stencil, self.dataRefR, self.dataRefG, self.dataRefB)

            # Check that the bboxes are all the same size.
            box = result.r.cutout.getBBox()
            self.assertEqual(box, result.g.cutout.getBBox())
            self.assertEqual(box, result.b.cutout.getBBox())
            self.assertEqual(box.width, 101)
            self.assertEqual(box.height, 101)
            # The galaxy should be near the center of the image.
            self.assertFloatsAlmostEqual(result.r.cutout.image.array[50, 49], 2.3940088748931885)
            self.assertFloatsAlmostEqual(result.g.cutout.image.array[50, 49], 2.083247661590576)
            self.assertFloatsAlmostEqual(result.b.cutout.image.array[50, 49], 1.2457562685012817)

    def test_process_ref_png(self):
        """Extract images with g->B, r->G, i->R and test that a PNG is saved.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            backend = ImageCutoutBackend(self.butler, self.projectionFinder, tempdir)
            cutoutBackend = RgbImageCutout(backend)
            result = cutoutBackend.process_ref_png(self.stencil, self.dataRefR, self.dataRefG, self.dataRefB)

            self.assertTrue(result.exists())
            expected_image_path = os.path.join(os.path.dirname(__file__),
                                               "data/galaxy-test-Q_10-stretch_0.5.png")
            expected_image = PIL.Image.open(expected_image_path)
            result_image = PIL.Image.open(result)
            # Check the pixel values for equality; we test metadata later.
            np.testing.assert_array_equal(np.asarray(expected_image), np.asarray(result_image))

            result_image.load()  # PIL does not guarantee that EXIF data is present until load().
            import ipdb; ipdb.set_trace();
            print(result_image.info)


if __name__ == "__main__":
    unittest.main()
