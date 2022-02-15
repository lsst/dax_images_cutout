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

import lsst.daf.butler
import lsst.geom
import lsst.resources
import lsst.utils.tests
from lsst.image_cutout_backend import ImageCutoutBackend, projection_finders, stencils


class TestImageCutoutsBackend(lsst.utils.tests.TestCase):
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

        self.dataId = dict(patch=24, tract=3828, band="r", skymap="DC2")

    def test_extract_ref(self):
        """Test that extract_ref produces a reasonable cutout."""
        dataRef = self.butler.registry.findDataset("deepCoadd_calexp", dataId=self.dataId)

        with tempfile.TemporaryDirectory() as tempdir:
            cutoutBackend = ImageCutoutBackend(self.butler, self.projectionFinder, tempdir)
            result = cutoutBackend.extract_ref(self.stencil, dataRef)
            box = result.cutout.getBBox()
            self.assertEqual(box.width, 101)
            self.assertEqual(box.height, 101)
            # The galaxy should be near the center of the image.
            self.assertFloatsAlmostEqual(result.cutout.image.array[50, 49], 2.083247661590576)


if __name__ == "__main__":
    unittest.main()
