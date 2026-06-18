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

import astropy

import lsst.daf.butler
import lsst.images
import lsst.sphgeom
import lsst.utils.tests
from lsst.dax.images.cutout import CutoutMode, ImageCutoutFactory, projection_finders, stencils

try:
    import lsst.afw.image
    import lsst.geom

    HAVE_AFW = True
except ImportError:
    HAVE_AFW = False


@unittest.skipUnless(HAVE_AFW, "lsst.afw not available")
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
        point = lsst.sphgeom.LonLat.fromDegrees(56.6400770, -36.4492250)
        radius = lsst.sphgeom.Angle((10 * lsst.geom.arcseconds).asRadians())
        self.stencil = stencils.SkyCircle(point, radius)

        self.projectionFinders = (
            projection_finders.ReadComponents(),
            projection_finders.ReadComponentsAstropyFits(),
        )

        self.dataId = {"patch": 24, "tract": 3828, "band": "r", "skymap": "DC2"}
        self.ref = self.butler.find_dataset("deepCoadd_calexp", data_id=self.dataId)

    def test_extract_ref(self):
        """Test that extract_ref produces a reasonable cutout."""
        with tempfile.TemporaryDirectory() as tempdir:
            for cutout_mode in CutoutMode:
                # Try each available projection finder at least once.
                proj_finder = self.projectionFinders[cutout_mode.value % len(self.projectionFinders)]
                cutoutBackend = ImageCutoutFactory(self.butler, proj_finder, tempdir)
                result = cutoutBackend.extract_ref(self.stencil, self.ref, cutout_mode=cutout_mode)
                # The galaxy should be near the center of the image.
                match result.cutout:
                    case lsst.afw.image.Exposure() | lsst.afw.image.MaskedImage():
                        box = result.cutout.getBBox()
                        array = result.cutout.image.array
                    case lsst.afw.image.Image():
                        array = result.cutout.array
                        box = result.cutout.getBBox()
                    case lsst.images.GeneralizedImage():
                        box = result.cutout.bbox.to_legacy()
                    case _:
                        raise RuntimeError(f"Unexpected cutout type: {type(result.cutout)}")
                self.assertEqual(box.width, 101)
                self.assertEqual(box.height, 101)
                self.assertFloatsAlmostEqual(array[50, 49], 2.083247661590576)
                self.assertFloatsAlmostEqual(array[1, 1], -0.14575983583927155)
                self.assertFloatsAlmostEqual(array[100, 100], 0.08674515783786774)

                output = cutoutBackend.process_ref(self.stencil, self.ref, cutout_mode=cutout_mode)
                self.assertTrue(output.exists())

                output = cutoutBackend.process_uuid(self.stencil, self.ref.id, cutout_mode=cutout_mode)
                self.assertTrue(output.exists())

    def test_provenance_in_primary_header(self):
        """Cutout provenance must land in the primary FITS header for every
        cutout mode, including the native ``lsst.images`` container modes.
        """
        provenance_keys = ("BTLRUUID", "BTLRNAME", "DATE-CUT", "CUTVERS")

        with tempfile.TemporaryDirectory() as tempdir:
            cutoutBackend = ImageCutoutFactory(self.butler, self.projectionFinders[0], tempdir)
            for cutout_mode in CutoutMode:
                with self.subTest(cutout_mode=cutout_mode):
                    output = cutoutBackend.process_ref(self.stencil, self.ref, cutout_mode=cutout_mode)
                    with output.open("rb") as fh, astropy.io.fits.open(fh) as hdul:
                        header = hdul[0].header
                    for key in provenance_keys:
                        self.assertIn(
                            key,
                            header,
                            f"{key} missing from primary header for {cutout_mode}",
                        )
                    self.assertEqual(header["BTLRUUID"], self.ref.id.hex)
                    self.assertEqual(header["BTLRNAME"], self.ref.datasetType.name)

                    # This test file has a BGMEAN header in the original
                    # primary header and we need to ensure that it still
                    # exists in the cutout to indicate that the primary
                    # header was propagated. IMAGE does not have it since
                    # IMAGE only reads the second header without merging
                    if cutout_mode != CutoutMode.IMAGE_ONLY:
                        self.assertIn("BGMEAN", header)


if __name__ == "__main__":
    unittest.main()
