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
import lsst.images.serialization
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

        # Projection finders are irrelevant in V2 but the constructor
        # still requires at least one.
        self.projectionFinders = (
            projection_finders.ReadComponents(),
            projection_finders.ReadComponentsAstropyFits(),
        )

        self.dataId = {"patch": 76, "tract": 5428, "band": "g", "skymap": "lsst_cells_v2"}
        ref = self.butler.find_dataset("deep_coadd", data_id=self.dataId)
        assert ref is not None
        self.ref = ref

    def test_extract_ref(self):
        """Test that extract_ref produces a reasonable cutout for all modes."""
        for cutout_mode in CutoutMode:
            # Projection finder has no effect for V2 and tempdir is
            # not relevant for this test.
            proj_finder = self.projectionFinders[0]
            cutoutBackend = ImageCutoutFactory(self.butler, proj_finder, ".")
            result = cutoutBackend.extract_ref(self.stencil, self.ref, cutout_mode=cutout_mode)
            match result.cutout:
                case lsst.images.MaskedImage():
                    box = result.cutout.bbox
                    array = result.cutout.image.array
                case lsst.images.Image():
                    array = result.cutout.array
                    box = result.cutout.bbox
                case _:
                    raise RuntimeError(f"Unexpected cutout type: {type(result.cutout)}")

            with self.subTest(cutout_mode=str(cutout_mode)):
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

    def test_outside_stencil_mask(self) -> None:
        """The ``OUTSIDE_STENCIL`` plane masks exactly the pixels outside the
        circular stencil for every mask-bearing cutout mode.

        The image-only modes carry no mask and are exempt.  The expected count
        is fixed by the stencil geometry and the test data WCS: it is the
        circle's complement within the ``29 x 28`` bounding box, so it is a
        non-empty, non-full subset (which also guards against an inverted
        mask).  It will change if the stencil or the fuzzed WCS changes.
        """
        image_only_modes = {CutoutMode.IMAGE_ONLY, CutoutMode.ASTROPY_IMAGE}
        expected_outside = 195
        for cutout_mode in CutoutMode:
            with self.subTest(cutout_mode=str(cutout_mode)):
                cutoutBackend = ImageCutoutFactory(self.butler, self.projectionFinders[0], ".")
                result = cutoutBackend.extract_ref(self.stencil, self.ref, cutout_mode=cutout_mode)
                result.mask()
                if cutout_mode in image_only_modes:
                    self.assertIsInstance(result.cutout, lsst.images.Image)
                    continue
                mask = result.cutout.mask
                self.assertIn("OUTSIDE_STENCIL", mask.schema.names)
                outside = mask.get("OUTSIDE_STENCIL")
                total = result.cutout.bbox.x.size * result.cutout.bbox.y.size
                n_outside = int(outside.sum())
                # A real circle complement: neither empty nor the full box.
                self.assertGreater(n_outside, 0)
                self.assertLess(n_outside, total)
                self.assertEqual(n_outside, expected_outside)

    def test_off_edge_cutout(self) -> None:
        """Test that we get a truncated cutout at the edge of the image."""
        # Shift the default position slightly so we fall partly off the edge.
        # Shift Y such that the bounding box is [-17:12] vs [0:64]
        point = lsst.sphgeom.LonLat.fromDegrees(0.25708, -3.03894)
        radius = lsst.sphgeom.Angle((3 * u.arcsec).to_value(u.rad))
        self.stencil = stencils.SkyCircle(point, radius, clip=True)

        proj_finder = self.projectionFinders[0]
        # Should not write any file out so tempdir is irrelevant.
        cutoutBackend = ImageCutoutFactory(self.butler, proj_finder, ".")
        result = cutoutBackend.extract_ref(self.stencil, self.ref, cutout_mode=CutoutMode.MASKED_IMAGE)
        box = result.cutout.bbox
        self.assertEqual(box.x.size, 29)
        self.assertEqual(box.y.size, 12)

    def test_process_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            for cutout_mode in CutoutMode:
                proj_finder = self.projectionFinders[0]
                cutoutBackend = ImageCutoutFactory(self.butler, proj_finder, tempdir)

                output = cutoutBackend.process_ref(self.stencil, self.ref, cutout_mode=cutout_mode)
                self.assertTrue(output.exists())

                # We should be able to read this back in using generic reader.
                result = lsst.images.serialization.read(output)
                self.assertIsInstance(result, lsst.images.GeneralizedImage)

    def test_process_uuid(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            for cutout_mode in CutoutMode:
                proj_finder = self.projectionFinders[0]
                cutoutBackend = ImageCutoutFactory(self.butler, proj_finder, tempdir)

                output = cutoutBackend.process_uuid(self.stencil, self.ref.id, cutout_mode=cutout_mode)
                self.assertTrue(output.exists())

                # We should be able to read this back in using generic reader.
                result = lsst.images.serialization.read(output)
                self.assertIsInstance(result, lsst.images.GeneralizedImage)

    def test_provenance_in_primary_header(self):
        """Cutout provenance must land in the primary FITS header for every
        cutout mode, including the native ``lsst.images`` container modes.
        """
        provenance_keys = ("CUTVERS",)
        # Card comments that fit within the 80-column FITS card limit and so
        # survive the round trip, covering both the provenance keys (DATE-CUT)
        # and the stencil keys (ST_TYPE).  Some cards can potentially have
        # truncated comments.
        provenance_comments = {
            "BTLRUUID": "Butler UUID of full image",
            "BTLRNAME": "Butler dataset type",
            "ST_TYPE": "Type of stencil used to create this cutout",
            "ST_RA": "[deg] Circle center Right Ascension",
            "DATE-CUT": "Time of cutout extraction",
        }

        with tempfile.TemporaryDirectory() as tempdir:
            cutoutBackend = ImageCutoutFactory(self.butler, self.projectionFinders[0], tempdir)
            for cutout_mode in CutoutMode:
                with self.subTest(cutout_mode=str(cutout_mode)):
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
                    for key, comment in provenance_comments.items():
                        self.assertEqual(
                            header.comments[key],
                            comment,
                            f"{key} comment wrong in primary header for {cutout_mode}",
                        )


if __name__ == "__main__":
    unittest.main()
