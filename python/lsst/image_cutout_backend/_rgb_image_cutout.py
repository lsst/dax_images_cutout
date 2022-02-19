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

__all__ = ["RgbImageCutout"]

import dataclasses
from uuid import uuid4

from astropy.visualization import make_lupton_rgb
import PIL

from lsst.daf.butler import DatasetRef
from lsst.resources import ResourcePath

from . import Extraction, ImageCutoutBackend
from .stencils import SkyStencil


@dataclasses.dataclass
class RgbExtraction:
    """The three separate cutouts as color channels to be combined to make an
    RGB image.
    """
    r: Extraction
    g: Extraction
    b: Extraction
    """The three color channels as image cutouts.
    """

    def create_rgb_image(self, Q=10, stretch=0.5):
        """Combine the RGB channels into a color image.

        Parameters
        ----------
        Q : `int`, optional
            The asinh softening parameter to apply to the channels.
        stretch : `float`, optional
            The linear stretch to apply to the channels.

        Returns
        -------
        image : `numpy.ndarray`, (3, N)
            The color image, at 8-bits per pixel.
        """
        return make_lupton_rgb(self.r.cutout.image.array,
                               self.g.cutout.image.array,
                               self.b.cutout.image.array,
                               Q=Q, stretch=stretch)

    def write_png(self, path):
        """Write the cutouts to a PNG file.

        Each channel's metata is written to EXIF fields in the output.

        Parameters
        ----------
        path : `str`
            Local path to the file to write.
        """
        data = self.create_rgb_image()
        image = PIL.Image.fromarray(data)
        image.info = self.r.metadata.toString()
        image.save(path, format="png", pnginfo=image.info)


class RgbImageCutout:
    """Get cutouts from three images and create an RGB color image.

    Parameters
    ----------
    backend : `ImageCutoutBackend`
        Backend to handle extracting the cutouts.
    """
    def __init__(self, backend: ImageCutoutBackend):
        self.backend = backend

    def extract_ref(self, stencil: SkyStencil,
                    ref_r: DatasetRef,
                    ref_g: DatasetRef,
                    ref_b: DatasetRef) -> RgbExtraction:
        """Extract a subimages from three fully-resolved `DatasetRef`s.
        """
        r = self.backend.extract_ref(stencil, ref_r)
        g = self.backend.extract_ref(stencil, ref_g)
        b = self.backend.extract_ref(stencil, ref_b)
        return RgbExtraction(r, g, b)

    def process_ref_png(self, stencil: SkyStencil,
                        ref_r: DatasetRef,
                        ref_g: DatasetRef,
                        ref_b: DatasetRef):
        """Extract a subimages from three fully-resolved `DatasetRef`s and
        write a PNG image to the backend output location.
        """
        result = self.extract_ref(stencil, ref_r, ref_g, ref_b)
        return self.write_png(result)

    def write_png(self, extract_result: RgbExtraction) -> ResourcePath:
        """Write an `RgbExtraction` to a remote PNG file .

        Parameters
        ----------
        extract_result : `RgbExtraction`
            Result of a call to an ``extract_*`` method.

        Returns
        -------
        uri : `ResourcePath`
            Full path to the saved cutout.
        """
        output_uuid = uuid4()
        remote_uri = self.backend.output_root.join(output_uuid.hex + ".png")
        with ResourcePath.temporary_uri(prefix=self.backend.temporary_root, suffix=".png") as tmp_uri:
            extract_result.write_png(tmp_uri.ospath)
            remote_uri.transfer_from(tmp_uri, transfer="copy")
        return remote_uri
