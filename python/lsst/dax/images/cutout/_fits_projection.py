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

__all__ = ("projection_and_bbox_from_fits_header",)

from collections.abc import Sequence

import astropy.io.fits
import astropy.units as u
import numpy as np

from lsst.images import Box, GeneralFrame, SkyProjection
from lsst.images._transforms import _ast

# Identifier AST gives the alternate ("A") FITS WCS that encodes the parent
# pixel origin (XY0) for LSST image data.
_PARENT_WCS_IDENT = "A"

# Pixel coordinate frame for the projection.  Only the unit is significant.
_PIXEL_FRAME = GeneralFrame(unit=u.pix)


def projection_and_bbox_from_fits_header(
    header: astropy.io.fits.Header, shape: Sequence[int]
) -> tuple[SkyProjection, Box]:
    """Build a sky projection and parent bounding box from a FITS header.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        FITS header carrying a primary celestial WCS and an ``"A"`` alternate
        WCS that encodes the parent pixel origin.
    shape : `~collections.abc.Sequence` [ `int` ]
        Array shape ``(ny, nx)`` of the image the header describes, as returned
        by ``astropy.io.fits.ImageHDU.shape``.

    Returns
    -------
    projection : `lsst.images.SkyProjection`
        Sky projection for the primary WCS, in ICRS.
    bbox : `lsst.images.Box`
        Parent bounding box: the origin comes from the ``"A"`` WCS and the
        size from ``shape``.

    Notes
    -----
    All AST handling goes through the wrapper ``lsst.images`` exposes in
    ``lsst.images._transforms._ast``, which presents the astshim interface
    regardless of whether astshim or starlink-pyast backs it.  This function is
    deliberately confined to that wrapper plus the public ``lsst.images`` APIs
    so it can move into ``lsst.images`` later (e.g. as
    ``SkyProjection.from_fits_header``).
    """
    frame_set = _ast.FitsChan(_ast.StringStream(header.tostring())).read()
    projection = SkyProjection.from_ast_frame_set(frame_set, _PIXEL_FRAME)

    parent_index = _frame_index_by_ident(frame_set, _PARENT_WCS_IDENT)
    grid_to_parent = frame_set.getMapping(frame_set.base, parent_index)
    # AST GRID coordinates are 1-based, so grid (1, 1) is the first pixel; its
    # parent coordinate is the integer XY0 origin.
    origin = grid_to_parent.applyForward(np.array([[1.0], [1.0]])).ravel()
    x0 = int(round(float(origin[0])))
    y0 = int(round(float(origin[1])))
    ny, nx = int(shape[0]), int(shape[1])
    bbox = Box.factory[y0 : y0 + ny, x0 : x0 + nx]
    return projection, bbox


def _frame_index_by_ident(frame_set: _ast.FrameSet, ident: str) -> int:
    """Return the 1-based index of the frame whose identifier is ``ident``."""
    for index in range(1, frame_set.nFrame + 1):
        if frame_set.getFrame(index).ident.strip() == ident:
            return index
    raise LookupError(f"FITS header has no AST frame with identifier {ident!r}.")
