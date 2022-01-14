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

from __future__ import annotations

__all__ = ("ImageCutoutBackend", "ReadResult")

import dataclasses
from typing import Iterable, Optional, Union
from uuid import UUID

from lsst.afw.image import Exposure, Image, Mask, MaskedImage
from lsst.daf.base import PropertyList
from lsst.daf.butler import Butler, DataId, DatasetRef

from .projection_finders import ProjectionFinder
from .stencils import PixelStencil, SkyStencil


@dataclasses.dataclass
class ReadResult:
    """A struct that aggregates the results of reading a cutout in the image
    cutout backend.
    """

    cutout: Union[Image, Mask, MaskedImage, Exposure]
    """The image cutout itself.
    """

    sky_stencil: SkyStencil
    """The original sky-coordinate stencil.
    """

    pixel_stencil: PixelStencil
    """A pixel-coordinate representation of the stencil.
    """

    metadata: PropertyList
    """Additional FITS metadata about the cutout process.

    This should be merged with ``cutout.getMetadata()`` on write, for types
    that carry their own metadata.
    """

    origin_ref: DatasetRef
    """Fully-resolved reference to the dataset the cutout is from.
    """

    def mask(self, name: str = "STENCIL") -> None:
        """Set the bitmask to show the approximate coverage of nonrectangular
        stencils.

        Parameters
        ----------
        name : `str`, optional
            Name of the mask plane to add and set.

        Notes
        -----
        This method modifies `cutout` in place if it is a `Mask`,
        `MaskedImage`, or `Exposure`.  It does nothing if `cutout` is an
        `Image`.
        """
        if isinstance(self.cutout, Exposure):
            mask = self.cutout.mask
        elif isinstance(self.cutout, MaskedImage):
            mask = self.cutout.mask
        elif isinstance(self.cutout, Mask):
            mask = self.cutout
        else:
            return
        mask.addMaskPlane(name)
        bits = mask.getPlaneBitMask(name)
        self.pixel_stencil.set_mask(mask, bits)


class ImageCutoutBackend:
    """High-level interface to the image cutout backend.

    Parameters
    ----------
    butler : `Butler`
        Butler that image cutouts are read from.
    projection_finder : `ProjectionObject`
        Object that obtains the WCS and bounding box for butler datasets of
        different types.  May include caches.
    """

    def __init__(self, butler: Butler, projection_finder: ProjectionFinder):
        self.butler = butler
        self.projection_finder = projection_finder

    butler: Butler
    """Butler client used to read cutouts (`Butler`).
    """

    projection_finder: ProjectionFinder
    """Object that obtains the WCS and bounding box for butler datasets of
    different types (`ProjectionFinder`).
    """

    def read_ref(self, stencil: SkyStencil, ref: DatasetRef) -> ReadResult:
        """Read a cutout from a fully-resolved `DatasetRef`.

        Parameters
        ----------
        stencil : `SkyStencil`
            Definition of the cutout region, in sky coordinates.
        ref : `DatasetRef`
            Fully-resolved reference to the dataset to obtain the cutout from.
            Must have ``DatasetRef.id`` not `None` (use `read_search` instead
            when this is not the case).  Need not have an expanded data ID.
            May represent an image-like dataset component.

        Returns
        -------
        result : `ReadResult`
            Struct that combines the cutout itself with additional metadata
            and the pixel-coordinate stencil.  The cutout is not masked;
            `ReadResult.mask` must be called explicitly if desired.
        """
        if ref.id is None:
            raise ValueError(f"A resolved DatasetRef is required; got {ref}.")
        # Get the WCS and bbox of this dataset.
        wcs, bbox = self.projection_finder(ref, self.butler)
        # Transform the stencil to pixel coordinates.
        pixel_stencil = stencil.to_pixels(wcs, bbox)
        # Actually read the cutout.  Leave it to the butler to cache remote
        # files locally or do partial remote reads.
        cutout = self.butler.getDirect(ref, parameters={"bbox": pixel_stencil.bbox})
        # Create some FITS metadata with the cutout parameters.
        metadata = PropertyList()
        metadata.set("BTLRUUID", ref.id, "Butler dataset UUID this cutout was extracted from.")
        metadata.set(
            "BTLRNAME", ref.datasetType.name, "Butler dataset type name this cutout was extracted from."
        )
        # TODO: write cutout timestamp to metadata.  Need to read up on how to
        # represent times in FITS headers.
        for n, (k, v) in enumerate(ref.dataId.items()):
            # Write data ID dictionary sort of like a list of 2-tuples, to make
            # it easier to stay within the FITS 8-char key limit.
            metadata.set(f"BTLRK{n:03}", k, f"Name of dimension {n} in the data ID.")
            metadata.set(f"BTLRV{n:03}", v, f"Value of dimension {n} in the data ID.")
        stencil.to_fits_metadata(metadata)
        return ReadResult(
            cutout=cutout,
            sky_stencil=stencil,
            pixel_stencil=pixel_stencil,
            metadata=metadata,
            origin_ref=ref,
        )

    def read_uuid(self, stencil: SkyStencil, uuid: UUID, *, component: Optional[str] = None) -> ReadResult:
        """Read a cutout from a dataset identified by its UUID.

        Parameters
        ----------
        stencil : `SkyStencil`
            Definition of the cutout region, in sky coordinates.
        uuid : `UUID`
            Unique ID of the dataset to read from.
        component : `str`, optional
            If not `None` (default), read this component instead of the
            composite dataset.

        Returns
        -------
        result : `ReadResult`
            Struct that combines the cutout itself with additional metadata
            and the pixel-coordinate stencil.  The cutout is not masked;
            `ReadResult.mask` must be called explicitly if desired.
        """
        ref = self.butler.registry.getDataset(uuid)
        if component is not None:
            ref = ref.makeComponentRef(component)
        return self.read_ref(stencil, ref)

    def read_search(
        self, stencil: SkyStencil, dataset_type_name: str, data_id: DataId, collections: Iterable[str]
    ) -> ReadResult:
        """Read a cutout from a dataset identified by a (dataset type, data ID,
        collection path) tuple.

        Parameters
        ----------
        stencil : `SkyStencil`
            Definition of the cutout region, in sky coordinates.
        dataset_type_name : `str`
            Name of the butler dataset.  Use ``.``-separate terms to read an
            image-like component.
        data_id : `dict` or `DataCoordinate`
            Mapping-of-dimensions identifier for the dataset within its
            collection.
        collections : `Iterable` [ `str` ]
            Collections to search for the dataset, in the order they should be
            searched.

        Returns
        -------
        result : `ReadResult`
            Struct that combines the cutout itself with additional metadata
            and the pixel-coordinate stencil.  The cutout is not masked;
            `ReadResult.mask` must be called explicitly if desired.
        """
        ref = self.butler.registry.findDataset(dataset_type_name, data_id, collections)
        return self.read_ref(stencil, ref)
