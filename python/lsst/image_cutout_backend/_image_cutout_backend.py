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

__all__ = ("ImageCutoutBackend", "Extraction")

import dataclasses
from typing import Iterable, Optional, Union
from uuid import UUID, uuid4

from lsst.afw.image import Exposure, Image, Mask, MaskedImage
from lsst.daf.base import PropertyList
from lsst.daf.butler import Butler, DataId, DatasetRef
from lsst.resources import ResourcePath, ResourcePathExpression

from .projection_finders import ProjectionFinder
from .stencils import PixelStencil, SkyStencil


@dataclasses.dataclass
class Extraction:
    """A struct that aggregates the results of extracting subimagines in the
    image cutout backend.
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

    def write_fits(self, path: str) -> None:
        """Write the cutout to a FITS file.

        Parameters
        ----------
        path : `str`
            Local path to the file.

        Notes
        -----
        If ``cutout`` is an `Exposure`, this will merge `metadata` into the
        cutout's attached metadata.  In other cases, `metadata` is written
        to the primary header without modifying `cutout`.
        """
        if isinstance(self.cutout, Exposure):
            self.cutout.getMetadata().update(self.metadata)
            self.cutout.writeFits(path)
        else:
            self.cutout.writeFits(path, metadata=self.metadata)


class ImageCutoutBackend:
    """High-level interface to the image cutout backend.

    Parameters
    ----------
    butler : `Butler`
        Butler that subimages are extracted from.
    projection_finder : `ProjectionObject`
        Object that obtains the WCS and bounding box for butler datasets of
        different types.  May include caches.
    output_root : convertible to `ResourcePath`
        Root of output file URIs.  This will be combined with the originating
        dataset's UUID and an encoding of the stencil to form the complete URI.
    temporary_root : convertible to `ResourcePath`, optional
        Local filesystem root to write files to before they are transferred to
        ``output_root`` (passed as the prefix argument to
        `ResourcePath.temporary_uri`).
    """

    def __init__(
        self,
        butler: Butler,
        projection_finder: ProjectionFinder,
        output_root: ResourcePathExpression,
        temporary_root: Optional[ResourcePathExpression] = None,
    ):
        self.butler = butler
        self.projection_finder = projection_finder
        self.output_root = ResourcePath(output_root, forceAbsolute=True, forceDirectory=True)
        self.temporary_root = (
            ResourcePath(temporary_root, forceDirectory=False) if temporary_root is not None else None
        )

    butler: Butler
    """Butler that subimage are extracted from (`Butler`).
    """

    projection_finder: ProjectionFinder
    """Object that obtains the WCS and bounding box for butler datasets of
    different types (`ProjectionFinder`).
    """

    output_root: ResourcePath
    """Root path that extracted cutouts are written to (`ResourcePath`).
    """

    temporary_root: Optional[ResourcePath]
    """Local filesystem root to write files to before they are transferred to
    ``output_root``
    """

    def process_ref(
        self, stencil: SkyStencil, ref: DatasetRef, *, mask_plane: Optional[str] = "STENCIL"
    ) -> ResourcePath:
        """Extract and write a cutout from a fully-resolved `DatasetRef`.

        Parameters
        ----------
        stencil : `SkyStencil`
            Definition of the cutout region, in sky coordinates.
        ref : `DatasetRef`
            Fully-resolved reference to the dataset to obtain the cutout from.
            Must have ``DatasetRef.id`` not `None` (use `extract_search`
            instead when this is not the case).  Need not have an expanded data
            ID.  May represent an image-like dataset component.
        mask : `str`, optional
            If not `None`, set this mask plane in the extracted cutout showing
            the approximate stencil region.  Does nothing if the image type
            does not have a mask plane.  Defaults to ``STENCIL``.

        Returns
        -------
        uri : `ResourcePath`
            Full path to the extracted cutout.
        """
        extract_result = self.extract_ref(stencil, ref)
        if mask_plane is not None:
            extract_result.mask(mask_plane)
        return self.write_fits(extract_result)

    def process_uuid(
        self,
        stencil: SkyStencil,
        uuid: UUID,
        *,
        component: Optional[str] = None,
        mask_plane: Optional[str] = "STENCIL",
    ) -> ResourcePath:
        """Extract and write a cutout from a dataset identified by its UUID.

        Parameters
        ----------
        stencil : `SkyStencil`
            Definition of the cutout region, in sky coordinates.
        uuid : `UUID`
            Unique ID of the dataset to extract the subimage from.
        component : `str`, optional
            If not `None` (default), read this component instead of the
            composite dataset.
        mask : `str`, optional
            If not `None`, set this mask plane in the extracted cutout showing
            the approximate stencil region.  Does nothing if the image type
            does not have a mask plane.  Defaults to ``STENCIL``.

        Returns
        -------
        uri : `ResourcePath`
            Full path to the extracted cutout.
        """
        extract_result = self.extract_uuid(stencil, uuid, component=component)
        if mask_plane is not None:
            extract_result.mask(mask_plane)
        return self.write_fits(extract_result)

    def process_search(
        self,
        stencil: SkyStencil,
        dataset_type_name: str,
        data_id: DataId,
        collections: Iterable[str],
        *,
        mask_plane: Optional[str] = "STENCIL",
    ) -> ResourcePath:
        """Extract and write a cutout from a dataset identified by its UUID.

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
        mask : `str`, optional
            If not `None`, set this mask plane in the extracted cutout showing
            the approximate stencil region.  Does nothing if the image type
            does not have a mask plane.  Defaults to ``STENCIL``.

        Returns
        -------
        uri : `ResourcePath`
            Full path to the extracted cutout.
        """
        extract_result = self.extract_search(stencil, dataset_type_name, data_id, collections)
        if mask_plane is not None:
            extract_result.mask(mask_plane)
        return self.write_fits(extract_result)

    def extract_ref(self, stencil: SkyStencil, ref: DatasetRef) -> Extraction:
        """Extract a subimage from a fully-resolved `DatasetRef`.

        Parameters
        ----------
        stencil : `SkyStencil`
            Definition of the cutout region, in sky coordinates.
        ref : `DatasetRef`
            Fully-resolved reference to the dataset to obtain the cutout from.
            Must have ``DatasetRef.id`` not `None` (use `extract_search`
            instead when this is not the case).  Need not have an expanded data
            ID.  May represent an image-like dataset component.

        Returns
        -------
        result : `Extraction`
            Struct that combines the cutout itself with additional metadata
            and the pixel-coordinate stencil.  The cutout is not masked;
            `Extraction.mask` must be called explicitly if desired.
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
        metadata.set("BTLRUUID", ref.id.hex, "Butler dataset UUID this cutout was extracted from.")
        metadata.set(
            "BTLRNAME", ref.datasetType.name, "Butler dataset type name this cutout was extracted from."
        )
        # TODO: write cutout timestamp to metadata.  Need to read up on how to
        # represent times in FITS headers.
        for n, (k, v) in enumerate(ref.dataId.items()):
            # Write data ID dictionary sort of like a list of 2-tuples, to make
            # it easier to stay within the FITS 8-char key limit.
            metadata.set(f"BTLRK{n:03}", k.name, f"Name of dimension {n} in the data ID.")
            metadata.set(f"BTLRV{n:03}", v, f"Value of dimension {n} in the data ID.")
        stencil.to_fits_metadata(metadata)
        return Extraction(
            cutout=cutout,
            sky_stencil=stencil,
            pixel_stencil=pixel_stencil,
            metadata=metadata,
            origin_ref=ref,
        )

    def extract_uuid(self, stencil: SkyStencil, uuid: UUID, *, component: Optional[str] = None) -> Extraction:
        """Extract a subimage from a dataset identified by its UUID.

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
        result : `Extraction`
            Struct that combines the cutout itself with additional metadata
            and the pixel-coordinate stencil.  The cutout is not masked;
            `Extraction.mask` must be called explicitly if desired.
        """
        ref = self.butler.registry.getDataset(uuid)
        if component is not None:
            ref = ref.makeComponentRef(component)
        return self.extract_ref(stencil, ref)

    def extract_search(
        self, stencil: SkyStencil, dataset_type_name: str, data_id: DataId, collections: Iterable[str]
    ) -> Extraction:
        """Extract a subimage from a dataset identified by a (dataset type,
        data ID, collection path) tuple.

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
        result : `Extraction`
            Struct that combines the cutout itself with additional metadata
            and the pixel-coordinate stencil.  The cutout is not masked;
            `Extraction.mask` must be called explicitly if desired.
        """
        ref = self.butler.registry.findDataset(dataset_type_name, data_id, collections)
        return self.extract_ref(stencil, ref)

    def write_fits(self, extract_result: Extraction) -> ResourcePath:
        """Write a `Extraction` to a remote FITS file in `output_root`.

        Parameters
        ----------
        extract_result : `Extraction`
            Result of a call to a ``extract_*`` method.

        Returns
        -------
        uri : `ResourcePath`
            Full path to the extracted cutout.
        """
        output_uuid = uuid4()
        remote_uri = self.output_root.join(output_uuid.hex + ".fits")
        with ResourcePath.temporary_uri(prefix=self.temporary_root, suffix=".fits") as tmp_uri:
            extract_result.write_fits(tmp_uri.ospath)
            remote_uri.transfer_from(tmp_uri, transfer="copy")
        return remote_uri
