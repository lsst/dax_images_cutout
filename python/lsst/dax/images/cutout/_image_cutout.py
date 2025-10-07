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

__all__ = ("CutoutMode", "Extraction", "ImageCutoutFactory")

import dataclasses
import logging
from collections.abc import Sequence
from enum import Enum, auto
from uuid import UUID, uuid4

import astropy.io.fits
import astropy.time

import lsst.geom
from lsst.afw.geom.wcsUtils import getImageXY0FromMetadata
from lsst.afw.image import Exposure, Image, Mask, MaskedImage, makeExposure, makeMaskedImage
from lsst.daf.base import PropertyList
from lsst.daf.butler import Butler, DataId, DatasetRef
from lsst.resources import ResourcePath, ResourcePathExpression
from lsst.utils.timer import time_this

from .projection_finders import ProjectionFinder
from .stencils import PixelStencil, SkyStencil
from .version import __version__

# Default logger.
_LOG = logging.getLogger(__name__)
_TIMER_LOG_LEVEL = logging.INFO


class CutoutMode(Enum):
    # The entire Exposure plus all associated metadata.
    FULL_EXPOSURE = auto()
    # Retrieve the full Exposure but only return MaskedImage/SkyWcs/metadata
    STRIPPED_EXPOSURE = auto()
    # Retrieve and return Image with no WCS or extra metadata.
    IMAGE_ONLY = auto()
    # MaskedImage using butler+afw
    MASKED_IMAGE = auto()
    # Astropy Image + primary HDU + WCS
    ASTROPY_IMAGE = auto()
    # Astropy with masked image.
    ASTROPY_MASKED_IMAGE = auto()


@dataclasses.dataclass
class Extraction:
    """A struct that aggregates the results of extracting subimages in the
    image cutout backend.
    """

    cutout: Image | Mask | MaskedImage | Exposure | astropy.io.fits.HDUList
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

    def write_fits(self, path: str, logger: logging.Logger | None = None) -> None:
        """Write the cutout to a FITS file.

        Parameters
        ----------
        path : `str`
            Local path to the file.
        logger : `logging.Logger`, optional
            Logger to use for timing messages.  If `None`, a default logger
            will be used.

        Notes
        -----
        If ``cutout`` is an `Exposure`, this will merge `metadata` into the
        cutout's attached metadata.  In other cases, `metadata` is written
        to the primary header without modifying `cutout`.
        """
        logger = logger if logger is not None else _LOG
        with time_this(logger, msg="Writing FITS file to %s", args=(path,), level=_TIMER_LOG_LEVEL):
            if isinstance(self.cutout, Exposure):
                self.cutout.getMetadata().update(self.metadata)
                self.cutout.writeFits(path)
            elif isinstance(self.cutout, astropy.io.fits.HDUList):
                self.cutout[0].header.update(self.metadata)
                with open(path, "wb") as fh:
                    self.cutout.writeto(fh)
            else:
                self.cutout.writeFits(path, metadata=self.metadata)


class ImageCutoutFactory:
    """High-level interface to the image cutout functionality.

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
    logger : `logging.Logger`, optional
        Logger to use for timing messages.  If `None`, a default logger
        will be used.
    """

    def __init__(
        self,
        butler: Butler,
        projection_finder: ProjectionFinder,
        output_root: ResourcePathExpression,
        temporary_root: ResourcePathExpression | None = None,
        logger: logging.Logger | None = None,
    ):
        self.butler = butler
        self.projection_finder = projection_finder
        self.output_root = ResourcePath(output_root, forceAbsolute=True, forceDirectory=True)
        self.temporary_root = (
            ResourcePath(temporary_root, forceDirectory=True) if temporary_root is not None else None
        )
        self.logger = logger if logger is not None else _LOG

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

    temporary_root: ResourcePath | None
    """Local filesystem root to write files to before they are transferred to
    ``output_root``
    """

    def process_ref(
        self,
        stencil: SkyStencil,
        ref: DatasetRef,
        *,
        mask_plane: str | None = "STENCIL",
        cutout_mode: CutoutMode = CutoutMode.FULL_EXPOSURE,
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
        extract_result = self.extract_ref(stencil, ref, cutout_mode=cutout_mode)
        if mask_plane is not None:
            extract_result.mask(mask_plane)
        return self.write_fits(extract_result)

    def process_uuid(
        self,
        stencil: SkyStencil,
        uuid: UUID,
        *,
        component: str | None = None,
        mask_plane: str | None = "STENCIL",
        cutout_mode: CutoutMode = CutoutMode.FULL_EXPOSURE,
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
        extract_result = self.extract_uuid(stencil, uuid, component=component, cutout_mode=cutout_mode)
        if mask_plane is not None:
            extract_result.mask(mask_plane)
        return self.write_fits(extract_result)

    def process_search(
        self,
        stencil: SkyStencil,
        dataset_type_name: str,
        data_id: DataId,
        collections: Sequence[str],
        *,
        mask_plane: str | None = "STENCIL",
    ) -> ResourcePath:
        """Extract and write a cutout from a dataset identified by a
        (dataset type, data ID, collection path) tuple.

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

    def extract_ref(
        self, stencil: SkyStencil, ref: DatasetRef, cutout_mode: CutoutMode = CutoutMode.FULL_EXPOSURE
    ) -> Extraction:
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
        # Timestamp of the cutout extraction.
        now = astropy.time.Time.now()
        timesys = "UTC"
        # Get the WCS and bbox of this dataset.
        wcs, bbox = self.projection_finder(ref, self.butler, logger=self.logger)
        # Transform the stencil to pixel coordinates.
        pixel_stencil = stencil.to_pixels(wcs, bbox)
        # Somewhere to store metadata.
        metadata = PropertyList()
        # Actually read the cutout.  Leave it to the butler to cache remote
        # files locally or do partial remote reads.
        with time_this(
            self.logger,
            msg="Extract cutout",
            kwargs={"id": str(ref.id), "cutout_mode": str(cutout_mode), "stencil": str(stencil)},
            level=_TIMER_LOG_LEVEL,
        ):
            match cutout_mode:
                case CutoutMode.FULL_EXPOSURE:
                    cutout = self.butler.get(ref, parameters={"bbox": pixel_stencil.bbox})
                    timesys = cutout.metadata.get("TIMESYS", timesys)
                case CutoutMode.STRIPPED_EXPOSURE:
                    cutout = self.butler.get(ref, parameters={"bbox": pixel_stencil.bbox})
                    metadata = cutout.metadata  # Track metadata externally.
                    timesys = metadata.get("TIMESYS", timesys)
                    cutout = makeExposure(cutout.maskedImage, wcs=cutout.wcs)
                case CutoutMode.IMAGE_ONLY:
                    cutout = self.butler.get(
                        ref.makeComponentRef("image"), parameters={"bbox": pixel_stencil.bbox}
                    )
                    # No metadata so UTC is default.
                    timesys = "UTC"
                case CutoutMode.MASKED_IMAGE:
                    # Rely on the file being cached on first read. Faster than
                    # reading entire exposure.
                    image = self.butler.get(
                        ref.makeComponentRef("image"), parameters={"bbox": pixel_stencil.bbox}
                    )
                    variance = self.butler.get(
                        ref.makeComponentRef("variance"), parameters={"bbox": pixel_stencil.bbox}
                    )
                    mask = self.butler.get(
                        ref.makeComponentRef("mask"), parameters={"bbox": pixel_stencil.bbox}
                    )
                    wcs = self.butler.get(ref.makeComponentRef("wcs"))
                    masked_image = makeMaskedImage(image, mask, variance)
                    cutout = makeExposure(masked_image, wcs=wcs)

                    metadata = self.butler.get(ref.makeComponentRef("metadata"))
                    timesys = metadata.get("TIMESYS", timesys)
                case CutoutMode.ASTROPY_IMAGE | CutoutMode.ASTROPY_MASKED_IMAGE:
                    # Bypass butler and try to find the pixel HDU directly.
                    # Approximate WCS is attached to DP1 image data but needs
                    # to be shifted.
                    pixel_components = {"image", "mask", "variance"}
                    if cutout_mode == CutoutMode.ASTROPY_IMAGE:
                        pixel_components = {"image"}
                    bbox = pixel_stencil.bbox
                    uri = self.butler.getURI(ref)
                    fs, fspath = uri.to_fsspec()
                    hdul = []
                    # Want the primary header and the requested pixel HDU.
                    # Stop scanning once IMAGE has been located.
                    found_primary = False
                    with fs.open(fspath) as f, astropy.io.fits.open(f) as fits_obj:
                        for hdu in fits_obj:
                            if not found_primary:
                                hdul.append(hdu.copy())
                                timesys = hdul[0].header.get("TIMESYS", timesys)
                                found_primary = True
                                continue

                            hdr = hdu.header
                            extname = hdr.get("EXTNAME")
                            if extname and extname.lower() in pixel_components:
                                pixel_components.remove(extname.lower())
                                # Get BBOX for full HDU.
                                # Use shape to prevent reading the data array.
                                shape = hdu.shape
                                dimensions = lsst.geom.Extent2I(shape[1], shape[0])

                                # XY0 is defined in the A WCS.
                                pl = PropertyList()
                                pl.update(hdr)
                                xy0 = getImageXY0FromMetadata(pl, "A", strip=False)

                                # This is the PARENT bbox.
                                full_bbox = lsst.geom.Box2I(xy0, dimensions)

                                # Work out the required cutout of the HDU.
                                minX = bbox.getBeginX() - full_bbox.getBeginX()
                                maxX = bbox.getEndX() - full_bbox.getBeginX()
                                minY = bbox.getBeginY() - full_bbox.getBeginY()
                                maxY = bbox.getEndY() - full_bbox.getBeginY()
                                # Get the cutout and detach from remote.
                                data = hdu.section[minY:maxY, minX:maxX].copy()

                                # Must correct the header to take into account
                                # the offset.
                                for x, y in (("CRPIX1", "CRPIX2"), ("LTV1", "LTV2")):
                                    if x in hdr:
                                        hdr[x] -= bbox.getBeginX()
                                    if y in hdr:
                                        hdr[y] -= bbox.getBeginY()
                                if "CRVAL1A" in hdr:
                                    hdr["CRVAL1A"] += bbox.getBeginX()
                                if "CRVAL2A" in hdr:
                                    hdr["CRVAL2A"] += bbox.getBeginY()

                                # Create new HDU with the cutout.
                                cutout_hdu = astropy.io.fits.ImageHDU(data=data, header=hdr.copy())
                                hdul.append(cutout_hdu)

                            # Stop looking for HDUs.
                            if not pixel_components:
                                break
                    cutout = astropy.io.fits.HDUList(hdus=hdul)
                case _:
                    raise ValueError(f"Unsupported cutout mode: {cutout_mode}")

        # Create some FITS metadata with the cutout parameters.
        metadata.set("BTLRUUID", ref.id.hex, "Butler dataset UUID this cutout was extracted from.")
        metadata.set(
            "BTLRNAME", ref.datasetType.name, "Butler dataset type name this cutout was extracted from."
        )
        for n, (k, v) in enumerate(ref.dataId.required.items()):
            # Write data ID dictionary sort of like a list of 2-tuples, to make
            # it easier to stay within the FITS 8-char key limit.
            metadata.set(f"BTLRK{n:03}", k, f"Name of dimension {n} in the data ID.")
            metadata.set(f"BTLRV{n:03}", v, f"Value of dimension {n} in the data ID.")
        stencil.to_fits_metadata(metadata)

        # Record the time and software version.
        now.format = "fits"
        now = now.tai if timesys.lower() == "tai" else now.utc
        metadata.set("DATE-CUT", str(now), "Time of cutout extraction")
        metadata.set("CUTVERS", __version__, "dax_images_cutout software version")

        return Extraction(
            cutout=cutout,
            sky_stencil=stencil,
            pixel_stencil=pixel_stencil,
            metadata=metadata,
            origin_ref=ref,
        )

    def extract_uuid(
        self,
        stencil: SkyStencil,
        uuid: UUID,
        *,
        component: str | None = None,
        cutout_mode: CutoutMode = CutoutMode.FULL_EXPOSURE,
    ) -> Extraction:
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
        ref = self.butler.get_dataset(uuid)
        if ref is None:
            raise LookupError(f"No dataset found with UUID {uuid}.")
        if component is not None:
            ref = ref.makeComponentRef(component)
        return self.extract_ref(stencil, ref, cutout_mode=cutout_mode)

    def extract_search(
        self, stencil: SkyStencil, dataset_type_name: str, data_id: DataId, collections: Sequence[str]
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
        ref = self.butler.find_dataset(dataset_type_name, data_id, collections=collections)
        if ref is None:
            raise LookupError(
                f"No {dataset_type_name} dataset found with data ID {data_id} in {collections}."
            )
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
            tmp_uri.parent().mkdir()
            extract_result.write_fits(tmp_uri.ospath, logger=self.logger)
            remote_uri.transfer_from(tmp_uri, transfer="copy")
        return remote_uri
