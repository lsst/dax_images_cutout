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
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import astropy.io.fits
import astropy.time

import lsst.images
import lsst.images.serialization
from lsst.daf.butler import Butler, DataId, DatasetRef
from lsst.images import Box, MaskPlane, MaskSchema
from lsst.images import Mask as ImagesMask
from lsst.images.fits import DEFAULT_PAGE_SIZE, READ_CACHE_MAX_BYTES, ExtensionKey
from lsst.images.fits._input_archive import _READ_CACHE_TYPE
from lsst.resources import ResourcePath, ResourcePathExpression
from lsst.utils.timer import time_this

from ._fits_projection import projection_and_bbox_from_fits_header
from .projection_finders import ProjectionFinder
from .stencils import PixelStencil, SkyStencil
from .version import __version__

if TYPE_CHECKING:
    from lsst.afw.image import Exposure, Image, Mask, MaskedImage
    from lsst.daf.base import PropertyList


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

    cutout: Image | Mask | MaskedImage | Exposure | lsst.images.GeneralizedImage
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

    def mask(self, name: str = "OUTSIDE_STENCIL") -> None:
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
        if isinstance(self.cutout, lsst.images.MaskedImage):
            mask = self.cutout.mask
            # Create the new plane if it's not there.
            if name not in mask.schema.names:
                mask.add_plane(name, "Pixel lies outside the stencil")
            self.pixel_stencil.set_mask(mask, name)
            return

        # Try afw variants. Protect the imports (if the imports fail it is
        # not possible for it to be an afe type).
        try:
            from lsst.afw.image import Exposure, Mask, MaskedImage

            match self.cutout:
                case Exposure() | MaskedImage():
                    mask = self.cutout.mask
                case Mask():
                    mask = self.cutout
                case _:
                    return
        except ImportError:
            # Can not determine the type so do not mask anything.
            return

        # Stage the coverage in an lsst.images.Mask, then OR it into the afw
        # mask plane so stencils never sees an afw object.
        images_mask = ImagesMask(
            schema=MaskSchema([MaskPlane(name, "stencil coverage")]),
            bbox=Box.from_legacy(mask.getBBox()),
        )
        self.pixel_stencil.set_mask(images_mask, name)
        covered = images_mask.get(name)
        mask.addMaskPlane(name)
        bits = mask.getPlaneBitMask(name)
        mask.array[:, :] |= (bits * covered).astype(mask.array.dtype)

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
            if isinstance(self.cutout, lsst.images.GeneralizedImage):
                # Write the cutout provenance into the primary FITS header via
                # the archive's update_header hook.  Storing it in the object's
                # flexible metadata would instead place it in the JSON tree,
                # where it would not appear in the primary header.
                self.cutout.write(path, update_header=lambda header: header.update(self.metadata))
            elif hasattr(self.cutout, "getMetadata") and hasattr(self.cutout, "writeFits"):
                self.cutout.metadata.update(self.metadata)
                self.cutout.writeFits(path)
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
        if issubclass(ref.datasetType.storageClass.pytype, lsst.images.GeneralizedImage):
            return self._extract_ref_v2(stencil, ref, cutout_mode=cutout_mode)
        return self._extract_ref_legacy(stencil, ref, cutout_mode=cutout_mode)

    def _read_astropy_hdulist(
        self,
        cutout_mode: CutoutMode,
        stencil: SkyStencil,
        ref: DatasetRef,
    ) -> tuple[lsst.images.GeneralizedImage, PixelStencil | None, str]:
        """Read the primary header and pixel HDUs, cut to the stencil.

        Parameters
        ----------
        cutout_mode
            The requested cutout mode. Must be either an Astropy image or
            masked image request.
        stencil
            Sky-coordinate stencil defining the cutout region.
        ref : `DatasetRef`
            Resolved reference to the dataset to read.

        Returns
        -------
        result : `lsst.images.GeneralizedImage`
            The resultant generalized image constructed from the individual
            HDUs.
        pixel_stencil : `PixelStencil` or `None`
            Pixel-coordinate stencil computed from the first pixel HDU, or
            `None` if no pixel HDU was found.
        timesys : `str`
            ``TIMESYS`` from the primary header, or ``"UTC"``.
        """
        if cutout_mode not in (CutoutMode.ASTROPY_IMAGE, CutoutMode.ASTROPY_MASKED_IMAGE):
            raise ValueError(f"Unexpected cutout mode {cutout_mode} encountered")

        pixel_components = (
            {"image"} if cutout_mode == CutoutMode.ASTROPY_IMAGE else {"image", "mask", "variance"}
        )
        # Tune the fsspec cache to match what we use in lsst.images.
        maxblocks = max(2, READ_CACHE_MAX_BYTES // DEFAULT_PAGE_SIZE)
        fsspec_kwargs = {
            "block_size": DEFAULT_PAGE_SIZE,
            "cache_type": _READ_CACHE_TYPE,
            "cache_options": {"maxblocks": maxblocks},
        }
        timesys = "UTC"
        pixel_stencil: PixelStencil | None = None
        bbox: Box | None = None
        uri = self.butler.getURI(ref)
        fs, fspath = uri.to_fsspec()
        hdul: list[astropy.io.fits.hdu.base.ExtensionHDU] = []
        found_primary = False
        with fs.open(fspath, **fsspec_kwargs) as f, astropy.io.fits.open(f) as fits_obj:
            for hdu in fits_obj:
                if not found_primary:
                    hdul.append(hdu.copy())
                    timesys_hdr = hdul[0].header.get("TIMESYS", timesys)
                    if timesys_hdr:
                        # For mypy since in theory a FITS header can exist
                        # but be undefined.
                        timesys = str(timesys_hdr)

                    found_primary = True
                    continue

                hdr = hdu.header
                extname = hdr.get("EXTNAME")
                if extname and extname.lower() in pixel_components:
                    pixel_components.remove(extname.lower())
                    projection, full_bbox = projection_and_bbox_from_fits_header(hdr, hdu.shape)
                    if pixel_stencil is None:
                        pixel_stencil = stencil.to_pixels(projection, full_bbox)
                        bbox = pixel_stencil.bbox

                    assert bbox is not None
                    # Offsets of the cutout within the full HDU (array order).
                    min_x = bbox.x.start - full_bbox.x.start
                    max_x = bbox.x.stop - full_bbox.x.start
                    min_y = bbox.y.start - full_bbox.y.start
                    max_y = bbox.y.stop - full_bbox.y.start
                    data = hdu.section[min_y:max_y, min_x:max_x].copy()

                    # Correct the header WCS for the cutout offset.
                    if (k := "CRPIX1") in hdr:
                        hdr[k] -= min_x
                    if (k := "CRPIX2") in hdr:
                        hdr[k] -= min_y
                    if (k := "LTV1") in hdr:
                        hdr[k] = -bbox.x.start
                    if (k := "LTV2") in hdr:
                        hdr[k] = -bbox.y.start
                    if (k := "CRVAL1A") in hdr:
                        hdr[k] = bbox.x.start
                    if (k := "CRVAL2A") in hdr:
                        hdr[k] = bbox.y.start

                    hdul.append(astropy.io.fits.ImageHDU(data=data, header=hdr.copy()))

                if not pixel_components:
                    break

        hdulist = astropy.io.fits.HDUList(hdus=hdul)
        result: lsst.images.GeneralizedImage
        if cutout_mode == CutoutMode.ASTROPY_IMAGE:
            result = lsst.images.Image.from_hdu_list(hdulist)
        else:
            result = lsst.images.MaskedImage.from_hdu_list(hdulist)

        return result, pixel_stencil, timesys

    def _extract_ref_legacy(
        self, stencil: SkyStencil, ref: DatasetRef, cutout_mode: CutoutMode = CutoutMode.FULL_EXPOSURE
    ) -> Extraction:
        """Extract a subimage from a fully-resolved `DatasetRef` associated
        with a legacy exposure.

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
        # We know that afw and daf_base are available here since we received
        # an afw Exposure.
        from lsst.afw.image import makeExposure, makeMaskedImage
        from lsst.daf.base import PropertyList

        if ref.id is None:
            raise ValueError(f"A resolved DatasetRef is required; got {ref}.")
        # Timestamp of the cutout extraction.
        now = astropy.time.Time.now()
        timesys = "UTC"
        # Get the WCS and bbox of this dataset unless we are in astropy mode.
        wcs = None
        bbox = None
        pixel_stencil = None
        if cutout_mode not in (CutoutMode.ASTROPY_IMAGE, CutoutMode.ASTROPY_MASKED_IMAGE):
            projection, bbox = self.projection_finder(ref, self.butler, logger=self.logger)
            # Transform the stencil to pixel coordinates.
            pixel_stencil = stencil.to_pixels(projection, bbox)
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
                    assert pixel_stencil is not None
                    cutout = self.butler.get(ref, parameters={"bbox": pixel_stencil.bbox.to_legacy()})
                    timesys = cutout.metadata.get("TIMESYS", timesys)
                case CutoutMode.STRIPPED_EXPOSURE:
                    assert pixel_stencil is not None
                    cutout = self.butler.get(ref, parameters={"bbox": pixel_stencil.bbox.to_legacy()})
                    metadata = cutout.metadata  # Track metadata externally.
                    timesys = metadata.get("TIMESYS", timesys)
                    cutout = makeExposure(cutout.maskedImage, wcs=cutout.wcs)
                case CutoutMode.IMAGE_ONLY:
                    assert pixel_stencil is not None
                    cutout = self.butler.get(
                        ref.makeComponentRef("image"), parameters={"bbox": pixel_stencil.bbox.to_legacy()}
                    )
                    # No metadata so UTC is default.
                    timesys = "UTC"
                case CutoutMode.MASKED_IMAGE:
                    # Rely on the file being cached on first read. Faster than
                    # reading entire exposure.
                    assert pixel_stencil is not None
                    image = self.butler.get(
                        ref.makeComponentRef("image"), parameters={"bbox": pixel_stencil.bbox.to_legacy()}
                    )
                    variance = self.butler.get(
                        ref.makeComponentRef("variance"), parameters={"bbox": pixel_stencil.bbox.to_legacy()}
                    )
                    mask = self.butler.get(
                        ref.makeComponentRef("mask"), parameters={"bbox": pixel_stencil.bbox.to_legacy()}
                    )
                    wcs = self.butler.get(ref.makeComponentRef("wcs"))
                    masked_image = makeMaskedImage(image, mask, variance)
                    cutout = makeExposure(masked_image, wcs=wcs)

                    metadata = self.butler.get(ref.makeComponentRef("metadata"))
                    timesys = metadata.get("TIMESYS", timesys)
                case CutoutMode.ASTROPY_IMAGE | CutoutMode.ASTROPY_MASKED_IMAGE:
                    # Bypass butler and read the pixel HDU directly.
                    cutout, pixel_stencil, timesys = self._read_astropy_hdulist(cutout_mode, stencil, ref)
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

        # Every supported cutout mode produces a pixel stencil above.
        assert pixel_stencil is not None
        return Extraction(
            cutout=cutout,
            sky_stencil=stencil,
            pixel_stencil=pixel_stencil,
            metadata=metadata,
            origin_ref=ref,
        )

    def _extract_ref_v2(
        self, stencil: SkyStencil, ref: DatasetRef, cutout_mode: CutoutMode = CutoutMode.FULL_EXPOSURE
    ) -> Extraction:
        """Extract a subimage from a fully-resolved `DatasetRef` associated
        with a lsst.images model.

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
        pixel_stencil = None

        # There are two distinct modes of operation. One is using the native
        # lsst.images interface, the other is using Astropy to deal with
        # IMAGE and MASKED_IMAGE by using standard FITS conventions.

        metadata: dict[str, str | int] = {}
        if cutout_mode not in (CutoutMode.ASTROPY_IMAGE, CutoutMode.ASTROPY_MASKED_IMAGE):
            # To reduce round-trips to an object store we want to open the file
            # once and then read out the components we need. In theory we can
            # ask Butler get to return multiple components but if we do that
            # we do not know the bounding box to use unless we add a skybox
            # parameter to get that lets us pass in a stencil directly.
            # This means we have to use the URI for direct access, bypassing
            # butler get for now.
            uri = self.butler.getURI(ref)

            # Time the full open and retrieval.
            with time_this(
                self.logger,
                msg="Extract cutout",
                kwargs={"id": str(ref.id), "cutout_mode": str(cutout_mode), "stencil": str(stencil)},
                level=_TIMER_LOG_LEVEL,
            ):
                with lsst.images.serialization.open(uri) as reader:
                    sky_projection = reader.get_component("sky_projection")
                    bbox = reader.get_component("bbox")

                    # Transform the stencil to pixel coordinates.
                    pixel_stencil = stencil.to_pixels(sky_projection, bbox)
                    modern_bbox = pixel_stencil.bbox

                    match cutout_mode:
                        case CutoutMode.FULL_EXPOSURE:
                            cutout = reader.read(bbox=modern_bbox)
                        case CutoutMode.IMAGE_ONLY:
                            cutout = reader.get_component("image", bbox=modern_bbox)
                        case CutoutMode.MASKED_IMAGE | CutoutMode.STRIPPED_EXPOSURE:
                            # A Stripped exposure is meaningless with the
                            # new models since MaskedImage does now carry a
                            # WCS and metadata.
                            cutout = reader.get_component("masked_image", bbox=modern_bbox)
                        case _:
                            raise ValueError(f"Unsupported cutout mode: {cutout_mode}")

            if cutout._opaque_metadata is not None:
                timesys = cutout._opaque_metadata.headers[ExtensionKey()].get("TIMESYS", timesys)

        else:
            # This is the Astropy direct branch.
            with time_this(
                self.logger,
                msg="Extract cutout",
                kwargs={"id": str(ref.id), "cutout_mode": str(cutout_mode), "stencil": str(stencil)},
                level=_TIMER_LOG_LEVEL,
            ):
                # Bypass butler and lsst.images and go straight to the HDUs.
                cutout, pixel_stencil, timesys = self._read_astropy_hdulist(cutout_mode, stencil, ref)

        # Create some FITS metadata with the cutout parameters.
        # Some of these are added as provenance by the butler on write so may
        # no longer be necessary.
        metadata["BTLRUUID"] = ref.id.hex
        metadata["BTLRNAME"] = ref.datasetType.name
        for n, (k, v) in enumerate(ref.dataId.required.items()):
            # Write data ID dictionary sort of like a list of 2-tuples, to make
            # it easier to stay within the FITS 8-char key limit.
            metadata[f"BTLRK{n:03}"] = k
            metadata[f"BTLRV{n:03}"] = v
        stencil.to_fits_metadata(metadata)

        # Record the time and software version.
        now.format = "fits"
        now = now.tai if timesys.lower() == "tai" else now.utc
        metadata["DATE-CUT"] = str(now)
        metadata["CUTVERS"] = __version__

        # Every supported cutout mode produces a pixel stencil above.
        assert pixel_stencil is not None
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
