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

__all__ = (
    "Chain",
    "MatchDatasetTypeName",
    "ProjectionFinder",
    "ReadComponents",
    "ReadComponentsAstropyFits",
    "TryComponentParents",
    "UseSkyMap",
)

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import cast

import astropy.io.fits

import lsst.geom
from lsst.afw.geom import SkyWcs, getImageXY0FromMetadata, makeSkyWcs
from lsst.daf.base import PropertyList
from lsst.daf.butler import Butler, DatasetRef
from lsst.geom import Box2I
from lsst.skymap import BaseSkyMap
from lsst.utils.timer import time_this

_LOG = logging.getLogger(__name__)
_TIMER_LOG_LEVEL = logging.INFO


class ProjectionFinder(ABC):
    """An interface for objects that can find the WCS and bounding box of a
    butler dataset.

    Notes
    -----
    Concrete `ProjectionFinder` implementations are intended to be composed to
    define rules for how to find this projection information for a particular
    dataset type; a finder that cannot handle a particular `DatasetRef` should
    implement `find_projection` to return `None`, and implementations that
    compose (e.g. `Chain`) can use this to decide when to try another nested
    finder.
    """

    @abstractmethod
    def find_projection(
        self, ref: DatasetRef, butler: Butler, logger: logging.Logger | None = None
    ) -> tuple[SkyWcs, Box2I] | None:
        """Run the finder on the given dataset with the given butler.

        Parameters
        ----------
        ref : `DatasetRef`
            Fully-resolved reference to the dataset.
        butler : `Butler`
            Butler client to use for reads.  Need not support writes, and any
            default search collections will be ignored.
        logger : `logging.Logger`, optional
            Logger to use for timing messages.  If `None`, a default logger
            will be used.

        Returns
        -------
        wcs : `SkyWcs` (only if result is not `None`)
            Mapping from sky to pixel coordinates for this dataset.
        bbox : `Box2I` (only if result is not `None`)
            Bounding box of the image dataset (or an image closely associated
            with the dataset) in pixel coordinates.
        """
        raise NotImplementedError()

    def __call__(
        self, ref: DatasetRef, butler: Butler, logger: logging.Logger | None = None
    ) -> tuple[SkyWcs, Box2I]:
        """Call `find_projection` but raise `LookupError` when no projection
        information is found.

        Parameters
        ----------
        ref : `DatasetRef`
            Fully-resolved reference to the dataset.
        butler : `Butler`
            Butler client to use for reads.  Need not support writes, and any
            default search collections will be ignored.
        logger : `logging.Logger`, optional
            Logger to use for timing messages.  If `None`, a default logger
            will be used.

        Returns
        -------
        wcs : `SkyWcs`
            Mapping from sky to pixel coordinates for this dataset.
        bbox : `Box2I`
            Bounding box of the image dataset (or an image closely associated
            with the dataset) in pixel coordinates.
        """
        result = self.find_projection(ref, butler, logger=logger)
        if result is None:
            raise LookupError(f"No way to obtain WCS and bounding box information for ref {ref}.")
        return result

    @staticmethod
    def make_default() -> ProjectionFinder:
        """Return a concrete finder appropriate for most pipelines.

        Returns
        -------
        finder : `ProjectionFinder`
            A finder that prefers to read, use, and cache a skymap when the
            data ID includes tract or patch, and falls back to reading the WCS
            and bbox from the dataset itself (or its parent, if the dataset is
            a component).
        """
        return TryComponentParents(
            Chain(
                UseSkyMap(),
                ReadComponentsAstropyFits(),
                ReadComponents(),
            )
        )


class ReadComponents(ProjectionFinder):
    """A `ProjectionFinder` implementation that reads ``wcs`` and ``bbox``
    from datasets that have them (e.g. ``Exposure``).

    Notes
    -----
    This should usually be the final finder attempted in any chain; it's the
    one most likely to work, but in many cases will not be the most efficient
    or yield the most accurate WCS.
    """

    def find_projection(
        self, ref: DatasetRef, butler: Butler, logger: logging.Logger | None = None
    ) -> tuple[SkyWcs, Box2I] | None:
        # Docstring inherited.
        if {"wcs", "bbox"}.issubset(ref.datasetType.storageClass.allComponents().keys()):
            logger = logger if logger is not None else _LOG
            with time_this(_LOG, msg="Read projection info from butler components", level=_TIMER_LOG_LEVEL):
                wcs = butler.get(ref.makeComponentRef("wcs"))
                bbox = butler.get(ref.makeComponentRef("bbox"))
                return wcs, bbox
        return None


class ReadComponentsAstropyFits(ProjectionFinder):
    """A `ProjectionFinder` implementation that reads ``wcs`` and ``bbox``
    from datasets that have them (e.g. ``Exposure``) and assumes there is
    a WCS associated with an IMAGE HDU in a FITS file.

    Notes
    -----
    This might be more efficient for remote datasets where a full file download
    is needed for butler to work using AFW.
    """

    def find_projection(
        self, ref: DatasetRef, butler: Butler, logger: logging.Logger | None = None
    ) -> tuple[SkyWcs, Box2I] | None:
        # Docstring inherited.
        logger = logger if logger is not None else _LOG
        with time_this(logger, msg="Read projection info using Astropy", level=_TIMER_LOG_LEVEL):
            if {"wcs", "bbox"}.issubset(ref.datasetType.storageClass.allComponents().keys()):
                try:
                    fs, fspath = butler.getURI(ref).to_fsspec()
                    with (
                        fs.open(fspath) as f,
                        astropy.io.fits.open(f) as fits_obj,
                    ):
                        # Look for first pixel HDU.
                        pixel_components = {"mask", "image", "variance"}
                        for i, hdu in enumerate(fits_obj):
                            if i == 0:
                                # Assumes WCS is in the IMAGE extension
                                # and not stored in the primary.
                                continue
                            hdr = hdu.header
                            extname = hdr.get("EXTNAME")
                            if extname and extname.lower() in pixel_components:
                                shape = hdu.shape
                                dimensions = lsst.geom.Extent2I(shape[1], shape[0])
                                pl = PropertyList()
                                pl.update(hdr)
                                # XY0 is defined in the A WCS.
                                xy0 = getImageXY0FromMetadata(pl, "A", strip=False)
                                bbox = lsst.geom.Box2I(xy0, dimensions)
                                wcs = makeSkyWcs(pl)
                                return wcs, bbox
                except Exception:
                    # Any failure and we will try the next option.
                    pass
        return None


class TryComponentParents(ProjectionFinder):
    """A composite `ProjectionFinder` that walks from component dataset to its
    parent composite until its nested finder succeeds.

    Parameters
    ----------
    nested : `ProjectionFinder`
        Nested finder to delegate to.

    Notes
    -----
    This is a good choice for the outermost composite finder, so that the same
    sequence of nested rules are applied to each level of the
    component-composite tree.
    """

    def __init__(self, nested: ProjectionFinder):
        self._nested = nested

    def find_projection(
        self, ref: DatasetRef, butler: Butler, logger: logging.Logger | None = None
    ) -> tuple[SkyWcs, Box2I] | None:
        # Docstring inherited.
        while True:
            if (result := self._nested.find_projection(ref, butler, logger=logger)) is not None:
                return result
            if ref.isComponent():
                ref = ref.makeCompositeRef()
            else:
                return None


class UseSkyMap(ProjectionFinder):
    """A `ProjectionFinder` implementation that reads and caches
    `lsst.skymap.BaseSkyMap` instances, allowing projections for coadds to be
    found without requiring I/O for each one.

    Parameters
    ----------
    dataset_type_name : `str`, optional
        Name of the dataset type used to load `BaseSkyMap` instances.
    collections : `Iterable` [ `str` ]
        Collection search path for skymap datasets.

    Notes
    -----
    This finder assumes any dataset with ``patch`` or ``tract`` dimensions
    should get its WCS and bounding box from the skymap, and that datasets
    without these dimensions never should (i.e. `find_projection` will return
    `None` for these).

    `BaseSkyMap` instances are never removed from the cache after being loaded;
    we expect the number of distinct skymaps to be very small.
    """

    def __init__(self, dataset_type_name: str = "skyMap", collections: Iterable[str] = ("skymaps",)):
        self._dataset_type_name = dataset_type_name
        self._collections = tuple(collections)
        self._cache: dict[str, BaseSkyMap] = {}

    def find_projection(
        self, ref: DatasetRef, butler: Butler, logger: logging.Logger | None = None
    ) -> tuple[SkyWcs, Box2I] | None:
        # Docstring inherited.
        if "tract" in ref.dataId.dimensions:
            assert "skymap" in ref.dataId.dimensions, "Guaranteed by expected dimension schema."
            if (skymap := self._cache.get(cast(str, ref.dataId["skymap"]))) is None:
                skymap = butler.get(
                    self._dataset_type_name, skymap=ref.dataId["skymap"], collections=self._collections
                )
            tractInfo = skymap[ref.dataId["tract"]]
            if "patch" in ref.dataId.dimensions:
                patchInfo = tractInfo[ref.dataId["patch"]]
                return patchInfo.wcs, patchInfo.outer_bbox
            else:
                return tractInfo.wcs, tractInfo.bbox
        return None


class Chain(ProjectionFinder):
    """A composite `ProjectionFinder` that attempts each finder in a sequence
    until one succeeds.

    Parameters
    ----------
    *nested : `ProjectionFinder`
        Nested finders to delegate to, in order.
    """

    def __init__(self, *nested: ProjectionFinder):
        self._nested = tuple(nested)

    def find_projection(
        self, ref: DatasetRef, butler: Butler, logger: logging.Logger | None = None
    ) -> tuple[SkyWcs, Box2I] | None:
        # Docstring inherited.
        for f in self._nested:
            if (result := f.find_projection(ref, butler, logger=logger)) is not None:
                return result
        return None


class MatchDatasetTypeName(ProjectionFinder):
    """A composite `ProjectionFinder` that delegates to different nested
    finders based on whether the dataset type name matches a regular
    expression.

    Parameters
    ----------
    regex : `str`
        Regular expression the dataset type name must match (in full).
    on_match : `ProjectionFinder`, optional
        Finder to try when the match succeeds, or `None` to return `None`.
    otherwise : `ProjectionFinder`, optional
        Finder to try when the match does not succeed, or `None` to return
        `None`.
    """

    def __init__(
        self,
        regex: str,
        on_match: ProjectionFinder | None = None,
        otherwise: ProjectionFinder | None = None,
    ):
        self._regex = re.compile(regex)
        self._on_match = on_match
        self._otherwise = otherwise

    def find_projection(
        self, ref: DatasetRef, butler: Butler, logger: logging.Logger | None = None
    ) -> tuple[SkyWcs, Box2I] | None:
        # Docstring inherited.
        if self._regex.match(ref.datasetType.name):
            if self._on_match is not None:
                return self._on_match.find_projection(ref, butler, logger=logger)
        else:
            if self._otherwise is not None:
                return self._otherwise.find_projection(ref, butler, logger=logger)
        return None
