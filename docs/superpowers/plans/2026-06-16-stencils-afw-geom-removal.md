# Removing afw and lsst.geom from the stencils package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `stencils.py` free of `lsst.afw` and `lsst.geom`, take `lsst.sphgeom` types in its constructors, mask onto an `lsst.images.Mask` by plane name, and move all FITS-WCS parsing into one `lsst.images`-based helper that `projection_finders.py` and `_image_cutout.py` share, so `find_projection` returns modern types.

**Architecture:** A new `_fits_projection.py` helper parses a FITS header into an `lsst.images.SkyProjection` and parent `Box` using the AST wrapper that `lsst.images` already exposes (`lsst.images._transforms._ast`), so no astshim/starlink code lives in this package. `stencils.py` keeps both masking backends (`AST`, `SPHGEOM`) but speaks only `lsst.images`/`sphgeom`/astropy/starlink-pyast. afw objects produced by legacy data and skymaps are converted at three seams, all outside `stencils.py`.

**Tech Stack:** `lsst.images` (`SkyProjection`, `Box`, `Mask`, `MaskSchema`, `MaskPlane`, `GeneralFrame`, `_transforms._ast`), `lsst.sphgeom` (`LonLat`, `Angle`, `UnitVector3d`, `Circle`, `ConvexPolygon`), `starlink.Ast`, astropy, numpy. afw/`lsst.geom` only in `projection_finders.py`/`_image_cutout.py` seams and in tests.

**Design spec:** `docs/superpowers/specs/2026-06-16-stencils-afw-geom-removal-design.md`.

---

## Background the engineer needs

- Files in scope: `python/lsst/dax/images/cutout/{stencils.py,projection_finders.py,_image_cutout.py}`, a new `_fits_projection.py`, and the tests `tests/{test_stencils.py,bench_stencils.py}` plus a new `tests/test_fits_projection.py`.
- The afw/starlink/sphgeom C++ extensions need the EUPS-configured library path. **Run every test/Python command in a shell prepared like this** (copy verbatim; it is the proven recipe in this checkout):

  ```bash
  cd /Users/timj/work/lsstsw/build/dax_images_cutout
  source /Users/timj/work/lsstsw/bin/envconfig
  source "$EUPS_DIR/bin/setups.sh"
  setup -k -r .
  FP=""; for p in sphgeom afw images geom daf_base utils pex_exceptions cpputils base log astshim; do \
    loc=$(eups list -d -s "$p" 2>/dev/null | awk '{print $NF}'); \
    [ -n "$loc" ] && [ -d "$loc/lib" ] && FP="$FP:$loc/lib"; done
  export DYLD_FALLBACK_LIBRARY_PATH="$FP"
  # now: pytest tests/test_stencils.py -v   (etc.)
  ```

- Coordinate conventions (validated in this checkout):
  - `lsst.images.Box.factory[y_slice, x_slice]` is `[y, x]`. `Box.shape` is `YX(y, x)`; `Box.x`/`Box.y` are `Interval` with `.min`, `.max` (inclusive), `.start`, `.stop` (= max + 1), `.size`. `Box.meshgrid()` returns `XY(x, y)`, each a `(ny, nx)` float array.
  - `lsst.images.Mask` is constructed as `Mask(schema=MaskSchema([MaskPlane(name, desc)]), bbox=box)`. `mask.set(plane, bool2d)` sets the named plane where `bool2d` (shape `mask.bbox.shape == (ny, nx)`) is `True`, leaving other pixels unchanged. `mask.get(plane)` returns the `(ny, nx)` boolean array. `mask.schema.names` lists plane names. There is no way to add a plane after construction; this plan never tries to.
  - `lsst.sphgeom`: `LonLat.fromRadians(lon, lat)`, `LonLat.fromDegrees(lon, lat)`, `LonLat(unit_vector3d)`, `lonlat.getLon()/getLat()` → `sphgeom.Angle` with `.asRadians()/.asDegrees()`. `sphgeom.Angle(radians)`. `UnitVector3d(lonlat)`. `Circle(UnitVector3d, Angle)`, `circle.getCenter()` → `UnitVector3d`, `circle.getOpeningAngle()` → `sphgeom.Angle`. `ConvexPolygon([UnitVector3d, ...])`.
  - `astropy.coordinates.SkyCoord.directional_offset_by(position_angle, separation)` is the great-circle offset (position angle measured East of North), matching `lsst.geom.SpherePoint.offset`.
- AST masking semantics (unchanged from the existing backend): `region.mask(map, inside, lbnd, ubnd, array, val)` with `inside=1`, `lbnd=[x_min, y_min]`, `ubnd=[x_max, y_max]`, `array` indexed `[y, x]`. Always rasterize into a fresh zero scratch array, never a live mask plane.
- The `lsst.images` AST wrapper lives at `lsst.images._transforms._ast`. It re-exports astshim when present, else starlink-pyast-backed shims with the **astshim** interface (`FitsChan`, `FrameSet`, `StringStream`, `USING_STARLINK_PYAST`). `SkyProjection.from_ast_frame_set` consumes a wrapper `FrameSet`. Validated: `_ast.FitsChan(_ast.StringStream(header.tostring())).read()` yields one `FrameSet` with a base `GRID` frame, a current `SKY` frame, and a frame whose `.ident` is `"A"`; the `GRID -> "A"` mapping applied to grid `(1, 1)` reproduces `CRVAL*A` (the parent origin / XY0).

## File structure

- Create `python/lsst/dax/images/cutout/_fits_projection.py` — sole home of FITS-header → `(SkyProjection, Box)` parsing; shaped to move upstream into `lsst.images` later.
- Modify `python/lsst/dax/images/cutout/stencils.py` — sphgeom constructors, sphgeom/astropy internal geometry, `to_pixels(SkyProjection, Box)`, `set_mask(Mask, plane)`; no afw/`lsst.geom`.
- Modify `python/lsst/dax/images/cutout/projection_finders.py` — return `(SkyProjection, Box)`; use the helper; convert afw inputs with `from_legacy`.
- Modify `python/lsst/dax/images/cutout/_image_cutout.py` — unify the two astropy branches through one new method that calls the helper; pass modern types to `to_pixels`; mask via `lsst.images.Mask`.
- Modify `tests/test_stencils.py` — sphgeom construction, `SkyProjection` input, `lsst.images.Mask` coverage read-back.
- Create `tests/test_fits_projection.py` — unit test for the helper, with `getImageXY0FromMetadata` as the XY0 oracle.
- Modify `tests/bench_stencils.py` — new constructor/types.

---

## Task 1: FITS-header projection helper

**Files:**
- Create: `python/lsst/dax/images/cutout/_fits_projection.py`
- Test: `tests/test_fits_projection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fits_projection.py`:

```python
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

import unittest

import astropy.io.fits
import numpy as np

from lsst.afw.geom import getImageXY0FromMetadata
from lsst.daf.base import PropertyList
from lsst.dax.images.cutout._fits_projection import projection_and_bbox_from_fits_header
from lsst.geom import Box2I, Extent2I
from lsst.images import Box, SkyProjection


def _make_header() -> astropy.io.fits.Header:
    """A primary gnomonic sky WCS plus an 'A' alternate WCS encoding XY0."""
    header = astropy.io.fits.Header()
    cards = {
        "WCSAXES": 2,
        "CRPIX1": 10.0,
        "CRPIX2": 20.0,
        "CRVAL1": 12.0,
        "CRVAL2": 13.0,
        "CD1_1": -2.78e-05,
        "CD2_2": 2.78e-05,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1A": 1.0,
        "CRPIX2A": 1.0,
        "CRVAL1A": 100.0,
        "CRVAL2A": 200.0,
        "CD1_1A": 1.0,
        "CD2_2A": 1.0,
        "CTYPE1A": "LINEAR",
        "CTYPE2A": "LINEAR",
    }
    for key, value in cards.items():
        header[key] = value
    return header


class FitsProjectionTestCase(unittest.TestCase):
    """Tests for the FITS-header projection helper."""

    def setUp(self) -> None:
        self.header = _make_header()
        # (ny, nx) as returned by astropy ``hdu.shape``.
        self.shape = (64, 48)

    def test_projection_round_trips_reference_pixel(self) -> None:
        projection, _ = projection_and_bbox_from_fits_header(self.header, self.shape)
        self.assertIsInstance(projection, SkyProjection)
        sky = projection.pixel_to_sky(x=np.array([10.0]), y=np.array([20.0]))
        # The reference pixel CRPIX maps to CRVAL.
        self.assertAlmostEqual(float(sky.ra.deg[0]), 12.0, places=6)
        self.assertAlmostEqual(float(sky.dec.deg[0]), 13.0, places=6)

    def test_bbox_matches_afw_reference(self) -> None:
        _, bbox = projection_and_bbox_from_fits_header(self.header, self.shape)
        self.assertIsInstance(bbox, Box)
        # afw reference for the parent bbox (XY0 from the 'A' WCS + dimensions).
        pl = PropertyList()
        pl.update(self.header)
        xy0 = getImageXY0FromMetadata(pl, "A", strip=False)
        dimensions = Extent2I(self.shape[1], self.shape[0])
        self.assertEqual(bbox, Box.from_legacy(Box2I(xy0, dimensions)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_fits_projection.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` for `_fits_projection`.

- [ ] **Step 3: Implement the helper**

Create `python/lsst/dax/images/cutout/_fits_projection.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_fits_projection.py -v`
Expected: PASS (both tests). If `test_bbox_matches_afw_reference` fails by a constant offset, the XY0 convention is wrong: inspect `getImageXY0FromMetadata`'s result versus `origin` and correct the grid sample point in `projection_and_bbox_from_fits_header` (do not add an ad-hoc offset — match the reference).

- [ ] **Step 5: Commit**

```bash
git add python/lsst/dax/images/cutout/_fits_projection.py tests/test_fits_projection.py
git commit -m "Add FITS-header to SkyProjection/Box helper"
```

---

## Task 2: Rewrite `stencils.py` onto sphgeom and `lsst.images.Mask`

**Files:**
- Modify (full rewrite of body): `python/lsst/dax/images/cutout/stencils.py`
- Test: `tests/test_stencils.py` (rewritten in Step 1)

- [ ] **Step 1: Replace the test file with the new-API version**

Overwrite `tests/test_stencils.py` with:

```python
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

import unittest

import astropy.coordinates
import astropy.units as u
import numpy as np

import lsst.sphgeom
from lsst.afw.geom import SkyWcs, makeCdMatrix, makeSkyWcs
from lsst.dax.images.cutout.stencils import (
    MaskBackend,
    SkyCircle,
    SkyStencil,
    _round_box_from_bounds,
)
from lsst.geom import Box2I, Point2D, Point2I, SpherePoint, arcseconds, degrees
from lsst.images import Box, GeneralFrame, Mask, MaskPlane, MaskSchema, SkyProjection
from lsst.sphgeom import Angle, LonLat, UnitVector3d  # noqa: F401  (LonLat/Angle used in eval(repr))


def _arcsec(value: float) -> Angle:
    """A `lsst.sphgeom.Angle` for ``value`` arcseconds."""
    return Angle((value * u.arcsec).to_value(u.rad))


class ModuleHelpersTestCase(unittest.TestCase):
    """Tests for module-level helpers that survive the rewrite."""

    def test_round_box_from_bounds(self) -> None:
        # x in [4.6, 9.4], y in [2.6, 5.4] -> box [3:6, 5:10] in [y, x].
        box = _round_box_from_bounds(4.6, 9.4, 2.6, 5.4)
        self.assertEqual(box, Box.factory[3:6, 5:10])

    def test_mask_backend_members(self) -> None:
        self.assertEqual({b.name for b in MaskBackend}, {"AST", "SPHGEOM"})


class SkyCircleTestCase(unittest.TestCase):
    """Tests for `SkyCircle`."""

    def setUp(self) -> None:
        self.center = LonLat.fromDegrees(12.0, 13.0)
        self.instance = SkyCircle(self.center, _arcsec(1.0))
        # An afw WCS at the same center, used for the brute-force reference and
        # to build a SkyProjection input.
        self.wcs_center = SpherePoint(12.0, 13.0, degrees)

    def _wcs(self) -> SkyWcs:
        return makeSkyWcs(Point2D(5.0, 7.0), self.wcs_center, makeCdMatrix(0.1 * arcseconds))

    def test_from_astropy(self) -> None:
        other = SkyCircle.from_astropy(
            astropy.coordinates.SkyCoord(
                frame="icrs", ra=12.0 * astropy.units.deg, dec=13.0 * astropy.units.deg
            ),
            astropy.coordinates.Angle(1.0 * astropy.units.arcsec),
        )
        self.assertEqual(self.instance.region, other.region)

    def test_repr(self) -> None:
        self.assertEqual(eval(repr(self.instance)).region, self.instance.region)

    def test_to_pixel(self) -> None:
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        _check_to_pixel(self, self.instance, self._wcs(), bbox, backend=MaskBackend.AST, max_missing=2, max_extra=2)

    def test_to_polygon(self) -> None:
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        polygon_stencil = self.instance.to_polygon()
        self.assertNotEqual(
            self.instance.region.relate(polygon_stencil.region.getBoundingCircle()), lsst.sphgeom.DISJOINT
        )
        _check_to_pixel(self, polygon_stencil, self._wcs(), bbox, backend=MaskBackend.AST, max_missing=6, max_extra=6)

    def test_ast_sky_region_circle_contains_center(self) -> None:
        region = self.instance._ast_sky_region()
        self.assertTrue(
            region.pointinregion([self.center.getLon().asRadians(), self.center.getLat().asRadians()])
        )

    def test_to_pixel_sphgeom(self) -> None:
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        _check_to_pixel(self, self.instance, self._wcs(), bbox, backend=MaskBackend.SPHGEOM, max_missing=0, max_extra=0)

    def test_to_pixel_sphgeom_polygon(self) -> None:
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        polygon_stencil = self.instance.to_polygon()
        _check_to_pixel(self, polygon_stencil, self._wcs(), bbox, backend=MaskBackend.SPHGEOM, max_missing=0, max_extra=0)


class SkyPolygonTestCase(unittest.TestCase):
    """Tests for `SkyPolygon` orientation handling."""

    def setUp(self) -> None:
        self.instance = SkyCircle(LonLat.fromDegrees(12.0, 13.0), _arcsec(2.0)).to_polygon(n_vertices=8)

    def test_ast_sky_region_polygon_contains_centroid(self) -> None:
        region = self.instance._ast_sky_region()
        lonlat = lsst.sphgeom.LonLat(self.instance.region.getCentroid())
        self.assertTrue(region.pointinregion([lonlat.getLon().asRadians(), lonlat.getLat().asRadians()]))


class BackendComparisonTestCase(unittest.TestCase):
    """Assert the AST and sphgeom backends agree on bbox and masked pixels."""

    def setUp(self) -> None:
        self.center = LonLat.fromDegrees(12.0, 13.0)
        wcs = makeSkyWcs(Point2D(5.0, 7.0), SpherePoint(12.0, 13.0, degrees), makeCdMatrix(0.1 * arcseconds))
        self.projection = SkyProjection.from_legacy(wcs, GeneralFrame(unit=u.pix))
        self.box = Box.from_legacy(Box2I(Point2I(-16, -13), Point2I(26, 27)))

    def _masked_array(self, stencil: SkyStencil, backend: MaskBackend) -> tuple[np.ndarray, Box]:
        pixel_stencil = stencil.to_pixels(self.projection, self.box, backend=backend)
        mask = Mask(schema=MaskSchema([MaskPlane("STENCIL", "stencil coverage")]), bbox=self.box)
        pixel_stencil.set_mask(mask, "STENCIL")
        return mask.get("STENCIL"), pixel_stencil.bbox

    def test_backends_agree_circle(self) -> None:
        circle = SkyCircle(self.center, _arcsec(1.0))
        ast_mask, ast_box = self._masked_array(circle, MaskBackend.AST)
        sph_mask, sph_box = self._masked_array(circle, MaskBackend.SPHGEOM)
        self.assertEqual(ast_box, sph_box)
        self.assertEqual(int(np.sum(ast_mask != sph_mask)), 0)

    def test_backends_agree_polygon(self) -> None:
        polygon = SkyCircle(self.center, _arcsec(1.0)).to_polygon()
        ast_mask, ast_box = self._masked_array(polygon, MaskBackend.AST)
        sph_mask, sph_box = self._masked_array(polygon, MaskBackend.SPHGEOM)
        self.assertEqual(ast_box, sph_box)
        self.assertLessEqual(int(np.sum(ast_mask != sph_mask)), 12)


def _brute_force_stencil_array(sky_stencil: SkyStencil, wcs: SkyWcs, bbox: Box2I) -> np.ndarray:
    """Boolean ``(ny, nx)`` array, `True` where a pixel center is in the stencil."""
    x1 = np.arange(bbox.getBeginX(), bbox.getEndX())
    y1 = np.arange(bbox.getBeginY(), bbox.getEndY())
    x2, y2 = np.meshgrid(x1, y1)
    pixels = np.zeros((2, bbox.getArea()), dtype=float)
    pixels[0, :] = x2.flatten()
    pixels[1, :] = y2.flatten()
    sky = wcs.getTransform().getMapping().applyForward(pixels)
    contained = sky_stencil.region.contains(sky[0, :], sky[1, :])
    return contained.reshape(bbox.getHeight(), bbox.getWidth())


def _check_to_pixel(
    test_case: unittest.TestCase,
    sky_stencil: SkyStencil,
    wcs: SkyWcs,
    bbox: Box2I,
    *,
    backend: MaskBackend = MaskBackend.AST,
    max_missing: int = 0,
    max_extra: int = 0,
    plot: bool = False,
) -> None:
    """Check a `SkyStencil.to_pixels` result against a brute-force reference."""
    projection = SkyProjection.from_legacy(wcs, GeneralFrame(unit=u.pix))
    box = Box.from_legacy(bbox)
    pixel_stencil = sky_stencil.to_pixels(projection, box, backend=backend)
    test_case.assertTrue(box.contains(pixel_stencil.bbox))
    mask = Mask(schema=MaskSchema([MaskPlane("STENCIL", "stencil coverage")]), bbox=box)
    pixel_stencil.set_mask(mask, "STENCIL")
    got = mask.get("STENCIL")
    check_array = _brute_force_stencil_array(sky_stencil, wcs, bbox)
    missing = np.logical_and(check_array, np.logical_not(got))
    extra = np.logical_and(got, np.logical_not(check_array))
    if plot:
        from matplotlib import pyplot

        display_array = np.zeros((bbox.getHeight(), bbox.getWidth(), 3), dtype=np.uint8)
        display_array[:, :, 0] = 255 * check_array
        display_array[:, :, 1] = 255 * got
        pyplot.imshow(display_array, origin="lower", interpolation="nearest")
        pyplot.title("red=check, green=SkyStencil.to_pixel, yellow=both")
        pyplot.show()
    test_case.assertLessEqual(int(missing.sum()), max_missing)
    test_case.assertLessEqual(int(extra.sum()), max_extra)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_stencils.py -v`
Expected: FAIL — `ImportError` (e.g. `_as_box` removed from the import is fine, but `SkyCircle(LonLat, Angle)` and `set_mask(mask, "STENCIL")` hit the old afw-based module) or `TypeError`. Any failure here is acceptable; it drives the rewrite.

- [ ] **Step 3: Rewrite `stencils.py`**

Replace the entire contents of `python/lsst/dax/images/cutout/stencils.py` with:

```python
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
    "MaskBackend",
    "PixelStencil",
    "SkyCircle",
    "SkyPolygon",
    "SkyStencil",
    "StencilNotContainedError",
)

import enum
import struct
from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableMapping
from hashlib import blake2b
from typing import Any

import astropy.coordinates
import astropy.units as u
import numpy as np
import starlink.Ast as Ast
from astropy.coordinates import SkyCoord

import lsst.sphgeom
from lsst.daf.base import PropertyList
from lsst.images import Box, Mask, NoOverlapError, SkyProjection
from lsst.images.utils import round_half_down, round_half_up
from lsst.sphgeom import Angle, LonLat, UnitVector3d


class MaskBackend(enum.Enum):
    """Selects the algorithm used to rasterize a stencil onto pixels."""

    AST = enum.auto()
    """Mask using starlink-pyast ``Region.mask`` on the true sky region."""

    SPHGEOM = enum.auto()
    """Mask by testing pixel centers against the ``lsst.sphgeom`` region."""


def _round_box_from_bounds(x_min: float, x_max: float, y_min: float, y_max: float) -> Box:
    """Build an integer pixel `Box` from continuous coordinate bounds.

    Uses the same rounding convention as `lsst.images.Region.bbox`, so that
    pixels whose centers lie within the bounds are included.
    """
    return Box.factory[
        round_half_up(y_min) : round_half_down(y_max) + 1,
        round_half_up(x_min) : round_half_down(x_max) + 1,
    ]


def _skycoord_from_lonlat(lonlat: LonLat) -> SkyCoord:
    """Return an ICRS `astropy.coordinates.SkyCoord` for a `sphgeom.LonLat`."""
    return SkyCoord(
        ra=lonlat.getLon().asRadians() * u.rad,
        dec=lonlat.getLat().asRadians() * u.rad,
        frame="icrs",
    )


class _AstLineSource:
    """Feeds AST-native text lines to a `starlink.Ast.Channel`."""

    def __init__(self, text: str) -> None:
        self._lines = text.splitlines()

    def astsource(self) -> str | None:
        return self._lines.pop(0) if self._lines else None


def _starlink_sky_to_pixel(projection: SkyProjection) -> Ast.Mapping:
    """Return the sky->pixel mapping of ``projection`` as a starlink-pyast
    Mapping.

    ``lsst.images`` may wrap AST with either astshim or starlink-pyast
    depending on the runtime environment.  The transform is serialized to
    AST's native text form via the public `~lsst.images.Transform.show`
    method and re-read with starlink-pyast, so that all region masking
    happens in starlink-pyast regardless of which wrapper ``lsst.images``
    uses internally.
    """
    return Ast.Channel(_AstLineSource(projection.sky_to_pixel_transform.show())).read()


class StencilNotContainedError(RuntimeError):
    """Exception that may be raised when a stencil is not with a desired
    bounding box.
    """


class PixelStencil(ABC):
    """An image cutout stencil defined in pixel coordinates."""

    @property
    @abstractmethod
    def bbox(self) -> Box:
        """Bounding box of this stencil, as a `lsst.images.Box`."""
        raise NotImplementedError()

    @abstractmethod
    def _coverage(self) -> np.ndarray:
        """Boolean array over `bbox`, `True` for pixels the stencil covers.

        The array has shape ``bbox.shape`` (``(ny, nx)``).
        """
        raise NotImplementedError()

    def set_mask(self, mask: Mask, plane: str) -> None:
        """Set a mask plane for pixels whose centers the stencil covers.

        Parameters
        ----------
        mask : `lsst.images.Mask`
            Mask to modify in-place.  Its schema must already define ``plane``
            and its bounding box must contain `bbox`.
        plane : `str`
            Name of the mask plane to set where the stencil covers a pixel.
        """
        covered = self._coverage()
        full = np.zeros(mask.bbox.shape, dtype=bool)
        y_off = self.bbox.y.min - mask.bbox.y.min
        x_off = self.bbox.x.min - mask.bbox.x.min
        full[y_off : y_off + self.bbox.shape.y, x_off : x_off + self.bbox.shape.x] = covered
        mask.set(plane, full)


class _AstPixelRegion(PixelStencil):
    """Pixel-coordinate stencil backed by a starlink-pyast sky `Region`.

    Parameters
    ----------
    sky_region : `starlink.Ast.Region`
        The stencil region expressed in an ICRS sky frame.
    sky_to_pixel : `starlink.Ast.Mapping`
        Mapping whose forward direction transforms sky coordinates to pixels,
        as required by ``Region.mask`` (region frame to grid).
    bbox : `lsst.images.Box`
        Bounding box the stencil is restricted to.
    """

    def __init__(self, sky_region: Ast.Region, sky_to_pixel: Ast.Mapping, bbox: Box) -> None:
        self._sky_region = sky_region
        self._sky_to_pixel = sky_to_pixel
        self._bbox = bbox

    @property
    def bbox(self) -> Box:
        # Docstring inherited.
        return self._bbox

    def _coverage(self) -> np.ndarray:
        # Docstring inherited.
        scratch = np.zeros(self._bbox.shape, dtype=np.int64)
        self._sky_region.mask(
            self._sky_to_pixel,
            1,
            [self._bbox.x.min, self._bbox.y.min],
            [self._bbox.x.max, self._bbox.y.max],
            scratch,
            1,
        )
        return scratch != 0


class _SphgeomPixelRegion(PixelStencil):
    """Pixel-coordinate stencil that tests pixel centers against a sphgeom
    region.

    Parameters
    ----------
    region : `lsst.sphgeom.Region`
        Sky region to test pixel centers against.
    projection : `lsst.images.SkyProjection`
        Mapping used to convert pixel centers to sky coordinates.
    bbox : `lsst.images.Box`
        Bounding box the stencil is restricted to.
    """

    def __init__(self, region: lsst.sphgeom.Region, projection: SkyProjection, bbox: Box) -> None:
        self._region = region
        self._projection = projection
        self._bbox = bbox

    @property
    def bbox(self) -> Box:
        # Docstring inherited.
        return self._bbox

    def _coverage(self) -> np.ndarray:
        # Docstring inherited.
        grid = self._bbox.meshgrid()
        sky = self._projection.pixel_to_sky(x=grid.x.ravel(), y=grid.y.ravel())
        return self._region.contains(sky.ra.radian, sky.dec.radian).reshape(self._bbox.shape)


class SkyStencil(ABC):
    """An image cutout stencil defined in sky (ICRS) coordinates."""

    _clip: bool

    def to_pixels(
        self,
        projection: SkyProjection,
        bbox: Box,
        *,
        backend: MaskBackend = MaskBackend.AST,
    ) -> PixelStencil:
        """Transform to a pixel-coordinate stencil.

        Parameters
        ----------
        projection : `lsst.images.SkyProjection`
            Mapping from sky coordinates to pixel coordinates.
        bbox : `lsst.images.Box`
            Bounds that the returned stencil must lie within.
        backend : `MaskBackend`, optional
            Algorithm used to rasterize the stencil.  Defaults to
            `MaskBackend.AST`.

        Returns
        -------
        pixels : `PixelStencil`
            Pixel-coordinate stencil object.  `PixelStencil.bbox` is guaranteed
            to be contained by the given ``bbox``.

        Raises
        ------
        StencilNotContainedError
            Raised when ``clip`` is `False` and the pixel-coordinate stencil
            does not lie within ``bbox``.
        """
        tight = self._pixel_bbox(projection)
        final = self._resolve_box(tight, bbox)
        if backend is MaskBackend.AST:
            return _AstPixelRegion(self._ast_sky_region(), _starlink_sky_to_pixel(projection), final)
        if backend is MaskBackend.SPHGEOM:
            return _SphgeomPixelRegion(self.region, projection, final)
        raise ValueError(f"Unknown mask backend: {backend!r}.")

    def _pixel_bbox(self, projection: SkyProjection) -> Box:
        """Compute the tight pixel bounding box of this stencil.

        The boundary is sampled on the sky and transformed to pixels, so both
        mask backends share an identical bounding box.
        """
        xy = projection.sky_to_pixel(self._boundary_skycoord())
        return _round_box_from_bounds(
            float(np.min(xy.x)), float(np.max(xy.x)), float(np.min(xy.y)), float(np.max(xy.y))
        )

    def _resolve_box(self, tight: Box, box: Box) -> Box:
        """Clip ``tight`` to ``box`` or raise if not contained.

        Honors the stencil's ``clip`` flag: when clipping, returns the
        intersection (raising `StencilNotContainedError` when disjoint); when
        not clipping, returns ``tight`` only if ``box`` contains it.
        """
        if self._clip:
            try:
                return tight.intersection(box)
            except NoOverlapError:
                raise StencilNotContainedError(f"{self} does not overlap {box}.") from None
        if not box.contains(tight):
            raise StencilNotContainedError(f"{self} has pixel bbox {tight}, which is not within {box}.")
        return tight

    @abstractmethod
    def _ast_sky_region(self) -> Ast.Region:
        """Return a starlink-pyast `Region` in an ICRS sky frame."""
        raise NotImplementedError()

    @abstractmethod
    def _boundary_skycoord(self) -> SkyCoord:
        """Return sky coordinates sampling the stencil boundary.

        Used to size the pixel bounding box.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def region(self) -> lsst.sphgeom.Region:
        """A `lsst.sphgeom.Region` that bounds this stencil on the sky."""
        raise NotImplementedError()

    @abstractmethod
    def to_fits_metadata(self, metadata: PropertyList | MutableMapping[str, Any]) -> None:
        """Write FITS header entries that describe the stencil."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def fingerprint(self) -> bytes:
        """A 16-byte blob that is unique to this stencil."""
        raise NotImplementedError()


class SkyCircle(SkyStencil):
    """A sky-coordinate circular stencil.

    Parameters
    ----------
    center : `lsst.sphgeom.LonLat`
        The center of the circle, in ICRS (longitude, latitude).
    radius : `lsst.sphgeom.Angle`
        Radius of the circle.
    clip : `bool`, optional
        If `True` (`False` is default), clip pixel stencils returned by
        `to_pixels` instead of raising `StencilNotContainedError`.
    """

    #: Number of points used to sample the circle boundary when sizing the
    #: pixel bounding box.
    BOUNDARY_SAMPLES = 64

    def __init__(self, center: LonLat, radius: Angle, clip: bool = False):
        self._center = center
        self._radius = radius
        self._clip = clip

    def __repr__(self) -> str:
        return (
            f"SkyCircle(LonLat.fromRadians({self._center.getLon().asRadians()!r}, "
            f"{self._center.getLat().asRadians()!r}), "
            f"Angle({self._radius.asRadians()!r}), clip={self._clip!r})"
        )

    @classmethod
    def from_astropy(
        cls, center: astropy.coordinates.SkyCoord, radius: astropy.coordinates.Angle, clip: bool = False
    ) -> SkyCircle:
        """Construct from `astropy.coordinates` arguments.

        Parameters
        ----------
        center : `astropy.coordinates.SkyCoord`
            The center of the circle, in ICRS (ra, dec).  Must be scalar.
        radius : `astropy.coordinates.Angle`
            Radius of the circle.  Must be scalar.
        clip : `bool`, optional
            If `True` (`False` is default), clip pixel stencils returned by
            `to_pixels` instead of raising `StencilNotContainedError`.

        Returns
        -------
        stencil : `SkyCircle`
            Circular stencil.
        """
        return cls(
            center=_lonlat_from_astropy(center),
            radius=_angle_from_astropy(radius),
            clip=clip,
        )

    @classmethod
    def from_sphgeom(cls, circle: lsst.sphgeom.Circle, clip: bool = False) -> SkyCircle:
        """Construct from a `lsst.sphgeom.Circle` instance."""
        return cls(LonLat(circle.getCenter()), circle.getOpeningAngle(), clip=clip)

    def to_polygon(self, n_vertices: int = 16) -> SkyPolygon:
        """Return a polygon sky stencil that approximates this circle.

        Parameters
        ----------
        n_vertices : `int`, optional
            Number of polygon vertices in the approximation.

        Returns
        -------
        polygon : `SkyPolygon`
            Polygon approximation.

        Notes
        -----
        This helper is retained for callers that want a polygon approximation;
        it is no longer used by `to_pixels`, which masks the true circle.
        """
        center = _skycoord_from_lonlat(self._center)
        position_angle = (np.arange(n_vertices) / n_vertices * 2.0 * np.pi) * u.rad
        radius = astropy.coordinates.Angle(self._radius.asRadians() * u.rad)
        points = center.directional_offset_by(position_angle, radius)
        vertices = [LonLat.fromRadians(float(p.ra.rad), float(p.dec.rad)) for p in points]
        return SkyPolygon(vertices, clip=self._clip)

    def _ast_sky_region(self) -> Ast.Region:
        # Docstring inherited.
        return Ast.Circle(
            Ast.SkyFrame("System=ICRS"),
            1,
            [self._center.getLon().asRadians(), self._center.getLat().asRadians()],
            [self._radius.asRadians()],
        )

    def _boundary_skycoord(self) -> SkyCoord:
        # Docstring inherited.
        center = _skycoord_from_lonlat(self._center)
        position_angle = (np.arange(self.BOUNDARY_SAMPLES) / self.BOUNDARY_SAMPLES * 2.0 * np.pi) * u.rad
        radius = astropy.coordinates.Angle(self._radius.asRadians() * u.rad)
        return center.directional_offset_by(position_angle, radius)

    @property
    def region(self) -> lsst.sphgeom.Region:
        # Docstring inherited.
        return lsst.sphgeom.Circle(UnitVector3d(self._center), self._radius)

    def to_fits_metadata(self, metadata: PropertyList | MutableMapping[str, Any]) -> None:
        # Docstring inherited.
        metadata["ST_TYPE"] = "CIRCLE"
        metadata["ST_RA"] = self._center.getLon().asDegrees()
        metadata["ST_DEC"] = self._center.getLat().asDegrees()
        metadata["ST_RAD"] = self._radius.asDegrees()

    @property
    def fingerprint(self) -> bytes:
        # Docstring inherited.
        hasher = blake2b(digest_size=16)
        hasher.update(b"CIRCLE")
        hasher.update(struct.pack("!d", self._center.getLon().asRadians()))
        hasher.update(struct.pack("!d", self._center.getLat().asRadians()))
        hasher.update(struct.pack("!d", self._radius.asRadians()))
        return hasher.digest()


class SkyPolygon(SkyStencil):
    """A sky-coordinate stencil in the shape of a great-circle polygon.

    Parameters
    ----------
    vertices : `Iterable` [ `lsst.sphgeom.LonLat` ]
        Vertices of the polygon, CCW when looking out from the origin.
        Implicitly closed (the first vertex should not be duplicated as the
        last).
    clip : `bool`, optional
        If `True` (`False` is default), clip pixel stencils returned by
        `to_pixels` instead of raising `StencilNotContainedError`.

    Notes
    -----
    Vertex orientation is not checked at construction, and incorrect
    orientation may result in unspecified failures in `to_pixels`.
    """

    def __init__(self, vertices: Iterable[LonLat], clip: bool = False):
        self._vertices = tuple(vertices)
        self._clip = clip

    @classmethod
    def from_astropy(cls, vertices: astropy.coordinates.SkyCoord, clip: bool = False) -> SkyPolygon:
        """Construct from an array-valued `astropy.coordinates.SkyCoord`.

        Parameters
        ----------
        vertices : `astropy.coordinates.SkyCoord`
            Array of vertices, CCW when looking out from the origin.
            Implicitly closed (the first vertex should not be duplicated as the
            last).
        clip : `bool`, optional
            If `True` (`False` is default), clip pixel stencils returned by
            `to_pixels` instead of raising `StencilNotContainedError`.

        Returns
        -------
        stencil : `SkyPolygon`
            Polygon stencil.
        """
        return cls((_lonlat_from_astropy(v) for v in vertices), clip=clip)

    def _ast_sky_region(self) -> Ast.Region:
        # Docstring inherited.
        sky_frame = Ast.SkyFrame("System=ICRS")
        ra = [v.getLon().asRadians() for v in self._vertices]
        dec = [v.getLat().asRadians() for v in self._vertices]
        centroid = lsst.sphgeom.LonLat(self.region.getCentroid())
        probe = [centroid.getLon().asRadians(), centroid.getLat().asRadians()]
        polygon = Ast.Polygon(sky_frame, np.array([ra, dec]))
        # AST's bounded interior depends on vertex winding: with the wrong
        # winding the polygon represents its own complement.  ``negate`` flips
        # ``pointinregion`` but not the ``mask`` polarity, so reverse the
        # vertices instead to obtain a region whose interior is the polygon.
        if not polygon.pointinregion(probe):
            polygon = Ast.Polygon(sky_frame, np.array([ra[::-1], dec[::-1]]))
        return polygon

    def _boundary_skycoord(self) -> SkyCoord:
        # Docstring inherited.
        return SkyCoord(
            ra=[v.getLon().asRadians() for v in self._vertices] * u.rad,
            dec=[v.getLat().asRadians() for v in self._vertices] * u.rad,
            frame="icrs",
        )

    @property
    def region(self) -> lsst.sphgeom.Region:
        # Docstring inherited.
        return lsst.sphgeom.ConvexPolygon([UnitVector3d(v) for v in self._vertices])

    def to_fits_metadata(self, metadata: PropertyList | MutableMapping[str, Any]) -> None:
        # Docstring inherited.
        metadata["ST_TYPE"] = "POLYGON"
        if len(self._vertices) > 100:
            raise NotImplementedError(
                "TODO: FITS limitations make it difficult to serialize big stencils to the header."
            )
        for n, v in enumerate(self._vertices):
            metadata[f"ST_RA{n:02d}"] = v.getLon().asDegrees()
            metadata[f"ST_DEC{n:02d}"] = v.getLat().asDegrees()

    @property
    def fingerprint(self) -> bytes:
        # Docstring inherited.
        hasher = blake2b(digest_size=16)
        hasher.update(b"POLYGON")
        for v in self._vertices:
            hasher.update(struct.pack("!dd", v.getLon().asRadians(), v.getLat().asRadians()))
        return hasher.digest()


def _angle_from_astropy(angle: astropy.coordinates.Angle) -> Angle:
    """Convert an `astropy.coordinates.Angle` to a `lsst.sphgeom.Angle`.

    Parameters
    ----------
    angle : `astropy.coordinates.Angle`
        Astropy Angle to convert.  Must be a scalar.

    Returns
    -------
    angle : `lsst.sphgeom.Angle`
        Equivalent sphgeom angle.
    """
    if not angle.isscalar:
        raise ValueError("Only scalar angles are supported.")
    return Angle(angle.to_value(u.rad))


def _lonlat_from_astropy(skycoord: astropy.coordinates.SkyCoord) -> LonLat:
    """Convert an `astropy.coordinates.SkyCoord` to a `lsst.sphgeom.LonLat`.

    Parameters
    ----------
    skycoord : `astropy.coordinates.SkyCoord`
        Astropy coordinates to convert.  Must be a scalar.

    Returns
    -------
    lonlat : `lsst.sphgeom.LonLat`
        Equivalent spherical point.
    """
    if not skycoord.isscalar:
        raise ValueError("Only scalar coordinates are supported.")
    icrs = skycoord.transform_to("icrs")
    return LonLat.fromRadians(float(icrs.ra.rad), float(icrs.dec.rad))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_stencils.py -v`
Expected: PASS for every test. If the AST circle/polygon counts differ slightly, the `max_missing`/`max_extra` thresholds in the test are the same ones used before the rewrite; a genuine failure indicates an x/y transposition (check `_coverage` scratch shape `(ny, nx)`) or mask-plane offset (check `set_mask` `y_off`/`x_off`).

- [ ] **Step 5: Commit**

```bash
git add python/lsst/dax/images/cutout/stencils.py tests/test_stencils.py
git commit -m "Move stencils onto sphgeom inputs and lsst.images.Mask"
```

---

## Task 3: `projection_finders.py` returns modern types

**Files:**
- Modify: `python/lsst/dax/images/cutout/projection_finders.py`

- [ ] **Step 1: Replace the imports**

Replace lines 40-48 (the import block from `import astropy.io.fits` through `from lsst.utils.timer import time_this`) with:

```python
import astropy.io.fits
import astropy.units as u

from lsst.daf.butler import Butler, DatasetRef
from lsst.images import Box, GeneralFrame, SkyProjection
from lsst.skymap import BaseSkyMap
from lsst.utils.timer import time_this

from ._fits_projection import projection_and_bbox_from_fits_header

# Pixel coordinate frame for projections built from afw SkyWcs objects.
_PIXEL_FRAME = GeneralFrame(unit=u.pix)
```

This drops `import lsst.geom`, `from lsst.afw.geom import SkyWcs, getImageXY0FromMetadata, makeSkyWcs`, `from lsst.daf.base import PropertyList`, and `from lsst.geom import Box2I`.

- [ ] **Step 2: Update the return-type annotations**

In every `find_projection` signature and in `__call__`, change the return annotation from `tuple[SkyWcs, Box2I] | None` to `tuple[SkyProjection, Box] | None` (and `__call__`'s `-> tuple[SkyWcs, Box2I]` to `-> tuple[SkyProjection, Box]`). There are six such annotations: `ProjectionFinder.find_projection`, `ProjectionFinder.__call__`, `ReadComponents.find_projection`, `ReadComponentsAstropyFits.find_projection`, `TryComponentParents.find_projection`, `UseSkyMap.find_projection`, `Chain.find_projection`, `MatchDatasetTypeName.find_projection`. Update the `wcs : `SkyWcs`` / `bbox : `Box2I`` docstring lines in the base class to `projection : `lsst.images.SkyProjection`` and `bbox : `lsst.images.Box``.

- [ ] **Step 3: Convert `ReadComponents.find_projection`**

Replace the body's return (currently `return wcs, bbox`) so the method reads:

```python
    def find_projection(
        self, ref: DatasetRef, butler: Butler, logger: logging.Logger | None = None
    ) -> tuple[SkyProjection, Box] | None:
        # Docstring inherited.
        if {"wcs", "bbox"}.issubset(ref.datasetType.storageClass.allComponents().keys()):
            logger = logger if logger is not None else _LOG
            with time_this(_LOG, msg="Read projection info from butler components", level=_TIMER_LOG_LEVEL):
                wcs = butler.get(ref.makeComponentRef("wcs"))
                bbox = butler.get(ref.makeComponentRef("bbox"))
                return SkyProjection.from_legacy(wcs, _PIXEL_FRAME), Box.from_legacy(bbox)
        return None
```

- [ ] **Step 4: Convert `ReadComponentsAstropyFits.find_projection`**

Replace its inner block (the part from `shape = hdu.shape` through `return wcs, bbox`) so the method reads:

```python
    def find_projection(
        self, ref: DatasetRef, butler: Butler, logger: logging.Logger | None = None
    ) -> tuple[SkyProjection, Box] | None:
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
                                return projection_and_bbox_from_fits_header(hdr, hdu.shape)
                except Exception:
                    # Any failure and we will try the next option.
                    pass
        return None
```

- [ ] **Step 5: Convert `UseSkyMap.find_projection`**

Replace the two `return` statements at the end of the method so they convert the afw skymap objects:

```python
            tractInfo = skymap[ref.dataId["tract"]]
            if "patch" in ref.dataId.dimensions:
                patchInfo = tractInfo[ref.dataId["patch"]]
                return (
                    SkyProjection.from_legacy(patchInfo.wcs, _PIXEL_FRAME),
                    Box.from_legacy(patchInfo.outer_bbox),
                )
            else:
                return (
                    SkyProjection.from_legacy(tractInfo.wcs, _PIXEL_FRAME),
                    Box.from_legacy(tractInfo.bbox),
                )
        return None
```

- [ ] **Step 6: Verify the module imports cleanly**

Run: `python -c "import lsst.dax.images.cutout.projection_finders"`
Expected: exit 0, no output.

- [ ] **Step 7: Run ruff to confirm no unused imports remain**

Run: `ruff check python/lsst/dax/images/cutout/projection_finders.py`
Expected: no errors. Fix any unused-import findings (e.g. a leftover `cast`/`re` is still used by `MatchDatasetTypeName`/`UseSkyMap`; do not remove those).

- [ ] **Step 8: Commit**

```bash
git add python/lsst/dax/images/cutout/projection_finders.py
git commit -m "Return SkyProjection and lsst.images.Box from projection finders"
```

---

## Task 4: Unify the astropy branches and modernize masking in `_image_cutout.py`

**Files:**
- Modify: `python/lsst/dax/images/cutout/_image_cutout.py`

- [ ] **Step 1: Update imports**

Replace line 38 (`from lsst.afw.geom import getImageXY0FromMetadata, makeSkyWcs`) and remove line 35 (`import lsst.geom`).  Add the helper and mask types. The relevant import region should read:

```python
import astropy.io.fits
import astropy.time

import lsst.images
import lsst.images.serialization
from lsst.afw.image import Exposure, Image, Mask, MaskedImage, makeExposure, makeMaskedImage
from lsst.daf.base import PropertyList
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
```

(`lsst.geom` is no longer used anywhere in the file after this task; confirm with `grep -n "lsst.geom" python/lsst/dax/images/cutout/_image_cutout.py` returning nothing in Step 7.)

- [ ] **Step 2: Add the shared astropy-HDU reader method**

Add this method to the cutout backend class (the class that defines `_extract_ref_legacy`/`_extract_ref_v2`; place it just above `_extract_ref_legacy`, around line 355). It contains the single copy of the FITS cutout logic that both branches will call:

```python
    def _read_astropy_hdulist(
        self,
        stencil: SkyStencil,
        ref: DatasetRef,
        pixel_components: set[str],
        fsspec_kwargs: dict[str, object],
    ) -> tuple[astropy.io.fits.HDUList, PixelStencil | None, str]:
        """Read the primary header and requested pixel HDUs, cut to the stencil.

        Parameters
        ----------
        stencil : `SkyStencil`
            Sky-coordinate stencil defining the cutout region.
        ref : `DatasetRef`
            Resolved reference to the dataset to read.
        pixel_components : `set` [ `str` ]
            Lower-case EXTNAMEs to extract (e.g. ``{"image"}``).  Consumed in
            place as components are found.
        fsspec_kwargs : `dict`
            Keyword arguments forwarded to ``fsspec`` ``open``.

        Returns
        -------
        hdulist : `astropy.io.fits.HDUList`
            Primary HDU plus one cutout HDU per requested pixel component.
        pixel_stencil : `PixelStencil` or `None`
            Pixel-coordinate stencil computed from the first pixel HDU, or
            `None` if no pixel HDU was found.
        timesys : `str`
            ``TIMESYS`` from the primary header, or ``"UTC"``.
        """
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
                    timesys = hdul[0].header.get("TIMESYS", timesys)
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
                    # Offsets of the cutout within the full HDU, in array order.
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
        return astropy.io.fits.HDUList(hdus=hdul), pixel_stencil, timesys
```

- [ ] **Step 3: Replace the legacy astropy branch**

In `_extract_ref_legacy`, replace the entire `case CutoutMode.ASTROPY_IMAGE | CutoutMode.ASTROPY_MASKED_IMAGE:` block (currently lines ~434-511, ending at `cutout = astropy.io.fits.HDUList(hdus=hdul)`) with:

```python
                case CutoutMode.ASTROPY_IMAGE | CutoutMode.ASTROPY_MASKED_IMAGE:
                    # Bypass butler and read the pixel HDU directly.
                    pixel_components = {"image"} if cutout_mode == CutoutMode.ASTROPY_IMAGE else {
                        "image",
                        "mask",
                        "variance",
                    }
                    cutout, pixel_stencil, timesys = self._read_astropy_hdulist(
                        stencil, ref, pixel_components, {}
                    )
```

- [ ] **Step 4: Replace the v2 astropy branch**

In `_extract_ref_v2`, replace the body of the `else:` astropy branch (currently lines ~617-715, from the `wcs = None` / `bbox = None` initialization through `cutout = astropy.io.fits.HDUList(hdus=hdul)`) with:

```python
        else:
            # This is the Astropy direct branch.
            with time_this(
                self.logger,
                msg="Extract cutout",
                kwargs={"id": str(ref.id), "cutout_mode": str(cutout_mode), "stencil": str(stencil)},
                level=_TIMER_LOG_LEVEL,
            ):
                pixel_components = {"image"} if cutout_mode == CutoutMode.ASTROPY_IMAGE else {
                    "image",
                    "mask",
                    "variance",
                }
                # Tune the fsspec cache to match what we use in lsst.images.
                maxblocks = max(2, READ_CACHE_MAX_BYTES // DEFAULT_PAGE_SIZE)
                fsspec_kwargs = {
                    "block_size": DEFAULT_PAGE_SIZE,
                    "cache_type": _READ_CACHE_TYPE,
                    "cache_options": {"maxblocks": maxblocks},
                }
                cutout, pixel_stencil, timesys = self._read_astropy_hdulist(
                    stencil, ref, pixel_components, fsspec_kwargs
                )
```

- [ ] **Step 5: Update the non-astropy `to_pixels` call sites**

In `_extract_ref_legacy` (line ~388) the finder now returns modern types, so rename the variable for clarity:

```python
            projection, bbox = self.projection_finder(ref, self.butler, logger=self.logger)
            # Transform the stencil to pixel coordinates.
            pixel_stencil = stencil.to_pixels(projection, bbox)
```

`_extract_ref_v2` already calls `stencil.to_pixels(sky_projection, bbox)` with modern types (lines ~593-599); leave it unchanged.

- [ ] **Step 6: Modernize `Extraction.mask`**

Replace the `mask` method (lines ~100-129) with:

```python
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
        if isinstance(self.cutout, lsst.images.MaskedImage):
            mask = self.cutout.mask
            # Adding a new plane to an lsst.images.Mask after construction is
            # not yet supported; set the plane only if it already exists.
            if name in mask.schema.names:
                self.pixel_stencil.set_mask(mask, name)
            return
        if isinstance(self.cutout, Exposure):
            mask = self.cutout.mask
        elif isinstance(self.cutout, MaskedImage):
            mask = self.cutout.mask
        elif isinstance(self.cutout, Mask):
            mask = self.cutout
        else:
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
```

- [ ] **Step 7: Verify imports and run the integration tests**

Run: `grep -n "lsst.geom\|getImageXY0\|makeSkyWcs" python/lsst/dax/images/cutout/_image_cutout.py`
Expected: no matches.

Run: `python -c "import lsst.dax.images.cutout"`
Expected: exit 0.

Run: `pytest tests/test_imageCutoutsBackend.py tests/test_imageCutoutsBackendV2.py -v`
Expected: PASS (or the same skips as on `main` if these require data not present locally — compare against a clean `git stash` run if unsure). If a masking assertion fails, check that the afw OR-in uses the same `(ny, nx)` coverage orientation as before.

- [ ] **Step 8: Commit**

```bash
git add python/lsst/dax/images/cutout/_image_cutout.py
git commit -m "Unify astropy cutout branches and mask via lsst.images.Mask"
```

---

## Task 5: Benchmark update and full verification

**Files:**
- Modify: `tests/bench_stencils.py`

- [ ] **Step 1: Rewrite the benchmark for the new types**

Replace lines 30-62 of `tests/bench_stencils.py` (imports through the end of `_run`) with:

```python
from __future__ import annotations

import timeit

import astropy.units as u

from lsst.afw.geom import makeCdMatrix, makeSkyWcs
from lsst.dax.images.cutout.stencils import MaskBackend, SkyCircle
from lsst.geom import Point2D, SpherePoint, arcseconds, degrees
from lsst.images import Box, GeneralFrame, Mask, MaskPlane, MaskSchema, SkyProjection
from lsst.sphgeom import Angle, LonLat

CENTER = LonLat.fromDegrees(12.0, 13.0)
_WCS = makeSkyWcs(Point2D(0.0, 0.0), SpherePoint(12.0, 13.0, degrees), makeCdMatrix(0.2 * arcseconds))
PROJECTION = SkyProjection.from_legacy(_WCS, GeneralFrame(unit=u.pix))

# (label, circle radius in arcsec, half-size of the square bbox in pixels).
CASES = (
    ("small", 5.0, 64),
    ("medium", 30.0, 256),
    ("large", 120.0, 1024),
)
REPEATS = 20


def _arcsec(value: float) -> Angle:
    """A `lsst.sphgeom.Angle` for ``value`` arcseconds."""
    return Angle((value * u.arcsec).to_value(u.rad))


def _run(radius_arcsec: float, half: int, backend: MaskBackend) -> float:
    stencil = SkyCircle(CENTER, _arcsec(radius_arcsec))
    box = Box.factory[-half : half + 1, -half : half + 1]
    schema = MaskSchema([MaskPlane("STENCIL", "stencil coverage")])

    def once() -> None:
        pixel_stencil = stencil.to_pixels(PROJECTION, box, backend=backend)
        mask = Mask(schema=schema, bbox=box)
        pixel_stencil.set_mask(mask, "STENCIL")

    return min(timeit.repeat(once, number=1, repeat=REPEATS))
```

`main()` and the `__main__` guard (lines 65-76) are unchanged.

- [ ] **Step 2: Run the benchmark**

Run: `python tests/bench_stencils.py`
Expected: a three-row table of best-of-20 timings for AST and SPHGEOM across the small/medium/large sizes.

- [ ] **Step 3: Confirm pytest does not collect the benchmark**

Run: `pytest tests/ --collect-only -q`
Expected: `tests/bench_stencils.py` is not listed.

- [ ] **Step 4: Run the full suite and linters**

Run: `pytest tests/ -v`
Expected: PASS (modulo data-dependent skips identical to `main`).

Run: `ruff check python/lsst/dax/images/cutout/stencils.py python/lsst/dax/images/cutout/_fits_projection.py python/lsst/dax/images/cutout/projection_finders.py python/lsst/dax/images/cutout/_image_cutout.py tests/test_stencils.py tests/test_fits_projection.py tests/bench_stencils.py`
Then: `ruff format --check` on the same files.
Expected: no errors. Fix any reported issues.

Run: `mypy python/lsst/dax/images/cutout/stencils.py python/lsst/dax/images/cutout/_fits_projection.py python/lsst/dax/images/cutout/projection_finders.py`
Expected: no errors. The `lsst.images._transforms._ast` import may need a `# type: ignore[import]` comment if mypy cannot resolve the private module; add it only if reported, with a short note.

- [ ] **Step 5: Confirm afw/geom are gone from the stencils module**

Run: `grep -nE "lsst\.afw|lsst\.geom" python/lsst/dax/images/cutout/stencils.py`
Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add tests/bench_stencils.py
git commit -m "Update stencil benchmark for sphgeom and lsst.images.Mask"
```

---

## Self-review notes

- **Spec coverage:** dependency purge + sphgeom constructors + `set_mask(Mask, plane)` (Task 2); shared FITS helper on the `lsst.images` AST wrapper (Task 1) used by both `ReadComponentsAstropyFits` (Task 3) and the unified astropy branches (Task 4); `find_projection` returns `(SkyProjection, Box)` with `from_legacy` conversions (Task 3); afw-mask seam in `Extraction.mask` (Task 4); v2 mask left as a guarded no-op until `lsst.images.Mask` can add a plane (Task 4, Step 6); tests + bench (Tasks 1, 2, 5).
- **Deferred deliberately:** the v2 `lsst.images.Mask` masking path beyond an existing-plane demonstration; moving the FITS helper into `lsst.images`.
- **XY0 convention:** validated against `getImageXY0FromMetadata` in `tests/test_fits_projection.py` rather than approximated (Task 1, Step 4).
- **Type consistency:** the helper returns `(SkyProjection, Box)` everywhere; `to_pixels(projection: SkyProjection, bbox: Box, *, backend)`; `set_mask(mask: lsst.images.Mask, plane: str)`; `_coverage() -> np.ndarray`; `SkyCircle(LonLat, Angle)`/`SkyPolygon(Iterable[LonLat])`; `Box` accessors `.x.start/.x.stop/.x.min/.shape.x` used consistently in `stencils.py` and `_image_cutout.py`.
