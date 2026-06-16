# Stencils Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `stencils.py` so its internal logic uses only `lsst.images`, starlink-pyast, and `lsst.sphgeom`, while still accepting afw `SkyWcs`/`Box2I`/`Mask` at the public boundary, with two interchangeable masking backends for comparison.

**Architecture:** Sky stencils (`SkyCircle`, `SkyPolygon`) keep their public surface. `to_pixels(wcs, bbox, *, backend=MaskBackend.AST)` normalizes afw inputs to `SkyProjection` and `lsst.images.Box`, then dispatches to one of two `PixelStencil` implementations: an AST backend that remaps a sky `Ast.Region` into the pixel frame and rasterizes with `Region.mask`, and a sphgeom backend that tests pixel centers against the `sphgeom` region. Both produce a boolean array applied to the mask via numpy bitwise-OR.

**Tech Stack:** `lsst.images` (`SkyProjection`, `Box`, `Interval`, `XY`, `GeneralFrame`), `starlink.Ast`, `lsst.sphgeom`, numpy, astropy. afw (`SkyWcs`, `Box2I`, `Mask`) only at the boundary.

---

## Background the engineer needs

- The module under change is `python/lsst/dax/images/cutout/stencils.py`; its tests are `tests/test_stencils.py`. The only other consumer is `python/lsst/dax/images/cutout/_image_cutout.py`.
- Run the tests with the project's normal EUPS-set-up environment (`setup -k -r .` then `pytest`), because the afw and sphgeom C++ extensions need the dynamic-library path that `setup` configures. Running bare `python` from an arbitrary directory will fail to load `libbase.dylib`/`libsphgeom.dylib`.
- Coordinate conventions: `lsst.images.Box.factory[y_slice, x_slice]` is `[y, x]`. `Box.start` and `Box.shape` are `YX(y, x)` named tuples. `Box.meshgrid()` returns `XY(x, y)` where each is a `(ny, nx)` float array. `Box.x`/`Box.y` are `Interval` objects with `.min`, `.max` (inclusive integer extremes), `.start`, `.stop`, `.size`.
- AST masking semantics (verified): `region.mask(map, inside, lbnd, ubnd, array, val)` where `map` maps array-grid coordinates to the region's frame (use `Ast.UnitMap(2)` when the region is already in the pixel frame), `inside=1` sets the pixels whose centers are **inside** the region to `val` and leaves the rest unchanged, `lbnd`/`ubnd` are `[x_min, y_min]`/`[x_max, y_max]`, and `array` is indexed `[y, x]` with shape `(ny, nx)`. The call mutates `array` in place and returns the number of pixels set.
- **Masking safety rule:** `region.mask` overwrites array values; it is not mask-plane aware. Always rasterize into a fresh zero scratch array, convert to boolean, then bitwise-OR the bits into the real mask. Never call `region.mask` on a live mask plane.
- A `starlink.Ast.Region` is also a `Mapping`: `region.applyForward(points)` returns the input coordinates for points inside the region and `nan` for points outside. This is how we check interior membership to fix polygon orientation.

## File structure

- Modify (full rewrite): `python/lsst/dax/images/cutout/stencils.py` — sky/pixel stencil abstractions and the two masking backends.
- Modify: `python/lsst/dax/images/cutout/_image_cutout.py` — adapt two call sites to the `lsst.images.Box` return type of `PixelStencil.bbox`.
- Modify: `tests/test_stencils.py` — parametrize correctness checks over both backends and add a backend-comparison test.
- Create: `tests/bench_stencils.py` — opt-in benchmark (not collected by pytest because the filename does not start with `test_`).

---

## Task 1: Module scaffolding and boundary helpers

**Files:**
- Modify: `python/lsst/dax/images/cutout/stencils.py` (replace the header/imports and add module-level helpers and the `MaskBackend` enum)
- Test: `tests/test_stencils.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stencils.py` (new imports at the top alongside the existing ones, and a new test case):

```python
import astropy.units as u
from lsst.images import Box, SkyProjection
from lsst.dax.images.cutout.stencils import (
    MaskBackend,
    _as_box,
    _as_projection,
    _round_box_from_bounds,
)


class BoundaryHelpersTestCase(unittest.TestCase):
    """Tests for the afw->lsst.images boundary normalization helpers."""

    def setUp(self) -> None:
        self.center = SpherePoint(12.0, 13.0, degrees)
        self.wcs = makeSkyWcs(Point2D(5.0, 7.0), self.center, makeCdMatrix(0.1 * arcseconds))
        self.box2i = Box2I(Point2I(-16, -13), Point2I(26, 27))

    def test_as_projection_passthrough(self) -> None:
        projection = SkyProjection.from_legacy(self.wcs, _PIXEL_FRAME)
        self.assertIs(_as_projection(projection), projection)

    def test_as_projection_from_skywcs(self) -> None:
        projection = _as_projection(self.wcs)
        self.assertIsInstance(projection, SkyProjection)

    def test_as_box_passthrough(self) -> None:
        box = Box.from_legacy(self.box2i)
        self.assertIs(_as_box(box), box)

    def test_as_box_from_box2i(self) -> None:
        box = _as_box(self.box2i)
        self.assertEqual(box, Box.from_legacy(self.box2i))

    def test_round_box_from_bounds(self) -> None:
        # x in [4.6, 9.4], y in [2.6, 5.4] -> integer pixel box [3:6, 5:10] in [y, x].
        box = _round_box_from_bounds(4.6, 9.4, 2.6, 5.4)
        self.assertEqual(box, Box.factory[3:6, 5:10])

    def test_mask_backend_members(self) -> None:
        self.assertEqual({b.name for b in MaskBackend}, {"AST", "SPHGEOM"})
```

Add a module-level constant in the test (after imports):

```python
from lsst.dax.images.cutout.stencils import _PIXEL_FRAME  # noqa: E402
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_stencils.py::BoundaryHelpersTestCase -v`
Expected: FAIL with `ImportError`/`AttributeError` because the names do not exist yet.

- [ ] **Step 3: Replace the top of `stencils.py` with the new imports, enum, and helpers**

Replace everything from the `__all__` declaration through the end of the imports, and add the helpers, so the top of the module reads:

```python
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
from lsst.afw.geom import SkyWcs
from lsst.afw.image import Mask
from lsst.daf.base import PropertyList
from lsst.geom import Angle, Box2I, SpherePoint, radians
from lsst.images import Box, GeneralFrame, NoOverlapError, SkyProjection
from lsst.images.utils import round_half_down, round_half_up

# Generic pixel coordinate frame used when converting a legacy ``SkyWcs`` into a
# ``SkyProjection``.  The frame identity only affects AST domain labels; the
# unit must be pixels so ``SkyProjection`` accepts the transform.
_PIXEL_FRAME = GeneralFrame(unit=u.pix)


class MaskBackend(enum.Enum):
    """Selects the algorithm used to rasterize a stencil onto pixels."""

    AST = enum.auto()
    """Mask using starlink-pyast ``Region.mask`` on the true sky region."""

    SPHGEOM = enum.auto()
    """Mask by testing pixel centers against the ``lsst.sphgeom`` region."""


def _as_projection(wcs: SkyWcs | SkyProjection) -> SkyProjection:
    """Normalize a WCS argument to a `lsst.images.SkyProjection`."""
    if isinstance(wcs, SkyProjection):
        return wcs
    if isinstance(wcs, SkyWcs):
        return SkyProjection.from_legacy(wcs, _PIXEL_FRAME)
    raise TypeError(f"Expected SkyWcs or SkyProjection; got {type(wcs).__name__}.")


def _as_box(bbox: Box2I | Box) -> Box:
    """Normalize a bounding-box argument to a `lsst.images.Box`."""
    if isinstance(bbox, Box):
        return bbox
    if isinstance(bbox, Box2I):
        return Box.from_legacy(bbox)
    raise TypeError(f"Expected Box2I or lsst.images.Box; got {type(bbox).__name__}.")


def _round_box_from_bounds(x_min: float, x_max: float, y_min: float, y_max: float) -> Box:
    """Build an integer pixel `Box` from continuous coordinate bounds.

    Uses the same rounding convention as `lsst.images.Region.bbox`, so that
    pixels whose centers lie within the bounds are included.
    """
    return Box.factory[
        round_half_up(y_min) : round_half_down(y_max) + 1,
        round_half_up(x_min) : round_half_down(x_max) + 1,
    ]
```

Leave the rest of the original module in place for now; later tasks replace the class bodies. The module must still import.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_stencils.py::BoundaryHelpersTestCase -v`
Expected: PASS (all six tests).

- [ ] **Step 5: Commit**

```bash
git add python/lsst/dax/images/cutout/stencils.py tests/test_stencils.py
git commit -m "Add MaskBackend enum and afw boundary helpers to stencils"
```

---

## Task 2: Rewrite the `PixelStencil` abstraction and the two backends

**Files:**
- Modify: `python/lsst/dax/images/cutout/stencils.py` (replace `PixelStencil` and `PixelPolygon`)
- Test: covered indirectly; direct backend tests arrive with Tasks 4 and 5.

- [ ] **Step 1: Replace the `PixelStencil` ABC and delete `PixelPolygon`**

Replace the existing `PixelStencil` ABC and the entire `PixelPolygon` class with:

```python
class PixelStencil(ABC):
    """An image cutout stencil defined in pixel coordinates."""

    @property
    @abstractmethod
    def bbox(self) -> Box:
        """Bounding box of this stencil in pixel coordinates (`lsst.images.Box`)."""
        raise NotImplementedError()

    @abstractmethod
    def set_mask(self, mask: Mask, bits: int) -> None:
        """Set mask planes for the pixels whose centers are covered by this stencil.

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Mask to modify in-place.
        bits : `int`
            Integer bitmask to bitwise-OR into pixels covered by the stencil.
        """
        raise NotImplementedError()


def _apply_boolean_to_mask(mask: Mask, box: Box, covered: np.ndarray, bits: int) -> None:
    """Bitwise-OR ``bits`` into ``mask`` for the `True` entries of ``covered``.

    ``covered`` must be a boolean array with shape ``box.shape`` (``(ny, nx)``).
    """
    submask = mask[box.to_legacy()]
    submask.array[:, :] |= (bits * covered).astype(submask.array.dtype)


class _AstPixelRegion(PixelStencil):
    """Pixel-coordinate stencil backed by a starlink-pyast `Region`.

    Parameters
    ----------
    pixel_region : `starlink.Ast.Region`
        The stencil region expressed in the pixel coordinate frame.
    bbox : `lsst.images.Box`
        Bounding box the stencil is restricted to.
    """

    def __init__(self, pixel_region: Ast.Region, bbox: Box) -> None:
        self._pixel_region = pixel_region
        self._bbox = bbox

    @property
    def bbox(self) -> Box:
        # Docstring inherited.
        return self._bbox

    def set_mask(self, mask: Mask, bits: int) -> None:
        # Docstring inherited.
        scratch = np.zeros(self._bbox.shape, dtype=np.int64)
        self._pixel_region.mask(
            Ast.UnitMap(2),
            1,
            [self._bbox.x.min, self._bbox.y.min],
            [self._bbox.x.max, self._bbox.y.max],
            scratch,
            1,
        )
        _apply_boolean_to_mask(mask, self._bbox, scratch != 0, bits)


class _SphgeomPixelRegion(PixelStencil):
    """Pixel-coordinate stencil that tests pixel centers against a sphgeom region.

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

    def set_mask(self, mask: Mask, bits: int) -> None:
        # Docstring inherited.
        grid = self._bbox.meshgrid()
        sky = self._projection.pixel_to_sky(x=grid.x.ravel(), y=grid.y.ravel())
        covered = self._region.contains(sky.ra.radian, sky.dec.radian).reshape(self._bbox.shape)
        _apply_boolean_to_mask(mask, self._bbox, covered, bits)
```

- [ ] **Step 2: Verify the module still imports**

Run: `python -c "import lsst.dax.images.cutout.stencils"`
Expected: no output, exit 0. (The `SkyStencil`/`SkyCircle`/`SkyPolygon` bodies are still the originals at this point; they are replaced in Task 3.)

- [ ] **Step 3: Commit**

```bash
git add python/lsst/dax/images/cutout/stencils.py
git commit -m "Replace PixelStencil with AST and sphgeom backed pixel regions"
```

---

## Task 3: Rewrite the sky stencils with `to_pixels` dispatch and sky regions

**Files:**
- Modify: `python/lsst/dax/images/cutout/stencils.py` (replace `SkyStencil`, `SkyCircle`, `SkyPolygon`; remove `_make_local_gnomonic_wcs`)
- Test: `tests/test_stencils.py` (existing `test_from_astropy`, `test_repr` continue to pass; add an interior-region test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stencils.py` inside `SkyCircleTestCase` (and a new polygon test case):

```python
    def test_ast_sky_region_circle_contains_center(self) -> None:
        """The AST sky region must contain its own center."""
        region = self.instance._ast_sky_region()
        probe = region.applyForward(
            np.array([[self.center.getRa().asRadians()], [self.center.getDec().asRadians()]])
        )
        self.assertTrue(np.all(np.isfinite(probe)))


class SkyPolygonTestCase(unittest.TestCase):
    """Tests for `SkyPolygon` orientation handling."""

    def setUp(self) -> None:
        center = SpherePoint(12.0, 13.0, degrees)
        self.instance = SkyCircle(center, Angle(2.0, arcseconds)).to_polygon(n_vertices=8)
        self.center = center

    def test_ast_sky_region_polygon_contains_centroid(self) -> None:
        """The AST polygon must select the bounded interior, not its complement."""
        region = self.instance._ast_sky_region()
        centroid = self.instance.region.getCentroid()
        lonlat = lsst.sphgeom.LonLat(centroid)
        probe = region.applyForward(
            np.array([[lonlat.getLon().asRadians()], [lonlat.getLat().asRadians()]])
        )
        self.assertTrue(np.all(np.isfinite(probe)))
```

Add `import lsst.sphgeom` to the test imports if not already present (it is).

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_stencils.py::SkyCircleTestCase::test_ast_sky_region_circle_contains_center tests/test_stencils.py::SkyPolygonTestCase -v`
Expected: FAIL with `AttributeError` (`_ast_sky_region` does not exist).

- [ ] **Step 3: Replace the `SkyStencil` ABC**

Replace the entire `SkyStencil` ABC with:

```python
class SkyStencil(ABC):
    """An image cutout stencil defined in sky (ICRS) coordinates."""

    _clip: bool

    def to_pixels(
        self,
        wcs: SkyWcs | SkyProjection,
        bbox: Box2I | Box,
        *,
        backend: MaskBackend = MaskBackend.AST,
    ) -> PixelStencil:
        """Transform to a pixel-coordinate stencil.

        Parameters
        ----------
        wcs : `lsst.afw.geom.SkyWcs` or `lsst.images.SkyProjection`
            Mapping from sky coordinates to pixel coordinates.  A legacy
            ``SkyWcs`` is converted to a `SkyProjection` internally.
        bbox : `lsst.geom.Box2I` or `lsst.images.Box`
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
        projection = _as_projection(wcs)
        box = _as_box(bbox)
        if backend is MaskBackend.AST:
            sky_region = self._ast_sky_region()
            sky_to_pixel = projection.sky_to_pixel_transform._ast_mapping
            # TODO: replace this private attribute with a public lsst.images
            # accessor if the AST backend is selected after benchmarking.
            pixel_region = sky_region.mapregion(sky_to_pixel, Ast.Frame(2))
            lbnd, ubnd = pixel_region.getregionbounds()
            tight = _round_box_from_bounds(lbnd[0], ubnd[0], lbnd[1], ubnd[1])
            return _AstPixelRegion(pixel_region, self._resolve_box(tight, box))
        if backend is MaskBackend.SPHGEOM:
            boundary = self._boundary_skycoord()
            xy = projection.sky_to_pixel(boundary)
            tight = _round_box_from_bounds(
                float(np.min(xy.x)), float(np.max(xy.x)), float(np.min(xy.y)), float(np.max(xy.y))
            )
            return _SphgeomPixelRegion(self.region, projection, self._resolve_box(tight, box))
        raise ValueError(f"Unknown mask backend: {backend!r}.")

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
            raise StencilNotContainedError(
                f"{self} has pixel bbox {tight}, which is not within {box}."
            )
        return tight

    @abstractmethod
    def _ast_sky_region(self) -> Ast.Region:
        """Return a starlink-pyast `Region` for this stencil in an ICRS sky frame."""
        raise NotImplementedError()

    @abstractmethod
    def _boundary_skycoord(self) -> SkyCoord:
        """Return sky coordinates sampling the stencil boundary.

        Used by the sphgeom backend to size the pixel bounding box.
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
```

- [ ] **Step 4: Replace `SkyCircle`**

Replace the entire `SkyCircle` class with the version below (drops the `print`, drops the polygon round-trip in `to_pixels`, adds `_ast_sky_region` and `_boundary_skycoord`):

```python
class SkyCircle(SkyStencil):
    """A sky-coordinate circular stencil.

    Parameters
    ----------
    center : `SpherePoint`
        The center of the circle, in ICRS (ra, dec).
    radius : `Angle`
        Radius of the circle.
    clip : `bool`, optional
        If `True` (`False` is default), clip pixel stencils returned by
        `to_pixels` instead of raising `StencilNotContainedError`.
    """

    #: Number of points used to sample the circle boundary when sizing the
    #: pixel bounding box for the sphgeom backend.
    BOUNDARY_SAMPLES = 64

    def __init__(self, center: SpherePoint, radius: Angle, clip: bool = False):
        self._center = center
        self._radius = radius
        self._clip = clip

    def __repr__(self) -> str:
        return f"SkyCircle({self._center!r}, {self._radius!r}, clip={self._clip!r})"

    @classmethod
    def from_astropy(
        cls, center: astropy.coordinates.SkyCoord, radius: astropy.coordinates.Angle, clip: bool = False
    ) -> SkyCircle:
        """Construct from `astropy.coordinates` arguments."""
        return cls(
            center=_spherepoint_from_astropy(center),
            radius=_angle_from_astropy(radius),
            clip=clip,
        )

    @classmethod
    def from_sphgeom(cls, circle: lsst.sphgeom.Circle, clip: bool = False) -> SkyCircle:
        """Construct from a `lsst.sphgeom.Circle` instance."""
        return cls(SpherePoint(circle.getCenter()), Angle(circle.getOpeningAngle()), clip=clip)

    def to_polygon(self, n_vertices: int = 16) -> SkyPolygon:
        """Return a polygon sky stencil that approximates this circle.

        Notes
        -----
        This helper is retained for callers that want a polygon approximation;
        it is no longer used by `to_pixels`, which masks the true circle.
        """
        factor = (2 * np.pi / n_vertices) * radians
        return SkyPolygon(
            (self._center.offset(b * factor, self._radius) for b in range(n_vertices)), clip=self._clip
        )

    def _ast_sky_region(self) -> Ast.Region:
        # Docstring inherited.
        return Ast.Circle(
            Ast.SkyFrame("System=ICRS"),
            1,
            [self._center.getRa().asRadians(), self._center.getDec().asRadians()],
            [self._radius.asRadians()],
        )

    def _boundary_skycoord(self) -> SkyCoord:
        # Docstring inherited.
        factor = (2 * np.pi / self.BOUNDARY_SAMPLES) * radians
        points = [self._center.offset(b * factor, self._radius) for b in range(self.BOUNDARY_SAMPLES)]
        return SkyCoord(
            ra=[p.getRa().asRadians() for p in points] * u.rad,
            dec=[p.getDec().asRadians() for p in points] * u.rad,
            frame="icrs",
        )

    @property
    def region(self) -> lsst.sphgeom.Region:
        # Docstring inherited.
        return lsst.sphgeom.Circle(self._center.getVector(), self._radius)

    def to_fits_metadata(self, metadata: PropertyList | MutableMapping[str, Any]) -> None:
        # Docstring inherited.
        metadata["ST_TYPE"] = "CIRCLE"
        metadata["ST_RA"] = self._center.getRa().asDegrees()
        metadata["ST_DEC"] = self._center.getDec().asDegrees()
        metadata["ST_RAD"] = self._radius.asDegrees()

    @property
    def fingerprint(self) -> bytes:
        # Docstring inherited.
        hasher = blake2b(digest_size=16)
        hasher.update(b"CIRCLE")
        hasher.update(struct.pack("!d", self._center.getRa().asRadians()))
        hasher.update(struct.pack("!d", self._center.getDec().asRadians()))
        hasher.update(struct.pack("!d", self._radius.asRadians()))
        return hasher.digest()
```

- [ ] **Step 5: Replace `SkyPolygon`**

Replace the entire `SkyPolygon` class with the version below (adds `_ast_sky_region` with orientation correction and `_boundary_skycoord`; fixes the `fingerprint` bug):

```python
class SkyPolygon(SkyStencil):
    """A sky-coordinate stencil in the shape of a great-circle polygon.

    Parameters
    ----------
    vertices : `Iterable` [ `SpherePoint` ]
        Vertices of the polygon, CCW when looking out from the origin.
        Implicitly closed (the first vertex should not be duplicated as the
        last).
    clip : `bool`, optional
        If `True` (`False` is default), clip pixel stencils returned by
        `to_pixels` instead of raising `StencilNotContainedError`.
    """

    def __init__(self, vertices: Iterable[SpherePoint], clip: bool = False):
        self._vertices = tuple(vertices)
        self._clip = clip

    @classmethod
    def from_astropy(cls, vertices: astropy.coordinates.SkyCoord, clip: bool = False) -> SkyPolygon:
        """Construct from `astropy.coordinates` arguments."""
        return cls((_spherepoint_from_astropy(v) for v in vertices), clip=clip)

    def _ast_sky_region(self) -> Ast.Region:
        # Docstring inherited.
        points = np.array(
            [
                [v.getRa().asRadians() for v in self._vertices],
                [v.getDec().asRadians() for v in self._vertices],
            ]
        )
        polygon = Ast.Polygon(Ast.SkyFrame("System=ICRS"), points)
        # AST's inside/outside choice depends on vertex traversal order; ensure
        # the bounded interior (which contains the centroid) is selected.
        centroid = lsst.sphgeom.LonLat(self.region.getCentroid())
        probe = polygon.applyForward(
            np.array([[centroid.getLon().asRadians()], [centroid.getLat().asRadians()]])
        )
        if not np.all(np.isfinite(probe)):
            polygon.negate()
        return polygon

    def _boundary_skycoord(self) -> SkyCoord:
        # Docstring inherited.
        return SkyCoord(
            ra=[v.getRa().asRadians() for v in self._vertices] * u.rad,
            dec=[v.getDec().asRadians() for v in self._vertices] * u.rad,
            frame="icrs",
        )

    @property
    def region(self) -> lsst.sphgeom.Region:
        # Docstring inherited.
        return lsst.sphgeom.ConvexPolygon([v.getVector() for v in self._vertices])

    def to_fits_metadata(self, metadata: PropertyList | MutableMapping[str, Any]) -> None:
        # Docstring inherited.
        metadata["ST_TYPE"] = "POLYGON"
        if len(self._vertices) > 100:
            raise NotImplementedError(
                "TODO: FITS limitations make it difficult to serialize big stencils to the header."
            )
        for n, v in enumerate(self._vertices):
            metadata[f"ST_RA{n:02d}"] = v.getRa().asDegrees()
            metadata[f"ST_DEC{n:02d}"] = v.getDec().asDegrees()

    @property
    def fingerprint(self) -> bytes:
        # Docstring inherited.
        hasher = blake2b(digest_size=16)
        hasher.update(b"POLYGON")
        for v in self._vertices:
            hasher.update(struct.pack("!dd", v.getRa().asRadians(), v.getDec().asRadians()))
        return hasher.digest()
```

- [ ] **Step 6: Delete the obsolete `_make_local_gnomonic_wcs` helper**

Remove the `_make_local_gnomonic_wcs` function at the bottom of the module entirely. Keep `_angle_from_astropy` and `_spherepoint_from_astropy` unchanged.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest tests/test_stencils.py::SkyCircleTestCase::test_from_astropy tests/test_stencils.py::SkyCircleTestCase::test_repr tests/test_stencils.py::SkyCircleTestCase::test_ast_sky_region_circle_contains_center tests/test_stencils.py::SkyPolygonTestCase -v`
Expected: PASS. (`test_to_pixel`/`test_to_polygon` are updated in Task 4.)

- [ ] **Step 8: Commit**

```bash
git add python/lsst/dax/images/cutout/stencils.py tests/test_stencils.py
git commit -m "Rewrite sky stencils with AST/sphgeom backends and bug fixes"
```

---

## Task 4: Validate the AST backend against the brute-force reference

**Files:**
- Modify: `tests/test_stencils.py` (parametrize `_check_to_pixel` over `backend`; point existing tests at `MaskBackend.AST`)

- [ ] **Step 1: Update `_check_to_pixel` to accept a backend and write the failing tests**

Change the helper signature and the `to_pixels` call in `tests/test_stencils.py`:

```python
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
    pixel_stencil = sky_stencil.to_pixels(wcs, bbox, backend=backend)
    test_case.assertTrue(bbox.contains(pixel_stencil.bbox.to_legacy()))
    mask = Mask(bbox)
    mask.addMaskPlane("STENCIL")
    bits = mask.getPlaneBitMask("STENCIL")
    pixel_stencil.set_mask(mask, bits)
    check_array = _brute_force_stencil_array(sky_stencil, wcs, bbox)
    missing = np.logical_and(check_array, np.logical_not(mask.array & bits))
    extra = np.logical_and(mask.array & bits, np.logical_not(check_array))
    if plot:
        from matplotlib import pyplot

        display_array = np.zeros((bbox.getHeight(), bbox.getWidth(), 3), dtype=np.uint8)
        display_array[:, :, 0] = 255 * check_array
        display_array[:, :, 1] = 255 * (mask.array & bits).astype(bool)
        pyplot.imshow(display_array, origin="lower", interpolation="nearest")
        pyplot.title("red=check, green=SkyStencil.to_pixel, yellow=both")
        pyplot.show()
    test_case.assertLessEqual(sum(missing.flatten()), max_missing)
    test_case.assertLessEqual(sum(extra.flatten()), max_extra)
```

Note the one behavioral change in the helper: `bbox.contains(pixel_stencil.bbox.to_legacy())` because `PixelStencil.bbox` is now an `lsst.images.Box`.

Update the two existing tests to run the AST backend explicitly:

```python
    def test_to_pixel(self) -> None:
        wcs = makeSkyWcs(Point2D(5.0, 7.0), self.center, makeCdMatrix(0.1 * arcseconds))
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        _check_to_pixel(self, self.instance, wcs, bbox, backend=MaskBackend.AST, max_missing=12)

    def test_to_polygon(self) -> None:
        wcs = makeSkyWcs(Point2D(5.0, 7.0), self.center, makeCdMatrix(0.1 * arcseconds))
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        polygon_stencil = self.instance.to_polygon()
        self.assertNotEqual(
            self.instance.region.relate(polygon_stencil.region.getBoundingCircle()), lsst.sphgeom.DISJOINT
        )
        _check_to_pixel(self, polygon_stencil, wcs, bbox, backend=MaskBackend.AST, max_missing=4)
```

- [ ] **Step 2: Run the tests to verify behavior**

Run: `pytest tests/test_stencils.py::SkyCircleTestCase::test_to_pixel tests/test_stencils.py::SkyCircleTestCase::test_to_polygon -v`
Expected: PASS. If the AST circle masks more accurately than the old polygon approximation, the `max_missing` tolerances may now be loose; that is acceptable. If a test FAILS because of mask polarity or x/y transposition, fix it in `stencils.py`: confirm `inside=1`, and confirm the scratch array shape is `box.shape` (`(ny, nx)`); for a polygon orientation failure, confirm the `negate()` branch in `_ast_sky_region` ran (the `test_ast_sky_region_polygon_contains_centroid` test from Task 3 guards this).

- [ ] **Step 3: Commit**

```bash
git add tests/test_stencils.py
git commit -m "Validate AST stencil backend against brute-force reference"
```

---

## Task 5: Validate the sphgeom backend against the brute-force reference

**Files:**
- Modify: `tests/test_stencils.py` (add sphgeom-backend tests)

- [ ] **Step 1: Write the failing tests**

Add to `SkyCircleTestCase`:

```python
    def test_to_pixel_sphgeom(self) -> None:
        """The sphgeom backend should reproduce the brute-force reference."""
        wcs = makeSkyWcs(Point2D(5.0, 7.0), self.center, makeCdMatrix(0.1 * arcseconds))
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        _check_to_pixel(self, self.instance, wcs, bbox, backend=MaskBackend.SPHGEOM, max_missing=0)

    def test_to_pixel_sphgeom_polygon(self) -> None:
        wcs = makeSkyWcs(Point2D(5.0, 7.0), self.center, makeCdMatrix(0.1 * arcseconds))
        bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))
        polygon_stencil = self.instance.to_polygon()
        _check_to_pixel(self, polygon_stencil, wcs, bbox, backend=MaskBackend.SPHGEOM, max_missing=4)
```

- [ ] **Step 2: Run the tests to verify behavior**

Run: `pytest tests/test_stencils.py -k sphgeom -v`
Expected: PASS. The sphgeom backend tests pixel centers against the same sphgeom region the brute force uses, so `max_missing`/`max_extra` should be 0 for the circle. If `test_to_pixel_sphgeom` fails with off-by-one mask placement, confirm `meshgrid` is consumed as `XY(x, y)` and that `contains(ra, dec)` receives raveled arrays reshaped to `box.shape`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_stencils.py
git commit -m "Validate sphgeom stencil backend against brute-force reference"
```

---

## Task 6: Backend comparison test

**Files:**
- Modify: `tests/test_stencils.py` (add a test that asserts both backends agree)

- [ ] **Step 1: Write the comparison test**

Add a new test case:

```python
class BackendComparisonTestCase(unittest.TestCase):
    """Assert the AST and sphgeom backends agree on bbox and masked pixels."""

    def setUp(self) -> None:
        self.center = SpherePoint(12.0, 13.0, degrees)
        self.wcs = makeSkyWcs(Point2D(5.0, 7.0), self.center, makeCdMatrix(0.1 * arcseconds))
        self.bbox = Box2I(Point2I(-16, -13), Point2I(26, 27))

    def _masked_array(self, stencil: SkyStencil, backend: MaskBackend) -> tuple[np.ndarray, Box]:
        pixel_stencil = stencil.to_pixels(self.wcs, self.bbox, backend=backend)
        mask = Mask(self.bbox)
        mask.addMaskPlane("STENCIL")
        bits = mask.getPlaneBitMask("STENCIL")
        pixel_stencil.set_mask(mask, bits)
        return (mask.array & bits).astype(bool), pixel_stencil.bbox

    def test_backends_agree_circle(self) -> None:
        ast_mask, ast_box = self._masked_array(SkyCircle(self.center, Angle(1.0, arcseconds)), MaskBackend.AST)
        sph_mask, sph_box = self._masked_array(
            SkyCircle(self.center, Angle(1.0, arcseconds)), MaskBackend.SPHGEOM
        )
        self.assertEqual(ast_box, sph_box)
        # Allow a small number of edge-pixel disagreements between the two
        # rasterizers; tighten or document the threshold after benchmarking.
        self.assertLessEqual(int(np.sum(ast_mask != sph_mask)), 2)

    def test_backends_agree_polygon(self) -> None:
        polygon = SkyCircle(self.center, Angle(1.0, arcseconds)).to_polygon()
        ast_mask, ast_box = self._masked_array(polygon, MaskBackend.AST)
        sph_mask, sph_box = self._masked_array(polygon, MaskBackend.SPHGEOM)
        self.assertEqual(ast_box, sph_box)
        self.assertLessEqual(int(np.sum(ast_mask != sph_mask)), 2)
```

- [ ] **Step 2: Run the comparison test**

Run: `pytest tests/test_stencils.py::BackendComparisonTestCase -v`
Expected: PASS. If the bounding boxes differ by a pixel, this is real comparison data: record it and decide during review whether to accept it or unify the bbox computation. Adjust the assertion to document the observed difference rather than masking it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_stencils.py
git commit -m "Add AST vs sphgeom backend comparison test"
```

---

## Task 7: Adapt `_image_cutout.py` to the `lsst.images.Box` return type

**Files:**
- Modify: `python/lsst/dax/images/cutout/_image_cutout.py:403`, `:406`, `:411-412`, `:419-426` (legacy path) and `:599` (v2 path)

- [ ] **Step 1: Update the legacy extraction path to pass a legacy box to butler**

In `_extract_ref_legacy`, every `parameters={"bbox": pixel_stencil.bbox}` must become `parameters={"bbox": pixel_stencil.bbox.to_legacy()}`. There are several occurrences in the `match cutout_mode` block (FULL_EXPOSURE, STRIPPED_EXPOSURE, IMAGE_ONLY, MASKED_IMAGE). Change each one. For example:

```python
                case CutoutMode.FULL_EXPOSURE:
                    cutout = self.butler.get(ref, parameters={"bbox": pixel_stencil.bbox.to_legacy()})
                    timesys = cutout.metadata.get("TIMESYS", timesys)
```

Apply the same `.to_legacy()` change to the `STRIPPED_EXPOSURE`, `IMAGE_ONLY`, and `MASKED_IMAGE` cases.

- [ ] **Step 2: Update the astropy-branch bbox arithmetic in the legacy path**

In `_extract_ref_legacy`, the astropy branch computes `minX`/`maxX`/`minY`/`maxY` from `bbox.getBeginX()` etc., where `bbox` came from `pixel_stencil.bbox`. Since `pixel_stencil.bbox` is now an `lsst.images.Box`, set `bbox` from its legacy form right after the stencil is computed:

```python
                                if pixel_stencil is None:
                                    # Use FITS WCS.
                                    wcs = makeSkyWcs(pl)
                                    pixel_stencil = stencil.to_pixels(wcs, full_bbox)
                                    bbox = pixel_stencil.bbox.to_legacy()
```

The same edit applies in `_extract_ref_v2`'s astropy branch (the block that does `pixel_stencil = stencil.to_pixels(wcs, full_bbox)` then `bbox = pixel_stencil.bbox`).

- [ ] **Step 3: Simplify the v2 native path**

In `_extract_ref_v2`, the native branch currently does:

```python
                    pixel_stencil = stencil.to_pixels(sky_projection, bbox)
                    modern_bbox = lsst.images.Box.from_legacy(pixel_stencil.bbox)
```

Replace those two lines with:

```python
                    pixel_stencil = stencil.to_pixels(sky_projection, bbox)
                    modern_bbox = pixel_stencil.bbox
```

- [ ] **Step 4: Verify the package imports and the stencil tests still pass**

Run: `python -c "import lsst.dax.images.cutout"`
Expected: exit 0.
Run: `pytest tests/test_stencils.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/dax/images/cutout/_image_cutout.py
git commit -m "Adapt image cutout to lsst.images.Box stencil bbox"
```

---

## Task 8: Full verification and lint/type cleanliness

**Files:**
- No new code; runs the full suite and the linters.

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/test_stencils.py -v`
Expected: PASS for every test (boundary helpers, sky regions, AST backend, sphgeom backend, comparison).

- [ ] **Step 2: Run ruff**

Run: `ruff check python/lsst/dax/images/cutout/stencils.py python/lsst/dax/images/cutout/_image_cutout.py tests/test_stencils.py`
Then: `ruff format --check python/lsst/dax/images/cutout/stencils.py tests/test_stencils.py`
Expected: no errors. Fix any reported issues (e.g., unused imports such as `Box2D`/`Point2D` that were removed, or import ordering).

- [ ] **Step 3: Run mypy**

Run: `mypy python/lsst/dax/images/cutout/stencils.py`
Expected: no errors. The private `projection.sky_to_pixel_transform._ast_mapping` access may need a `# type: ignore[attr-defined]` comment with a short explanation; add it only if mypy reports it.

- [ ] **Step 4: Commit any lint/type fixes**

```bash
git add -A
git commit -m "Satisfy ruff and mypy for modernized stencils"
```

---

## Task 9: Opt-in benchmark for the two backends

**Files:**
- Create: `tests/bench_stencils.py`

- [ ] **Step 1: Write the benchmark script**

Create `tests/bench_stencils.py`:

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

"""Opt-in benchmark comparing the AST and sphgeom stencil masking backends.

This module is intentionally not named ``test_*`` so that pytest does not
collect it.  Run it directly::

    python tests/bench_stencils.py
"""

from __future__ import annotations

import timeit

from lsst.afw.geom import makeCdMatrix, makeSkyWcs
from lsst.afw.image import Mask
from lsst.dax.images.cutout.stencils import MaskBackend, SkyCircle
from lsst.geom import Angle, Box2I, Point2D, Point2I, SpherePoint, arcseconds, degrees

CENTER = SpherePoint(12.0, 13.0, degrees)
WCS = makeSkyWcs(Point2D(0.0, 0.0), CENTER, makeCdMatrix(0.2 * arcseconds))

# (label, circle radius in arcsec, half-size of the square bbox in pixels).
CASES = (
    ("small", 5.0, 64),
    ("medium", 30.0, 256),
    ("large", 120.0, 1024),
)
REPEATS = 20


def _run(radius_arcsec: float, half: int, backend: MaskBackend) -> float:
    stencil = SkyCircle(CENTER, Angle(radius_arcsec, arcseconds))
    bbox = Box2I(Point2I(-half, -half), Point2I(half, half))

    def once() -> None:
        pixel_stencil = stencil.to_pixels(WCS, bbox, backend=backend)
        mask = Mask(bbox)
        mask.addMaskPlane("STENCIL")
        bits = mask.getPlaneBitMask("STENCIL")
        pixel_stencil.set_mask(mask, bits)

    return min(timeit.repeat(once, number=1, repeat=REPEATS))


def main() -> None:
    print(f"{'case':>8} {'bbox':>12} {'AST (ms)':>12} {'SPHGEOM (ms)':>14}")
    for label, radius_arcsec, half in CASES:
        side = 2 * half + 1
        ast_ms = _run(radius_arcsec, half, MaskBackend.AST) * 1e3
        sph_ms = _run(radius_arcsec, half, MaskBackend.SPHGEOM) * 1e3
        print(f"{label:>8} {f'{side}x{side}':>12} {ast_ms:>12.3f} {sph_ms:>14.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark to confirm it executes and prints a table**

Run: `python tests/bench_stencils.py`
Expected: a three-row table of best-of-20 timings for each backend across the small/medium/large cutout sizes. Record the output in the review notes; it is the basis for choosing a backend.

- [ ] **Step 3: Confirm pytest does not collect the benchmark**

Run: `pytest tests/ --collect-only -q`
Expected: `tests/bench_stencils.py` does not appear in the collected items.

- [ ] **Step 4: Commit**

```bash
git add tests/bench_stencils.py
git commit -m "Add opt-in benchmark for stencil masking backends"
```

---

## Self-review notes

- **Spec coverage:** boundary policy (Task 1), sky classes preserved + `_ast_sky_region` (Task 3), AST backend (Tasks 2-4), sphgeom backend (Tasks 2, 5), bbox type change + `_image_cutout.py` edits (Task 7), comparison (Task 6), benchmark (Task 9), bug fixes — `print` removed and `to_polygon` off the masking path (Task 3, Step 4), `fingerprint` fixed (Task 3, Step 5), `_make_local_gnomonic_wcs` removed (Task 3, Step 6), lint/type (Task 8).
- **Open items deliberately deferred:** the modern `lsst.images.Mask` masking path (separate ticket; needs new plane-assignment API); a public `lsst.images` accessor for the AST mapping (only if the AST backend is chosen).
- **Design choice:** the `SkyCircle`/`SkyPolygon` constructors keep their existing `lsst.geom` `SpherePoint`/`Angle` inputs; `lsst.geom` is retained for those types and the `Box2I` boundary, since changing the public constructor signature is out of scope for this rewrite.
