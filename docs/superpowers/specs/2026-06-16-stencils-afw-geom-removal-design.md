# Removing `lsst.afw` and `lsst.geom` from the stencils package

- Date: 2026-06-16
- Ticket: DM-55210
- Status: Design pending written-spec review.

## Background

The first phase of DM-55210 rewrote `stencils.py` onto `lsst.images`, starlink-pyast, and `lsst.sphgeom`, with two interchangeable masking backends (`AST` and `SPHGEOM`).
That phase deliberately kept `lsst.afw` and `lsst.geom` at the public boundary: `to_pixels` accepted an afw `SkyWcs` or `lsst.geom.Box2I`, `set_mask` targeted an afw `Mask`, and `SkyCircle`/`SkyPolygon` constructors took `lsst.geom` `SpherePoint`/`Angle`.
Two reasons drove those deferrals: changing the constructor signatures looked like a public-API break, and the modern `lsst.images.Mask` masking path depended on `lsst.images` functionality that did not yet exist.

Both reasons have since lapsed.
The stencils module is private to this package (its only consumer is `_image_cutout.py`, with projection discovery in `projection_finders.py`), so the constructor signatures are free to change.
`lsst.images.Mask` now provides `Mask.set(plane, boolean_mask)`, which assigns a named mask plane from a boolean coverage array — exactly what both backends already compute.

This phase removes `lsst.afw` and `lsst.geom` from the stencils package entirely.
`stencils.py` becomes afw/geom-free; the afw types that legacy data still produces are converted to `lsst.images` types in the consumer, and the FITS-WCS parsing that currently relies on afw is rewritten onto starlink-pyast and `lsst.images.SkyProjection`.

## Goals

- `stencils.py` imports none of `lsst.afw.*` or `lsst.geom`.
- `SkyCircle`/`SkyPolygon` constructors take `lsst.sphgeom` `LonLat` and `Angle`.
- `set_mask` targets an `lsst.images.Mask` and is keyed by mask-plane name.
- `projection_finders.find_projection` returns `(lsst.images.SkyProjection, lsst.images.Box)`.
- FITS-WCS parsing uses `starlink.Ast.FitsChan` and `SkyProjection`, with no `makeSkyWcs`, `getImageXY0FromMetadata`, or `PropertyList`; the two duplicated astropy branches in `_image_cutout.py` are unified.
- Keep both masking backends (`AST` default and `SPHGEOM`).

## Non-goals

- Removing `lsst.afw`/`lsst.geom` from the wider package.
  `lsst.skymap` returns afw `SkyWcs`, and the legacy cutout path reads afw `Exposure` datasets and writes afw mask planes; those afw objects are converted to `lsst.images` types at well-defined seams but are not themselves eliminated.
- Adding the ability to insert a new plane into an `lsst.images.Mask` after construction.
  That functionality is still missing from `lsst.images`; this work assumes the mask handed to `set_mask` already carries a usable plane and leaves the full v2 mask wiring as a follow-up.
- Changing the masking algorithms or the backend-selection API beyond the type changes described here.

## Boundary policy

afw types are no longer accepted anywhere in the stencils package.
They are converted to `lsst.images` types at three seams, all outside `stencils.py`.

| afw object | Origin | Converted to | Where |
| --- | --- | --- | --- |
| `SkyWcs` (butler `wcs` component) | `ReadComponents` | `SkyProjection.from_legacy(wcs, GeneralFrame(unit=u.pix))` | `projection_finders.py` |
| `SkyWcs` (skymap) | `UseSkyMap` | `SkyProjection.from_legacy(...)` | `projection_finders.py` |
| `Box2I` (butler/skymap) | `ReadComponents`, `UseSkyMap` | `Box.from_legacy(bbox)` | `projection_finders.py` |
| FITS header WCS | `ReadComponentsAstropyFits`, astropy cutout branches | `SkyProjection` via the `lsst.images` AST wrapper (see below) | shared FITS helper |
| afw `Mask` plane | legacy `Extraction.mask` | empty `lsst.images.Mask`, result copied back | `_image_cutout.py` |

## `stencils.py` changes

### Dependencies

Retained: `numpy`, `astropy.coordinates`/`astropy.units`, `starlink.Ast`, `lsst.sphgeom`, `lsst.images` (`SkyProjection`, `Box`, `Mask`, `MaskSchema`, `MaskPlane`, `NoOverlapError`), `lsst.images.utils`.
`lsst.daf.base.PropertyList` is retained only as a type in the `to_fits_metadata` signature union (the legacy path passes a `PropertyList`); it is neither afw nor geom.
Removed: `from lsst.afw.geom import SkyWcs`, `from lsst.afw.image import Mask`, and all of `from lsst.geom import Angle, Box2I, SpherePoint, radians`.
The `_as_projection`, `_as_box`, and `_PIXEL_FRAME` helpers are removed from the module (their afw-normalization role moves to the consumer).

### Constructors

- `SkyCircle(center: lsst.sphgeom.LonLat, radius: lsst.sphgeom.Angle, clip: bool = False)`.
- `SkyPolygon(vertices: Iterable[lsst.sphgeom.LonLat], clip: bool = False)`.
- `from_astropy` and `from_sphgeom` classmethods are kept.
  `from_astropy` converts an astropy `SkyCoord`/`Angle` to `sphgeom` `LonLat`/`Angle`.
  `SkyCircle.from_sphgeom(circle)` becomes `cls(LonLat(circle.getCenter()), circle.getOpeningAngle(), clip=clip)`.

### Internal geometry

- The `lsst.geom` `radians` unit and `SpherePoint.offset(bearing, distance)` great-circle offset (used by `_boundary_skycoord` and `SkyCircle.to_polygon`) are replaced with `lsst.sphgeom.Angle` and astropy `SkyCoord.directional_offset_by(position_angle, separation)`.
- `region` builds `lsst.sphgeom.Circle(UnitVector3d(center), radius)` and `lsst.sphgeom.ConvexPolygon([UnitVector3d(v) for v in vertices])`.
- `_ast_sky_region`, `fingerprint`, and `to_fits_metadata` read longitudes, latitudes, and the radius via the `sphgeom` `LonLat`/`Angle` accessors (`getLon()`/`getLat()`/`asRadians()`/`asDegrees()`).
- The module helpers `_angle_from_astropy` and `_spherepoint_from_astropy` become `sphgeom`-valued: the former returns a `sphgeom.Angle`, the latter (renamed `_lonlat_from_astropy`) returns a `sphgeom.LonLat`.

### `to_pixels` and `set_mask`

- `to_pixels(projection: SkyProjection, bbox: Box, *, backend: MaskBackend = MaskBackend.AST) -> PixelStencil`.
  No afw `SkyWcs`/`Box2I` acceptance and no internal normalization.
- `set_mask(mask: lsst.images.Mask, plane: str)` replaces `set_mask(mask, bits)`.
  Each backend implements a `_coverage() -> np.ndarray` primitive returning a boolean array over `self.bbox`; the base `set_mask` writes that coverage into the named plane with `Mask.set(plane, ...)`.
  This collapses the two `_apply_boolean_to_mask` call sites into one and removes the afw bitmask-OR idiom.
  The mask passed in must already define `plane`; inserting a new plane is out of scope (see Non-goals).

## Shared FITS-WCS helper

A new private module (working name `_fits_projection.py`) exposes:

```
projection_and_bbox_from_fits_header(header, shape) -> tuple[SkyProjection, Box]
```

The helper is written entirely against the AST wrapper that `lsst.images` already
provides in `lsst.images._transforms._ast`, so that all astshim-versus-starlink
detection and conversion is handled inside `lsst.images` and never appears in this
package.
That wrapper re-exports astshim when it is installed and otherwise supplies
starlink-pyast-backed shims with the same (astshim-style) interface, exposing a
`USING_STARLINK_PYAST` flag and `FitsChan`/`FrameSet`/`StringStream`/`Channel`
types.
`SkyProjection.from_ast_frame_set` consumes a wrapper `FrameSet` directly, so a
`FrameSet` produced by the wrapper's `FitsChan` needs no bridging regardless of
backend.

1. Feed the header into the wrapper's `StringStream` (it accepts the single
   concatenated block from `astropy.io.fits.Header.tostring()`) and read it with
   the wrapper's `FitsChan`, taking the single resulting `FrameSet`.
   For LSST FITS data this FrameSet has a base `GRID` frame (pixel coordinates), a
   current `SKY` frame (the primary celestial WCS), and a third frame whose
   identifier is `"A"` (the alternate WCS that encodes the parent origin).
2. Build the projection with
   `SkyProjection.from_ast_frame_set(frame_set, GeneralFrame(unit=u.pix))`.
3. Derive the parent origin (XY0) by locating the frame with identifier `"A"`,
   applying the `GRID -> "A"` mapping to the grid origin, and build
   `full_bbox` as an `lsst.images.Box` from XY0 and `shape`.
   The exact integer convention (1-based AST `GRID` versus 0-based `lsst.images`
   pixels) is validated during implementation against the current
   `getImageXY0FromMetadata(pl, "A")` result, which is the reference; no heuristic
   offset is introduced.

This helper is the single home for FITS-WCS parsing and is used by both
`projection_finders.ReadComponentsAstropyFits` and the astropy branches of
`_image_cutout.py`.

Because the helper depends only on `lsst.images` (its public `SkyProjection`/`Box`
APIs plus the `_transforms._ast` wrapper) and astropy, it is deliberately shaped so
it can be lifted into `lsst.images` later — for example as a
`SkyProjection.from_fits_header` constructor — by moving the file and changing the
wrapper import from `lsst.images._transforms._ast` to a package-relative one.
The dependency on the currently-private `_transforms._ast` module is accepted as an
interim coupling for exactly this reason.

## `projection_finders.py` changes

- `ProjectionFinder.find_projection` and `ProjectionFinder.__call__` return `tuple[SkyProjection, Box] | None` (and `tuple[SkyProjection, Box]`), with docstrings updated accordingly.
- `ReadComponentsAstropyFits.find_projection` uses the shared FITS helper and drops `makeSkyWcs`, `getImageXY0FromMetadata`, `PropertyList`, and `lsst.geom.Extent2I`/`Box2I`.
- `ReadComponents.find_projection` reads the afw `wcs`/`bbox` butler components and converts them with `SkyProjection.from_legacy` and `Box.from_legacy`.
- `UseSkyMap.find_projection` converts the afw `SkyWcs` and `Box2I` it receives from `lsst.skymap` with `from_legacy`.
  `lsst.skymap` remains afw-based; the afw objects are a transient input and never leave the finder.
- `Chain`, `TryComponentParents`, and `MatchDatasetTypeName` are unchanged apart from the return-type annotations.

## `_image_cutout.py` changes

- The non-astropy paths call `to_pixels(projection, box)` with the modern types returned by `find_projection`; no conversion remains at those call sites.
- The two near-identical astropy branches (in `_extract_ref_legacy` and `_extract_ref_v2`) are unified into one block that calls the shared FITS helper to obtain the `SkyProjection` and `full_bbox`, then `to_pixels`.
- `Extraction.mask`:
  - For the afw branch (Exposure/MaskedImage/Mask), build an empty `lsst.images.Mask` sized to `pixel_stencil.bbox` with a schema that includes the `STENCIL` plane, call `pixel_stencil.set_mask(images_mask, "STENCIL")`, read `images_mask.get("STENCIL")`, and bitwise-OR it into the afw mask plane, preserving current afw output.
  - For the v2 `lsst.images` branch (today a no-op stub), call `set_mask` on the cutout's `lsst.images` mask; full wiring depends on the pending `lsst.images.Mask` add-plane functionality.

## Testing

- `test_stencils.py`: construct stencils from `sphgeom` `LonLat`/`Angle`; pass a `SkyProjection` to `to_pixels` (converting any test `SkyWcs` with `SkyProjection.from_legacy`); assert coverage via `lsst.images.Mask.get("STENCIL")`.
  Remove the `_as_projection`/`_as_box`/`_PIXEL_FRAME` boundary-helper tests.
  Keep the brute-force reference and the AST-vs-SPHGEOM comparison test, now reading coverage from the `lsst.images.Mask`.
- A new unit test covers `projection_and_bbox_from_fits_header` with a synthetic header carrying a primary celestial WCS and an `A` alternate WCS, asserting the round-trip sky coordinates and the derived `full_bbox`/XY0.
- `bench_stencils.py` is updated to the new constructor inputs, `SkyProjection`, and `lsst.images.Mask`.

## Risks and validated facts

- Validated in this environment (with astshim present, `USING_STARLINK_PYAST` false): `lsst.images.Mask.set`/`get` round-trip a boolean plane; `sphgeom` `LonLat`/`UnitVector3d`/`Circle`/`Angle` and astropy `directional_offset_by` behave as needed; the wrapper's `FitsChan.read()` (fed from `Header.tostring()` via the wrapper `StringStream`) yields one `FrameSet` with `GRID`/`SKY`/`A` frames; `SkyProjection.from_ast_frame_set` consumes that wrapper `FrameSet` and reproduces `CRPIX -> CRVAL`; the `GRID -> "A"` mapping reproduces `CRVAL*A` at the reference pixel.
- The XY0 integer convention must match `getImageXY0FromMetadata` exactly; this is checked against that function during implementation rather than approximated.
- The starlink-pyast wrapper path (`USING_STARLINK_PYAST` true) cannot be exercised where astshim is installed.  The helper uses only the wrapper interface, which is identical across backends, so correctness there rests on the wrapper's own guarantee; it should still be confirmed once in an astshim-free environment.
