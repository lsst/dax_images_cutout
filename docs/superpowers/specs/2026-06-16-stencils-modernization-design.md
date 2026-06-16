# Modernizing `stencils.py` onto `lsst.images` + starlink-pyast + sphgeom

- Date: 2026-06-16
- Ticket: DM-55210
- Status: Design approved; pending written-spec review.

## Background

`stencils.py` defines the cutout geometry abstractions used by `dax_images_cutout`.
`SkyStencil` (with `SkyCircle` and `SkyPolygon`) describes a cutout region in ICRS sky coordinates.
`PixelStencil` (with `PixelPolygon`) is the pixel-coordinate projection of a sky stencil for a particular WCS and bounding box.
The module is consumed only within this package, primarily by `_image_cutout.py`.

The current implementation leans heavily on `lsst.afw`.
It uses `afw.geom.SkyWcs` for coordinate transforms, `afw.geom.polygon.Polygon` plus `Polygon.createImage` for rasterization, and `lsst.geom` `Box2I`/`Box2D`/`Point2D` for geometry.
We want to modernize the module so its internal logic depends only on `lsst.images` (`SkyProjection`, `Box`, `Interval`, `XY`), starlink-pyast, and `lsst.sphgeom`.
The legacy cutout path reads `lsst.afw.image.Exposure` datasets, so we must continue to accept afw objects (`SkyWcs`, `Box2I`, and an afw `Mask`) at the public boundary.

## Goals

- Rewrite `stencils.py` so its internal logic uses only `lsst.images`, starlink-pyast, and `lsst.sphgeom`.
- Keep accepting afw `SkyWcs` and `geom.Box2I` as inputs to `to_pixels`, and an afw `Mask` as the `set_mask` target.
- Replace afw polygon rasterization with center-based masking of the true sky region.
- Implement two masking backends side by side so we can compare accuracy and performance before committing to one.
- Benchmark both backends, because this is time-critical code.
- Update `test_stencils.py` to cover and compare both backends.

## Non-goals

- Wiring the `lsst.images.Mask` (bit-packed `MaskSchema`) masking path in `Extraction.mask`.
  That path is currently a no-op stub and depends on new `lsst.images` functionality to assign a mask plane; it is a separate follow-up.
- Changing the public method names or call patterns of `SkyStencil`/`PixelStencil` beyond what is described here.
- Reworking `projection_finders.py` or the cutout-extraction logic except for the minimal `bbox`-type edits described below.

## Key decisions

These were settled during brainstorming.

1. Masking uses the **true sky region** rather than a pixel-space polygon approximation.
   Both backends evaluate pixel-center containment, which matches the brute-force reference in the test and removes the circle-to-polygon approximation from the masking path.
2. Scope is `stencils.py` and `test_stencils.py`, plus the minimal `_image_cutout.py` edits required by the `bbox` type change.
3. We build **two masking backends** and compare them during review: a starlink-pyast `Region.mask` backend and a `sphgeom` brute-force backend that uses only public `lsst.images` APIs.
4. `PixelStencil.bbox` returns `lsst.images.Box`; touching `_image_cutout.py` to accommodate this is acceptable.

## Boundary policy

afw types are accepted only at the public boundary and normalized to `lsst.images` types at the top of `to_pixels`, so the internal logic is single-path and modern.

| Boundary | Accepted (legacy) | Normalized to (internal) |
| --- | --- | --- |
| WCS argument | `lsst.afw.geom.SkyWcs` | `SkyProjection` via `SkyProjection.from_legacy(wcs, GeneralFrame(unit="pix"))` |
| `bbox` argument | `lsst.geom.Box2I` | `lsst.images.Box` via `Box.from_legacy` |
| `set_mask` target | `lsst.afw.image.Mask` | unchanged for now; the modern `Mask` path is a follow-up |

`SkyProjection` is always available internally, because afw `SkyWcs` is converted with `SkyProjection.from_legacy`, and the v2 path already supplies a `SkyProjection`.
The following afw usages are removed entirely: `afw.geom.polygon`, `afw.geom.SkyWcs` direct use, `makeSkyWcs`/`makeCdMatrix`, `lsst.geom` `Box2D`/`Point2D`, and the unused `_make_local_gnomonic_wcs` helper.
`lsst.sphgeom` remains; it is the `region` representation and the basis of one backend.

## Sky-side classes

`SkyStencil` (ABC), `SkyCircle`, and `SkyPolygon` keep their existing public surface.
Preserved members include `from_astropy`, `SkyCircle.from_sphgeom`, `SkyCircle.to_polygon`, the `region` property (returning a `lsst.sphgeom.Region`), `to_fits_metadata`, and `fingerprint`.

Changes:

- Add an internal `_ast_sky_region()` returning a `starlink.Ast.Region`.
  `SkyCircle` builds an `Ast.Circle` in an ICRS `Ast.SkyFrame`.
  `SkyPolygon` builds an `Ast.Polygon` whose edges are great circles in an ICRS `Ast.SkyFrame`, consistent with the `sphgeom.ConvexPolygon` representation.
- `SkyCircle.to_polygon` and the circle-to-polygon approximation are kept as public helpers but are no longer on the masking path.
- `to_fits_metadata` continues to accept either a `PropertyList` or a plain mutable mapping, because the legacy extraction path passes a `PropertyList` and the v2 path passes a `dict`.

## Two masking backends

`to_pixels(wcs, bbox, *, backend=...)` selects which concrete `PixelStencil` it returns via a `backend` argument (an enum with `AST` and `SPHGEOM` members).
Both backends normalize the boundary inputs, compute a tight pixel `Box` clipped to the requested `bbox` (or raise `StencilNotContainedError` when `clip` is `False` and the stencil is not contained), and rasterize a boolean inside-pixel array with center-based semantics.

### `AstRegionPixelStencil` (backend `AST`, the starlink-pyast option)

- Build the sky region with `_ast_sky_region()`.
- Obtain the sky-to-pixel AST mapping from the `SkyProjection`.
  Today this is the private `SkyProjection.sky_to_pixel_transform._ast_mapping`; a `TODO` records that we should add a public accessor to `lsst.images` if we keep this backend.
- Remap the sky region into a 2-d pixel frame with `region.mapregion(sky_to_pixel_mapping, Ast.Frame(2))`.
- Derive the tight pixel bounding box from `pixel_region.getregionbounds()`.
- `set_mask` rasterizes by calling `pixel_region.mask(...)` over the box and OR-ing the resulting boolean array into the afw mask plane.
- The masking polarity and the AST polygon vertex orientation (AST's inside/outside convention, with `negate()` if required) are verified against the brute-force test during implementation.

### `SphgeomPixelStencil` (backend `SPHGEOM`, the public-API option)

- Use the existing `sphgeom` `region` property.
- Compute the tight pixel bounding box by transforming a boundary sample of the region to pixels with the public `SkyProjection.sky_to_pixel`.
  A circle samples points around its circumference; a polygon uses its vertices.
- `set_mask` transforms every pixel center in the box to the sky with public `SkyProjection.pixel_to_sky`, tests containment with `region.contains(ra, dec)`, reshapes, and OR-s the result into the afw mask plane.
- This backend uses no starlink-pyast and no private `lsst.images` attributes.

Both backends produce a boolean array over the same bounding box and apply it identically to the afw mask, so the only differences under comparison are the bounding box and the set of masked pixels.

## `bbox` type change and `_image_cutout.py` edits

`PixelStencil.bbox` returns `lsst.images.Box` instead of `lsst.geom.Box2I`.
This requires two edits in `_image_cutout.py`:

- The legacy extraction path passes `pixel_stencil.bbox.to_legacy()` to `butler.get(..., parameters={"bbox": ...})`.
- The v2 extraction path uses `pixel_stencil.bbox` directly and drops the `lsst.images.Box.from_legacy(pixel_stencil.bbox)` wrapper.

These are the only changes outside `stencils.py` and the tests.

## Testing and benchmarking

`test_stencils.py` keeps the brute-force checker, which is the reference for correctness and is effectively the sphgeom backend's own algorithm.

- Run the existing `to_pixels`/`set_mask` checks for both backends against the brute-force reference, within the established `max_missing` and `max_extra` tolerances.
- Add a comparison test that builds both backends for the same stencil, WCS, and bbox and asserts they agree on the bounding box and on the masked-pixel set.
- Cover both the `SkyWcs` and `SkyProjection` input forms and both the `Box2I` and `lsst.images.Box` input forms, so the boundary normalization is exercised.

Add an opt-in benchmark, because this is time-critical code.

- Time `to_pixels` and `set_mask` for each backend across a few representative bounding-box sizes (small and large cutouts).
- Report a table of per-backend timings and the accuracy deltas versus the brute-force reference.
- The benchmark is not part of the normal pytest run; it is invoked explicitly, in the spirit of the existing `plot` debugging flag in the test helper.

## Cleanups and bug fixes

These are corrected as part of the rewrite.

- Remove the stray `print(...)` at `stencils.py:297`.
- Fix `SkyPolygon.fingerprint`: the `getDecx()` typo and the invalid `struct.pack("!d!d", ...)` format string, which would raise if the property were ever called.
- Remove the unused `_make_local_gnomonic_wcs` helper.

## Risks and open implementation details

- AST mask polarity and AST polygon vertex orientation must be confirmed against the brute-force test; AST chooses an inside/outside convention that may require `Region.negate()`.
- Remapping a `SkyFrame` `Ast.Circle` through a non-linear sky-to-pixel mapping yields a generic AST `Region`; `getregionbounds` and `mask` are still valid on it.
- Using the private `_ast_mapping` couples the AST backend to a `lsst.images` internal; if that backend wins, the public accessor becomes a `lsst.images` prerequisite.
- The `GeneralFrame(unit="pix")` pixel frame is sufficient for the math; the frame identity only affects AST domain labeling.

## Decision criteria after review

After the comparison and benchmark, we choose one backend based on:

- Correctness against the brute-force reference across circle and polygon stencils.
- Performance across representative cutout sizes.
- Dependency cleanliness, weighing the AST backend's need for a public `lsst.images` mapping accessor against the sphgeom backend's pure-public-API implementation.
