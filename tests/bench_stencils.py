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

import astropy.units as u
import astropy.wcs

from lsst.dax.images.cutout.stencils import MaskBackend, SkyCircle
from lsst.images import Box, GeneralFrame, Mask, MaskPlane, MaskSchema, SkyProjection
from lsst.sphgeom import Angle, LonLat

CENTER = LonLat.fromDegrees(12.0, 13.0)
PIXEL_SCALE = 0.2  # arcsec/pixel

# (label, circle radius in arcsec, half-size of the square bbox in pixels).
CASES = (
    ("small", 5.0, 64),
    ("medium", 30.0, 256),
    ("large", 120.0, 1024),
)
REPEATS = 20


def _arcsec(value: float) -> Angle:
    """Return a `lsst.sphgeom.Angle` for ``value`` arcseconds."""
    return Angle((value * u.arcsec).to_value(u.rad))


def _projection() -> SkyProjection:
    """Build a gnomonic `SkyProjection` centered on ``CENTER``."""
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [CENTER.getLon().asDegrees(), CENTER.getLat().asDegrees()]
    scale = PIXEL_SCALE / 3600.0
    wcs.wcs.cd = [[-scale, 0.0], [0.0, scale]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return SkyProjection.from_fits_wcs(wcs, GeneralFrame(unit=u.pix))


def _run(projection: SkyProjection, radius_arcsec: float, half: int, backend: MaskBackend) -> float:
    stencil = SkyCircle(CENTER, _arcsec(radius_arcsec))
    box = Box.factory[-half : half + 1, -half : half + 1]
    schema = MaskSchema([MaskPlane("STENCIL", "stencil coverage")])

    def once() -> None:
        pixel_stencil = stencil.to_pixels(projection, box, backend=backend)
        mask = Mask(schema=schema, bbox=box)
        pixel_stencil.set_mask(mask, "STENCIL")

    return min(timeit.repeat(once, number=1, repeat=REPEATS))


def main() -> None:
    """Print a table of best-of-N timings for each backend and cutout size."""
    projection = _projection()
    print(f"{'case':>8} {'bbox':>12} {'AST (ms)':>12} {'SPHGEOM (ms)':>14}")
    for label, radius_arcsec, half in CASES:
        side = 2 * half + 1
        ast_ms = _run(projection, radius_arcsec, half, MaskBackend.AST) * 1e3
        sph_ms = _run(projection, radius_arcsec, half, MaskBackend.SPHGEOM) * 1e3
        print(f"{label:>8} {f'{side}x{side}':>12} {ast_ms:>12.3f} {sph_ms:>14.3f}")


if __name__ == "__main__":
    main()
