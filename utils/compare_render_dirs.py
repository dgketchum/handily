"""Compare two GNN render directories raster by raster (cross-machine parity check).

For every ``*.tif`` present in both directories: same grid (shape + transform),
same nodata mask, and the max |difference| over valid cells, with the count of
cells differing by more than ``--tol``. Exit 1 if any raster differs.

    uv run python utils/compare_render_dirs.py --a <zoran gnn dir> --b <zephyr gnn dir> --tol 1e-3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tol", type=float, default=1e-3, help="metres (or layer units)")
    args = ap.parse_args()
    a, b = Path(args.a), Path(args.b)
    names = sorted(
        {p.name for p in a.glob("*.tif")} & {p.name for p in b.glob("*.tif")}
    )
    if not names:
        raise SystemExit("no common rasters")
    print(
        f"{'raster':32s} {'grid':5s} {'mask':5s} {'n_valid':>9s} {'max_abs':>10s} {'n_gt_tol':>9s}"
    )
    bad = 0
    for n in names:
        with rasterio.open(a / n) as ra, rasterio.open(b / n) as rb:
            grid = ra.shape == rb.shape and ra.transform == rb.transform
            xa, xb = ra.read(1).astype("float64"), rb.read(1).astype("float64")
            va = (
                np.isfinite(xa)
                if ra.nodata is None
                else (xa != ra.nodata) & np.isfinite(xa)
            )
            vb = (
                np.isfinite(xb)
                if rb.nodata is None
                else (xb != rb.nodata) & np.isfinite(xb)
            )
        mask = grid and bool((va == vb).all())
        if grid:
            d = np.abs(xa - xb)[va & vb]
            mx = float(d.max()) if d.size else 0.0
            ngt = int((d > args.tol).sum())
        else:
            mx, ngt = float("nan"), -1
        ok = grid and mask and ngt == 0
        bad += 0 if ok else 1
        print(
            f"{n:32s} {'ok' if grid else 'DIFF':5s} {'ok' if mask else 'DIFF':5s} {int(va.sum()):9d} {mx:10.4g} {ngt:9d}"
        )
    print("IDENTICAL within tol" if bad == 0 else f"{bad} raster(s) DIFFER")
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
