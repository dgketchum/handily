"""2 km moving-window mean of the 100 m CONUS DEM on the canonical 100 m lattice.

Replacement for ``hydrography90m/coarse_surface.tif`` (a 2 km BLOCK mean, 2000 m
pixels on 2 km multiples) as the reference surface of the query feature
``elev_above_coarse_m = z_surf - coarse``. Nearest-sampled, the block raster
steps by 40-160 m on 2 km lines across flat valley floors, and the renderer
draws those steps into the DTW map (notes/NV_RASTER_DEBUG_2026-10-07.md
section 2). A centred moving window on the 100 m lattice carries the same 2 km
scale with no block edges.

Recipe:
  window  = 21 x 21 cells (2.1 km x 2.1 km, centred on the cell)
  value   = sum(valid DEM in window) / count(valid DEM in window)   (m)
  nodata  = cells whose window holds no valid DEM cell (stay 32767)
  output  = Int16 (rounded to 1 m, like the block raster), deflate, tiled,
            BIGTIFF, same CRS / transform / shape as the 100 m DEM

Out-of-lattice neighbours count as invalid (zero-padded sum and count), so the
lattice edge is nodata-aware exactly like an interior nodata hole.

Processed in horizontal strips with a 10-row halo (scipy.ndimage.uniform_filter
on float64 sums and counts); peak memory is a few strip-sized float64 arrays.

Usage:
    uv run python utils/build_coarse_surface_mw2km.py
    uv run python utils/build_coarse_surface_mw2km.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_coarse_surface_mw2km")

DEM_100M = "/data/ssd2/handily/conus/covariates/elev48i0100a.tif"
OLD_COARSE = "/data/ssd2/handily/conus/hydrography90m/coarse_surface.tif"
OUT_TIF = "/data/ssd2/handily/conus/hydrography90m/coarse_surface_mw2km_100m.tif"
WINDOW_CELLS = 21  # 21 x 100 m = 2.1 km, odd so the window is centred
NODATA = 32767
STRIP_ROWS = 2048
# Nevada bounding box (EPSG:5070) for the 2 km-line enrichment check
NV_BOX = (-2_060_000.0, 1_600_000.0, -1_580_000.0, 2_460_000.0)


def moving_mean(dem: np.ndarray, nodata: float, size: int) -> np.ndarray:
    """Nodata-aware centred ``size`` x ``size`` window mean (float64, NaN = no data).

    Zero-padded outside the array: the out-of-array neighbours add nothing to
    the sum or the count, so the mean is over the valid in-array cells only.
    """
    valid = dem != nodata
    vals = np.where(valid, dem, 0).astype("float64")
    s = uniform_filter(vals, size=size, mode="constant", cval=0.0)
    c = uniform_filter(valid.astype("float64"), size=size, mode="constant", cval=0.0)
    # uniform_filter returns window MEANS; counts below half a cell are zero
    # valid neighbours up to float round-off.
    out = np.full(dem.shape, np.nan)
    ok = c * size * size > 0.5
    out[ok] = s[ok] / c[ok]
    return out


def build(dem_path: str, out_path: str, size: int, strip_rows: int) -> dict:
    halo = size // 2
    t0 = time.time()
    with rasterio.open(dem_path) as src:
        prof = src.profile.copy()
        h, w = src.height, src.width
        nodata = src.nodata
        if nodata is None:
            raise SystemExit(f"DEM has no nodata value: {dem_path}")
        prof.update(
            dtype="int16",
            nodata=NODATA,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="YES",
        )
        n_valid_in = n_valid_out = 0
        with rasterio.open(out_path, "w", **prof) as dst:
            for r0 in range(0, h, strip_rows):
                r1 = min(r0 + strip_rows, h)
                p0, p1 = max(0, r0 - halo), min(h, r1 + halo)
                dem = src.read(1, window=Window(0, p0, w, p1 - p0))
                mm = moving_mean(dem, nodata, size)[r0 - p0 : r0 - p0 + (r1 - r0)]
                n_valid_in += int((dem[r0 - p0 : r0 - p0 + (r1 - r0)] != nodata).sum())
                ok = np.isfinite(mm)
                n_valid_out += int(ok.sum())
                out = np.full(mm.shape, NODATA, dtype="int16")
                out[ok] = np.rint(mm[ok]).astype("int16")
                dst.write(out, 1, window=Window(0, r0, w, r1 - r0))
                log.info("rows %d-%d / %d", r0, r1, h)
    runtime = time.time() - t0
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    log.info("built %s in %.0f s, peak RSS %.1f GB", out_path, runtime, peak_gb)
    return {
        "runtime_s": runtime,
        "peak_rss_gb": peak_gb,
        "n_valid_dem": n_valid_in,
        "n_valid_out": n_valid_out,
        "width": w,
        "height": h,
    }


def sample_nearest(path: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as ds:
        vals = np.array([v[0] for v in ds.sample(zip(x, y))], dtype="float64")
        if ds.nodata is not None:
            vals[vals == ds.nodata] = np.nan
    return vals


def jump_enrichment(path: str, box: tuple, res_line: float = 2000.0) -> dict:
    """Column/row jump enrichment on ``res_line`` lines vs elsewhere, 100 m cells.

    The ``state_2km.py`` statistic of notes/NV_RASTER_DEBUG_2026-10-07.md
    section 2, on a raster resampled (nearest) to the 100 m lattice inside
    ``box``: mean |jump| between adjacent columns whose shared edge falls on a
    ``res_line`` multiple, divided by the mean |jump| on all other edges
    (dimensionless; 1.0 = no excess at the lines). Same for the share of jumps
    > 5 m, and for rows. ``1km_not_2km`` is the control (edges on 1 km but not
    2 km multiples).
    """
    xmin, ymin, xmax, ymax = box
    xs = np.arange(xmin + 50.0, xmax, 100.0)
    ys = np.arange(ymax - 50.0, ymin, -100.0)
    with rasterio.open(path) as ds:
        t = ds.transform
        col = np.floor((xs - t.c) / t.a).astype(int)
        row = np.floor((ys - t.f) / t.e).astype(int)
        win = Window(
            col.min(), row.min(), col.max() - col.min() + 1, row.max() - row.min() + 1
        )
        a = ds.read(1, window=win).astype("float64")
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
    g = a[np.ix_(row - row.min(), col - col.min())]
    return edge_enrichment(g, xs, ys, res_line)


def edge_enrichment(
    g: np.ndarray, xs: np.ndarray, ys: np.ndarray, res_line: float = 2000.0
) -> dict:
    """Enrichment of |adjacent-cell jumps| on ``res_line`` grid lines (see jump_enrichment).

    ``g`` is a 100 m grid with cell-centre coords ``xs`` (columns) and ``ys``
    (rows). Edge between columns j and j+1 sits at ``xs[j] + 50``.
    """
    out = {}
    for axis, centres in (("col", xs), ("row", ys)):
        d = np.abs(np.diff(g, axis=1 if axis == "col" else 0))
        step = 50.0 if axis == "col" else -50.0
        edges = centres[:-1] + step
        on2 = np.isclose(np.mod(edges, res_line), 0) | np.isclose(
            np.mod(edges, res_line), res_line
        )
        on1 = np.isclose(np.mod(edges, res_line / 2), 0) | np.isclose(
            np.mod(edges, res_line / 2), res_line / 2
        )
        sel = (lambda m: d[:, m]) if axis == "col" else (lambda m: d[m, :])
        lines = sel(on2)
        ctrl = sel(on1 & ~on2)
        other = sel(~on1)
        lines, ctrl, other = (v[np.isfinite(v)] for v in (lines, ctrl, other))
        if not (len(lines) and len(other)):
            raise SystemExit("no finite jumps on the 2 km lines or elsewhere")
        out[axis] = {
            "mean_abs_jump_2km_m": float(lines.mean()),
            "mean_abs_jump_elsewhere_m": float(other.mean()),
            "enrich_mean": float(lines.mean() / other.mean()),
            "enrich_gt5m": float((lines > 5).mean() / max((other > 5).mean(), 1e-12)),
            "enrich_mean_1km_not_2km": float(ctrl.mean() / other.mean()),
            "n_edges_2km": int(len(lines)),
        }
    return out


def verify(out_path: str, n: int = 1000, seed: int = 0) -> dict:
    with rasterio.open(out_path) as ds:
        t, w, h = ds.transform, ds.width, ds.height
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    while len(xs) < n:
        c = rng.integers(0, w, 4 * n)
        r = rng.integers(0, h, 4 * n)
        x = t.c + (c + 0.5) * t.a
        y = t.f + (r + 0.5) * t.e
        new = sample_nearest(out_path, x, y)
        dem = sample_nearest(DEM_100M, x, y)
        old = sample_nearest(OLD_COARSE, x, y)
        ok = np.isfinite(new) & np.isfinite(dem) & np.isfinite(old)
        xs += list(x[ok])
        ys += list(y[ok])
    x, y = np.array(xs[:n]), np.array(ys[:n])
    d = sample_nearest(out_path, x, y) - sample_nearest(OLD_COARSE, x, y)
    res = {
        "n_cells": int(n),
        "new_minus_old_median_m": float(np.median(d)),
        "new_minus_old_mean_m": float(np.mean(d)),
        "new_minus_old_mad_m": float(np.median(np.abs(d))),
        "new_minus_old_p05_p95_m": [
            float(np.percentile(d, 5)),
            float(np.percentile(d, 95)),
        ],
        "enrichment_nv_box_new": jump_enrichment(out_path, NV_BOX),
        "enrichment_nv_box_old": jump_enrichment(OLD_COARSE, NV_BOX),
    }
    log.info("verify: %s", json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem", default=DEM_100M)
    ap.add_argument("--out", default=OUT_TIF)
    ap.add_argument("--window-cells", type=int, default=WINDOW_CELLS)
    ap.add_argument("--strip-rows", type=int, default=STRIP_ROWS)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if args.window_cells % 2 != 1:
        raise SystemExit("--window-cells must be odd (centred window)")
    man_path = Path(args.out).with_suffix(".json")
    if args.verify_only:
        man = json.loads(man_path.read_text())
    else:
        stats = build(args.dem, args.out, args.window_cells, args.strip_rows)
        man = {
            "recipe": "centred moving-window mean of the 100 m DEM, nodata-aware "
            "(sum of valid / count of valid; zero valid neighbours -> nodata), "
            "rounded to Int16 m",
            "window_cells": args.window_cells,
            "window_m": args.window_cells * 100.0,
            "window_note": f"{args.window_cells} x {args.window_cells} cells of "
            f"100 m = {args.window_cells * 0.1:.1f} km square, centred",
            "source_dem": args.dem,
            "replaces": OLD_COARSE,
            "consumer": "infer_conus_gnn.py --coarse-surface (renderer query "
            "feature elev_above_coarse_m only; bundle unchanged)",
            "nodata": NODATA,
            "units": "m",
            **stats,
        }
    man["verify"] = verify(args.out)
    man_path.write_text(json.dumps(man, indent=2))
    log.info("wrote %s", man_path)


if __name__ == "__main__":
    main()
