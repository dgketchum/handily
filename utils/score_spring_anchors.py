"""Phase 5 of the Ruby HUC8 pilot: spring anchor scoring.

NHD springs (``fcode == 45800``) are candidate zero-DTW anchors, but the mapped
point is rarely the exact emergence pixel. Rather than trust an exact-point REM
sample, score each spring by the *minimum* depth-to-water within local buffers
(30 / 60 / 100 m). A spring is "captured" when the buffered minimum DTW is at or
below ``capture_threshold`` (0.5 m by default).

    spring_residual = min_buffered_dtw - 0.0

The same routine scores any DTW-like raster (the FAC REM or the Ma WTD product),
so Phase 6 reuses it to compare capture rates head to head.

Primary metric: capture rate at 60 m. Secondary: median residual and the
fraction of springs above 2 / 5 / 10 m (the misses to inspect in QGIS).

Usage:
    uv run python utils/score_spring_anchors.py \
        --rem /data/ssd2/handily/mt/regional/ruby_huc8/rem/ruby_fac10_baseline/fac_head_depth_rem_10m.tif \
        --springs /data/ssd2/handily/mt/regional/ruby_huc8/evidence/springs/nhd_springs_45800.fgb \
        --label fac --out-dir /data/ssd2/handily/mt/regional/ruby_huc8/validation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds

log = logging.getLogger("score_spring_anchors")

DEFAULT_BUFFERS = (30.0, 60.0, 100.0)
CAPTURE_THRESHOLD_M = 0.5
PRIMARY_BUFFER_M = 60.0
MISS_LEVELS = (2.0, 5.0, 10.0)


def _valid(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    m = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        m &= arr != nodata
    return m


def min_in_buffer(
    src: rasterio.DatasetReader, x: float, y: float, radius_m: float, band: int = 1
) -> float:
    """Minimum valid raster value within ``radius_m`` of point (x, y), raster CRS."""
    if radius_m <= 0:
        val = next(src.sample([(x, y)], indexes=band))[0]
        return float(val) if _valid(np.array([val]), src.nodata)[0] else np.nan
    win = (
        from_bounds(
            x - radius_m, y - radius_m, x + radius_m, y + radius_m, src.transform
        )
        .round_offsets()
        .round_lengths()
    )
    fill = src.nodata if src.nodata is not None else np.nan
    arr = src.read(band, window=win, boundless=True, fill_value=fill)
    wt = src.window_transform(win)
    ny, nx = arr.shape
    cols_i, rows_i = np.meshgrid(np.arange(nx), np.arange(ny))
    xs = wt.c + wt.a * (cols_i + 0.5) + wt.b * (rows_i + 0.5)
    ys = wt.f + wt.d * (cols_i + 0.5) + wt.e * (rows_i + 0.5)
    dist = np.hypot(xs - x, ys - y)
    mask = _valid(arr, src.nodata) & (dist <= radius_m)
    return float(arr[mask].min()) if mask.any() else np.nan


def score_springs(
    rem_path: str,
    springs: gpd.GeoDataFrame,
    buffers: tuple[float, ...] = DEFAULT_BUFFERS,
    capture_threshold: float = CAPTURE_THRESHOLD_M,
    primary_buffer: float = PRIMARY_BUFFER_M,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Score each spring against a DTW raster; return per-spring residuals and a
    summary dict (capture rates, median residual, miss fractions)."""
    with rasterio.open(rem_path) as src:
        pts = springs.to_crs(src.crs)
        rows = []
        for geom in pts.geometry:
            x, y = geom.x, geom.y
            rec = {"exact_dtw": min_in_buffer(src, x, y, 0.0)}
            for r in buffers:
                rec[f"min_dtw_{int(r)}m"] = min_in_buffer(src, x, y, r)
            rows.append(rec)

    out = springs.copy().reset_index(drop=True)
    for k in rows[0]:
        out[k] = [r[k] for r in rows]
    pcol = f"min_dtw_{int(primary_buffer)}m"
    out["residual_m"] = out[pcol]
    out["captured"] = out[pcol] <= capture_threshold

    n = len(out)
    n_valid = int(out[pcol].notna().sum())
    summary = {
        "n_springs": n,
        "n_with_raster_value": n_valid,
        "capture_threshold_m": capture_threshold,
        "primary_buffer_m": primary_buffer,
        "capture_rate": {
            f"{int(r)}m": float(
                (out[f"min_dtw_{int(r)}m"] <= capture_threshold).sum() / n
            )
            for r in buffers
        },
        "median_residual_m": float(out["residual_m"].median(skipna=True)),
        "miss_fraction_above": {
            f"{int(lvl)}m": float((out["residual_m"] > lvl).sum() / n)
            for lvl in MISS_LEVELS
        },
    }
    return out, summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rem", required=True, help="DTW raster (FAC REM or Ma WTD)")
    p.add_argument("--springs", required=True, help="spring points (fgb)")
    p.add_argument("--label", default="fac", help="tag for outputs (fac|ma)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--buffers", default="30,60,100")
    p.add_argument("--capture-threshold", type=float, default=CAPTURE_THRESHOLD_M)
    args = p.parse_args()

    buffers = tuple(float(b) for b in args.buffers.split(","))
    springs = gpd.read_file(args.springs)
    out, summary = score_springs(
        args.rem, springs, buffers=buffers, capture_threshold=args.capture_threshold
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fgb = out_dir / f"spring_residuals_{args.label}.fgb"
    out.to_file(fgb, driver="FlatGeobuf")
    summary["label"] = args.label
    summary["rem"] = args.rem
    with open(out_dir / f"spring_score_{args.label}.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", fgb)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
