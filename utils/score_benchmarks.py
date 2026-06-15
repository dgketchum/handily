"""Score published WT/DTW benchmark maps against the Ruby GWX wells + NHD springs.

Extends the FAC-vs-Ma comparison (``compare_fac_ma.py``) to an arbitrary set of
gridded depth-to-water products (Zell & Sanford 2020, Fan 2013, SSURGO, etc. from
``/nas/gwx/studies/benchmark_maps_mt/``). Every product is scored on the SAME
in-grid well set as the FAC REM, with the same metrics, so FAC, Ma, and each
benchmark are directly rankable.

Fairness: the in-grid well set is defined by the FAC REM footprint (out-of-grid
wells -> NaN, dropped). Each product is then scored on the wells where IT is
finite (its own nodata masked), and ``n_usable_valid`` is reported per product so
sparse / partial-coverage layers (e.g. SSURGO, which only maps <~2 m wetland
soils) are visible as a small-n caveat rather than a silent biased subset. The
benchmarks are references to beat, never tuning targets.

Residual sign: ``residual = predicted_dtw - observed_dtw`` (positive = too deep).
Springs are zero-DTW anchors (residual = buffered-min predicted DTW).

Usage:
    uv run python utils/score_benchmarks.py \
        --fac-rem .../rem/ruby_v3_wetter/fac_head_depth_rem_10m.tif \
        --ma /nas/gwx/wtd_states/wtd_montana.tif \
        --benchmarks \
            zell_sanford:/nas/gwx/studies/benchmark_maps_mt/dtw_positive_down_m/zell_sanford_2020_dtw_mt_5070.tif \
            fan_2013:/nas/gwx/studies/benchmark_maps_mt/dtw_positive_down_m/fan_2013_wtd_mt_5070.tif \
            ssurgo_min:/nas/gwx/studies/benchmark_maps_mt/dtw_positive_down_m/ssurgo_wtdep_min_m_mt_5070.tif \
        --wells .../evidence/gwx/ruby_well_observation_labels.parquet \
        --springs .../evidence/springs/nhd_springs_45800.fgb \
        --streams .../streams_regional.fgb \
        --out-dir .../validation/benchmarks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_fac_ma import (  # noqa: E402
    sample_raster_at_points,
    spring_rows,
    well_metrics,
)
from score_spring_anchors import score_springs  # noqa: E402

log = logging.getLogger("score_benchmarks")

MA_DEFAULT = "/nas/gwx/wtd_states/wtd_montana.tif"


def _parse_named(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items or []:
        if ":" not in it:
            raise SystemExit(f"--benchmarks entries must be name:path, got {it!r}")
        name, path = it.split(":", 1)
        out[name] = path
    return out


def _get(sl: pd.DataFrame, gt: str, g: str, st: str):
    m = sl[(sl.group_type == gt) & (sl.group == g) & (sl.statistic == st)]
    return None if m.empty else m["value"].iloc[0]


def build_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per product with the headline accuracy metrics."""
    rows = []
    for label in summary["raster"].unique():
        sl = summary[summary["raster"] == label]
        rows.append(
            {
                "product": label,
                "n_usable_valid": _get(sl, "tier", "primary+secondary", "n"),
                "primary_mad_m": _get(sl, "tier", "primary", "median_abs_residual_m"),
                "primary_bias_m": _get(sl, "tier", "primary", "bias_m"),
                "valley_mad_m": _get(sl, "setting", "valley", "median_abs_residual_m"),
                "upland_mad_m": _get(sl, "setting", "upland", "median_abs_residual_m"),
                "all_mad_m": _get(sl, "tier", "all", "median_abs_residual_m"),
                "all_rmse_m": _get(sl, "tier", "all", "rmse_m"),
                "shallow_recall_5m": _get(sl, "shallow_class", "<5m", "recall"),
                "shallow_prec_5m": _get(sl, "shallow_class", "<5m", "precision"),
                "shallow_recall_2m": _get(sl, "shallow_class", "<2m", "recall"),
                "spring_capture_60m": _get(sl, "capture", "60m", "capture_rate"),
                "spring_med_resid_m": _get(sl, "residual", "all", "median_residual_m"),
            }
        )
    rank = pd.DataFrame(rows)
    return rank.sort_values("valley_mad_m", na_position="last").reset_index(drop=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fac-rem", required=True, help="FAC REM DTW raster (defines in-grid)"
    )
    p.add_argument("--ma", default=MA_DEFAULT, help="Ma WTD raster")
    p.add_argument(
        "--benchmarks",
        nargs="*",
        default=[],
        help="name:path entries for benchmark DTW rasters",
    )
    p.add_argument("--wells", required=True)
    p.add_argument("--springs", required=True)
    p.add_argument(
        "--streams", default=None, help="streams_regional.fgb (valley/upland)"
    )
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wells = gpd.read_parquet(args.wells)
    springs = gpd.read_file(args.springs)
    streams = gpd.read_file(args.streams) if args.streams else None

    products = {"fac": args.fac_rem, "ma": args.ma, **_parse_named(args.benchmarks)}

    # In-grid restriction off the FAC REM footprint (same as compare_fac_ma).
    in_grid = np.isfinite(sample_raster_at_points(args.fac_rem, wells))
    n_drop = int((~in_grid).sum())
    if n_drop:
        log.info(
            "Dropping %d / %d wells outside the FAC REM grid; scoring all products "
            "on the %d in-grid wells",
            n_drop,
            len(wells),
            int(in_grid.sum()),
        )
    wells = wells.loc[in_grid].reset_index(drop=True)

    all_rows: list[dict] = []
    well_resids = wells[
        [
            "canonical_id",
            "tier",
            "weight",
            "confinement_class",
            "dtw_label_m",
            "geometry",
        ]
    ].copy()
    # Resolution-fair spring metric: the buffered-min capture@60m is sub-pixel for
    # the coarse CONUS products (250 m / 1 km), so it reads NaN there. The
    # exact-cell predicted DTW at the spring (the cell the spring falls in) is
    # defined at any resolution and is the cross-product spring number.
    spring_exact: dict[str, float] = {}
    for label, path in products.items():
        log.info("Scoring %s: %s", label, path)
        rows, resid = well_metrics(wells, path, label, streams)
        all_rows += rows
        well_resids[f"pred_{label}"] = sample_raster_at_points(path, wells)
        well_resids[f"residual_{label}"] = resid
        s_out, s_sum = score_springs(path, springs)
        all_rows += spring_rows(label, s_sum)
        ex = s_out["exact_dtw"].to_numpy(dtype="float64")
        spring_exact[label] = float(np.nanmedian(ex)) if np.isfinite(ex).any() else None

    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "benchmark_score_summary.csv"
    summary.to_csv(summary_path, index=False)
    well_resids.to_file(out_dir / "benchmark_well_residuals.fgb", driver="FlatGeobuf")

    ranking = build_ranking(summary)
    ranking["spring_exact_med_dtw_m"] = ranking["product"].map(spring_exact)
    ranking_path = out_dir / "benchmark_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    with open(out_dir / "benchmark_run.json", "w") as f:
        json.dump(
            {
                "products": products,
                "n_in_grid_wells": len(wells),
                "outputs": {"summary": str(summary_path), "ranking": str(ranking_path)},
            },
            f,
            indent=2,
        )
    log.info("Wrote %s", ranking_path)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
