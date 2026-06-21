"""Validate a FAC10 REM DTW prior against GWX unconfined wells in a region window.

Replaces the thin NWIS-only ``nm_independent_validation_wells.geoparquet`` (which
was ~99% ``confinement=unknown``) with the GWX national well index
(``/data/ssd2/gwx/products/current/wells.geoparquet``, ~6.45M wells), whose v2
confinement classifier supplies a large *labeled* unconfined population. For
Mesilla this took the headline set from 6 to ~5k wells.

Independence: NWIS/NGWMN monitoring wells are the canonical training set for
national WTD models (Ma, Janssen), so they are EXCLUDED by default
(``--exclude-sources nwis,ngwmn``) — keeping them inflates the benchmark's
apparent skill via leakage (on Mesilla, Ma scored MAD 0.49 m on the 357 NWIS
monitoring wells vs ~2.6 m on the independent nm_ose PODs). FAC/Janssen never
trained on any of these wells; the exclusion levels the field for the benchmark.

Confinement: only ``unconfined`` / ``unconfined_marginal`` enter the headline
(depth-to-water = depth to the *unconfined* table; confined wells measure a
potentiometric surface). The GWX model_x/model_s unconfined call is high-precision
(0.94-0.99 on DE ground truth); in arid deep-table basins the global HAS cuts
*over*-call confinement, so the unconfined set is conservative, not contaminated.

Residual sign: ``residual = predicted_dtw - observed_dtw`` (positive = too deep).
Ma/Janssen are benchmarks to beat, never tuning targets; FAC may be tuned to
these GWX unconfined wells + NHD springs.

Usage:
    uv run python utils/validate_fac_gwx_wells.py \
        --fac-rem /data/.../rem/nm_mesilla_v5_arid/fac_head_depth_rem_10m.tif \
        --streams /data/.../mesilla/streams_regional.fgb \
        --ma /nas/gwx/wtd_states/wtd_new_mexico.tif \
        --janssen Jan_V1=/nas/gwx/janssen/V1_140.tif \
        --out-dir /data/.../mesilla/validation/fac_v5_arid_gwx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwx_wells import (  # noqa: E402
    DEPTH_BANDS,
    GWX_INDEX,
    WT_CLASSES,
    load_window_wells,
    resid_stats,
    sample_raster,
    tag_setting,
)

log = logging.getLogger("validate_fac_gwx_wells")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fac-rem", required=True)
    p.add_argument("--streams", required=True, help="FAC streams_regional.fgb")
    p.add_argument("--ma", required=True, help="Ma WTD raster (benchmark)")
    p.add_argument(
        "--janssen",
        action="append",
        default=[],
        help="LABEL=path (repeatable), e.g. Jan_V1=/nas/gwx/janssen/V1_140.tif",
    )
    p.add_argument("--gwx-index", default=GWX_INDEX)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--exclude-sources", default="nwis,ngwmn")
    p.add_argument(
        "--include-sources",
        default="",
        help="If set (comma-list), keep ONLY these sources (overrides exclude); "
        "e.g. --include-sources nwis for the NWIS-only tuning set.",
    )
    p.add_argument("--confinement", default=",".join(WT_CLASSES))
    p.add_argument("--valley-dist-m", type=float, default=500.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = {s for s in args.exclude_sources.split(",") if s}
    include = {s for s in args.include_sources.split(",") if s}
    conf = tuple(c for c in args.confinement.split(",") if c)

    with rasterio.open(args.fac_rem) as src:
        b = src.bounds
    bbox = (b.left, b.bottom, b.right, b.top)

    wells = load_window_wells(args.gwx_index, bbox, conf, exclude, include)
    log.info(
        "GWX wells in window: %d (confinement=%s, %s)",
        len(wells),
        conf,
        f"including sources={sorted(include)}"
        if include
        else f"excluding sources={sorted(exclude)}",
    )

    lon = wells["longitude"].to_numpy()
    lat = wells["latitude"].to_numpy()
    preds = {"FAC": args.fac_rem, "Ma": args.ma}
    for spec in args.janssen:
        label, path = spec.split("=", 1)
        preds[label] = path
    for label, path in preds.items():
        wells[f"pred_{label}"] = sample_raster(path, lon, lat)
        wells[f"resid_{label}"] = wells[f"pred_{label}"] - wells["mean_dtw"]

    setting, dist = tag_setting(wells, args.streams, args.valley_dist_m)
    wells["setting"] = setting
    wells["dist_stream_m"] = dist

    # Common footprint: every predictor finite (fair head-to-head).
    finite = np.ones(len(wells), dtype=bool)
    for label in preds:
        finite &= wells[f"pred_{label}"].notna().to_numpy()
    cw = wells.loc[finite].copy()
    log.info("common-footprint set (all predictors finite): %d", len(cw))

    obs = cw["mean_dtw"].to_numpy()
    rows: list[dict] = []

    def emit(group_type: str, group: str, mask: np.ndarray) -> None:
        for label in preds:
            st = resid_stats(cw.loc[mask, f"pred_{label}"].to_numpy(), obs[mask])
            if st:
                rows.append(
                    {"group_type": group_type, "group": group, "predictor": label, **st}
                )

    emit("all", "all", np.ones(len(cw), dtype=bool))
    for s in ("valley", "upland"):
        emit("setting", s, (cw["setting"] == s).to_numpy())
    for wc in sorted(cw["well_class"].dropna().unique()):
        emit("well_class", str(wc), (cw["well_class"] == wc).to_numpy())
    for src_name in sorted(cw["source"].unique()):
        emit("source", str(src_name), (cw["source"] == src_name).to_numpy())
    for lo, hi in DEPTH_BANDS:
        emit(
            "obs_depth", f"{lo}-{hi if hi < 1e9 else 'inf'}m", (obs >= lo) & (obs < hi)
        )

    # Spatial error structure: distance-to-stream and along-valley (northing)
    # bins, to localize where the prior over/under-predicts.
    for lo, hi in ((0, 100), (100, 250), (250, 500), (500, 1000), (1000, 1e9)):
        emit(
            "fac_dist_stream",
            f"{lo}-{hi if hi < 1e9 else 'inf'}m",
            (cw["dist_stream_m"].to_numpy() >= lo)
            & (cw["dist_stream_m"].to_numpy() < hi),
        )
    ybins = np.quantile(cw["y5070"], [0, 0.25, 0.5, 0.75, 1.0])
    for i in range(4):
        mask = (cw["y5070"].to_numpy() >= ybins[i]) & (
            cw["y5070"].to_numpy() <= ybins[i + 1]
        )
        emit("northing_quartile", f"q{i + 1}_S_to_N", mask)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "score_summary.csv"
    summary.to_csv(summary_path, index=False)

    keep_cols = [
        "source",
        "well_class",
        "well_use",
        "obs_count",
        "mean_dtw",
        "confinement_class",
        "confinement_source",
        "setting",
        "dist_stream_m",
        *[f"pred_{label}" for label in preds],
        *[f"resid_{label}" for label in preds],
        "geometry",
    ]
    cw[keep_cols].to_file(out_dir / "fac_well_residuals.fgb", driver="FlatGeobuf")

    run = {
        "fac_rem": args.fac_rem,
        "gwx_index": args.gwx_index,
        "excluded_sources": sorted(exclude),
        "included_sources": sorted(include),
        "confinement_classes": list(conf),
        "n_window": int(len(wells)),
        "n_common_footprint": int(len(cw)),
        "predictors": preds,
        "headline": {
            label: resid_stats(cw[f"pred_{label}"].to_numpy(), obs) for label in preds
        },
    }
    with open(out_dir / "validation_run.json", "w") as f:
        json.dump(run, f, indent=2)

    # Console summary.
    src_note = f"incl {sorted(include)}" if include else f"excl {sorted(exclude)}"
    print(f"\n=== FAC vs benchmarks on {len(cw)} GWX unconfined wells ({src_note}) ===")
    for gt in ("all", "setting", "well_class", "obs_depth", "fac_dist_stream"):
        sl = summary[summary["group_type"] == gt]
        for g in sl["group"].unique():
            line = f"{gt:16} {g:14}"
            for label in preds:
                r = sl[(sl["group"] == g) & (sl["predictor"] == label)]
                if not r.empty:
                    line += f"  {label} MAD={r['mad_m'].iloc[0]:5.2f} bias={r['bias_m'].iloc[0]:+5.2f}"
            print(line)
        print()
    log.info("Wrote %s and fac_well_residuals.fgb", summary_path)


if __name__ == "__main__":
    main()
