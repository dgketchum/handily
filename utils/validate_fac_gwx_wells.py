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
    SW_DIST_BANDS,
    WT_CLASSES,
    load_window_wells,
    resid_stats,
    sample_raster,
    shallow_skill,
    surface_water_distance,
    tag_setting,
)

log = logging.getLogger("validate_fac_gwx_wells")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fac-rem", required=True)
    p.add_argument(
        "--pred-label",
        default="FAC",
        help="label for the --fac-rem predictor under test (e.g. WTE for the "
        "regional water-table surface); names its pred_/resid_ columns and the "
        "<label>_well_residuals.fgb output",
    )
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
    p.add_argument(
        "--holdout-oof",
        default=None,
        help="OOF predictions parquet (x5070/y5070). Wells co-located with a "
        "training well (exact 1 m 5070 key) are DROPPED, giving the fair "
        "spatial-holdout sub-panel for an inference mosaic the model saw part of "
        "at training. Applied AFTER confinement/source screening.",
    )
    p.add_argument("--exclude-sources", default="nwis,ngwmn")
    p.add_argument(
        "--include-sources",
        default="",
        help="If set (comma-list), keep ONLY these sources (overrides exclude); "
        "e.g. --include-sources nwis for the NWIS-only tuning set.",
    )
    p.add_argument("--confinement", default=",".join(WT_CLASSES))
    p.add_argument("--valley-dist-m", type=float, default=500.0)
    p.add_argument(
        "--surface-water",
        default=None,
        help="binary surface-water raster (e.g. JRC-GSW permanent water). Enables the "
        "sw_dist stratification — the regional-prior (R) test: bands of horizontal "
        "distance from surface water, where the FAR bands judge how well a predictor "
        "expresses the regional (drainage-decoupled) water table.",
    )
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

    if args.holdout_oof:
        oof = pd.read_parquet(args.holdout_oof, columns=["x5070", "y5070"])

        def _key(x, y):
            xi = np.round(np.asarray(x, "float64")).astype("int64")
            yi = np.round(np.asarray(y, "float64")).astype("int64")
            return np.char.add(np.char.add(xi.astype(str), "_"), yi.astype(str))

        train_keys = set(_key(oof["x5070"].to_numpy(), oof["y5070"].to_numpy()))
        wk = _key(wells["x5070"].to_numpy(), wells["y5070"].to_numpy())
        keep = ~pd.Series(wk, index=wells.index).isin(train_keys).to_numpy()
        n0 = len(wells)
        wells = wells.loc[keep].copy()
        log.info(
            "holdout: dropped %d/%d wells co-located with a training well -> %d held-out",
            n0 - len(wells),
            n0,
            len(wells),
        )

    lon = wells["longitude"].to_numpy()
    lat = wells["latitude"].to_numpy()
    preds = {args.pred_label: args.fac_rem, "Ma": args.ma}
    for spec in args.janssen:
        label, path = spec.split("=", 1)
        preds[label] = path
    for label, path in preds.items():
        wells[f"pred_{label}"] = sample_raster(path, lon, lat)
        wells[f"resid_{label}"] = wells[f"pred_{label}"] - wells["mean_dtw"]

    setting, dist = tag_setting(wells, args.streams, args.valley_dist_m)
    wells["setting"] = setting
    wells["dist_stream_m"] = dist
    if args.surface_water:
        wells["sw_dist_m"] = surface_water_distance(wells, args.surface_water)
        log.info(
            "surface-water distance: median %.0f m, %.0f%% within 2 km",
            np.nanmedian(wells["sw_dist_m"]),
            100 * np.nanmean(wells["sw_dist_m"].to_numpy() < 2000),
        )

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

    # Regional-prior (R) test: distance from surface water. Far bands decide whether
    # a predictor expresses the drainage-decoupled regional table (near bands are
    # non-diagnostic — the channel bed is the answer there).
    if args.surface_water:
        swd = cw["sw_dist_m"].to_numpy()
        for lo, hi in SW_DIST_BANDS:
            lab = f"{lo / 1000:g}-{hi / 1000:g}km" if hi < 1e9 else f"{lo / 1000:g}+km"
            emit("sw_dist", lab, (swd >= lo) & (swd < hi))

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "score_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Depth-specific accuracy is the CENTRAL accuracy statement, not the aggregate.
    # The aggregate MAD is dominated by whichever depth band is most populous (in
    # arid NM that is the 30+ m tail), so it hides the shallow-prior crossover the
    # product is actually for. Ship the full per-band panel (n + its share of the
    # population, MAD, bias AND medR, RMSE, p95, catastrophic-miss fractions) for
    # every predictor as a first-class block of the accuracy assessment.
    def _band_label(lo: float, hi: float) -> str:
        return f"{lo:g}-{hi:g}m" if hi < 1e9 else f"{lo:g}+m"

    n_cf = len(cw)
    by_depth_band: dict[str, dict] = {}
    for label in preds:
        pred = cw[f"pred_{label}"].to_numpy()
        bands: dict[str, dict] = {}
        for lo, hi in DEPTH_BANDS:
            m = (obs >= lo) & (obs < hi)
            st = resid_stats(pred[m], obs[m])
            if st is None:
                st = {"n": 0}
            st["frac_of_wells"] = float(m.sum()) / n_cf if n_cf else float("nan")
            bands[_band_label(lo, hi)] = st
        by_depth_band[label] = bands

    # Shallow-class precision/recall (<2/<5/<10 m) per predictor, on the common
    # footprint and split valley/upland -- the shallow water-table call is the
    # GW-subsidy use case, and MAD alone hides the precision/recall trade
    # (a shallow prior that over-calls shallow has high recall but low precision).
    shallow_pr: dict[str, dict] = {}
    for scope, mask in [
        ("all", np.ones(len(cw), dtype=bool)),
        ("valley", (cw["setting"] == "valley").to_numpy()),
        ("upland", (cw["setting"] == "upland").to_numpy()),
    ]:
        if mask.sum() < 25:
            continue
        shallow_pr[scope] = {
            label: shallow_skill(cw.loc[mask, f"pred_{label}"].to_numpy(), obs[mask])
            for label in preds
        }

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
        *(["sw_dist_m"] if args.surface_water else []),
        *[f"pred_{label}" for label in preds],
        *[f"resid_{label}" for label in preds],
        "geometry",
    ]
    resid_name = f"{args.pred_label.lower()}_well_residuals.fgb"
    cw[keep_cols].to_file(out_dir / resid_name, driver="FlatGeobuf")

    run = {
        "fac_rem": args.fac_rem,
        "gwx_index": args.gwx_index,
        "excluded_sources": sorted(exclude),
        "included_sources": sorted(include),
        "confinement_classes": list(conf),
        "holdout_oof": args.holdout_oof,
        "n_window": int(len(wells)),
        "n_common_footprint": int(len(cw)),
        "predictors": preds,
        "headline": {
            label: resid_stats(cw[f"pred_{label}"].to_numpy(), obs) for label in preds
        },
        "by_depth_band": by_depth_band,
        "shallow_pr": shallow_pr,
    }
    with open(out_dir / "validation_run.json", "w") as f:
        json.dump(run, f, indent=2)

    # Console summary.
    src_note = f"incl {sorted(include)}" if include else f"excl {sorted(exclude)}"
    print(
        f"\n=== {args.pred_label} vs benchmarks on {len(cw)} GWX unconfined wells "
        f"({src_note}) ==="
    )
    console_groups = ["all", "setting", "well_class", "fac_dist_stream"]
    if args.surface_water:
        console_groups.append("sw_dist")
    for gt in console_groups:
        sl = summary[summary["group_type"] == gt]
        for g in sl["group"].unique():
            line = f"{gt:16} {g:14}"
            for label in preds:
                r = sl[(sl["group"] == g) & (sl["predictor"] == label)]
                if not r.empty:
                    line += f"  {label} MAD={r['mad_m'].iloc[0]:5.2f} bias={r['bias_m'].iloc[0]:+5.2f}"
            print(line)
        print()

    # Depth-specific accuracy panel -- the full metric set per band, the accuracy
    # statement the product is judged on (never MAD alone). One block per band so
    # the shallow-prior crossover and the deep-tail spread are both visible.
    print("=== depth-specific accuracy (obs-depth banded, common footprint) ===")
    first_pred = next(iter(preds))
    for lo, hi in DEPTH_BANDS:
        band = _band_label(lo, hi)
        n = int(by_depth_band[first_pred][band]["n"])
        share = by_depth_band[first_pred][band].get("frac_of_wells", 0.0)
        print(f"-- {band}  (n={n}, {100 * share:.0f}% of wells) --")
        for label in preds:
            st = by_depth_band[label][band]
            if not st.get("n"):
                continue
            print(
                f"   {label:10} MAD={st['mad_m']:6.2f} bias={st['bias_m']:+7.2f} "
                f"medR={st['median_residual_m']:+7.2f} RMSE={st['rmse_m']:6.2f} "
                f"p95={st['p95_abs_err_m']:7.2f} f>10m={st['frac_abs_err_gt_10m']:.2f}"
            )
    print()

    if "all" in shallow_pr:
        print("=== shallow-class precision/recall (common footprint, 'all') ===")
        for label in preds:
            sk = shallow_pr["all"][label]
            cells = " ".join(
                f"{thr}:P{v['precision']:.2f}/R{v['recall']:.2f}"
                for thr, v in sk.items()
            )
            print(f"  {label:12} {cells}")
        print()
    log.info("Wrote %s and %s", summary_path, resid_name)


if __name__ == "__main__":
    main()
