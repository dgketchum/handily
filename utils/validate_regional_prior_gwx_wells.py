"""Validate the regional WTE prior on independent (non-NWIS) GWX wells.

Mirrors ``validate_fac_gwx_wells.py`` but for the well-interpolated regional
water-table prior (``build_wte_regional_prior.py``). Because the prior is a point
model (no native raster), it is scored with **blocked-CV out-of-fold**
predictions, not raster sampling: spatial blocks (``block_20km``, the same
integer-divide as ``build_wte_covariate_table.py``) are held out under GroupKFold
so every scored well is predicted from spatially separated training wells.

Design (see notes/plans/regional_prior_gwx_revalidation.md):
- Blocked CV over ALL unconfined GWX wells in the window. The per-fold training
  pool INCLUDES NWIS -- NWIS are legitimate water-table observations; the rule is
  never tune to Ma, not never use NWIS. Ma is benchmark-only, never an input.
- HEADLINE scored on the non-NWIS (independent) held-out wells, with FAC + Ma
  sampled at the same wells -> clean, leakage-free comparison vs Ma.
- SECONDARY scored on NWIS held-out wells for deep-regime visibility (the
  independent set is shallow-dominated), with Ma FLAGGED contaminated (Ma trained
  on NWIS) -- read FAC vs regional-prior there, not Ma.

WTE residual == -(DTW residual) because obs_wte = dem - obs_dtw and
pred_wte = dem - pred_dtw share the per-well DEM; metrics are reported in DTW
space and the equivalence is recorded in the run json.

Example:
    uv run python utils/validate_regional_prior_gwx_wells.py \\
        --fac-rem /data/ssd2/handily/nm/regional/rio_grande_albuquerque/rem/nm_rga_v5_arid_full/fac_head_depth_rem_10m.tif \\
        --dem    /data/ssd2/handily/nm/regional/rio_grande_albuquerque/dem_10m.tif \\
        --streams /data/ssd2/handily/nm/regional/rio_grande_albuquerque/streams_regional.fgb \\
        --ma /nas/gwx/wtd_states/wtd_new_mexico.tif \\
        --out-dir /data/ssd2/handily/nm/regional/rio_grande_albuquerque/hybrid/regional_prior/gwx_validation
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
from build_wte_regional_prior import (  # noqa: E402
    _grouped_folds,
    parse_method_name,
    rbf_predict,
    regional_fit_predict,
)
from gwx_wells import (  # noqa: E402
    DEPTH_BANDS,
    GWX_INDEX,
    WT_CLASSES,
    load_window_wells,
    resid_stats,
    sample_raster,
    tag_setting,
)

log = logging.getLogger("validate_regional_prior_gwx_wells")

DEFAULT_METHODS = "rbf_tps_s25,idw_k32_p2"


def assign_blocks(x5070: np.ndarray, y5070: np.ndarray, block_m: int) -> np.ndarray:
    """block_{km} label, identical to build_wte_covariate_table.py."""
    bx = (x5070 // block_m).astype("int64").astype(str)
    by = (y5070 // block_m).astype("int64").astype(str)
    return np.char.add(np.char.add(bx, "_"), by)


def crossfit_oof_dtw(
    wells: pd.DataFrame, blocks: np.ndarray, spec, n_splits: int, rbf_neighbors: int = 0
) -> np.ndarray:
    """Out-of-fold regional DTW prediction (dem - OOF WTE, clipped >=0).

    Each held-out spatial block is predicted from training wells in the OTHER
    blocks only -- no same-block (and hence no near-coincident) leakage.
    ``rbf_neighbors`` > 0 switches RBF to a local (k-neighbor) solve so large
    training folds don't trigger a dense O(N^3) kernel solve; 0 keeps the dense
    fit the builder's recommended method was tuned with.
    """
    folds = _grouped_folds(blocks, n_splits_max=n_splits)
    xy = wells[["x5070", "y5070"]].to_numpy("float64")
    ywte = wells["obs_wte_m"].to_numpy("float64")
    w = np.ones(len(wells))
    use_local_rbf = spec.family == "rbf" and rbf_neighbors > 0
    pred_wte = np.full(len(wells), np.nan)
    for tr, te in folds:
        if use_local_rbf:
            pred_wte[te] = rbf_predict(
                xy[tr],
                ywte[tr],
                xy[te],
                kernel=spec.params["kernel"],
                smoothing=spec.params["smoothing"],
                neighbors=min(rbf_neighbors, len(tr)),
            )
        else:
            pred_wte[te] = regional_fit_predict(xy[tr], ywte[tr], w[tr], xy[te], spec)
    return np.clip(wells["dem_m"].to_numpy("float64") - pred_wte, 0, None)


def emit_panel(
    cw: pd.DataFrame, preds: list[str], group_type: str, group: str, mask: np.ndarray
) -> list[dict]:
    obs = cw["mean_dtw"].to_numpy("float64")
    rows = []
    for label in preds:
        st = resid_stats(cw.loc[mask, f"pred_{label}"].to_numpy(), obs[mask])
        if st:
            rows.append(
                {"group_type": group_type, "group": group, "predictor": label, **st}
            )
    return rows


def score_scope(
    cw: pd.DataFrame, preds: list[str], valley_dist_m: float
) -> pd.DataFrame:
    """Full metric panel for one scope (already common-footprint subset)."""
    obs = cw["mean_dtw"].to_numpy("float64")
    rows: list[dict] = []
    rows += emit_panel(cw, preds, "all", "all", np.ones(len(cw), bool))
    for s in ("valley", "upland"):
        rows += emit_panel(cw, preds, "setting", s, (cw["setting"] == s).to_numpy())
    for wc in sorted(cw["well_class"].dropna().unique()):
        rows += emit_panel(
            cw, preds, "well_class", str(wc), (cw["well_class"] == wc).to_numpy()
        )
    for src in sorted(cw["source"].unique()):
        rows += emit_panel(
            cw, preds, "source", str(src), (cw["source"] == src).to_numpy()
        )
    for lo, hi in DEPTH_BANDS:
        lab = f"{lo}-{hi if hi < 1e9 else 'inf'}m"
        rows += emit_panel(cw, preds, "obs_depth", lab, (obs >= lo) & (obs < hi))
    d = cw["dist_stream_m"].to_numpy("float64")
    for lo, hi in ((0, 100), (100, 250), (250, 500), (500, 1000), (1000, 1e9)):
        lab = f"{lo}-{hi if hi < 1e9 else 'inf'}m"
        rows += emit_panel(cw, preds, "dist_stream", lab, (d >= lo) & (d < hi))
    y = cw["y5070"].to_numpy("float64")
    ybins = np.quantile(y, [0, 0.25, 0.5, 0.75, 1.0])
    for i in range(4):
        rows += emit_panel(
            cw,
            preds,
            "northing_quartile",
            f"q{i + 1}_S_to_N",
            (y >= ybins[i]) & (y <= ybins[i + 1]),
        )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fac-rem", required=True, help="FAC REM raster (predicted DTW)")
    p.add_argument(
        "--dem", required=True, help="3DEP ground-elevation raster (WTE = dem - dtw)"
    )
    p.add_argument("--streams", required=True, help="FAC streams_regional.fgb")
    p.add_argument("--ma", required=True, help="Ma WTD raster (benchmark)")
    p.add_argument("--gwx-index", default=GWX_INDEX)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--exclude-sources", default="nwis,ngwmn")
    p.add_argument("--confinement", default=",".join(WT_CLASSES))
    p.add_argument("--valley-dist-m", type=float, default=500.0)
    p.add_argument("--methods", default=DEFAULT_METHODS, help="comma method names")
    p.add_argument("--block-size-m", type=int, default=20000)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument(
        "--rbf-neighbors",
        type=int,
        default=0,
        help="local RBF k-neighbors (0=dense fit; set >0 for large well counts)",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = {s for s in args.exclude_sources.split(",") if s}
    conf = tuple(c for c in args.confinement.split(",") if c)
    specs = [s for s in (parse_method_name(n) for n in args.methods.split(",")) if s]
    if not specs:
        raise SystemExit(f"no parseable methods in {args.methods!r}")

    with rasterio.open(args.fac_rem) as src:
        b = src.bounds
    bbox = (b.left, b.bottom, b.right, b.top)

    # Independent (non-NWIS) + NWIS pools, combined for blocked-CV training.
    indep = load_window_wells(args.gwx_index, bbox, conf, exclude, None)
    indep["pool"] = "independent"
    nwis = load_window_wells(args.gwx_index, bbox, conf, set(), exclude)
    nwis["pool"] = "nwis"
    wells = pd.concat([indep, nwis], ignore_index=True)
    wells = wells.set_geometry("geometry")
    log.info(
        "window wells: independent=%d nwis=%d total=%d",
        len(indep),
        len(nwis),
        len(wells),
    )

    # WTE label: dem - dtw. Off-grid wells (no DEM) are dropped, not patched.
    lon, lat = wells["longitude"].to_numpy(), wells["latitude"].to_numpy()
    wells["dem_m"] = sample_raster(args.dem, lon, lat)
    n_nodem = int(wells["dem_m"].isna().sum())
    if n_nodem:
        log.info("dropping %d wells with no DEM coverage (off-grid)", n_nodem)
    wells = wells[wells["dem_m"].notna()].reset_index(drop=True)
    wells["obs_wte_m"] = wells["dem_m"] - wells["mean_dtw"]

    blocks = assign_blocks(
        wells["x5070"].to_numpy(), wells["y5070"].to_numpy(), args.block_size_m
    )
    n_blocks = int(np.unique(blocks).size)
    log.info("spatial blocks (%dkm): %d", args.block_size_m // 1000, n_blocks)

    # Predictors: rasters (FAC, Ma) + blocked-CV OOF regional methods.
    lon, lat = wells["longitude"].to_numpy(), wells["latitude"].to_numpy()
    wells["pred_FAC"] = sample_raster(args.fac_rem, lon, lat)
    wells["pred_Ma"] = sample_raster(args.ma, lon, lat)
    reg_labels = []
    for spec in specs:
        label = f"Reg_{spec.name}"
        wells[f"pred_{label}"] = crossfit_oof_dtw(
            wells, blocks, spec, args.n_splits, args.rbf_neighbors
        )
        reg_labels.append(label)
    preds = ["FAC", "Ma", *reg_labels]
    for label in preds:
        wells[f"resid_{label}"] = wells[f"pred_{label}"] - wells["mean_dtw"]

    setting, dist = tag_setting(wells, args.streams, args.valley_dist_m)
    wells["setting"] = setting
    wells["dist_stream_m"] = dist

    # Common footprint: every predictor finite (fair head-to-head).
    finite = np.ones(len(wells), bool)
    for label in preds:
        finite &= wells[f"pred_{label}"].notna().to_numpy()
    cw = wells.loc[finite].copy()
    log.info("common-footprint wells (all predictors finite): %d", len(cw))

    indep_cw = cw[cw["pool"] == "independent"].copy()
    nwis_cw = cw[cw["pool"] == "nwis"].copy()

    sum_indep = score_scope(indep_cw, preds, args.valley_dist_m)
    sum_nwis = (
        score_scope(nwis_cw, preds, args.valley_dist_m)
        if len(nwis_cw)
        else pd.DataFrame()
    )
    sum_indep.to_csv(out_dir / "score_summary_independent.csv", index=False)
    if not sum_nwis.empty:
        sum_nwis.to_csv(out_dir / "score_summary_nwis.csv", index=False)

    keep = [
        "pool",
        "source",
        "well_class",
        "well_use",
        "obs_count",
        "mean_dtw",
        "dem_m",
        "confinement_class",
        "confinement_source",
        "setting",
        "dist_stream_m",
        *[f"pred_{label}" for label in preds],
        *[f"resid_{label}" for label in preds],
        "geometry",
    ]
    cw[keep].to_file(out_dir / "regional_prior_well_residuals.fgb", driver="FlatGeobuf")

    def headline(sub: pd.DataFrame) -> dict:
        o = sub["mean_dtw"].to_numpy("float64")
        return {
            label: resid_stats(sub[f"pred_{label}"].to_numpy(), o) for label in preds
        }

    run = {
        "fac_rem": args.fac_rem,
        "dem": args.dem,
        "ma": args.ma,
        "gwx_index": args.gwx_index,
        "excluded_sources": sorted(exclude),
        "confinement_classes": list(conf),
        "methods": [s.name for s in specs],
        "block_size_m": args.block_size_m,
        "n_blocks": n_blocks,
        "n_splits": min(args.n_splits, n_blocks),
        "n_window_independent": int(len(indep)),
        "n_window_nwis": int(len(nwis)),
        "n_dropped_no_dem": n_nodem,
        "n_common_independent": int(len(indep_cw)),
        "n_common_nwis": int(len(nwis_cw)),
        "scoring": "blocked-CV out-of-fold (train pool incl NWIS); Ma benchmark-only",
        "wte_residual_note": "WTE residual == -(DTW residual); metrics in DTW space",
        "caveats": [
            "Independent (non-NWIS) set is shallow-dominated; deep-band n is small "
            "-- read the deep regime on the NWIS panel.",
            "On the NWIS panel Ma is leakage-inflated (Ma trained on NWIS); compare "
            "FAC vs regional-prior there, not Ma.",
        ],
        "headline_independent": headline(indep_cw),
        "secondary_nwis_Ma_contaminated": headline(nwis_cw) if len(nwis_cw) else None,
    }
    with open(out_dir / "validation_run.json", "w") as f:
        json.dump(run, f, indent=2)

    print(
        f"\n=== regional prior vs FAC/Ma on INDEPENDENT (non-NWIS) wells: n={len(indep_cw)} ==="
    )
    for gt in ("all", "setting", "obs_depth", "dist_stream"):
        sl = sum_indep[sum_indep["group_type"] == gt]
        for g in sl["group"].unique():
            line = f"{gt:16} {g:12}"
            for label in preds:
                r = sl[(sl["group"] == g) & (sl["predictor"] == label)]
                if not r.empty:
                    line += f"  {label}:MAD={r['mad_m'].iloc[0]:5.2f}/n={int(r['n'].iloc[0])}"
            print(line)
        print()
    if not sum_nwis.empty:
        sl = sum_nwis[sum_nwis["group_type"] == "all"]
        print(
            f"=== NWIS panel (Ma CONTAMINATED -- FAC vs Reg only): n={len(nwis_cw)} ==="
        )
        for label in preds:
            r = sl[sl["predictor"] == label]
            if not r.empty:
                print(
                    f"  {label:18} MAD={r['mad_m'].iloc[0]:6.2f}  bias={r['bias_m'].iloc[0]:+7.2f}  RMSE={r['rmse_m'].iloc[0]:6.2f}"
                )
    log.info(
        "wrote score_summary_*.csv, regional_prior_well_residuals.fgb, validation_run.json -> %s",
        out_dir,
    )


if __name__ == "__main__":
    main()
