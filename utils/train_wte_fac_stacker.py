"""Pilot level-1 stacker: learn DTW from ConusWTE + ConusFAC under blocked CV.

Trains a HistGradientBoosting stacker on the ConusFAC-covered wells of the feature
table (`build_stacker_features.py`) and scores it against its two level-0 inputs and
the national benchmarks, on a common footprint, with the metric panel the project
mandates (MAD + bias + median residual + RMSE + n, stratified).

Why this is leakage-free:
  * ConusWTE (`wte_dtw`) is a frozen surface fit on a RETIRED well partition -- none
    of these wells informed it (exogenous feature).
  * ConusFAC (`fac_rem_dtw_m`) is target-blind terrain (never sees a label).
  * The stacker itself is evaluated out-of-fold under LEAVE-ONE-HUC4-OUT CV, so the
    learned blend is tested on a region whose wells it never trained on. A random
    K-fold stacker is also reported as the spatial-autocorrelation canary (if random
    >> blocked, the blend is memorising regions, not generalising).

Benchmarks (to beat, never tuning targets): Janssen V1 (CONUS) at every well; Ma
per-state (MT/NM/TX) coalesced where available. Janssen/Ma trained on nwis/ngwmn, so
the panel is also reported on the INDEPENDENT footprint (excl nwis/ngwmn) -- the fair
head-to-head. ConusWTE/ConusFAC/stacker are fair on all wells regardless.

Usage:
    uv run python utils/train_wte_fac_stacker.py \
        --features /data/ssd2/handily/conus/stacker/wte_fac_features.parquet \
        --out-dir  /data/ssd2/handily/conus/stacker/stacker_eval
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwx_wells import DEPTH_BANDS, resid_stats, sample_raster  # noqa: E402

log = logging.getLogger("train_wte_fac_stacker")

FEATURES = "/data/ssd2/handily/conus/stacker/wte_fac_features.parquet"
JANSSEN = "/nas/gwx/janssen/V1_140.tif"
# All per-state Ma WTD rasters (CONUS coverage; sample_ma coalesces the first finite
# value per well across states -- states don't overlap, so a well takes its own state's
# value). Globbed so new states are picked up automatically.
MA_STATES = tuple(sorted(glob.glob("/nas/gwx/wtd_states/wtd_*.tif")))
# Feature tiers (mirror notes/CONUS_DTW_STACKER.md). Ablations share ONE footprint
# (the need mask below always requires the FULL set finite), so level0/relief/full are
# compared on an identical well population -- the difference is only which columns the
# HGB sees, never which wells it sees.
LEVEL0 = (
    "wte_dtw",
    "fac_rem_dtw_m",
    "wte_support_dist_m",
    "land_surface_elev_m",
)
RELIEF = LEVEL0 + (
    "elev_above_coarse_m",  # artifact-deep flag: well perched above its 2 km cell mean
    "slope_deg",
    "tri_100m",
    "dist_to_stream_m",
    "log_drainage_area",
)
FULL = RELIEF + (
    "aridity_index",  # real-deep flag: arid basin (low P/PET) -> genuinely deep table
    "mean_annual_precip_mm",
)
# ETRM water-balance fluxes: the dynamic-flux test of the real-deep hypothesis static
# gridMET aridity failed. recharge integrates P-ET-runoff (low recharge -> no water
# reaching the table -> genuinely deep); ETa flags where water is actually consumed
# (shallow/accessible table). NOT in the footprint mask (see FEATURE_COLS), so the well
# population is identical to the no-ETRM run -- ETRM enters only as NaN-tolerant columns
# the HGB may or may not use, isolating its marginal value.
ETRM = (
    "etrm_recharge_mm",
    "etrm_eta_mm",
    "etrm_runoff_mm",
)
RELIEF_ETRM = RELIEF + ETRM
FULL_ETRM = FULL + ETRM
FEATURE_SETS = {
    "level0": LEVEL0,
    "relief": RELIEF,
    "full": FULL,
    "relief_etrm": RELIEF_ETRM,
    "full_etrm": FULL_ETRM,
}
# Superset for the finite-feature footprint mask (kept fixed across ablations so every
# tier is scored on an identical well population). Deliberately FULL, NOT FULL_ETRM:
# requiring ETRM finite would shrink/shift the population and break comparability with
# the prior relief baseline; ETRM is NaN-tolerant in the HGB instead.
FEATURE_COLS = FULL
BENCH_SOURCES = ("nwis", "ngwmn")  # Janssen/Ma training set -> independent footprint
MIN_FOLD_N = 50


def _model() -> HistGradientBoostingRegressor:
    # absolute_error aligns the objective with the headline MAD and is robust to the
    # deep-DTW tail; modest capacity for a ~26k-row pilot.
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=0,
    )


def sample_ma(lon: np.ndarray, lat: np.ndarray, ma_paths) -> np.ndarray:
    """Coalesce per-state Ma rasters: first finite value per well across states."""
    out = np.full(lon.shape, np.nan, dtype=np.float64)
    for path in ma_paths:
        if not Path(path).exists():
            continue
        need = ~np.isfinite(out)
        if not need.any():
            break
        v = sample_raster(path, lon[need], lat[need])
        idx = np.flatnonzero(need)
        out[idx] = v
    return out


def oof_blocked(df: pd.DataFrame, feats, target: str, block_col: str, min_n: int):
    """Out-of-fold stacker predictions under leave-one-block-out CV."""
    blocks = df[block_col].to_numpy()
    x = df[list(feats)].to_numpy()
    y = df[target].to_numpy()
    vc = pd.Series(blocks).value_counts()
    eligible = sorted(vc[vc >= min_n].index)
    keep = pd.Series(blocks).isin(eligible).to_numpy()
    oof = np.full(len(df), np.nan, dtype=np.float64)
    for b in eligible:
        test = (blocks == b) & keep
        train = keep & (blocks != b)
        m = _model().fit(x[train], y[train])
        oof[test] = m.predict(x[test])
    return oof, keep, eligible


def oof_random(df: pd.DataFrame, feats, target: str, k: int, seed: int):
    """Out-of-fold predictions under random K-fold (autocorrelation canary)."""
    x = df[list(feats)].to_numpy()
    y = df[target].to_numpy()
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, len(df))
    oof = np.full(len(df), np.nan, dtype=np.float64)
    for f in range(k):
        test = fold == f
        m = _model().fit(x[~test], y[~test])
        oof[test] = m.predict(x[test])
    return oof


def _panel(df, predictors, footprints, obs_col, fold_col):
    """Emit metric-panel rows: per footprint x group x predictor, common footprint."""
    obs = df[obs_col].to_numpy()
    rows: list[dict] = []

    def emit(fp_label, fp_mask, gt, g, gmask):
        # common footprint across ALL predictors present (fair head-to-head)
        common = fp_mask & gmask & np.isfinite(obs)
        for p in predictors:
            common &= np.isfinite(df[p].to_numpy())
        for p in predictors:
            st = resid_stats(df[p].to_numpy()[common], obs[common])
            if st:
                rows.append(
                    {
                        "footprint": fp_label,
                        "group_type": gt,
                        "group": g,
                        "predictor": p,
                        **st,
                    }
                )

    for fp_label, fp_mask in footprints.items():
        emit(fp_label, fp_mask, "all", "all", np.ones(len(df), dtype=bool))
        for b in sorted(pd.Series(df[fold_col]).dropna().unique()):
            emit(fp_label, fp_mask, "huc4_fold", str(b), (df[fold_col] == b).to_numpy())
        for lo, hi in DEPTH_BANDS:
            lbl = f"{lo}-{hi if hi < 1e9 else 'inf'}m"
            emit(fp_label, fp_mask, "obs_depth", lbl, ((obs >= lo) & (obs < hi)))
    return rows


def build(
    features_path: str,
    out_dir: str,
    *,
    feature_cols=FEATURE_COLS,
    janssen=JANSSEN,
    ma_states=MA_STATES,
    random_k: int = 5,
    seed: int = 0,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(features_path)
    df = df[df.fac_covered].copy()
    # need finite target + the FULL feature superset, so every ablation (level0/relief/
    # full) trains and evals on the IDENTICAL well population (footprint fixed; only the
    # columns fed to the HGB change).
    need = df.mean_dtw.notna()
    for c in FEATURE_COLS:
        need &= df[c].notna()
    df = df[need].reset_index(drop=True)
    log.info(
        "ConusFAC-covered wells with complete features: %d (feature set: %s, %d cols)",
        len(df),
        [k for k, v in FEATURE_SETS.items() if v == tuple(feature_cols)] or ["custom"],
        len(feature_cols),
    )

    lon, lat = df.longitude.to_numpy(), df.latitude.to_numpy()
    df["pred_ConusWTE"] = df.wte_dtw
    df["pred_ConusFAC"] = df.fac_rem_dtw_m
    df["pred_Janssen"] = sample_raster(janssen, lon, lat)
    df["pred_Ma"] = sample_ma(lon, lat, ma_states)
    log.info(
        "Janssen finite: %d | Ma finite: %d / %d",
        int(df.pred_Janssen.notna().sum()),
        int(df.pred_Ma.notna().sum()),
        len(df),
    )

    oof, keep, eligible = oof_blocked(df, feature_cols, "mean_dtw", "huc4", MIN_FOLD_N)
    df["pred_Stacker"] = oof
    df["pred_Stacker_rand"] = oof_random(df, feature_cols, "mean_dtw", random_k, seed)
    n_drop = int((~keep).sum())
    log.info(
        "blocked folds (huc4 >= %d): %s | dropped %d wells in tiny blocks",
        MIN_FOLD_N,
        eligible,
        n_drop,
    )

    # Restrict the panel to fold-eligible wells (those with an OOF blocked prediction).
    ev = df[keep].reset_index(drop=True)
    indep = ~ev.source.isin(BENCH_SOURCES)
    footprints = {"all": np.ones(len(ev), dtype=bool), "independent": indep.to_numpy()}
    log.info(
        "eval set: %d (independent excl %s: %d)",
        len(ev),
        BENCH_SOURCES,
        int(indep.sum()),
    )

    main_preds = ["pred_Stacker", "pred_ConusWTE", "pred_ConusFAC", "pred_Janssen"]
    rows = _panel(ev, main_preds, footprints, "mean_dtw", "huc4")
    # Ma sub-panel: only where Ma exists (MT/NM regimes), include Ma in the lineup.
    ma_mask = ev.pred_Ma.notna().to_numpy()
    ev_ma = ev[ma_mask].reset_index(drop=True)
    if len(ev_ma):
        indep_ma = (~ev_ma.source.isin(BENCH_SOURCES)).to_numpy()
        fps_ma = {"all": np.ones(len(ev_ma), dtype=bool), "independent": indep_ma}
        ma_preds = main_preds + ["pred_Ma"]
        for r in _panel(ev_ma, ma_preds, fps_ma, "mean_dtw", "huc4"):
            r["group_type"] = "ma_" + r["group_type"]
            rows.append(r)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "score_summary.csv", index=False)
    keep_cols = [
        "canonical_id",
        "source",
        "huc4",
        "huc8",
        "longitude",
        "latitude",
        "mean_dtw",
        *[
            f"pred_{p}"
            for p in (
                "ConusWTE",
                "ConusFAC",
                "Janssen",
                "Ma",
                "Stacker",
                "Stacker_rand",
            )
        ],
        "wte_support_dist_m",
    ]
    ev[keep_cols].to_parquet(out / "well_predictions.parquet", index=False)

    # Final stacker on all eligible wells, persisted for inference once ConusFAC is
    # CONUS-wide (feature order = FEATURE_COLS).
    import joblib

    final = _model().fit(ev[list(feature_cols)].to_numpy(), ev.mean_dtw.to_numpy())
    joblib.dump(
        {"model": final, "feature_cols": list(feature_cols)}, out / "stacker_hgb.joblib"
    )
    run = {
        "n_eval": int(len(ev)),
        "n_independent": int(indep.sum()),
        "feature_cols": list(feature_cols),
        "blocked_folds": list(map(str, eligible)),
        "n_dropped_tiny_blocks": n_drop,
        "feature_importance_permutation": None,
        "model": "HistGradientBoostingRegressor(absolute_error)",
        "headline": {
            fp: {
                p: resid_stats(
                    ev.loc[ev_mask, p].to_numpy(),
                    ev.loc[ev_mask, "mean_dtw"].to_numpy(),
                )
                for p in main_preds
            }
            for fp, ev_mask in {
                "all": np.ones(len(ev), bool),
                "independent": indep.to_numpy(),
            }.items()
        },
    }
    with open(out / "train_run.json", "w") as f:
        json.dump(run, f, indent=2, default=float)

    # Console summary.
    print("\n=== Pilot stacker: leave-one-HUC4-out vs level-0 + benchmarks ===")
    print(f"eval wells {len(ev)} (independent {int(indep.sum())}); folds {eligible}\n")
    s = summary[(summary.group_type == "all") & (summary.footprint == "all")]
    si = summary[(summary.group_type == "all") & (summary.footprint == "independent")]
    print(
        f"{'predictor':14} {'MAD':>6} {'bias':>6} {'med':>6} {'RMSE':>7} {'n':>6}   "
        f"|  independent MAD/bias"
    )
    for p in main_preds:
        r = s[s.predictor == p]
        ri = si[si.predictor == p]
        if not r.empty:
            print(
                f"{p.replace('pred_', ''):14} {r.mad_m.iloc[0]:6.2f} "
                f"{r.bias_m.iloc[0]:+6.2f} {r.median_residual_m.iloc[0]:+6.2f} "
                f"{r.rmse_m.iloc[0]:7.2f} {int(r.n.iloc[0]):6d}   |  "
                f"{ri.mad_m.iloc[0]:5.2f} {ri.bias_m.iloc[0]:+5.2f}"
            )
    # canary
    cst = resid_stats(ev.pred_Stacker.to_numpy(), ev.mean_dtw.to_numpy())
    crd = resid_stats(ev.pred_Stacker_rand.to_numpy(), ev.mean_dtw.to_numpy())
    print(
        f"\ncanary  blocked-CV stacker MAD {cst['mad_m']:.2f}  vs  "
        f"random-CV {crd['mad_m']:.2f}  "
        f"(gap = region-transfer cost)"
    )
    print("\nper-HUC4-fold (footprint=all), MAD:")
    pf = summary[(summary.group_type == "huc4_fold") & (summary.footprint == "all")]
    for g in sorted(pf.group.unique()):
        line = f"  {g:8}"
        for p in main_preds:
            r = pf[(pf.group == g) & (pf.predictor == p)]
            if not r.empty:
                line += f"  {p.replace('pred_', ''):8}={r.mad_m.iloc[0]:5.2f}"
        print(line)
    log.info("wrote %s", out / "score_summary.csv")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default=FEATURES)
    p.add_argument("--janssen", default=JANSSEN)
    p.add_argument("--out-dir", default="/data/ssd2/handily/conus/stacker/stacker_eval")
    p.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="relief",
        help="feature tier fed to the HGB (footprint fixed across tiers): "
        "level0 | relief (default; aridity twice shown NEGATIVE) | full (+aridity expt)",
    )
    p.add_argument("--random-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    build(
        args.features,
        args.out_dir,
        feature_cols=FEATURE_SETS[args.feature_set],
        janssen=args.janssen,
        random_k=args.random_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
