"""Score the MAE neighborhood-embedding GNN arm against the frozen production
baseline under the frozen v0.2 evaluation contract.

Companion to ``utils/train_conus_gnn.py --mae-embeddings`` (the additive per-query
head slot wired in ``utils/train_wte_gnn.py``) and ``utils/extract_mae_embeddings.py``
/ ``utils/train_neighborhood_mae.py`` (the self-supervised pretrain that produces the
embedding). The falsifiable question this answers: does the learned neighborhood
embedding carry DTW skill BEYOND the same rasters point-sampled at the pixel once it
rides the FULL production feature set (gate + mirror + sigma), or does the win seen in
the isolated HistGBT probe wash out inside the GNN?

Discipline (matches every other v0.2 scorer):
- real wells only (is_water_pseudo == False) with finite obs_dtw_m;
- UNCONFINED water-table wells only (confinement_class in {unconfined,
  unconfined_marginal}); confined / likely_confined wells measure a potentiometric
  surface and never enter accuracy metrics (they are already absent from this bundle,
  but the filter is asserted, not assumed);
- sacrificial regional-holdout HUC4s (0707, 1019, 1605) dropped from EVERY report;
- the MAE arm and the frozen baseline are compared on the SAME common footprint (both
  arms + obs finite), joined on canonical_id, so paired bootstrap CIs are honest;
- no Ma / Janssen columns enter any metric (benchmarks only, scored elsewhere);
- full panel, never MAD alone: MAD + mean bias + median residual + RMSE, depth-banded
  (0-2/2-5/5-10/10-30/30+ m) with n per cell, plus the NV closed-basin slice (HUC2=16)
  and the wide-basin slice (HUC2 in 13/15/16) for continuity with the probe round;
- units on every number (m unless noted dimensionless).

Usage:
    uv run python utils/v02_score_mae.py \\
        --arm-dir /data/ssd2/handily/conus/wte_gnn/gnn_conus_monitoring_water_mae_pyrwide_pilot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_conus_gnn import core_metrics, depth_banded, shallow_skill  # noqa: E402
from v02_metrics import paired_improvement  # noqa: E402

log = logging.getLogger("v02_score_mae")

WTE_GNN = "/data/ssd2/handily/conus/wte_gnn"
BASELINE = f"{WTE_GNN}/v02/contract/frozen_baseline/gnn_oof_predictions.parquet"
PANELS = f"{WTE_GNN}/v02/contract/wells_panels.parquet"

SACRIFICIAL_HUC4 = ("0707", "1019", "1605")
UNCONFINED = ("unconfined", "unconfined_marginal")
NV_HUC2 = ("16",)  # Great Basin closed basins (score_conus_gnn NV sub-panel literal)
WIDE_BASIN_HUC2 = ("13", "15", "16")  # Rio Grande/NM rift, Lower Colorado, Great Basin
DEPTH_BANDS = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
MIN_SLICE_N = 50  # match score_conus_gnn's >=50-well guard for sub-panels


def _round(obj, nd: int = 4):
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), nd)
    return obj


def _band_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"


def _band_paired(pred_m: np.ndarray, pred_b: np.ndarray, obs: np.ndarray) -> dict:
    """Per depth-band paired MAD/RMSE reduction of the MAE arm over the baseline on
    the SAME rows (positive skill => MAE arm better)."""
    out = {}
    for lo, hi in DEPTH_BANDS:
        sel = np.isfinite(obs) & (obs >= lo) & (obs < hi)
        out[_band_label(lo, hi)] = paired_improvement(
            (pred_m - obs)[sel], (pred_b - obs)[sel]
        )
    return out


def slice_panel(df: pd.DataFrame, mask: np.ndarray, name: str) -> dict:
    """Full panel for one population slice: both arms' core + banded metrics + shallow
    P/R, overall paired improvement, per-band paired reductions."""
    sub = df[mask]
    obs = sub["obs_dtw_m"].to_numpy(float)
    mae = sub["mae_dtw_m"].to_numpy(float)
    base = sub["base_dtw_m"].to_numpy(float)
    return {
        "panel": name,
        "n": int(mask.sum()),
        "mae_arm": {
            "overall": core_metrics(mae, obs),
            "by_depth_band": depth_banded(mae, obs),
            "shallow_skill": shallow_skill(mae, obs),
        },
        "baseline": {
            "overall": core_metrics(base, obs),
            "by_depth_band": depth_banded(base, obs),
            "shallow_skill": shallow_skill(base, obs),
        },
        "paired_overall": paired_improvement(mae - obs, base - obs),
        "paired_by_depth_band": _band_paired(mae, base, obs),
    }


def load_and_join(args) -> tuple[pd.DataFrame, dict]:
    """Scored unconfined-well frame on the common footprint. Sacrificial HUC4s and
    confined wells dropped; join diagnostics returned."""
    oof = pd.read_parquet(args.arm_oof)
    if "gnn_dtw_m" not in oof.columns:
        raise SystemExit(f"{args.arm_oof} lacks gnn_dtw_m")
    base = pd.read_parquet(
        args.baseline, columns=["canonical_id", "is_water_pseudo", "gnn_dtw_m"]
    ).rename(columns={"gnn_dtw_m": "base_dtw_m"})
    panels = pd.read_parquet(
        args.panels,
        columns=["canonical_id", "is_water_pseudo", "huc4", "confinement_class"],
    )

    oof["huc2"] = oof["huc2"].astype(str).str.zfill(2)
    is_water = oof["is_water_pseudo"].astype(bool)
    real = oof[~is_water].copy().rename(columns={"gnn_dtw_m": "mae_dtw_m"})
    n_real = len(real)
    if real["canonical_id"].duplicated().any():
        raise ValueError("canonical_id not unique among real MAE-arm wells")

    base_real = base[~base["is_water_pseudo"].astype(bool)].drop(
        columns=["is_water_pseudo"]
    )
    panels_real = panels[~panels["is_water_pseudo"].astype(bool)].drop(
        columns=["is_water_pseudo"]
    )
    df = real.merge(base_real, on="canonical_id", how="left", validate="one_to_one")
    df = df.merge(panels_real, on="canonical_id", how="left", validate="one_to_one")

    n_missing_base = int(df["base_dtw_m"].isna().sum())
    n_missing_panel = int(df["huc4"].isna().sum())

    fin_obs = np.isfinite(df["obs_dtw_m"].to_numpy(float))
    sac = df["huc4"].isin(SACRIFICIAL_HUC4).to_numpy()
    unconf = df["confinement_class"].isin(UNCONFINED).to_numpy()
    n_confined_dropped = int((fin_obs & ~sac & ~unconf).sum())
    keep = fin_obs & ~sac & unconf
    df = df[keep].reset_index(drop=True)

    diag = {
        "arm_oof": args.arm_oof,
        "baseline": args.baseline,
        "n_oof_rows": int(len(oof)),
        "n_real_wells": int(n_real),
        "n_real_finite_obs": int(fin_obs.sum()),
        "n_sacrificial_dropped": int((fin_obs & sac).sum()),
        "n_confined_dropped": n_confined_dropped,
        "n_scored_unconfined": int(len(df)),
        "n_missing_baseline_join": n_missing_base,
        "n_missing_panel_join": n_missing_panel,
        "confinement_classes_kept": sorted(df["confinement_class"].unique().tolist()),
    }
    log.info("join/population: %s", json.dumps(_round(diag)))
    return df, diag


def definitions_block() -> dict:
    return {
        "target": "obs_dtw_m = observed depth to the unconfined water table (m).",
        "residual": "pred_dtw - obs_dtw (m; positive = predicted too deep).",
        "mad_m": "median(|residual|) (m).",
        "bias_mean_m": "mean(residual) (m).",
        "median_resid_m": "median(residual) (m).",
        "rmse_m": "sqrt(mean(residual^2)) (m).",
        "mad_skill / rmse_skill": "1 - metric_mae_arm/metric_baseline (dimensionless; "
        "positive = MAE arm better than the frozen baseline).",
        "mad_skill_ci95": "percentile bootstrap 95% CI on the MAD skill over well "
        "resamples (paired on the common footprint); excludes 0 => significant.",
        "shallow_skill": "precision/recall of the 'shallow water table' call at <2/<5/"
        "<10 m thresholds (dimensionless).",
        "nv_slice": "HUC2 == 16 (Great Basin closed basins; score_conus_gnn NV panel).",
        "wide_basin_slice": "HUC2 in {13,15,16} (Rio Grande/NM rift, Lower Colorado, "
        "Great Basin) -- the wide arid closed-basin regime from the probe round.",
        "sacrificial_excluded": f"HUC4 {', '.join(SACRIFICIAL_HUC4)} dropped from every "
        "metric (frozen regional-holdout lockout).",
        "unconfined_only": "confinement_class in {unconfined, unconfined_marginal}; "
        "confined wells measure a potentiometric surface and are excluded.",
    }


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def write_summary(report: dict, path: Path) -> None:
    d = report["population"]
    allp = report["all"]
    lines = [
        "# MAE neighborhood-embedding arm — score summary",
        "",
        f"Scored population: {d['n_scored_unconfined']} unconfined real wells "
        f"(is_water_pseudo == False, finite obs_dtw_m, confinement_class in "
        f"{{{', '.join(UNCONFINED)}}}, sacrificial HUC4s {', '.join(SACRIFICIAL_HUC4)} "
        "dropped). MAE arm vs frozen w25_prod baseline on the common footprint, joined "
        "on canonical_id. All errors in metres; skills/CI dimensionless.",
        "",
        "## Overall (all scored wells)",
        "",
        "| arm | n | MAD_m | bias_mean_m | median_resid_m | RMSE_m |",
        "|---|---|---|---|---|---|",
    ]
    for label, node in (("MAE arm", allp["mae_arm"]), ("baseline", allp["baseline"])):
        o = node["overall"]
        lines.append(
            f"| {label} | {o.get('n')} | {_fmt(o.get('mad_m'))} | "
            f"{_fmt(o.get('bias_mean_m'))} | {_fmt(o.get('median_resid_m'))} | "
            f"{_fmt(o.get('rmse_m'))} |"
        )
    p = allp["paired_overall"]
    lines += [
        "",
        f"Paired overall: MAD skill {_fmt(p.get('mad_skill'))} "
        f"(CI95 {p.get('mad_skill_ci95')}), RMSE skill {_fmt(p.get('rmse_skill'))}; "
        f"MAE arm MAD {_fmt(p.get('mad_model_m'))} m vs baseline "
        f"{_fmt(p.get('mad_base_m'))} m (n={p.get('n')}).",
        "",
        "## By observed-depth band (MAE arm | baseline; MAD_m / RMSE_m / n) + paired MAD skill",
        "",
        "| band | MAE MAD | base MAD | MAE RMSE | base RMSE | n | paired MAD skill [CI95] |",
        "|---|---|---|---|---|---|---|",
    ]
    mb = allp["mae_arm"]["by_depth_band"]
    bb = allp["baseline"]["by_depth_band"]
    pb = allp["paired_by_depth_band"]
    for lo, hi in DEPTH_BANDS:
        lbl = _band_label(lo, hi)
        m, b, pr = mb.get(lbl, {}), bb.get(lbl, {}), pb.get(lbl, {})
        lines.append(
            f"| {lbl} | {_fmt(m.get('mad_m'))} | {_fmt(b.get('mad_m'))} | "
            f"{_fmt(m.get('rmse_m'))} | {_fmt(b.get('rmse_m'))} | {m.get('n', 0)} | "
            f"{_fmt(pr.get('mad_skill'))} {pr.get('mad_skill_ci95')} |"
        )
    for key, title in (
        ("nv_closed_basin_huc2_16", "NV closed basins (HUC2=16)"),
        ("wide_basin_huc2_13_15_16", "Wide basins (HUC2 in 13/15/16)"),
    ):
        sl = report["slices"].get(key)
        lines += ["", f"## {title}", ""]
        if sl is None or sl.get("n", 0) < MIN_SLICE_N:
            lines.append(f"(< {MIN_SLICE_N} wells; slice skipped.)")
            continue
        mo = sl["mae_arm"]["overall"]
        bo = sl["baseline"]["overall"]
        pp = sl["paired_overall"]
        lines += [
            f"n = {sl['n']}. MAE arm MAD {_fmt(mo.get('mad_m'))} m "
            f"(bias {_fmt(mo.get('bias_mean_m'))}, median {_fmt(mo.get('median_resid_m'))}, "
            f"RMSE {_fmt(mo.get('rmse_m'))}) vs baseline MAD {_fmt(bo.get('mad_m'))} m "
            f"(bias {_fmt(bo.get('bias_mean_m'))}, RMSE {_fmt(bo.get('rmse_m'))}).",
            f"Paired MAD skill {_fmt(pp.get('mad_skill'))} "
            f"(CI95 {pp.get('mad_skill_ci95')}), RMSE skill {_fmt(pp.get('rmse_skill'))}.",
        ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", required=True, help="MAE-arm output dir")
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--panels", default=PANELS)
    ap.add_argument("--out-dir", default=None, help="defaults to --arm-dir")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    arm_dir = Path(args.arm_dir)
    args.arm_oof = str(arm_dir / "gnn_oof_predictions.parquet")
    out_dir = Path(args.out_dir) if args.out_dir else arm_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, diag = load_and_join(args)

    cf = (
        np.isfinite(df["obs_dtw_m"].to_numpy(float))
        & np.isfinite(df["mae_dtw_m"].to_numpy(float))
        & np.isfinite(df["base_dtw_m"].to_numpy(float))
    )
    cf_df = df[cf].reset_index(drop=True)
    log.info(
        "common footprint (MAE arm + baseline + obs finite): %d of %d scored wells",
        int(cf.sum()),
        len(df),
    )

    all_panel = slice_panel(cf_df, np.ones(len(cf_df), bool), "all")
    huc2 = cf_df["huc2"].to_numpy()
    slices = {
        "nv_closed_basin_huc2_16": slice_panel(
            cf_df, np.isin(huc2, NV_HUC2), "nv_closed_basin_huc2_16"
        ),
        "wide_basin_huc2_13_15_16": slice_panel(
            cf_df, np.isin(huc2, WIDE_BASIN_HUC2), "wide_basin_huc2_13_15_16"
        ),
    }

    report = {
        "work": "MAE neighborhood-embedding arm vs frozen w25_prod baseline (v0.2 contract)",
        "arm_dir": str(arm_dir),
        "baseline": str(args.baseline),
        "definitions": definitions_block(),
        "join": diag,
        "population": {
            "n_scored_unconfined": int(len(df)),
            "n_common_footprint": int(cf.sum()),
            "n_folds": int(cf_df["cv_fold"].nunique()),
            "sacrificial_huc4_excluded": list(SACRIFICIAL_HUC4),
            "unconfined_classes": list(UNCONFINED),
        },
        "all": all_panel,
        "slices": slices,
    }
    report = _round(report)

    out_report = out_dir / "mae_score_report.json"
    out_report.write_text(json.dumps(report, indent=2, default=str))
    out_md = out_dir / "mae_score_summary.md"
    write_summary(report, out_md)
    log.info("wrote %s", out_report)
    log.info("wrote %s", out_md)
    p = report["all"]["paired_overall"]
    log.info(
        "MAE arm overall MAD %.4f m vs baseline %.4f m; MAD skill %.4f (CI %s), "
        "RMSE skill %.4f (n=%d)",
        p["mad_model_m"],
        p["mad_base_m"],
        p["mad_skill"],
        p["mad_skill_ci95"],
        p["rmse_skill"],
        p["n"],
    )


if __name__ == "__main__":
    main()
