"""WP5 — score the v0.2 GNN ordinal shallow-water head against the frozen controls.

Companion to ``v02_shallow_ordinal_controls.py``. That script froze the bar the
ordinal head must clear; this one scores the trained ordinal arm on the SAME
scoring population, folds and label so the comparison is apples-to-apples.

The ordinal arm is the production ``w25_prod`` GNN (prior-gate + mirror-anchor +
sigma head + 0.25 water-label weight) with an added monotone ordinal shallow head
emitting per-well OOF probabilities

    p_dtw_lt_2m <= p_dtw_lt_5m <= p_dtw_lt_10m   (nested by construction),

where each column is P(true DTW < t) for t in {2, 5, 10} m. It is scored against
the class ``y_t = (mean_dtw < t)`` (mean_dtw = observed depth to the unconfined
water table, m — identical to the controls' label).

Metrics mirror ``v02_metrics.prob_class_panel`` (Brier, Brier skill vs the
in-stratum prevalence forecast, PR-AUC) plus ECE on the controls' 10 equal-width
reliability bins and the descriptive p>=0.5 operating point. All probabilities /
PR-AUC / Brier / Brier-skill / ECE values are dimensionless; every length is in m.

The four preregistered WP5 bars (v2 plan §10.3), reported per threshold:
  (a) beat the strongest tabular control (GBT) on threshold-free PR-AUC;
  (b) beat the honest production readout (gnn_laplace_cdf) on Brier skill;
  (c) 0 monotone-ordinal violations (p2 <= p5 <= p10 for every well);
  (d) no material regression of the jointly-trained point surface vs the frozen
      baseline (overall MAD and RMSE within --regression-tol-frac).

Discipline (matches the controls exactly): real wells only
(is_water_pseudo == False) with finite mean_dtw; sacrificial regional-holdout
HUC4s (0707, 1019, 1605) dropped from every report; no Ma/Janssen columns in any
metric; no threshold selected on any geography (headline = PR-AUC / Brier skill).

Outputs to ``.../v02/wp5/``:
  ordinal_score_report.json    full stratified panel + bar comparison + point
                               non-regression table + definitions
  ordinal_score_summary.md     PASS/FAIL verdict per bar per threshold

Usage:
    uv run python utils/v02_score_ordinal.py
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
from score_conus_gnn import core_metrics, depth_banded  # noqa: E402
from v02_metrics import prob_class_panel  # noqa: E402
from v02_shallow_ordinal_controls import (  # noqa: E402
    THRESHOLDS_M,
    build_strata,
    load_population,
    ordinal_monotonicity,
    pr_at_op,
    reliability_table,
)

log = logging.getLogger("v02_score_ordinal")

WTE_GNN = "/data/ssd2/handily/conus/wte_gnn"
CONTRACT_DIR = f"{WTE_GNN}/v02/contract"
WP5_DIR = f"{WTE_GNN}/v02/wp5"
ORDINAL_OOF = f"{WP5_DIR}/gnn_w25_ordinal/gnn_oof_predictions.parquet"
CONTROLS_REPORT = f"{WP5_DIR}/shallow_controls_report.json"
OUT_DIR = WP5_DIR

# ordinal head OOF probability column per class threshold (m -> column name)
PROB_COL = {2.0: "p_dtw_lt_2m", 5.0: "p_dtw_lt_5m", 10.0: "p_dtw_lt_10m"}
REGRESSION_TOL_FRAC = (
    0.02  # matches contract acceptance_gates regression_tolerance_frac
)
POINT_METRIC_KEYS = ("mad_m", "bias_mean_m", "median_resid_m", "rmse_m")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def expected_calibration_error(p, y, n_bins: int = 10) -> float | None:
    """ECE over ``n_bins`` equal-width probability bins (same binning as
    ``v02_shallow_ordinal_controls.reliability_table``).

    ECE = sum_b (n_b / N) * |mean_pred_b - obs_freq_b|; dimensionless in [0, 1];
    lower = better calibrated. The last bin is closed on the right so p == 1 falls
    in it, matching the reliability table.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    n = len(p)
    if n == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        nb = int(m.sum())
        if nb:
            ece += (nb / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return round(float(ece), 4)


def _round_metrics(d: dict, nd: int = 4) -> dict:
    """Round numeric leaves of a (possibly nested) metric dict for the report."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _round_metrics(v, nd)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = round(float(v), nd)
        else:
            out[k] = v
    return out


def _delta(model: dict, base: dict) -> dict:
    """model - base for the shared point-metric keys (m); positive = model worse
    for magnitude metrics (mad/rmse), signed for bias/median."""
    return {
        k: round(float(model[k] - base[k]), 4)
        for k in POINT_METRIC_KEYS
        if k in model and k in base
    }


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_scored_population(
    contract_dir: str, ordinal_oof: str, feature_cols: list
) -> tuple[pd.DataFrame, dict]:
    """Load the frozen controls population and join the ordinal arm OOF onto it.

    ``load_population`` returns the identical 33 245-well scoring set the controls
    used (real, finite mean_dtw, non-sacrificial), carrying the frozen-baseline
    point surface as ``gnn_dtw_m``. The ordinal arm's point + probability OOF are
    merged on canonical_id; the frozen point is renamed so both survive.
    """
    df = load_population(contract_dir, feature_cols)
    df = df.rename(
        columns={"gnn_dtw_m": "gnn_dtw_frozen_m", "gnn_sigma_m": "gnn_sigma_frozen_m"}
    )

    ordn = pd.read_parquet(
        ordinal_oof,
        columns=[
            "canonical_id",
            "is_water_pseudo",
            "gnn_dtw_m",
            "gnn_sigma_m",
            *PROB_COL.values(),
        ],
    )
    ordn = ordn[~ordn["is_water_pseudo"].astype(bool)].drop(columns=["is_water_pseudo"])
    ordn = ordn.rename(
        columns={"gnn_dtw_m": "gnn_dtw_ord_m", "gnn_sigma_m": "gnn_sigma_ord_m"}
    )
    if ordn["canonical_id"].duplicated().any():
        raise ValueError("canonical_id not unique among real ordinal-arm wells")

    n_pop = len(df)
    df = df.merge(ordn, on="canonical_id", how="left", validate="one_to_one")
    prob_missing = int(df[PROB_COL[2.0]].isna().sum())
    point_missing = int(df["gnn_dtw_ord_m"].isna().sum())
    join = {
        "n_population": int(n_pop),
        "n_joined_ordinal": int(n_pop - prob_missing),
        "n_missing_ordinal_prob": prob_missing,
        "n_missing_ordinal_point": point_missing,
        "n_ordinal_real_wells": int(len(ordn)),
    }
    log.info("join: %s", json.dumps(join))
    if prob_missing or point_missing:
        log.warning(
            "%d wells lack ordinal probabilities / %d lack ordinal point — "
            "excluded from ordinal metrics (row-count caveat)",
            prob_missing,
            point_missing,
        )
    return df, join


# --------------------------------------------------------------------------- #
# metric panels
# --------------------------------------------------------------------------- #
def ordinal_stratum_panel(mask: np.ndarray, y: dict, p: dict) -> dict:
    """Per-threshold ordinal-head probability panel restricted to ``mask``."""
    out = {"n": int(mask.sum())}
    for t in THRESHOLDS_M:
        pm, ym = p[t][mask], y[t][mask]
        out[f"t{int(t)}"] = {
            "prob_panel": prob_class_panel(pm, ym),
            "op_0p5": pr_at_op(pm, ym, 0.5),
            "ece": expected_calibration_error(pm, ym),
        }
    return out


def bar_comparison(stratum: str, ordinal_panel: dict, ctrl_strata: dict) -> dict:
    """Compare the ordinal head to the WP5 bars on one stratum.

    Bar (a): ordinal PR-AUC vs GBT PR-AUC (threshold-free, dimensionless).
    Bar (b): ordinal Brier skill vs gnn_laplace_cdf Brier skill (dimensionless).
    Degenerate class panels (single-class stratum) yield no bar -> pass = None.
    """
    ctrl = ctrl_strata.get(stratum, {})
    out = {}
    for t in THRESHOLDS_M:
        key = f"t{int(t)}"
        op = ordinal_panel[key]["prob_panel"]
        gbt = ctrl.get("gbt", {}).get(key, {}).get("prob_panel", {})
        lap = ctrl.get("gnn_laplace_cdf", {}).get(key, {}).get("prob_panel", {})

        ord_pr = op.get("pr_auc")
        ord_bs = op.get("brier_skill")
        gbt_pr = gbt.get("pr_auc")
        lap_bs = lap.get("brier_skill")

        pass_a = None if (ord_pr is None or gbt_pr is None) else bool(ord_pr > gbt_pr)
        pass_b = None if (ord_bs is None or lap_bs is None) else bool(ord_bs > lap_bs)
        out[key] = {
            "base_rate": op.get("base_rate"),
            "ordinal_pr_auc": ord_pr,
            "gbt_pr_auc": gbt_pr,
            "pass_a_beats_gbt_pr_auc": pass_a,
            "ordinal_brier_skill": ord_bs,
            "laplace_brier_skill": lap_bs,
            "pass_b_beats_laplace_brier_skill": pass_b,
            "ordinal_brier": op.get("brier"),
            "laplace_pr_auc": lap.get("pr_auc"),
            "gbt_brier_skill": gbt.get("brier_skill"),
        }
    return out


def point_non_regression(
    ord_pt: np.ndarray, frozen_pt: np.ndarray, obs: np.ndarray, tol_frac: float
) -> dict:
    """Point-surface non-regression: ordinal vs frozen baseline on the same rows.

    Reports MAD/bias/median-resid/RMSE (m) overall and by observed-depth band for
    both arms plus the paired delta (ordinal - frozen). Bar (d) passes when the
    overall MAD and RMSE do not exceed the frozen baseline by more than tol_frac.
    """
    ord_ov = core_metrics(ord_pt, obs)
    fro_ov = core_metrics(frozen_pt, obs)
    mad_regress = ord_ov["mad_m"] > fro_ov["mad_m"] * (1.0 + tol_frac)
    rmse_regress = ord_ov["rmse_m"] > fro_ov["rmse_m"] * (1.0 + tol_frac)

    ord_bands = depth_banded(ord_pt, obs)
    fro_bands = depth_banded(frozen_pt, obs)
    by_band = {}
    for band in ord_bands:
        ob, fb = ord_bands[band], fro_bands[band]
        by_band[band] = {
            "n": ob.get("n", 0),
            "ordinal": _round_metrics(ob),
            "frozen": _round_metrics(fb),
            "delta_ord_minus_frozen": _delta(ob, fb)
            if ob.get("n") and fb.get("n")
            else {},
        }
    return {
        "regression_tol_frac": tol_frac,
        "overall": {
            "ordinal": _round_metrics(ord_ov),
            "frozen": _round_metrics(fro_ov),
            "delta_ord_minus_frozen": _delta(ord_ov, fro_ov),
        },
        "mad_regression": bool(mad_regress),
        "rmse_regression": bool(rmse_regress),
        "pass_d_no_material_regression": bool(not (mad_regress or rmse_regress)),
        "by_depth_band": by_band,
    }


# --------------------------------------------------------------------------- #
# report / summary assembly
# --------------------------------------------------------------------------- #
def definitions_block(tol_frac: float) -> dict:
    return {
        "target": "mean_dtw = observed depth to the unconfined water table (m); label "
        "identical to the frozen controls.",
        "class_y_t": "y_t = (mean_dtw < t); binary shallow-water class at threshold t m.",
        "thresholds_m": list(THRESHOLDS_M),
        "ordinal_head": "monotone GNN head emitting P(true DTW < t) as p_dtw_lt_{2,5,10}m; "
        "nested by construction (p2 <= p5 <= p10).",
        "population": "real wells (is_water_pseudo == False) with finite mean_dtw, minus "
        "sacrificial regional-holdout HUC4s; identical rows/folds to the controls.",
        "brier": "mean((p - y)^2); dimensionless (probability^2); lower better.",
        "brier_skill": "1 - brier / (base*(1-base)); dimensionless; positive beats the "
        "in-stratum prevalence forecast.",
        "pr_auc": "average precision (area under PR curve); dimensionless in [0,1]; the "
        "no-skill baseline equals base_rate.",
        "ece": "expected calibration error over 10 equal-width prob bins; dimensionless "
        "in [0,1]; lower = better calibrated.",
        "op_0p5": "descriptive operating point at p >= 0.5: precision = TP/(TP+FP), "
        "recall = TP/(TP+FN) (dimensionless). NOT a selected threshold.",
        "point_metrics": "residual = pred - obs (m); mad_m = median|resid|, "
        "bias_mean_m = mean resid, median_resid_m = median resid, rmse_m = RMSE.",
        "bars": {
            "a": "ordinal PR-AUC > GBT control PR-AUC, per threshold.",
            "b": "ordinal Brier skill > gnn_laplace_cdf control Brier skill, per threshold.",
            "c": "0 monotone-ordinal violations over the whole population.",
            "d": f"overall point MAD and RMSE within {tol_frac:.0%} of the frozen baseline.",
        },
    }


def _fmt(x, nd=4):
    return (
        f"{x:.{nd}f}"
        if isinstance(x, (int, float)) and not isinstance(x, bool)
        else "-"
    )


def _verdict(flag) -> str:
    return "PASS" if flag is True else ("FAIL" if flag is False else "n/a")


def write_summary(report: dict, path: Path) -> None:
    pop = report["population"]
    allbars = report["bars_by_stratum"]["all"]
    mono = report["monotonicity"]
    point = report["point_non_regression"]
    lines = [
        "# WP5 ordinal shallow-head score — summary",
        "",
        f"Scoring population: {pop['n_scoring_wells']} real wells "
        f"(sacrificial HUC4s dropped; finite mean_dtw); OOF over {pop['n_folds']} folds. "
        f"Identical rows/folds/label to the frozen controls "
        f"(join: {report['join']['n_missing_ordinal_prob']} wells missing ordinal probs).",
        "",
        "All PR-AUC / Brier / Brier-skill / ECE values are dimensionless; every length "
        "is in metres. Bars are the frozen controls (v2 plan §10.3).",
        "",
        "## Verdict — all-wells scoring stratum",
        "",
        "### (a) beat GBT control PR-AUC (threshold-free)",
    ]
    for t in THRESHOLDS_M:
        b = allbars[f"t{int(t)}"]
        lines.append(
            f"- P(WTD < {int(t)} m): ordinal PR-AUC {_fmt(b['ordinal_pr_auc'])} vs "
            f"GBT {_fmt(b['gbt_pr_auc'])} -> {_verdict(b['pass_a_beats_gbt_pr_auc'])}"
        )
    lines += ["", "### (b) beat gnn_laplace_cdf control Brier skill"]
    for t in THRESHOLDS_M:
        b = allbars[f"t{int(t)}"]
        lines.append(
            f"- P(WTD < {int(t)} m): ordinal Brier skill {_fmt(b['ordinal_brier_skill'])} vs "
            f"Laplace {_fmt(b['laplace_brier_skill'])} -> "
            f"{_verdict(b['pass_b_beats_laplace_brier_skill'])}"
        )
    lines += [
        "",
        "### (c) monotone-ordinal contract",
        f"- p2 <= p5 <= p10 violations: {mono['n_violations']} of {mono['n']} wells "
        f"(frac_monotone {_fmt(mono['frac_monotone'], 6)}) -> "
        f"{_verdict(report['pass_c_zero_violations'])}",
        "",
        "### (d) point-surface non-regression (jointly-trained point head)",
    ]
    ov_o = point["overall"]["ordinal"]
    ov_f = point["overall"]["frozen"]
    dl = point["overall"]["delta_ord_minus_frozen"]
    lines += [
        f"- overall MAD: ordinal {_fmt(ov_o['mad_m'])} m vs frozen {_fmt(ov_f['mad_m'])} m "
        f"(delta {_fmt(dl['mad_m'])} m)",
        f"- overall RMSE: ordinal {_fmt(ov_o['rmse_m'])} m vs frozen {_fmt(ov_f['rmse_m'])} m "
        f"(delta {_fmt(dl['rmse_m'])} m)",
        f"- overall bias: ordinal {_fmt(ov_o['bias_mean_m'])} m vs frozen "
        f"{_fmt(ov_f['bias_mean_m'])} m; median resid ordinal {_fmt(ov_o['median_resid_m'])} m "
        f"vs frozen {_fmt(ov_f['median_resid_m'])} m",
        f"- verdict (tol {point['regression_tol_frac']:.0%}): "
        f"{_verdict(point['pass_d_no_material_regression'])}",
        "",
        "MAD by observed-depth band (m), ordinal vs frozen (delta = ordinal - frozen):",
        "",
        "| band | n | ordinal MAD | frozen MAD | delta | ordinal RMSE | frozen RMSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for band, node in point["by_depth_band"].items():
        o, f = node["ordinal"], node["frozen"]
        d = node["delta_ord_minus_frozen"]
        lines.append(
            f"| {band} | {node['n']} | {_fmt(o.get('mad_m'))} | {_fmt(f.get('mad_m'))} | "
            f"{_fmt(d.get('mad_m'))} | {_fmt(o.get('rmse_m'))} | {_fmt(f.get('rmse_m'))} |"
        )

    # per-panel pass rollup for a and b
    lines += ["", "## Independent-panel bar pass rate (a & b)", ""]
    panels = [s for s in report["bars_by_stratum"] if s != "all"]
    for label, flag_key in [
        ("(a) beats GBT PR-AUC", "pass_a_beats_gbt_pr_auc"),
        ("(b) beats Laplace Brier skill", "pass_b_beats_laplace_brier_skill"),
    ]:
        n_pass = n_eval = 0
        for s in ["all"] + panels:
            for t in THRESHOLDS_M:
                v = report["bars_by_stratum"][s][f"t{int(t)}"][flag_key]
                if v is None:
                    continue
                n_eval += 1
                n_pass += int(v)
        lines.append(
            f"- {label}: {n_pass}/{n_eval} stratum×threshold cells pass "
            f"(across {len(panels) + 1} strata; degenerate class cells excluded)."
        )

    lines += [
        "",
        "## Overall WP5 gate",
        "",
        f"- (a) beats GBT PR-AUC (all-wells, all thresholds): "
        f"{_verdict(report['gate']['pass_a_all_thresholds'])}",
        f"- (b) beats Laplace Brier skill (all-wells, all thresholds): "
        f"{_verdict(report['gate']['pass_b_all_thresholds'])}",
        f"- (c) 0 monotone violations: {_verdict(report['pass_c_zero_violations'])}",
        f"- (d) point non-regression: "
        f"{_verdict(point['pass_d_no_material_regression'])}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract-dir", default=CONTRACT_DIR)
    ap.add_argument("--ordinal-oof", default=ORDINAL_OOF)
    ap.add_argument("--controls-report", default=CONTROLS_REPORT)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--regression-tol-frac", type=float, default=REGRESSION_TOL_FRAC)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    manifest = json.loads(
        (
            Path(args.contract_dir) / "frozen_baseline" / "graph_manifest.json"
        ).read_text()
    )
    feature_cols = list(manifest["query_feature_cols"])

    controls = json.loads(Path(args.controls_report).read_text())
    ctrl_strata = controls["strata"]
    ctrl_pop = controls["population"]

    df, join = load_scored_population(args.contract_dir, args.ordinal_oof, feature_cols)

    # scoring arrays on the common footprint
    obs = df["mean_dtw"].to_numpy(float)
    y = {t: obs < t for t in THRESHOLDS_M}
    p = {t: df[PROB_COL[t]].to_numpy(float) for t in THRESHOLDS_M}
    ord_pt = df["gnn_dtw_ord_m"].to_numpy(float)
    frozen_pt = df["gnn_dtw_frozen_m"].to_numpy(float)

    # footprint sanity vs controls
    footprint = {
        "n_ordinal_scoring_wells": int(len(df)),
        "n_controls_scoring_wells": int(ctrl_pop["n_scoring_wells"]),
        "match": bool(
            len(df) == ctrl_pop["n_scoring_wells"]
            and join["n_missing_ordinal_prob"] == 0
        ),
    }

    strata = build_strata(df)
    ordinal_panels = {
        name: ordinal_stratum_panel(m, y, p) for name, m in strata.items()
    }
    bars = {
        name: bar_comparison(name, ordinal_panels[name], ctrl_strata) for name in strata
    }

    mono = ordinal_monotonicity(p[2.0], p[5.0], p[10.0])
    point = point_non_regression(ord_pt, frozen_pt, obs, args.regression_tol_frac)

    all_bars = bars["all"]
    gate = {
        "pass_a_all_thresholds": all(
            all_bars[f"t{int(t)}"]["pass_a_beats_gbt_pr_auc"] for t in THRESHOLDS_M
        ),
        "pass_b_all_thresholds": all(
            all_bars[f"t{int(t)}"]["pass_b_beats_laplace_brier_skill"]
            for t in THRESHOLDS_M
        ),
    }

    report = {
        "work_package": "WP5 ordinal shallow-head score (v2 plan §10)",
        "definitions": definitions_block(args.regression_tol_frac),
        "ordinal_arm_oof": str(args.ordinal_oof),
        "controls_report": str(args.controls_report),
        "join": join,
        "footprint_vs_controls": footprint,
        "population": {
            "n_scoring_wells": int(len(df)),
            "n_folds": int(df["cv_fold"].nunique()),
            "prevalence_by_threshold": {
                f"t{int(t)}": round(float(y[t].mean()), 4) for t in THRESHOLDS_M
            },
        },
        "ordinal_panels": ordinal_panels,
        "bars_by_stratum": bars,
        "reliability_all": {
            f"t{int(t)}": reliability_table(p[t], y[t]) for t in THRESHOLDS_M
        },
        "monotonicity": mono,
        "pass_c_zero_violations": bool(mono["n_violations"] == 0),
        "point_non_regression": point,
        "gate": gate,
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "ordinal_score_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float))
    md_path = out / "ordinal_score_summary.md"
    write_summary(report, md_path)

    log.info("wrote %s", report_path)
    log.info("wrote %s", md_path)
    # terse console verdict
    for t in THRESHOLDS_M:
        b = all_bars[f"t{int(t)}"]
        log.info(
            "P(<%dm): PR-AUC ord %.4f vs gbt %.4f (%s) | BrierSkill ord %.4f vs lap %.4f (%s)",
            int(t),
            b["ordinal_pr_auc"],
            b["gbt_pr_auc"],
            _verdict(b["pass_a_beats_gbt_pr_auc"]),
            b["ordinal_brier_skill"],
            b["laplace_brier_skill"],
            _verdict(b["pass_b_beats_laplace_brier_skill"]),
        )
    log.info(
        "monotone violations %d (%s); point non-regression %s",
        mono["n_violations"],
        _verdict(report["pass_c_zero_violations"]),
        _verdict(point["pass_d_no_material_regression"]),
    )


if __name__ == "__main__":
    main()
