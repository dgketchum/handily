"""WP2 — score the v0.2 latent two-surface (phreatic + regional) GNN mixture arm
against the frozen production baseline and the preregistered WP2 acceptance gates.

Companion to ``utils/train_conus_gnn.py --two-surface`` (see
``notes/v02_implementation/WP2_MODEL_DESIGN.md`` §Scoring for the spec this
implements) and ``utils/build_two_surface_priors.py`` (the privileged assignment
prior used ONLY in the training loss, read here as held-out evidence).

The arm trains a latent two-component Laplace mixture: a PHREATIC component anchored
on the FAC/terrain-mirror shallow prior and a REGIONAL component anchored on the
deep-well IDW datum, with a membership head ``pi = P(phreatic)``. The blended point
prediction ``gnn_dtw_m = pi * H_p + (1-pi) * H_r`` is what early-stopping and legacy
scoring use; the estimand-correct view is the two components scored separately.

Component DTWs (both in DTW space, m; wte_residual native target convention):
    component WTE = regional_wte_idw_oof_m + ts_native_{p,r}_m
    component DTW = z_surf_well_m - (regional_wte_idw_oof_m + ts_native_{p,r}_m)
By construction ``gnn_dtw_m == pi*phreatic_dtw + (1-pi)*regional_dtw`` (QA-checked).

Discipline (matches every other v0.2 scorer):
- real wells only (is_water_pseudo == False) with finite obs_dtw_m;
- sacrificial regional-holdout HUC4s (0707, 1019, 1605) dropped from EVERY report;
- the mixture point pred and the frozen baseline are compared on the SAME common
  footprint (both arms + obs finite), joined on canonical_id;
- no Ma / Janssen columns enter any WP2 metric (they are benchmarks only);
- units on every number (m unless noted dimensionless).

Sections (WP2_MODEL_DESIGN.md §Scoring):
  1. Point non-regression + deep gate: mixture point pred vs frozen baseline, MAD /
     bias / median-residual / RMSE overall and depth-banded, paired reductions, on
     all wells + the separated deep panels (mech_deep_far_arid, buffered >=2.5 km) +
     the shallow-irrigated panel.
  2. Estimand-separated readout: the same depth-banded panel for the phreatic
     component alone, the regional component alone, the oracle-best component
     (min |error|, the identifiability ceiling) and the pi-hard-assigned component.
  3. Component-collapse check: pi distribution (mean, share <0.05 / >0.95) globally,
     by fold, by HUC2 and by depth band; water rows separately; collapse flagged if
     either component carries <5% global usage.
  4. Assignment agreement (OOF pi vs held-out construction evidence): among wells
     with confident construction priors (|2p-1| >= 0.5), AUC of pi against
     (p_phreatic_construction >= 0.5) and Spearman(pi, depth_rank_q), vs the trivial
     baselines (constant; the prior itself predicting depth-rank).
  5. Verdict block: PASS/FAIL per preregistered gate with the numbers cited.

Gate arithmetic is read from ``<contract-dir>/acceptance_gates.json`` at runtime and
applied exactly as frozen (see ``wp2_gate`` in the report for the literal spec).

Usage:
    uv run python utils/v02_score_two_surface.py \\
        --run-dir /data/ssd2/handily/conus/wte_gnn/v02/wp2/gnn_w25_two_surface
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

log = logging.getLogger("v02_score_two_surface")

WTE_GNN = "/data/ssd2/handily/conus/wte_gnn"
RUN_DIR = f"{WTE_GNN}/v02/wp2/gnn_w25_two_surface"
BASELINE = f"{WTE_GNN}/v02/contract/frozen_baseline/gnn_oof_predictions.parquet"
CONTRACT_DIR = f"{WTE_GNN}/v02/contract"
PRIORS = f"{WTE_GNN}/v02/wp2/assignment_priors.parquet"

SACRIFICIAL_HUC4 = ("0707", "1019", "1605")
DEPTH_BANDS = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
SHALLOW_BAND_LABELS = ("0-2m", "2-5m")
DEEP_BAND_LABEL = "30+m"
MID_DEEP_BAND_LABEL = "10-30m"
SHALLOW_THRESHOLDS = ("<2m", "<5m", "<10m")
CONFIDENT_PRIOR_MIN = 0.5  # |2p-1| >= 0.5 => held-out construction evidence is usable
COLLAPSE_MIN_USAGE = 0.05  # a component used < this fraction globally = collapse
PI_LO, PI_HI = 0.05, 0.95  # membership-share diagnostic bounds


# --------------------------------------------------------------------------- #
# pure math (unit-testable)
# --------------------------------------------------------------------------- #
def component_dtw(z_surf, regional_wte, ts_native) -> np.ndarray:
    """DTW (m) of one mixture component.

    In the wte_residual native target space a component's water-table ELEVATION is
    ``regional_wte_idw_oof_m + ts_native``; its DEPTH below ground is
    ``z_surf - (regional_wte_idw_oof_m + ts_native)``. Elementwise; NaN propagates.
    """
    z = np.asarray(z_surf, float)
    r = np.asarray(regional_wte, float)
    t = np.asarray(ts_native, float)
    return z - (r + t)


def oracle_best_dtw(phreatic_dtw, regional_dtw, obs) -> np.ndarray:
    """Per-row component DTW (m) with the smaller absolute error vs obs.

    The identifiability ceiling: if the mixture could route every well to its truly
    better component, this is the error it would achieve. Where exactly one component
    is finite it is chosen; ties (and both-NaN) fall to the regional value.
    """
    p = np.asarray(phreatic_dtw, float)
    r = np.asarray(regional_dtw, float)
    o = np.asarray(obs, float)
    ep = np.abs(p - o)
    er = np.abs(r - o)
    both = np.isfinite(ep) & np.isfinite(er)
    # choose phreatic when it is the (weakly) better finite option, or the only one
    choose_p = np.where(both, ep <= er, np.isfinite(ep) & ~np.isfinite(er))
    return np.where(choose_p, p, r)


def hard_assigned_dtw(pi, phreatic_dtw, regional_dtw, thr: float = 0.5) -> np.ndarray:
    """pi-hard-assigned component DTW (m): pi >= thr -> phreatic else regional."""
    return np.where(np.asarray(pi, float) >= thr, phreatic_dtw, regional_dtw)


# --------------------------------------------------------------------------- #
# report helpers
# --------------------------------------------------------------------------- #
def _round(obj, nd: int = 4):
    """Round numeric leaves of a nested dict/list for compact JSON (bools kept)."""
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


def arm_panel(pred: np.ndarray, obs: np.ndarray) -> dict:
    """Overall core metrics + depth-banded structure + shallow P/R for one arm."""
    return {
        "overall": core_metrics(pred, obs),
        "by_depth_band": depth_banded(pred, obs),
        "shallow_skill": shallow_skill(pred, obs),
    }


def _band_reductions(pred_m: np.ndarray, pred_b: np.ndarray, obs: np.ndarray) -> dict:
    """Per depth-band paired MAD/RMSE reduction (skill = 1 - model/base) of the
    mixture over the frozen baseline on the SAME rows (common footprint assumed)."""
    out = {}
    for lo, hi in DEPTH_BANDS:
        label = f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"
        sel = np.isfinite(obs) & (obs >= lo) & (obs < hi)
        out[label] = paired_improvement((pred_m - obs)[sel], (pred_b - obs)[sel])
    return out


def point_non_regression_panel(df: pd.DataFrame, mask: np.ndarray, name: str) -> dict:
    """Mixture vs frozen baseline on ``mask`` (already on the common footprint).

    Reports both arms' core + banded metrics, the overall paired improvement, and
    the per-band paired reductions (the raw material for the deep gate)."""
    sub = df[mask]
    obs = sub["obs_dtw_m"].to_numpy(float)
    mix = sub["gnn_dtw_m"].to_numpy(float)
    base = sub["base_dtw_m"].to_numpy(float)
    return {
        "panel": name,
        "n": int(mask.sum()),
        "mixture": arm_panel(mix, obs),
        "frozen": arm_panel(base, obs),
        "paired_overall": paired_improvement(mix - obs, base - obs),
        "paired_by_depth_band": _band_reductions(mix, base, obs),
    }


def estimand_panel(df: pd.DataFrame, mask: np.ndarray, name: str) -> dict:
    """Depth-banded panel for each estimand-separated surface on ``mask``."""
    sub = df[mask]
    obs = sub["obs_dtw_m"].to_numpy(float)
    surfaces = {
        "phreatic_component": sub["phreatic_dtw_m"].to_numpy(float),
        "regional_component": sub["regional_dtw_m"].to_numpy(float),
        "oracle_best_component": sub["oracle_dtw_m"].to_numpy(float),
        "pi_hard_assigned": sub["hard_dtw_m"].to_numpy(float),
        "mixture_blend": sub["gnn_dtw_m"].to_numpy(float),
    }
    return {
        "panel": name,
        "n": int(mask.sum()),
        "surfaces": {k: arm_panel(v, obs) for k, v in surfaces.items()},
    }


def _pi_dist(pi: np.ndarray) -> dict:
    """Membership distribution: mean/median + soft usage of both components +
    hard-share tails. ``phreatic_usage = mean(pi)``, ``regional_usage = mean(1-pi)``
    (all dimensionless)."""
    p = np.asarray(pi, float)
    p = p[np.isfinite(p)]
    n = len(p)
    if n == 0:
        return {"n": 0}
    return {
        "n": int(n),
        "mean_pi": float(np.mean(p)),
        "median_pi": float(np.median(p)),
        "phreatic_usage_mean_pi": float(np.mean(p)),
        "regional_usage_mean_1_minus_pi": float(np.mean(1.0 - p)),
        "share_pi_lt_0.05": float(np.mean(p < PI_LO)),
        "share_pi_gt_0.95": float(np.mean(p > PI_HI)),
        "share_pi_ge_0.5_hard_phreatic": float(np.mean(p >= 0.5)),
    }


def collapse_check(df: pd.DataFrame, water_pi: np.ndarray) -> dict:
    """pi distribution globally / by fold / by HUC2 / by depth band + collapse flag.

    Collapse (WP2 gate) is flagged when the minimum soft component usage
    ``min(mean pi, mean(1-pi))`` falls below ``COLLAPSE_MIN_USAGE`` globally."""
    obs = df["obs_dtw_m"].to_numpy(float)
    pi = df["ts_pi_phreatic"].to_numpy(float)
    overall = _pi_dist(pi)
    min_usage = min(
        overall.get("phreatic_usage_mean_pi", np.nan),
        overall.get("regional_usage_mean_1_minus_pi", np.nan),
    )
    by_fold = {
        str(f): _pi_dist(sub["ts_pi_phreatic"].to_numpy(float))
        for f, sub in df.groupby("cv_fold")
    }
    by_huc2 = {
        str(h): _pi_dist(sub["ts_pi_phreatic"].to_numpy(float))
        for h, sub in df.groupby("huc2")
        if len(sub) >= 25
    }
    by_band = {}
    for lo, hi in DEPTH_BANDS:
        label = f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"
        sel = np.isfinite(obs) & (obs >= lo) & (obs < hi)
        by_band[label] = _pi_dist(pi[sel])
    return {
        "global": overall,
        "min_component_usage": float(min_usage),
        "collapse_threshold": COLLAPSE_MIN_USAGE,
        "component_collapsed": bool(min_usage < COLLAPSE_MIN_USAGE),
        "by_fold": by_fold,
        "by_huc2": by_huc2,
        "by_depth_band": by_band,
        "water_rows": _pi_dist(water_pi),
    }


def assignment_agreement(
    pi: np.ndarray,
    p_constr: np.ndarray,
    depth_rank_q: np.ndarray,
    confident: np.ndarray,
    nested: np.ndarray,
) -> dict:
    """OOF pi vs held-out construction evidence on confident-prior wells.

    ``pi`` for a test-fold well comes from a model that never saw that well's prior
    (the prior is loss-only and the well is out-of-fold), so this is an honest
    agreement measure. AUC of pi against the binarized construction prior; Spearman
    of pi against the local completion-depth rank. Sign note: depth_rank_q rises with
    LOCAL completion depth, and pi = P(phreatic) should FALL with depth, so an
    honest model gives a NEGATIVE Spearman(pi, depth_rank_q) that tracks the prior's
    own (strongly negative, since p_rank = 1 - q)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    def _auc(mask):
        m = mask & np.isfinite(pi) & np.isfinite(p_constr)
        n = int(m.sum())
        y = (p_constr[m] >= 0.5).astype(int)
        if n == 0 or y.sum() in (0, n):
            return {"n": n, "auc": None, "base_rate": None, "degenerate": True}
        return {
            "n": n,
            "auc": float(roc_auc_score(y, pi[m])),
            "base_rate": float(y.mean()),
            "baseline_constant_auc": 0.5,
        }

    def _spearman(mask):
        m = mask & np.isfinite(pi) & np.isfinite(depth_rank_q)
        n = int(m.sum())
        if n < 3:
            return {"n": n, "spearman_pi": None, "spearman_prior_baseline": None}
        return {
            "n": n,
            "spearman_pi_vs_depth_rank_q": float(
                spearmanr(pi[m], depth_rank_q[m]).statistic
            ),
            "spearman_prior_vs_depth_rank_q_baseline": float(
                spearmanr(p_constr[m], depth_rank_q[m]).statistic
            ),
        }

    return {
        "confident_prior_definition": "|2*p_phreatic_construction - 1| >= "
        f"{CONFIDENT_PRIOR_MIN}",
        "auc_all_confident": _auc(confident),
        "auc_nested_confident": _auc(confident & nested),
        "spearman_all_confident": _spearman(confident),
        "spearman_nested_confident": _spearman(confident & nested),
    }


# --------------------------------------------------------------------------- #
# gate evaluation (arithmetic read from acceptance_gates.json)
# --------------------------------------------------------------------------- #
def _band_pass(reds: dict, band: str, min_red: float) -> dict:
    """Does the paired reduction in ``band`` meet ``min_red`` for MAD or RMSE?"""
    b = reds.get(band, {})
    mad_red = b.get("mad_skill")
    rmse_red = b.get("rmse_skill")
    pass_mad = mad_red is not None and mad_red >= min_red
    pass_rmse = rmse_red is not None and rmse_red >= min_red
    return {
        "n": b.get("n", 0),
        "mad_reduction": mad_red,
        "rmse_reduction": rmse_red,
        "mad_model_m": b.get("mad_model_m"),
        "mad_base_m": b.get("mad_base_m"),
        "rmse_model_m": b.get("rmse_model_m"),
        "rmse_base_m": b.get("rmse_base_m"),
        "pass_mad": bool(pass_mad),
        "pass_rmse": bool(pass_rmse),
        "pass": bool(pass_mad or pass_rmse),
    }


def evaluate_gates(
    gate: dict,
    point_panels: dict,
    collapse: dict,
    agreement: dict,
) -> dict:
    """Apply the frozen WP2 gate arithmetic literally.

    ``gate`` is the ``wp2_two_surface`` block from acceptance_gates.json. The frozen
    deep gate binds on the 30+m band on a separated panel; the 10-30m band is
    reported as a diagnostic (the design note asks for it) but is NOT part of the
    frozen PASS/FAIL, per the literal ``deep_band`` field."""
    min_red = float(gate["min_deep_band_error_reduction"])
    shallow_tol = float(gate["shallow_skill_regression_tolerance_frac"])

    # --- deep gate: 30+m band reduction on the separated deep panel(s) ---
    deep_panels = {}
    for pname in ("mech_deep_far_arid", "panel_buffered_2p5km"):
        reds = point_panels[pname]["paired_by_depth_band"]
        deep_panels[pname] = {
            "30+m_binding": _band_pass(reds, DEEP_BAND_LABEL, min_red),
            "10-30m_diagnostic": _band_pass(reds, MID_DEEP_BAND_LABEL, min_red),
        }
    binding_panel = "mech_deep_far_arid"
    deep_pass = deep_panels[binding_panel]["30+m_binding"]["pass"]

    # --- shallow non-regression: shallow bands MAD/RMSE + shallow P/R within tol ---
    all_panel = point_panels["all"]
    mix_bands = all_panel["mixture"]["by_depth_band"]
    fro_bands = all_panel["frozen"]["by_depth_band"]
    shallow_band_checks = {}
    shallow_ok = True
    for label in SHALLOW_BAND_LABELS:
        mb, fb = mix_bands.get(label, {}), fro_bands.get(label, {})
        rec = {"n": mb.get("n", 0)}
        band_ok = True
        for metric in ("mad_m", "rmse_m"):  # lower is better
            mv, fv = mb.get(metric), fb.get(metric)
            reg = mv is not None and fv is not None and mv > fv * (1.0 + shallow_tol)
            rec[metric] = {
                "mixture": mv,
                "frozen": fv,
                "material_regression": bool(reg),
            }
            band_ok = band_ok and not reg
        rec["pass"] = bool(band_ok)
        shallow_band_checks[label] = rec
        shallow_ok = shallow_ok and band_ok

    mix_sk = all_panel["mixture"]["shallow_skill"]
    fro_sk = all_panel["frozen"]["shallow_skill"]
    shallow_skill_checks = {}
    for thr in SHALLOW_THRESHOLDS:
        rec = {}
        thr_ok = True
        for metric in ("precision", "recall"):  # higher is better
            mv, fv = mix_sk.get(thr, {}).get(metric), fro_sk.get(thr, {}).get(metric)
            reg = (
                mv is not None
                and fv is not None
                and np.isfinite(mv)
                and np.isfinite(fv)
                and mv < fv * (1.0 - shallow_tol)
            )
            rec[metric] = {
                "mixture": mv,
                "frozen": fv,
                "material_regression": bool(reg),
            }
            thr_ok = thr_ok and not reg
        rec["pass"] = bool(thr_ok)
        shallow_skill_checks[thr] = rec
        shallow_ok = shallow_ok and thr_ok

    # --- component collapse ---
    collapse_pass = not collapse["component_collapsed"]

    # --- component assignment beats one-surface on nested wells ---
    # the one-surface baseline emits a single component (no membership) -> its
    # implicit assignment AUC is the constant 0.5; the mixture must beat it.
    nested_auc = agreement["auc_nested_confident"].get("auc")
    all_auc = agreement["auc_all_confident"].get("auc")
    nested_n = agreement["auc_nested_confident"].get("n", 0)
    used_scope = (
        "nested_confident"
        if (nested_auc is not None and nested_n >= 25)
        else "all_confident"
    )
    used_auc = nested_auc if used_scope == "nested_confident" else all_auc
    assignment_pass = used_auc is not None and used_auc > 0.5

    overall_pass = bool(deep_pass and shallow_ok and collapse_pass and assignment_pass)
    return {
        "frozen_wp2_gate_literal": gate,
        "gate_arithmetic_notes": {
            "min_deep_band_error_reduction": min_red,
            "deep_band_binding": f"{DEEP_BAND_LABEL} (per frozen 'deep_band' field)",
            "deep_band_diagnostic_only": f"{MID_DEEP_BAND_LABEL} (design note asks "
            "for it; NOT in the frozen 30+m gate)",
            "error_reduction_operator": "reduction = 1 - metric_model/metric_base; "
            "a band passes when MAD OR RMSE reduction >= the threshold (mirrors the "
            "WP1 'MAD or RMSE' contract phrasing)",
            "separated_deep_panels": [
                "mech_deep_far_arid (binding)",
                "panel_buffered_2p5km (support-separated view)",
            ],
            "shallow_non_regression": "shallow bands (0-2m, 2-5m) MAD/RMSE and "
            "shallow-class precision/recall at <2/<5/<10 m must stay within "
            f"{shallow_tol:.0%} of the frozen baseline",
            "no_collapse": f"min(mean pi, mean(1-pi)) >= {COLLAPSE_MIN_USAGE}",
            "assignment_beats_one_surface": "AUC(pi vs binarized construction prior) "
            "> 0.5 on nested confident wells (one-surface implicit AUC = 0.5); falls "
            "back to all-confident wells when nested n < 25",
        },
        "deep_gate": {
            "binding_panel": binding_panel,
            "panels": deep_panels,
            "pass": bool(deep_pass),
        },
        "shallow_non_regression": {
            "tolerance_frac": shallow_tol,
            "band_mad_rmse": shallow_band_checks,
            "shallow_skill_pr": shallow_skill_checks,
            "pass": bool(shallow_ok),
        },
        "component_collapse": {
            "min_component_usage": collapse["min_component_usage"],
            "threshold": COLLAPSE_MIN_USAGE,
            "pass": bool(collapse_pass),
        },
        "assignment_beats_one_surface": {
            "scope_used": used_scope,
            "auc_used": used_auc,
            "auc_nested_confident": nested_auc,
            "auc_all_confident": all_auc,
            "one_surface_baseline_auc": 0.5,
            "pass": bool(assignment_pass),
        },
        "overall_wp2_pass": overall_pass,
    }


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_and_join(args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the scored real-well frame + the water-row frame (for the collapse
    readout) + a join-diagnostics dict. Sacrificial HUC4s dropped everywhere."""
    oof = pd.read_parquet(args.run_dir_oof)
    base = pd.read_parquet(
        args.baseline, columns=["canonical_id", "is_water_pseudo", "gnn_dtw_m"]
    ).rename(columns={"gnn_dtw_m": "base_dtw_m"})
    panels = pd.read_parquet(
        args.panels,
        columns=[
            "canonical_id",
            "is_water_pseudo",
            "regional_holdout",
            "huc4",
            "mech_deep_far_arid",
            "mech_shallow_irrigated",
            "mech_nested_collocated",
            "panel_buffered_2p5km",
            "is_nest_sibling",
        ],
    )
    priors = pd.read_parquet(
        args.priors,
        columns=["canonical_id", "p_phreatic_construction", "depth_rank_q"],
    )

    oof["huc2"] = oof["huc2"].astype(str).str.zfill(2)
    is_water = oof["is_water_pseudo"].astype(bool)

    # ---- water rows (collapse readout only): drop sacrificial via panel join ----
    water = oof[is_water].merge(
        panels[["canonical_id", "regional_holdout"]], on="canonical_id", how="left"
    )
    n_water_all = len(water)
    water = water[water["regional_holdout"] != "sacrificial"]

    # ---- real wells: assemble the scored population ----
    real = oof[~is_water].copy()
    n_real = len(real)
    base_real = base[~base["is_water_pseudo"].astype(bool)].drop(
        columns=["is_water_pseudo"]
    )
    panels_real = panels[~panels["is_water_pseudo"].astype(bool)].drop(
        columns=["is_water_pseudo"]
    )
    if real["canonical_id"].duplicated().any():
        raise ValueError("canonical_id not unique among real two-surface wells")

    df = real.merge(base_real, on="canonical_id", how="left", validate="one_to_one")
    df = df.merge(panels_real, on="canonical_id", how="left", validate="one_to_one")
    df = df.merge(priors, on="canonical_id", how="left", validate="one_to_one")

    n_missing_base = int(df["base_dtw_m"].isna().sum())
    n_missing_panel = int(df["regional_holdout"].isna().sum())
    n_missing_prior = int(df["p_phreatic_construction"].isna().sum())

    fin_obs = np.isfinite(df["obs_dtw_m"].to_numpy(float))
    n_finite = int(fin_obs.sum())
    sac = df["huc4"].isin(SACRIFICIAL_HUC4) | (df["regional_holdout"] == "sacrificial")
    df = df[fin_obs & ~sac.to_numpy()].reset_index(drop=True)

    # ---- estimand-separated surfaces (DTW space) ----
    z = df["z_surf_well_m"].to_numpy(float)
    r_wte = df["regional_wte_idw_oof_m"].to_numpy(float)
    obs = df["obs_dtw_m"].to_numpy(float)
    pi = df["ts_pi_phreatic"].to_numpy(float)
    df["phreatic_dtw_m"] = component_dtw(z, r_wte, df["ts_native_p_m"].to_numpy(float))
    df["regional_dtw_m"] = component_dtw(z, r_wte, df["ts_native_r_m"].to_numpy(float))
    df["oracle_dtw_m"] = oracle_best_dtw(
        df["phreatic_dtw_m"].to_numpy(float), df["regional_dtw_m"].to_numpy(float), obs
    )
    df["hard_dtw_m"] = hard_assigned_dtw(
        pi, df["phreatic_dtw_m"].to_numpy(float), df["regional_dtw_m"].to_numpy(float)
    )

    # ---- QA identity: gnn_dtw_m == pi*phreatic + (1-pi)*regional ----
    recon = pi * df["phreatic_dtw_m"].to_numpy(float) + (1.0 - pi) * df[
        "regional_dtw_m"
    ].to_numpy(float)
    mix = df["gnn_dtw_m"].to_numpy(float)
    ok = np.isfinite(recon) & np.isfinite(mix)
    identity_max = float(np.max(np.abs(recon[ok] - mix[ok]))) if ok.any() else None

    diag = {
        "n_oof_rows": int(len(oof)),
        "n_real_wells": int(n_real),
        "n_water_rows_all": int(n_water_all),
        "n_water_rows_nonsacrificial": int(len(water)),
        "n_real_finite_obs": n_finite,
        "n_sacrificial_dropped": int((fin_obs & sac.to_numpy()).sum()),
        "n_scored_wells": int(len(df)),
        "n_missing_baseline_join": n_missing_base,
        "n_missing_panel_join": n_missing_panel,
        "n_missing_prior_join": n_missing_prior,
        "mixture_identity_max_abs_delta_m": identity_max,
    }
    log.info("join/population: %s", json.dumps(_round(diag)))
    if identity_max is not None and identity_max > 1e-3:
        log.warning(
            "mixture identity delta %.3e m > 1e-3 m: gnn_dtw_m may not equal "
            "pi*phreatic + (1-pi)*regional (check native-target convention)",
            identity_max,
        )
    return df, water, diag


# --------------------------------------------------------------------------- #
# summary markdown
# --------------------------------------------------------------------------- #
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def _v(flag) -> str:
    return "PASS" if flag is True else ("FAIL" if flag is False else "n/a")


def write_summary(report: dict, path: Path) -> None:
    d = report["population"]
    g = report["gate_verdict"]
    lines = [
        "# WP2 two-surface arm — score summary",
        "",
        f"Scored population: {d['n_scored_wells']} real wells "
        f"(is_water_pseudo == False, finite obs_dtw_m, sacrificial HUC4s "
        f"{', '.join(SACRIFICIAL_HUC4)} dropped). Mixture point pred vs frozen "
        "baseline on the common footprint (both arms + obs finite), joined on "
        "canonical_id. All errors in metres; skills/AUC/Spearman dimensionless.",
        "",
        f"Mixture identity check max |gnn_dtw - (pi*phreatic + (1-pi)*regional)| = "
        f"{_fmt(report['join']['mixture_identity_max_abs_delta_m'], 6)} m.",
        "",
        "## Frozen WP2 gate (acceptance_gates.json, verbatim)",
        "",
        "```json",
        json.dumps(report["frozen_wp2_gate_literal"], indent=2),
        "```",
        "",
        "## Verdict per preregistered gate",
        "",
        f"- component assignment beats one-surface on nested wells: "
        f"{_v(g['assignment_beats_one_surface']['pass'])} "
        f"(AUC {_fmt(g['assignment_beats_one_surface']['auc_used'])} on "
        f"{g['assignment_beats_one_surface']['scope_used']} vs one-surface 0.5)",
        f"- deep-band error reduction >= "
        f"{_fmt(report['frozen_wp2_gate_literal']['min_deep_band_error_reduction'], 2)} "
        f"on 30+m separated panel ({g['deep_gate']['binding_panel']}): "
        f"{_v(g['deep_gate']['pass'])}",
        f"- no global component collapse "
        f"(min usage {_fmt(g['component_collapse']['min_component_usage'])} >= "
        f"{_fmt(g['component_collapse']['threshold'], 2)}): "
        f"{_v(g['component_collapse']['pass'])}",
        f"- shallow no material regression (tol "
        f"{_fmt(g['shallow_non_regression']['tolerance_frac'], 2)}): "
        f"{_v(g['shallow_non_regression']['pass'])}",
        "",
        f"### OVERALL WP2: {_v(g['overall_wp2_pass'])}",
        "",
        "## Deep gate detail (30+m binding; 10-30m diagnostic)",
        "",
        "| panel | band | n | MAD reduction | RMSE reduction | mix MAD | base MAD | pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for pname, node in g["deep_gate"]["panels"].items():
        for band_key, label in (
            ("30+m_binding", "30+m*"),
            ("10-30m_diagnostic", "10-30m"),
        ):
            b = node[band_key]
            lines.append(
                f"| {pname} | {label} | {b['n']} | {_fmt(b['mad_reduction'])} | "
                f"{_fmt(b['rmse_reduction'])} | {_fmt(b['mad_model_m'])} | "
                f"{_fmt(b['mad_base_m'])} | {_v(b['pass'])} |"
            )
    lines += [
        "",
        "(* = frozen-binding band; 10-30m shown for the design-note view only.)",
        "",
        "## Estimand-separated readout (all scored wells) — MAD by observed-depth band",
        "",
        "| surface | overall MAD | 0-2m | 2-5m | 5-10m | 10-30m | 30+m |",
        "|---|---|---|---|---|---|---|",
    ]
    est = report["estimand_separated"]["all"]["surfaces"]
    for sname, node in est.items():
        bands = node["by_depth_band"]

        def _bmad(lbl):
            return _fmt(bands.get(lbl, {}).get("mad_m"))

        lines.append(
            f"| {sname} | {_fmt(node['overall'].get('mad_m'))} | {_bmad('0-2m')} | "
            f"{_bmad('2-5m')} | {_bmad('5-10m')} | {_bmad('10-30m')} | {_bmad('30+m')} |"
        )
    ag = report["assignment_agreement"]
    lines += [
        "",
        "## Assignment agreement (OOF pi vs held-out construction evidence)",
        "",
        f"- AUC(pi vs binarized prior), all confident wells: "
        f"{_fmt(ag['auc_all_confident'].get('auc'))} "
        f"(n={ag['auc_all_confident'].get('n')}, base 0.5)",
        f"- AUC(pi vs binarized prior), nested confident wells: "
        f"{_fmt(ag['auc_nested_confident'].get('auc'))} "
        f"(n={ag['auc_nested_confident'].get('n')})",
        f"- Spearman(pi, depth_rank_q), all confident: "
        f"{_fmt(ag['spearman_all_confident'].get('spearman_pi_vs_depth_rank_q'))} "
        f"(prior baseline "
        f"{_fmt(ag['spearman_all_confident'].get('spearman_prior_vs_depth_rank_q_baseline'))})",
        "",
        "## Component collapse",
        "",
        f"- global mean pi = {_fmt(report['component_collapse']['global'].get('mean_pi'))}; "
        f"phreatic usage {_fmt(report['component_collapse']['global'].get('phreatic_usage_mean_pi'))}, "
        f"regional usage {_fmt(report['component_collapse']['global'].get('regional_usage_mean_1_minus_pi'))}",
        f"- share pi<0.05 = {_fmt(report['component_collapse']['global'].get('share_pi_lt_0.05'))}; "
        f"share pi>0.95 = {_fmt(report['component_collapse']['global'].get('share_pi_gt_0.95'))}",
        f"- water rows mean pi = {_fmt(report['component_collapse']['water_rows'].get('mean_pi'))} "
        f"(n={report['component_collapse']['water_rows'].get('n')}; should be high)",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def definitions_block() -> dict:
    return {
        "target": "obs_dtw_m = observed depth to the unconfined water table (m); "
        "identical to the contract mean_dtw.",
        "residual": "pred_dtw - obs_dtw (m; positive = predicted too deep).",
        "mad_m": "median(|residual|) (m).",
        "bias_mean_m": "mean(residual) (m).",
        "median_resid_m": "median(residual) (m).",
        "rmse_m": "sqrt(mean(residual^2)) (m).",
        "mad_skill / rmse_skill": "1 - metric_mixture/metric_frozen (dimensionless "
        "reduction; positive = mixture better than the frozen baseline).",
        "phreatic_dtw_m": "z_surf_well_m - (regional_wte_idw_oof_m + ts_native_p_m) "
        "(m); the shallow terrain-anchored component's depth.",
        "regional_dtw_m": "z_surf_well_m - (regional_wte_idw_oof_m + ts_native_r_m) "
        "(m); the deep-regional-anchored component's depth.",
        "oracle_best_component": "per-well component with the smaller |error|; the "
        "identifiability ceiling (upper bound on what perfect routing could achieve).",
        "pi_hard_assigned": "pi >= 0.5 -> phreatic_dtw else regional_dtw (m).",
        "ts_pi_phreatic": "membership pi = P(phreatic) (dimensionless in [0,1]).",
        "component_usage": "soft usage: phreatic = mean(pi), regional = mean(1-pi) "
        "(dimensionless); collapse when min < 0.05 globally.",
        "assignment_auc": "AUC of OOF pi against (p_phreatic_construction >= 0.5); "
        "honest because pi never saw the well's own prior (loss-only, out-of-fold). "
        "One-surface baseline has no membership -> implicit AUC 0.5.",
        "spearman_pi_depth_rank": "Spearman(pi, depth_rank_q); depth_rank_q rises "
        "with local completion depth so an honest pi (falling with depth) gives a "
        "NEGATIVE value, compared to the prior's own (strongly negative) baseline.",
        "separated_deep_panels": "mech_deep_far_arid (deep/far/arid regime) and "
        "panel_buffered_2p5km (support-separated: >=2.5 km from any training well).",
        "sacrificial_excluded": f"HUC4 {', '.join(SACRIFICIAL_HUC4)} dropped from "
        "every metric (frozen regional-holdout lockout).",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=RUN_DIR, help="two-surface arm output dir")
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--contract-dir", default=CONTRACT_DIR)
    ap.add_argument("--priors", default=PRIORS)
    ap.add_argument("--out-dir", default=None, help="defaults to --run-dir")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    run_dir = Path(args.run_dir)
    args.run_dir_oof = str(run_dir / "gnn_oof_predictions.parquet")
    args.panels = str(Path(args.contract_dir) / "wells_panels.parquet")
    gate = json.loads((Path(args.contract_dir) / "acceptance_gates.json").read_text())[
        "wp2_two_surface"
    ]
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, water, join = load_and_join(args)

    # ---- section 1: point non-regression + deep gate (common footprint) ----
    cf = (
        np.isfinite(df["obs_dtw_m"].to_numpy(float))
        & np.isfinite(df["gnn_dtw_m"].to_numpy(float))
        & np.isfinite(df["base_dtw_m"].to_numpy(float))
    )
    cf_df = df[cf].reset_index(drop=True)
    log.info(
        "common footprint (mixture + frozen + obs finite): %d of %d scored wells",
        int(cf.sum()),
        len(df),
    )
    panel_masks = {
        "all": np.ones(len(cf_df), bool),
        "mech_deep_far_arid": cf_df["mech_deep_far_arid"].to_numpy(bool),
        "panel_buffered_2p5km": cf_df["panel_buffered_2p5km"].to_numpy(bool),
        "mech_shallow_irrigated": cf_df["mech_shallow_irrigated"].to_numpy(bool),
    }
    point_panels = {
        name: point_non_regression_panel(cf_df, mask, name)
        for name, mask in panel_masks.items()
    }

    # ---- section 2: estimand-separated readout ----
    estimand = {
        name: estimand_panel(cf_df, mask, name) for name, mask in panel_masks.items()
    }

    # ---- section 3: component collapse ----
    collapse = collapse_check(df, water["ts_pi_phreatic"].to_numpy(float))

    # ---- section 4: assignment agreement ----
    agreement = assignment_agreement(
        df["ts_pi_phreatic"].to_numpy(float),
        df["p_phreatic_construction"].to_numpy(float),
        df["depth_rank_q"].to_numpy(float),
        confident=(
            np.abs(2.0 * df["p_phreatic_construction"].to_numpy(float) - 1.0)
            >= CONFIDENT_PRIOR_MIN
        ),
        nested=df["mech_nested_collocated"].to_numpy(bool),
    )

    # ---- section 5: gate verdict ----
    verdict = evaluate_gates(gate, point_panels, collapse, agreement)

    report = {
        "work_package": "WP2 two-surface mixture score (v2 plan §7)",
        "run_dir": str(run_dir),
        "baseline": str(args.baseline),
        "contract_dir": str(args.contract_dir),
        "priors": str(args.priors),
        "definitions": definitions_block(),
        "frozen_wp2_gate_literal": gate,
        "join": join,
        "population": {
            "n_scored_wells": int(len(df)),
            "n_common_footprint": int(cf.sum()),
            "n_folds": int(df["cv_fold"].nunique()),
            "sacrificial_huc4_excluded": list(SACRIFICIAL_HUC4),
        },
        "point_non_regression": point_panels,
        "estimand_separated": estimand,
        "component_collapse": collapse,
        "assignment_agreement": agreement,
        "gate_verdict": verdict,
    }
    report = _round(report)
    # write_summary reads a couple of raw fields from the report
    report["join"]["mixture_identity_max_abs_delta_m"] = join[
        "mixture_identity_max_abs_delta_m"
    ]

    out_report = out_dir / "two_surface_score_report.json"
    out_report.write_text(json.dumps(report, indent=2, default=str))
    out_md = out_dir / "two_surface_score_summary.md"
    write_summary(report, out_md)

    log.info("wrote %s", out_report)
    log.info("wrote %s", out_md)
    v = verdict
    log.info(
        "WP2 verdict: assignment=%s deep=%s collapse=%s shallow=%s -> OVERALL %s",
        _v(v["assignment_beats_one_surface"]["pass"]),
        _v(v["deep_gate"]["pass"]),
        _v(v["component_collapse"]["pass"]),
        _v(v["shallow_non_regression"]["pass"]),
        _v(v["overall_wp2_pass"]),
    )


if __name__ == "__main__":
    main()
