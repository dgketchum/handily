"""Shared metric panel for the Handily v0.2 evaluation contract (v2 plan WP0 §5.2).

Reuses the deterministic point metrics from ``score_conus_gnn.py`` (``core_metrics``,
``depth_banded``, ``shallow_skill``) and adds what the v0.2 plan requires on top:
distance-to-context bands, Brier / PR-AUC for shallow-class probabilities, Laplace
CRPS, central-interval coverage and sharpness for the sigma head, error-vs-sigma
monotonicity, and coverage/abstention accounting.

Conventions (fixed across all v0.2 arms):
- every error metric is in metres unless suffixed otherwise; residual = pred - obs
  (positive = predicted too deep, DTW space), matching ``score_conus_gnn``;
- skill = 1 - metric_model / metric_baseline (dimensionless fraction; positive =
  better than baseline), stated per use;
- distance-to-context bands are km to the nearest ELIGIBLE context well (eligibility
  defined by the caller's mask contract, WP1 §6.3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_conus_gnn import core_metrics  # noqa: E402

DIST_CONTEXT_BANDS_KM = [0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, np.inf]


def _band_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}km" if np.isfinite(hi) else f"{lo:g}+km"


def dist_banded(
    pred: np.ndarray, obs: np.ndarray, dist_km: np.ndarray, bands=None
) -> dict:
    """Central metrics stratified by distance to nearest eligible context well (km)."""
    bands = DIST_CONTEXT_BANDS_KM if bands is None else bands
    out = {}
    d = np.asarray(dist_km, float)
    for lo, hi in zip(bands[:-1], bands[1:]):
        m = np.isfinite(d) & (d >= lo) & (d < hi)
        out[_band_label(lo, hi)] = core_metrics(pred[m], obs[m])
    return out


def paired_improvement(
    err_model: np.ndarray, err_base: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Relative improvement of model over baseline on the SAME rows.

    Reports MAD and RMSE for both, plus skill = 1 - model/baseline (dimensionless;
    positive = model better) with a bootstrap 95% CI on the MAD skill (percentile
    CI over well resamples).
    """
    e_m = np.abs(np.asarray(err_model, float))
    e_b = np.abs(np.asarray(err_base, float))
    ok = np.isfinite(e_m) & np.isfinite(e_b)
    e_m, e_b = e_m[ok], e_b[ok]
    n = len(e_m)
    if n == 0:
        return {"n": 0}
    mad_m, mad_b = float(np.median(e_m)), float(np.median(e_b))
    rmse_m = float(np.sqrt(np.mean(e_m**2)))
    rmse_b = float(np.sqrt(np.mean(e_b**2)))
    rng = np.random.default_rng(seed)
    sk = np.empty(n_boot)
    chunk = max(1, min(n_boot, int(2e7 // max(n, 1))))
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - s), n))
        sk[s : s + len(idx)] = 1.0 - np.median(e_m[idx], axis=1) / np.maximum(
            np.median(e_b[idx], axis=1), 1e-9
        )
    return {
        "n": n,
        "mad_model_m": round(mad_m, 4),
        "mad_base_m": round(mad_b, 4),
        "mad_skill": round(1.0 - mad_m / max(mad_b, 1e-9), 4),
        "mad_skill_ci95": [round(float(q), 4) for q in np.percentile(sk, [2.5, 97.5])],
        "rmse_model_m": round(rmse_m, 4),
        "rmse_base_m": round(rmse_b, 4),
        "rmse_skill": round(1.0 - rmse_m / max(rmse_b, 1e-9), 4),
    }


# --------------------------------------------------------------------------- #
# probabilistic metrics
# --------------------------------------------------------------------------- #
def prob_class_panel(p: np.ndarray, y: np.ndarray) -> dict:
    """Brier / PR-AUC panel for one binary shallow class.

    p = predicted probability, y = observed class (bool). brier_skill = 1 -
    brier / brier_prevalence (dimensionless; positive = beats the
    prevalence-only forecast).
    """
    from sklearn.metrics import average_precision_score

    p = np.asarray(p, float)
    y = np.asarray(y, bool)
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    n = len(p)
    if n == 0 or y.sum() in (0, n):
        return {"n": int(n), "degenerate": True}
    base = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    brier_ref = base * (1 - base)
    return {
        "n": int(n),
        "base_rate": round(base, 4),
        "brier": round(brier, 5),
        "brier_ref_prevalence": round(brier_ref, 5),
        "brier_skill": round(1.0 - brier / brier_ref, 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
    }


def laplace_crps(pred: np.ndarray, scale_b: np.ndarray, obs: np.ndarray) -> dict:
    """Mean CRPS (m) for Laplace(pred, b) predictive distributions.

    Closed form: CRPS = |y-mu| + b*exp(-|y-mu|/b) - 3b/4. Lower is better;
    same units as the target (m).
    """
    mu = np.asarray(pred, float)
    b = np.asarray(scale_b, float)
    y = np.asarray(obs, float)
    ok = np.isfinite(mu) & np.isfinite(b) & np.isfinite(y) & (b > 0)
    a = np.abs(y[ok] - mu[ok])
    crps = a + b[ok] * np.exp(-a / b[ok]) - 0.75 * b[ok]
    return {"n": int(ok.sum()), "crps_mean_m": round(float(np.mean(crps)), 4)}


def interval_panel_laplace(
    pred: np.ndarray,
    scale_b: np.ndarray,
    obs: np.ndarray,
    levels=(0.5, 0.8, 0.9),
) -> dict:
    """Central-interval coverage + sharpness for Laplace predictive intervals.

    For level a the central interval half-width is t = -b*ln(1-a); coverage is
    the fraction of |resid| <= t (target: == a); sharpness is the mean full
    width 2t in metres (smaller = sharper, only meaningful at honest coverage).
    """
    mu = np.asarray(pred, float)
    b = np.asarray(scale_b, float)
    y = np.asarray(obs, float)
    ok = np.isfinite(mu) & np.isfinite(b) & np.isfinite(y) & (b > 0)
    a_err = np.abs(y[ok] - mu[ok])
    out = {"n": int(ok.sum())}
    for lv in levels:
        t = -b[ok] * np.log(1.0 - lv)
        out[f"cov_{int(lv * 100)}"] = round(float(np.mean(a_err <= t)), 4)
        out[f"sharp_{int(lv * 100)}_m"] = round(float(np.mean(2.0 * t)), 4)
    return out


def sigma_monotonicity(
    pred: np.ndarray, sigma: np.ndarray, obs: np.ndarray, n_bins: int = 10
) -> dict:
    """Is error monotone in predicted scale? (WP1 gate: error monotone with
    predicted scale.) Reports median |err| per sigma-decile and the Spearman
    rank correlation between sigma and |err| (dimensionless)."""
    from scipy.stats import spearmanr

    mu = np.asarray(pred, float)
    s = np.asarray(sigma, float)
    y = np.asarray(obs, float)
    ok = np.isfinite(mu) & np.isfinite(s) & np.isfinite(y)
    a_err, s = np.abs(y[ok] - mu[ok]), s[ok]
    if len(s) < n_bins * 5:
        return {"n": int(len(s)), "degenerate": True}
    q = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    q[-1] += 1e-9
    med = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (s >= lo) & (s < hi)
        med.append(round(float(np.median(a_err[m])), 4) if m.any() else None)
    rho = float(spearmanr(s, a_err).statistic)
    return {
        "n": int(len(s)),
        "median_abs_err_by_sigma_decile_m": med,
        "spearman_sigma_abs_err": round(rho, 4),
    }


def coverage_abstention(n_population: int, n_scored: int) -> dict:
    """Coverage/abstention accounting (WP0 §5.2). Fractions are dimensionless."""
    return {
        "n_population": int(n_population),
        "n_scored": int(n_scored),
        "coverage_frac": round(n_scored / max(n_population, 1), 4),
        "abstention_frac": round(1.0 - n_scored / max(n_population, 1), 4),
    }
