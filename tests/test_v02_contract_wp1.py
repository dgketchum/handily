"""Unit tests for the v0.2 evaluation contract (WP0) and WP1 recoverability harness.

Covers the leakage-critical mechanics: physical-site union-find, constrained
nearest-context distances, the empty-context zero-increment identity, fold+site
exclusion in the LOO predictors, PSD correlation-model recovery, and the
probabilistic metric panel (Laplace CRPS / interval coverage / paired skill).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vm = _load("v02_metrics")
fc = _load("v02_freeze_eval_contract")
ir = _load("v02_innovation_recoverability")


# --------------------------------------------------------------------------- #
# v02_metrics
# --------------------------------------------------------------------------- #
def test_laplace_crps_at_perfect_center_is_quarter_scale():
    out = vm.laplace_crps(np.zeros(4), np.full(4, 2.0), np.zeros(4))
    assert abs(out["crps_mean_m"] - 0.5) < 1e-9  # b/4 = 0.5


def test_interval_coverage_matches_laplace_draws():
    rng = np.random.default_rng(0)
    b = 3.0
    y = rng.laplace(0.0, b, size=200_000)
    out = vm.interval_panel_laplace(np.zeros_like(y), np.full_like(y, b), y)
    assert abs(out["cov_50"] - 0.5) < 0.01
    assert abs(out["cov_90"] - 0.9) < 0.01


def test_paired_improvement_halved_error_gives_half_skill():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 4, 5000)
    out = vm.paired_improvement(base / 2.0, base, n_boot=200)
    assert abs(out["mad_skill"] - 0.5) < 0.02
    lo, hi = out["mad_skill_ci95"]
    assert lo <= 0.5 <= hi


def test_prob_class_panel_perfect_and_prevalence():
    y = np.array([True] * 30 + [False] * 70)
    perfect = vm.prob_class_panel(y.astype(float), y)
    assert perfect["brier"] == 0.0 and perfect["brier_skill"] == 1.0
    prev = vm.prob_class_panel(np.full(100, 0.3), y)
    assert abs(prev["brier_skill"]) < 1e-9


def test_sigma_monotonicity_detects_heteroscedastic_error():
    rng = np.random.default_rng(2)
    s = rng.uniform(0.5, 5.0, 4000)
    y = rng.normal(0, s)
    out = vm.sigma_monotonicity(np.zeros_like(y), s, y)
    assert out["spearman_sigma_abs_err"] > 0.3


# --------------------------------------------------------------------------- #
# WP0 identity + panels mechanics
# --------------------------------------------------------------------------- #
def test_union_find_sites_transitive_closure():
    # chain a-b-c each 80 m apart: all one site despite a-c being 160 m apart
    xy = np.array([[0.0, 0.0], [80.0, 0.0], [160.0, 0.0], [5000.0, 0.0]])
    site = fc.union_find_sites(xy, 100.0)
    assert site[0] == site[1] == site[2]
    assert site[3] != site[0]


def test_nearest_constrained_km_excludes_same_site_and_fold():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [1000.0, 0.0], [3000.0, 0.0]])
    site = np.array([0, 0, 1, 2])
    fold = np.array([0, 1, 0, 1])
    d_any = fc.nearest_constrained_km(xy, False, fold, site)
    # well 0: same-site well at 10 m is skipped -> nearest is 1000 m
    assert abs(d_any[0] - 1.0) < 1e-6
    d_train = fc.nearest_constrained_km(xy, True, fold, site)
    # well 0 (fold 0): well 2 shares the fold -> nearest usable is 3000 m
    assert abs(d_train[0] - 3.0) < 1e-6


# --------------------------------------------------------------------------- #
# WP1 harness mechanics
# --------------------------------------------------------------------------- #
def _toy_df(n=200, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 50_000, n)
    y = rng.uniform(0, 50_000, n)
    smooth = 10.0 * np.sin(x / 20_000.0) + 5.0 * np.cos(y / 15_000.0)
    innov = smooth + rng.normal(0, 1.0, n)
    return pd.DataFrame(
        {
            "x5070": x,
            "y5070": y,
            "innovation_m": innov,
            "z_surf_well_m": rng.uniform(100, 200, n),
            "site_id": np.arange(n),
            "cv_fold": rng.integers(0, 4, n),
        }
    )


def test_idw_loo_recovers_smooth_field():
    df = _toy_df()
    pred, dist_used = ir.idw_loo(df, k=8, power=2.0, buffer_km=0.0)
    innov = df["innovation_m"].to_numpy()
    assert np.isfinite(pred).all()
    mad_base = np.median(np.abs(innov))
    mad_pred = np.median(np.abs(innov - pred))
    assert mad_pred < 0.6 * mad_base  # smooth signal is recoverable


def test_idw_loo_empty_context_returns_zero_increment():
    df = _toy_df(n=5)
    df["cv_fold"] = 0  # single fold: every context pool is empty
    pred, dist_used = ir.idw_loo(df, k=8, power=2.0, buffer_km=0.0)
    assert np.allclose(pred, 0.0)
    assert np.isnan(dist_used).all()


def test_idw_loo_site_exclusion():
    # two coincident wells at the same site in different folds: the sibling must
    # not be used as context, so the prediction cannot equal its value exactly
    df = _toy_df(n=50)
    df.loc[0, ["x5070", "y5070"]] = [25_000.0, 25_000.0]
    df.loc[1, ["x5070", "y5070"]] = [25_001.0, 25_000.0]
    df.loc[0, "cv_fold"], df.loc[1, "cv_fold"] = 0, 1
    df.loc[[0, 1], "site_id"] = 999
    df.loc[1, "innovation_m"] = 500.0  # a leak would drag pred_0 toward 500
    pred, _ = ir.idw_loo(df, k=4, power=2.0, buffer_km=0.0)
    assert abs(pred[0]) < 100.0


def test_krige_loo_empty_context_returns_background_sigma():
    df = _toy_df(n=5)
    df["cv_fold"] = 0
    fm = {
        0: {"exponential": {"nugget_frac": 0.4, "range_km": 30.0, "sill_var_m2": 9.0}}
    }
    pred, sig, dist_used = ir.krige_loo(df, fm, 0.0)
    assert np.allclose(pred, 0.0)
    assert np.allclose(sig, 3.0)


def test_krige_loo_reduces_error_and_sigma_below_background():
    df = _toy_df()
    var = float(df["innovation_m"].var())
    fm = {
        f: {"exponential": {"nugget_frac": 0.1, "range_km": 20.0, "sill_var_m2": var}}
        for f in range(4)
    }
    pred, sig, dist_used = ir.krige_loo(df, fm, 0.0)
    innov = df["innovation_m"].to_numpy()
    assert np.median(np.abs(innov - pred)) < np.median(np.abs(innov))
    assert (sig[np.isfinite(dist_used)] < np.sqrt(var)).all()


def test_fit_correlation_models_recovers_exponential():
    h = np.linspace(0.5, 200, 40)
    nug, rng_km = 0.3, 60.0
    cg = {
        "bin_mid_km": h.tolist(),
        "rho": ((1 - nug) * np.exp(-h / rng_km)).tolist(),
        "n_pairs": [10_000] * len(h),
    }
    fits = ir.fit_correlation_models(cg, var=25.0)
    assert abs(fits["exponential"]["nugget_frac"] - nug) < 0.02
    assert abs(fits["exponential"]["range_km"] - rng_km) < 3.0
    assert fits["exponential"]["mappable_variance_fraction"] > 0.65
