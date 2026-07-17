"""Unit tests for the WP3 temporal-alias census numeric core (synthetic, no disk).

Exercises the alias-innovation link, leave-one-fold-out beta fit, and the
correction metrics. The controlling requirement (plan §8): a population whose
innovations ARE exactly the (negated) implied drift must show R^2 ~ 1 and a
large MAD reduction under the honest OLS-fit correction, while an uncorrelated
population must show neither.
"""

import importlib.util
from pathlib import Path

import numpy as np


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tac = _load("v02_temporal_alias_census")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_mad_and_rmse_basic():
    assert abs(tac.mad(np.array([-3.0, -1.0, 1.0, 3.0])) - 2.0) < 1e-9
    assert abs(tac.rmse(np.array([3.0, 4.0])) - np.sqrt(12.5)) < 1e-9
    # non-finite entries are dropped, not patched
    assert abs(tac.mad(np.array([np.nan, 2.0, -2.0])) - 2.0) < 1e-9


def test_ols_slope_recovers_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, 5000)
    y = -1.7 * x + rng.normal(0, 0.01, 5000)
    assert abs(tac.ols_slope(x, y) - (-1.7)) < 0.01


# --------------------------------------------------------------------------- #
# perfect alias: innovation == -drift  ->  R^2 ~ 1, large MAD reduction
# --------------------------------------------------------------------------- #
def test_perfect_alias_link_and_correction():
    rng = np.random.default_rng(0)
    n = 3000
    drift = rng.normal(0.0, 3.0, n)
    innovation = -drift  # innovations ARE exactly -drift
    fold = rng.integers(0, 8, n)

    link = tac.alias_innovation_link(innovation, drift)
    assert link["ols_r2"] > 0.99
    assert abs(link["pearson_r"] - (-1.0)) < 0.01
    assert abs(link["ols_slope"] - (-1.0)) < 0.01

    beta = tac.fit_beta_loo(innovation, drift, fold)
    assert np.all(np.abs(beta - (-1.0)) < 1e-6)  # each LOO fit recovers -1
    harm = innovation - beta * drift
    m = tac.correction_metrics(innovation, harm)
    assert m["mad_reduction_frac"] > 0.99  # harmonized collapses to ~0
    assert m["rmse_reduction_frac"] > 0.99


def test_bootstrap_ci_brackets_perfect_slope():
    rng = np.random.default_rng(1)
    drift = rng.normal(0.0, 3.0, 2000)
    innovation = -drift
    lo, hi = tac.bootstrap_ols_slope_ci(drift, innovation, n_boot=300, seed=0)
    assert lo <= -1.0 <= hi


# --------------------------------------------------------------------------- #
# uncorrelated: innovation independent of drift  ->  R^2 ~ 0, no MAD reduction
# --------------------------------------------------------------------------- #
def test_uncorrelated_population_shows_no_alias():
    rng = np.random.default_rng(2)
    n = 3000
    drift = rng.normal(0.0, 3.0, n)
    innovation = rng.normal(0.0, 3.0, n)  # independent of drift
    fold = rng.integers(0, 8, n)

    link = tac.alias_innovation_link(innovation, drift)
    assert link["ols_r2"] < 0.05

    beta = tac.fit_beta_loo(innovation, drift, fold)
    harm = innovation - beta * drift
    m = tac.correction_metrics(innovation, harm)
    assert abs(m["mad_reduction_frac"]) < 0.10


def test_empty_and_degenerate_inputs_are_safe():
    # no finite pairs -> link returns Nones, not a crash
    link = tac.alias_innovation_link(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
    assert link["ols_r2"] is None
    # zero-variance drift -> slope undefined (nan), not an inf/exception
    assert not np.isfinite(tac.ols_slope(np.ones(10), np.arange(10.0)))
