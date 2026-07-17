"""Unit tests for the WP5 shallow-water ordinal controls (v2 plan §10).

Synthetic small-data only -- no disk / contract dependencies. Covers the four
mechanics the report leans on: prevalence-only calibration is skill-neutral,
the OOF logistic recovers a separable feature, the Laplace CDF is correct at a
known quantile, and the ordinal-monotonicity checker flags a nesting violation.
"""

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


soc = _load("v02_shallow_ordinal_controls")
vm = _load("v02_metrics")


def test_prevalence_oof_is_skill_neutral():
    rng = np.random.default_rng(0)
    y = (rng.random(6000) < 0.3).astype(int)
    folds = rng.integers(0, 8, 6000)
    p = soc.prevalence_oof(y, folds)
    # per-fold constant ~ overall prevalence -> Brier skill ~ 0
    assert np.all(np.isfinite(p))
    panel = vm.prob_class_panel(p, y.astype(bool))
    assert abs(panel["brier_skill"]) < 0.02


def test_logistic_oof_learns_separable_feature():
    rng = np.random.default_rng(1)
    n = 2000
    y = (rng.random(n) < 0.4).astype(int)
    signal = y + rng.normal(0, 0.3, n)  # strongly separating feature
    noise = rng.normal(0, 1, (n, 3))
    X = np.column_stack([signal, noise])
    folds = rng.integers(0, 5, n)
    p = soc.logistic_oof(X, y, folds)
    assert np.all(np.isfinite(p))
    assert average_precision_score(y, p) > 0.9


def test_laplace_cdf_known_quantiles():
    b = 3.0
    loc = 10.0
    # median at loc; loc +/- b*ln2 are the 0.75 / 0.25 quantiles
    assert abs(float(soc.laplace_cdf(loc, loc, b)) - 0.5) < 1e-12
    assert abs(float(soc.laplace_cdf(loc + b * np.log(2), loc, b)) - 0.75) < 1e-12
    assert abs(float(soc.laplace_cdf(loc - b * np.log(2), loc, b)) - 0.25) < 1e-12


def test_laplace_cdf_is_monotone_in_threshold():
    loc = np.array([5.0, 20.0])
    scale = np.array([2.0, 8.0])
    p2 = soc.laplace_cdf(2.0, loc, scale)
    p5 = soc.laplace_cdf(5.0, loc, scale)
    p10 = soc.laplace_cdf(10.0, loc, scale)
    assert np.all(p2 <= p5) and np.all(p5 <= p10)


def test_ordinal_monotonicity_flags_violation():
    # rows 0,1 monotone; row 2 violates p5 < p2
    p2 = np.array([0.1, 0.2, 0.6])
    p5 = np.array([0.3, 0.4, 0.4])
    p10 = np.array([0.5, 0.9, 0.8])
    out = soc.ordinal_monotonicity(p2, p5, p10)
    assert out["n"] == 3
    assert out["n_violations"] == 1
    assert abs(out["frac_monotone"] - 2 / 3) < 1e-6


def test_ordinal_monotonicity_all_pass():
    p2 = np.array([0.1, 0.2])
    p5 = np.array([0.2, 0.5])
    p10 = np.array([0.9, 0.9])
    out = soc.ordinal_monotonicity(p2, p5, p10)
    assert out["frac_monotone"] == 1.0 and out["n_violations"] == 0
