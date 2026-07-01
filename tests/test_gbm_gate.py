"""Unit tests for utils/gbm_gate_wte_residual.py.

Covers the gate's decision contract on a synthetic wte_residual query table:
  * a candidate column that IS the residual target (+ noise) clears the gate;
  * a pure-noise candidate does not move R^2 -> the gate fails;
  * a NaN-bearing candidate is reported in finite_fraction (never silently imputed),
    and the GBM's native NaN handling still lets a real signal pass.
The GBM is shrunk (max_iter=60) so the tests run fast; the decision logic is
config-independent.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_MOD = Path(__file__).resolve().parents[1] / "utils" / "gbm_gate_wte_residual.py"
_spec = importlib.util.spec_from_file_location("gbm_gate_wte_residual", _MOD)
gg = importlib.util.module_from_spec(_spec)
sys.modules["gbm_gate_wte_residual"] = gg
_spec.loader.exec_module(gg)


def _fast_model():
    return gg._hgb(max_iter=60, learning_rate=0.1, min_samples_leaf=20)


def _synthetic(n=1500, seed=0):
    """A residual target with a weak baseline feature + a strong candidate.

    obs_dtw = dtw_base - resid so a perfect resid_hat reconstructs DTW exactly; the
    baseline feature is weakly informative, ``good_cand`` ~ target, ``bad_cand`` noise.
    """
    rng = np.random.RandomState(seed)
    signal = rng.normal(0.0, 5.0, n)
    resid = signal + rng.normal(0.0, 1.0, n)  # residual target
    x0 = 0.3 * signal + rng.normal(0.0, 3.0, n)  # weak baseline feature
    good = resid + rng.normal(0.0, 0.5, n)  # candidate ~ target (strong)
    bad = rng.normal(0.0, 5.0, n)  # pure-noise candidate
    dtw_base = rng.normal(20.0, 5.0, n)
    return pd.DataFrame(
        {
            "wte_residual_m": resid,
            "x0": x0,
            "good_cand": good,
            "bad_cand": bad,
            "wte_resid_base_m": dtw_base,
            "mean_dtw": dtw_base - resid,
            "cv_fold": rng.randint(0, 4, n),
        }
    )


def _run(df, candidate_cols):
    return gg.run_gate(
        df,
        ["x0"],
        candidate_cols,
        target_col="wte_residual_m",
        dtw_base_col="wte_resid_base_m",
        obs_dtw_col="mean_dtw",
        fold_col="cv_fold",
        make_model=_fast_model,
        perm_repeats=2,
        seed=1,
    )


def test_gate_passes_when_candidate_is_target():
    res = _run(_synthetic(), ["good_cand"])
    assert res["gate"]["overall_ok"]  # ΔR^2 >> 0.02
    assert res["gate"]["passes"]
    assert res["delta_r2"] > gg.GATE_R2_DELTA
    # candidate improves the reconstructed-DTW MAD too.
    assert (
        res["panel_candidate"]["core"]["mad_m"] < res["panel_baseline"]["core"]["mad_m"]
    )
    # and the permutation importance flags it as the load-bearing column.
    assert res["permutation_importance"]["good_cand"] > 0.05


def test_gate_fails_on_pure_noise_candidate():
    res = _run(_synthetic(), ["bad_cand"])
    assert not res["gate"]["passes"]
    assert res["delta_r2"] < gg.GATE_R2_DELTA
    # a noise column carries ~no importance (allow a small negative from reuse noise).
    assert res["permutation_importance"]["bad_cand"] < 0.02


def test_nan_candidate_reported_not_imputed():
    df = _synthetic()
    df.loc[df.index[::10], "good_cand"] = np.nan  # 10% missing
    res = _run(df, ["good_cand"])
    # finite-fraction is reported (not silently 1.0), and no imputation happened.
    assert 0.85 < res["finite_fraction"]["good_cand"] < 0.95
    assert df["good_cand"].isna().sum() > 0  # the input frame is untouched
    # native GBM NaN handling still lets a real signal pass.
    assert res["gate"]["passes"]


def test_run_gate_dedups_candidate_already_in_baseline():
    # a candidate that duplicates a baseline column must not crash the feature list.
    df = _synthetic()
    res = gg.run_gate(
        df,
        ["x0", "good_cand"],
        ["good_cand"],  # already a baseline feature
        target_col="wte_residual_m",
        dtw_base_col="wte_resid_base_m",
        obs_dtw_col="mean_dtw",
        fold_col="cv_fold",
        make_model=_fast_model,
        perm_repeats=1,
        seed=1,
    )
    # baseline already contains the signal, so adding the dup gains ~nothing.
    assert abs(res["delta_r2"]) < 0.02
