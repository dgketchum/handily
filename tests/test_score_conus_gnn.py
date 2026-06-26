"""Unit tests for utils/score_conus_gnn.py.

Covers the handoff-mandated additions: depth-band error STRUCTURE (RMSE / p90-95 /
catastrophic fractions, not MAD alone) and the aquifer-router gate diagnostics
(the fail-flat monitor). A 30+ m band where RMSE >> MAD is the catastrophic tail
the acceptance criteria turn on, so we assert the split is actually surfaced.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_MOD = Path(__file__).resolve().parents[1] / "utils" / "score_conus_gnn.py"
_spec = importlib.util.spec_from_file_location("score_conus_gnn", _MOD)
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def test_depth_banded_emits_rmse_p95_and_catastrophic_fracs():
    # one shallow band + one deep band, deep band has a catastrophic outlier.
    obs = np.array([1.0, 1.0, 1.0, 40.0, 40.0, 40.0, 40.0])
    pred = np.array([1.5, 0.5, 1.0, 41.0, 39.0, 40.0, 140.0])  # +100 m miss in deep
    bands = sc.depth_banded(pred, obs)
    deep = bands["30+m"]
    for k in (
        "mad_m",
        "median_resid_m",
        "bias_mean_m",
        "rmse_m",
        "p90_abs_err_m",
        "p95_abs_err_m",
        "frac_abs_err_gt_10m",
        "frac_abs_err_gt_25m",
    ):
        assert k in deep, f"missing {k}"
    # one of four deep wells misses by 100 m -> exactly 1/4 over both thresholds.
    assert deep["frac_abs_err_gt_10m"] == 0.25
    assert deep["frac_abs_err_gt_25m"] == 0.25


def test_depth_band_rmse_exceeds_mad_on_high_variance_tail():
    # MAD suppresses the tail; RMSE must expose it (the whole point of the 30+ band).
    obs = np.full(8, 40.0)
    pred = np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 200.0])  # one huge miss
    deep = sc.depth_banded(pred, obs)["30+m"]
    assert deep["mad_m"] == 0.0  # median abs err is 0
    assert deep["rmse_m"] > 50.0  # but RMSE catches the 160 m miss
    assert deep["rmse_m"] > deep["mad_m"]


def test_gate_diagnostics_drop_nan_and_band_by_depth():
    df = pd.DataFrame(
        {
            "aquifer_gate": [0.01, 0.02, np.nan, 0.8, 0.9, 0.85],
            "obs_dtw_m": [1.0, 1.5, 1.0, 40.0, 50.0, 45.0],
            "gnn_dtw_m": [1.2, 1.4, 1.1, 41.0, 49.0, 44.0],
            "huc2": ["13", "13", "13", "13", "13", "13"],
        }
    )
    diag = sc.gate_diagnostics(df, "aquifer_gate")
    # NaN gate row is dropped from the finite count.
    assert diag["n_finite_gate"] == 5
    shallow = diag["by_obs_depth"]["0-2m"]
    deep = diag["by_obs_depth"]["30+m"]
    # the NaN-gate well sits in the shallow band, so it has 2 finite, not 3.
    assert shallow["n"] == 2
    assert deep["n"] == 3
    # gate localizes the deep regime (low shallow, high deep) -- the intended pattern.
    assert shallow["mean_gate"] < 0.1
    assert deep["mean_gate"] > 0.5


def test_gate_diagnostics_absent_column_returns_empty():
    df = pd.DataFrame({"obs_dtw_m": [1.0], "gnn_dtw_m": [1.0]})
    assert sc.gate_diagnostics(df, "aquifer_gate") == {}
