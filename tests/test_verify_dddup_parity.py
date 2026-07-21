"""Unit tests for the canonical-vs-derived dd/dup parity comparator
(utils/verify_dddup_parity.py). Loads the module by path (repo test convention).

Covers the comparator's core guarantees: (1) identical values joined on canonical_id
pass even when row ORDER differs; (2) a value drift beyond tolerance fails; (3) a
divergent NaN pattern fails even where finite values agree; (4) a column missing from
one frame fails.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vp = _load("verify_dddup_parity")


def _frame(ids, vals, nan_col=None):
    df = pd.DataFrame({"canonical_id": ids, "drilled_depth_idw_m": vals})
    if nan_col is not None:
        df["dupuit_hang_dtw_m"] = nan_col
    return df


def test_identical_values_pass_despite_row_order():
    a = _frame(["gwx_1", "gwx_2", "water_3"], [1.0, 2.0, 3.0])
    # same canonical_id -> value mapping, shuffled order
    b = _frame(["water_3", "gwx_1", "gwx_2"], [3.0, 1.0, 2.0])
    rep = vp.compare_columns(a, b, ["drilled_depth_idw_m"])
    assert rep["passed"]
    assert rep["n_common"] == 3
    assert rep["columns"]["drilled_depth_idw_m"]["max_abs_diff"] == 0.0


def test_value_drift_beyond_tolerance_fails():
    a = _frame(["gwx_1", "gwx_2"], [1.0, 2.0])
    b = _frame(["gwx_1", "gwx_2"], [1.0, 2.01])  # 0.01 m drift > atol 1e-6
    rep = vp.compare_columns(a, b, ["drilled_depth_idw_m"])
    assert not rep["passed"]
    assert not rep["columns"]["drilled_depth_idw_m"]["allclose"]
    assert rep["columns"]["drilled_depth_idw_m"]["max_abs_diff"] > 0.0


def test_divergent_nan_pattern_fails():
    a = _frame(["gwx_1", "gwx_2"], [1.0, 2.0], nan_col=[np.nan, 5.0])
    b = _frame(
        ["gwx_1", "gwx_2"], [1.0, 2.0], nan_col=[4.0, 5.0]
    )  # NaN vs finite at row 0
    rep = vp.compare_columns(a, b, ["dupuit_hang_dtw_m"])
    assert not rep["passed"]
    assert not rep["columns"]["dupuit_hang_dtw_m"]["nan_pattern_match"]


def test_matching_nan_pattern_passes():
    a = _frame(["gwx_1", "gwx_2"], [1.0, 2.0], nan_col=[np.nan, 5.0])
    b = _frame(["gwx_2", "gwx_1"], [2.0, 1.0], nan_col=[5.0, np.nan])
    rep = vp.compare_columns(a, b, ["dupuit_hang_dtw_m"])
    assert rep["passed"]


def test_missing_column_fails():
    a = _frame(["gwx_1"], [1.0])
    b = _frame(["gwx_1"], [1.0])
    rep = vp.compare_columns(a, b, ["nonexistent_col"])
    assert not rep["passed"]
    col = rep["columns"]["nonexistent_col"]
    assert not col["present_a"] and not col["present_b"]
