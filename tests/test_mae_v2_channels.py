"""Tests for the v2 payload channel derivation: Landsat index math and band mapping."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v2c = _load("build_mae_v2_channels")

BAND_NAMES = [f"p{p}_b{b}" for p in range(5) for b in (2, 3, 4, 5, 6, 7, 10)]


def _stack(values: dict) -> np.ndarray:
    """35-band [35,2,2] stack, constant per band from a {band_name: value} dict."""
    s = np.zeros((35, 2, 2), "float32")
    for i, n in enumerate(BAND_NAMES):
        s[i] = values.get(n, 1.0)
    return s


def test_landsat_index_values():
    vals = {}
    # ndvi per period = (b5-b4)/(b5+b4): make period p have ndvi = p/10
    for p in range(5):
        vals[f"p{p}_b4"] = 10.0 - p
        vals[f"p{p}_b5"] = 10.0 + p  # nd = 2p/20 = p/10
        vals[f"p{p}_b10"] = 280.0 + 5 * p
    vals["p2_b6"] = 6.0  # ndmi_p2 = (12-6)/18, mndwi_p2 = (b3-6)/(b3+6)
    vals["p2_b3"] = 3.0
    idx = v2c.compute_landsat_indices(_stack(vals), BAND_NAMES)
    assert idx.shape == (10, 2, 2)
    named = dict(zip(v2c.INDEX_NAMES, idx[:, 0, 0]))
    for p in range(5):
        assert named[f"lst_ndvi_p{p}"] == pytest.approx(p / 10, abs=1e-6)
    assert named["lst_ndvi_amp"] == pytest.approx(0.4, abs=1e-6)
    assert named["lst_ndmi_p2"] == pytest.approx(6.0 / 18.0, abs=1e-6)
    assert named["lst_mndwi_p2"] == pytest.approx(-3.0 / 9.0, abs=1e-6)
    assert named["lst_b10_p2_k"] == pytest.approx(290.0)
    assert named["lst_b10_amp_k"] == pytest.approx(20.0)


def test_landsat_index_nan_propagation():
    vals = {f"p{p}_b{b}": np.nan for p in range(5) for b in (2, 3, 4, 5, 6, 7, 10)}
    idx = v2c.compute_landsat_indices(_stack(vals), BAND_NAMES)
    assert np.isnan(idx).all()


def test_zero_denominator_is_nan():
    a = np.array([[0.0]])
    assert np.isnan(v2c._nd(a, a)).all()
