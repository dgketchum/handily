"""Unit tests for the pure-math helpers in utils/v02_score_two_surface.py.

No disk dependencies: exercise the component-DTW conversion, the oracle-best
component selection (including single-finite fallbacks) and the pi-hard
assignment, plus the mixture identity that ties them to the blended point pred.
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


sc = _load("v02_score_two_surface")


def test_component_dtw_formula():
    # DTW = z_surf - (R + ts_native); elementwise
    z = np.array([100.0, 50.0, 200.0])
    r = np.array([90.0, 45.0, 150.0])
    ts = np.array([2.0, -3.0, 10.0])
    got = sc.component_dtw(z, r, ts)
    assert np.allclose(got, z - (r + ts))
    assert np.allclose(got, [8.0, 8.0, 40.0])


def test_component_dtw_nan_propagates():
    got = sc.component_dtw([100.0, np.nan], [90.0, 90.0], [1.0, 1.0])
    assert np.isclose(got[0], 9.0)
    assert not np.isfinite(got[1])


def test_oracle_best_picks_smaller_error():
    obs = np.array([10.0, 10.0, 10.0])
    phreatic = np.array([9.0, 20.0, 10.5])  # errors 1, 10, 0.5
    regional = np.array([15.0, 11.0, 25.0])  # errors 5, 1, 15
    got = sc.oracle_best_dtw(phreatic, regional, obs)
    # row0 phreatic(1<5), row1 regional(1<10), row2 phreatic(0.5<15)
    assert np.allclose(got, [9.0, 11.0, 10.5])


def test_oracle_best_single_finite_fallback():
    obs = np.array([10.0, 10.0])
    phreatic = np.array([np.nan, 12.0])
    regional = np.array([13.0, np.nan])
    got = sc.oracle_best_dtw(phreatic, regional, obs)
    # only regional finite -> regional; only phreatic finite -> phreatic
    assert np.isclose(got[0], 13.0)
    assert np.isclose(got[1], 12.0)


def test_oracle_best_tie_prefers_phreatic():
    obs = np.array([10.0])
    got = sc.oracle_best_dtw(np.array([12.0]), np.array([8.0]), obs)  # |2| == |2|
    assert np.isclose(got[0], 12.0)


def test_hard_assigned_threshold():
    pi = np.array([0.9, 0.5, 0.49, 0.1])
    phreatic = np.array([1.0, 2.0, 3.0, 4.0])
    regional = np.array([10.0, 20.0, 30.0, 40.0])
    got = sc.hard_assigned_dtw(pi, phreatic, regional)
    # pi>=0.5 -> phreatic (rows 0,1); else regional (rows 2,3)
    assert np.allclose(got, [1.0, 2.0, 30.0, 40.0])


def test_mixture_identity():
    # gnn_dtw = pi*phreatic + (1-pi)*regional must reconstruct from components
    rng = np.random.default_rng(0)
    z = rng.uniform(50, 300, 500)
    r = rng.uniform(40, 250, 500)
    ts_p = rng.normal(0, 5, 500)
    ts_r = rng.normal(0, 20, 500)
    pi = rng.uniform(0, 1, 500)
    phreatic = sc.component_dtw(z, r, ts_p)
    regional = sc.component_dtw(z, r, ts_r)
    blend = pi * phreatic + (1.0 - pi) * regional
    # native blend: z - (r + pi*ts_p + (1-pi)*ts_r) is the same surface
    native_blend = z - (r + pi * ts_p + (1.0 - pi) * ts_r)
    assert np.allclose(blend, native_blend)
