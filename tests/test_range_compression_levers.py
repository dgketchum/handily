"""Unit tests for the range-compression loss levers in ``train_conus_gnn``:
the signed-log1p loss transform, the sigma-to-metres conversion per loss space,
and the inverse-density label weights (``notes/RANGE_COMPRESSION_PLAN.md``)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tc = _load("train_conus_gnn")


def test_loss_transform_signed_log1p_is_odd_monotone_and_torch_numpy_consistent():
    x = np.array([-5.0, -1.0, 0.0, 0.5, 2.0, 30.0])
    t = tc.loss_transform(x, "log_dtw")
    assert np.allclose(t, -tc.loss_transform(-x, "log_dtw"))
    assert (np.diff(t) > 0).all()
    assert np.allclose(t[3:], np.log1p(x[3:]))
    tt = tc.loss_transform(torch.as_tensor(x), "log_dtw").numpy()
    assert np.allclose(tt, t)
    assert np.array_equal(tc.loss_transform(x, "std"), x)
    assert np.array_equal(tc.loss_transform(x, "dtw"), x)


def test_sigma_to_m_per_space():
    lb = np.log(np.array([0.5, 2.0]))
    assert np.allclose(tc.sigma_to_m(lb, "std", 4.0), [2.0, 8.0])
    assert np.allclose(tc.sigma_to_m(lb, "dtw", 4.0), [0.5, 2.0])
    # delta method: b * (1 + |dtw|) at the prediction
    assert np.allclose(
        tc.sigma_to_m(lb, "log_dtw", 4.0, np.array([0.0, 9.0])), [0.5, 20.0]
    )
    try:
        tc.sigma_to_m(lb, "log_dtw", 4.0)
    except ValueError:
        pass
    else:
        raise AssertionError("log_dtw without dtw_pred must raise")


def test_density_weights_downweight_the_dense_band_and_leave_pseudo_rows_alone():
    rng = np.random.default_rng(0)
    # dense at 2-5 m, sparse at 30-60 m, plus pseudo rows at 0 m
    obs = np.concatenate(
        [rng.uniform(2, 5, 2000), rng.uniform(30, 60, 100), np.zeros(300)]
    )
    real = np.concatenate([np.ones(2100, bool), np.zeros(300, bool)])
    w, info = tc.density_label_weights(obs, real, 0.5, 0.25, 0.1, 10.0)
    assert w.shape == obs.shape and w.dtype == np.float32
    assert np.allclose(w[~real], 1.0)
    assert abs(w[real].mean() - 1.0) < 1e-5
    assert np.median(w[real][obs[real] < 5]) < np.median(w[real][obs[real] > 30])
    assert w[real].min() >= 0.1 - 1e-6 and w[real].max() <= 10.0 * (1 + 1e-6) * (
        1 / w[real].mean()
    )
    assert (
        info["n_real"] == 2100
        and info["w_median_by_band"]["30+"] > info["w_median_by_band"]["2-5"]
    )


def test_density_alpha_zero_is_uniform():
    obs = np.linspace(0, 50, 500)
    real = np.ones(500, bool)
    w, _ = tc.density_label_weights(obs, real, 0.0, 0.25, 0.1, 10.0)
    assert np.allclose(w, 1.0)
