"""Unit tests for the anti-compression pair loss in utils/train_conus_gnn.py (item 4).

Covers:
  * build_train_pairs -- <=k nearest neighbours within radius_m, i<j, deduped,
    no self-pairs, radius/k monotonicity (the neighbour set the pair term acts on).
  * the per-fold train-train restriction (leak-free: a pair with a held-out end
    is dropped) -- the exact expression main() uses.
  * a 1-epoch train_fold smoke: the pair term is finite AND, from an identical
    init/step, a positive weight makes the predictions LESS compressed (larger
    std) than the point loss alone -- the mechanism the term exists to provide.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_MOD = Path(__file__).resolve().parents[1] / "utils" / "train_conus_gnn.py"
_spec = importlib.util.spec_from_file_location("train_conus_gnn", _MOD)
tc = importlib.util.module_from_spec(_spec)
# train_wte_gnn (imported transitively) resolves PyG message() globals via
# sys.modules, so register before exec.
sys.modules["train_conus_gnn"] = tc
_spec.loader.exec_module(tc)


def _pair_dists(xy, pairs):
    return np.hypot(
        xy[pairs[:, 0], 0] - xy[pairs[:, 1], 0],
        xy[pairs[:, 0], 1] - xy[pairs[:, 1], 1],
    )


def test_build_train_pairs_exact_small_case():
    # x = 0,1,2 (a triplet 1 m apart) then 10,11 (a distant couple). radius 1.5 m
    # links only immediate neighbours: (0,1),(1,2),(3,4). No 2-3 (8 m apart).
    x = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
    y = np.zeros_like(x)
    pairs = tc.build_train_pairs(x, y, radius_m=1.5, k=10)
    assert set(map(tuple, pairs)) == {(0, 1), (1, 2), (3, 4)}
    assert (pairs[:, 0] < pairs[:, 1]).all()  # i<j, so no self-pairs
    assert len(np.unique(pairs, axis=0)) == len(pairs)  # deduped


def test_build_train_pairs_radius_and_k_properties():
    rng = np.random.RandomState(0)
    xy = rng.uniform(0.0, 1000.0, size=(60, 2))
    x, y = xy[:, 0], xy[:, 1]
    p1 = tc.build_train_pairs(x, y, radius_m=300.0, k=1)
    p3 = tc.build_train_pairs(x, y, radius_m=300.0, k=3)
    for p in (p1, p3):
        assert (p[:, 0] < p[:, 1]).all()  # i<j, no self
        assert len(np.unique(p, axis=0)) == len(p)  # deduped
        assert (_pair_dists(xy, p) <= 300.0 + 1e-9).all()  # radius respected
    assert len(p3) >= len(p1)  # more neighbours -> at least as many pairs
    # a point beyond radius of every other never appears in any pair.
    xf = np.append(x, 1e6)
    yf = np.append(y, 1e6)
    pf = tc.build_train_pairs(xf, yf, radius_m=300.0, k=3)
    assert (pf != len(xf) - 1).all()


def test_build_train_pairs_empty_when_all_far():
    x = np.array([0.0, 1000.0, 2000.0])
    y = np.zeros_like(x)
    pairs = tc.build_train_pairs(x, y, radius_m=10.0, k=3)
    assert pairs.shape == (0, 2)
    assert pairs.dtype == np.int64


def test_pair_train_mask_filter_is_leak_free():
    # the exact restriction main() applies: keep a pair only if BOTH ends train.
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.zeros_like(x)
    pairs = tc.build_train_pairs(x, y, radius_m=1.5, k=10)  # (0,1),(1,2),(2,3),(3,4)
    tr = np.array([True, True, False, True, True])  # hold out node 2
    both_tr = tr[pairs[:, 0]] & tr[pairs[:, 1]]
    kept = pairs[both_tr]
    dropped = pairs[~both_tr]
    assert tr[kept[:, 0]].all() and tr[kept[:, 1]].all()  # no held-out end survives
    assert (
        ~(tr[dropped[:, 0]] & tr[dropped[:, 1]])
    ).all()  # every drop had a held-out end
    assert set(map(tuple, kept)) == {(0, 1), (3, 4)}  # the two pairs clear of node 2


class _LinModel(torch.nn.Module):
    """pred = a * query_x (+ b); starts flat (a=b=0) so it begins fully compressed."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)

    def forward(self, feat):
        return self.lin(feat["query_x"]).squeeze(-1)


def _smoke_args():
    return SimpleNamespace(
        lr=0.1,
        weight_decay=0.0,
        epochs=1,
        min_epochs=1,
        patience=1,
        pinball=False,
        pinball_weight=0.0,
        huber_delta=1.0,
        huber_delta_m=1.0,
        deep_regime_threshold_m=30.0,
    )


def _run_smoke_fold(pair_w):
    n = 24
    device = "cpu"
    xm = np.arange(n, dtype="float64") * 100.0  # 100 m spacing -> local neighbours
    coords_y = np.zeros(n)
    ycol = np.linspace(-3.0, 3.0, n)  # real spatial spread in standardized target
    feat = {"query_x": torch.as_tensor(ycol.reshape(-1, 1), dtype=torch.float32)}
    y_std = torch.as_tensor(ycol, dtype=torch.float32)
    base = np.zeros(n)  # dtw_residual: dtw = base + native = native
    obs_dtw = ycol.copy()
    va = np.zeros(n, bool)
    va[[5, 18]] = True  # small held-out val block so best_mad is finite
    tr = ~va
    pairs = tc.build_train_pairs(xm, coords_y, radius_m=250.0, k=3)
    both_tr = tr[pairs[:, 0]] & tr[pairs[:, 1]]
    pair_idx = torch.as_tensor(pairs[both_tr].T, dtype=torch.long)
    torch.manual_seed(0)
    model = _LinModel()
    native, best_mad, _ = tc.train_fold(
        model,
        feat,
        y_std,
        tr,
        va,
        base,
        obs_dtw,
        0.0,  # y_c
        1.0,  # y_s (native == pred)
        tc.TARGET_DTW_RESIDUAL,
        0.85,  # eff_tau (unused: pinball off)
        _smoke_args(),
        device,
        None,  # sample_w_t
        pair_idx=pair_idx,
        pair_w=pair_w,
    )
    return native, best_mad


def test_pair_loss_smoke_finite_and_reduces_compression():
    native0, mad0 = _run_smoke_fold(pair_w=0.0)
    native_p, mad_p = _run_smoke_fold(pair_w=8.0)
    assert np.isfinite(native0).all() and np.isfinite(native_p).all()
    assert np.isfinite(mad0) and np.isfinite(mad_p)
    # identical init + one identical point-loss step; the pair term adds gradient
    # that matches nearby DIFFERENCES -> after the step the weighted predictions
    # carry more amplitude (less compressed) than the point loss alone.
    assert np.std(native_p) > np.std(native0)


class _SigModel(torch.nn.Module):
    """Linear point head + linear log-b head; exposes sigma_log_b like WTEGraphNet."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)
        self.sig = torch.nn.Linear(1, 1)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)
        torch.nn.init.zeros_(self.sig.weight)
        torch.nn.init.zeros_(self.sig.bias)
        self.sigma_log_b = None

    def forward(self, feat):
        x = feat["query_x"]
        self.sigma_log_b = torch.clamp(self.sig(x).squeeze(-1), -4.0, 4.0)
        return self.lin(x).squeeze(-1)


def test_sigma_head_nll_smoke_finite_and_trains():
    n = 24
    ycol = np.linspace(-3.0, 3.0, n)
    feat = {"query_x": torch.as_tensor(ycol.reshape(-1, 1), dtype=torch.float32)}
    y_std = torch.as_tensor(ycol, dtype=torch.float32)
    va = np.zeros(n, bool)
    va[[5, 18]] = True
    args = _smoke_args()
    args.sigma_head = True
    args.epochs = 5
    torch.manual_seed(0)
    model = _SigModel()
    native, best_mad, _ = tc.train_fold(
        model,
        feat,
        y_std,
        ~va,
        va,
        np.zeros(n),
        ycol.copy(),
        0.0,
        1.0,
        tc.TARGET_DTW_RESIDUAL,
        0.85,
        args,
        "cpu",
        None,
    )
    assert np.isfinite(native).all() and np.isfinite(best_mad)
    # the NLL branch actually ran: both heads moved off their zero init.
    assert model.lin.weight.abs().sum() > 0
    assert model.sig.weight.abs().sum() > 0 or model.sig.bias.abs().sum() > 0
