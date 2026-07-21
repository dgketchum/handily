"""Unit tests for the WP5 monotone ordinal shallow head on WTEGraphNet.

Covers the head's hard guarantees: nested probabilities by construction
(P(DTW<2) <= P(DTW<5) <= P(DTW<10) for every query), gradient flow into the
score/cutpoint parameters through the BCE, and exact no-op behavior when the
flag is off (ordinal_logits stays None).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    # torch_geometric's MessagePassing inspector resolves the defining module
    # through sys.modules, so the by-path load must register itself first.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tw = _load("train_wte_gnn")


def _tiny_graph(n_reach=6, n_query=5, f_reach=4, f_query=3, f_ch=2, f_lat=2, seed=0):
    rng = np.random.default_rng(seed)
    g = {
        "reach_x": torch.tensor(
            rng.normal(size=(n_reach, f_reach)), dtype=torch.float32
        ),
        "query_x": torch.tensor(
            rng.normal(size=(n_query, f_query)), dtype=torch.float32
        ),
        "ch_ei": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        "ch_ea": torch.tensor(rng.normal(size=(4, f_ch)), dtype=torch.float32),
        "lat_ei": torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long),
        "lat_ea": torch.tensor(rng.normal(size=(5, f_lat)), dtype=torch.float32),
    }
    return g


def _model(**kw):
    torch.manual_seed(0)
    return tw.WTEGraphNet(4, 3, 2, 2, 8, 1, 0.0, **kw)


def test_ordinal_probabilities_nested_for_every_query():
    m = _model(sigma=True, ordinal=True)
    m.eval()
    with torch.no_grad():
        m(_tiny_graph())
    assert m.ordinal_logits is not None and m.ordinal_logits.shape == (5, 3)
    p = torch.sigmoid(m.ordinal_logits).numpy()
    assert (np.diff(p, axis=1) >= 0).all()


def test_ordinal_nesting_survives_training_steps():
    # push the head hard toward arbitrary labels; nesting must hold afterwards
    m = _model(ordinal=True)
    g = _tiny_graph()
    y = torch.tensor(
        [[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32
    )
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(60):
        opt.zero_grad()
        m(g)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m.ordinal_logits, y)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        m(g)
    p = torch.sigmoid(m.ordinal_logits).numpy()
    assert (np.diff(p, axis=1) >= -1e-7).all()


def test_ordinal_bce_grad_reaches_score_and_cutpoints():
    m = _model(ordinal=True)
    m(_tiny_graph())
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        m.ordinal_logits, torch.ones(5, 3)
    )
    loss.backward()
    assert m.ordinal_cut_raw.grad is not None
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in m.ordinal_score.parameters()
    )


def test_ordinal_off_is_inert():
    m = _model(sigma=True)
    m.eval()
    with torch.no_grad():
        out = m(_tiny_graph())
    assert m.ordinal_logits is None
    assert out.shape == (5,)


def test_ordinal_off_baseline_weights_identical():
    # RNG discipline: constructing with ordinal=True must not perturb the init
    # of any shared module (the ordinal modules are constructed LAST).
    m0 = _model(sigma=True)
    m1 = _model(sigma=True, ordinal=True)
    s0, s1 = m0.state_dict(), m1.state_dict()
    for k, v in s0.items():
        assert torch.equal(v, s1[k]), f"shared weight {k} changed by ordinal flag"
