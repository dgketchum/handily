"""Unit tests for the WP2 two-surface mixture arm on WTEGraphNet.

Covers the arm's hard guarantees: epoch-0 components ARE their anchors (zero-init
corrections), the mixture point prediction is the pi-weighted component mean, the
trainer's mixture Laplace NLL matches the direct per-row formula, exclusivity
raises, gradient flow into every ts_* module, and exact no-op behavior for shared
weights when the flag is off (RNG discipline).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
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


def _anchors(n_query=5, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "fac_base": torch.tensor(rng.normal(size=n_query), dtype=torch.float32),
        "fac_present": torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float32),
        "mirror_base": torch.tensor(rng.normal(size=n_query), dtype=torch.float32),
        "deep_base": torch.tensor(rng.normal(size=n_query), dtype=torch.float32),
        "deep_present": torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    }


def _model(**kw):
    torch.manual_seed(0)
    return tw.WTEGraphNet(4, 3, 2, 2, 8, 1, 0.0, **kw)


def test_epoch0_components_are_their_anchors():
    m = _model(two_surface=True)
    m.eval()
    g = _tiny_graph()
    a = _anchors()
    with torch.no_grad():
        primary = m({**g, **a})
    ts = m.ts_out
    fp = a["fac_present"]
    want_hp = fp * a["fac_base"] + (1.0 - fp) * a["mirror_base"]
    want_hr = a["deep_present"] * a["deep_base"]
    assert torch.allclose(ts["hp"], want_hp, atol=1e-6)
    assert torch.allclose(ts["hr"], want_hr, atol=1e-6)
    pi = torch.sigmoid(ts["m"])
    assert torch.allclose(primary, pi * ts["hp"] + (1.0 - pi) * ts["hr"], atol=1e-6)


def test_mixture_nll_matches_direct_formula():
    # replicate the trainer's logsumexp branch and compare against the literal
    # -log(pi*Lap_p + (1-pi)*Lap_r) with the log(2) constant dropped from both.
    m = _model(two_surface=True)
    g = {**_tiny_graph(), **_anchors()}
    m(g)
    ts = m.ts_out
    y = torch.tensor([0.3, -1.2, 0.0, 2.0, -0.5])
    ll_p = -(torch.abs(ts["hp"] - y) * torch.exp(-ts["lbp"]) + ts["lbp"])
    ll_r = -(torch.abs(ts["hr"] - y) * torch.exp(-ts["lbr"]) + ts["lbr"])
    row = -torch.logsumexp(
        torch.stack(
            [
                torch.nn.functional.logsigmoid(ts["m"]) + ll_p,
                torch.nn.functional.logsigmoid(-ts["m"]) + ll_r,
            ]
        ),
        dim=0,
    )
    pi = torch.sigmoid(ts["m"])
    direct = -torch.log(pi * torch.exp(ll_p) + (1.0 - pi) * torch.exp(ll_r))
    assert torch.allclose(row, direct, atol=1e-5)


def test_grads_reach_every_ts_module():
    m = _model(two_surface=True)
    g = {**_tiny_graph(), **_anchors()}
    m(g)
    ts = m.ts_out
    y = torch.tensor([0.3, -1.2, 0.0, 2.0, -0.5])
    a = torch.tensor([0.9, 0.1, 0.5, 0.98, 0.2])
    ll_p = -(torch.abs(ts["hp"] - y) * torch.exp(-ts["lbp"]) + ts["lbp"])
    ll_r = -(torch.abs(ts["hr"] - y) * torch.exp(-ts["lbr"]) + ts["lbr"])
    nll = -torch.logsumexp(
        torch.stack(
            [
                torch.nn.functional.logsigmoid(ts["m"]) + ll_p,
                torch.nn.functional.logsigmoid(-ts["m"]) + ll_r,
            ]
        ),
        dim=0,
    )
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        ts["m"], a, reduction="none"
    )
    loss = (nll + 0.3 * (2.0 * a - 1.0).abs() * bce).mean()
    loss.backward()
    for name in ("ts_head_p", "ts_head_r", "ts_logb_p", "ts_logb_r", "ts_member"):
        mod = getattr(m, name)
        assert any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for p in mod.parameters()
        ), f"no gradient reached {name}"


def test_neutral_prior_contributes_nothing():
    # conf = |2a-1| zeroes the assignment term at a=0.5 exactly, so unmatched
    # wells impose nothing on the membership head.
    a = torch.full((5,), 0.5)
    logits = torch.tensor([1.0, -2.0, 0.3, 4.0, 0.0], requires_grad=True)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, a, reduction="none"
    )
    term = ((2.0 * a - 1.0).abs() * bce).sum()
    term.backward()
    assert float(term) == 0.0
    assert float(logits.grad.abs().sum()) == 0.0


def test_exclusivity_raises():
    with pytest.raises(ValueError):
        _model(two_surface=True, prior_gate=True)
    with pytest.raises(ValueError):
        _model(two_surface=True, sigma=True)
    with pytest.raises(ValueError):
        _model(two_surface=True, pinball=True)
    with pytest.raises(ValueError):
        _model(two_surface=True, fac_skip=True)


def test_two_surface_off_is_inert():
    m = _model()
    m.eval()
    with torch.no_grad():
        out = m(_tiny_graph())
    assert m.ts_out is None
    assert out.shape == (5,)


def test_two_surface_off_baseline_weights_identical():
    # RNG discipline: the ts_* modules are constructed LAST (after ordinal), so
    # turning the flag on must not perturb the init of any shared module.
    m0 = _model(ordinal=True)
    m1 = _model(ordinal=True, two_surface=True)
    s0, s1 = m0.state_dict(), m1.state_dict()
    for k, v in s0.items():
        assert torch.equal(v, s1[k]), f"shared weight {k} changed by two_surface flag"
