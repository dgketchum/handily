"""Unit tests for utils/train_wte_gnn.py.

Covers the leak-relevant / numeric pieces: train-only median-impute + z-score
with missingness flags, whole-block validation carving that hits the target row
fraction (blocks vary >100x in density), and the edge-gated message-passing conv
(reach->query direction, gate in (0,1), empty-incoming -> zero context).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_MOD = Path(__file__).resolve().parents[1] / "utils" / "train_wte_gnn.py"
_spec = importlib.util.spec_from_file_location("train_wte_gnn", _MOD)
tg = importlib.util.module_from_spec(_spec)
# PyG's MessagePassing Inspector resolves message() globals via sys.modules,
# so the dynamically loaded module must be registered before exec.
sys.modules["train_wte_gnn"] = tg
_spec.loader.exec_module(tg)


def test_fit_apply_stats_zscore_and_missingness_flag():
    df = pd.DataFrame({"a": [1.0, 3.0, np.nan, 100.0], "b": [0.0, 0.0, 0.0, 0.0]})
    train = np.array([True, True, True, False])  # exclude the outlier from stats
    stats = tg.fit_stats(df, ["a", "b"], train)
    # train median of 'a' (1,3,nan) = 2; constant 'b' std -> guarded to 1.
    assert stats["med"]["a"] == 2.0
    assert stats["std"]["b"] == 1.0
    assert stats["nan_cols"] == ["a"]  # only 'a' is ever NaN

    x = tg.apply_stats(df, stats)
    # 2 feature cols + 1 missingness flag (for 'a') = 3 columns.
    assert x.shape == (4, 3)
    # NaN row imputed to median -> z 0 for 'a'; flag column marks it.
    assert np.isclose(x[2, 0], 0.0)
    assert list(x[:, 2]) == [0.0, 0.0, 1.0, 0.0]


def test_val_blocks_whole_block_and_fraction():
    # one dense block (1000 rows) + many sparse blocks (10 each).
    blocks = np.array(["dense"] * 1000 + sum([[f"s{i}"] * 10 for i in range(40)], []))
    trainval = np.ones(len(blocks), bool)
    trainval[:5] = False  # a few test rows -> excluded from val
    rng = np.random.RandomState(0)
    va = tg.val_blocks(trainval, blocks, 0.15, rng)

    assert (va & ~trainval).sum() == 0  # val never leaks into test rows
    # whole-block: no block is split between val and the rest.
    for b in np.unique(blocks):
        inb = blocks == b
        v = va[inb & trainval]
        if v.any():
            assert v.all()
    # fraction of TRAIN-VAL rows is near 0.15 (accumulated by row count).
    frac = va.sum() / trainval.sum()
    assert 0.08 < frac < 0.25


def test_edge_gated_conv_direction_gate_and_empty_incoming():
    torch.manual_seed(0)
    conv = tg.EdgeGatedConv(in_src=3, in_dst=2, edge_dim=1, out_dim=4)
    x_src = torch.randn(5, 3)  # reaches
    x_dst = torch.randn(3, 2)  # queries
    # reach 0 -> query 0, reach 1 -> query 0; query 2 has no incoming edge.
    edge_index = torch.tensor([[0, 1], [0, 0]])
    edge_attr = torch.randn(2, 1)
    out = conv(x_src, x_dst, edge_index, edge_attr)

    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()
    g = conv.last_gate
    assert g.shape == (2, 1)
    assert ((g > 0) & (g < 1)).all()  # sigmoid gate strictly in (0,1)
    # query 2 has no incoming message -> mean-agg is zeros -> update on [x_dst, 0].
    expect2 = conv.upd_mlp(torch.cat([x_dst[2], torch.zeros(4)]))
    assert torch.allclose(out[2], expect2, atol=1e-6)


# ---------------------------------------------------------------------------
# Aquifer branch: exact no-op contract + learned gate/delta wiring
# ---------------------------------------------------------------------------
F_REACH, F_QUERY, F_CH, F_LAT = 3, 2, 1, 1
F_AQ, F_AQ_E, F_AQ_Q = 4, 2, 2
HIDDEN = 8


def _tiny_graph():
    """Small non-anchor graph dict with the aquifer tensors the model reads."""
    torch.manual_seed(7)
    return {
        "reach_x": torch.randn(4, F_REACH),
        "query_x": torch.randn(3, F_QUERY),
        "ch_ei": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "ch_ea": torch.randn(3, F_CH),
        # lateral edges are reach(src) -> query(dst); query 2 left with no edge.
        "lat_ei": torch.tensor([[0, 1, 2], [0, 0, 1]]),
        "lat_ea": torch.randn(3, F_LAT),
        "aquifer_x": torch.randn(5, F_AQ),
        "aq_node_ei": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        "aq_node_ea": torch.randn(4, F_AQ_E),
        "aq_query_ei": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]]),
        "aq_query_ea": torch.randn(4, F_AQ_Q),
    }


def _base_model(seed=0):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1)


def _aquifer_model(route, seed=0, delta_zero=True, gate_init=-6.0):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(
        F_REACH,
        F_QUERY,
        F_CH,
        F_LAT,
        HIDDEN,
        2,
        0.1,
        f_aquifer=F_AQ,
        f_aquifer_edge=F_AQ_E,
        f_aquifer_query=F_AQ_Q,
        n_aquifer_layers=2,
        aquifer_route=route,
        aquifer_gate_init=gate_init,
        aquifer_delta_init_zero=delta_zero,
    )


def test_aquifer_fixed_stream_is_exact_noop():
    g = _tiny_graph()
    base = _base_model().eval()
    aq = _aquifer_model("fixed_stream").eval()
    # copy the shared stream weights so the test isolates the branch contribution.
    aq.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        assert torch.allclose(base(g), aq(g), atol=1e-6)


def test_aquifer_fixed_stream_same_seed_param_and_rng_identity():
    # The run-level no-op contract, proved on a deterministic (CPU) device: two models
    # built from the SAME seed must (a) share byte-identical stream params -- proving the
    # aquifer modules are constructed LAST so they never perturb baseline init -- and (b)
    # consume identical RNG through a TRAIN-mode forward -- proving fixed_stream draws no
    # aquifer dropout, so two independent training runs stay locked step-for-step. (The
    # ~140 m cross-process run diff is GPU scatter/index_add nondeterminism, identical in
    # magnitude to baseline-vs-baseline, NOT a branch contribution.)
    g = _tiny_graph()
    base = _base_model(seed=3)
    aq = _aquifer_model("fixed_stream", seed=3)
    bsd = aq.state_dict()
    for k, v in base.state_dict().items():
        assert torch.equal(v, bsd[k]), f"stream param {k} diverged at init"
    base.train()
    aq.train()
    # a train-mode forward must consume the same RNG in both -> trajectories stay locked.
    torch.manual_seed(123)
    base(g)
    rng_after_base = torch.rand(4)
    torch.manual_seed(123)
    aq(g)
    rng_after_aq = torch.rand(4)
    assert torch.equal(rng_after_base, rng_after_aq)
    # and the train-mode outputs match given the same pre-forward seed.
    torch.manual_seed(123)
    ob = base(g)
    torch.manual_seed(123)
    oa = aq(g)
    assert torch.allclose(ob, oa, atol=1e-6)


def test_aquifer_learned_delta_init_zero_is_noop_at_init():
    # delta head zeroed at init => learned route == stream output at epoch 0 (eval).
    g = _tiny_graph()
    base = _base_model().eval()
    aq = _aquifer_model("learned", delta_zero=True).eval()
    aq.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        assert torch.allclose(base(g), aq(g), atol=1e-6)


def test_aquifer_learned_gate_init_pins_to_sigmoid_gate_init():
    g = _tiny_graph()
    aq = _aquifer_model("learned", gate_init=-6.0).eval()
    with torch.no_grad():
        aq(g)
    gate = aq.last_aquifer_gate
    assert gate.shape == (3,)  # one gate per query
    # final gate weight is zeroed at init, so the gate is exactly sigmoid(gate_init).
    assert torch.allclose(
        gate, torch.full((3,), torch.sigmoid(torch.tensor(-6.0))), atol=1e-6
    )


def test_aquifer_learned_delta_changes_output_when_head_nonzero():
    # with a non-zero delta head and an opened gate, the branch must move the output.
    g = _tiny_graph()
    base = _base_model().eval()
    aq = _aquifer_model("learned", delta_zero=False, gate_init=4.0).eval()
    aq.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        torch.nn.init.normal_(aq.aquifer_delta_head[-1].weight, std=1.0)
        torch.nn.init.constant_(aq.aquifer_delta_head[-1].bias, 0.5)
        assert not torch.allclose(base(g), aq(g), atol=1e-4)


def test_aquifer_missing_tensor_errors_loudly():
    g = _tiny_graph()
    del g["aquifer_x"]
    aq = _aquifer_model("learned").eval()
    with pytest.raises(KeyError):
        aq(g)


def test_aquifer_disallowed_with_pinball():
    with pytest.raises(ValueError):
        tg.WTEGraphNet(
            F_REACH,
            F_QUERY,
            F_CH,
            F_LAT,
            HIDDEN,
            2,
            0.1,
            pinball=True,
            f_aquifer=F_AQ,
            f_aquifer_edge=F_AQ_E,
            f_aquifer_query=F_AQ_Q,
            n_aquifer_layers=2,
            aquifer_route="learned",
        )


# ---------------------------------------------------------------------------
# Mainstem-read conv (item 2): query -> downstream-datum read edge
# ---------------------------------------------------------------------------
F_MS = 5


def _ms_graph():
    """_tiny_graph + one datum-read edge for queries 0 and 1; query 2 has none."""
    g = _tiny_graph()
    g["ms_ei"] = torch.tensor([[0, 1], [0, 1]])  # reach 0->query 0, reach 1->query 1
    g["ms_ea"] = torch.randn(2, F_MS)
    return g


def _ms_model(seed=0):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, f_ms=F_MS)


def test_mainstem_read_forward_shape_and_zero_context():
    g = _ms_graph()
    m = _ms_model().eval()
    assert m.has_ms
    assert m.head[0].in_features == HIDDEN * 3  # [q, ctx_reach, ctx_ms]
    with torch.no_grad():
        out = m(g)
    assert out.shape == (3,)
    # query 2 has no incoming ms edge -> zero context -> still a finite prediction.
    assert torch.isfinite(out).all()


def test_mainstem_read_none_is_baseline_state_dict():
    # f_ms=None must add no params and keep the plain hidden*2 head (keys unchanged).
    base = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1)
    assert not base.has_ms
    assert not hasattr(base, "ms_read")
    assert base.head[0].in_features == HIDDEN * 2
    assert not any(k.startswith("ms_read") for k in base.state_dict())


def test_mainstem_read_uses_the_edge_context():
    # dropping the ms edge attrs to zero magnitude changes the gate -> output moves,
    # proving ctx_ms is actually wired into the head (not a dead branch).
    g = _ms_graph()
    m = _ms_model().eval()
    with torch.no_grad():
        out_full = m(g)
        g2 = dict(g)
        g2["ms_ei"] = torch.empty(2, 0, dtype=torch.long)  # no ms edges at all
        g2["ms_ea"] = torch.empty(0, F_MS)
        out_none = m(g2)
    # queries 0/1 had edges; removing them must change their prediction.
    assert not torch.allclose(out_full[:2], out_none[:2], atol=1e-5)


def test_mainstem_read_rejects_anchor_combo():
    with pytest.raises(ValueError):
        tg.WTEGraphNet(
            F_REACH,
            F_QUERY,
            F_CH,
            F_LAT,
            HIDDEN,
            2,
            0.1,
            f_ms=F_MS,
            f_anchor=F_AQ,
            f_anchor_reach=F_AQ_E,
            f_anchor_query=F_AQ_Q,
        )


# ---------------------------------------------------------------------------
# Portfolio-read conv (6B): segment-softmax attention over <=4 typed site edges
# ---------------------------------------------------------------------------
F_PF = 6


def _pf_graph():
    """_tiny_graph + a portfolio read: query 0 gets 3 site edges, query 1 gets 1, 2 none."""
    g = _tiny_graph()
    g["pf_ei"] = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 1]])  # reach(src) -> query(dst)
    g["pf_ea"] = torch.randn(4, F_PF)
    return g


def _pf_model(seed=0):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, f_pf=F_PF)


def test_portfolio_conv_softmax_sums_to_one_per_query():
    torch.manual_seed(0)
    conv = tg.PortfolioReadConv(in_src=3, in_dst=2, edge_dim=1, out_dim=4)
    x_src = torch.randn(5, 3)
    x_dst = torch.randn(3, 2)
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 0, 0, 1]]
    )  # q0: 3 edges, q1: 1, q2: none
    edge_attr = torch.randn(4, 1)
    out = conv(x_src, x_dst, edge_index, edge_attr)
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()
    a = conv.last_attn.reshape(-1)
    assert a.shape == (4,)
    # per-query segment softmax: query 0's three edges sum to 1; query 1's single edge = 1.
    assert torch.isclose(a[:3].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(a[3], torch.tensor(1.0), atol=1e-6)
    assert ((a > 0) & (a <= 1)).all()


def test_portfolio_conv_zero_edge_query_is_finite():
    torch.manual_seed(0)
    conv = tg.PortfolioReadConv(3, 2, 1, 4)
    x_src = torch.randn(5, 3)
    x_dst = torch.randn(3, 2)
    edge_index = torch.tensor([[0], [0]])  # only query 0 has a site edge
    edge_attr = torch.randn(1, 1)
    out = conv(x_src, x_dst, edge_index, edge_attr)
    # query 2 (no edge) -> zero aggregate -> update on [x_dst, 0], still finite.
    expect2 = conv.upd_mlp(torch.cat([x_dst[2], torch.zeros(4)]))
    assert torch.allclose(out[2], expect2, atol=1e-6)


def test_portfolio_conv_permutation_invariant():
    torch.manual_seed(1)
    conv = tg.PortfolioReadConv(3, 2, 2, 4).eval()
    x_src = torch.randn(5, 3)
    x_dst = torch.randn(3, 2)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 1]])
    edge_attr = torch.randn(4, 2)
    with torch.no_grad():
        out = conv(x_src, x_dst, edge_index, edge_attr)
        perm = torch.tensor([2, 0, 3, 1])
        out_p = conv(x_src, x_dst, edge_index[:, perm], edge_attr[perm])
    assert torch.allclose(
        out, out_p, atol=1e-6
    )  # attention read is edge-order invariant


def test_portfolio_read_forward_shape_and_zero_context():
    g = _pf_graph()
    m = _pf_model().eval()
    assert m.has_pf
    assert m.head[0].in_features == HIDDEN * 3  # [q, ctx_reach, ctx_pf]
    with torch.no_grad():
        out = m(g)
    assert out.shape == (3,)
    assert torch.isfinite(
        out
    ).all()  # query 2 has no portfolio edge -> zero ctx, finite


def test_portfolio_read_none_is_baseline_state_dict():
    base = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1)
    assert not base.has_pf
    assert not hasattr(base, "pf_read")
    assert base.head[0].in_features == HIDDEN * 2
    assert not any(k.startswith("pf_read") for k in base.state_dict())


def test_portfolio_read_uses_the_edge_context():
    g = _pf_graph()
    m = _pf_model().eval()
    with torch.no_grad():
        out_full = m(g)
        g2 = dict(g)
        g2["pf_ei"] = torch.empty(2, 0, dtype=torch.long)  # no portfolio edges at all
        g2["pf_ea"] = torch.empty(0, F_PF)
        out_none = m(g2)
    assert not torch.allclose(out_full[:2], out_none[:2], atol=1e-5)


def test_portfolio_read_rejects_ms_and_anchor():
    with pytest.raises(ValueError):
        tg.WTEGraphNet(
            F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, f_pf=F_PF, f_ms=F_MS
        )
    with pytest.raises(ValueError):
        tg.WTEGraphNet(
            F_REACH,
            F_QUERY,
            F_CH,
            F_LAT,
            HIDDEN,
            2,
            0.1,
            f_pf=F_PF,
            f_anchor=F_AQ,
            f_anchor_reach=F_AQ_E,
            f_anchor_query=F_AQ_Q,
        )


# ---------------------------------------------------------------------------
# Query->reach write-back (6C): reversed lateral residual before the channel stack
# ---------------------------------------------------------------------------
def _wb_graph():
    g = _tiny_graph()
    g["lat_ei_reversed"] = g["lat_ei"].flip(0)  # query(src) -> reach(dst)
    return g


def test_writeback_off_adds_no_params():
    base = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1)
    assert not base.writeback
    assert not hasattr(base, "writeback_conv")
    assert not any(k.startswith("writeback_conv") for k in base.state_dict())


def test_writeback_zeroed_conv_matches_baseline():
    # the 6C residual r = r + writeback(...): a zero-output writeback_conv reduces EXACTLY
    # to the baseline, proving the query_enc reorder + residual are a no-op when nothing is
    # written back (the reordered query_enc draws no RNG, so the stream trajectory is intact).
    g = _wb_graph()
    base = _base_model(seed=5).eval()
    wb = tg.WTEGraphNet(
        F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, writeback=True
    ).eval()
    wb.load_state_dict(base.state_dict(), strict=False)  # share the stream params
    with torch.no_grad():
        torch.nn.init.zeros_(wb.writeback_conv.upd_mlp[-1].weight)
        torch.nn.init.zeros_(wb.writeback_conv.upd_mlp[-1].bias)
        assert torch.allclose(base(g), wb(g), atol=1e-6)


def test_writeback_on_changes_output_and_is_finite():
    g = _wb_graph()
    base = _base_model(seed=5).eval()
    wb = tg.WTEGraphNet(
        F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, writeback=True
    ).eval()
    wb.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        out = wb(g)
    assert out.shape == (3,) and torch.isfinite(out).all()
    with torch.no_grad():
        assert not torch.allclose(
            base(g), out, atol=1e-5
        )  # write-back moves the output


def test_writeback_rejects_anchor_combo():
    with pytest.raises(ValueError):
        tg.WTEGraphNet(
            F_REACH,
            F_QUERY,
            F_CH,
            F_LAT,
            HIDDEN,
            2,
            0.1,
            writeback=True,
            f_anchor=F_AQ,
            f_anchor_reach=F_AQ_E,
            f_anchor_query=F_AQ_Q,
        )


# --- fac-lambda convex blend + heteroscedastic sigma head (levers A / E) ---------
def _fac_tensors(pres=(1.0, 1.0, 0.0)):
    torch.manual_seed(11)
    p = torch.tensor(pres)
    return {
        "fac_base": torch.randn(3) * p,  # absent FAC is zeroed (as _fac_feat does)
        "fac_present": p,
        "fac_pred_dtw": torch.randn(3) * p,
    }


def _lambda_model(seed=0):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(
        F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, fac_lambda=True
    )


def test_fac_lambda_convex_blend_limits():
    g = {**_tiny_graph(), **_fac_tensors()}
    base = _base_model(seed=0).eval()
    m = _lambda_model(seed=0).eval()
    # lambda modules are constructed LAST -> shared modules identical at same seed.
    with torch.no_grad():
        # lam -> 1 everywhere FAC present: pred == fac_base there.
        torch.nn.init.zeros_(m.fac_lambda_mlp[-1].weight)
        torch.nn.init.constant_(m.fac_lambda_mlp[-1].bias, 50.0)
        hi = m(g)
        assert torch.allclose(hi[:2], g["fac_base"][:2], atol=1e-4)
        # lam -> 0: pure GNN head == baseline output exactly.
        torch.nn.init.constant_(m.fac_lambda_mlp[-1].bias, -50.0)
        lo = m(g)
        assert torch.allclose(lo, base(g), atol=1e-5)
        # absent FAC (query 2) is pure GNN regardless of the gate bias.
        torch.nn.init.constant_(m.fac_lambda_mlp[-1].bias, 50.0)
        assert torch.allclose(m(g)[2], base(g)[2], atol=1e-5)
        lam = m.last_fac_lambda.squeeze(-1)
        assert lam[2] == 0.0 and (lam[:2] > 0.99).all()


def test_fac_lambda_off_is_baseline_state_dict():
    base = _base_model(seed=0)
    assert set(_base_model(seed=1).state_dict()) == set(base.state_dict())
    lam = _lambda_model(seed=0)
    extra = set(lam.state_dict()) - set(base.state_dict())
    assert extra and all(k.startswith("fac_lambda_mlp") for k in extra)


def test_fac_lambda_rejects_skip_gate_pinball():
    with pytest.raises(ValueError):
        tg.WTEGraphNet(
            F_REACH,
            F_QUERY,
            F_CH,
            F_LAT,
            HIDDEN,
            2,
            0.1,
            fac_lambda=True,
            fac_skip=True,
        )
    with pytest.raises(ValueError):
        tg.WTEGraphNet(
            F_REACH,
            F_QUERY,
            F_CH,
            F_LAT,
            HIDDEN,
            2,
            0.1,
            fac_lambda=True,
            pinball=True,
        )


def test_sigma_head_log_b_clamped_and_backprops():
    torch.manual_seed(0)
    m = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, sigma=True)
    g = _tiny_graph()
    out = m(g)
    lb = m.sigma_log_b
    assert lb.shape == (3,) and torch.isfinite(lb).all()
    assert (lb >= -4.0).all() and (lb <= 4.0).all()
    # Laplace NLL backprops through BOTH heads.
    y = torch.zeros(3)
    nll = (torch.abs(out - y) * torch.exp(-lb) + lb).mean()
    nll.backward()
    assert m.sigma_head[0].weight.grad is not None
    assert torch.isfinite(m.sigma_head[0].weight.grad).all()
    assert m.head[0].weight.grad is not None


def test_sigma_rejects_pinball():
    with pytest.raises(ValueError):
        tg.WTEGraphNet(
            F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, sigma=True, pinball=True
        )


# --- 3-way prior gate: fac anchor / deep-regional anchor / free head --------------
def _deep_tensors(pres=(1.0, 0.0, 1.0)):
    torch.manual_seed(13)
    p = torch.tensor(pres)
    return {
        "deep_base": torch.randn(3) * p,  # absent deep is zeroed (as _fac_feat does)
        "deep_present": p,
        "deep_pred_dtw": torch.randn(3) * p,
    }


def _gate_model(seed=0):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(
        F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, prior_gate=True
    )


def test_prior_gate_expert_limits_and_masking():
    # fac present on queries 0,1; deep present on queries 0,2.
    g = {**_tiny_graph(), **_fac_tensors(), **_deep_tensors()}
    base = _base_model(seed=0).eval()
    m = _gate_model(seed=0).eval()
    # gate modules are constructed LAST -> shared modules identical at same seed.
    with torch.no_grad():
        torch.nn.init.zeros_(m.prior_gate_mlp[-1].weight)
        bias = m.prior_gate_mlp[-1].bias
        # force FAC expert: pred == fac_base where FAC present; masked where absent.
        bias.zero_()
        bias[0] = 50.0
        out = m(g)
        assert torch.allclose(out[:2], g["fac_base"][:2], atol=1e-4)
        w = m.last_prior_gate
        assert torch.allclose(w.sum(dim=-1), torch.ones(3), atol=1e-5)
        assert w[2, 0] == 0.0  # absent FAC masked out of the softmax
        # force DEEP expert: pred == deep_base where deep present; masked where absent.
        bias.zero_()
        bias[1] = 50.0
        out = m(g)
        assert torch.allclose(out[[0, 2]], g["deep_base"][[0, 2]], atol=1e-4)
        assert m.last_prior_gate[1, 1] == 0.0
        # force the FREE HEAD: pred == baseline model output exactly, everywhere.
        bias.zero_()
        bias[2] = 50.0
        assert torch.allclose(m(g), base(g), atol=1e-5)


def test_prior_gate_off_is_baseline_state_dict():
    base = _base_model(seed=0)
    m = _gate_model(seed=0)
    extra = set(m.state_dict()) - set(base.state_dict())
    assert extra and all(k.startswith("prior_gate_mlp") for k in extra)
    # shared modules bitwise-identical at the same seed (RNG-order preservation).
    sd = m.state_dict()
    assert all(torch.equal(v, sd[k]) for k, v in base.state_dict().items())


def test_prior_gate_rejects_anchor_and_pinball_combos():
    for kw in ({"fac_skip": True}, {"fac_lambda": True}, {"pinball": True}):
        with pytest.raises(ValueError):
            tg.WTEGraphNet(
                F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, prior_gate=True, **kw
            )


# --- terrain-mirror expert (4-way gate: fac / deep / mirror / head) ---------------
def _mirror_tensors():
    torch.manual_seed(17)
    return {
        "mirror_base": torch.randn(3),  # base - d, standardized; always present
        "mirror_pred_dtw": torch.zeros(3),  # constant d -> standardized 0
    }


def _mirror_gate_model(seed=0):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(
        F_REACH,
        F_QUERY,
        F_CH,
        F_LAT,
        HIDDEN,
        2,
        0.1,
        prior_gate=True,
        mirror_anchor=True,
    )


def test_mirror_gate_expert_limits_and_masking():
    g = {**_tiny_graph(), **_fac_tensors(), **_deep_tensors(), **_mirror_tensors()}
    base = _base_model(seed=0).eval()
    m = _mirror_gate_model(seed=0).eval()
    with torch.no_grad():
        torch.nn.init.zeros_(m.prior_gate_mlp[-1].weight)
        bias = m.prior_gate_mlp[-1].bias
        # force the MIRROR expert: pred == mirror_base EVERYWHERE (always present).
        bias.zero_()
        bias[2] = 50.0
        out = m(g)
        assert torch.allclose(out, g["mirror_base"], atol=1e-4)
        w = m.last_prior_gate
        assert w.shape == (3, 4)
        assert torch.allclose(w.sum(dim=-1), torch.ones(3), atol=1e-5)
        # absent-prior masking still applies to fac/deep in the 4-way softmax.
        bias.zero_()
        bias[0] = 50.0
        m(g)
        assert m.last_prior_gate[2, 0] == 0.0  # absent FAC on query 2
        # force the FREE HEAD (last expert): pred == baseline model output.
        bias.zero_()
        bias[3] = 50.0
        assert torch.allclose(m(g), base(g), atol=1e-5)


def test_mirror_gate_off_is_3way_and_requires_prior_gate():
    # without mirror_anchor the gate stays 3-way (bias length 3).
    assert _gate_model(seed=0).prior_gate_mlp[-1].bias.shape[0] == 3
    assert _mirror_gate_model(seed=0).prior_gate_mlp[-1].bias.shape[0] == 4
    with pytest.raises(ValueError):
        tg.WTEGraphNet(
            F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, mirror_anchor=True
        )


# --- last_head_out capture: the inference-render decomposition contract ------------
# The 10 m renderer recomposes wte_hat = sum w_i * expert_i (GNN_INFERENCE_10M_PLAN);
# these pin the identity against the live forward via the captured head output.
def test_last_head_out_plain_model_equals_primary():
    g = _tiny_graph()
    m = _base_model(seed=0).eval()
    with torch.no_grad():
        out = m(g)
    assert m.last_head_out is not None
    assert not m.last_head_out.requires_grad  # detached QA capture
    # no skip/gate -> the primary IS the head output.
    assert torch.equal(out, m.last_head_out)


def test_prior_gate_mixture_identity_via_last_head_out():
    g = {**_tiny_graph(), **_fac_tensors(), **_deep_tensors()}
    m = _gate_model(seed=0).eval()
    with torch.no_grad():
        out = m(g)
    w = m.last_prior_gate
    recomposed = (
        w[:, 0] * g["fac_base"] + w[:, 1] * g["deep_base"] + w[:, 2] * m.last_head_out
    )
    assert torch.allclose(out, recomposed, atol=1e-6)


def test_mirror_gate_mixture_identity_via_last_head_out():
    g = {**_tiny_graph(), **_fac_tensors(), **_deep_tensors(), **_mirror_tensors()}
    m = _mirror_gate_model(seed=0).eval()
    with torch.no_grad():
        out = m(g)
    w = m.last_prior_gate
    recomposed = (
        w[:, 0] * g["fac_base"]
        + w[:, 1] * g["deep_base"]
        + w[:, 2] * g["mirror_base"]
        + w[:, 3] * m.last_head_out
    )
    assert torch.allclose(out, recomposed, atol=1e-6)


# --- Dupuit hang expert (gate expert: fac / deep / [mirror] / hang / head) ---------
def _hang_tensors(pres=(1.0, 1.0, 0.0)):
    torch.manual_seed(19)
    p = torch.tensor(pres)
    return {
        "hang_base": torch.randn(3) * p,  # base - hang_dtw, standardized
        "hang_present": p,
        "hang_pred_dtw": torch.randn(3) * p,
    }


def _hang_gate_model(seed=0, mirror=False):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(
        F_REACH,
        F_QUERY,
        F_CH,
        F_LAT,
        HIDDEN,
        2,
        0.1,
        prior_gate=True,
        mirror_anchor=mirror,
        hang_anchor=True,
    )


def test_hang_gate_expert_limits_and_masking():
    g = {**_tiny_graph(), **_fac_tensors(), **_deep_tensors(), **_hang_tensors()}
    base = _base_model(seed=0).eval()
    m = _hang_gate_model(seed=0).eval()
    with torch.no_grad():
        torch.nn.init.zeros_(m.prior_gate_mlp[-1].weight)
        bias = m.prior_gate_mlp[-1].bias
        assert bias.shape[0] == 4  # fac, deep, hang, head
        # force the HANG expert: pred == hang_base where present, masked on query 2.
        bias.zero_()
        bias[2] = 50.0
        out = m(g)
        assert torch.allclose(out[:2], g["hang_base"][:2], atol=1e-4)
        w = m.last_prior_gate
        assert torch.allclose(w.sum(dim=-1), torch.ones(3), atol=1e-5)
        assert w[2, 2] == 0.0  # absent hang masked out of the softmax
        # force the FREE HEAD (last expert): pred == baseline model output.
        bias.zero_()
        bias[3] = 50.0
        assert torch.allclose(m(g), base(g), atol=1e-5)


def test_hang_with_mirror_is_5way_and_mixture_identity():
    g = {
        **_tiny_graph(),
        **_fac_tensors(),
        **_deep_tensors(),
        **_mirror_tensors(),
        **_hang_tensors(),
    }
    m = _hang_gate_model(seed=0, mirror=True).eval()
    assert m.prior_gate_mlp[-1].bias.shape[0] == 5
    with torch.no_grad():
        out = m(g)
    w = m.last_prior_gate
    recomposed = (
        w[:, 0] * g["fac_base"]
        + w[:, 1] * g["deep_base"]
        + w[:, 2] * g["mirror_base"]
        + w[:, 3] * g["hang_base"]
        + w[:, 4] * m.last_head_out
    )
    assert torch.allclose(out, recomposed, atol=1e-6)


def test_hang_anchor_requires_prior_gate():
    with pytest.raises(ValueError):
        tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, hang_anchor=True)


# ---------------------------------------------------------------------------
# Spatial-context read: additive attention slot over lattice ring cells
# ---------------------------------------------------------------------------
F_SC_N, F_SC_E = 5, 7


def _sc_graph():
    """_tiny_graph + a spatial-context read: q0 gets 3 cell edges, q1 gets 2, q2 none."""
    g = _tiny_graph()
    g["sc_x"] = torch.randn(6, F_SC_N)  # 6 shared lattice cells
    g["sc_ei"] = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 0, 0, 1, 1]]
    )  # cell(src) -> query(dst)
    g["sc_ea"] = torch.randn(5, F_SC_E)
    return g


def _sc_model(seed=0, **kw):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(
        F_REACH,
        F_QUERY,
        F_CH,
        F_LAT,
        HIDDEN,
        2,
        0.1,
        f_sc_node=F_SC_N,
        f_sc_edge=F_SC_E,
        **kw,
    )


def test_spatial_context_forward_shape_and_attention():
    g = _sc_graph()
    m = _sc_model().eval()
    assert m.has_sc
    assert m.head[0].in_features == HIDDEN * 3  # [q, ctx_reach] + ctx_sc
    with torch.no_grad():
        out = m(g)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()  # query 2 has no sc edge -> zero ctx, finite
    a = m.sc_read.last_attn.reshape(-1)
    assert a.shape == (5,)
    # segment softmax per query: q0's 3 edges and q1's 2 edges each sum to 1
    assert torch.isclose(a[:3].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(a[3:].sum(), torch.tensor(1.0), atol=1e-6)


def test_spatial_context_composes_with_portfolio_read():
    # the SC read is an ADDITIVE slot: it must stack on the exclusive pf slot,
    # because the production arms keep their portfolio/gate configuration.
    g = _sc_graph()
    g["pf_ei"] = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 1]])
    g["pf_ea"] = torch.randn(4, F_PF)
    torch.manual_seed(0)
    m = tg.WTEGraphNet(
        F_REACH,
        F_QUERY,
        F_CH,
        F_LAT,
        HIDDEN,
        2,
        0.1,
        f_pf=F_PF,
        f_sc_node=F_SC_N,
        f_sc_edge=F_SC_E,
    ).eval()
    assert m.has_pf and m.has_sc
    assert m.head[0].in_features == HIDDEN * 4  # [q, ctx_reach, ctx_pf, ctx_sc]
    with torch.no_grad():
        out = m(g)
    assert out.shape == (3,) and torch.isfinite(out).all()


def test_spatial_context_composes_with_prior_gate_mirror():
    # constructor-level width check for the production arm shape (gate + mirror + sigma):
    # head/gate/sigma all read head_in = hidden*3 ([q, ctx_reach, ctx_sc]).
    m = _sc_model(prior_gate=True, mirror_anchor=True, sigma=True)
    assert m.head[0].in_features == HIDDEN * 3
    assert m.sigma_head[0].in_features == HIDDEN * 3
    assert m.prior_gate_mlp[0].in_features == HIDDEN * 3 + 7 + 3
    assert m.prior_gate_mlp[-1].bias.shape[0] == 4  # fac/deep/mirror/head


def test_spatial_context_off_is_baseline_state_dict():
    base = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1)
    assert not base.has_sc
    assert not hasattr(base, "sc_read") and not hasattr(base, "sc_enc")
    assert not any(k.startswith(("sc_read", "sc_enc")) for k in base.state_dict())


def test_spatial_context_uses_the_edge_context():
    g = _sc_graph()
    m = _sc_model().eval()
    with torch.no_grad():
        out_full = m(g)
        g2 = dict(g)
        g2["sc_ei"] = torch.empty(2, 0, dtype=torch.long)  # no sc edges at all
        g2["sc_ea"] = torch.empty(0, F_SC_E)
        out_none = m(g2)
    assert not torch.allclose(out_full[:2], out_none[:2], atol=1e-5)


def test_spatial_context_requires_both_widths():
    with pytest.raises(ValueError):
        tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, f_sc_node=F_SC_N)
    with pytest.raises(ValueError):
        tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, f_sc_edge=F_SC_E)


# ---------------------------------------------------------------------------
# MAE neighborhood embedding: additive per-query head slot (no graph read)
# ---------------------------------------------------------------------------
F_MAE = 9


def _mae_graph():
    """_tiny_graph + a per-query MAE embedding (one dense vector per query node)."""
    g = _tiny_graph()
    torch.manual_seed(11)
    g["mae_x"] = torch.randn(3, F_MAE)  # 3 query nodes
    return g


def _mae_model(seed=0, **kw):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(
        F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, f_mae=F_MAE, **kw
    )


def test_mae_forward_shape_and_head_width():
    g = _mae_graph()
    m = _mae_model().eval()
    assert m.has_mae
    assert m.head[0].in_features == HIDDEN * 3  # [q, ctx_reach] + ctx_mae
    with torch.no_grad():
        out = m(g)
    assert out.shape == (3,) and torch.isfinite(out).all()


def test_mae_off_is_baseline_state_dict():
    # f_mae=None must add no params and keep the plain hidden*2 head (keys unchanged),
    # i.e. a run without --mae-embeddings is byte-identical to the baseline model.
    base = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1)
    assert not base.has_mae
    assert not hasattr(base, "mae_enc")
    assert base.head[0].in_features == HIDDEN * 2
    assert not any(k.startswith("mae_enc") for k in base.state_dict())


def test_mae_composes_with_prior_gate_mirror_sigma():
    # the production arm shape (gate + mirror + sigma): head + sigma read head_in =
    # hidden*3 ([q, ctx_reach, ctx_mae]); the gate keeps its 4 experts.
    m = _mae_model(prior_gate=True, mirror_anchor=True, sigma=True)
    assert m.head[0].in_features == HIDDEN * 3
    assert m.sigma_head[0].in_features == HIDDEN * 3
    assert m.prior_gate_mlp[0].in_features == HIDDEN * 3 + 7 + 3
    assert m.prior_gate_mlp[-1].bias.shape[0] == 4  # fac/deep/mirror/head


def test_mae_composes_with_spatial_context():
    # two ADDITIVE slots stack: head_in = [q, ctx_reach, ctx_sc, ctx_mae] = hidden*4.
    g = _mae_graph()
    g["sc_x"] = torch.randn(6, F_SC_N)
    g["sc_ei"] = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 1]])
    g["sc_ea"] = torch.randn(5, F_SC_E)
    torch.manual_seed(0)
    m = tg.WTEGraphNet(
        F_REACH,
        F_QUERY,
        F_CH,
        F_LAT,
        HIDDEN,
        2,
        0.1,
        f_sc_node=F_SC_N,
        f_sc_edge=F_SC_E,
        f_mae=F_MAE,
    ).eval()
    assert m.has_sc and m.has_mae
    assert m.head[0].in_features == HIDDEN * 4
    with torch.no_grad():
        out = m(g)
    assert out.shape == (3,) and torch.isfinite(out).all()


def test_mae_uses_the_embedding():
    # two different embeddings must yield different predictions -> the slot is live.
    g = _mae_graph()
    m = _mae_model().eval()
    with torch.no_grad():
        out_a = m(g)
        g2 = dict(g)
        g2["mae_x"] = g["mae_x"] + 3.0
        out_b = m(g2)
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_mae_missing_tensor_errors_loudly():
    g = _mae_graph()
    del g["mae_x"]
    m = _mae_model().eval()
    with pytest.raises(KeyError):
        m(g)
