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
