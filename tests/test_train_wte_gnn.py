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
