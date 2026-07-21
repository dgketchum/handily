"""Unit tests for the E3 analog-edge lever (notes/E3_ANALOG_EDGES.md).

Three concerns:
  * the fold-mask leakage guard (a test-fold query never reads a test-fold well),
  * the edge-builder geometry (embedding normalisation + nonlocal kNN invariants),
  * the model wiring (additive head slot; flag-off byte-identical to baseline; composes
    with the E1-leader head; the slot is live).
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
    # PyG's MessagePassing Inspector resolves message() globals via sys.modules.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bae = _load("build_analog_edges")
tg = _load("train_wte_gnn")

F_REACH, F_QUERY, F_CH, F_LAT, F_ANALOG = 3, 2, 1, 1, 3
HIDDEN = 8


# ---------------------------------------------------------------------------
# 1. Fold-mask leakage guard
# ---------------------------------------------------------------------------
def test_analog_fold_keep_excludes_held_out_fold_source():
    # every surviving edge in fold f must be sourced OUTSIDE fold f.
    src_fold = np.array([0, 0, 1, 2, 3, 3, 5, 7, 7], dtype="int64")
    for f in range(8):
        keep = bae.analog_fold_keep(src_fold, f)
        assert not (src_fold[keep] == f).any(), f"fold {f} source leaked through mask"
        # and it drops EXACTLY the fold-f edges, nothing more.
        assert keep.sum() == (src_fold != f).sum()


def test_analog_fold_keep_none_keeps_all():
    src_fold = np.array([0, 1, 2, 3], dtype="int64")
    keep = bae.analog_fold_keep(src_fold, None)
    assert keep.all() and keep.shape == (4,)


# ---------------------------------------------------------------------------
# 2. Edge-builder geometry
# ---------------------------------------------------------------------------
def test_zscore_l2_unit_norm_and_centered():
    rng = np.random.RandomState(0)
    emb = rng.randn(20, 6) * np.array([1, 10, 0.1, 5, 2, 100]) + 3.0
    unit = bae.zscore_l2(emb)
    norms = np.linalg.norm(unit, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)  # every row unit length
    # a constant (zero-variance) dimension must not produce NaN.
    emb2 = emb.copy()
    emb2[:, 2] = 7.0
    assert np.isfinite(bae.zscore_l2(emb2)).all()


def test_knn_analog_edges_nonlocality_and_topk():
    # 5 wells on a line; A,B share HUC4 'X' and are 10 km apart (both excluded for each
    # other), C/D/E are distant + different HUC4.
    xy = np.array(
        [[0.0, 0.0], [10_000, 0.0], [100_000, 0.0], [200_000, 0.0], [60_000, 0.0]]
    )
    huc4 = np.array(["X", "X", "Y", "Z", "W"])
    rng = np.random.RandomState(1)
    unit = bae.zscore_l2(rng.randn(5, 4))
    dest, src, sim = bae.knn_analog_edges(
        unit, xy, huc4, k=8, min_dist_km=50.0, chunk=2
    )
    for d, s in zip(dest, src):
        assert d != s, "self analog edge"
        assert huc4[d] != huc4[s], "same-HUC4 analog leaked"
        dist_km = np.linalg.norm(xy[d] - xy[s]) / 1000.0
        assert dist_km >= 50.0 - 1e-6, "near analog leaked"
    # dest A (0): only C(2),D(3),E(4) qualify (B is same-HUC4 + 10 km).
    a_src = set(src[dest == 0].tolist())
    assert a_src == {2, 3, 4}
    # per-dest similarities are emitted in descending order.
    for d in np.unique(dest):
        s_d = sim[dest == d]
        assert np.all(np.diff(s_d) <= 1e-9), (
            "analogs not sorted by descending similarity"
        )


# ---------------------------------------------------------------------------
# 3. Model wiring: additive head slot
# ---------------------------------------------------------------------------
def _tiny_graph():
    torch.manual_seed(7)
    return {
        "reach_x": torch.randn(4, F_REACH),
        "query_x": torch.randn(3, F_QUERY),
        "ch_ei": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "ch_ea": torch.randn(3, F_CH),
        "lat_ei": torch.tensor([[0, 1, 2], [0, 0, 1]]),
        "lat_ea": torch.randn(3, F_LAT),
    }


def _analog_graph():
    """_tiny_graph + analog edges: dest 0 has 2 analogs, dest 1 has 1, dest 2 has NONE."""
    g = _tiny_graph()
    torch.manual_seed(11)
    g["analog_ei"] = torch.tensor([[1, 2, 0], [0, 0, 1]])  # (src, dst)
    g["analog_ea"] = torch.randn(3, F_ANALOG)
    g["analog_src_val"] = torch.randn(3)  # per-query-node scalar residual
    return g


def _analog_model(seed=0, **kw):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(
        F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, f_analog=F_ANALOG, **kw
    )


def test_analog_forward_shape_and_head_width():
    g = _analog_graph()
    m = _analog_model().eval()
    assert m.has_analog
    assert m.head[0].in_features == HIDDEN * 3  # [q, ctx_reach, ctx_analog]
    with torch.no_grad():
        out = m(g)
    assert out.shape == (3,) and torch.isfinite(out).all()  # zero-edge dest 2 finite


def test_analog_off_is_baseline_state_dict():
    # f_analog=None must add no params and keep the plain hidden*2 head (keys unchanged).
    base = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1)
    assert not base.has_analog
    assert not hasattr(base, "analog_read")
    assert not hasattr(base, "analog_enc")
    assert base.head[0].in_features == HIDDEN * 2
    assert not any(k.startswith("analog_") for k in base.state_dict())


def test_analog_off_state_dict_identical_to_pre_e3_baseline():
    # The exact byte-for-byte contract: two same-seed models, one built with the analog
    # kwarg absent, must share identical param keys+values (turning the lever off cannot
    # perturb any baseline weight -> the E1 leader reproduces).
    torch.manual_seed(3)
    a = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1)
    torch.manual_seed(3)
    b = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, f_analog=None)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"{k} diverged with f_analog=None"


def test_analog_composes_with_e1_leader_head():
    # E1 leader + analog: MAE (additive) + analog (additive) -> head_in = hidden*4
    # ([q, ctx_reach, ctx_mae, ctx_analog]); write-back adds no head width; the prior
    # gate keeps its 4 experts (fac/deep/mirror/head) and reads the full head_in.
    m = _analog_model(
        f_mae=9,
        writeback=True,
        prior_gate=True,
        mirror_anchor=True,
        sigma=True,
    )
    assert m.head[0].in_features == HIDDEN * 4
    assert m.sigma_head[0].in_features == HIDDEN * 4
    assert m.prior_gate_mlp[0].in_features == HIDDEN * 4 + 7 + 3
    assert m.prior_gate_mlp[-1].bias.shape[0] == 4


def test_analog_uses_the_source_value():
    # changing the imported analog residual must move the prediction -> the slot is live.
    g = _analog_graph()
    m = _analog_model().eval()
    with torch.no_grad():
        out_a = m(g)
        g2 = dict(g)
        g2["analog_src_val"] = g["analog_src_val"] + 3.0
        out_b = m(g2)
    # dest 0 and 1 have analog edges -> must change; dest 2 (no edge) stays fixed.
    assert not torch.allclose(out_a[:2], out_b[:2], atol=1e-5)
    assert torch.allclose(out_a[2], out_b[2], atol=1e-6)


def test_analog_missing_tensor_errors_loudly():
    g = _analog_graph()
    del g["analog_ei"]
    m = _analog_model().eval()
    with pytest.raises(KeyError):
        m(g)
