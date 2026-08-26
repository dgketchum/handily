"""Unit tests for source-well assimilation rungs 0 (--source-obs) and 1 (--source-edges).

See notes/SOURCE_WELL_ASSIMILATION_PLAN.md. Three concerns:
  * model wiring -- the src block requires the 6C write-back (it is the only
    source->query path at rung 0), widens ONLY query_enc, is byte-identical to
    baseline when off, errors loudly when src_x is missing, and actually carries a
    source obs to a NEIGHBOURING query through writeback -> channel -> lateral;
  * the masked-label protocol in train_fold -- a held-out (test) row never sees its
    own obs in the final/OOF forward, the per-epoch source draw is redrawn within the
    configured fraction range from the train pool only, and the val / final forwards
    use their deterministic source semantics (all-train / train+val);
  * the leak guard on the loss -- a drawn source contributes zero loss (its label,
    visible to the model, cannot pull the parameters).
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


tg = _load("train_wte_gnn")
tc = _load("train_conus_gnn")

F_REACH, F_QUERY, F_CH, F_LAT, F_SRC = 3, 2, 1, 1, 2
HIDDEN = 8


# ---------------------------------------------------------------------------
# 1. Model wiring
# ---------------------------------------------------------------------------
def _wb_graph():
    """2 reaches, 3 queries; queries 0 and 1 share reach 0, query 2 sits on reach 1."""
    torch.manual_seed(7)
    lat_ei = torch.tensor([[0, 0, 1], [0, 1, 2]])  # (reach src, query dst)
    return {
        "reach_x": torch.randn(2, F_REACH),
        "query_x": torch.randn(3, F_QUERY),
        "ch_ei": torch.tensor([[0], [1]]),
        "ch_ea": torch.randn(1, F_CH),
        "lat_ei": lat_ei,
        "lat_ea": torch.randn(3, F_LAT),
        "lat_ei_reversed": lat_ei.flip(0),
        "src_x": torch.zeros(3, F_SRC),
    }


def _src_model(seed=0, **kw):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(
        F_REACH,
        F_QUERY,
        F_CH,
        F_LAT,
        HIDDEN,
        2,
        0.1,
        f_src=F_SRC,
        writeback=True,
        **kw,
    )


def test_source_obs_requires_writeback():
    with pytest.raises(ValueError, match="query-writeback"):
        tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, f_src=F_SRC)


def test_source_obs_widens_only_query_enc():
    torch.manual_seed(1)
    base = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, writeback=True)
    m = _src_model(seed=1)
    sa, sb = base.state_dict(), m.state_dict()
    assert sa.keys() == sb.keys()  # no new modules, no head widening
    for k in sa:
        if k == "query_enc.0.weight":
            assert sb[k].shape == (HIDDEN, F_QUERY + F_SRC)
            assert sa[k].shape == (HIDDEN, F_QUERY)
        else:
            assert sa[k].shape == sb[k].shape, f"{k} changed shape"


def test_source_obs_off_state_dict_identical_to_baseline():
    torch.manual_seed(3)
    a = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, writeback=True)
    torch.manual_seed(3)
    b = tg.WTEGraphNet(
        F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, f_src=None, writeback=True
    )
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"{k} diverged with f_src=None"


def test_source_obs_missing_tensor_errors_loudly():
    g = _wb_graph()
    del g["src_x"]
    m = _src_model().eval()
    with pytest.raises(KeyError):
        m(g)


def test_source_obs_propagates_to_neighbouring_query():
    # flipping query 0's obs must move query 1 (shared reach 0, two hops through the
    # write-back) and query 0 itself; query 2 changes only via the channel edge 0->1,
    # so at least assert the shared-reach neighbour moves.
    g = _wb_graph()
    m = _src_model().eval()
    with torch.no_grad():
        out_a = m(g)
        g2 = dict(g)
        src2 = g["src_x"].clone()
        src2[0, 0], src2[0, 1] = 5.0, 1.0  # obs value + valid bit on query 0
        g2["src_x"] = src2
        out_b = m(g2)
    assert not torch.allclose(out_a[0], out_b[0], atol=1e-6)
    assert not torch.allclose(out_a[1], out_b[1], atol=1e-6)


# ---------------------------------------------------------------------------
# 2. train_fold masked-label protocol
# ---------------------------------------------------------------------------
N = 20
FMIN, FMAX = 0.3, 0.9
SEED = 123


def _partition():
    tr = np.zeros(N, bool)
    va = np.zeros(N, bool)
    tr[:12] = True  # 0-11 train
    va[12:16] = True  # 12-15 val
    return tr, va  # 16-19 test (neither)


def _args(epochs=1):
    return SimpleNamespace(
        lr=0.1,
        weight_decay=0.0,
        epochs=epochs,
        min_epochs=epochs,
        patience=epochs + 1,
        pinball=False,
        pinball_weight=0.0,
        huber_delta=1.0,
        huber_delta_m=1.0,
        deep_regime_threshold_m=30.0,
    )


def _src_ctx(y, tr, va, fmin=FMIN, fmax=FMAX):
    eligible = np.ones(N, bool)
    return {
        "val_std": torch.as_tensor(y, dtype=torch.float32),
        "tr_pool": tr & eligible,
        "trva_pool": (tr | va) & eligible,
        "frac": (fmin, fmax),
        "rng": np.random.default_rng(SEED),
    }


class _CopyModel(torch.nn.Module):
    """pred = own src obs value (the cheat the mask protocol must defuse)."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, feat):
        return feat["src_x"][:, 0] + 0.0 * self.p


def test_final_forward_masks_test_rows_and_shows_trva_sources():
    tr, va = _partition()
    y = np.linspace(1.0, 4.0, N)  # nonzero everywhere so masking is observable
    y_std = torch.as_tensor(y, dtype=torch.float32)
    feat = {"query_x": torch.zeros(N, 1)}
    native, _, _ = tc.train_fold(
        _CopyModel(),
        feat,
        y_std,
        tr,
        va,
        np.zeros(N),
        y.copy(),
        0.0,
        1.0,
        tc.TARGET_DTW_RESIDUAL,
        0.85,
        _args(),
        "cpu",
        None,
        src_ctx=_src_ctx(y, tr, va),
    )
    test = ~(tr | va)
    # a test row NEVER carries its own obs -> the copy model reads exactly 0 there.
    assert np.allclose(native[test], 0.0)
    # train+val rows are sources in the final forward.
    assert np.allclose(native[tr | va], y[tr | va])
    # and feat["src_x"] was left at the final semantics (valid bit = trva pool).
    assert np.array_equal(feat["src_x"][:, 1].numpy().astype(bool), tr | va)


class _RecorderModel(torch.nn.Module):
    """records the src valid-bit mask of every forward, split train/eval."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.train_masks: list[np.ndarray] = []
        self.eval_masks: list[np.ndarray] = []

    def forward(self, feat):
        mask = feat["src_x"][:, 1].detach().numpy().astype(bool)
        (self.train_masks if self.training else self.eval_masks).append(mask)
        return feat["query_x"].squeeze(-1) * self.p


def test_per_epoch_source_redraw_fraction_and_eval_semantics():
    tr, va = _partition()
    y = np.linspace(-1.0, 1.0, N)
    epochs = 6
    model = _RecorderModel()
    tc.train_fold(
        model,
        {"query_x": torch.zeros(N, 1)},
        torch.as_tensor(y, dtype=torch.float32),
        tr,
        va,
        np.zeros(N),
        y.copy(),
        0.0,
        1.0,
        tc.TARGET_DTW_RESIDUAL,
        0.85,
        _args(epochs=epochs),
        "cpu",
        None,
        src_ctx=_src_ctx(y, tr, va),
    )
    pool = tr
    n_pool = int(pool.sum())
    assert len(model.train_masks) == epochs
    for m in model.train_masks:
        assert not m[~pool].any(), "source drawn outside the train pool"
        frac = m.sum() / n_pool
        assert FMIN - 0.5 / n_pool <= frac <= FMAX + 0.5 / n_pool
    # redrawn, not frozen: the epochs do not all share one mask.
    assert len({m.tobytes() for m in model.train_masks}) > 1
    # eval forwards: one per epoch with ALL train sources, then the final trva forward.
    assert len(model.eval_masks) == epochs + 1
    for m in model.eval_masks[:-1]:
        assert np.array_equal(m, pool)
    assert np.array_equal(model.eval_masks[-1], tr | va)


class _BiasModel(torch.nn.Module):
    """pred = b everywhere; only a loss that bites can move b off 0."""

    def __init__(self):
        super().__init__()
        self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, feat):
        return self.b.expand(feat["query_x"].shape[0])


def test_drawn_sources_contribute_zero_loss():
    tr, va = _partition()
    # replicate the fold rng to learn which train rows the first epoch draws.
    ctx_probe = _src_ctx(np.zeros(N), tr, va)
    rng = np.random.default_rng(SEED)
    pool_idx = np.flatnonzero(ctx_probe["tr_pool"])
    k = int(round(rng.uniform(FMIN, FMAX) * len(pool_idx)))
    drawn = np.zeros(N, bool)
    drawn[rng.choice(pool_idx, size=k, replace=False)] = True
    assert drawn.any() and (tr & ~drawn).any()
    # huge labels ONLY on the drawn sources; every scored target has label 0. If the
    # sources leaked into the loss, the bias would take a step toward +100.
    y = np.where(drawn, 100.0, 0.0)
    model = _BiasModel()
    tc.train_fold(
        model,
        {"query_x": torch.zeros(N, 1)},
        torch.as_tensor(y, dtype=torch.float32),
        tr,
        va,
        np.zeros(N),
        y.copy(),
        0.0,
        1.0,
        tc.TARGET_DTW_RESIDUAL,
        0.85,
        _args(epochs=1),
        "cpu",
        None,
        src_ctx=_src_ctx(y, tr, va),
    )
    assert float(model.b.detach()) == 0.0
    # control: the SAME setup without the source protocol trains on those labels.
    model2 = _BiasModel()
    tc.train_fold(
        model2,
        {"query_x": torch.zeros(N, 1)},
        torch.as_tensor(y, dtype=torch.float32),
        tr,
        va,
        np.zeros(N),
        y.copy(),
        0.0,
        1.0,
        tc.TARGET_DTW_RESIDUAL,
        0.85,
        _args(epochs=1),
        "cpu",
        None,
    )
    assert float(model2.b.detach()) > 0.0


# ---------------------------------------------------------------------------
# 3. Rung 1: source-edge slot (--source-edges)
# ---------------------------------------------------------------------------
F_SE = 4  # edge-feature width (build_source_edges.EDGE_COLS)


def _se_graph():
    """2 reaches, 3 queries; source edges 0->2 and 1->2 (query 2 reads both)."""
    torch.manual_seed(11)
    g = {
        "reach_x": torch.randn(2, F_REACH),
        "query_x": torch.randn(3, F_QUERY),
        "ch_ei": torch.tensor([[0], [1]]),
        "ch_ea": torch.randn(1, F_CH),
        "lat_ei": torch.tensor([[0, 0, 1], [0, 1, 2]]),
        "lat_ea": torch.randn(3, F_LAT),
        "srcedge_ei": torch.tensor([[0, 1], [2, 2]]),
        "srcedge_ea": torch.randn(2, F_SE),
        "srcedge_val": torch.zeros(3),
    }
    return g


def _se_model(seed=0, **kw):
    torch.manual_seed(seed)
    return tg.WTEGraphNet(
        F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, f_srcedge=F_SE, **kw
    )


def test_source_edges_do_not_require_writeback():
    m = _se_model()  # no ValueError: the edges are their own transmission path
    assert m.has_srcedge and not m.writeback


def test_source_edges_add_only_the_two_read_modules():
    torch.manual_seed(2)
    base = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1)
    m = _se_model(seed=2)
    sa, sb = base.state_dict(), m.state_dict()
    new = {k for k in sb if k not in sa}
    assert new and all(k.startswith(("source_enc.", "source_read.")) for k in new), (
        f"unexpected new params: {sorted(new)}"
    )
    for k in sa:
        if k.startswith("head."):
            continue  # head widens by design (additive slot)
        assert sa[k].shape == sb[k].shape, f"{k} changed shape"


def test_source_edges_off_state_dict_identical_to_baseline():
    torch.manual_seed(4)
    a = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1)
    torch.manual_seed(4)
    b = tg.WTEGraphNet(F_REACH, F_QUERY, F_CH, F_LAT, HIDDEN, 2, 0.1, f_srcedge=None)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"{k} diverged with f_srcedge=None"


def test_source_edge_value_reaches_dest_not_source():
    # flipping source 0's obs value must move the edge DEST (query 2) and must NOT
    # move query 0 itself (no incoming edge -> a source never reads its own obs
    # through the edge path) nor query 1 (its value unchanged, no path to it).
    g = _se_graph()
    m = _se_model().eval()
    with torch.no_grad():
        out_a = m(g)
        g2 = dict(g)
        v = g["srcedge_val"].clone()
        v[0] = 5.0
        g2["srcedge_val"] = v
        out_b = m(g2)
    assert not torch.allclose(out_a[2], out_b[2], atol=1e-6)
    assert torch.allclose(out_a[0], out_b[0], atol=1e-6)
    assert torch.allclose(out_a[1], out_b[1], atol=1e-6)


def test_zero_edge_queries_stay_finite():
    g = _se_graph()
    g["srcedge_ei"] = torch.zeros((2, 0), dtype=torch.long)
    g["srcedge_ea"] = torch.zeros((0, F_SE))
    m = _se_model().eval()
    with torch.no_grad():
        out = m(g)
    assert torch.isfinite(out).all()


def test_source_edge_gated_swaps_read_module_and_keeps_semantics():
    """--source-edge-gated: sigmoid-gate read (EdgeGatedConv) replaces softmax
    attention; the dest-not-source and zero-edge invariants must still hold."""
    m = _se_model(srcedge_gated=True).eval()
    keys = [k for k in m.state_dict() if k.startswith("source_read.")]
    assert any(".gate_mlp." in k for k in keys)
    assert not any(".score_mlp." in k for k in keys)
    g = _se_graph()
    with torch.no_grad():
        out_a = m(g)
        g2 = dict(g)
        v = g["srcedge_val"].clone()
        v[0] = 5.0
        g2["srcedge_val"] = v
        out_b = m(g2)
    assert torch.isfinite(out_a).all()
    assert not torch.allclose(out_a[2], out_b[2], atol=1e-6)
    assert torch.allclose(out_a[0], out_b[0], atol=1e-6)
    assert torch.allclose(out_a[1], out_b[1], atol=1e-6)
    g["srcedge_ei"] = torch.zeros((2, 0), dtype=torch.long)
    g["srcedge_ea"] = torch.zeros((0, F_SE))
    with torch.no_grad():
        out = m(g)
    assert torch.isfinite(out).all()


class _EdgeRecorderModel(torch.nn.Module):
    """records the visible-source set implied by feat's edge subset each forward."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.train_srcs: list[set] = []
        self.eval_srcs: list[set] = []
        self.saw_src_x: bool | None = None

    def forward(self, feat):
        srcs = set(feat["srcedge_ei"][0].tolist())
        (self.train_srcs if self.training else self.eval_srcs).append(srcs)
        self.saw_src_x = "src_x" in feat
        return feat["query_x"].squeeze(-1) * self.p


def _edge_ctx(y, tr, va):
    """ring edges i -> (i+1) % N: every node is a potential source of one edge."""
    ctx = _src_ctx(y, tr, va)
    ctx["has_x"] = False  # rung-1-only arm: no src_x feature block
    src = torch.arange(N, dtype=torch.long)
    dst = (src + 1) % N
    ctx["edge_src"] = src
    ctx["edge_ei"] = torch.stack([src, dst])
    ctx["edge_ea"] = torch.randn(N, F_SE)
    return ctx


def test_source_edge_visibility_follows_the_mask_protocol():
    tr, va = _partition()
    y = np.linspace(-1.0, 1.0, N)
    epochs = 5
    model = _EdgeRecorderModel()
    torch.manual_seed(0)
    tc.train_fold(
        model,
        {"query_x": torch.zeros(N, 1)},
        torch.as_tensor(y, dtype=torch.float32),
        tr,
        va,
        np.zeros(N),
        y.copy(),
        0.0,
        1.0,
        tc.TARGET_DTW_RESIDUAL,
        0.85,
        _args(epochs=epochs),
        "cpu",
        None,
        src_ctx=_edge_ctx(y, tr, va),
    )
    pool = set(np.flatnonzero(tr))
    n_pool = len(pool)
    assert len(model.train_srcs) == epochs
    for srcs in model.train_srcs:
        assert srcs <= pool, "edge kept from a source outside the train pool"
        frac = len(srcs) / n_pool
        assert FMIN - 0.5 / n_pool <= frac <= FMAX + 0.5 / n_pool
    assert len({frozenset(s) for s in model.train_srcs}) > 1, "edge mask frozen"
    # eval forwards: all-train sources per epoch, then the final train+val forward.
    assert len(model.eval_srcs) == epochs + 1
    for srcs in model.eval_srcs[:-1]:
        assert srcs == pool
    assert model.eval_srcs[-1] == set(np.flatnonzero(tr | va))
    # has_x=False: the rung-0 feature block is never set on an edges-only arm.
    assert model.saw_src_x is False


def test_val_mix_adds_a_no_source_eval_forward_each_epoch():
    """--source-val-mix: early stopping sees TWO eval forwards per epoch -- an
    empty-source one then the all-train one -- so model selection cannot favor
    edge-reliant weights (rung 1's second protocol defect)."""
    tr, va = _partition()
    y = np.linspace(-1.0, 1.0, N)
    epochs = 3
    model = _EdgeRecorderModel()
    ctx = _edge_ctx(y, tr, va)
    ctx["val_mix"] = True
    tc.train_fold(
        model,
        {"query_x": torch.zeros(N, 1)},
        torch.as_tensor(y, dtype=torch.float32),
        tr,
        va,
        np.zeros(N),
        y.copy(),
        0.0,
        1.0,
        tc.TARGET_DTW_RESIDUAL,
        0.85,
        _args(epochs=epochs),
        "cpu",
        None,
        src_ctx=ctx,
    )
    pool = set(np.flatnonzero(tr))
    # per epoch: empty-source forward, then all-train forward; final trva forward.
    assert len(model.eval_srcs) == 2 * epochs + 1
    for i in range(epochs):
        assert model.eval_srcs[2 * i] == set(), "no-source forward missing"
        assert model.eval_srcs[2 * i + 1] == pool
    assert model.eval_srcs[-1] == set(np.flatnonzero(tr | va))


def test_edges_only_arms_keep_drawn_source_loss():
    """Rung 1b: without src_x a source never sees its own label (self/same-site
    edges are dropped at build time), so its loss is legitimate supervision and
    must NOT be zeroed -- zeroing it anyway threw away ~60% of train labels per
    epoch and caused rung 1's cold regression."""
    tr, va = _partition()
    # replicate the fold rng to learn which train rows the first epoch draws.
    rng = np.random.default_rng(SEED)
    pool_idx = np.flatnonzero(tr)
    k = int(round(rng.uniform(FMIN, FMAX) * len(pool_idx)))
    drawn = np.zeros(N, bool)
    drawn[rng.choice(pool_idx, size=k, replace=False)] = True
    assert drawn.any()
    # huge labels ONLY on the drawn sources: the bias moves iff their loss bites.
    y = np.where(drawn, 100.0, 0.0)
    model = _BiasModel()
    tc.train_fold(
        model,
        {"query_x": torch.zeros(N, 1)},
        torch.as_tensor(y, dtype=torch.float32),
        tr,
        va,
        np.zeros(N),
        y.copy(),
        0.0,
        1.0,
        tc.TARGET_DTW_RESIDUAL,
        0.85,
        _args(epochs=1),
        "cpu",
        None,
        src_ctx=_edge_ctx(y, tr, va),
    )
    assert float(model.b.detach()) > 0.0
