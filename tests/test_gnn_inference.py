"""Unit tests for the GNN inference pipeline (utils/infer_conus_gnn.py + render_gnn_10m.py).

Covers the pure pieces of the Phases 1-3 contract (the end-to-end persisted-
contract replay is validated separately by the runner's --oof-check gate,
which reproduces the archived OOF predictions to <1e-3 m):

  * snapped_window -- canonical-origin snap, coverage, 100 m / 10 m nesting
  * build_anchors -- presence from finiteness, pred_dtw = base - raw, mirror
  * assert_mixture_identity -- passes on a hand-built convex mixture (incl. an
    absent-FAC row with masked weight 0), raises on a perturbed output
  * prune_for_queries -- hop-limited reach retention + exact index remap
  * recompose -- exact convex identity where FAC present, (1 - w_fac)
    renormalization where absent, head fallback under the guard
  * master_window -- 10 m window nests the coarse window exactly
  * leak_gate -- enforced on the two-signal leak signature, informational (never
    failing) for source-assimilation arms that read the well pool by design
  * ckpt_extensions / ckpt_f_src / ckpt_f_srcedge -- state-dict introspection of
    the MAE head, writeback branch and source-read slot
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

_UTILS = Path(__file__).resolve().parents[1] / "utils"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _UTILS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


inf = _load("infer_conus_gnn")
ren = _load("render_gnn_10m")


# ---------------------------------------------------------------------------
# snapped_window
# ---------------------------------------------------------------------------
def test_snapped_window_on_canonical_origin():
    t, w, h = inf.snapped_window((-1000037.0, 1500011.0, -999512.0, 1500499.0), 100.0)
    assert (t.c - inf.X0) % 100 == 0 and (t.f - inf.Y0) % 100 == 0
    # snapped window covers the requested bounds
    assert t.c <= -1000037.0 and t.f >= 1500499.0
    assert t.c + w * 100 >= -999512.0 and t.f - h * 100 <= 1500011.0
    # snap is tight: within one cell of the raw bounds
    assert -1000037.0 - t.c < 100 and t.f - 1500499.0 < 100


def test_snapped_window_10m_nests_100m():
    bounds = (-1000037.0, 1500011.0, -999512.0, 1500499.0)
    t100, w100, h100 = inf.snapped_window(bounds, 100.0)
    # the renderer's master grid: same top-left, x10 finer
    assert (t100.c - inf.X0) % 10 == 0 and (t100.f - inf.Y0) % 10 == 0
    # every 100 m cell edge is a 10 m cell edge
    assert (100 * w100) % 10 == 0 and (100 * h100) % 10 == 0


# ---------------------------------------------------------------------------
# build_anchors
# ---------------------------------------------------------------------------
def test_build_anchors_contract():
    base = np.array([10.0, 20.0, 30.0])
    fac_raw = np.array([2.0, np.nan, -1.0])
    deep_raw = np.array([-5.0, -8.0, np.nan])
    a = inf.build_anchors(base, fac_raw, deep_raw, 3.0, mirror_on=True)
    np.testing.assert_array_equal(a["fac_present"], [True, False, True])
    np.testing.assert_array_equal(a["deep_present"], [True, True, False])
    # pred_dtw is the prior's own DTW estimate: base - raw
    np.testing.assert_allclose(a["fac_pred_dtw"][[0, 2]], [8.0, 31.0])
    assert np.isnan(a["fac_pred_dtw"][1])
    np.testing.assert_allclose(a["deep_pred_dtw"][:2], [15.0, 28.0])
    # mirror: constant-depth surface in residual space, pred_dtw = d
    np.testing.assert_allclose(a["mirror_raw"], base - 3.0)
    assert a["mirror_present"].all()
    np.testing.assert_allclose(a["mirror_pred_dtw"], 3.0)


def test_build_anchors_no_mirror():
    a = inf.build_anchors(np.zeros(2), np.zeros(2), np.zeros(2), 3.0, mirror_on=False)
    assert "mirror_raw" not in a


# ---------------------------------------------------------------------------
# mixture identity
# ---------------------------------------------------------------------------
def _toy_mixture():
    r = np.array([100.0, 200.0])
    base = np.array([5.0, 8.0])
    anchors = inf.build_anchors(
        base,
        np.array([1.0, np.nan]),  # FAC absent on row 1
        np.array([-2.0, -3.0]),
        3.0,
        mirror_on=True,
    )
    # masked softmax: absent FAC -> weight exactly 0 on row 1
    w = np.array([[0.4, 0.3, 0.2, 0.1], [0.0, 0.5, 0.3, 0.2]])
    head_wte = np.array([104.0, 207.0])
    wte = np.zeros(2)
    experts = [
        r + np.nan_to_num(anchors["fac_raw"]),
        r + anchors["deep_raw"],
        r + anchors["mirror_raw"],
        head_wte,
    ]
    pres = [
        anchors["fac_present"],
        np.ones(2, bool),
        np.ones(2, bool),
        np.ones(2, bool),
    ]
    for i, (e, p) in enumerate(zip(experts, pres)):
        wte += w[:, i] * np.where(p, e, 0.0)
    return {"wte": wte, "w": w, "head_wte": head_wte}, r, anchors


def test_mixture_identity_passes():
    out, r, anchors = _toy_mixture()
    inf.assert_mixture_identity(out, r, anchors, fold=0, where="toy")


def test_mixture_identity_fails_on_perturbation():
    out, r, anchors = _toy_mixture()
    out["wte"] = out["wte"] + 0.01
    with pytest.raises(SystemExit, match="mixture identity FAILED"):
        inf.assert_mixture_identity(out, r, anchors, fold=0, where="toy")


def test_mixture_identity_rejects_gate_width_skew():
    out, r, anchors = _toy_mixture()
    out["w"] = out["w"][:, :3]
    with pytest.raises(SystemExit, match="gate width"):
        inf.assert_mixture_identity(out, r, anchors, fold=0, where="toy")


# ---------------------------------------------------------------------------
# prune_for_queries
# ---------------------------------------------------------------------------
def test_prune_for_queries_hops_and_remap():
    # chain 0-1-2-3-4 (downstream edges), query attached to reach 0 only.
    rn = pd.DataFrame({"reach_node_idx": range(5), "feat": np.arange(5.0)})
    ce = pd.DataFrame(
        {
            "src_reach_idx": [0, 1, 2, 3],
            "dst_reach_idx": [1, 2, 3, 4],
            "direction": [1, 1, 1, 1],
        }
    )
    lat = pd.DataFrame({"query_node_idx": [0], "reach_node_idx": [0]})
    rn2, ce2, lat2 = inf.prune_for_queries(rn, ce, lat, channel_layers=2)
    # 2 hops from reach 0 keeps {0,1,2}; edge 2->3 dropped (endpoint pruned)
    assert list(rn2["feat"]) == [0.0, 1.0, 2.0]
    assert len(ce2) == 2
    assert set(zip(ce2["src_reach_idx"], ce2["dst_reach_idx"])) == {(0, 1), (1, 2)}
    assert lat2["reach_node_idx"].tolist() == [0]


# ---------------------------------------------------------------------------
# recompose (renderer)
# ---------------------------------------------------------------------------
def test_recompose_exact_where_fac_present():
    dem = np.full((1, 3), 100.0)
    fac = np.array([[2.0, 4.0, 1.0]])
    w = {
        "fac": np.full((1, 3), 0.4),
        "deep": np.full((1, 3), 0.3),
        "mirror": np.full((1, 3), 0.2),
        "head": np.full((1, 3), 0.1),
    }
    deep = np.full((1, 3), 95.0)
    head = np.full((1, 3), 99.0)
    wte, fb = ren.recompose(dem, fac, w, deep, head, d=3.0, mirror_on=True)
    expect = 0.4 * (dem - fac) + 0.3 * deep + 0.2 * (dem - 3.0) + 0.1 * head
    np.testing.assert_allclose(wte, expect)
    assert not fb.any()


def test_recompose_renormalizes_where_fac_absent():
    dem = np.full((1, 2), 100.0)
    fac = np.array([[np.nan, np.nan]])
    w = {
        "fac": np.array([[0.4, 0.99]]),
        "deep": np.array([[0.3, 0.004]]),
        "mirror": np.array([[0.2, 0.003]]),
        "head": np.array([[0.1, 0.003]]),
    }
    deep = np.full((1, 2), 95.0)
    head = np.full((1, 2), 99.0)
    wte, fb = ren.recompose(dem, fac, w, deep, head, d=3.0, mirror_on=True)
    # cell 0: survivors renormalize by (1 - 0.4)
    num0 = 0.3 * 95.0 + 0.2 * 97.0 + 0.1 * 99.0
    np.testing.assert_allclose(wte[0, 0], num0 / 0.6)
    # renormalized mixture stays convex: within survivor min/max
    assert 95.0 <= wte[0, 0] <= 99.0
    # cell 1: 1 - w_fac = 0.01 < guard -> head fallback
    assert fb[0, 1] and not fb[0, 0]
    np.testing.assert_allclose(wte[0, 1], 99.0)


def test_recompose_three_expert_arm():
    dem = np.full((1, 1), 50.0)
    fac = np.array([[5.0]])
    w = {
        "fac": np.array([[0.5]]),
        "deep": np.array([[0.25]]),
        "head": np.array([[0.25]]),
    }
    wte, _ = ren.recompose(
        dem, fac, w, np.array([[40.0]]), np.array([[48.0]]), d=3.0, mirror_on=False
    )
    np.testing.assert_allclose(wte, 0.5 * 45.0 + 0.25 * 40.0 + 0.25 * 48.0)


# ---------------------------------------------------------------------------
# master_window (renderer)
# ---------------------------------------------------------------------------
def test_master_window_nests_coarse(tmp_path):
    t100, w100, h100 = inf.snapped_window(
        (-1200050.0, 1400020.0, -1199210.0, 1400900.0), 100.0
    )
    p = tmp_path / "gnn_wte_100m.tif"
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        dtype="float32",
        count=1,
        height=h100,
        width=w100,
        crs="EPSG:5070",
        transform=t100,
        nodata=-9999.0,
    ) as dst:
        dst.write(np.zeros((1, h100, w100), "float32"))
    t10, w10, h10 = ren.master_window(p)
    assert t10.c == t100.c and t10.f == t100.f
    assert w10 == w100 * 10 and h10 == h100 * 10
    assert t10.a == 10.0


def test_master_window_rejects_off_origin(tmp_path):
    p = tmp_path / "bad.tif"
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        dtype="float32",
        count=1,
        height=2,
        width=2,
        crs="EPSG:5070",
        transform=from_origin(inf.X0 + 37.0, inf.Y0, 100.0, 100.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(np.zeros((1, 2, 2), "float32"))
    with pytest.raises(SystemExit, match="canonical origin"):
        ren.master_window(p)


# ---------------------------------------------------------------------------
# manifest guard
# ---------------------------------------------------------------------------
def test_load_models_rejects_non_gate_arm(tmp_path):
    mdir = tmp_path / "models"
    mdir.mkdir()
    man = {"flags": {"prior_gate": True, "fac_skip": True}, "folds": []}
    (mdir / "inference_manifest.json").write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="prior-gate arms only"):
        inf.load_models(tmp_path)


# ---------------------------------------------------------------------------
# water pseudo-row pool discipline
# ---------------------------------------------------------------------------
def test_well_pool_drops_water_pseudo_rows():
    qn = pd.DataFrame(
        {
            "canonical_id": ["w1", "water_0000001", "w2"],
            "is_water_pseudo": [False, True, False],
        }
    )
    out = inf.well_pool(qn)
    assert list(out["canonical_id"]) == ["w1", "w2"]
    assert list(out.index) == [0, 1]


def test_well_pool_noop_without_column():
    qn = pd.DataFrame({"canonical_id": ["w1", "w2"]})
    out = inf.well_pool(qn)
    assert out is qn


# ---------------------------------------------------------------------------
# leak gate: enforced for plain arms, informational for source-assimilation arms
# ---------------------------------------------------------------------------
def _leak_panel_inputs():
    """A map pinned exactly to obs while the archived OOF sits 8 m away.

    Both leak signals fire: map MAD 0.00 m vs OOF MAD 8.00 m (ratio 0 < 0.5) and
    median |map - oof| 8.00 m > the 2.0 m tracking tolerance.
    """
    transform = from_origin(inf.X0, inf.Y0, inf.RES, inf.RES)
    rows, cols = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    obs = 10.0 + np.arange(len(rows), dtype="float64")
    grid = np.full((5, 5), np.nan)
    grid[rows, cols] = obs
    oof = pd.DataFrame(
        {
            "x5070": inf.X0 + (cols + 0.5) * inf.RES,
            "y5070": inf.Y0 - (rows + 0.5) * inf.RES,
            "obs_dtw_m": obs,
            "gnn_dtw_m": obs + 8.0,
        }
    )
    return grid, transform, oof


def test_leak_gate_fails_on_leak_signature():
    grid, transform, oof = _leak_panel_inputs()
    with pytest.raises(SystemExit, match="LEAK GATE FAILED"):
        inf.leak_gate("00000000", grid, transform, oof, 0.5, 5, 2.0)


def test_leak_gate_informational_for_source_arm():
    grid, transform, oof = _leak_panel_inputs()
    panel = inf.leak_gate(
        "00000000", grid, transform, oof, 0.5, 5, 2.0, informational=True
    )
    assert panel["status"] == "informational_source_arm"
    assert panel["n_wells"] == 25
    assert panel["mad_map_vs_obs_m"] == 0.0
    assert panel["mad_oof_vs_obs_m"] == 8.0
    assert panel["median_abs_map_minus_oof_m"] == 8.0


# ---------------------------------------------------------------------------
# checkpoint introspection (the manifest records neither MAE nor the src slot)
# ---------------------------------------------------------------------------
def test_ckpt_introspection_plain_arm():
    ck = {"state_dict": {"query_enc.0.weight": torch.zeros(48, 46)}}
    assert inf.ckpt_extensions(ck) == (None, False)
    assert inf.ckpt_f_src(ck, 46) is None
    assert inf.ckpt_f_srcedge(ck) is None


def test_ckpt_introspection_source_edge_writeback_arm():
    ck = {
        "state_dict": {
            "query_enc.0.weight": torch.zeros(48, 46),
            "writeback_conv.gate_mlp.0.weight": torch.zeros(48, 96),
            "source_enc.0.weight": torch.zeros(48, 1),
            # (hidden, in_src + in_dst + f_srcedge) = (48, 48 + 48 + 4)
            "source_read.score_mlp.0.weight": torch.zeros(48, 100),
        }
    }
    assert inf.ckpt_extensions(ck) == (None, True)
    assert inf.ckpt_f_src(ck, 46) is None
    assert inf.ckpt_f_srcedge(ck) == 4


def test_ckpt_introspection_gated_source_read_and_mae():
    ck = {
        "state_dict": {
            "query_enc.0.weight": torch.zeros(48, 48),  # +2 src-obs block
            "mae_enc.0.weight": torch.zeros(48, 64),
            "source_enc.0.weight": torch.zeros(48, 1),
            "source_read.gate_mlp.0.weight": torch.zeros(48, 100),
        }
    }
    assert inf.ckpt_extensions(ck) == (64, False)
    assert inf.ckpt_f_src(ck, 46) == 2
    assert inf.ckpt_f_srcedge(ck) == 4
