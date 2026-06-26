"""Unit tests for utils/build_regional_aquifer_graph.py.

Covers the graph-construction invariants the trainer relies on (dense node ids,
adjacency gating, barrier separation, stable harmonic-K, query attachment ranks)
and the Phase-1 leakage guard (no target/benchmark column on aquifer features).
All pure-function (no rasterio / no GPU), so they run in the standard test pass.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MOD = Path(__file__).resolve().parents[1] / "utils" / "build_regional_aquifer_graph.py"
_spec = importlib.util.spec_from_file_location("build_regional_aquifer_graph", _MOD)
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)


def _grid_nodes(codes, res=1000.0, logk=-13.0):
    """3x3 grid of nodes with given principal-aquifer codes (0 = absent -> dropped)."""
    codes = np.asarray(codes)
    rows = []
    for i in range(3):
        for j in range(3):
            c = int(codes[i, j])
            rows.append(
                {
                    "x5070": (j + 0.5) * res,
                    "y5070": -(i + 0.5) * res,
                    "grid_i": i,
                    "grid_j": j,
                    "principal_aquifer_code": c,
                    "aquifer_rocktype_code": 100,
                    "logk_glhymps": logk,
                }
            )
    df = pd.DataFrame(rows)
    df = df[df["principal_aquifer_code"] >= 0].reset_index(drop=True)
    df.insert(0, "aquifer_node_idx", np.arange(len(df), dtype="int64"))
    return df


def test_assemble_nodes_dense_idx_and_drop_all_nodata():
    # 3x3 centres; the centre cell (idx 4) is all-nodata -> dropped; ids stay dense.
    xc = np.arange(9, dtype="float64")
    yc = np.arange(9, dtype="float64")
    gi = np.repeat([0, 1, 2], 3).astype("int32")
    gj = np.tile([0, 1, 2], 3).astype("int32")
    aq = np.array([101, 101, 101, 101, 0, 101, 101, 101, 101])
    logk = np.array([-13.0] * 9)
    logk[4] = np.nan  # centre has no continuous signal either
    samples = {
        "principal_aquifer_code": aq.astype("float64"),
        "aquifer_rocktype_code": np.zeros(9),  # rocktype absent everywhere
        "logk_glhymps": logk,
    }
    nodes = ba.assemble_nodes(samples, xc, yc, gi, gj)
    assert len(nodes) == 8  # centre dropped
    assert (nodes["aquifer_node_idx"].to_numpy() == np.arange(8)).all()
    assert nodes["principal_aquifer_present"].sum() == 8


def test_grid_edges_respect_principal_aquifer_boundary():
    # left column class 101, right two columns class 202: no edge crosses the boundary
    # (both endpoints have a valid, differing code -> dropped; rocktype is uniform but
    # the same-aquifer rule wins when both codes are present).
    codes = np.array([[101, 202, 202], [101, 202, 202], [101, 202, 202]])
    nodes = _grid_nodes(codes)
    edges, comp = ba.build_grid_edges(nodes, neighborhood=8)
    cls = {
        int(r.aquifer_node_idx): int(r.principal_aquifer_code)
        for r in nodes.itertuples()
    }
    crossing = [
        (s, d)
        for s, d in zip(edges["src_aquifer_idx"], edges["dst_aquifer_idx"])
        if cls[s] != cls[d]
    ]
    assert crossing == []
    # two classes -> at least two components.
    assert len(np.unique(comp[nodes["aquifer_node_idx"].to_numpy()])) >= 2


def test_grid_edges_barrier_separates_components():
    # uniform class, but a full barrier row (i=1) between the top (i=0) and bottom
    # (i=2) rows: no edge may touch a barrier cell, so top and bottom disconnect.
    codes = np.full((3, 3), 101)
    nodes = _grid_nodes(codes)
    barrier = nodes["grid_i"].to_numpy() == 1
    edges, comp = ba.build_grid_edges(nodes, neighborhood=8, barrier=barrier)
    touches = [
        (s, d)
        for s, d in zip(edges["src_aquifer_idx"], edges["dst_aquifer_idx"])
        if barrier[s] or barrier[d]
    ]
    assert touches == []
    top = nodes.index[nodes["grid_i"] == 0].to_numpy()
    bot = nodes.index[nodes["grid_i"] == 2].to_numpy()
    assert comp[top[0]] != comp[bot[0]]  # barrier row severs the two halves


def test_harmonic_k_linear_and_log_agree():
    # known endpoints: K1=1e-12, K2=1e-14 -> H = 2/(1e12+1e14) ~= 1.9802e-14.
    h_lin = ba.harmonic_k_linear(np.array(1e-12), np.array(1e-14))
    assert np.isclose(h_lin, 2.0 / (1e12 + 1e14))
    logh = ba.log_harmonic_k(np.array(-12.0), np.array(-14.0))
    assert np.isclose(logh, np.log10(h_lin), atol=1e-9)
    # symmetric and stable for deep-negative logk (no overflow).
    assert np.isclose(
        ba.log_harmonic_k(np.array(-16.0), np.array(-12.0)),
        ba.log_harmonic_k(np.array(-12.0), np.array(-16.0)),
    )


def test_aquifer_query_edges_ranks_and_controlling():
    # 2 queries, 4 aquifer nodes on a line; each query gets k edges, ranks 0..k-1,
    # the nearest node is is_controlling=1.
    aq_xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    aq_code = np.array([101, 101, 101, 101])
    aq_comp = np.array([0, 0, 0, 0])
    q_xy = np.array([[1.0, 0.0], [29.0, 0.0]])
    q_code = np.array([101, 101])
    k = 3
    e = ba.build_aquifer_query_edges(
        aq_xy, aq_code, aq_comp, np.array([5, 9]), q_xy, q_code, k=k, max_dist_m=1e9
    )
    for qn in (5, 9):
        sub = e[e["query_node_idx"] == qn].sort_values("rank")
        assert len(sub) == k
        assert list(sub["rank"]) == list(range(k))
        assert sub["is_controlling"].sum() == 1.0
        ctrl = sub[sub["is_controlling"] == 1.0].iloc[0]
        assert ctrl["aq_query_dist_m"] == sub["aq_query_dist_m"].min()
    # query 0 nearest is aquifer node 0 (dist 1); query 1 nearest is node 3 (dist 1).
    c0 = e[(e["query_node_idx"] == 5) & (e["is_controlling"] == 1.0)].iloc[0]
    assert c0["aquifer_node_idx"] == 0


def test_leakage_guard_blocks_target_and_benchmark_cols():
    # the real feature set is clean.
    ba.assert_target_blind(ba.AQUIFER_FEATURE_COLS)
    # any of the forbidden label/benchmark columns must raise.
    for bad in ("mean_dtw", "wte_obs_m", "obs_wte_m", "janssen", "ma", "regional_deep"):
        with pytest.raises(SystemExit):
            ba.assert_target_blind(ba.AQUIFER_FEATURE_COLS + [bad])
