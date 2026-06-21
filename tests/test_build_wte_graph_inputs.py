"""Unit tests for utils/build_wte_graph_inputs.py (pure functions).

Synthetic Y-network (reaches A,B -> C): verifies channel adjacency matches the
FAC topology with reverse edges, directional edge features, lateral edges connect
each query to its true nearest reach (rank 0 == controlling) with k respected and
exact distances, node indices stay in range, no query->query edges, and the
residual-WTE target equals obs_wte - regional_wte_oof.
"""

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point

_MOD = Path(__file__).resolve().parents[1] / "utils" / "build_wte_graph_inputs.py"
_spec = importlib.util.spec_from_file_location("build_wte_graph_inputs", _MOD)
bg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bg)

M = "idw_k32_p2"  # residual-method suffix the assembler auto-detects


def _reach_gdf():
    # A,B are headwater reaches whose down_node is the shared junction (0,50);
    # C's up_node is that junction -> downstream edges A->C, B->C.
    rows = [
        dict(
            stream_id=10,
            strahler=1,
            length_m=50.0,
            relief_m=10.0,
            up_elev_m=110.0,
            down_elev_m=100.0,
            drainage_km2=10.0,
            bed_elev_m=99.0,
            channel_head_m=101.0,
            hard_pin=0,
            up_node_x=0.0,
            up_node_y=100.0,
            down_node_x=0.0,
            down_node_y=50.0,
            geometry=LineString([(0, 100), (0, 50)]),
        ),
        dict(
            stream_id=11,
            strahler=1,
            length_m=70.7,
            relief_m=12.0,
            up_elev_m=112.0,
            down_elev_m=100.0,
            drainage_km2=8.0,
            bed_elev_m=99.0,
            channel_head_m=101.0,
            hard_pin=0,
            up_node_x=50.0,
            up_node_y=100.0,
            down_node_x=0.0,
            down_node_y=50.0,
            geometry=LineString([(50, 100), (0, 50)]),
        ),
        dict(
            stream_id=12,
            strahler=2,
            length_m=50.0,
            relief_m=10.0,
            up_elev_m=100.0,
            down_elev_m=90.0,
            drainage_km2=20.0,
            bed_elev_m=89.0,
            channel_head_m=91.0,
            hard_pin=0,
            up_node_x=0.0,
            up_node_y=50.0,
            down_node_x=0.0,
            down_node_y=0.0,
            geometry=LineString([(0, 50), (0, 0)]),
        ),
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=5070)


def _query_frames():
    pts = [Point(1, 75), Point(49, 75), Point(0, 25)]
    cids = ["q0", "q1", "q2"]
    cov = gpd.GeoDataFrame(
        {"canonical_id": cids, "dem_m": [105.0, 106.0, 95.0], "geometry": pts},
        geometry="geometry",
        crs=5070,
    )
    oof = gpd.GeoDataFrame(
        {
            "canonical_id": cids,
            "cv_fold": [0, 1, 2],
            "obs_wte_m": [100.0, 100.0, 100.0],
            "obs_dtw_m": [5.0, 6.0, 7.0],
            f"regional_wte_oof__{M}": [70.0, 80.0, 60.0],
            f"regional_dtw_oof__{M}": [35.0, 26.0, 35.0],
            f"residual_wte_obs__{M}": [30.0, 20.0, 40.0],
            f"residual_wte_hat__{M}": [0.0, 0.0, 0.0],
            "depth_bin": ["5-10", "5-10", "5-10"],
            "geometry": pts,
        },
        geometry="geometry",
        crs=5070,
    )
    labels = pd.DataFrame({"canonical_id": cids, "source": ["nwis", "other", "other"]})
    return cov, oof, labels


def test_channel_edges_match_topology_with_reverse():
    rn = bg.build_reach_nodes(_reach_gdf(), 20000)
    ce = bg.build_channel_edges(rn, node_precision=3)

    fwd = ce[ce.direction == 1.0]
    rev = ce[ce.direction == -1.0]
    assert set(map(tuple, fwd[["src_reach_idx", "dst_reach_idx"]].to_numpy())) == {
        (0, 2),
        (1, 2),
    }
    assert set(map(tuple, rev[["src_reach_idx", "dst_reach_idx"]].to_numpy())) == {
        (2, 0),
        (2, 1),
    }
    assert len(ce) == 4  # 2 downstream + 2 reverse, no self-loops

    e = fwd[(fwd.src_reach_idx == 0) & (fwd.dst_reach_idx == 2)].iloc[0]
    assert e.elev_drop_m == 100.0 - 90.0  # down_elev[A] - down_elev[C]
    assert e.strahler_change == 2 - 1
    assert e.log_drainage_ratio == np.log(20.0 / 10.0)

    r = rev[(rev.src_reach_idx == 2) & (rev.dst_reach_idx == 0)].iloc[0]
    assert r.elev_drop_m == 90.0 - 100.0
    assert r.strahler_change == 1 - 2
    assert r.log_drainage_ratio == np.log(10.0 / 20.0)


def test_channel_edge_indices_in_range():
    rn = bg.build_reach_nodes(_reach_gdf(), 20000)
    ce = bg.build_channel_edges(rn, node_precision=3)
    assert ce.src_reach_idx.between(0, len(rn) - 1).all()
    assert ce.dst_reach_idx.between(0, len(rn) - 1).all()
    assert (ce.src_reach_idx != ce.dst_reach_idx).all()


def test_lateral_edges_pick_true_nearest_reach():
    rn = bg.build_reach_nodes(_reach_gdf(), 20000)
    cov, oof, labels = _query_frames()
    qn, method = bg.build_query_nodes(cov, oof, labels)
    knn = 2
    le = bg.build_lateral_edges(qn, rn, knn=knn)

    assert method == M
    assert len(le) == knn * len(qn)

    reach_geoms = rn.geometry.to_numpy()
    qgeom = qn.geometry.to_numpy()
    for qi in range(len(qn)):
        sub = le[le.query_node_idx == qi].sort_values("rank")
        assert list(sub["rank"]) == list(range(knn))
        assert sub["is_controlling"].sum() == 1.0
        # rank-0 edge == brute-force nearest reach, exact distance
        brute = int(np.argmin(shapely.distance(qgeom[qi], reach_geoms)))
        top = sub.iloc[0]
        assert int(top.reach_node_idx) == brute
        assert top.lateral_dist_m == shapely.distance(qgeom[qi], reach_geoms[brute])
    # the three queries sit nearest A,B,C respectively
    rank0 = le[le["rank"] == 0].sort_values("query_node_idx")
    assert list(rank0.reach_node_idx) == [0, 1, 2]


def test_lateral_edges_are_query_to_reach_only():
    rn = bg.build_reach_nodes(_reach_gdf(), 20000)
    cov, oof, labels = _query_frames()
    qn, _ = bg.build_query_nodes(cov, oof, labels)
    le = bg.build_lateral_edges(qn, rn, knn=2)
    # dst side is always a valid reach node; src side a valid query node.
    assert le.reach_node_idx.between(0, len(rn) - 1).all()
    assert le.query_node_idx.between(0, len(qn) - 1).all()
    assert "height_above_bed_m" in le.columns


def test_target_is_obs_minus_regional_oof():
    cov, oof, labels = _query_frames()
    qn, _ = bg.build_query_nodes(cov, oof, labels)
    expect = qn["obs_wte_m"] - qn["regional_wte_oof_m"]
    assert np.allclose(qn["target_residual_wte_m"], expect)
    assert np.allclose(qn["target_residual_wte_m"], [30.0, 20.0, 40.0])
    # source rejoined for the non-NWIS split
    assert list(qn.sort_values("query_node_idx")["source"]) == [
        "nwis",
        "other",
        "other",
    ]


def test_network_distance_to_mainstem():
    rn = bg.build_reach_nodes(_reach_gdf(), 20000)
    ce = bg.build_channel_edges(rn, node_precision=3)
    # C (idx 2) is the only Strahler>=2 reach -> the mainstem source.
    net = bg.network_distance_to_order(rn, ce, min_strahler=2)
    assert net[2] == 0.0
    d_ac = ce[(ce.src_reach_idx == 0) & (ce.dst_reach_idx == 2)].centroid_dist_m.iloc[0]
    d_bc = ce[(ce.src_reach_idx == 1) & (ce.dst_reach_idx == 2)].centroid_dist_m.iloc[0]
    assert np.isclose(net[0], d_ac)
    assert np.isclose(net[1], d_bc)
    # no reach reaches Strahler>=3 -> all NaN (disconnected from any mainstem)
    assert np.all(np.isnan(bg.network_distance_to_order(rn, ce, min_strahler=3)))


def test_classify_fcode():
    codes = np.array([46006, 46003, 46007, 33600, 55800, 12345, np.nan])
    cls = bg.classify_fcode(codes)
    assert list(cls) == [
        "perennial",
        "intermittent",
        "ephemeral",
        "canal_ditch",
        "artificial_path",
        "other",
        "other",
    ]


def test_blocks_match_covariate_builder():
    x = np.array([0.0, 19999.0, 20001.0, -1.0])
    y = np.array([0.0, 0.0, 40000.0, 0.0])
    got = bg.assign_blocks(x, y, 20000)
    want = (
        (pd.Series(x) // 20000).astype("int64").astype(str)
        + "_"
        + (pd.Series(y) // 20000).astype("int64").astype(str)
    ).to_numpy()
    assert list(got) == list(want)
