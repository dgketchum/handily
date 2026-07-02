"""Unit tests for the deep-aquifer-datum helpers in build_conus_graph_inputs.py.

Covers the two properties the deep regional datum depends on:
  * deep_well_mask is LOCAL (per-HUC6 quantile), so a locally-deep well in a
    shallow regime is selected while a globally-deep-but-locally-shallow well is
    not -- the whole point of not using a single global cut. Sparse HUC6s fall
    back to the HUC4 threshold.
  * crossfit_deep_idw is leak-free (a held-out fold is predicted only from deep
    wells in OTHER folds), predicts for ALL wells, and guards thin deep folds
    (kk = min(k, n_deep_train)).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MOD = Path(__file__).resolve().parents[1] / "utils" / "build_conus_graph_inputs.py"
_spec = importlib.util.spec_from_file_location("build_conus_graph_inputs", _MOD)
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)


def test_deep_well_mask_is_local_not_global():
    # HUC6 100100 (HUC4 1001): shallow regime, dtw 1..8 -> q0.75=6.25 -> {7,8}.
    # HUC6 100200 (HUC4 1002): deep regime, dtw 100..130 -> q0.75=122.5 -> {130}.
    wells = pd.DataFrame(
        {
            "huc8": ["10010000"] * 8 + ["10020000"] * 4,
            "mean_dtw": [1, 2, 3, 4, 5, 6, 7, 8, 100, 110, 120, 130],
        }
    )
    mask = bc.deep_well_mask(wells, quantile=0.75, unit="huc6", min_per_unit=2)
    # locally-deep shallow-regime wells ARE selected (dtw 7,8)
    assert mask[6] and mask[7]
    # a globally-deep but locally-shallow well is NOT selected (dtw 100 < 122.5)
    assert not mask[8]
    # only the locally-deepest of the deep regime is selected (dtw 130)
    assert mask[11] and not mask[9] and not mask[10]


def test_deep_well_mask_sparse_huc6_falls_back_to_huc4():
    # HUC4 1001: dense HUC6 100101 (40 wells) + sparse HUC6 100100 (3 wells).
    # min_per_unit=30 -> sparse 100100 uses the HUC4 quartile, not its own.
    dense_dtw = list(range(1, 41))  # HUC6 100101
    sparse_dtw = [100, 100, 100]  # HUC6 100100 (above the HUC4 q0.75)
    wells = pd.DataFrame(
        {
            "huc8": ["10010100"] * 40 + ["10010000"] * 3,
            "mean_dtw": dense_dtw + sparse_dtw,
        }
    )
    mask = bc.deep_well_mask(wells, quantile=0.75, unit="huc6", min_per_unit=30)
    # HUC4 1001 q0.75 over all 43 wells is well below 100, so the sparse trio
    # (fallback to HUC4 threshold) are all deep.
    assert mask[40] and mask[41] and mask[42]


def test_crossfit_deep_idw_leak_free_and_predicts_all():
    # Deep wells on a line, one per fold; a query sits on top of the same-fold
    # deep well. Leak-free => its prediction comes from the NEXT-nearest deep
    # well in another fold, not its own.
    q = [0.1, 0.0]
    d = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
    xy_all = np.array([q] + d, dtype="float64")
    fold_all = np.array([0, 0, 1, 2, 3])
    xy_deep = np.array(d, dtype="float64")
    dtw_deep = np.array([100.0, 200.0, 300.0, 400.0])
    fold_deep = np.array([0, 1, 2, 3])

    pred = bc.crossfit_deep_idw(
        xy_all, xy_deep, dtw_deep, fold_all, fold_deep, k=2, power=6.0
    )
    assert np.isfinite(pred).all()  # predicts for ALL wells
    # query in fold 0: excludes the same-fold deep well (dtw 100), so the
    # prediction is pulled toward the next-nearest other-fold well (dtw 200).
    assert abs(pred[0] - 200.0) < abs(pred[0] - 100.0)
    assert 200.0 <= pred[0] < 300.0


def test_crossfit_deep_idw_thin_fold_guard():
    # k larger than the deep training pool per held-out fold -> kk clamps, no crash.
    xy_all = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype="float64")
    fold_all = np.array([0, 1, 2, 3])
    xy_deep = xy_all.copy()
    dtw_deep = np.array([10.0, 20.0, 30.0, 40.0])
    fold_deep = fold_all.copy()
    pred = bc.crossfit_deep_idw(
        xy_all, xy_deep, dtw_deep, fold_all, fold_deep, k=10, power=2.0
    )
    assert np.isfinite(pred).all()


# ---------------------------------------------------------------------------
# downstream_datum: the shared walk primitive (items 1/2/5). Toy basin "A":
# an order-3 mainstem chain 0->1->2, two tributaries joining it (3->4->1 and a
# braided 5 with edges 5->6 and 5->1), a small 6->2, and a disconnected fragment
# 7->8 whose local max order (1) never reaches the basin band -> datum_missing.
# ---------------------------------------------------------------------------
def _toy_reach_graph():
    lengths = [1000, 1000, 1000, 500, 500, 800, 300, 400, 400]  # reaches 0..8
    orders = [3, 3, 3, 0, 1, 1, 0, 0, 1]
    elevs = [100, 90, 80, 120, 110, 130, 125, 200, 190]
    totda = [100, 200, 300, 5, 10, 8, 2, 3, 6]
    rn = pd.DataFrame(
        {
            "reach_node_idx": np.arange(9, dtype="int64"),
            "basin": ["A"] * 9,
            "streamorde": np.array(orders, "float64"),
            "reach_elev_m": np.array(elevs, "float64"),
            "totdasqkm": np.array(totda, "float64"),
            "log1p_length_m": np.log1p(np.array(lengths, "float64")),
        }
    )
    # direction==1 downstream edges + one reverse (-1) row to prove it is filtered.
    src = [0, 1, 3, 4, 5, 5, 6, 7, 1]
    dst = [1, 2, 4, 1, 6, 1, 2, 8, 0]
    direction = [1.0] * 8 + [-1.0]
    ce = pd.DataFrame(
        {
            "src_reach_idx": np.array(src, "int64"),
            "dst_reach_idx": np.array(dst, "int64"),
            "direction": np.array(direction, "float64"),
        }
    )
    return rn, ce


def test_downstream_datum_order_band_walk_and_self():
    rn, ce = _toy_reach_graph()
    dd = bc.downstream_datum(rn, ce, order_band=1)  # datum = order >= 3-1 = 2
    idx = dd["datum_reach_idx"].to_numpy()
    # order-3 mainstem reaches are their own datum; tributaries walk down to it.
    assert list(idx) == [0, 1, 2, 1, 1, 1, 2, -1, -1]
    assert list(dd["datum_is_self"]) == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert list(dd["datum_missing"]) == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    # distances are the SUM of the traversed reaches' lengths (m); expm1(log1p())
    # round-trips to ~1e-12, so compare approximately.
    dist = dd["datum_dist_m"].to_numpy()
    assert dist[3] == pytest.approx(1000.0)  # 3->4->1 = 500+500
    assert dist[4] == pytest.approx(500.0)
    assert dist[6] == pytest.approx(300.0)
    assert dist[0] == 0.0  # self-datum: exact, no walk
    assert np.isnan(dist[7])  # fragment: no datum
    assert list(dd["datum_n_hops"].to_numpy()[[0, 4, 3]]) == [0, 1, 2]
    # datum elevation/order come from the landed datum reach.
    assert dd["datum_elev_m"].to_numpy()[3] == 90.0  # reach 1 elev
    assert dd["datum_order"].to_numpy()[6] == 3.0  # reach 2 order
    assert (dd["basin_max_order"].to_numpy() == 3.0).all()


def test_downstream_datum_braid_takes_max_drainage_branch():
    rn, ce = _toy_reach_graph()
    dd = bc.downstream_datum(rn, ce, order_band=1)
    # reach 5 has edges 5->6 (totda 2) and 5->1 (totda 200): the braid must resolve to
    # the max-drainage branch (reach 1), NOT the small 5->6 tributary.
    assert dd["datum_reach_idx"].to_numpy()[5] == 1
    assert dd["datum_dist_m"].to_numpy()[5] == pytest.approx(800.0)  # reach-5 length
    assert dd["datum_n_hops"].to_numpy()[5] == 1


def test_downstream_datum_stop_mask_downstream_wet_distance():
    # item 5 generalization: stop at the nearest DOWNSTREAM wet reach (here reach 2).
    rn, ce = _toy_reach_graph()
    wet = np.zeros(9, bool)
    wet[2] = True
    dd = bc.downstream_datum(rn, ce, stop_mask=wet)
    idx = dd["datum_reach_idx"].to_numpy()
    dist = dd["datum_dist_m"].to_numpy()
    assert idx[2] == 2 and dist[2] == 0.0  # wet reach is its own datum
    assert idx[1] == 2 and dist[1] == pytest.approx(1000.0)  # 1->2
    assert idx[0] == 2 and dist[0] == pytest.approx(2000.0)
    assert dd["datum_n_hops"].to_numpy()[0] == 2
    assert idx[6] == 2 and dist[6] == pytest.approx(300.0)
    # the disconnected fragment has no downstream wet reach.
    assert dd["datum_missing"].to_numpy()[7] and dd["datum_missing"].to_numpy()[8]


def test_downstream_datum_requires_contiguous_reach_idx():
    rn, ce = _toy_reach_graph()
    rn.loc[0, "reach_node_idx"] = 99  # break the 0..n-1 invariant
    with pytest.raises(SystemExit):
        bc.downstream_datum(rn, ce)


def test_project_datum_to_queries_rank0_attachment():
    # three wells attach (rank 0) to reaches 3, 6, and the fragment reach 7; well 0 also
    # has a rank-1 edge to reach 4 that must be ignored.
    rn, ce = _toy_reach_graph()
    dd = bc.downstream_datum(rn, ce, order_band=1)
    lat = pd.DataFrame(
        {
            "query_node_idx": [0, 0, 1, 2],
            "reach_node_idx": [3, 4, 6, 7],
            "lateral_dist_m": [50.0, 999.0, 20.0, 10.0],
            "rank": [0, 1, 0, 0],
        }
    )
    well_surf = np.array([200.0, 150.0, 300.0])
    proj = bc.project_datum_to_queries(dd, lat, rn, well_surf)

    assert list(proj["reach0"]) == [3, 6, 7]  # rank-0 attachment, not rank-1
    assert list(proj["lat0"]) == [50.0, 20.0, 10.0]
    assert list(proj["datum_reach_idx"]) == [
        1,
        2,
        -1,
    ]  # walk lands on reach 1 / 2 / none
    assert list(proj["reach0_strah"]) == [0.0, 0.0, 0.0]
    assert list(proj["datum_missing"]) == [0.0, 0.0, 1.0]
    np.testing.assert_allclose(proj["datum_elev"], [90.0, 80.0, np.nan])
    np.testing.assert_allclose(proj["datum_dist"], [1000.0, 300.0, np.nan])
    np.testing.assert_allclose(proj["datum_order"], [3.0, 3.0, np.nan])
    np.testing.assert_allclose(proj["datum_da"], [200.0, 300.0, np.nan])

    # the item-1 query columns derived from the projection.
    drop = well_surf - proj["datum_elev"]
    np.testing.assert_allclose(drop, [110.0, 70.0, np.nan])
    log_dist = np.log1p(proj["lat0"] + proj["datum_dist"])
    np.testing.assert_allclose(log_dist, [np.log1p(1050.0), np.log1p(320.0), np.nan])
    order_rel = proj["datum_order"] - proj["reach0_strah"]
    np.testing.assert_allclose(order_rel, [3.0, 3.0, np.nan])


# ---------------------------------------------------------------------------
# wet_propagation_reach_features (item 5): propagate a per-reach wet flag over
# the Phase-0b toy basin. Distances are network-metric, so the UNDIRECTED
# net-distance can beat the DOWNSTREAM-only distance (reach 5 reaches wet reach
# 2 via 5-6-2 = 1100 m, not the downstream 5->1->2 = 1800 m).
# ---------------------------------------------------------------------------
def test_wet_propagation_distances_wet_at_outlet():
    rn, ce = _toy_reach_graph()
    wet = np.zeros(9, bool)
    wet[2] = True  # the order-3 mainstem outlet is the only wet reach
    wf = bc.wet_propagation_reach_features(rn, ce, wet)

    # the helper's columns ARE the manifest contract (trainer reads these generically).
    assert list(wf.columns) == bc.WET_PROP_REACH_FEATURE_COLS

    np.testing.assert_allclose(wf["wet_reach"].to_numpy(), [0, 0, 1, 0, 0, 0, 0, 0, 0])
    # UNDIRECTED net-distance to reach 2 (edge weight = src reach length).
    net = np.expm1(wf["log1p_net_dist_wet_m"].to_numpy())
    np.testing.assert_allclose(
        net,
        [2000.0, 1000.0, 0.0, 2000.0, 1500.0, 1100.0, 300.0, np.nan, np.nan],
    )
    # DOWNSTREAM-only distance (the 0b walk with stop_mask=wet): reach 5 = 1800.
    ds = np.expm1(wf["log1p_ds_dist_wet_m"].to_numpy())
    np.testing.assert_allclose(
        ds,
        [2000.0, 1000.0, 0.0, 2000.0, 1500.0, 1800.0, 300.0, np.nan, np.nan],
    )
    # the fragment 7->8 is a component with no wet reach.
    np.testing.assert_allclose(
        wf["no_wet_in_component"].to_numpy(), [0, 0, 0, 0, 0, 0, 0, 1, 1]
    )


def test_wet_propagation_upstream_fraction_headwater_wet():
    rn, ce = _toy_reach_graph()
    wet = np.zeros(9, bool)
    wet[3] = True  # a headwater (len 500) that flows 3->4->1->2
    wf = bc.wet_propagation_reach_features(rn, ce, wet)
    frac = wf["upstream_wet_fraction"].to_numpy()
    # length-weighted wet/total over the inclusive upstream set of each reach.
    assert frac[3] == pytest.approx(1.0)  # headwater: only itself upstream, wet
    assert frac[4] == pytest.approx(500.0 / 1000.0)  # {4,3}: 500 wet / (500+500)
    assert frac[1] == pytest.approx(500.0 / 3800.0)  # {1,0,3,4,5}
    assert frac[2] == pytest.approx(500.0 / 5100.0)  # {2,1,0,3,4,5,6}
    assert frac[0] == pytest.approx(0.0)  # dry upstream set {0}
    assert frac[5] == pytest.approx(0.0)
    assert frac[7] == pytest.approx(0.0)  # dry fragment


def test_wet_propagation_no_wet_reaches_all_nan():
    rn, ce = _toy_reach_graph()
    wet = np.zeros(9, bool)  # no wet reach anywhere
    wf = bc.wet_propagation_reach_features(rn, ce, wet)
    assert np.isnan(wf["log1p_net_dist_wet_m"].to_numpy()).all()
    assert np.isnan(wf["log1p_ds_dist_wet_m"].to_numpy()).all()
    assert (wf["no_wet_in_component"].to_numpy() == 1.0).all()
    assert (wf["upstream_wet_fraction"].to_numpy() == 0.0).all()  # dry, not NaN


def test_wet_propagation_length_mismatch_raises():
    rn, ce = _toy_reach_graph()
    with pytest.raises(SystemExit):
        bc.wet_propagation_reach_features(rn, ce, np.zeros(3, bool))


# ---------------------------------------------------------------------------
# Phase 6A refactors: build_down_ptr (max-drainage braid resolution) +
# accumulate_upstream (length-weighted upstream inclusive mean via Kahn).
# ---------------------------------------------------------------------------
def test_build_down_ptr_braid_takes_max_drainage_and_counts():
    rn, ce = _toy_reach_graph()
    down_ptr, n_braids = bc.build_down_ptr(rn, ce)
    # reach 5 braids (5->6 totda 2 vs 5->1 totda 200): keep the max-drainage parent (1).
    assert list(down_ptr) == [1, 2, -1, 4, 1, 1, 2, 8, -1]
    assert n_braids == 1  # only reach 5 has >1 downstream edge


def test_accumulate_upstream_confluence_headwater_and_nan_exclusion():
    rn, ce = _toy_reach_graph()
    down_ptr, _ = bc.build_down_ptr(rn, ce)
    seg_len = np.expm1(rn["log1p_length_m"].to_numpy("float64"))
    v = (np.arange(9, dtype="float64") + 1.0) * 10.0  # 10,20,...,90
    up = bc.accumulate_upstream(v, seg_len, down_ptr)
    # reach 1 inclusive upstream {0,1,3,4,5}: length-weighted mean.
    assert up[1] == pytest.approx(123000.0 / 3800.0)
    # reach 2 inclusive upstream {0,1,2,3,4,5,6}.
    assert up[2] == pytest.approx(174000.0 / 5100.0)
    assert up[4] == pytest.approx(45.0)  # {3,4}
    assert up[0] == pytest.approx(10.0)  # headwater == own value
    assert up[8] == pytest.approx(85.0)  # fragment {7,8}

    # NaN local is excluded from BOTH numerator and denominator; an all-NaN upstream -> NaN.
    v2 = v.copy()
    v2[3] = np.nan
    up2 = bc.accumulate_upstream(v2, seg_len, down_ptr)
    assert up2[1] == pytest.approx(
        103000.0 / 3300.0
    )  # reach 3 dropped from {0,1,3,4,5}
    assert up2[4] == pytest.approx(50.0)  # {3,4} with 3 dropped -> just reach 4
    assert np.isnan(up2[3])  # sole member is NaN


def test_accumulate_upstream_matches_wet_fraction():
    # the length-weighted upstream mean of a 0/1 wet flag IS the wet channel-length fraction.
    rn, ce = _toy_reach_graph()
    down_ptr, _ = bc.build_down_ptr(rn, ce)
    seg_len = np.expm1(rn["log1p_length_m"].to_numpy("float64"))
    wet = np.zeros(9, "float64")
    wet[3] = 1.0
    frac = bc.accumulate_upstream(wet, seg_len, down_ptr)
    wf = bc.wet_propagation_reach_features(rn, ce, wet.astype(bool))
    np.testing.assert_allclose(frac, wf["upstream_wet_fraction"].to_numpy())


def test_accumulate_upstream_2d_columns():
    rn, ce = _toy_reach_graph()
    down_ptr, _ = bc.build_down_ptr(rn, ce)
    seg_len = np.expm1(rn["log1p_length_m"].to_numpy("float64"))
    v = (np.arange(9, dtype="float64") + 1.0) * 10.0
    stack = np.column_stack([v, 2.0 * v])
    up = bc.accumulate_upstream(stack, seg_len, down_ptr)
    assert up.shape == (9, 2)
    np.testing.assert_allclose(up[:, 1], 2.0 * up[:, 0])  # linear in the value column


def test_accumulate_upstream_cycle_guard():
    # a 2-node down-pointer cycle (0<->1) has no headwater -> the Kahn guard must fire.
    with pytest.raises(RuntimeError):
        bc.accumulate_upstream(
            np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([1, 0], dtype="int64")
        )


# ---------------------------------------------------------------------------
# upstream_head (6B up_head site): mirror of downstream_datum walking UP the
# max-drainage PARENT pointer to a terminal channel head.
# ---------------------------------------------------------------------------
def test_upstream_head_max_drainage_parent_and_fragment():
    rn, ce = _toy_reach_graph()
    hd = bc.upstream_head(rn, ce)
    # reach 1's parents are 0 (totda 100), 4 (10), 5 (8): the head walk must follow the
    # MAX-drainage parent (0), so both reach 1 and reach 2 head at reach 0. reach 6's only
    # parent is the headwater reach 5 (edge 5->6), so it heads at 5 -- reach 7 is a
    # disconnected fragment (7->8) and never enters reach 6's upstream walk.
    assert list(hd["head_reach_idx"].to_numpy()) == [0, 0, 0, 3, 3, 5, 5, 7, 7]
    np.testing.assert_allclose(
        hd["head_dist_m"].to_numpy(), [0, 1000, 2000, 0, 500, 0, 300, 0, 400]
    )
    assert list(hd["head_n_hops"].to_numpy()) == [0, 1, 2, 0, 1, 0, 1, 0, 1]
    # head attrs come from the landed head reach.
    assert hd["head_elev_m"].to_numpy()[2] == 100.0  # reach 0 elev
    assert hd["head_da_km2"].to_numpy()[2] == 100.0  # reach 0 totda
    assert hd["head_reach_idx"].to_numpy()[8] == 7  # fragment head


def test_upstream_head_requires_contiguous_reach_idx():
    rn, ce = _toy_reach_graph()
    rn.loc[0, "reach_node_idx"] = 99
    with pytest.raises(SystemExit):
        bc.upstream_head(rn, ce)


# ---------------------------------------------------------------------------
# serving_wet_source (6B wet site): nearest wet reach node identity + net dist,
# checked against a brute-force per-source Dijkstra on the undirected toy graph.
# ---------------------------------------------------------------------------
def _brute_wet(rn, ce, wet_idx):
    """Per-node (min-distance, argmin-source) over the undirected length-weighted graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    n = len(rn)
    seg = np.expm1(rn["log1p_length_m"].to_numpy("float64"))
    down = ce[ce["direction"] == 1]
    s = down["src_reach_idx"].to_numpy("int64")
    d = down["dst_reach_idx"].to_numpy("int64")
    g = csr_matrix((seg[s], (s, d)), shape=(n, n))
    dists = np.vstack([dijkstra(g, directed=False, indices=int(w)) for w in wet_idx])
    return dists  # (len(wet_idx), n)


def test_serving_wet_source_single_matches_wet_propagation():
    rn, ce = _toy_reach_graph()
    wet = np.zeros(9, bool)
    wet[2] = True
    serving, net = bc.serving_wet_source(rn, ce, wet)
    # only reach 2 is wet: every reach in its component serves reach 2 at the net distance.
    assert list(serving) == [2, 2, 2, 2, 2, 2, 2, -1, -1]
    wf = bc.wet_propagation_reach_features(rn, ce, wet)
    np.testing.assert_allclose(net, np.expm1(wf["log1p_net_dist_wet_m"].to_numpy()))
    assert np.isnan(net[7]) and np.isnan(net[8])  # dry fragment component


def test_serving_wet_source_multi_picks_nearest():
    rn, ce = _toy_reach_graph()
    wet = np.zeros(9, bool)
    wet[[3, 6]] = True
    serving, net = bc.serving_wet_source(rn, ce, wet)
    dists = _brute_wet(rn, ce, [3, 6])
    expected = dists.min(axis=0)
    for i in range(7):  # reaches in the wet component
        assert net[i] == pytest.approx(expected[i])
        # the served reach is a wet reach that achieves the min distance.
        assert serving[i] in (3, 6)
        served_row = 0 if serving[i] == 3 else 1
        assert dists[served_row, i] == pytest.approx(net[i])
    assert serving[7] == -1 and np.isnan(net[7])  # dry fragment


def test_serving_wet_source_no_wet_all_missing():
    rn, ce = _toy_reach_graph()
    serving, net = bc.serving_wet_source(rn, ce, np.zeros(9, bool))
    assert (serving == -1).all()
    assert np.isnan(net).all()


# ---------------------------------------------------------------------------
# 6A.1 sample_reach_covariates: column assembly + arithmetic (perm /100 decode,
# ndvi_amp = jja - djf, GSW occurrence pass-through). sample_coarse is stubbed so
# the test is about the sampler's logic, not raster IO.
# ---------------------------------------------------------------------------
def test_sample_reach_covariates_arithmetic(monkeypatch):
    vals = {
        bc.TERRAIN_MULTISCALE_RASTERS["tpi_2km"]: 1.0,
        bc.TERRAIN_MULTISCALE_RASTERS["tpi_10km"]: 2.0,
        bc.TERRAIN_MULTISCALE_RASTERS["twi_2km"]: 3.0,
        bc.ETRM_RECHARGE: 4.0,
        bc.ETRM_ETA: 5.0,
        bc.ETRM_RUNOFF: 6.0,
        bc.GRIDMET_AI: 7.0,
        bc.GRIDMET_P: 8.0,
        bc.PERM_LOGK: 1500.0,  # log10(k)x100 -> /100 = 15.0
        bc.SED_THICKNESS: 9.0,
        bc.NDVI_JJA: 0.8,
        bc.NDVI_DJF: 0.3,
    }
    monkeypatch.setattr(
        bc, "sample_coarse", lambda path, x, y: np.full(len(x), vals[path], "float64")
    )
    rcx = np.array([0.0, 100.0])  # finite EPSG:5070 rep-points
    rcy = np.array([0.0, 100.0])
    occ = np.array([50.0, np.nan])  # pre-sampled GSW occurrence (shared)
    cov = bc.sample_reach_covariates(rcx, rcy, occ)
    assert set(cov) == set(bc.REACH_COVARIATE_FEATURE_COLS)
    np.testing.assert_allclose(cov["r_perm_logk_m2"], [15.0, 15.0])  # /100 decode
    np.testing.assert_allclose(cov["r_ndvi_amp"], [0.5, 0.5])  # jja - djf
    np.testing.assert_allclose(cov["r_ndvi_jja"], [0.8, 0.8])
    np.testing.assert_allclose(cov["r_aridity_index"], [7.0, 7.0])  # gridMET (lon/lat)
    np.testing.assert_array_equal(cov["r_gsw_occ"], occ)  # GSW pass-through (incl NaN)
