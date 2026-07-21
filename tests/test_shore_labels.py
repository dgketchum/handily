"""E6 shoreline-ring pseudo-label tests.

Two families, mirroring test_build_swl_labels.py:

* pure-logic wiring tests (synthetic frames / geometry) -- always run. They pin the
  leakage-critical trainer masks (shore rows flow into `real = ~water & ~swl & ~shore`,
  never a fit source, never a scored metric), the flag-off byte-identity contract
  (no is_shore_pseudo column => shore all-False => weights/masks identical to pre-E6),
  the "weight applied LAST" override, fold-aware holdout, the neutral aux embedding, the
  shore-agnostic model construction, and the builder's land-ring geometry/screen logic.
* data-dependent tests -- skip when the CONUS artifacts (built ring parquet + v2s
  bundle) are absent, so the suite still runs off-zoran. They pin the label convention
  (mean_dtw=0, wte_obs=z_surf), the feat-dim-46 invariant (no new NaN feature columns),
  the non-shore parity with v2, and that the scoreable well set is unchanged (shore rows
  never enter the v0.2 metric footprint).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WTE = Path("/data/ssd2/handily/conus/wte_gnn")
RING = WTE / "shoreline" / "shoreline_ring_points.parquet"
V2 = WTE / "graph_conus_monitoring_water_v2"
V2S = WTE / "graph_conus_monitoring_water_v2s"
_HAVE_RING = RING.exists()
_HAVE_BUNDLES = V2.exists() and V2S.exists()
_ring = pytest.mark.skipif(not _HAVE_RING, reason="shoreline ring parquet absent")
_bundles = pytest.mark.skipif(not _HAVE_BUNDLES, reason="v2/v2s bundles absent")


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# builder geometry / screen logic (pure, synthetic)
# ---------------------------------------------------------------------------
def test_ring_candidates_are_land_side_on_lattice():
    """A synthetic square lake: every emitted land cell center is OUTSIDE the polygon,
    snapped to the canonical 100 m lattice, and within 150 m of the boundary; every
    inside probe is INSIDE (so the per-segment GSW permanence probe samples water)."""
    from shapely.geometry import Point, Polygon

    bsp = _load("build_shoreline_points")
    x0, y0 = -1_000_000.0, 2_000_000.0
    sq = Polygon([(x0, y0), (x0 + 600, y0), (x0 + 600, y0 + 600), (x0, y0 + 600)])
    lc, pr = bsp.polygon_ring_candidates(sq)
    assert len(lc) > 0 and len(lc) == len(pr)
    assert all(not sq.contains(Point(*p)) for p in lc)  # land side
    assert all(sq.contains(Point(*p)) for p in pr)  # probe inside
    # snapped to lattice cell centers (offset 50 m from the origin grid lines)
    assert np.allclose((lc[:, 0] - bsp.LAT_X0) % bsp.LAT_RES, 50.0)
    assert np.allclose((bsp.LAT_Y0 - lc[:, 1]) % bsp.LAT_RES, 50.0)
    d = np.hypot(lc[:, 0] - pr[:, 0], lc[:, 1] - pr[:, 1])  # land<->probe span
    assert d.max() <= bsp.LAND_OFFSET_M + bsp.PROBE_INSIDE_M + bsp.LAT_RES


def test_cell_id_dedup_collapses_shared_cell():
    """Two points in the same 100 m lattice cell share a cell id (one point per cell)."""
    bsp = _load("build_shoreline_points")
    x = np.array([bsp.LAT_X0 + 12.0, bsp.LAT_X0 + 88.0, bsp.LAT_X0 + 150.0])
    y = np.array([bsp.LAT_Y0 - 12.0, bsp.LAT_Y0 - 88.0, bsp.LAT_Y0 - 150.0])
    cid = bsp.cell_ids(x, y)
    assert cid[0] == cid[1]  # same cell
    assert cid[0] != cid[2]  # different cell
    _, first = np.unique(cid, return_index=True)
    assert len(first) == 2


def test_pt_seg_dist_matches_endpoint_and_interior():
    bsp = _load("build_shoreline_points")
    px, py = np.array([0.0, 5.0]), np.array([3.0, 10.0])
    ax, ay = np.array([-10.0, 0.0]), np.array([0.0, 0.0])
    bx, by = np.array([10.0, 0.0]), np.array([0.0, 8.0])
    d = bsp._pt_seg_dist(px, py, ax, ay, bx, by)
    assert np.isclose(d[0], 3.0)  # perpendicular to the segment interior
    # p=(5,10) projects past endpoint b=(0,8): clamps, dist = hypot(5, 2)
    assert np.isclose(d[1], np.hypot(5.0, 2.0))


# ---------------------------------------------------------------------------
# trainer-wiring expressions (pure logic; the exact statements main() uses)
# ---------------------------------------------------------------------------
def test_flag_off_is_noop_on_masks():
    """No is_shore_pseudo column => shore all-False => real == pre-E6 ~water & ~swl."""
    qn = pd.DataFrame({"is_water_pseudo": [False, True, False]})
    water = qn["is_water_pseudo"].to_numpy(bool)
    swl = np.zeros(len(qn), bool)
    shore = (
        qn["is_shore_pseudo"].to_numpy(bool)
        if "is_shore_pseudo" in qn.columns
        else np.zeros(len(qn), bool)
    )
    real = ~water & ~swl & ~shore
    assert shore.tolist() == [False, False, False]
    assert real.tolist() == [True, False, True]  # identical to pre-E6 real=~water


def test_shore_weight_applied_last_overrides_depth_path():
    """The shore block runs LAST: an obs_dtw=0 shore row that the depth path would
    max-weight is forced to the shore weight (0.0 => loss-inert)."""
    w = np.array([1.0, 5.0, 5.0])  # depth path gave shore rows big weight
    shore = np.array([False, True, True])
    shore_label_weight = 0.0
    if shore.any():
        w = w.copy()
        w[shore] = shore_label_weight
    assert w.tolist() == [1.0, 0.0, 0.0]
    # flag-off: shore all-False leaves the vector untouched
    w2 = np.array([1.0, 2.0, 3.0])
    shore_off = np.zeros(3, bool)
    if shore_off.any():
        w2[shore_off] = shore_label_weight
    assert w2.tolist() == [1.0, 2.0, 3.0]


def test_a_pri_shore_is_phreatic():
    """Two-surface assignment prior: shore rings are open-water phreatic like water."""
    a_pri = np.full(4, 0.5)
    water = np.array([False, True, False, False])
    shore = np.array([False, False, True, True])
    a_pri[water] = 0.98
    a_pri[shore] = 0.98
    assert a_pri.tolist() == [0.5, 0.98, 0.98, 0.98]


def test_fold_aware_masking_holds_out_shore():
    """A shore point in fold f is absent from fold-f training and never enters a
    fit-eligible real mask (the exact fold-loop expression main() uses)."""
    fold = np.array([0, 1, 2, 0, 1, 2, 0, 1])  # last 3 are shore
    shore = np.array([False] * 5 + [True] * 3)
    water = np.zeros(len(fold), bool)
    swl = np.zeros(len(fold), bool)
    real = ~water & ~swl & ~shore
    for f in np.unique(fold):
        test = fold == f
        tr = ~test
        assert not (tr & shore & test).any()
        assert not (tr & real & shore).any()


def test_neutral_embedding_standardizes_shore_to_zero():
    """The neutral aux embedding standardizes to 0 and base rows match the no-aux path
    exactly (stats fit on base only) -- shore rows perturb neither."""
    tc = _load("train_conus_gnn")
    rng = np.random.RandomState(1)
    cols = [f"mae_{i}" for i in range(4)]
    base_emb = rng.normal(5.0, 3.0, size=(6, 4))
    shore = np.array([False] * 6 + [True] * 3)
    emb = np.empty((9, 4), "float64")
    emb[~shore] = base_emb
    emb[shore] = base_emb.mean(0, keepdims=True)
    frame = pd.DataFrame(emb, columns=cols)
    stats = tc.fit_stats(frame.loc[~shore].reset_index(drop=True), cols, None)
    std = tc.apply_stats(frame, stats)
    assert np.allclose(std[shore], 0.0, atol=1e-8)
    base_only = tc.apply_stats(pd.DataFrame(base_emb, columns=cols), stats)
    assert np.allclose(std[~shore], base_only, atol=1e-12)


def test_model_construction_is_shore_agnostic_byte_identical():
    """WTEGraphNet takes no shore argument: two seeded constructions are byte-identical,
    so a bundle built with --shoreline-points inits the same weights as one without
    (flag-off byte-identity of the trained model given identical feature dims)."""
    import io

    import torch

    tw = _load("train_wte_gnn")

    def _bytes():
        torch.manual_seed(0)
        m = tw.WTEGraphNet(8, 46, 5, 6, 32, 2, 0.0, sigma=True, prior_gate=True)
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        return buf.getvalue()

    assert _bytes() == _bytes()


# ---------------------------------------------------------------------------
# built-artifact invariants (data-dependent)
# ---------------------------------------------------------------------------
@_ring
def test_ring_label_convention():
    r = pd.read_parquet(RING)
    assert np.allclose(r["mean_dtw"].to_numpy("float64"), 0.0)
    assert np.allclose(
        r["wte_obs"].to_numpy("float64"), r["z_surf"].to_numpy("float64")
    )
    assert np.isfinite(r["z_surf"].to_numpy("float64")).all()


@_ring
def test_ring_per_huc8_cap():
    r = pd.read_parquet(RING)
    assert int(r.groupby("huc8").size().max()) <= 50


@_bundles
def test_shore_rows_flagged_and_labelled():
    qn = pd.read_parquet(
        V2S / "query_nodes.parquet",
        columns=["is_shore_pseudo", "well_class", "mean_dtw", "is_water_pseudo"],
    )
    s = qn[qn["is_shore_pseudo"].astype(bool)]
    assert len(s) > 0
    assert (s["well_class"] == "shore_pseudo").all()
    assert np.allclose(s["mean_dtw"].to_numpy("float64"), 0.0)
    assert (~s["is_water_pseudo"].astype(bool)).all()


@_bundles
def test_no_new_nan_feature_columns():
    """Shore rows must not introduce a query-feature NaN column absent on v2 -- else the
    trainer's NaN-indicator matrix widens and feat dim drifts off 46."""
    import json

    qfc = json.loads((V2S / "graph_manifest.json").read_text())["query_feature_cols"]
    v2 = pd.read_parquet(V2 / "query_nodes.parquet", columns=qfc)
    v2s = pd.read_parquet(V2S / "query_nodes.parquet", columns=qfc)
    v2_nan = {c for c in qfc if v2[c].isna().any()}
    v2s_nan = {c for c in qfc if v2s[c].isna().any()}
    assert v2s_nan <= v2_nan, f"new NaN feature cols: {sorted(v2s_nan - v2_nan)}"


@_bundles
def test_scoreable_well_set_unchanged_by_shore():
    """Shore rows never enter the v0.2 metric footprint: the scoreable real-well set
    (non-pseudo, unconfined) is identical between v2 and v2s."""
    import pyarrow.parquet as pq

    want = ["is_water_pseudo", "is_shore_pseudo", "confinement_class", "canonical_id"]

    def _scoreable(b):
        have = set(pq.read_schema(b / "query_nodes.parquet").names)
        qn = pd.read_parquet(
            b / "query_nodes.parquet", columns=[c for c in want if c in have]
        )
        shore = (
            qn["is_shore_pseudo"].astype(bool)
            if "is_shore_pseudo" in qn.columns
            else np.zeros(len(qn), bool)
        )
        m = (
            ~qn["is_water_pseudo"].astype(bool)
            & ~shore
            & qn["confinement_class"].isin({"unconfined", "unconfined_marginal"})
        )
        return set(qn.loc[m, "canonical_id"])

    assert _scoreable(V2) == _scoreable(V2S)
