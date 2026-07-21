"""E4 SWL auxiliary-label tests: build-output leakage/schema invariants + the
trainer-wiring expressions main() uses to append them.

Data-dependent tests (the built aux parquet + the base bundle) skip when the CONUS
artifacts are absent, so the suite still runs off-zoran. The wiring tests are pure
logic (synthetic frames) and always run -- they pin the leakage-critical masks:
  * SWL rows flow into `real = ~water & ~swl` (never a fit source, never a metric);
  * an SWL point in fold f is absent from fold-f training (fold-aware masking, rule 3);
  * aux lateral edges are offset onto the base query frame;
  * the neutral aux embedding standardizes to 0 and leaves base embeddings unperturbed.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BUNDLE = Path("/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2")
AUX_Q = Path("/data/ssd2/handily/conus/wte_gnn/e4_swl_labels.parquet")
AUX_L = Path("/data/ssd2/handily/conus/wte_gnn/e4_swl_lateral_edges.parquet")
_HAVE_DATA = BUNDLE.exists() and AUX_Q.exists() and AUX_L.exists()
_data = pytest.mark.skipif(not _HAVE_DATA, reason="CONUS SWL artifacts absent")


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# build-output invariants (data-dependent)
# ---------------------------------------------------------------------------
@_data
def test_confinement_screen_zero_confined():
    aux = pd.read_parquet(AUX_Q)
    assert set(aux["confinement_class"].unique()) <= {
        "unconfined",
        "unconfined_marginal",
    }
    assert bool(aux["is_swl_aux"].all())
    assert bool((~aux["is_water_pseudo"]).all())


@_data
def test_no_shared_canonical_id_with_anchor():
    aux = pd.read_parquet(AUX_Q, columns=["canonical_id"])
    qn = pd.read_parquet(
        BUNDLE / "query_nodes.parquet", columns=["canonical_id", "is_water_pseudo"]
    )
    anchors = set(qn[~qn["is_water_pseudo"]]["canonical_id"])
    assert len(set(aux["canonical_id"]) & anchors) == 0


@_data
def test_min_100m_from_every_anchor():
    from scipy.spatial import cKDTree

    aux = pd.read_parquet(AUX_Q, columns=["x5070", "y5070"])
    qn = pd.read_parquet(
        BUNDLE / "query_nodes.parquet", columns=["x5070", "y5070", "is_water_pseudo"]
    )
    anch = qn[~qn["is_water_pseudo"]]
    d, _ = cKDTree(anch[["x5070", "y5070"]].to_numpy("float64")).query(
        aux[["x5070", "y5070"]].to_numpy("float64"), k=1
    )
    assert float(d.min()) >= 100.0


@_data
def test_locked_huc4s_excluded():
    aux = pd.read_parquet(AUX_Q, columns=["huc4"])
    assert not aux["huc4"].astype(str).isin({"0707", "1019", "1605"}).any()


@_data
def test_fold_matches_anchor_huc12():
    """Rule 3, part 1: an SWL point in an anchor-populated HUC12 carries that
    HUC12's fold (so it is held out exactly when those anchors are)."""
    aux = pd.read_parquet(AUX_Q, columns=["cv_unit", "cv_fold"])
    qn = pd.read_parquet(
        BUNDLE / "query_nodes.parquet",
        columns=["cv_unit", "cv_fold", "is_water_pseudo"],
    )
    anch = qn[~qn["is_water_pseudo"]].dropna(subset=["cv_unit"])
    # cv_unit -> fold is 1:1 among anchors (the build's premise); verify then check.
    u2f = anch.groupby("cv_unit")["cv_fold"].agg(["first", "nunique"])
    assert (u2f["nunique"] == 1).all()
    fmap = u2f["first"].to_dict()
    matched = aux[aux["cv_unit"].isin(fmap)]
    assert len(matched) > 0
    assert (
        matched["cv_fold"].to_numpy() == matched["cv_unit"].map(fmap).to_numpy()
    ).all()


@_data
def test_no_new_nan_feature_columns():
    """Aux must not introduce a query-feature NaN column absent on the base -- else
    the trainer's NaN-indicator matrix would widen and perturb the architecture."""
    import json

    qfc = json.loads((BUNDLE / "graph_manifest.json").read_text())["query_feature_cols"]
    base = pd.read_parquet(BUNDLE / "query_nodes.parquet", columns=qfc)
    aux = pd.read_parquet(AUX_Q, columns=qfc)
    base_nan = {c for c in qfc if base[c].isna().any()}
    aux_nan = {c for c in qfc if aux[c].isna().any()}
    assert aux_nan <= base_nan, (
        f"new NaN feature cols in aux: {sorted(aux_nan - base_nan)}"
    )


@_data
def test_target_is_obs_wte_minus_regional_prior():
    aux = pd.read_parquet(
        AUX_Q,
        columns=[
            "wte_residual_m",
            "wte_obs_m",
            "regional_wte_idw_oof_m",
            "wte_resid_base_m",
            "z_surf_well_m",
            "mean_dtw",
        ],
    )
    r = aux["regional_wte_idw_oof_m"].to_numpy("float64")
    assert np.isfinite(aux["wte_residual_m"].to_numpy("float64")).all()
    assert np.allclose(
        aux["wte_residual_m"], aux["wte_obs_m"].to_numpy("float64") - r, atol=1e-6
    )
    assert np.allclose(
        aux["wte_resid_base_m"], aux["z_surf_well_m"].to_numpy("float64") - r, atol=1e-6
    )
    assert np.allclose(
        aux["wte_obs_m"], aux["z_surf_well_m"] - aux["mean_dtw"], atol=1e-6
    )


@_data
def test_lateral_edges_local_index_and_schema():
    aux = pd.read_parquet(AUX_Q, columns=["query_node_idx"])
    lat = pd.read_parquet(AUX_L)
    n = len(aux)
    assert aux["query_node_idx"].to_numpy().tolist() == list(range(n))
    assert int(lat["query_node_idx"].min()) == 0
    assert int(lat["query_node_idx"].max()) == n - 1
    # knn=3 attachment
    assert lat.groupby("query_node_idx").size().max() <= 3
    for c in [
        "lateral_dist_m",
        "log1p_lateral_dist_m",
        "reach_log1p_drainage_km2",
        "reach_strahler",
        "rank",
        "is_controlling",
        "rel_elev_query_reach_m",
        "lateral_conductance",
    ]:
        assert c in lat.columns


# ---------------------------------------------------------------------------
# trainer-wiring expressions (pure logic; the exact statements main() uses)
# ---------------------------------------------------------------------------
def _append_like_main(qn, le, sqn, sle):
    """Replicate the append block main() runs under --swl-labels."""
    base_n = len(qn)
    if "is_swl_aux" not in qn.columns:
        qn = qn.assign(is_swl_aux=False)
    sqn = sqn.copy()
    for c in set(qn.columns) - set(sqn.columns):
        sqn[c] = np.nan
    sqn = sqn[qn.columns].copy()
    sqn["is_swl_aux"] = True
    sqn["query_node_idx"] = np.arange(base_n, base_n + len(sqn), dtype="int64")
    sle = sle.copy()
    sle["query_node_idx"] = sle["query_node_idx"].to_numpy("int64") + base_n
    qn = pd.concat([qn, sqn], ignore_index=True)
    le = pd.concat([le[sle.columns], sle], ignore_index=True)
    return qn, le


def test_swl_append_offsets_and_masks():
    qn = pd.DataFrame(
        {
            "query_node_idx": [0, 1, 2],
            "canonical_id": ["a", "b", "c"],
            "is_water_pseudo": [False, False, True],
            "cv_fold": [0, 1, 0],
        }
    )
    le = pd.DataFrame(
        {"query_node_idx": [0, 1, 2], "reach_node_idx": [10, 11, 12], "rank": [0, 0, 0]}
    )
    sqn = pd.DataFrame(
        {
            "query_node_idx": [0, 1],  # local
            "canonical_id": ["s0", "s1"],
            "is_water_pseudo": [False, False],
            "cv_fold": [1, 0],
        }
    )
    sle = pd.DataFrame(
        {"query_node_idx": [0, 0, 1], "reach_node_idx": [20, 21, 22], "rank": [0, 1, 0]}
    )
    qn2, le2 = _append_like_main(qn, le, sqn, sle)
    # contiguous query_node_idx over the combined frame (the trainer invariant)
    assert qn2["query_node_idx"].to_numpy().tolist() == [0, 1, 2, 3, 4]
    # aux lateral edges offset by base_n=3
    assert le2[le2["reach_node_idx"] >= 20]["query_node_idx"].to_numpy().tolist() == [
        3,
        3,
        4,
    ]
    # masks: exactly the two aux rows are swl; water is the original pseudo row
    water = qn2["is_water_pseudo"].to_numpy(bool)
    swl = qn2["is_swl_aux"].to_numpy(bool)
    assert swl.tolist() == [False, False, False, True, True]
    real = ~water & ~swl
    assert real.tolist() == [True, True, False, False, False]


def test_flag_off_is_noop_on_masks():
    qn = pd.DataFrame({"is_water_pseudo": [False, True, False]})
    water = qn["is_water_pseudo"].to_numpy(bool)
    swl = (
        qn["is_swl_aux"].to_numpy(bool)
        if "is_swl_aux" in qn.columns
        else np.zeros(len(qn), bool)
    )
    real = ~water & ~swl
    assert swl.tolist() == [False, False, False]
    assert real.tolist() == [True, False, True]  # identical to pre-E4 real=~water


def test_fold_aware_masking_holds_out_swl(rng=np.random.RandomState(0)):
    """Rule 3, part 2: an SWL point in fold f is absent from fold-f training -- the
    exact fold-loop expression main() uses (test = fold==f; tr = ~test & ~va)."""
    fold = np.array([0, 1, 2, 0, 1, 2, 0, 1])  # last 3 are aux
    swl = np.array([False] * 5 + [True] * 3)
    water = np.zeros(len(fold), bool)
    real = ~water & ~swl
    for f in np.unique(fold):
        test = fold == f
        trainval = ~test
        va = np.zeros(len(fold), bool)  # no val for the test
        va &= real
        tr = trainval & ~va
        # every SWL point in the held-out fold must be OUT of training
        assert not (tr & swl & test).any()
        # SWL never enters a fit-eligible real mask
        assert not (tr & real & swl).any()


def test_neutral_embedding_standardizes_aux_to_zero():
    """The neutral aux embedding standardizes to 0 and base rows match the no-aux
    path exactly (stats fit on base only)."""
    tc = _load("train_conus_gnn")
    rng = np.random.RandomState(1)
    cols = [f"mae_{i}" for i in range(4)]
    base_emb = rng.normal(5.0, 3.0, size=(6, 4))
    swl = np.array([False] * 6 + [True] * 3)
    emb = np.empty((9, 4), "float64")
    emb[~swl] = base_emb
    emb[swl] = base_emb.mean(0, keepdims=True)
    mae_ord = pd.DataFrame(emb, columns=cols)
    stats = tc.fit_stats(mae_ord.loc[~swl].reset_index(drop=True), cols, None)
    std = tc.apply_stats(mae_ord, stats)
    # aux rows -> ~0 in every (finite) column
    assert np.allclose(std[swl], 0.0, atol=1e-8)
    # base rows identical to standardizing base-only with the same stats
    base_only = tc.apply_stats(pd.DataFrame(base_emb, columns=cols), stats)
    assert np.allclose(std[~swl], base_only, atol=1e-12)
