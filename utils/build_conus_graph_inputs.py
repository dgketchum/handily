"""Step 3 of the CONUS WTE/DTW GNN: well (query) side + lateral edges + bundle.

Joins the well HAND table (step 1) and the national reach graph (step 2) into the
full framework-agnostic hetero-graph bundle the trainer loads:

    query_nodes.parquet   one row per CONUS unconfined GWX well: model features
                          (hand_m, cross-fit regional IDW-DTW prior), the DTW
                          target / residual target, HUC4 CV fold, and carried
                          diagnostics (Janssen/source/coords -- never features).
    lateral_edges.parquet directed query->reach edges (k-nearest reach by
                          flowline geometry; rank 0 == controlling reach).
    graph_manifest.json   the bundle manifest the trainer/scorer read.

Leakage discipline (mirrors build_wte_graph_inputs.py):
  * Query model features = hand_m + regional_idw_dtw_oof only. Absolute coords,
    Janssen, Ma, and observed DTW are carried for scoring/diagnostics but flagged
    NON-features. Janssen/Ma are benchmark products (potential well leakage) and
    never enter the feature set, exactly as zell/fan are blocked at RGA.
  * Reaches carry NO labels; there are no query->query edges.
  * The regional IDW prior is cross-fit leave-one-HUC4-fold-out on the SAME folds
    the GNN trains with, so a held-out well's DTW never informs its own prior.
  * Target = residual over the regional prior (mean_dtw - regional_idw_dtw_oof);
    final DTW = regional + residual_hat. Better-conditioned than raw DTW and makes
    the GNN OOF directly comparable to the regional baseline it must beat.

    uv run python utils/build_conus_graph_inputs.py \\
        --hand   /data/ssd2/handily/conus/wte_gnn/conus_wells_hand.parquet \\
        --geom   /data/ssd2/handily/conus/wte_gnn/nhd_flowline_geom.parquet \\
        --graph-dir /data/ssd2/handily/conus/wte_gnn/graph
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_conus_graph_inputs")

NWIS = {"nwis", "ngwmn"}
# Well model features (leak-free). Everything else on the query node is carried
# for scoring/diagnostics only.
QUERY_FEATURE_COLS = ["hand_m", "regional_idw_dtw_oof_m"]
QUERY_DIAGNOSTIC_COLS = [
    "canonical_id",
    "query_node_idx",
    "source",
    "is_nwis",
    "x5070",
    "y5070",
    "huc8",
    "huc4",
    "huc2",
    "mean_dtw",
    "janssen_dtw",
    "regional_idw_dtw_oof_m",
    "regional_deep_idw_dtw_oof_m",
    "cv_fold",
    "block_40km",
]
LATERAL_EDGE_FEATURE_COLS = [
    "lateral_dist_m",
    "log1p_lateral_dist_m",
    "reach_log1p_drainage_km2",
    "reach_strahler",
    "rank",
    "is_controlling",
]


def load_wells_hand(path: str) -> pd.DataFrame:
    """Dedup the HAND shards by well: overlapping FIM HUC8 domains re-sample a
    boundary well in each, so a canonical_id appears multiple times with different
    hand_m. Keep the min-HAND row (height above the *nearest* drainage across all
    overlapping domains) and that row's HUC8.
    """
    df = pd.read_parquet(path)
    n0 = len(df)
    # min hand first (NaN last), so drop_duplicates keep="first" == min-HAND row.
    df = df.sort_values("hand_m", na_position="last").drop_duplicates(
        "canonical_id", keep="first"
    )
    df = df.reset_index(drop=True)
    log.info("HAND table: %d rows -> %d unique wells (overlap dedup)", n0, len(df))
    return df


def assign_folds(huc4: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """HUC4-blocked CV: each whole HUC4 basin assigned to one fold (round-robin on
    a seeded shuffle), so train/test never share a basin -- the cross-basin
    transfer test the CONUS scale is meant to attack.
    """
    uh = np.array(sorted(pd.unique(huc4)))
    rng = np.random.RandomState(seed)
    fold_of = {h: i % folds for i, h in enumerate(rng.permutation(uh))}
    return np.array([fold_of[h] for h in huc4], dtype="int64")


def crossfit_idw(
    xy: np.ndarray, dtw: np.ndarray, fold: np.ndarray, k: int, power: float
) -> np.ndarray:
    """Leave-one-fold-out IDW(kNN) of well DTW -- the leak-free regional prior."""
    pred = np.full(len(dtw), np.nan)
    for f in np.unique(fold):
        te = fold == f
        tr = ~te
        tree = cKDTree(xy[tr])
        dist, idx = tree.query(xy[te], k=k)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        w = 1.0 / np.maximum(dist, 1.0) ** power
        pred[te] = (w * dtw[tr][idx]).sum(1) / w.sum(1)
    return pred


def deep_well_mask(
    wells: pd.DataFrame, quantile: float, unit: str, min_per_unit: int
) -> np.ndarray:
    """Local deepest-quartile mask for the deep-aquifer datum.

    A well is 'deep' if its DTW is at/above the ``quantile`` of DTW WITHIN its
    local hydrologic unit (HUC6 by default). 'Deep' is thus relative to the local
    regime, so a deep-well pool -- and the datum built from it -- exists in every
    region; a single global cut would select almost only arid-West wells and leave
    humid CONUS with no deep wells.

    Leak-safe to compute once over all wells: every well in a HUC6 shares one HUC4
    (= one CV fold), so the per-unit threshold is computed within a single fold,
    and the leave-one-HUC4-out cross-fit of the surface (``crossfit_deep_idw``)
    excludes that whole fold from the deep training pool. Sparse units
    (< ``min_per_unit`` wells) fall back to the HUC4 threshold.
    """
    dtw = wells["mean_dtw"].astype("float64")
    nchar = {"huc6": 6, "huc4": 4}[unit]
    unit_code = wells["huc8"].astype(str).str[:nchar]
    huc4 = wells["huc8"].astype(str).str[:4]
    thr = dtw.groupby(unit_code).transform(lambda s: s.quantile(quantile))
    cnt = dtw.groupby(unit_code).transform("size")
    thr4 = dtw.groupby(huc4).transform(lambda s: s.quantile(quantile))
    thr = thr.where(cnt >= min_per_unit, thr4)
    return (dtw >= thr).to_numpy()


def crossfit_deep_idw(
    xy_all: np.ndarray,
    xy_deep: np.ndarray,
    dtw_deep: np.ndarray,
    fold_all: np.ndarray,
    fold_deep: np.ndarray,
    k: int,
    power: float,
) -> np.ndarray:
    """Leave-one-HUC4-fold-out IDW of DTW for ALL wells from DEEP wells only.

    The deep-aquifer datum: each held-out fold's wells are predicted from the
    deepest-quartile wells in the OTHER folds (``tr = fold_deep != f``), so a
    held-out basin never informs its own deep prior. ``kk = min(k, n_deep_train)``
    guards thin deep folds; a smaller ``k`` than the all-well prior keeps the
    surface local rather than collapsing to a global deep mean.
    """
    pred = np.full(len(xy_all), np.nan)
    for f in np.unique(fold_all):
        te = fold_all == f
        tr = fold_deep != f
        if tr.sum() == 0:
            raise SystemExit(f"deep training pool empty for held-out fold {f}")
        tree = cKDTree(xy_deep[tr])
        kk = min(k, int(tr.sum()))
        dist, idx = tree.query(xy_all[te], k=kk)
        if kk == 1:
            dist, idx = dist[:, None], idx[:, None]
        w = 1.0 / np.maximum(dist, 1.0) ** power
        pred[te] = (w * dtw_deep[tr][idx]).sum(1) / w.sum(1)
    return pred


def build_lateral_edges(
    qxy: np.ndarray, geom: gpd.GeoDataFrame, comid_to_idx: dict, knn: int
) -> pd.DataFrame:
    """k-nearest reach per well by flowline representative-point distance.

    A representative-point cKDTree (vectorised over all wells) is used rather than
    exact point-to-line distance: at CONUS scale (3.4M wells x 2.7M reaches) the
    per-well shapely refinement the RGA builder does is intractable, and rep-point
    nearest is an adequate controlling-reach proxy on the dense V2 network. The
    approximation is recorded in the manifest.
    """
    gx = geom["cx"].to_numpy("float64")
    gy = geom["cy"].to_numpy("float64")
    gidx = geom["reach_node_idx"].to_numpy("int64")
    tree = cKDTree(np.c_[gx, gy])
    dist, cand = tree.query(qxy, k=knn)
    if knn == 1:
        dist, cand = dist[:, None], cand[:, None]
    n_q = len(qxy)
    q_rep = np.repeat(np.arange(n_q, dtype="int64"), knn)
    rank = np.tile(np.arange(knn, dtype="int64"), n_q)
    reach_idx = gidx[cand.ravel()]
    d = dist.ravel()
    return pd.DataFrame(
        {
            "query_node_idx": q_rep,
            "reach_node_idx": reach_idx,
            "lateral_dist_m": d,
            "log1p_lateral_dist_m": np.log1p(d),
            "rank": rank,
            "is_controlling": (rank == 0).astype("float64"),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--hand", default="/data/ssd2/handily/conus/wte_gnn/conus_wells_hand.parquet"
    )
    ap.add_argument(
        "--geom", default="/data/ssd2/handily/conus/wte_gnn/nhd_flowline_geom.parquet"
    )
    ap.add_argument("--graph-dir", default="/data/ssd2/handily/conus/wte_gnn/graph")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--knn-lateral", type=int, default=3)
    ap.add_argument("--idw-k", type=int, default=32)
    ap.add_argument("--idw-power", type=float, default=2.0)
    ap.add_argument("--block-size-m", type=float, default=40000.0)
    ap.add_argument("--seed", type=int, default=0)
    # Deep regional aquifer datum (built from the deepest-quartile wells only).
    ap.add_argument("--deep-quantile", type=float, default=0.75)
    ap.add_argument("--deep-unit", choices=["huc6", "huc4"], default="huc6")
    ap.add_argument("--min-deep-per-unit", type=int, default=30)
    ap.add_argument("--idw-k-deep", type=int, default=16)
    ap.add_argument(
        "--deep-as-feature",
        action="store_true",
        help="add the deep datum as a query feature (Mode A / B)",
    )
    ap.add_argument(
        "--rebase-on-deep",
        action="store_true",
        help="re-base the residual target on the deep datum (Mode B)",
    )
    args = ap.parse_args()
    gdir = Path(args.graph_dir)

    reach_nodes = pd.read_parquet(gdir / "reach_nodes.parquet")
    comid_to_idx = dict(
        zip(
            reach_nodes["comid"].to_numpy("int64"),
            reach_nodes["reach_node_idx"].to_numpy("int64"),
        )
    )
    r_logdr = reach_nodes.set_index("reach_node_idx")["log1p_totda_km2"]
    r_strah = reach_nodes.set_index("reach_node_idx")["streamorde"]

    wells = load_wells_hand(args.hand)
    # HUC4/2 from the dedup'd (min-HAND-domain) HUC8.
    wells["huc8"] = wells["huc8"].astype(str).str.zfill(8)
    wells["huc4"] = wells["huc8"].str[:4]
    wells["huc2"] = wells["huc8"].str[:2]
    wells["is_nwis"] = wells["source"].isin(NWIS)
    wells = wells.reset_index(drop=True)
    wells["query_node_idx"] = np.arange(len(wells), dtype="int64")

    # CV folds (HUC4-blocked) + within-train val blocks (40 km).
    wells["cv_fold"] = assign_folds(wells["huc4"].to_numpy(), args.folds, args.seed)
    bx = (wells["x5070"].to_numpy() // args.block_size_m).astype("int64")
    by = (wells["y5070"].to_numpy() // args.block_size_m).astype("int64")
    wells["block_40km"] = np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))
    log.info(
        "wells=%d  HUC4 basins=%d  folds=%d  non-NWIS=%d",
        len(wells),
        wells["huc4"].nunique(),
        args.folds,
        int((~wells["is_nwis"]).sum()),
    )

    # Leak-free regional IDW-DTW prior, cross-fit on the GNN's HUC4 folds.
    xy = wells[["x5070", "y5070"]].to_numpy("float64")
    dtw = wells["mean_dtw"].to_numpy("float64")
    fold = wells["cv_fold"].to_numpy()
    wells["regional_idw_dtw_oof_m"] = crossfit_idw(
        xy, dtw, fold, args.idw_k, args.idw_power
    )
    log.info(
        "regional IDW prior: MAD=%.2f m (in-CV)",
        float(np.nanmedian(np.abs(wells["regional_idw_dtw_oof_m"] - dtw))),
    )

    # Deep regional aquifer datum: cross-fit IDW from the deepest-quartile wells
    # only (local per-HUC6), a smooth deep base free of riparian/shallow pull.
    deep = deep_well_mask(
        wells, args.deep_quantile, args.deep_unit, args.min_deep_per_unit
    )
    wells["regional_deep_idw_dtw_oof_m"] = crossfit_deep_idw(
        xy, xy[deep], dtw[deep], fold, fold[deep], args.idw_k_deep, args.idw_power
    )
    log.info(
        "deep datum: %d/%d deep wells (%.0f%%, %s q%.2f); surface MAD=%.2f m (in-CV)",
        int(deep.sum()),
        len(wells),
        100 * deep.mean(),
        args.deep_unit,
        args.deep_quantile,
        float(np.nanmedian(np.abs(wells["regional_deep_idw_dtw_oof_m"] - dtw))),
    )

    # Residual target over the chosen base. Mode B (--rebase-on-deep) measures the
    # residual against the deep datum, so the GNN only explains the riparian *rise*
    # above it -- the targeted attack on the too-shallow 30+m deep tail.
    regional_prior_col = (
        "regional_deep_idw_dtw_oof_m"
        if args.rebase_on_deep
        else "regional_idw_dtw_oof_m"
    )
    wells["target_residual_dtw_m"] = dtw - wells[regional_prior_col].to_numpy()
    query_feature_cols = list(QUERY_FEATURE_COLS)
    if args.deep_as_feature:
        query_feature_cols.append("regional_deep_idw_dtw_oof_m")
    log.info(
        "base=%s  features=%s  rebase=%s",
        regional_prior_col,
        query_feature_cols,
        args.rebase_on_deep,
    )

    # Lateral edges (query -> k-nearest reach by flowline rep-point).
    geom = gpd.read_parquet(args.geom)
    geom["reach_node_idx"] = geom["comid"].map(comid_to_idx)
    geom = geom[geom["reach_node_idx"].notna()].copy()
    geom["reach_node_idx"] = geom["reach_node_idx"].astype("int64")
    log.info("flowline geom: %d reaches matched to graph nodes", len(geom))
    lat = build_lateral_edges(xy, geom, comid_to_idx, args.knn_lateral)
    lat["reach_log1p_drainage_km2"] = r_logdr.reindex(lat["reach_node_idx"]).to_numpy()
    lat["reach_strahler"] = r_strah.reindex(lat["reach_node_idx"]).to_numpy()
    log.info(
        "lateral edges: %d (%d wells x knn=%d)", len(lat), len(wells), args.knn_lateral
    )

    q_keep = list(
        dict.fromkeys(
            QUERY_DIAGNOSTIC_COLS
            + query_feature_cols
            + ["target_residual_dtw_m", "well_class", "confinement_class"]
        )
    )
    wells[q_keep].to_parquet(gdir / "query_nodes.parquet")
    lat[["query_node_idx", "reach_node_idx", *LATERAL_EDGE_FEATURE_COLS]].to_parquet(
        gdir / "lateral_edges.parquet"
    )

    reach_manifest = json.loads((gdir / "reach_graph_manifest.json").read_text())
    manifest = {
        "stage": "full_bundle",
        "crs": "EPSG:5070",
        "counts": {
            "reach_nodes": reach_manifest["counts"]["reach_nodes"],
            "channel_edges": reach_manifest["counts"]["channel_edges"],
            "query_nodes": int(len(wells)),
            "query_nodes_non_nwis": int((~wells["is_nwis"]).sum()),
            "query_nodes_nwis": int(wells["is_nwis"].sum()),
            "lateral_edges": int(len(lat)),
            "hand_finite_frac": float(np.isfinite(wells["hand_m"]).mean()),
            "regional_deep_finite_frac": float(
                np.isfinite(wells["regional_deep_idw_dtw_oof_m"]).mean()
            ),
            "deep_well_count": int(deep.sum()),
        },
        "target_col": "target_residual_dtw_m",
        "target_definition": f"mean_dtw - {regional_prior_col} (cross-fit IDW prior)",
        "final_dtw_definition": f"{regional_prior_col} + residual_hat",
        "obs_dtw_col": "mean_dtw",
        "regional_prior_col": regional_prior_col,
        "cv_fold_col": "cv_fold",
        "cv_group_col": "block_40km",
        "cv_scheme": f"HUC4-blocked, {args.folds} folds",
        "reach_feature_cols": reach_manifest["reach_feature_cols"],
        "reach_structural_nan_cols": reach_manifest["reach_structural_nan_cols"],
        "channel_edge_feature_cols": reach_manifest["channel_edge_feature_cols"],
        "query_feature_cols": query_feature_cols,
        "query_diagnostic_cols": QUERY_DIAGNOSTIC_COLS,
        "lateral_edge_feature_cols": LATERAL_EDGE_FEATURE_COLS,
        "knn_lateral": args.knn_lateral,
        "idw_k": args.idw_k,
        "idw_power": args.idw_power,
        "deep_datum": {
            "quantile": args.deep_quantile,
            "unit": args.deep_unit,
            "min_per_unit": args.min_deep_per_unit,
            "idw_k_deep": args.idw_k_deep,
            "deep_as_feature": args.deep_as_feature,
            "rebase_on_deep": args.rebase_on_deep,
        },
        "leakage_notes": [
            "Query features = hand_m + cross-fit regional IDW prior only.",
            "Janssen/Ma/coords/obs DTW carried for scoring, never features.",
            "Reaches carry no labels; no query->query edges.",
            "Regional prior cross-fit leave-one-HUC4-fold-out on the GNN folds.",
            "Deep datum: per-HUC6 deepest-quartile mask (each HUC6 in one HUC4=one "
            "fold) + leave-one-HUC4-out cross-fit, so held-out basins never inform "
            "their own deep prior.",
        ],
        "approximations": [
            "Lateral attachment uses flowline representative-point nearest, not "
            "exact point-to-line distance (CONUS-scale tractability).",
        ],
        "deferred": [
            "StreamCat per-COMID covariates (recharge/BFI/bedrock/soils): pynhd "
            "0.19.4 StreamCat is broken (year-range parse bug); reach side uses VAA "
            "features for v1. Add once StreamCat is reachable if v1 shows life.",
        ],
        "sources": {"hand": args.hand, "geom": args.geom, "reach_graph": str(gdir)},
    }
    (gdir / "graph_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        "wrote query_nodes.parquet, lateral_edges.parquet, graph_manifest.json -> %s",
        gdir,
    )


if __name__ == "__main__":
    main()
