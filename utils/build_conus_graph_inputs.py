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
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stacker_features import (  # noqa: E402
    ACCUM,
    COARSE_SURFACE,
    DIST_STREAM,
    ETRM_ETA,
    ETRM_RECHARGE,
    ETRM_RUNOFF,
    SLOPE,
    TRI,
    sample_coarse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_conus_graph_inputs")

NWIS = {"nwis", "ngwmn"}
DEM = "/data/ssd1/streamflow-ml-data/conus-dem/data/elev48i0100a.tif"
# Target modes. dtw_residual = predict (mean_dtw - regional prior), reconstruct
# DTW = regional + residual (v1/v2). wte = predict real-magnitude water-table
# elevation directly, reconstruct DTW = z_surf_well - wte_hat. The two paths share
# the bundle/edges; only the query target + head-space priors differ.
TARGET_DTW_RESIDUAL = "dtw_residual"
TARGET_WTE = "wte"
TARGET_CHOICES = (TARGET_DTW_RESIDUAL, TARGET_WTE)
SURFACE_ELEV_COL = "z_surf_well_m"  # DEM land-surface elev at the well (datum)
OBS_WTE_COL = "wte_obs_m"  # observed head = z_surf_well - mean_dtw
REGIONAL_WTE_COL = "regional_wte_idw_oof_m"  # cross-fit IDW of observed WTE
DEEP_REGIONAL_WTE_COL = "deep_regional_wte_idw_oof_m"  # cross-fit IDW of deep WTE
FAC_REM_WTE_COL = "fac_rem_wte_m"  # z_surf - fac_rem_dtw (legit local DTW product)
HAND_WTE_COL = "hand_wte_m"  # z_surf - hand_m (head-like FIM-HAND feature)
# Well model features (leak-free). Everything else on the query node is carried
# for scoring/diagnostics only.
QUERY_FEATURE_COLS = ["hand_m", "regional_idw_dtw_oof_m"]
# v2: the proven RELIEF_ETRM query bank (terrain relief discriminators + ETRM water-
# balance fluxes -- the only covariate family that moved the deep tail). Exogenous /
# target-blind. Appended to query features behind --relief-etrm-features. gridMET
# aridity is deliberately excluded (it was NEGATIVE in the stacker sweep).
RELIEF_ETRM_FEATURE_COLS = [
    "slope_deg",
    "tri_100m",
    "dist_to_stream_m",
    "log_drainage_area",
    "elev_above_coarse_m",
    "etrm_recharge_mm",
    "etrm_eta_mm",
    "etrm_runoff_mm",
]
# Anchor BC schema (v2). anchor_x = class/source one-hot + head_uncertainty ONLY
# (head_m is audit-only -- feeding raw head would violate the no-absolute-elevation
# rule). Anchor classes/sources mirror build_conus_anchors.py.
ANCHOR_CLASSES = ("spring", "open_water", "wetland")
ANCHOR_SOURCES = ("nhd_hr", "nwi", "3dhp")
ANCHOR_FEATURE_COLS = (
    [f"anchor_is_{c}" for c in ANCHOR_CLASSES]
    + [f"anchor_src_{s}" for s in ANCHOR_SOURCES]
    + ["head_uncertainty_m"]
)
ANCHOR_REACH_EDGE_FEATURE_COLS = [
    "anchor_dist_m",
    "log1p_anchor_dist_m",
    "rel_elev_anchor_reach_m",
    "anchor_conductance",
    "head_uncertainty_m",
    "is_within_R",
]
ANCHOR_QUERY_EDGE_FEATURE_COLS = [
    "anchor_dist_m",
    "log1p_anchor_dist_m",
    "rel_elev_anchor_query_m",
    "head_uncertainty_m",
    "is_within_R",
    "is_controlling",
]
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
    # v2 Darcy attrs: the missing shallow signal (well-vs-reach rel-elev) in RGA-safe
    # relative form + conductance from lateral distance and reach drainage.
    "rel_elev_query_reach_m",
    "lateral_conductance",
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
    xy: np.ndarray, value: np.ndarray, fold: np.ndarray, k: int, power: float
) -> np.ndarray:
    """Leave-one-fold-out IDW(kNN) of a scalar well value -- the leak-free prior.

    Scalar-generic: ``value`` is DTW for the regional DTW prior and observed WTE
    for the head-space prior. A held-out fold's wells are never in their own
    neighbor set.
    """
    pred = np.full(len(value), np.nan)
    for f in np.unique(fold):
        te = fold == f
        tr = ~te
        tree = cKDTree(xy[tr])
        dist, idx = tree.query(xy[te], k=k)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        w = 1.0 / np.maximum(dist, 1.0) ** power
        pred[te] = (w * value[tr][idx]).sum(1) / w.sum(1)
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
    value_deep: np.ndarray,
    fold_all: np.ndarray,
    fold_deep: np.ndarray,
    k: int,
    power: float,
) -> np.ndarray:
    """Leave-one-HUC4-fold-out IDW for ALL wells from the DEEP-well pool.

    The deep datum: each held-out fold's wells are predicted from the deepest-
    quartile wells in the OTHER folds (``tr = fold_deep != f``), so a held-out
    basin never informs its own deep prior. ``kk = min(k, n_deep_train)`` guards
    thin deep folds; a smaller ``k`` than the all-well prior keeps the surface
    local rather than collapsing to a global deep mean. Scalar-generic in
    ``value_deep`` (deep DTW for the DTW datum; deep observed WTE for the head
    datum -- the deep pool is a DTW depth class either way).
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
        pred[te] = (w * value_deep[tr][idx]).sum(1) / w.sum(1)
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


def sample_relief_etrm(
    x: np.ndarray, y: np.ndarray, well_surf_m: np.ndarray
) -> dict[str, np.ndarray]:
    """The RELIEF_ETRM query bank, sampled at well 5070 coords (target-blind rasters).

    All rasters are EPSG:5070 (sampled at x5070/y5070). elev_above_coarse_m uses the
    same well land-surface elevation (DEM-sampled) used for the rel-elev edge attrs,
    so the relative-elevation features are datum-consistent across the whole graph.
    Off-footprint / nodata stay NaN (the trainer median-imputes + flags).
    """
    return {
        "slope_deg": sample_coarse(SLOPE, x, y),
        "tri_100m": sample_coarse(TRI, x, y),
        "dist_to_stream_m": sample_coarse(DIST_STREAM, x, y),
        "log_drainage_area": np.log1p(np.abs(sample_coarse(ACCUM, x, y))),
        "elev_above_coarse_m": well_surf_m - sample_coarse(COARSE_SURFACE, x, y),
        "etrm_recharge_mm": sample_coarse(ETRM_RECHARGE, x, y),
        "etrm_eta_mm": sample_coarse(ETRM_ETA, x, y),
        "etrm_runoff_mm": sample_coarse(ETRM_RUNOFF, x, y),
    }


def build_anchor_query_edges(
    axy: np.ndarray, qxy: np.ndarray, knn: int, max_dist_m: float
) -> pd.DataFrame:
    """Directed anchor->query edges: per well, its ``knn`` nearest anchors.

    Degree is capped low (k<=2, the BC is meant to nudge, not to collapse to a
    distance-to-nearest-spring lookup). ``is_within_R`` flags (does not drop) far
    attachments; ``is_controlling`` marks the nearest anchor (rank 0).
    """
    tree = cKDTree(axy)
    dist, cand = tree.query(qxy, k=knn)
    if knn == 1:
        dist, cand = dist[:, None], cand[:, None]
    n_q = len(qxy)
    q_rep = np.repeat(np.arange(n_q, dtype="int64"), knn)
    rank = np.tile(np.arange(knn, dtype="int64"), n_q)
    anchor_idx = cand.ravel().astype("int64")
    d = dist.ravel()
    return pd.DataFrame(
        {
            "query_node_idx": q_rep,
            "anchor_node_idx": anchor_idx,
            "anchor_dist_m": d,
            "log1p_anchor_dist_m": np.log1p(d),
            "rank": rank,
            "is_controlling": (rank == 0).astype("float64"),
            "is_within_R": (d <= max_dist_m).astype("float64"),
        }
    )


def build_anchor_x(anchor_nodes: pd.DataFrame) -> pd.DataFrame:
    """anchor_x = class one-hot + source one-hot + head_uncertainty (NO head_m)."""
    out = pd.DataFrame({"anchor_node_idx": anchor_nodes["anchor_node_idx"].to_numpy()})
    for c in ANCHOR_CLASSES:
        out[f"anchor_is_{c}"] = (
            (anchor_nodes["anchor_class"] == c).astype("float64").to_numpy()
        )
    for s in ANCHOR_SOURCES:
        out[f"anchor_src_{s}"] = (
            (anchor_nodes["source"] == s).astype("float64").to_numpy()
        )
    out["head_uncertainty_m"] = anchor_nodes["head_uncertainty_m"].to_numpy("float64")
    # carried for the rel-elev edge attrs + QGIS audit (NOT model features).
    for c in ("x5070", "y5070", "head_m", "anchor_class", "source"):
        out[c] = anchor_nodes[c].to_numpy()
    return out


def join_stacker_features(
    wells: pd.DataFrame, path: str | None
) -> tuple[bool, float | None]:
    """Map ``fac_rem_dtw_m`` onto wells by canonical_id (ONLY these two columns).

    Deliberately reads just canonical_id + fac_rem_dtw_m -- the stacker table also
    carries frozen ConusWTE / retired-well accounting that would leak labels if
    blindly merged. A dict-map (not a join) preserves wells row order exactly.
    Returns (joined, finite_fraction).
    """
    if not path:
        return False, None
    sf = pd.read_parquet(path, columns=["canonical_id", "fac_rem_dtw_m"])
    sf = sf.drop_duplicates("canonical_id")
    m = dict(
        zip(sf["canonical_id"].to_numpy(), sf["fac_rem_dtw_m"].to_numpy("float64"))
    )
    vals = wells["canonical_id"].map(m).astype("float64").to_numpy()
    wells["fac_rem_dtw_m"] = vals
    return True, float(np.isfinite(vals).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--hand", default="/data/ssd2/handily/conus/wte_gnn/conus_wells_hand.parquet"
    )
    ap.add_argument(
        "--geom", default="/data/ssd2/handily/conus/wte_gnn/nhd_flowline_geom.parquet"
    )
    ap.add_argument("--graph-dir", default="/data/ssd2/handily/conus/wte_gnn/graph")
    ap.add_argument(
        "--target",
        choices=list(TARGET_CHOICES),
        default=TARGET_DTW_RESIDUAL,
        help="training target mode; dtw_residual preserves the v1/v2 behavior, "
        "wte predicts real-magnitude water-table elevation",
    )
    ap.add_argument(
        "--stacker-features",
        default=None,
        help="optional parquet keyed by canonical_id with fac_rem_dtw_m; used to "
        "build fac_rem_wte_m in --target wte mode (only those two cols are read)",
    )
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
    # v2 additions ------------------------------------------------------------
    ap.add_argument(
        "--relief-etrm-features",
        action="store_true",
        help="add the RELIEF_ETRM query bank (terrain relief + ETRM fluxes)",
    )
    ap.add_argument("--dem", default=DEM, help="land-surface DEM for rel-elev attrs")
    ap.add_argument(
        "--anchors-dir",
        default=None,
        help="anchor stage out-dir; enables anchor BC nodes + anchor edges",
    )
    ap.add_argument(
        "--knn-anchor-query", type=int, default=2, help="anchors per well (k<=2)"
    )
    ap.add_argument("--max-attach-dist-m", type=float, default=5000.0)
    ap.add_argument("--conductance-p", type=float, default=0.5)
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
    if "reach_elev_m" not in reach_nodes.columns:
        raise SystemExit(
            "reach_nodes lacks reach_elev_m -- rebuild the reach graph with the v2 "
            "build_conus_reach_graph.py (rel-elev edge attrs)"
        )
    r_elev = reach_nodes.set_index("reach_node_idx")["reach_elev_m"]
    r_totda = reach_nodes.set_index("reach_node_idx")["totdasqkm"]

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

    # Well land-surface elevation (DEM): the SHARED land-surface datum for every
    # relative-elevation feature/edge attr (lateral + anchor), sampled from the same
    # DEM the reaches and anchors use, so all rel-elev attrs are datum-consistent.
    well_surf_m = sample_coarse(args.dem, xy[:, 0], xy[:, 1])
    log.info(
        "well land-surface elev: %.4f finite frac",
        float(np.isfinite(well_surf_m).mean()),
    )
    # The DEM land-surface elevation is the shared datum. In WTE mode it is also the
    # target datum (wte_obs = z_surf - dtw) and the DTW reconstruction term, so it
    # MUST be finite for every well -- a non-finite surface cannot reconstruct DTW.
    wells[SURFACE_ELEV_COL] = well_surf_m
    if args.target == TARGET_WTE and not np.isfinite(well_surf_m).all():
        n_bad = int((~np.isfinite(well_surf_m)).sum())
        raise SystemExit(
            f"{n_bad} wells lack finite {SURFACE_ELEV_COL}; the WTE target cannot "
            "reconstruct DTW for them (add a deliberate drop flag + audit if a few "
            "off-DEM wells must be retained -- do not silently drop)"
        )
    wells[OBS_WTE_COL] = well_surf_m - dtw  # observed head (NaN-tolerant in resid mode)
    if args.relief_etrm_features:
        for col, vals in sample_relief_etrm(xy[:, 0], xy[:, 1], well_surf_m).items():
            wells[col] = vals
            log.info("  %s: %.3f finite frac", col, float(np.isfinite(vals).mean()))

    # --- target + feature set by mode ----------------------------------------
    fac_joined, fac_finite_frac = False, None
    if args.target == TARGET_WTE:
        # Head-space priors: leave-one-HUC4-out IDW of OBSERVED WTE directly. Never
        # z_surf - dtw_prior -- that re-injects rough local terrain into a feature
        # that is supposed to be a smooth head. The deep pool is the same DTW-
        # selected deepest-quartile wells, but the interpolated quantity is WTE.
        wte = wells[OBS_WTE_COL].to_numpy("float64")
        if not np.isfinite(wte).all():
            raise SystemExit(
                f"{int((~np.isfinite(wte)).sum())} non-finite {OBS_WTE_COL}"
            )
        wells[REGIONAL_WTE_COL] = crossfit_idw(
            xy, wte, fold, args.idw_k, args.idw_power
        )
        wells[DEEP_REGIONAL_WTE_COL] = crossfit_deep_idw(
            xy, xy[deep], wte[deep], fold, fold[deep], args.idw_k_deep, args.idw_power
        )
        log.info(
            "WTE priors (DTW-reconstructed MAD): regional=%.2f m  deep=%.2f m",
            float(
                np.nanmedian(
                    np.abs((well_surf_m - wells[REGIONAL_WTE_COL].to_numpy()) - dtw)
                )
            ),
            float(
                np.nanmedian(
                    np.abs(
                        (well_surf_m - wells[DEEP_REGIONAL_WTE_COL].to_numpy()) - dtw
                    )
                )
            ),
        )
        wells[HAND_WTE_COL] = well_surf_m - wells["hand_m"].to_numpy("float64")
        target_col = OBS_WTE_COL
        regional_prior_col = None
        query_feature_cols = [
            SURFACE_ELEV_COL,
            "hand_m",
            HAND_WTE_COL,
            REGIONAL_WTE_COL,
            DEEP_REGIONAL_WTE_COL,
        ]
        # FAC-REM head feature only if the stacker join yields finite coverage; an
        # all-NaN feature carries no signal and would poison the train-only scaler.
        fac_joined, fac_finite_frac = join_stacker_features(
            wells, args.stacker_features
        )
        if fac_joined and fac_finite_frac and fac_finite_frac > 0:
            wells[FAC_REM_WTE_COL] = well_surf_m - wells["fac_rem_dtw_m"].to_numpy(
                "float64"
            )
            query_feature_cols.append(FAC_REM_WTE_COL)
        if args.relief_etrm_features:
            query_feature_cols += RELIEF_ETRM_FEATURE_COLS
        log.info(
            "target=wte  features=%s  fac_rem_wte joined=%s (finite frac=%s)",
            query_feature_cols,
            fac_joined,
            f"{fac_finite_frac:.3f}" if fac_finite_frac is not None else "n/a",
        )
    else:
        # Residual target over the chosen base. Mode B (--rebase-on-deep) measures
        # the residual against the deep datum, so the GNN only explains the riparian
        # *rise* above it -- the targeted attack on the too-shallow 30+m deep tail.
        regional_prior_col = (
            "regional_deep_idw_dtw_oof_m"
            if args.rebase_on_deep
            else "regional_idw_dtw_oof_m"
        )
        wells["target_residual_dtw_m"] = dtw - wells[regional_prior_col].to_numpy()
        target_col = "target_residual_dtw_m"
        query_feature_cols = list(QUERY_FEATURE_COLS)
        if args.deep_as_feature:
            query_feature_cols.append("regional_deep_idw_dtw_oof_m")
        if args.relief_etrm_features:
            query_feature_cols += RELIEF_ETRM_FEATURE_COLS
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
    # Darcy attrs: well-vs-reach relative elevation (the missing shallow signal, in
    # RGA-safe relative form) + conductance from lateral distance and reach drainage.
    lat_reach_elev = r_elev.reindex(lat["reach_node_idx"]).to_numpy()
    lat["rel_elev_query_reach_m"] = (
        well_surf_m[lat["query_node_idx"].to_numpy()] - lat_reach_elev
    )
    lat_reach_drain = r_totda.reindex(lat["reach_node_idx"]).to_numpy()
    lat["lateral_conductance"] = np.log1p(
        np.clip(lat_reach_drain, 0, None)
    ) - args.conductance_p * np.log1p(
        np.clip(lat["lateral_dist_m"].to_numpy(), 0, None)
    )
    log.info(
        "lateral edges: %d (%d wells x knn=%d)", len(lat), len(wells), args.knn_lateral
    )

    # Carry the surface datum + observed WTE in both modes (cheap, enables cross-
    # mode diagnostics + the WTE identity check); mode-specific target/priors added.
    extra_keep = [SURFACE_ELEV_COL, OBS_WTE_COL, "well_class", "confinement_class"]
    if args.target == TARGET_WTE:
        extra_keep += [REGIONAL_WTE_COL, DEEP_REGIONAL_WTE_COL, HAND_WTE_COL]
        if FAC_REM_WTE_COL in wells.columns:
            extra_keep += [FAC_REM_WTE_COL, "fac_rem_dtw_m"]
    else:
        extra_keep.append("target_residual_dtw_m")
    q_keep = list(
        dict.fromkeys(QUERY_DIAGNOSTIC_COLS + query_feature_cols + extra_keep)
    )
    wells[q_keep].to_parquet(gdir / "query_nodes.parquet")
    lat[["query_node_idx", "reach_node_idx", *LATERAL_EDGE_FEATURE_COLS]].to_parquet(
        gdir / "lateral_edges.parquet"
    )

    # --- anchor BC nodes + anchor->reach / anchor->query edges (v2) -----------
    anchor_block = None
    if args.anchors_dir:
        adir = Path(args.anchors_dir)
        anodes = pd.read_parquet(adir / "anchor_nodes.parquet")
        ar = pd.read_parquet(adir / "anchor_edges.parquet")  # anchor->reach attachment
        a_head = anodes.set_index("anchor_node_idx")["head_m"]
        a_unc = anodes.set_index("anchor_node_idx")["head_uncertainty_m"]

        # enrich anchor->reach attachment with rel-elev + conductance + uncertainty.
        ar_reach_elev = r_elev.reindex(ar["reach_node_idx"]).to_numpy()
        ar["rel_elev_anchor_reach_m"] = (
            ar_reach_elev - a_head.reindex(ar["anchor_node_idx"]).to_numpy()
        )
        ar_drain = r_totda.reindex(ar["reach_node_idx"]).to_numpy()
        ar["anchor_conductance"] = np.log1p(
            np.clip(ar_drain, 0, None)
        ) - args.conductance_p * np.log1p(
            np.clip(ar["anchor_dist_m"].to_numpy(), 0, None)
        )
        ar["head_uncertainty_m"] = a_unc.reindex(ar["anchor_node_idx"]).to_numpy()

        # anchor->query edges (per well, k<=2 nearest anchors; gated, never a lookup).
        axy = anodes[["x5070", "y5070"]].to_numpy("float64")
        aq = build_anchor_query_edges(
            axy, xy, args.knn_anchor_query, args.max_attach_dist_m
        )
        aq["rel_elev_anchor_query_m"] = (
            well_surf_m[aq["query_node_idx"].to_numpy()]
            - a_head.reindex(aq["anchor_node_idx"]).to_numpy()
        )
        aq["head_uncertainty_m"] = a_unc.reindex(aq["anchor_node_idx"]).to_numpy()

        ax = build_anchor_x(anodes)
        ax.to_parquet(gdir / "anchor_nodes.parquet")
        ar[
            ["anchor_node_idx", "reach_node_idx", *ANCHOR_REACH_EDGE_FEATURE_COLS]
        ].to_parquet(gdir / "anchor_to_reach_edges.parquet")
        aq[
            ["anchor_node_idx", "query_node_idx", *ANCHOR_QUERY_EDGE_FEATURE_COLS]
        ].to_parquet(gdir / "anchor_to_query_edges.parquet")
        assert not (
            (
                ax["anchor_is_spring"]
                + ax["anchor_is_open_water"]
                + ax["anchor_is_wetland"]
            )
            == 0
        ).any(), "anchor with no class one-hot"
        anchor_block = {
            "anchors_dir": str(adir),
            "anchor_nodes": int(len(ax)),
            "anchor_to_reach_edges": int(len(ar)),
            "anchor_to_query_edges": int(len(aq)),
            "anchor_feature_cols": ANCHOR_FEATURE_COLS,
            "anchor_reach_edge_feature_cols": ANCHOR_REACH_EDGE_FEATURE_COLS,
            "anchor_query_edge_feature_cols": ANCHOR_QUERY_EDGE_FEATURE_COLS,
            "knn_anchor_query": args.knn_anchor_query,
            "by_class": {c: int(ax[f"anchor_is_{c}"].sum()) for c in ANCHOR_CLASSES},
            # anchor head is the Dirichlet BC value. In WTE mode the trainer reads
            # this column and injects it (fold-standardized in target space) on a
            # dedicated value channel -- it is NOT a member of anchor_feature_cols.
            "anchor_bc_col": "head_m",
            "anchor_bc_units": "m_same_datum_as_target_wte",
        }
        log.info(
            "anchors: %d nodes, %d->reach edges, %d->query edges (k=%d)",
            len(ax),
            len(ar),
            len(aq),
            args.knn_anchor_query,
        )

    # Mode-aware target metadata: the trainer/scorer read target_mode to decide how
    # to de-standardize the model's scalar output and reconstruct DTW.
    leakage_notes = [
        "Query features exclude Janssen/Ma/coords/obs DTW (carried for scoring).",
        "Reaches carry no labels; no query->query edges.",
        "Deep datum: per-HUC6 deepest-quartile mask (each HUC6 in one HUC4=one "
        "fold) + leave-one-HUC4-out cross-fit, so held-out basins never inform "
        "their own deep prior.",
        "RELIEF_ETRM query features are exogenous target-blind rasters (terrain "
        "relief + ETRM fluxes); gridMET aridity excluded (NEGATIVE in stacker).",
        "Anchors carry DEM head + fixed BC DTW=0, never an observed well DTW; "
        "anchor_x = class/source one-hot + head_uncertainty ONLY (no head_m). "
        "Anchors are a fixed BC in all CV folds (no label to hold out).",
        "All rel-elev attrs use one shared DEM land-surface datum (well/reach/"
        "anchor), so they are translation-invariant differences.",
    ]
    if args.target == TARGET_WTE:
        target_mode = TARGET_WTE
        target_units = "m (same datum as the land-surface DEM)"
        native_prediction_col = "gnn_wte_hat_m"
        target_definition = "z_surf_well_m - mean_dtw (observed water-table elevation)"
        final_dtw_definition = "z_surf_well_m - wte_hat"
        dtw_reconstruction = "surface_elev_col - native_prediction"
        wte_features = {
            REGIONAL_WTE_COL: "cross-fit leave-one-HUC4-out IDW of observed WTE",
            DEEP_REGIONAL_WTE_COL: "cross-fit IDW of deep-well observed WTE (direct)",
            FAC_REM_WTE_COL: "joined_from_stacker_features"
            if FAC_REM_WTE_COL in query_feature_cols
            else False,
            HAND_WTE_COL: "z_surf_well - hand_m",
        }
        leakage_notes += [
            "WTE mode: target = observed water-table elevation; loss/scoring on "
            "reconstructed DTW (z_surf_well - wte_hat). |WTE err| == |DTW err|.",
            "Regional + deep WTE priors are cross-fit leave-one-HUC4-out IDW of "
            "OBSERVED WTE (head-space direct), never z_surf - dtw_prior.",
            "fac_rem_wte = z_surf - fac_rem_dtw is target-blind; NaN where FAC-REM "
            "is unavailable (NaN+indicator); only canonical_id + fac_rem_dtw_m read "
            "from the stacker table (no frozen-ConusWTE / retired-well columns).",
            "Absolute land-surface elevation (z_surf_well_m) IS a feature in WTE "
            "mode; HUC4-blocked CV is the overfit monitor (watch train/test gap).",
        ]
    else:
        target_mode = TARGET_DTW_RESIDUAL
        target_units = "m"
        native_prediction_col = "gnn_residual_hat_m"
        target_definition = f"mean_dtw - {regional_prior_col} (cross-fit IDW prior)"
        final_dtw_definition = f"{regional_prior_col} + residual_hat"
        dtw_reconstruction = "regional_prior_col + native_prediction"
        wte_features = None
        leakage_notes.insert(
            0, "Query features = hand_m + cross-fit regional IDW prior (+ deep/RELIEF)."
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
        "target_mode": target_mode,
        "target_col": target_col,
        "target_units": target_units,
        "target_definition": target_definition,
        "native_prediction_col": native_prediction_col,
        "final_dtw_definition": final_dtw_definition,
        "dtw_reconstruction": dtw_reconstruction,
        "obs_dtw_col": "mean_dtw",
        "obs_wte_col": OBS_WTE_COL,
        "surface_elev_col": SURFACE_ELEV_COL,
        "regional_prior_col": regional_prior_col,
        "wte_features": wte_features,
        "cv_fold_col": "cv_fold",
        "cv_group_col": "block_40km",
        "cv_scheme": f"HUC4-blocked, {args.folds} folds",
        "reach_feature_cols": reach_manifest["reach_feature_cols"],
        "reach_structural_nan_cols": reach_manifest["reach_structural_nan_cols"],
        "channel_edge_feature_cols": reach_manifest["channel_edge_feature_cols"],
        "query_feature_cols": query_feature_cols,
        "query_diagnostic_cols": QUERY_DIAGNOSTIC_COLS,
        "lateral_edge_feature_cols": LATERAL_EDGE_FEATURE_COLS,
        "relief_etrm_features": bool(args.relief_etrm_features),
        "relief_etrm_feature_cols": RELIEF_ETRM_FEATURE_COLS
        if args.relief_etrm_features
        else [],
        "anchors": anchor_block,
        "conductance_p": args.conductance_p,
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
        "leakage_notes": leakage_notes,
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
