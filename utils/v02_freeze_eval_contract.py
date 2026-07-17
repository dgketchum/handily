"""WP0 — freeze the Handily v0.2 evaluation contract.

Implements `notes/plans/HANDILY_V2_FIELD_MODEL_PLAN.md` §5: before any v0.2
architecture arm starts, freeze (1) the production background baseline, (2) well
identities (physical-site groups, nest siblings, cross-record duplicates), (3) panel
memberships, and (4) the numeric acceptance gates. Everything downstream (WP1-WP6)
reads these artifacts and nothing else.

Stages (all run by default; --stages to subset):
  freeze    sha256 + provenance manifest for the w25_prod OOF predictions, run
            config, score panel, bundle manifest and query nodes; small files are
            copied into <out>/frozen_baseline/ so later work cannot silently drift.
  identity  physical-site groups among the query wells (union-find over pairs
            within --site-radius-m), nest siblings (same site, distinct completion
            depths), and GWX cross-record counts at 100/300 m (the coordinate-jitter
            duplicate risk from notes/LEAKAGE_AUDIT.md L4).
  panels    per-well panel membership: distance-to-training-context, buffered
            interpolation booleans, regional dev/sacrificial HUC4 holdouts,
            source/site-disjoint, and the six mechanism panels (permanent water,
            shallow irrigated, deep/far arid, montane valley, closed basin,
            nested/collocated).
  gates     preregistered numeric acceptance gates for WP1/WP2/WP3/WP4/WP5, frozen
            before any learned run.

Outputs (--out-dir, default /data/ssd2/handily/conus/wte_gnn/v02/contract):
  frozen_manifest.json        hashes + provenance of every frozen artifact
  frozen_baseline/            copies of the small frozen artifacts
  wells_panels.parquet        one row per query node: identity + strata + panels
  panel_definitions.json      every threshold/radius/seed that defines the panels
  regional_holdout_selection.json   the held-out HUC4s (dev + sacrificial)
  acceptance_gates.json       preregistered numeric gates

The sacrificial HUC4s are selected here and marked in wells_panels.parquet, but no
v0.2 script may report metrics on them until work-package settings are frozen
(plan §5.1.5, §5.3). Downstream scripts must drop rows with
regional_holdout == 'sacrificial'.

Usage:
    uv run python utils/v02_freeze_eval_contract.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

log = logging.getLogger("v02_freeze_eval_contract")

WTE_GNN = "/data/ssd2/handily/conus/wte_gnn"
PROD_ARM = "gnn_conus_monitoring_water_gate_mirror_sigma_w25_prod"
OOF_PARQUET = f"{WTE_GNN}/{PROD_ARM}/gnn_oof_predictions.parquet"
RUN_JSON = f"{WTE_GNN}/{PROD_ARM}/gnn_run.json"
SCORE_JSON = f"{WTE_GNN}/{PROD_ARM}/conus_score_panel.json"
GRAPH_DIR = f"{WTE_GNN}/graph_conus_monitoring_water"
GWX_WELLS = "/data/ssd2/gwx/products/current/wells.geoparquet"
IRRIGATION_TIF = "/nas/handily/covariates/anthropogenic/irrigation_2017.tif"
COLOCATED_FGB = f"{WTE_GNN}/colocated_wells/colocated_cells_500m.fgb"
OUT_DIR = f"{WTE_GNN}/v02/contract"

SITE_RADIUS_M = 100.0
JITTER_RADIUS_M = 300.0
NEST_DEPTH_DIFF_M = 10.0
BUFFER_KM = (1.0, 2.5, 5.0)
MIN_REGIONAL_WELLS = 150
DEV_HOLDOUT_PER_TERCILE = 3
SACRIFICIAL_PER_TERCILE = 1
REGIONAL_SEED = 0
ARID_AI_MAX = 0.5  # gridMET aridity index = P/PET (dimensionless); <0.5 = semi-arid+
DEEP_FAR_DEPTH_M = 10.0  # NM regime boundary, plan §3.1
DEEP_FAR_WATER_KM = 2.0
MONTANE_HAF_MAX_M = 15.0  # on the valley floor/terrace
MONTANE_TPI10_QUANTILE = 0.10  # tpi_10km at/below this population quantile = montane


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# freeze
# --------------------------------------------------------------------------- #
def stage_freeze(args, out: Path) -> dict:
    """Hash + copy the production background artifacts (plan §5.3)."""
    frozen = out / "frozen_baseline"
    frozen.mkdir(parents=True, exist_ok=True)
    to_copy = {
        "oof_predictions": Path(args.oof),
        "gnn_run": Path(args.run_json),
        "score_panel": Path(args.score_json),
        "graph_manifest": Path(args.graph_dir) / "graph_manifest.json",
        "query_nodes": Path(args.graph_dir) / "query_nodes.parquet",
    }
    hash_only = {
        "lateral_edges": Path(args.graph_dir) / "lateral_edges.parquet",
        "reach_nodes": Path(args.graph_dir) / "reach_nodes.parquet",
        "gwx_wells": Path(args.gwx),
    }
    man = {"arm": PROD_ARM, "frozen_date": str(date.today()), "artifacts": {}}
    for key, src in {**to_copy, **hash_only}.items():
        entry = {
            "source_path": str(src),
            "sha256": sha256(src),
            "bytes": src.stat().st_size,
        }
        if key in to_copy:
            dst = frozen / src.name
            shutil.copy2(src, dst)
            entry["frozen_copy"] = str(dst)
        man["artifacts"][key] = entry
    (out / "frozen_manifest.json").write_text(json.dumps(man, indent=2))
    log.info("frozen_manifest.json written (%d artifacts)", len(man["artifacts"]))
    return man


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def union_find_sites(xy: np.ndarray, radius_m: float) -> np.ndarray:
    """Cluster wells into physical-site groups: any pair within radius_m shares a
    site. Returns an integer site id per row (transitive closure via union-find)."""
    n = len(xy)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = cKDTree(xy)
    for i, j in tree.query_pairs(radius_m):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    _, site_id = np.unique(roots, return_inverse=True)
    return site_id


def stage_identity(args, wells: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Physical-site groups, nest siblings, and GWX cross-record duplicate stats."""
    xy = wells[["x5070", "y5070"]].to_numpy()
    wells["site_id"] = union_find_sites(xy, args.site_radius_m)
    counts = wells.groupby("site_id")["site_id"].transform("size")
    wells["site_n_query_wells"] = counts.astype(int)

    # nest siblings: same physical site, materially different completion depth.
    depth = wells["well_depth"].to_numpy(float)
    screen = wells["screen_bottom"].to_numpy(float)
    comp = np.where(np.isfinite(screen), screen, depth)
    nest = np.zeros(len(wells), bool)
    for _, g in wells[counts > 1].groupby("site_id"):
        c = comp[g.index.to_numpy()]
        c = c[np.isfinite(c)]
        if len(c) >= 2 and (c.max() - c.min()) >= args.nest_depth_diff_m:
            nest[g.index.to_numpy()] = True
    wells["is_nest_sibling"] = nest

    # GWX cross-record duplicates (coordinate-jitter risk, LEAKAGE_AUDIT L4):
    # how many *records* (any source, any class) sit within 100/300 m of each
    # query well. Same-site records under different canonical ids are the known
    # 88.5% failure mode for id-based exclusion.
    import pyarrow.parquet as pq
    from pyproj import Transformer

    gt = pq.read_table(args.gwx, columns=["longitude", "latitude", "canonical_id"])
    lon = gt["longitude"].to_numpy(zero_copy_only=False)
    lat = gt["latitude"].to_numpy(zero_copy_only=False)
    tr = Transformer.from_crs(4326, 5070, always_xy=True)
    gx, gy = tr.transform(lon, lat)
    gxy = np.column_stack([gx, gy])
    gxy = gxy[np.isfinite(gxy).all(axis=1)]
    gtree = cKDTree(gxy)
    wells["n_gwx_records_100m"] = gtree.query_ball_point(
        xy, args.site_radius_m, return_length=True
    )
    wells["n_gwx_records_300m"] = gtree.query_ball_point(
        xy, args.jitter_radius_m, return_length=True
    )

    n_multi = int((wells["site_n_query_wells"] > 1).sum())
    diag = {
        "n_query_wells": int(len(wells)),
        "site_radius_m": args.site_radius_m,
        "n_sites": int(wells["site_id"].nunique()),
        "n_wells_multi_well_site": n_multi,
        "n_nest_siblings": int(nest.sum()),
        "nest_depth_diff_m": args.nest_depth_diff_m,
        "frac_with_gwx_record_100m_beyond_self": round(
            float((wells["n_gwx_records_100m"] > 1).mean()), 4
        ),
        "frac_gwx_gap_100_300m": round(
            float((wells["n_gwx_records_300m"] > wells["n_gwx_records_100m"]).mean()),
            4,
        ),
    }
    log.info("identity: %s", json.dumps(diag))
    return wells, diag


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def nearest_constrained_km(
    xy: np.ndarray,
    require_other_fold: bool,
    fold: np.ndarray,
    site: np.ndarray,
    k0: int = 64,
) -> np.ndarray:
    """Distance (km) from each well to the nearest OTHER well that is (a) at a
    different physical site and (b) if require_other_fold, in a different fold
    (i.e. a well the fold's training run actually saw)."""
    n = len(xy)
    tree = cKDTree(xy)
    out = np.full(n, np.nan)
    k = min(k0, n)
    d, idx = tree.query(xy, k=k)
    for i in range(n):
        hit = np.nan
        for dist, j in zip(d[i], idx[i]):
            if j == i or site[j] == site[i]:
                continue
            if require_other_fold and fold[j] == fold[i]:
                continue
            hit = dist
            break
        out[i] = hit
    missing = np.isnan(out)
    if missing.any():  # fall back to a full scan for the rare deep-rural cases
        for i in np.where(missing)[0]:
            mask = (site != site[i]) & (np.arange(n) != i)
            if require_other_fold:
                mask &= fold != fold[i]
            if mask.any():
                dd = np.sqrt(((xy[mask] - xy[i]) ** 2).sum(axis=1))
                out[i] = dd.min()
    return out / 1000.0


def select_regional_holdouts(wells: pd.DataFrame, args) -> dict:
    """Pick dev-regional and sacrificial HUC4s, stratified by aridity terciles."""
    g = wells.groupby("huc4").agg(
        n=("canonical_id", "size"), ai=("aridity_index", "median")
    )
    cand = g[g["n"] >= args.min_regional_wells].dropna(subset=["ai"])
    terciles = np.quantile(cand["ai"], [1 / 3, 2 / 3])
    rng = np.random.default_rng(args.regional_seed)
    sel = {"dev": [], "sacrificial": []}
    for t in range(3):
        lo = -np.inf if t == 0 else terciles[t - 1]
        hi = np.inf if t == 2 else terciles[t]
        pool = cand[(cand["ai"] > lo) & (cand["ai"] <= hi)].index.to_numpy()
        pool = rng.permutation(np.sort(pool))
        take = pool[: args.dev_per_tercile + args.sacrificial_per_tercile]
        sel["dev"] += list(take[: args.dev_per_tercile])
        sel["sacrificial"] += list(take[args.dev_per_tercile :])
    return {
        "seed": args.regional_seed,
        "min_wells_per_huc4": args.min_regional_wells,
        "aridity_tercile_edges": [round(float(x), 4) for x in terciles],
        "dev_huc4": sorted(sel["dev"]),
        "sacrificial_huc4": sorted(sel["sacrificial"]),
        "n_candidate_huc4": int(len(cand)),
        "note": (
            "dev_huc4 = regional-separation development panel (plan §5.1.3); "
            "sacrificial_huc4 = independent geography, LOCKED until work-package "
            "settings are frozen (plan §5.1.5). Regional-separation evaluation "
            "requires retraining without these units; membership is frozen here "
            "so every arm holds out the same geography."
        ),
    }


def stage_panels(args, wells: pd.DataFrame, water: pd.DataFrame):
    xy = wells[["x5070", "y5070"]].to_numpy()
    fold = wells["cv_fold"].to_numpy()
    site = wells["site_id"].to_numpy()

    # distance to nearest well the fold's training run saw (other fold, other site)
    wells["dist_train_km"] = nearest_constrained_km(xy, True, fold, site)
    # distance to nearest other-site well regardless of fold (analysis-time context)
    wells["dist_anyother_km"] = nearest_constrained_km(xy, False, fold, site)
    for r in args.buffer_km:
        wells[f"panel_buffered_{str(r).replace('.', 'p')}km"] = (
            wells["dist_train_km"] >= r
        )

    # regional holdouts
    sel = select_regional_holdouts(wells, args)
    wells["regional_holdout"] = ""
    wells.loc[wells["huc4"].isin(sel["dev_huc4"]), "regional_holdout"] = "dev"
    wells.loc[wells["huc4"].isin(sel["sacrificial_huc4"]), "regional_holdout"] = (
        "sacrificial"
    )

    # source/site separation: non-NWIS wells with no NWIS well inside the jitter
    # radius (site-disjoint from the dominant source, plan §5.1.4)
    nwis_xy = xy[wells["is_nwis"].to_numpy(bool)]
    if len(nwis_xy):
        dn, _ = cKDTree(nwis_xy).query(xy)
    else:
        dn = np.full(len(wells), np.inf)
    wells["dist_nearest_nwis_km"] = dn / 1000.0
    wells["panel_source_disjoint"] = (~wells["is_nwis"].astype(bool)) & (
        dn > args.jitter_radius_m
    )

    # mechanism panels (plan §5.1.6)
    obs = wells["mean_dtw"].to_numpy(float)
    dist_water_m = np.expm1(wells["log1p_dist_perm_water_m"].to_numpy(float))
    ai = wells["aridity_index"].to_numpy(float)

    import rasterio

    with rasterio.open(args.irrigation_tif) as src:
        from pyproj import Transformer

        if src.crs and src.crs.to_epsg() == 5070:
            px, py = wells["x5070"].to_numpy(), wells["y5070"].to_numpy()
        else:
            tr = Transformer.from_crs(5070, src.crs, always_xy=True)
            px, py = tr.transform(wells["x5070"].to_numpy(), wells["y5070"].to_numpy())
        vals = np.array(
            [v[0] for v in src.sample(np.column_stack([px, py]))], dtype=float
        )
        if src.nodata is not None:
            vals[vals == src.nodata] = 0.0
    wells["irrigated_2017"] = vals > 0

    tpi10 = wells["tpi_10km"].to_numpy(float)
    tpi10_thr = float(np.nanquantile(tpi10, args.montane_tpi10_q))
    haf = wells["haf_500m"].to_numpy(float)

    wells["mech_permanent_water"] = False
    wells["mech_shallow_irrigated"] = wells["irrigated_2017"] & (obs < 10.0)
    wells["mech_deep_far_arid"] = (
        (obs >= args.deep_far_depth_m)
        & (dist_water_m > args.deep_far_water_km * 1000.0)
        & (ai < args.arid_ai_max)
    )
    wells["mech_montane_valley"] = (tpi10 <= tpi10_thr) & (
        haf <= args.montane_haf_max_m
    )
    wells["mech_closed_basin"] = wells["huc2"].astype(str) == "16"

    # nested/collocated: multi-well site OR inside a collocated 500 m cell
    import geopandas as gpd

    cells = gpd.read_file(args.colocated_fgb)
    pts = gpd.GeoDataFrame(
        wells[["canonical_id"]],
        geometry=gpd.points_from_xy(wells["x5070"], wells["y5070"]),
        crs="EPSG:5070",
    )
    joined = gpd.sjoin(pts, cells.to_crs("EPSG:5070"), how="left", predicate="within")
    in_cell = ~joined.groupby(level=0)["index_right"].first().isna()
    wells["in_colocated_cell_500m"] = in_cell.reindex(wells.index, fill_value=False)
    wells["mech_nested_collocated"] = (wells["site_n_query_wells"] > 1) | wells[
        "in_colocated_cell_500m"
    ]

    # water pseudo rows: permanent-water mechanism panel only
    water = water.copy()
    water["mech_permanent_water"] = True

    defs = {
        "site_radius_m": args.site_radius_m,
        "jitter_radius_m": args.jitter_radius_m,
        "nest_depth_diff_m": args.nest_depth_diff_m,
        "buffer_km": list(args.buffer_km),
        "deep_far_depth_m": args.deep_far_depth_m,
        "deep_far_water_km": args.deep_far_water_km,
        "arid_ai_max": args.arid_ai_max,
        "aridity_index_def": "gridMET P/PET, dimensionless; lower = more arid",
        "montane_tpi10_quantile": args.montane_tpi10_q,
        "montane_tpi10_threshold_m": round(tpi10_thr, 3),
        "montane_haf_max_m": args.montane_haf_max_m,
        "irrigation_source": args.irrigation_tif,
        "colocated_source": args.colocated_fgb,
        "sacrificial_locked": True,
        "counts": {},
    }
    panel_cols = [c for c in wells.columns if c.startswith(("panel_", "mech_"))]
    for c in panel_cols:
        defs["counts"][c] = int(wells[c].sum())
    defs["counts"]["mech_permanent_water"] = int(len(water))
    defs["counts"]["regional_holdout_dev"] = int(
        (wells["regional_holdout"] == "dev").sum()
    )
    defs["counts"]["regional_holdout_sacrificial"] = int(
        (wells["regional_holdout"] == "sacrificial").sum()
    )
    return wells, water, sel, defs


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def stage_gates(out: Path) -> dict:
    """Preregistered numeric acceptance gates, copied from the plan and frozen
    before any learned v0.2 run (plan §6.6, §7.5, §8.3, §9.6, §10.3)."""
    gates = {
        "frozen_date": str(date.today()),
        "frozen_before_any_learned_v02_run": True,
        "wp1_conditional_assimilation": {
            "min_relative_improvement": 0.10,
            "improvement_metric": "MAD or RMSE vs frozen background, same rows",
            "min_qualifying_support_bands": 2,
            "support_band_range_km": [2, 50],
            "no_material_regression_panels": ["buffered", "regional_dev", "shallow"],
            "regression_tolerance_frac": 0.02,
            "empty_context_identity_tol_m": 1e-6,
            "interval_coverage_tolerance": 0.05,
            "require_error_monotone_in_scale": True,
            "no_headline_from_assimilated_or_sub_km_targets": True,
            "ship_kriging_if_learned_does_not_beat_it": True,
        },
        "wp2_two_surface": {
            "component_assignment_beats_one_surface_on_nested_wells": True,
            "min_deep_band_error_reduction": 0.15,
            "deep_band": "30+m on separated panel",
            "no_global_component_collapse": True,
            "shallow_skill_regression_tolerance_frac": 0.02,
        },
        "wp3_common_epoch": {
            "replace_mean_dtw_only_if": [
                "reduced spatial residual structure",
                "improved held-out WTE/DTW in pumping-sensitive regions",
                "no degradation in stable monitoring regions",
            ]
        },
        "wp4_regional_operator": {
            "evaluate_background_with_nearby_obs_removed": True,
            "min_deep_error_reduction": 0.15,
            "bands": ["10-30m", "30+m"],
            "max_shallow_giveback_frac": 0.02,
            "must_beat": "transparent differentiable-solver or kriging/background control",
            "three_pilot_failure_stops_conus_buildout": True,
        },
        "wp5_ordinal_shallow": {
            "require_pr_auc_and_brier_improvement": True,
            "vs_controls": [
                "logistic/boosted-tree on frozen features",
                "FAC threshold calls",
                "current GNN point predictions",
                "sigma-selective GNN calls",
                "class-prevalence calibration",
            ],
            "no_threshold_selection_on_final_geography": True,
        },
        "tuning_discipline": {
            "background_baseline": PROD_ARM,
            "inner_split_for_selection": "block_40km within-fold validation blocks",
            "sacrificial_geography_opens": "once, after work-package settings frozen",
            "ma_janssen_are_benchmarks_only": True,
        },
    }
    (out / "acceptance_gates.json").write_text(json.dumps(gates, indent=2))
    return gates


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def load_query_nodes(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    qn = pd.read_parquet(Path(args.graph_dir) / "query_nodes.parquet")
    real = qn[~qn["is_water_pseudo"].astype(bool)].reset_index(drop=True)
    water = qn[qn["is_water_pseudo"].astype(bool)].reset_index(drop=True)
    if real["canonical_id"].duplicated().any():
        raise ValueError("canonical_id not unique among real query wells")
    return real, water


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", default=OOF_PARQUET)
    ap.add_argument("--run-json", default=RUN_JSON)
    ap.add_argument("--score-json", default=SCORE_JSON)
    ap.add_argument("--graph-dir", default=GRAPH_DIR)
    ap.add_argument("--gwx", default=GWX_WELLS)
    ap.add_argument("--irrigation-tif", default=IRRIGATION_TIF)
    ap.add_argument("--colocated-fgb", default=COLOCATED_FGB)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--site-radius-m", type=float, default=SITE_RADIUS_M)
    ap.add_argument("--jitter-radius-m", type=float, default=JITTER_RADIUS_M)
    ap.add_argument("--nest-depth-diff-m", type=float, default=NEST_DEPTH_DIFF_M)
    ap.add_argument("--buffer-km", type=float, nargs="+", default=list(BUFFER_KM))
    ap.add_argument("--min-regional-wells", type=int, default=MIN_REGIONAL_WELLS)
    ap.add_argument("--dev-per-tercile", type=int, default=DEV_HOLDOUT_PER_TERCILE)
    ap.add_argument(
        "--sacrificial-per-tercile", type=int, default=SACRIFICIAL_PER_TERCILE
    )
    ap.add_argument("--regional-seed", type=int, default=REGIONAL_SEED)
    ap.add_argument("--arid-ai-max", type=float, default=ARID_AI_MAX)
    ap.add_argument("--deep-far-depth-m", type=float, default=DEEP_FAR_DEPTH_M)
    ap.add_argument("--deep-far-water-km", type=float, default=DEEP_FAR_WATER_KM)
    ap.add_argument("--montane-haf-max-m", type=float, default=MONTANE_HAF_MAX_M)
    ap.add_argument("--montane-tpi10-q", type=float, default=MONTANE_TPI10_QUANTILE)
    ap.add_argument(
        "--stages",
        nargs="+",
        default=["freeze", "identity", "panels", "gates"],
        choices=["freeze", "identity", "panels", "gates"],
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "freeze" in args.stages:
        stage_freeze(args, out)

    if "identity" in args.stages or "panels" in args.stages:
        wells, water = load_query_nodes(args)
        wells, ident_diag = stage_identity(args, wells)
        if "panels" in args.stages:
            wells, water, sel, defs = stage_panels(args, wells, water)
            defs["identity"] = ident_diag
            (out / "panel_definitions.json").write_text(json.dumps(defs, indent=2))
            (out / "regional_holdout_selection.json").write_text(
                json.dumps(sel, indent=2)
            )
            keep = [
                "canonical_id",
                "query_node_idx",
                "source",
                "is_nwis",
                "x5070",
                "y5070",
                "huc2",
                "huc4",
                "huc8",
                "cv_fold",
                "cv_unit",
                "block_40km",
                "is_water_pseudo",
                "mean_dtw",
                "wte_obs_m",
                "z_surf_well_m",
                "well_class",
                "confinement_class",
                "well_depth",
                "screen_bottom",
                "obs_count",
                "por_start",
                "por_end",
                "aridity_index",
                "slope_deg",
                "tri_100m",
                "dist_to_stream_m",
                "log1p_dist_perm_water_m",
                "tpi_10km",
                "haf_500m",
                "site_id",
                "site_n_query_wells",
                "is_nest_sibling",
                "n_gwx_records_100m",
                "n_gwx_records_300m",
                "dist_train_km",
                "dist_anyother_km",
                "dist_nearest_nwis_km",
                "regional_holdout",
                "irrigated_2017",
                "in_colocated_cell_500m",
            ]
            keep += [c for c in wells.columns if c.startswith(("panel_", "mech_"))]
            wtab = wells[[c for c in keep if c in wells.columns]].copy()
            wat = water[[c for c in keep if c in water.columns]].copy()
            panels = pd.concat([wtab, wat], ignore_index=True)
            for c in panels.columns:
                if c.startswith(("panel_", "mech_", "irrigated", "in_colocated")):
                    panels[c] = panels[c].fillna(False).astype(bool)
            panels.to_parquet(out / "wells_panels.parquet")
            log.info(
                "wells_panels.parquet: %d rows (%d wells + %d water)",
                len(panels),
                len(wtab),
                len(wat),
            )

    if "gates" in args.stages:
        stage_gates(out)
    log.info("WP0 artifacts written to %s", out)


if __name__ == "__main__":
    main()
