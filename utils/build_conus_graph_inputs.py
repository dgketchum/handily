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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stacker_features import (  # noqa: E402
    ACCUM,
    COARSE_SURFACE,
    DIST_STREAM,
    ETRM_ETA,
    ETRM_RECHARGE,
    ETRM_RUNOFF,
    GRIDMET_AI,
    GRIDMET_P,
    PERM_LOGK,
    SED_THICKNESS,
    SLOPE,
    TRI,
    sample_coarse,
)
from fac_rem_registry import sample_fac_rem, sample_str_top2_wte  # noqa: E402

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
# wte_residual (the wall-to-wall keystone): predict the head residual above a smooth
# regional WTE prior R, target = obs_wte - R (a small, well-conditioned residual).
# Reconstruct dtw = z_surf - (R + resid_hat) = (z_surf - R) - resid_hat, so the DTW
# base is (z_surf - R) and reconstruction shares the WTE mode's `base - native` form.
# All head-space features are anomalies-from-R (translation-invariant); z_surf is the
# reconstruction datum only, never a feature; HAND features are dropped.
TARGET_WTE_RESIDUAL = "wte_residual"
TARGET_CHOICES = (TARGET_DTW_RESIDUAL, TARGET_WTE, TARGET_WTE_RESIDUAL)
SURFACE_ELEV_COL = "z_surf_well_m"  # DEM land-surface elev at the well (datum)
OBS_WTE_COL = "wte_obs_m"  # observed head = z_surf_well - mean_dtw
REGIONAL_WTE_COL = "regional_wte_idw_oof_m"  # cross-fit IDW of observed WTE (= R)
DEEP_REGIONAL_WTE_COL = "deep_regional_wte_idw_oof_m"  # cross-fit IDW of deep WTE
FAC_REM_WTE_COL = "fac_rem_wte_m"  # z_surf - fac_rem_dtw (legit local DTW product)
HAND_WTE_COL = "hand_wte_m"  # z_surf - hand_m (head-like FIM-HAND feature)
# wte_residual columns: the DTW base (z_surf - R), the residual target (obs_wte - R),
# and the two head-space prior anomalies-from-R (the core translation-invariant signal).
WTE_RESID_BASE_COL = "wte_resid_base_m"  # z_surf - R; dtw = base - resid_hat
WTE_RESIDUAL_TARGET_COL = "wte_residual_m"  # obs_wte - R
FAC_REM_WTE_ANOM_COL = "fac_rem_wte_anom_m"  # (z_surf - fac_rem_dtw) - R
DEEP_REGIONAL_WTE_ANOM_COL = "deep_regional_wte_anom_m"  # deep_wte_idw - R
# --residual-base fac_rem only: R = FAC water surface, so str_top2 becomes the
# regional-context anomaly feature (str_top2 WTE - R) + a diagnostic surface column.
STR_TOP2_WTE_COL = "str_top2_wte_m"  # the str_top2 regional WTE surface (diagnostic)
STR_TOP2_WTE_ANOM_COL = "str_top2_wte_anom_m"  # str_top2 WTE - R (regional anomaly)
# --ensemble-member-features only: the plain (vw=0, no relief-lift) well-IDW WTE member
# exposed as an anomaly-over-R feature. In relief_idw mode R IS the vw=100 member, so
# (simple - R) IS the relief-lift disagreement -- strongly negative where relief
# over-mounds a bench, which is the missing bench signal.
SIMPLE_IDW_WTE_COL = "simple_idw_wte_m"  # plain vw=0 well-IDW WTE surface (diagnostic)
SIMPLE_IDW_WTE_ANOM_COL = "simple_idw_wte_anom_m"  # simple(vw=0) WTE - R
# gridMET climate (EPSG:4326 -> sampled at lon/lat). Kept in wte_residual mode: a
# NEGATIVE result in the pilot stacker does not mean it cannot help the GNN.
CLIMATE_FEATURE_COLS = ["aridity_index", "mean_annual_precip_mm"]
# Evidence-feature bank (the "feed the evidence, not FAC's water-surface answer" path,
# behind --evidence-features). Observable, target-blind signals -- MODIS summer NDVI +
# summer-minus-winter amplitude (phreatophyte greenness, a shallow-GW proxy) and
# proximity to GSW-evidenced surface water -- added to the wte_residual query bank
# WITHOUT the un-calibrated FAC-REM head estimate. Tests whether the GNN can build the
# shallow correction from raw evidence rather than riding FAC's own rough water-surface
# solve (the "shortcut to a mediocre prediction"). The required fac-skip anchor on the
# FAC target-estimate is unchanged -- this is a feature change, not a skip change.
NDVI_JJA = "/data/ssd2/handily/conus/covariates/modis_ndvi_jja_mean.tif"
NDVI_DJF = "/data/ssd2/handily/conus/covariates/modis_ndvi_djf_mean.tif"
GSW_OCC_DIR = Path(
    "/nas/hydrography/gsw/occurrence"
)  # JRC GSW occurrence (0-100, 4326)
EVIDENCE_FEATURE_COLS = [
    "well_ndvi_jja",  # summer greenness at the well
    "well_ndvi_amp",  # jja - djf (phreatophyte amplitude: green in summer, dry in winter)
    "dist_to_wet_reach_m",  # distance to nearest stream reach with GSW water evidence
    "log1p_dist_to_wet_reach_m",
]
# IrrMapper irrigation frequency (behind --irrigation-features): % of 2015-2024 years
# a 30 m pixel was classified irrigated. Mechanism: sustained irrigation recharges a
# local shallow mound the terrain cannot see (flood-irrigated hay valleys), and the
# GW-subsidy deliverable is scored exactly on these lands -- where the GNN is
# currently ~2 m too deep. Target-blind (satellite land use, no observed WTE).
IRRMAPPER_FREQ_RASTERS = {  # per-state EPSG:5070 uint8, nodata None (0 = never irrigated)
    "MT": "/nas/irrmapper/tif_exports/conus_freq/irrmapper_freq_MT_2015_2024_30m_5070.tif",
    "NM": "/nas/irrmapper/tif_exports/conus_freq/irrmapper_freq_NM_2015_2024_30m_5070.tif",
    "NV": "/nas/irrmapper/tif_exports/conus_freq/irrmapper_freq_NV_2015_2024_30m_5070.tif",
}
IRRIGATION_FEATURE_COLS = ["irr_freq_pct"]
# Well model features (leak-free). Everything else on the query node is carried
# for scoring/diagnostics only.
QUERY_FEATURE_COLS = ["hand_m", "regional_idw_dtw_oof_m"]
# v2: the proven RELIEF_ETRM query bank (terrain relief discriminators + ETRM water-
# balance fluxes -- the only covariate family that moved the deep tail). Exogenous /
# target-blind. Appended to query features behind --relief-etrm-features. gridMET
# aridity is deliberately excluded (it was NEGATIVE in the stacker sweep).
# perm_logk_m2 + sediment_thickness_m complete the Toth R/K & water-table-ratio inputs
# (recharge/dist_to_stream/hand already here) so the GNN can form the regime interaction
# itself -- no explicit WTR product (the hand gate was a no-go; see notes/WTR_FINDING.md).
RELIEF_ETRM_FEATURE_COLS = [
    "slope_deg",
    "tri_100m",
    "dist_to_stream_m",
    "log_drainage_area",
    "elev_above_coarse_m",
    "etrm_recharge_mm",
    "etrm_eta_mm",
    "etrm_runoff_mm",
    "perm_logk_m2",
    "sediment_thickness_m",
]
# Multi-scale terrain-position family (built by build_terrain_covariates.py on the
# canonical 100 m grid, EPSG:5070). Behind --terrain-multiscale-features. The residual
# over R is organized by terrain position (valley floor vs terrace/bench) but the wired
# elev_above_coarse_m proxy is ~dead (corr ~0); height-above-local-floor (z - focal_min)
# is the proven lever (GBM R2 0.06->0.13 on obs_wte-R). TPI (DevFromMeanElev) adds the
# terrace-vs-valley position axis; multi-scale TWI (slope-at-scale, true SCA retained)
# adds convergence/wetness. 4 scales each (500 m/2 km/5 km/10 km). All target-blind +
# translation-invariant (relative-elevation / dimensionless), so RGA-safe.
TERRAIN_DIR = "/nas/handily/covariates/terrain"
_TERRAIN_SCALES = ("500m", "2km", "5km", "10km")
TERRAIN_MULTISCALE_FEATURE_COLS = (
    [f"haf_{s}" for s in _TERRAIN_SCALES]  # height above local floor (m)
    + [
        f"tpi_{s}" for s in _TERRAIN_SCALES
    ]  # elevation deviation from mean (DevFromMean)
    + [f"twi_{s}" for s in _TERRAIN_SCALES]  # multi-scale wetness index
)
TERRAIN_MULTISCALE_RASTERS = {
    c: f"{TERRAIN_DIR}/{c}.tif" for c in TERRAIN_MULTISCALE_FEATURE_COLS
}
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
    "hand_m",  # carried for the scorer's hand_cal benchmark in EVERY mode, not as a
    # model feature -- head-space modes drop HAND from features but still score vs it.
    "regional_idw_dtw_oof_m",
    "regional_deep_idw_dtw_oof_m",
    "cv_fold",
    "cv_unit",
    "block_40km",
]
# GWX observation metadata carried on query nodes (diagnostics, NEVER features):
# period-of-record, observation count, and construction depths. These feed the
# observation-model / label-noise analyses and the contemporary-vs-pre-development
# split downstream. Missing values stay NaN -- metadata, not model inputs.
OBS_METADATA_COLS = [
    "por_start",
    "por_end",
    "obs_count",
    "well_depth",
    "screen_bottom",
    "head_above_screen",
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
# --ds-datum-features (item 1): the regional-HAND-along-flowpath query columns -- the
# elevation drop / along-network distance / order-jump from a well to its down-gradient
# high-order (regional discharge) channel. Relative-elevation + topology only ->
# target-blind + translation-invariant (RGA-safe). ds_datum_drop_m is the regional HAND.
DS_DATUM_FEATURE_COLS = [
    "ds_datum_drop_m",  # z_surf_well - datum_elev (regional HAND, m)
    "log1p_ds_datum_dist_m",  # log1p(rank0_lateral_dist + along-network dist to datum)
    "ds_datum_order_rel",  # datum_order - attached-reach streamorde
    "ds_datum_missing",  # 0/1 the downstream walk dead-ended (no datum reachable)
]
# --mainstem-read (item 2): one query->downstream-datum edge per well whose datum exists,
# letting the query attend to the datum reach's LEARNED state (its seed evidence,
# drainage, channel context) -- fixes the med-7/p90-28-hop receptive-field gap without
# deepening the channel stack. All relative / dimensionless / topological -> leak-free.
MS_EDGE_FEATURE_COLS = [
    "rel_elev_query_datum_m",  # z_surf_well - datum_elev
    "log1p_ds_datum_dist_m",  # log1p(rank0_lateral_dist + along-network dist to datum)
    "datum_order_rel",  # datum_order - basin_max_order (<= 0)
    "datum_is_self",  # attached reach is already a datum (0/1)
    "log1p_datum_da_km2",  # log1p(totdasqkm at the datum reach)
]
# --portfolio-read (Phase 6B): per-query typed read edges DIRECT to <=4 heterogeneous
# reference reaches (supersedes the single-purpose mainstem read). All four site types are
# REACHES (no new node types); a missing site is simply an absent edge the trainer's
# segment-softmax renormalizes over (the structural fix for the wet-prop missingness killer).
PORTFOLIO_SITE_TYPES = ["ds_datum", "up_head", "wet", "ho_any"]
PORTFOLIO_EDGE_FEATURE_COLS = [
    "rel_elev_query_site_m",  # well land-surface - site reach elevation (load-bearing)
    "log1p_site_dist_m",  # network sites: rank0-lateral + along-network; ho_any: Euclidean
    "dist_is_network",  # 1 for ds_datum/up_head/wet, 0 for ho_any (Euclidean)
    "site_order_rel",  # site streamorde - query attached-reach basin_max_order
    "log1p_site_da_km2",  # log1p(totdasqkm at the site reach)
    "type_ds_datum",  # one-hot site-type indicators (mutually exclusive per edge)
    "type_up_head",
    "type_wet",
    "type_ho_any",
]
# indicator query features -- distinguish a genuinely-absent site (no edge) from a zero read.
PORTFOLIO_MISSING_COLS = [f"portfolio_missing_{t}" for t in PORTFOLIO_SITE_TYPES]
# --wet-propagation-features (item 5): REACH-SIDE, NETWORK-METRIC surface-water evidence
# (distinct from the rejected query-side + Euclidean evidence bank). A per-reach GSW wet
# flag is propagated along the FAC channel graph so every reach -- and through the convs
# every well -- carries distance-to-permanent-water in NETWORK metric (the physically
# right metric for GW connection to losing/gaining channels; dist-to-water != dist-to-
# drainage per the R-eval). Appended to the bundle manifest's reach_feature_cols; the
# trainer reads reach cols generically -> zero trainer changes.
WET_PROP_REACH_FEATURE_COLS = [
    "wet_reach",  # 0/1: reach rep-point on GSW occurrence >= threshold
    "log1p_net_dist_wet_m",  # along-network dist to nearest wet reach (any direction)
    "log1p_ds_dist_wet_m",  # along-network dist to nearest wet reach DOWNSTREAM only
    "upstream_wet_fraction",  # length-weighted wet / total upstream channel length
    "no_wet_in_component",  # 0/1: reach's channel component has zero wet reaches
]
# §5.2 gate: the rank-0 query projection of two reach-side features, carried on the query
# node (NOT model features) so the Phase-0a GBM gate can probe them tabularly.
WET_PROP_QUERY_GATE_COLS = ["q_log1p_net_dist_wet_m", "q_upstream_wet_fraction"]
# --reach-covariate-features (Phase 6A.1): a target-blind covariate bank sampled at each
# reach's flowline rep-point (cx/cy, EPSG:5070), giving the channel node LOCAL terrain/flux/
# geology/greenness/surface-water context. haf_* are EXCLUDED (height-above-local-floor is
# ~0 AT a reach by construction). Materialized into reach_nodes; the trainer reads reach cols
# from the manifest -> zero trainer change. The r_ prefix avoids query-col collisions.
REACH_COVARIATE_FEATURE_COLS = [
    "r_tpi_2km",  # terrace-vs-valley position at the floor (2 km)
    "r_tpi_10km",  # regional valley size/confinement (10 km)
    "r_twi_2km",  # moisture convergence at the reach
    "r_etrm_recharge_mm",  # local water-balance flux regime
    "r_etrm_eta_mm",
    "r_etrm_runoff_mm",
    "r_aridity_index",  # gridMET climate (4326 -> lon/lat)
    "r_precip_mm",
    "r_perm_logk_m2",  # GLHYMPS log10(k m^2) (/100 decode) + Pelletier sediment
    "r_sediment_thickness_m",
    "r_ndvi_jja",  # phreatophyte greenness ON the channel
    "r_ndvi_amp",  # jja - djf amplitude
    "r_gsw_occ",  # continuous GSW occurrence (not just the wet flag)
]
# --upstream-accumulation-features (Phase 6A.2): length-weighted upstream-catchment means of
# the 6A.1 locals (upstream recharge is the physical driver of the table AT the reach) + the
# ONE surviving Phase-5 column upstream_wet_fraction (always finite: 0.0 on dry catchments).
# The Phase-5 NaN-distance columns are DELIBERATELY excluded (adjudicated missingness killer;
# nearest-wet distance returns as the missingness-robust wet READ EDGE in 6B). Requires
# --reach-covariate-features (upstream means accumulate the r_* locals).
UPSTREAM_ACCUM_FEATURE_COLS = [
    "upstream_recharge_mm",
    "upstream_precip_mm",
    "upstream_ndvi_jja",
    "upstream_wet_fraction",
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


def assign_folds(unit: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Blocked CV: each whole spatial unit assigned to one fold (round-robin on a
    seeded shuffle), so train/test never share a unit. ``unit`` is a HUC12 code
    (HUC12-blocked CV, the default) or a geometric block id (the national fallback);
    a held-out unit's NEIGHBOURS stay in train, which is the between-wells
    interpolation regime the wall-to-wall product is deployed in.
    """
    uh = np.array(sorted(pd.unique(unit)))
    rng = np.random.RandomState(seed)
    fold_of = {h: i % folds for i, h in enumerate(rng.permutation(uh))}
    return np.array([fold_of[h] for h in unit], dtype="int64")


def spatial_block_ids(x: np.ndarray, y: np.ndarray, block_km: float) -> np.ndarray:
    """Geometric square-block ids at ``block_km`` scale on the EPSG:5070 grid.

    The national CV fallback (and the source of the synthetic unit for the rare
    well that matches no HUC12 polygon): a ~12 km block is the HUC12 areal scale
    (mean HUC12 ~ 100-250 km2, ~10-15 km across).
    """
    s = block_km * 1000.0
    bx = np.floor(x / s).astype("int64")
    by = np.floor(y / s).astype("int64")
    return np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))


def huc12_units(
    x: np.ndarray, y: np.ndarray, huc12_path: str, fallback_block_km: float
) -> np.ndarray:
    """HUC12 code per well by point-in-polygon (EPSG:5070), for HUC12-blocked CV.

    Wells matching no HUC12 polygon (rare border/coastal points) fall back to a
    geometric block id (``blk_<bx>_<by>``) so no well is dropped; the fallback
    count is logged, never silent.
    """
    polys = gpd.read_parquet(huc12_path)[["huc12", "geometry"]]
    if polys.crs is None or polys.crs.to_epsg() != 5070:
        raise SystemExit(f"HUC12 polys not EPSG:5070: {huc12_path} (crs={polys.crs})")
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=5070)
    joined = gpd.sjoin(pts, polys, predicate="within", how="left")
    # a point on a shared polygon edge can match >1 unit; keep the first.
    joined = joined[~joined.index.duplicated(keep="first")].sort_index()
    units = joined["huc12"].to_numpy(dtype=object).copy()
    miss = pd.isna(units)
    if miss.any():
        idx = np.where(miss)[0]
        block = spatial_block_ids(x[idx], y[idx], fallback_block_km)
        units[idx] = np.char.add("blk_", block)
        log.warning(
            "HUC12 join: %d/%d wells unmatched -> geometric-block fallback",
            int(miss.sum()),
            len(units),
        )
    return units.astype(str)


def _relief_coords(xy: np.ndarray, z: np.ndarray | None, vw: float) -> np.ndarray:
    """Coordinates for the IDW KD-tree, optionally relief-lifted into (x, y, vw*z).

    With ``vw>0`` and a land-surface elevation ``z``, neighbour selection AND the
    inverse-distance weights run in lifted space, so only wells at SIMILAR ground
    elevation inform a cell -- the fix for the high-relief WTE smear (a valley head
    bleeding onto an upland well; ``build_wte_idw_grid.py``, vw=100 cuts the
    leave-fold-out WTE MAD 8.45->6.18 and RMSE 80->54). ``vw=0`` is the original
    horizontal IDW (identity), so callers that pass no z/vw are byte-identical.
    """
    if vw and z is not None:
        return np.column_stack([xy[:, 0], xy[:, 1], vw * np.asarray(z, "float64")])
    return xy


def crossfit_idw(
    xy: np.ndarray,
    value: np.ndarray,
    fold: np.ndarray,
    k: int,
    power: float,
    z: np.ndarray | None = None,
    vw: float = 0.0,
) -> np.ndarray:
    """Leave-one-fold-out IDW(kNN) of a scalar well value -- the leak-free prior.

    Scalar-generic: ``value`` is DTW for the regional DTW prior and observed WTE
    for the head-space prior. A held-out fold's wells are never in their own
    neighbor set. ``z``/``vw`` enable the relief-aware lift (see ``_relief_coords``).
    """
    coords = _relief_coords(xy, z, vw)
    pred = np.full(len(value), np.nan)
    for f in np.unique(fold):
        te = fold == f
        tr = ~te
        tree = cKDTree(coords[tr])
        dist, idx = tree.query(coords[te], k=k)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        w = 1.0 / np.maximum(dist, 1.0) ** power
        pred[te] = (w * value[tr][idx]).sum(1) / w.sum(1)
    return pred


def idw_at_points(
    train_xy: np.ndarray,
    train_val: np.ndarray,
    query_xy: np.ndarray,
    k: int,
    power: float,
) -> np.ndarray:
    """Plain IDW(kNN) of a scalar from training points to arbitrary query points."""
    tree = cKDTree(train_xy)
    kk = min(k, len(train_xy))
    dist, idx = tree.query(query_xy, k=kk)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / np.maximum(dist, 1.0) ** power
    return (w * train_val[idx]).sum(1) / w.sum(1)


def crossfit_anchor_anomaly(
    well_xy: np.ndarray,
    well_wte: np.ndarray,
    fold: np.ndarray,
    anchor_xy: np.ndarray,
    anchor_head: np.ndarray,
    k: int,
    power: float,
) -> dict[int, np.ndarray]:
    """Per-fold leak-safe anchor head-anomaly: anchor_head - R_f(anchor).

    R_f(anchor) is the IDW of OBSERVED WTE from wells in folds != f -- the SAME
    leave-one-fold-out scheme as the well-side regional prior R. A held-out fold's
    wells therefore never enter the anchor BC injected when predicting that fold,
    so the Dirichlet BC carries no leakage into fold f's own wells. The anomaly is
    in the wte_residual target space (obs_wte - R), so the trainer standardizes it
    by the same per-fold target median/MAD. Returns {fold: anomaly over anchor_xy}.
    """
    out: dict[int, np.ndarray] = {}
    for f in np.unique(fold):
        tr = fold != f
        r_f = idw_at_points(well_xy[tr], well_wte[tr], anchor_xy, k, power)
        out[int(f)] = anchor_head - r_f
    return out


def deep_well_mask(
    wells: pd.DataFrame, quantile: float, unit: str, min_per_unit: int
) -> np.ndarray:
    """Local deepest-quartile mask for the deep-aquifer datum.

    A well is 'deep' if its DTW is at/above the ``quantile`` of DTW WITHIN its
    local hydrologic unit (HUC6 by default). 'Deep' is thus relative to the local
    regime, so a deep-well pool -- and the datum built from it -- exists in every
    region; a single global cut would select almost only arid-West wells and leave
    humid CONUS with no deep wells.

    Leak discipline: the per-HUC6 deepest-quartile THRESHOLD is a global quantile
    (under HUC12/geometric-block folds a HUC6 spans several folds, so the threshold
    is no longer computed within a single fold -- a mild, second-order relaxation:
    one held-out well's DTW can nudge the membership cut of OTHER wells, but never
    enters its OWN prediction). The actual leak-safety is enforced downstream by
    ``crossfit_deep_idw`` (``tr = fold_deep != f``), which excludes the held-out
    fold from the deep training pool regardless of how the threshold was computed.
    Sparse units (< ``min_per_unit`` wells) fall back to the HUC4 threshold.
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
    z_all: np.ndarray | None = None,
    z_deep: np.ndarray | None = None,
    vw: float = 0.0,
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
    coords_all = _relief_coords(xy_all, z_all, vw)
    coords_deep = _relief_coords(xy_deep, z_deep, vw)
    pred = np.full(len(xy_all), np.nan)
    for f in np.unique(fold_all):
        te = fold_all == f
        tr = fold_deep != f
        if tr.sum() == 0:
            raise SystemExit(f"deep training pool empty for held-out fold {f}")
        tree = cKDTree(coords_deep[tr])
        kk = min(k, int(tr.sum()))
        dist, idx = tree.query(coords_all[te], k=kk)
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


def build_octant_lateral_edges(
    qxy: np.ndarray,
    geom: gpd.GeoDataFrame,
    comid_to_idx: dict,
    k_search: int,
    n_sectors: int = 8,
) -> pd.DataFrame:
    """Azimuthal lateral attachment: the NEAREST reach in each of ``n_sectors`` compass
    sectors (chosen from the ``k_search`` nearest rep-points), giving each well a
    spatially-REPRESENTATIVE set of surrounding drainages instead of the redundant
    knn-nearest cluster the topology review flagged (50% of wells piled all knn onto
    one Strahler-0 fingertip; +-8 m rel-elev instability across the knn). A well sitting
    in a valley bottom gets reaches on many sides (small rel-elev, large drainage); one
    on a divide gets edges mostly downslope -- so the DISTRIBUTION of the per-sector
    edges' (distance, rel-elev, drainage) attrs encodes topographic position on the
    network. Orientation-invariant: NO absolute-bearing feature is emitted (a global
    N/E orientation signal would be arbitrary), so the schema matches build_lateral_edges
    and the trainer is unchanged; only the connectivity differs.

    Rank/is_controlling follow the knn build (rank 0 == globally nearest). Columns are
    dist-sorted from the KD-tree, so the nearest candidate per sector is the lowest
    column index landing in that sector. No distance cap (same as build_lateral_edges);
    every well keeps >=1 edge since k_search >> n_sectors.
    """
    gx = geom["cx"].to_numpy("float64")
    gy = geom["cy"].to_numpy("float64")
    gidx = geom["reach_node_idx"].to_numpy("int64")
    tree = cKDTree(np.c_[gx, gy])
    k = min(k_search, len(gx))
    dist, cand = tree.query(qxy, k=k)  # ascending by distance
    if k == 1:
        dist, cand = dist[:, None], cand[:, None]
    n_q = len(qxy)
    dx = gx[cand] - qxy[:, 0:1]
    dy = gy[cand] - qxy[:, 1:2]
    ang = np.arctan2(dy, dx) % (2.0 * np.pi)  # (n_q, k), 0..2pi
    sector = np.clip(
        (ang / (2.0 * np.pi / n_sectors)).astype("int64"), 0, n_sectors - 1
    )
    # Nearest candidate per (well, sector): walk columns high->low so the final write
    # per sector is the lowest (nearest) column index; -1 == that sector is empty.
    chosen = np.full((n_q, n_sectors), -1, dtype="int64")
    rows = np.arange(n_q)
    for j in range(k - 1, -1, -1):
        chosen[rows, sector[:, j]] = j
    occ = chosen >= 0
    qn_idx = np.repeat(rows, n_sectors)[occ.ravel()]
    col = chosen[occ]
    d = dist[qn_idx, col]
    out = pd.DataFrame(
        {
            "query_node_idx": qn_idx,
            "reach_node_idx": gidx[cand[qn_idx, col]],
            "lateral_dist_m": d,
            "log1p_lateral_dist_m": np.log1p(d),
        }
    )
    # rank by distance within each well; rank 0 == globally-nearest = the controller.
    out = out.sort_values(["query_node_idx", "lateral_dist_m"]).reset_index(drop=True)
    out["rank"] = out.groupby("query_node_idx").cumcount().astype("int64")
    out["is_controlling"] = (out["rank"] == 0).astype("float64")
    return out


def build_down_ptr(rn: pd.DataFrame, ce: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Single max-drainage downstream pointer per reach (braid/diffluence resolution).

    A reach with >1 downstream edge (a braid/diffluence -- expected rare) keeps the edge
    whose dst has the largest ``totdasqkm`` (the mainstem branch). Returns
    ``(down_ptr, n_braids)`` where ``down_ptr`` is length ``len(rn)`` int64 (-1 at a
    component outlet). ``rn`` must be reach_node_idx-sorted 0..n-1 (the reach-graph +
    trainer invariant); ``ce`` is ``channel_edges`` (``direction == 1`` rows are src->dst
    downstream). Shared by ``downstream_datum`` and the upstream accumulation.
    """
    n = len(rn)
    totda = rn["totdasqkm"].to_numpy("float64")
    down = ce[ce["direction"] == 1]
    s = down["src_reach_idx"].to_numpy("int64")
    d = down["dst_reach_idx"].to_numpy("int64")
    n_braids = int(len(s) - pd.unique(s).size)
    order = np.lexsort((totda[d], s))  # ascending dst-drainage within each src
    down_ptr = np.full(n, -1, dtype="int64")
    down_ptr[s[order]] = d[order]  # last write per src == the max-drainage dst
    return down_ptr, n_braids


def accumulate_upstream(
    values: np.ndarray,
    seg_len: np.ndarray,
    down_ptr: np.ndarray,
) -> np.ndarray:
    """Length-weighted upstream (inclusive) mean of each value column over the
    down-pointer forest, via a Kahn topological accumulation (headwaters first).

    ``values`` is (n,) or (n, k) per-reach local values (may contain NaN); ``seg_len`` is
    the (n,) per-reach channel length used as the weight; ``down_ptr`` the single
    max-drainage downstream pointer from :func:`build_down_ptr`. NaN locals are excluded
    from BOTH the weighted sum and the weight sum (nan-skipping weighted mean); a reach
    whose entire upstream catchment is NaN for a column yields NaN there. Cycle-guarded
    (the down-pointer forest is a DAG). Returns an array shaped like ``values``.
    """
    n = len(down_ptr)
    v = np.asarray(values, dtype="float64")
    single = v.ndim == 1
    if single:
        v = v[:, None]
    finite = np.isfinite(v)
    w = np.where(np.isfinite(seg_len) & (seg_len > 0.0), seg_len, 0.0).astype("float64")
    acc_num = np.where(finite, v * w[:, None], 0.0)  # sum of w*value over finite locals
    acc_den = np.where(finite, w[:, None], 0.0)  # sum of w over finite locals
    indeg = np.zeros(n, dtype="int64")
    has_down = down_ptr >= 0
    np.add.at(indeg, down_ptr[has_down], 1)  # #direct upstream neighbours per reach
    remaining = indeg.copy()
    stack = list(np.where(indeg == 0)[0])  # headwaters first (Kahn topological order)
    processed = 0
    while stack:
        u = stack.pop()
        processed += 1
        p = down_ptr[u]
        if p >= 0:
            acc_num[p] += acc_num[u]
            acc_den[p] += acc_den[u]
            remaining[p] -= 1
            if remaining[p] == 0:
                stack.append(p)
    if processed != n:
        raise RuntimeError(
            "accumulate_upstream: cycle in the down-pointer graph "
            f"({processed}/{n} reaches ordered)"
        )
    out = np.full_like(acc_num, np.nan)
    pos = (
        acc_den > 0.0
    )  # reaches with >=1 finite upstream local (else no weight -> NaN)
    np.divide(acc_num, acc_den, out=out, where=pos)
    return out[:, 0] if single else out


def downstream_datum(
    rn: pd.DataFrame,
    ce: pd.DataFrame,
    order_band: int = 1,
    stop_mask: np.ndarray | None = None,
    max_iter: int = 5000,
) -> pd.DataFrame:
    """Per-reach first-downstream-datum via vectorized pointer stepping.

    The single topology primitive behind the regional-HAND-along-flowpath features
    (item 1), the mainstem-read edge set (item 2), and -- with ``stop_mask`` -- the
    downstream-wet-distance feature (item 5). Walks each reach down the FAC channel
    network to the first reach in the DATUM set and reports the along-network
    distance/elevation/order/hops to it.

    ``rn`` must expose ``reach_node_idx`` (contiguous 0..n-1, the trainer invariant),
    ``basin``, ``streamorde``, ``reach_elev_m``, ``totdasqkm`` and ``log1p_length_m``;
    ``ce`` is ``channel_edges`` (``direction == 1`` rows are src->dst downstream).

    Datum set: an explicit ``stop_mask`` (item 5's wet reaches) OR, by default, the
    per-basin top-``(order_band+1)`` Strahler band -- the SAME mainstem definition as
    ``build_conus_fac_reach_graph.network_distance_to_mainstem`` / the str_top2 concept,
    so the feature and the regional R stay aligned. Braids/diffluences (a reach with
    >1 downstream edge -- expected rare) resolve to the downstream reach with the
    largest ``totdasqkm`` (the mainstem branch); the count is logged.

    Returns a DataFrame indexed by ``reach_node_idx`` (int64, -1/NaN where no datum is
    reachable -- NEVER silently imputed; ``datum_missing`` is the authoritative flag):
      datum_reach_idx  int64  (-1 where missing)
      datum_dist_m     float  (NaN where missing; 0 for a self-datum reach)
      datum_elev_m     float  reach_elev_m at the datum (NaN where missing)
      datum_order      float  streamorde at the datum (NaN where missing)
      datum_n_hops     int64  channel hops walked (0 for self-datum)
      datum_missing    bool   walk dead-ended at a component outlet (no datum)
      datum_is_self    bool   the reach is itself a datum
      basin_max_order  float  per-basin max streamorde (the mainstem band reference)
    """
    rn = rn.sort_values("reach_node_idx").reset_index(drop=True)
    n = len(rn)
    if not (rn["reach_node_idx"].to_numpy("int64") == np.arange(n)).all():
        raise SystemExit(
            "downstream_datum: reach_node_idx must be contiguous 0..n-1 "
            "(the reach-graph builder + trainer invariant)"
        )
    strah = rn["streamorde"].to_numpy("float64")
    basin = rn["basin"].to_numpy()
    elev = rn["reach_elev_m"].to_numpy("float64")
    seg_len = np.expm1(rn["log1p_length_m"].to_numpy("float64"))  # per-reach length (m)
    seg_len = np.where(np.isfinite(seg_len) & (seg_len > 0.0), seg_len, 0.0)

    # single downstream pointer per reach; on braids keep the max-drainage branch.
    down_ptr, n_braids = build_down_ptr(rn, ce)

    basin_max = np.full(n, np.nan)
    for b in np.unique(basin):
        m = basin == b
        basin_max[m] = np.nanmax(strah[m])

    if stop_mask is not None:
        is_datum = np.asarray(stop_mask, dtype=bool)
        if len(is_datum) != n:
            raise SystemExit("downstream_datum: stop_mask length != n reaches")
    else:
        is_datum = strah >= (basin_max - order_band)

    datum_idx = np.full(n, -1, dtype="int64")
    datum_dist = np.zeros(n, dtype="float64")
    datum_hops = np.zeros(n, dtype="int64")
    self_datum = is_datum.copy()
    datum_idx[self_datum] = np.where(self_datum)[0]  # datum reaches terminate at self
    cur = np.arange(n, dtype="int64")
    active = ~is_datum
    guard = 0
    while active.any():
        guard += 1
        if guard > max_iter:
            raise RuntimeError(
                f"downstream_datum did not converge in {max_iter} steps "
                "(cycle in the down-pointer graph?)"
            )
        nxt = np.where(active, down_ptr[cur], -1)
        active = active & ~(nxt < 0)  # dead-end at a component outlet -> stays missing
        step = active  # survivors all have a downstream reach
        datum_dist[step] += seg_len[cur[step]]  # length of the reach being left
        datum_hops[step] += 1
        cur[step] = nxt[step]
        landed = step & is_datum[cur]
        datum_idx[landed] = cur[landed]
        active = active & ~landed

    missing = datum_idx < 0
    found = ~missing
    datum_elev = np.full(n, np.nan)
    datum_order = np.full(n, np.nan)
    datum_elev[found] = elev[datum_idx[found]]
    datum_order[found] = strah[datum_idx[found]]
    datum_dist[missing] = np.nan
    log.info(
        "downstream_datum: order_band=%s datum-reaches=%d/%d missing=%d (%.2f%%) "
        "braids-resolved=%d hops med/p90 %.0f/%.0f",
        "stop_mask" if stop_mask is not None else order_band,
        int(is_datum.sum()),
        n,
        int(missing.sum()),
        100.0 * missing.mean(),
        n_braids,
        float(np.median(datum_hops[found])) if found.any() else 0.0,
        float(np.percentile(datum_hops[found], 90)) if found.any() else 0.0,
    )
    return pd.DataFrame(
        {
            "datum_reach_idx": datum_idx,
            "datum_dist_m": datum_dist,
            "datum_elev_m": datum_elev,
            "datum_order": datum_order,
            "datum_n_hops": datum_hops,
            "datum_missing": missing,
            "datum_is_self": self_datum,
            "basin_max_order": basin_max,
        },
        index=pd.Index(np.arange(n, dtype="int64"), name="reach_node_idx"),
    )


def project_datum_to_queries(
    dd: pd.DataFrame,
    lat: pd.DataFrame,
    reach_nodes: pd.DataFrame,
    well_surf_m: np.ndarray,
) -> dict:
    """Project the per-reach ``downstream_datum`` result onto each query via its rank-0
    (globally-nearest) attached lateral reach.

    Shared by ``--ds-datum-features`` (item 1) and ``--mainstem-read`` (item 2). Every
    well has a rank-0 lateral edge (both lateral builders guarantee >=1), so ``have`` is
    all-True in practice; it is carried so a query with no attachment falls to
    ``datum_missing`` rather than being silently dropped. Returns numpy arrays aligned to
    ``query_node_idx`` (0..n_wells-1) -- NaN/-1 where no datum is reachable (NEVER
    imputed; the trainer median-imputes + flags at standardization).
    """
    n = len(well_surf_m)
    r0 = (
        lat[lat["rank"] == 0]
        .drop_duplicates("query_node_idx")
        .set_index("query_node_idx")
    )
    reach0 = np.full(n, -1, dtype="int64")
    lat0 = np.full(n, np.nan)
    qi = r0.index.to_numpy("int64")
    reach0[qi] = r0["reach_node_idx"].to_numpy("int64")
    lat0[qi] = r0["lateral_dist_m"].to_numpy("float64")
    have = reach0 >= 0

    # per-reach arrays are positionally reach_node_idx-ordered (dd index is contiguous
    # 0..n-1, asserted in downstream_datum; reach_node_idx is arange in the reach graph).
    rn_sorted = reach_nodes.sort_values("reach_node_idx")
    totda = rn_sorted["totdasqkm"].to_numpy("float64")
    strah = rn_sorted["streamorde"].to_numpy("float64")

    def _gather(vals: np.ndarray, fill):
        out = np.full(n, fill, dtype="float64")
        out[have] = vals[reach0[have]]
        return out

    reach0_strah = _gather(strah, np.nan)
    datum_elev = _gather(dd["datum_elev_m"].to_numpy("float64"), np.nan)
    datum_dist = _gather(dd["datum_dist_m"].to_numpy("float64"), np.nan)
    datum_order = _gather(dd["datum_order"].to_numpy("float64"), np.nan)
    datum_missing = _gather(dd["datum_missing"].to_numpy("float64"), 1.0)
    datum_is_self = _gather(dd["datum_is_self"].to_numpy("float64"), 0.0)
    basin_max_order = _gather(dd["basin_max_order"].to_numpy("float64"), np.nan)

    # totdasqkm at the DATUM reach (not the attached reach) for the ms-edge drainage attr.
    q_datum_reach = np.full(n, -1, dtype="int64")
    q_datum_reach[have] = dd["datum_reach_idx"].to_numpy("int64")[reach0[have]]
    has_datum = q_datum_reach >= 0
    datum_da = np.full(n, np.nan)
    datum_da[has_datum] = totda[q_datum_reach[has_datum]]

    return {
        "reach0": reach0,
        "reach0_strah": reach0_strah,
        "lat0": lat0,
        "datum_reach_idx": q_datum_reach,
        "datum_elev": datum_elev,
        "datum_dist": datum_dist,
        "datum_order": datum_order,
        "datum_missing": datum_missing,
        "datum_is_self": datum_is_self,
        "basin_max_order": basin_max_order,
        "datum_da": datum_da,
    }


def wet_propagation_reach_features(
    reach_nodes: pd.DataFrame,
    channel_edges: pd.DataFrame,
    wet: np.ndarray,
) -> pd.DataFrame:
    """Propagate a per-reach wet flag along the FAC channel network (item 5).

    Given a boolean ``wet`` mask over reaches (rep-point on GSW occurrence >= threshold,
    sampled in ``main`` where geom coords live), returns a DataFrame indexed by
    ``reach_node_idx`` with the WET_PROP_REACH_FEATURE_COLS:

      * ``wet_reach``               the flag itself (float 0/1).
      * ``log1p_net_dist_wet_m``    UNDIRECTED along-network distance to the nearest wet
        reach: one multi-source Dijkstra with a weight-0 super-source over every wet reach
        on the length-weighted undirected channel graph (same construction as
        ``build_conus_fac_reach_graph.network_distance_to_mainstem``). NaN in a component
        with no wet reach.
      * ``log1p_ds_dist_wet_m``     DOWNSTREAM-only distance to the nearest wet reach: the
        Phase-0b ``downstream_datum`` walk with ``stop_mask=wet`` (one shared primitive,
        two stop conditions). NaN where no downstream wet reach.
      * ``upstream_wet_fraction``   length-weighted wet channel length upstream (inclusive)
        / total upstream channel length, via a topological accumulation over the single
        max-drainage down-pointer (a DAG; cycle-guarded). In [0, 1].
      * ``no_wet_in_component``     0/1 indicator that the reach's undirected channel
        component holds zero wet reaches -- the authoritative NaN flag for the two
        distances (NEVER silently imputed, per CLAUDE.md; the trainer additionally
        median-imputes + flags at standardization).

    Distances are log1p-transformed; the fraction and flags are raw.
    """
    rn = reach_nodes.sort_values("reach_node_idx").reset_index(drop=True)
    n = len(rn)
    if not (rn["reach_node_idx"].to_numpy("int64") == np.arange(n)).all():
        raise SystemExit(
            "wet_propagation_reach_features: reach_node_idx must be contiguous 0..n-1"
        )
    wet = np.asarray(wet, dtype=bool)
    if len(wet) != n:
        raise SystemExit("wet_propagation_reach_features: wet mask length != n reaches")
    seg_len = np.expm1(rn["log1p_length_m"].to_numpy("float64"))
    seg_len = np.where(np.isfinite(seg_len) & (seg_len > 0.0), seg_len, 0.0)

    down = channel_edges[channel_edges["direction"] == 1]
    s = down["src_reach_idx"].to_numpy("int64")
    d = down["dst_reach_idx"].to_numpy("int64")

    # --- undirected net-distance to nearest wet reach (super-source Dijkstra) ---------
    w = seg_len[s]
    w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)  # edge weight = src reach length
    wet_idx = np.where(wet)[0]
    if len(wet_idx) == 0:
        net_dist = np.full(n, np.nan)
    else:
        super_idx = n
        rows = np.concatenate([s, wet_idx])
        cols = np.concatenate([d, np.full(len(wet_idx), super_idx)])
        data = np.concatenate([w, np.zeros(len(wet_idx))])
        g = csr_matrix((data, (rows, cols)), shape=(n + 1, n + 1))
        net_dist = dijkstra(g, directed=False, indices=super_idx)[:n]
        net_dist[~np.isfinite(net_dist)] = np.nan

    # --- no-wet-component indicator (authoritative NaN flag for the distances) --------
    adj = csr_matrix((np.ones(len(s)), (s, d)), shape=(n, n))
    ncomp, comp = connected_components(adj, directed=False)
    comp_has_wet = np.zeros(ncomp, dtype=bool)
    np.logical_or.at(comp_has_wet, comp, wet)
    no_wet_in_component = ~comp_has_wet[comp]

    # --- downstream-only distance to nearest wet reach (shared 0b walk primitive) -----
    dd = downstream_datum(rn, channel_edges, stop_mask=wet)
    ds_dist = dd["datum_dist_m"].to_numpy("float64")

    # --- upstream wet fraction: length-weighted accumulation over the down-pointer forest
    # (a DAG; braid resolution + cycle guard live in the shared helpers). The length-
    # weighted upstream mean of the 0/1 wet flag IS the wet channel-length fraction.
    down_ptr, n_braids = build_down_ptr(rn, channel_edges)
    upstream_wet_fraction = accumulate_upstream(
        wet.astype("float64"), seg_len, down_ptr
    )

    for order_lab in sorted(rn["streamorde"].dropna().unique()):
        m = rn["streamorde"].to_numpy("float64") == order_lab
        log.info(
            "  wet fraction @ Strahler %g: %.3f (%d reaches)",
            order_lab,
            float(wet[m].mean()) if m.any() else 0.0,
            int(m.sum()),
        )
    log.info(
        "wet-propagation: %d/%d wet reaches (%.2f%%), %d channel components, "
        "no-wet-component reaches %d (%.2f%%), braids-resolved %d; net_dist NaN %.2f%%, "
        "ds_dist NaN %.2f%%",
        int(wet.sum()),
        n,
        100.0 * wet.mean(),
        ncomp,
        int(no_wet_in_component.sum()),
        100.0 * no_wet_in_component.mean(),
        n_braids,
        100.0 * np.isnan(net_dist).mean(),
        100.0 * np.isnan(ds_dist).mean(),
    )
    return pd.DataFrame(
        {
            "wet_reach": wet.astype("float64"),
            "log1p_net_dist_wet_m": np.log1p(net_dist),
            "log1p_ds_dist_wet_m": np.log1p(ds_dist),
            "upstream_wet_fraction": upstream_wet_fraction,
            "no_wet_in_component": no_wet_in_component.astype("float64"),
        },
        index=pd.Index(np.arange(n, dtype="int64"), name="reach_node_idx"),
    )


def upstream_head(rn: pd.DataFrame, ce: pd.DataFrame) -> pd.DataFrame:
    """Per-reach farthest-upstream channel head along the max-drainage UP-pointer.

    The 6B ``up_head`` reference site (the recharge-boundary end of the attached flowpath):
    a MIRROR of :func:`downstream_datum` walking UP instead of down. Build a single
    up-pointer per reach via the mirrored lexsort ``np.lexsort((totda[s], d))`` (last-write-
    wins == the max-``totdasqkm`` PARENT per dst -- the mainstem-upstream branch), then step
    each reach up that pointer until ``up_ptr == -1`` (a channel head with no parent). Every
    reach reaches a head (itself if already terminal), so there is no missing flag.

    ``rn`` must be reach_node_idx-contiguous 0..n-1; ``ce`` is ``channel_edges``. Returns a
    DataFrame indexed by ``reach_node_idx`` with head_reach_idx / head_dist_m / head_elev_m /
    head_order / head_da_km2 / head_n_hops.
    """
    rn = rn.sort_values("reach_node_idx").reset_index(drop=True)
    n = len(rn)
    if not (rn["reach_node_idx"].to_numpy("int64") == np.arange(n)).all():
        raise SystemExit("upstream_head: reach_node_idx must be contiguous 0..n-1")
    totda = rn["totdasqkm"].to_numpy("float64")
    strah = rn["streamorde"].to_numpy("float64")
    elev = rn["reach_elev_m"].to_numpy("float64")
    seg_len = np.expm1(rn["log1p_length_m"].to_numpy("float64"))
    seg_len = np.where(np.isfinite(seg_len) & (seg_len > 0.0), seg_len, 0.0)

    down = ce[ce["direction"] == 1]
    s = down["src_reach_idx"].to_numpy("int64")
    d = down["dst_reach_idx"].to_numpy("int64")
    up_ptr = np.full(n, -1, dtype="int64")
    order = np.lexsort((totda[s], d))  # ascending src-drainage within each dst
    up_ptr[d[order]] = s[order]  # last write per dst == the max-drainage parent

    head_idx = np.arange(n, dtype="int64")  # a reach with up_ptr==-1 is its own head
    head_dist = np.zeros(n, dtype="float64")
    head_hops = np.zeros(n, dtype="int64")
    cur = np.arange(n, dtype="int64")
    active = up_ptr[cur] >= 0
    guard = 0
    while active.any():
        guard += 1
        if guard > 5000:
            raise RuntimeError(
                "upstream_head did not converge in 5000 steps (cycle in up-pointer graph?)"
            )
        head_dist[active] += seg_len[cur[active]]  # length of the reach being left
        head_hops[active] += 1
        cur[active] = up_ptr[cur[active]]  # step up (all active have up_ptr >= 0)
        head_idx[active] = cur[active]
        active = active & (
            up_ptr[cur] >= 0
        )  # continue while the new reach has a parent
    log.info(
        "upstream_head: hops med/p90 %.0f/%.0f, dist med/p90 %.0f/%.0f m, %d self-heads",
        float(np.median(head_hops)),
        float(np.percentile(head_hops, 90)),
        float(np.median(head_dist)),
        float(np.percentile(head_dist, 90)),
        int((head_hops == 0).sum()),
    )
    return pd.DataFrame(
        {
            "head_reach_idx": head_idx,
            "head_dist_m": head_dist,
            "head_elev_m": elev[head_idx],
            "head_order": strah[head_idx],
            "head_da_km2": totda[head_idx],
            "head_n_hops": head_hops,
        },
        index=pd.Index(np.arange(n, dtype="int64"), name="reach_node_idx"),
    )


def serving_wet_source(
    rn: pd.DataFrame, ce: pd.DataFrame, wet: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-reach nearest wet reach along the UNDIRECTED length-weighted channel network,
    plus the along-network distance to it (the 6B ``wet`` reference site).

    Mirrors the super-source Dijkstra of :func:`wet_propagation_reach_features` (which
    returns only the DISTANCE) but resolves the SOURCE identity of each shortest path via
    ``return_predecessors``: walk predecessors toward the weight-0 super-source, resolving
    nodes in ascending-distance order (a node's predecessor is always closer to the source,
    hence already resolved). This lets a query READ the wet reach node's learned state, not
    just its distance -- the missingness-robust return of item-5's physics as an edge.

    Returns ``(serving, net_dist)``, both reach_node_idx-indexed 0..n-1: ``serving`` is the
    wet reach idx ending each shortest path (-1 in a component with no wet reach -- NEVER
    imputed); ``net_dist`` the along-network distance (NaN where serving == -1).
    """
    rn = rn.sort_values("reach_node_idx").reset_index(drop=True)
    n = len(rn)
    if not (rn["reach_node_idx"].to_numpy("int64") == np.arange(n)).all():
        raise SystemExit("serving_wet_source: reach_node_idx must be contiguous 0..n-1")
    wet = np.asarray(wet, dtype=bool)
    if len(wet) != n:
        raise SystemExit("serving_wet_source: wet mask length != n reaches")
    seg_len = np.expm1(rn["log1p_length_m"].to_numpy("float64"))
    seg_len = np.where(np.isfinite(seg_len) & (seg_len > 0.0), seg_len, 0.0)
    down = ce[ce["direction"] == 1]
    s = down["src_reach_idx"].to_numpy("int64")
    d = down["dst_reach_idx"].to_numpy("int64")
    w = seg_len[s]
    w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)  # edge weight = src reach length
    wet_idx = np.where(wet)[0]
    serving = np.full(n, -1, dtype="int64")
    net_dist = np.full(n, np.nan)
    if len(wet_idx) == 0:
        return serving, net_dist
    super_idx = n
    rows = np.concatenate([s, wet_idx])
    cols = np.concatenate([d, np.full(len(wet_idx), super_idx)])
    data = np.concatenate([w, np.zeros(len(wet_idx))])
    g = csr_matrix((data, (rows, cols)), shape=(n + 1, n + 1))
    dist, pred = dijkstra(
        g, directed=False, indices=super_idx, return_predecessors=True
    )
    # Resolve the serving wet reach per node. A wet reach's predecessor IS the super-source
    # (0-weight edge); a non-wet reach inherits its predecessor's source. Ascending distance
    # guarantees the predecessor (strictly closer to super, internal weights > 0) is resolved.
    for i in np.argsort(dist[:n]):
        di = dist[i]
        if not np.isfinite(di):
            continue  # unreachable: no wet reach in this component -> stays -1 / NaN
        p = pred[i]
        serving[i] = i if p == super_idx else serving[p]
        net_dist[i] = di
    return serving, net_dist


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
        # geology: perm raw is log10(k m^2)*100 -> /100; nodata already -> NaN.
        "perm_logk_m2": sample_coarse(PERM_LOGK, x, y) / 100.0,
        "sediment_thickness_m": sample_coarse(SED_THICKNESS, x, y),
    }


def sample_terrain_multiscale(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Multi-scale terrain-position family (haf/tpi/twi x 4 scales), EPSG:5070 rasters.

    height-above-local-floor is meters (relative elevation, translation-invariant); TPI
    and TWI are dimensionless. All sampled at x5070/y5070. Off-footprint / nodata stay
    NaN (the trainer median-imputes + flags), consistent with the other query rasters.
    """
    return {c: sample_coarse(p, x, y) for c, p in TERRAIN_MULTISCALE_RASTERS.items()}


def sample_gridmet(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """gridMET aridity + mean-annual-precip at EPSG:5070 coords.

    The gridMET rasters are EPSG:4326, so the 5070 coords are transformed to lon/lat
    first (NOT sampled at 5070 metres). Off-footprint / nodata stay NaN (the trainer
    median-imputes + flags). Kept in wte_residual mode despite the stacker's NEGATIVE
    result -- a tabular non-result is not a GNN non-result.
    """
    from pyproj import Transformer

    tr = Transformer.from_crs(5070, 4326, always_xy=True)
    lon, lat = tr.transform(x, y)
    lon, lat = np.asarray(lon), np.asarray(lat)
    return {
        "aridity_index": sample_coarse(GRIDMET_AI, lon, lat),
        "mean_annual_precip_mm": sample_coarse(GRIDMET_P, lon, lat),
    }


def sample_irrigation(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """IrrMapper irrigation frequency (%% of 2015-2024 years irrigated) at 5070 coords.

    Per-state rasters: first finite in-bounds value wins (state rasters do not
    overlap except trivial border slivers). The rasters carry nodata=None, so every
    in-bounds pixel is valid (0 = never irrigated). Wells outside ALL three state
    rasters get 0.0 -- outside the mapped-states footprint "no mapped irrigation" is
    the exact value, not an impute; the count is logged loudly.
    """
    out = np.full(np.shape(x), np.nan, dtype="float64")
    for st, path in IRRMAPPER_FREQ_RASTERS.items():
        m = ~np.isfinite(out)
        if not m.any():
            break
        out[m] = sample_coarse(path, np.asarray(x)[m], np.asarray(y)[m])
    n_outside = int((~np.isfinite(out)).sum())
    if n_outside:
        log.warning(
            "irrigation features: %d wells outside every IrrMapper state raster "
            "-> irr_freq_pct=0 (unmapped, not imputed)",
            n_outside,
        )
        out[~np.isfinite(out)] = 0.0
    return {"irr_freq_pct": out}


def sample_evidence_ndvi(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Summer NDVI + summer-minus-winter amplitude at well 5070 coords.

    Phreatophyte greenness: vegetation that stays green through summer (high JJA NDVI),
    especially where winter NDVI is low (high amplitude), flags shallow-groundwater
    access in water-limited settings -- the shallow signal the FAC pipeline solves for,
    fed here as raw OBSERVED evidence instead of FAC's water-surface estimate. Both
    MODIS NDVI rasters are EPSG:5070. Winter (DJF) nodata over snow/cloud stays NaN; the
    trainer median-imputes + flags it (fit_stats/apply_stats), so amp inherits that gap.
    """
    jja = sample_coarse(NDVI_JJA, x, y)
    djf = sample_coarse(NDVI_DJF, x, y)
    return {"well_ndvi_jja": jja, "well_ndvi_amp": jja - djf}


def _sample_gsw_occurrence(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """JRC GSW occurrence (0-100 = %% of valid months inundated) at lon/lat (EPSG:4326).

    Routes each point to its covering 10-deg tile by tile bounds; points outside every
    cached tile stay NaN. Half-open bounds [left,right)/(bottom,top] keep a point on a
    shared 10-deg tile seam from being sampled twice.
    """
    import rasterio  # local: heavy import only when the evidence bank is built

    lon = np.asarray(lon, "float64")
    lat = np.asarray(lat, "float64")
    out = np.full(lon.shape, np.nan)
    tiles = sorted(GSW_OCC_DIR.glob("*.tif"))
    if not tiles:
        raise SystemExit(f"no GSW occurrence tiles in {GSW_OCC_DIR}")
    for tp in tiles:
        with rasterio.open(tp) as ds:
            b = ds.bounds
            ins = (lon >= b.left) & (lon < b.right) & (lat > b.bottom) & (lat <= b.top)
            if not ins.any():
                continue
            vals = np.array(
                [v[0] for v in ds.sample(list(zip(lon[ins], lat[ins])))], "float64"
            )
            if ds.nodata is not None:
                vals[vals == ds.nodata] = np.nan
            out[ins] = vals
    return out


def reach_reppoint_coords(
    reach_nodes: pd.DataFrame, geom: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Rep-point (cx, cy) EPSG:5070 per reach, reach_node_idx-indexed (0..n-1); NaN where a
    reach has no flowline rep-point in ``geom``. Rep-points are reach_node_idx-keyed in
    ``geom`` (``cx``/``cy``); the scatter yields a reach_node_idx-indexed array regardless
    of ``reach_nodes`` row order (the reach graph guarantees contiguous 0..n-1)."""
    n = len(reach_nodes)
    rcx = np.full(n, np.nan)
    rcy = np.full(n, np.nan)
    gidx = geom["reach_node_idx"].to_numpy("int64")
    rcx[gidx] = geom["cx"].to_numpy("float64")
    rcy[gidx] = geom["cy"].to_numpy("float64")
    return rcx, rcy


def sample_reach_gsw_occ(rcx: np.ndarray, rcy: np.ndarray) -> np.ndarray:
    """JRC GSW occurrence (0-100) at reach rep-points (EPSG:5070 in, reach_node_idx-indexed).

    Shared by 6A.1 (r_gsw_occ), 6A.2 (the wet mask for upstream_wet_fraction), and 6B (the
    wet reference site) so the GSW tiles are sampled once. NaN where a reach lacks a
    rep-point or the point falls outside every cached 10-deg GSW tile.
    """
    from pyproj import (
        Transformer,
    )  # local: heavy import only when a GSW consumer is built

    occ = np.full(len(rcx), np.nan)
    finite = np.isfinite(rcx) & np.isfinite(rcy)
    if finite.any():
        lon, lat_deg = Transformer.from_crs(5070, 4326, always_xy=True).transform(
            rcx[finite], rcy[finite]
        )
        occ[finite] = _sample_gsw_occurrence(lon, lat_deg)
    return occ


def sample_reach_covariates(
    rcx: np.ndarray, rcy: np.ndarray, occ: np.ndarray
) -> dict[str, np.ndarray]:
    """The 6A.1 REACH_COVARIATE_FEATURE_COLS at reach rep-points (EPSG:5070, reach_node_idx-
    indexed). ``occ`` is the pre-sampled GSW occurrence (shared with the wet mask, so the
    GSW tiles are read once). All rasters are target-blind; gridMET is EPSG:4326 (transform
    to lon/lat) and perm decodes /100 exactly as the query side (B: sample_relief_etrm)."""
    from pyproj import Transformer

    lon = np.full(len(rcx), np.nan)
    lat_deg = np.full(len(rcx), np.nan)
    finite = np.isfinite(rcx) & np.isfinite(rcy)
    if finite.any():
        lo, la = Transformer.from_crs(5070, 4326, always_xy=True).transform(
            rcx[finite], rcy[finite]
        )
        lon[finite], lat_deg[finite] = lo, la
    jja = sample_coarse(NDVI_JJA, rcx, rcy)
    djf = sample_coarse(NDVI_DJF, rcx, rcy)
    return {
        "r_tpi_2km": sample_coarse(TERRAIN_MULTISCALE_RASTERS["tpi_2km"], rcx, rcy),
        "r_tpi_10km": sample_coarse(TERRAIN_MULTISCALE_RASTERS["tpi_10km"], rcx, rcy),
        "r_twi_2km": sample_coarse(TERRAIN_MULTISCALE_RASTERS["twi_2km"], rcx, rcy),
        "r_etrm_recharge_mm": sample_coarse(ETRM_RECHARGE, rcx, rcy),
        "r_etrm_eta_mm": sample_coarse(ETRM_ETA, rcx, rcy),
        "r_etrm_runoff_mm": sample_coarse(ETRM_RUNOFF, rcx, rcy),
        "r_aridity_index": sample_coarse(GRIDMET_AI, lon, lat_deg),
        "r_precip_mm": sample_coarse(GRIDMET_P, lon, lat_deg),
        "r_perm_logk_m2": sample_coarse(PERM_LOGK, rcx, rcy) / 100.0,
        "r_sediment_thickness_m": sample_coarse(SED_THICKNESS, rcx, rcy),
        "r_ndvi_jja": jja,
        "r_ndvi_amp": jja - djf,
        "r_gsw_occ": occ,
    }


def dist_to_wet_reach(
    qxy: np.ndarray,
    rx: np.ndarray,
    ry: np.ndarray,
    occ_threshold: float,
    search_km: float,
) -> np.ndarray:
    """Distance (m, EPSG:5070) from each well to the nearest stream reach carrying GSW
    surface-water evidence.

    "Water evidence" = a reach whose flowline rep-point sits on GSW occurrence
    >= ``occ_threshold`` (%% of valid months inundated). Candidate reaches are limited to
    within ``search_km`` of any well (the only ones that can be a near neighbour), so the
    per-point GSW sampling stays cheap. Distinct from the bundle's dist_to_stream_m
    (nearest stream, wet or dry): the GAP between them is the signal -- close to a channel
    but far from WET water flags an ephemeral/dry reach over a deeper table (the
    deep-regime flag the 30+m tail lacks). Every well gets a finite nearest wet reach.
    """
    from pyproj import Transformer

    near = cKDTree(qxy).query(np.column_stack([rx, ry]), k=1)[0] <= search_km * 1000.0
    if not near.any():
        raise SystemExit("no candidate reaches within --wet-search-km of any well")
    rx, ry = rx[near], ry[near]
    lon, lat = Transformer.from_crs(5070, 4326, always_xy=True).transform(rx, ry)
    occ = _sample_gsw_occurrence(lon, lat)
    wet = np.isfinite(occ) & (occ >= occ_threshold)
    if not wet.any():
        raise SystemExit(
            f"no GSW-wet reaches (occ>={occ_threshold}) within {search_km} km of any "
            "well -- lower --gsw-wet-threshold or widen --wet-search-km"
        )
    d = cKDTree(np.column_stack([rx[wet], ry[wet]])).query(qxy, k=1)[0]
    log.info(
        "dist_to_wet_reach: %d/%d candidate reaches wet (occ>=%g); well dist "
        "p50/p90/max %.0f/%.0f/%.0f m",
        int(wet.sum()),
        int(near.sum()),
        occ_threshold,
        float(np.median(d)),
        float(np.percentile(d, 90)),
        float(d.max()),
    )
    return d


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


def join_obs_metadata(wells: pd.DataFrame, path: str) -> dict[str, float]:
    """Attach GWX observation metadata (OBS_METADATA_COLS) by canonical_id.

    Reindex-maps (not a join) so the wells row order is untouched; wells absent
    from the GWX index keep NaN. Returns per-column non-null coverage for the
    manifest.
    """
    meta = pd.read_parquet(path, columns=["canonical_id", *OBS_METADATA_COLS])
    meta = meta.drop_duplicates("canonical_id").set_index("canonical_id")
    aligned = meta.reindex(wells["canonical_id"].to_numpy())
    coverage: dict[str, float] = {}
    for c in OBS_METADATA_COLS:
        vals = aligned[c].to_numpy()
        wells[c] = vals
        coverage[c] = float(pd.notna(vals).mean())
    return coverage


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
        "--residual-base",
        choices=["str_top2", "fac_rem", "relief_idw", "ensemble_median"],
        default="str_top2",
        help="(wte_residual only) the regional surface R the residual target/base/"
        "anomalies are taken over. str_top2 = the well-free streams-Strahler regional "
        "WTE prior (default; FAC enters as a feature/fac-skip anchor). fac_rem = the "
        "FAC-REM water surface (z_surf - fac_rem_dtw): FAC becomes the residual BASE and "
        "the model predicts the correction to FAC's DTW (dtw = fac_rem_dtw - resid_hat); "
        "str_top2 then enters as the regional-context anomaly feature. Use fac_rem (no "
        "--fac-skip) when FAC-REM is the better base than R -- the FAC-residual approach. "
        "relief_idw = the cross-fit (leave-fold-out) relief-aware well-IDW WTE surface "
        "(set --r-relief-vw 100): our most-accurate regional R. The GNN predicts the "
        "graph-structured residual over it; FAC + deep enter as anomaly features. Needs "
        "NO str_top2, so it runs over the full FAC footprint (not the trunk-only subset). "
        "ensemble_median = per-well median of three level-0 members {relief_idw well-IDW "
        "(vw=100), simple well-IDW (vw=0), FAC-REM water surface}: a robust R that damps "
        "the relief-lift over-mounding on benches while keeping the valley-floor accuracy. "
        "All three members are leak-free (well-IDW cross-fit leave-fold-out; FAC well-"
        "free); same full FAC footprint as relief_idw; FAC + deep enter as anomalies.",
    )
    ap.add_argument(
        "--stacker-features",
        default=None,
        help="optional parquet keyed by canonical_id with fac_rem_dtw_m; used to "
        "build fac_rem_wte_m in --target wte mode (only those two cols are read)",
    )
    ap.add_argument(
        "--require-fac",
        action="store_true",
        help="train where you serve: keep ONLY wells inside FAC-REM raster coverage "
        "(the built basins). Aligns the training footprint with the wall-to-wall "
        "render domain so every well carries FAC and the model actually learns it; "
        "folds + the regional prior R + deep datum all become basin-local.",
    )
    ap.add_argument(
        "--well-class",
        nargs="+",
        default=None,
        help="keep only queries whose GWX well_class is in this set (e.g. "
        "'monitoring' for the monitoring-wells-only lineage). Applied before "
        "folds, so the cross-fit priors R / deep-IDW are built from these wells "
        "only.",
    )
    ap.add_argument(
        "--obs-metadata",
        action="store_true",
        help="carry GWX observation metadata (por_start/por_end, obs_count, "
        "well_depth, screen_bottom, head_above_screen) onto query nodes as "
        "diagnostics (never features)",
    )
    ap.add_argument(
        "--gwx-wells",
        default="/data/ssd2/gwx/products/current/wells.geoparquet",
        help="GWX well index supplying --obs-metadata columns",
    )
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument(
        "--cv-scheme",
        choices=["huc12", "block", "huc4"],
        default="huc12",
        help="CV blocking unit: huc12 (real WBD HUC12 polygons, the default -- holds "
        "out small sub-HUC8 basins), block (geometric square blocks, the national "
        "fallback needing no polygons), huc4 (legacy, far-too-large holdout)",
    )
    ap.add_argument(
        "--huc12-polys",
        default="/nas/hydrography/HUC_Boundaries/wbd_national/wbdhu12_5070.parquet",
        help="EPSG:5070 HUC12 geoparquet for --cv-scheme huc12 (national WBD)",
    )
    ap.add_argument(
        "--cv-block-km",
        type=float,
        default=12.0,
        help="square-block size (km) for --cv-scheme block + the unmatched-HUC12 "
        "fallback; ~12 km is the HUC12 areal scale",
    )
    ap.add_argument("--knn-lateral", type=int, default=3)
    ap.add_argument("--idw-k", type=int, default=32)
    ap.add_argument("--idw-power", type=float, default=2.0)
    ap.add_argument(
        "--r-relief-vw",
        type=float,
        default=0.0,
        help="relief-aware lift for the wte_residual regional WTE prior R: vertical "
        "weight (horizontal-m per vertical-m). 0=horizontal IDW (original); 100 lifts "
        "neighbor selection + weights into (x,y,vw*z_surf) so only wells at similar "
        "ground elevation inform a cell (build_wte_idw_grid.py: MAD 8.45->6.18, "
        "RMSE 80->54 in high relief, no-op where dz~0).",
    )
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
    ap.add_argument(
        "--evidence-features",
        action="store_true",
        help="add the observable evidence bank (MODIS NDVI greenness + GSW wet-reach "
        "distance) to the wte_residual query features -- WITHOUT the FAC-REM head "
        "estimate (the fac-skip anchor is unchanged)",
    )
    ap.add_argument(
        "--gsw-wet-threshold",
        type=float,
        default=25.0,
        help="GSW occurrence %% for a reach to count as carrying water evidence",
    )
    ap.add_argument(
        "--irrigation-features",
        action="store_true",
        help="add IrrMapper irrigation frequency (%% of 2015-2024 years irrigated, "
        "per-state 30 m rasters) to the wte_residual query features -- the "
        "irrigation-recharge mound signal terrain covariates cannot see",
    )
    ap.add_argument(
        "--wet-search-km",
        type=float,
        default=50.0,
        help="candidate-reach radius (km) around wells for the wet-reach KD-tree",
    )
    ap.add_argument(
        "--fac-rem-feature",
        action="store_true",
        help="add fac_rem_dtw_m (the FAC-REM shallow DTW prior = properly-solved "
        "height above the FAC network) as a DIRECT query-node feature. Target-blind + "
        "translation-invariant (RGA-safe). Otherwise the graph sees the shallow height "
        "signal only indirectly (entangled in fac_rem_wte_anom_m / on lateral edges). "
        "The graph-topology rep-point rel-elev was rejected as a noisy ~2x-weaker proxy "
        "(50%% of wells attach to Strahler-0 fingertips; see the topology review).",
    )
    ap.add_argument(
        "--ensemble-member-features",
        action="store_true",
        help="(wte_residual only) expose the ensemble members that are NOT the base as "
        "anomaly-over-R query features, so the GNN sees where the members DISAGREE rather "
        "than being handed a pre-blended base (which washed out; see ensemble_median). "
        "Adds simple_idw_wte_anom_m = plain vw=0 well-IDW WTE - R; in relief_idw mode R is "
        "the vw=100 member so this IS the relief-lift disagreement (negative where relief "
        "over-mounds a bench = the missing bench signal). FAC-REM already enters as "
        "fac_rem_wte_anom_m (+ fac_rem_dtw_m); relief-IDW is the base frame (anomaly 0). "
        "Leak-free (crossfit leave-fold-out) + translation-invariant.",
    )
    ap.add_argument(
        "--terrain-multiscale-features",
        action="store_true",
        help="(wte_residual only) add the multi-scale terrain-position family "
        "(height-above-local-floor + TPI + multi-scale TWI, 4 scales each: "
        "500 m/2 km/5 km/10 km) as DIRECT query-node features from "
        "/nas/handily/covariates/terrain (build_terrain_covariates.py). The residual "
        "over R is terrain-organized but the wired elev_above_coarse_m proxy is ~dead; "
        "height-above-local-floor is the proven lever (GBM R2 0.06->0.13 on obs_wte-R). "
        "All target-blind + translation-invariant (relative elevation / dimensionless).",
    )
    ap.add_argument(
        "--ds-datum-features",
        action="store_true",
        help="(wte_residual only) add the regional-HAND-along-flowpath query features "
        "(item 1): from each well's rank-0 attached FAC reach, walk DOWN the channel "
        "network to the basin's mainstem discharge datum (top-(mainstem_order_band+1) "
        "Strahler band, the SAME definition as net_dist_mainstem / R) and emit the "
        "elevation drop (ds_datum_drop_m = the long-wavelength regional HAND), the "
        "along-network distance, the order-jump, and a walk-dead-ended flag. This is the "
        "flagged Tier-1 missing quantity: fac_rem_dtw_m is the LOCAL fine-grid HAND and "
        "net_dist_mainstem_m is a distance with no elevation. Relative-elevation + "
        "topology only -> target-blind + translation-invariant (RGA-safe, leak-free). "
        "See notes/GNN_TOPOLOGY_PLAN.md item 1.",
    )
    ap.add_argument(
        "--mainstem-read",
        action="store_true",
        help="(item 2) emit mainstem_edges.parquet: exactly one query->downstream-datum "
        "read edge per well whose datum exists, letting the query attend to the DATUM "
        "reach's LEARNED state (seed evidence, drainage, 2-hop channel context) -- fixes "
        "the med-7/p90-28-hop receptive-field gap without deepening the channel stack or "
        "touching component fragmentation. Shares the --ds-datum-features walk (enable "
        "both in the A/B). Recommend pairing with --ds-datum-features so ds_datum_missing "
        "lets the head tell zero-context from real context. All edge attrs relative / "
        "dimensionless / topological -> leak-free. See notes/GNN_TOPOLOGY_PLAN.md item 2.",
    )
    ap.add_argument(
        "--wet-propagation-features",
        action="store_true",
        help="(item 5) propagate a per-reach GSW wet flag along the FAC channel network "
        "and write an AUGMENTED (materialized, not symlinked) reach_nodes.parquet with "
        "network-metric surface-water features: log1p_net_dist_wet_m (undirected dist to "
        "nearest wet reach), log1p_ds_dist_wet_m (downstream-only, the 0b walk with a wet "
        "stop_mask), upstream_wet_fraction, and a no_wet_in_component NaN flag. REACH-side "
        "+ NETWORK-metric (distinct from the rejected query-side + Euclidean evidence "
        "bank): dist-to-water != dist-to-drainage. Reuses --gsw-wet-threshold; the trainer "
        "reads reach cols from the manifest -> no trainer change. See "
        "notes/GNN_TOPOLOGY_PLAN.md item 5.",
    )
    ap.add_argument(
        "--octant-lateral",
        action="store_true",
        help="build lateral edges by azimuthal sector instead of plain k-NN: the "
        "nearest FAC reach in each of --octant-sectors compass directions (from the "
        "--octant-k-search nearest rep-points). Gives every well DIRECTIONAL drainage "
        "geometry (valley-confluence vs divide, mainstem access from each side) rather "
        "than the k nearest reaches, which pile onto one dense cluster. "
        "Orientation-invariant (no absolute-bearing feature) so the edge-attr schema is "
        "identical to k-NN -> clean A/B; only connectivity changes.",
    )
    ap.add_argument(
        "--octant-k-search",
        type=int,
        default=64,
        help="nearest rep-points to scan when bucketing into octant sectors "
        "(--octant-lateral); larger fills far/empty sectors but costs KD-tree query time",
    )
    ap.add_argument(
        "--octant-sectors",
        type=int,
        default=8,
        help="number of azimuthal sectors for --octant-lateral (8 = octants)",
    )
    ap.add_argument(
        "--reach-covariate-features",
        action="store_true",
        help="(Phase 6A.1) sample a target-blind covariate bank (TPI/TWI, ETRM flux, "
        "gridMET climate, GLHYMPS perm + sediment, MODIS NDVI, GSW occurrence) at each "
        "reach's flowline rep-point and MATERIALIZE the augmented reach_nodes.parquet. "
        "haf_* excluded (~0 at a reach by construction). Any column >2%% NaN is DROPPED "
        "with a warning (only raster nodata at a rep-point is tolerated). The trainer reads "
        "reach cols from the manifest -> zero trainer change. See notes/GNN_PHASE6_PLAN.md "
        "6A.1.",
    )
    ap.add_argument(
        "--upstream-accumulation-features",
        action="store_true",
        help="(Phase 6A.2, requires --reach-covariate-features) add length-weighted "
        "upstream-catchment means of the 6A.1 locals (upstream_recharge_mm/precip_mm/"
        "ndvi_jja -- recharge upstream is the physical driver of the table AT the reach) + "
        "upstream_wet_fraction (the ONE always-finite Phase-5 survivor; the NaN-distance "
        "columns are excluded). Kahn accumulation over the max-drainage down-pointer. See "
        "notes/GNN_PHASE6_PLAN.md 6A.2.",
    )
    ap.add_argument(
        "--portfolio-read",
        action="store_true",
        help="(Phase 6B) build per-query typed read edges DIRECT to up to 4 heterogeneous "
        "reference reaches -- ds_datum (first downstream mainstem-band reach), up_head "
        "(farthest-upstream reach on the attached flowpath), wet (nearest downstream "
        "GSW-wet reach), and ho_any (nearest high-order reach by Euclidean distance, the "
        "NV closed-basin fallback where downstream never reaches a datum). A missing site "
        "is simply an ABSENT edge the trainer's segment-softmax renormalizes over (the "
        "structural fix for the wet-propagation missingness killer). Writes "
        "portfolio_edges.parquet + per-type portfolio_missing_* query flags; requires "
        "channel_edges + reach rep-points. See notes/GNN_PHASE6_PLAN.md 6B.",
    )
    args = ap.parse_args()
    gdir = Path(args.graph_dir)
    if args.evidence_features and args.target != TARGET_WTE_RESIDUAL:
        raise SystemExit(
            "--evidence-features is wired into the wte_residual query bank only; "
            f"got target={args.target}"
        )
    if args.fac_rem_feature and args.target != TARGET_WTE_RESIDUAL:
        raise SystemExit(
            "--fac-rem-feature is wired into the wte_residual query bank only; "
            f"got target={args.target}"
        )
    if args.irrigation_features and args.target != TARGET_WTE_RESIDUAL:
        raise SystemExit(
            "--irrigation-features is wired into the wte_residual query bank only; "
            f"got target={args.target}"
        )
    if args.ensemble_member_features and args.target != TARGET_WTE_RESIDUAL:
        raise SystemExit(
            "--ensemble-member-features is wired into the wte_residual query bank only; "
            f"got target={args.target}"
        )
    if args.terrain_multiscale_features and args.target != TARGET_WTE_RESIDUAL:
        raise SystemExit(
            "--terrain-multiscale-features is wired into the wte_residual query bank only; "
            f"got target={args.target}"
        )
    if args.ds_datum_features and args.target != TARGET_WTE_RESIDUAL:
        raise SystemExit(
            "--ds-datum-features is wired into the wte_residual query bank only; "
            f"got target={args.target}"
        )
    if args.reach_covariate_features and args.target != TARGET_WTE_RESIDUAL:
        raise SystemExit(
            "--reach-covariate-features is wired into the wte_residual bundle only; "
            f"got target={args.target}"
        )
    if args.upstream_accumulation_features and not args.reach_covariate_features:
        raise SystemExit(
            "--upstream-accumulation-features requires --reach-covariate-features "
            "(the upstream means accumulate the 6A.1 local r_* covariates)"
        )
    if args.portfolio_read and args.target != TARGET_WTE_RESIDUAL:
        raise SystemExit(
            "--portfolio-read is wired into the wte_residual bundle only; "
            f"got target={args.target}"
        )
    reach_cov_rasters = ["tpi_2km", "tpi_10km", "twi_2km"]
    missing_terr = (
        [p for p in TERRAIN_MULTISCALE_RASTERS.values() if not Path(p).exists()]
        if args.terrain_multiscale_features
        else [
            TERRAIN_MULTISCALE_RASTERS[c]
            for c in reach_cov_rasters
            if not Path(TERRAIN_MULTISCALE_RASTERS[c]).exists()
        ]
        if args.reach_covariate_features
        else []
    )
    if missing_terr:
        raise SystemExit(
            "missing terrain rasters "
            f"(build with build_terrain_covariates.py --only haf,twi_multiscale,wbt): {missing_terr}"
        )

    reach_nodes = pd.read_parquet(gdir / "reach_nodes.parquet")
    reach_manifest = json.loads((gdir / "reach_graph_manifest.json").read_text())
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

    # Query-population filter (e.g. monitoring-only lineage) BEFORE footprint,
    # folds, and the cross-fit priors, so R / deep-IDW / CV blocks are all built
    # from the population the model trains on.
    if args.well_class:
        n0 = len(wells)
        wells = wells[wells["well_class"].isin(args.well_class)].reset_index(drop=True)
        if wells.empty:
            raise SystemExit(f"--well-class {args.well_class}: no wells match")
        log.info("--well-class %s: %d/%d wells kept", args.well_class, len(wells), n0)
    obs_meta_coverage = None
    if args.obs_metadata:
        obs_meta_coverage = join_obs_metadata(wells, args.gwx_wells)
        log.info("obs metadata non-null coverage: %s", obs_meta_coverage)

    # Train-where-you-serve: restrict to wells inside FAC-REM raster coverage BEFORE
    # folds/priors, so the regional prior R, deep datum, and HUC12 folds are all
    # basin-local and every training well carries the FAC signal (no NaN'd critical
    # feature, no train/serve footprint mismatch).
    if args.require_fac:
        fac_probe = sample_fac_rem(
            wells["x5070"].to_numpy("float64"), wells["y5070"].to_numpy("float64")
        )
        keep = np.isfinite(fac_probe)
        n0 = len(wells)
        wells = wells[keep].reset_index(drop=True)
        if len(wells) < args.folds * 10:
            raise SystemExit(
                f"--require-fac kept only {len(wells)} FAC-covered wells -- too few "
                f"for {args.folds}-fold CV; build more FAC-REM basins first"
            )
        log.info(
            "--require-fac: %d/%d wells inside FAC-REM coverage (%.2f%%) -- training "
            "on the serve footprint",
            len(wells),
            n0,
            100 * len(wells) / n0,
        )
    # Streams-Strahler R footprint (wte_residual only): R is the well-free str_top2
    # WTE (huc8/{basin}/str_top2_idw_wte_100m.tif), finite only on the trunk-supported
    # valley. Drop off-trunk wells NOW (audited, not silent) so folds, the deep datum,
    # and the residual base/target are all defined on the same R footprint. Only when
    # str_top2 is actually used (as R, or as the fac_rem-base anomaly feature) -- the
    # relief_idw base needs no str_top2, so it keeps the full FAC footprint.
    uses_str_top2 = args.residual_base in ("str_top2", "fac_rem")
    if args.target == TARGET_WTE_RESIDUAL and uses_str_top2:
        r_probe = sample_str_top2_wte(
            wells["x5070"].to_numpy("float64"), wells["y5070"].to_numpy("float64")
        )
        keep_r = np.isfinite(r_probe)
        n0r = len(wells)
        if int(keep_r.sum()) < args.folds * 10:
            raise SystemExit(
                f"str_top2 R covers only {int(keep_r.sum())} wells -- too few for "
                f"{args.folds}-fold CV; widen the str_top2 anchors/footprint first"
            )
        if (~keep_r).any():
            log.info(
                "str_top2 R: dropping %d/%d wells off the stream-anchored footprint "
                "(no top-2 Strahler R there) -- residual base undefined off-trunk",
                int((~keep_r).sum()),
                n0r,
            )
        wells = wells[keep_r].reset_index(drop=True)

    wells["query_node_idx"] = np.arange(len(wells), dtype="int64")

    # CV folds (HUC12-blocked by default) + within-train val blocks (40 km).
    wx = wells["x5070"].to_numpy("float64")
    wy = wells["y5070"].to_numpy("float64")
    if args.cv_scheme == "huc12":
        cv_unit = huc12_units(wx, wy, args.huc12_polys, args.cv_block_km)
    elif args.cv_scheme == "block":
        cv_unit = spatial_block_ids(wx, wy, args.cv_block_km)
    else:  # huc4 (legacy)
        cv_unit = wells["huc4"].to_numpy().astype(str)
    wells["cv_unit"] = cv_unit
    wells["cv_fold"] = assign_folds(cv_unit, args.folds, args.seed)
    bx = (wx // args.block_size_m).astype("int64")
    by = (wy // args.block_size_m).astype("int64")
    wells["block_40km"] = np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))
    log.info(
        "wells=%d  cv_scheme=%s  cv_units=%d  folds=%d  non-NWIS=%d",
        len(wells),
        args.cv_scheme,
        int(pd.unique(cv_unit).size),
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
    if (
        args.target in (TARGET_WTE, TARGET_WTE_RESIDUAL)
        and not np.isfinite(well_surf_m).all()
    ):
        n_bad = int((~np.isfinite(well_surf_m)).sum())
        raise SystemExit(
            f"{n_bad} wells lack finite {SURFACE_ELEV_COL}; the head-space target "
            "cannot reconstruct DTW for them (add a deliberate drop flag + audit if "
            "a few off-DEM wells must be retained -- do not silently drop)"
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
    elif args.target == TARGET_WTE_RESIDUAL:
        # Keystone: predict the head residual above a smooth regional WTE prior R.
        # target = obs_wte - R (small, well-conditioned); features are anomalies-from-R
        # (translation-invariant -- no absolute elevation memorised); reconstruct
        # dtw = z_surf - (R + resid_hat) = (z_surf - R) - resid_hat. HAND is dropped.
        # Anchors (if requested) are injected as a per-fold head-anomaly BC
        # (anchor_head - R_f(anchor)) computed in the anchor block below.
        wte = wells[OBS_WTE_COL].to_numpy("float64")
        if not np.isfinite(wte).all():
            raise SystemExit(
                f"{int((~np.isfinite(wte)).sum())} non-finite {OBS_WTE_COL}"
            )
        # FAC-REM DTW from the raster registry -- the SAME source as the inference grid
        # (no shard concat, no lexical-sort precedence); finite for every retained well
        # under --require-fac. Sampled up front because it is either an anomaly feature
        # (str_top2 base) or the residual BASE itself (fac_rem base).
        fac_dtw = sample_fac_rem(xy[:, 0], xy[:, 1])
        wells["fac_rem_dtw_m"] = fac_dtw
        fac_joined, fac_finite_frac = True, float(np.isfinite(fac_dtw).mean())
        fac_wte = well_surf_m - fac_dtw  # FAC-REM water-surface ELEVATION
        # The well-free streams-Strahler regional WTE surface (top-2 orders,
        # str_top2_idw_wte_100m.tif from huc8/{basin}); zero well labels -> leakage-free,
        # no cross-fit. Finite for every retained well (off-trunk dropped up front).
        # str_top2 is only consumed by the str_top2 / fac_rem bases (as R, or the
        # fac_rem-base anomaly feature). relief_idw uses none of it -> sample only when
        # needed and require finiteness only there (the up-front drop already enforced
        # the footprint for those bases).
        if uses_str_top2:
            str_top2 = sample_str_top2_wte(xy[:, 0], xy[:, 1])
            if not np.isfinite(str_top2).all():
                raise SystemExit(
                    f"{int((~np.isfinite(str_top2)).sum())} wells lack finite str_top2 "
                    "after the up-front off-trunk drop -- investigate (do not patch)"
                )
        else:
            str_top2 = np.full(len(wells), np.nan)  # diagnostic columns only, unused
        # Choose the regional surface R the residual is taken over. Reconstruction is
        # dtw = (z_surf - R) - resid_hat, so R=fac_rem makes FAC the BASE (dtw =
        # fac_rem_dtw - resid_hat -> the model predicts the correction to FAC) and
        # str_top2 swaps in as the regional-context anomaly feature; R=str_top2 is the
        # default (FAC enters as the anomaly feature / fac-skip anchor); R=relief_idw is
        # the cross-fit relief-aware well-IDW WTE surface (our best regional R), with FAC
        # + deep as the anomaly features.
        if args.residual_base == "fac_rem":
            if not np.isfinite(fac_wte).all():
                raise SystemExit(
                    f"{int((~np.isfinite(fac_wte)).sum())} wells lack finite FAC-REM for "
                    "the --residual-base fac_rem base -- investigate (do not patch)"
                )
            r_wte = fac_wte
            head_anom_cols = [STR_TOP2_WTE_ANOM_COL, DEEP_REGIONAL_WTE_ANOM_COL]
        elif args.residual_base == "relief_idw":
            # Leave-fold-out relief-aware well-IDW of observed WTE (the validated
            # relief-IDW OOF R; pass --r-relief-vw 100). Cross-fit on the GNN's CV folds
            # -> leak-free w.r.t. evaluation; relief lift keeps valley heads off ridges.
            r_wte = crossfit_idw(
                xy,
                wte,
                fold,
                args.idw_k,
                args.idw_power,
                z=well_surf_m,
                vw=args.r_relief_vw,
            )
            if not np.isfinite(r_wte).all():
                raise SystemExit(
                    f"{int((~np.isfinite(r_wte)).sum())} wells lack finite relief-IDW R "
                    "-- investigate (do not patch)"
                )
            head_anom_cols = [FAC_REM_WTE_ANOM_COL, DEEP_REGIONAL_WTE_ANOM_COL]
        elif args.residual_base == "ensemble_median":
            # Per-well median of three physically-distinct level-0 members: the
            # relief-aware well-IDW (vw=100, valley heads off ridges), the plain
            # well-IDW (vw=0, no relief over-mounding -- best on benches), and the
            # FAC-REM water surface (well-free terrain HAND). The median damps the
            # relief-lift over-mound where members disagree while keeping the shared
            # valley-floor level. Both well-IDW members are cross-fit LEAVE-ONE-FOLD-
            # OUT (leak-free w.r.t. evaluation); FAC-REM is well-free.
            relief_wte = crossfit_idw(
                xy,
                wte,
                fold,
                args.idw_k,
                args.idw_power,
                z=well_surf_m,
                vw=args.r_relief_vw,
            )
            simple_wte = crossfit_idw(
                xy,
                wte,
                fold,
                args.idw_k,
                args.idw_power,
                z=well_surf_m,
                vw=0.0,
            )
            for nm, member in (
                ("relief", relief_wte),
                ("simple", simple_wte),
                ("fac", fac_wte),
            ):
                if not np.isfinite(member).all():
                    raise SystemExit(
                        f"{int((~np.isfinite(member)).sum())} wells lack finite "
                        f"{nm} member of ensemble_median R -- investigate (do not patch)"
                    )
            r_wte = np.median(np.vstack([relief_wte, simple_wte, fac_wte]), axis=0)
            head_anom_cols = [FAC_REM_WTE_ANOM_COL, DEEP_REGIONAL_WTE_ANOM_COL]
        else:
            r_wte = str_top2
            head_anom_cols = [FAC_REM_WTE_ANOM_COL, DEEP_REGIONAL_WTE_ANOM_COL]
        wells[REGIONAL_WTE_COL] = r_wte
        deep_wte = crossfit_deep_idw(
            xy,
            xy[deep],
            wte[deep],
            fold,
            fold[deep],
            args.idw_k_deep,
            args.idw_power,
            z_all=well_surf_m,
            z_deep=well_surf_m[deep],
            vw=args.r_relief_vw,
        )
        wells[DEEP_REGIONAL_WTE_COL] = deep_wte
        # target (small residual) + DTW base (z_surf - R; dtw = base - resid_hat).
        wells[WTE_RESIDUAL_TARGET_COL] = wte - r_wte
        wells[WTE_RESID_BASE_COL] = well_surf_m - r_wte
        # Head-space prior anomalies-from-R (the core translation-invariant signal). Both
        # anomaly columns are always written for schema/diagnostic consistency; only
        # head_anom_cols enter the feature set. The FAC anomaly is identically 0 when R IS
        # the FAC surface, so str_top2's anomaly takes its place in fac_rem mode.
        wells[FAC_REM_WTE_ANOM_COL] = fac_wte - r_wte
        wells[STR_TOP2_WTE_COL] = str_top2
        wells[STR_TOP2_WTE_ANOM_COL] = str_top2 - r_wte
        wells[DEEP_REGIONAL_WTE_ANOM_COL] = deep_wte - r_wte
        # Ensemble members AS FEATURES (not a blended base): let the GNN see where the
        # members DISAGREE. Adds the plain vw=0 well-IDW anomaly-over-R; in relief_idw
        # mode R is the vw=100 member, so (simple - R) IS the relief-lift disagreement
        # (negative where relief over-mounds a bench). Leak-free (crossfit leave-fold-out)
        # + translation-invariant (difference of two WTE surfaces).
        ensemble_member_anom_cols = []
        if args.ensemble_member_features:
            simple_wte_feat = crossfit_idw(
                xy, wte, fold, args.idw_k, args.idw_power, z=well_surf_m, vw=0.0
            )
            if not np.isfinite(simple_wte_feat).all():
                raise SystemExit(
                    f"{int((~np.isfinite(simple_wte_feat)).sum())} wells lack finite "
                    "simple-IDW (vw=0) ensemble-member feature -- investigate (do not patch)"
                )
            wells[SIMPLE_IDW_WTE_COL] = simple_wte_feat
            wells[SIMPLE_IDW_WTE_ANOM_COL] = simple_wte_feat - r_wte
            ensemble_member_anom_cols = [SIMPLE_IDW_WTE_ANOM_COL]
        # Exogenous target-blind covariates (terrain relief + ETRM fluxes + gridMET
        # climate), always sampled in this mode (not gated behind --relief-etrm).
        for col, vals in sample_relief_etrm(xy[:, 0], xy[:, 1], well_surf_m).items():
            wells[col] = vals
        for col, vals in sample_gridmet(xy[:, 0], xy[:, 1]).items():
            wells[col] = vals
        # Multi-scale terrain-position family (height-above-floor + TPI + TWI x 4 scales).
        if args.terrain_multiscale_features:
            for col, vals in sample_terrain_multiscale(xy[:, 0], xy[:, 1]).items():
                wells[col] = vals
                log.info("  %s: %.3f finite frac", col, float(np.isfinite(vals).mean()))
        # Observable evidence bank (NDVI greenness here; the GSW wet-reach distance is
        # added after the flowline geom loads below).
        if args.evidence_features:
            for col, vals in sample_evidence_ndvi(xy[:, 0], xy[:, 1]).items():
                wells[col] = vals
                log.info("  %s: %.3f finite frac", col, float(np.isfinite(vals).mean()))
        if args.irrigation_features:
            for col, vals in sample_irrigation(xy[:, 0], xy[:, 1]).items():
                wells[col] = vals
                log.info(
                    "  %s: %.3f finite frac, %.3f irrigated frac (>0)",
                    col,
                    float(np.isfinite(vals).mean()),
                    float((vals > 0).mean()),
                )
        target_col = WTE_RESIDUAL_TARGET_COL
        regional_prior_col = REGIONAL_WTE_COL  # carried; the DTW base is wte_resid_base
        query_feature_cols = (
            head_anom_cols
            + ensemble_member_anom_cols
            + RELIEF_ETRM_FEATURE_COLS
            + CLIMATE_FEATURE_COLS
            + (EVIDENCE_FEATURE_COLS if args.evidence_features else [])
            # fac_rem_dtw_m is already on the wells table (sampled up front); adding the
            # name here surfaces it as a direct query-node feature + lands it in the manifest.
            + (["fac_rem_dtw_m"] if args.fac_rem_feature else [])
            + (
                TERRAIN_MULTISCALE_FEATURE_COLS
                if args.terrain_multiscale_features
                else []
            )
            + (IRRIGATION_FEATURE_COLS if args.irrigation_features else [])
        )
        log.info(
            "target=wte_residual  residual_base=%s  R=%s  R-MAD(DTW)=%.2f m  "
            "R-RMSE(DTW)=%.2f m  fac finite frac=%.3f  features=%s",
            args.residual_base,
            {
                "fac_rem": "fac_rem(z_surf-fac_rem_dtw)",
                "relief_idw": f"relief_idw(crossfit well-IDW WTE, vw={args.r_relief_vw:g})",
                "str_top2": "str_top2(streams-Strahler, well-free)",
                "ensemble_median": f"ensemble_median(relief-IDW vw={args.r_relief_vw:g} "
                "/ simple-IDW vw=0 / FAC-REM surface)",
            }[args.residual_base],
            float(np.nanmedian(np.abs((well_surf_m - r_wte) - dtw))),
            float(np.sqrt(np.nanmean(((well_surf_m - r_wte) - dtw) ** 2))),
            fac_finite_frac,
            query_feature_cols,
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
    # GSW wet-reach distance (the second evidence feature). Computed here so it reuses
    # the already-loaded flowline rep-points instead of re-reading the 2.69M-row geom;
    # the column names were appended to query_feature_cols above.
    if args.evidence_features:
        dwr = dist_to_wet_reach(
            xy,
            geom["cx"].to_numpy("float64"),
            geom["cy"].to_numpy("float64"),
            args.gsw_wet_threshold,
            args.wet_search_km,
        )
        wells["dist_to_wet_reach_m"] = dwr
        wells["log1p_dist_to_wet_reach_m"] = np.log1p(dwr)
    if args.octant_lateral:
        lat = build_octant_lateral_edges(
            xy, geom, comid_to_idx, args.octant_k_search, args.octant_sectors
        )
    else:
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
    if args.octant_lateral:
        log.info(
            "lateral edges: %d (OCTANT: %d wells, %d sectors, k_search=%d, mean %.2f "
            "edges/well, %d wells with all sectors filled)",
            len(lat),
            len(wells),
            args.octant_sectors,
            args.octant_k_search,
            len(lat) / max(len(wells), 1),
            int((lat.groupby("query_node_idx").size() == args.octant_sectors).sum()),
        )
    else:
        log.info(
            "lateral edges: %d (%d wells x knn=%d)",
            len(lat),
            len(wells),
            args.knn_lateral,
        )
    if args.fac_rem_feature:
        log.info(
            "fac-rem-feature ON: fac_rem_dtw_m added as a direct query feature "
            "(finite frac %.3f, med %.1f m)",
            float(np.isfinite(wells["fac_rem_dtw_m"]).mean()),
            float(np.nanmedian(wells["fac_rem_dtw_m"])),
        )

    # --- regional-HAND-along-flowpath features (item 1) + mainstem-read edges (item 2)
    # Both share ONE downstream-datum walk (per-reach), projected onto each well via its
    # rank-0 attached reach. Order band mirrors net_dist_mainstem / R so the feature, the
    # read edge, and the prior share a single mainstem definition. Computed AFTER lateral
    # edges so the rank-0 attachment is available.
    ds_datum_block = None
    mainstem_read_block = None
    channel_edges = None
    if (
        args.ds_datum_features
        or args.mainstem_read
        or args.wet_propagation_features
        or args.upstream_accumulation_features
        or args.portfolio_read
    ):
        channel_edges = pd.read_parquet(gdir / "channel_edges.parquet")
    if args.ds_datum_features or args.mainstem_read or args.portfolio_read:
        order_band = int(reach_manifest.get("mainstem_order_band", 1))
        dd = downstream_datum(reach_nodes, channel_edges, order_band=order_band)
        proj = project_datum_to_queries(dd, lat, reach_nodes, well_surf_m)
        down = channel_edges[channel_edges["direction"] == 1]
        n_braids = int(len(down) - down["src_reach_idx"].nunique())

    if args.ds_datum_features:
        wells["ds_datum_drop_m"] = well_surf_m - proj["datum_elev"]
        wells["log1p_ds_datum_dist_m"] = np.log1p(proj["lat0"] + proj["datum_dist"])
        wells["ds_datum_order_rel"] = proj["datum_order"] - proj["reach0_strah"]
        wells["ds_datum_missing"] = proj["datum_missing"]
        query_feature_cols = query_feature_cols + DS_DATUM_FEATURE_COLS
        drop = wells["ds_datum_drop_m"].to_numpy("float64")
        dist_m = np.expm1(wells["log1p_ds_datum_dist_m"].to_numpy("float64"))
        miss_frac = float(np.mean(proj["datum_missing"]))
        log.info(
            "ds-datum-features ON (order_band=%d): drop med/p90 %.1f/%.1f m, dist "
            "med/p90 %.0f/%.0f m, missing %.3f, braids-resolved %d",
            order_band,
            float(np.nanmedian(drop)),
            float(np.nanpercentile(drop, 90)),
            float(np.nanmedian(dist_m)),
            float(np.nanpercentile(dist_m, 90)),
            miss_frac,
            n_braids,
        )
        if miss_frac > 0.02:
            log.warning(
                "ds-datum missing fraction %.3f > 2%% -- net_dist_mainstem was 1.00 on "
                "attached reaches; investigate the walk before trusting the feature",
                miss_frac,
            )
        ds_datum_block = {
            "order_band": order_band,
            "missing_fraction": miss_frac,
            "n_braids_resolved": n_braids,
            "feature_cols": DS_DATUM_FEATURE_COLS,
            "leakage_note": (
                "ds_datum_* are relative-elevation + channel-topology only (well "
                "land-surface minus the downstream mainstem-datum reach elevation, "
                "along-network distance, order jump). No absolute elevation / head / "
                "target -> translation-invariant + leak-free (RGA-safe)."
            ),
        }

    if args.mainstem_read:
        # One query->datum read edge per well whose downstream walk reached a datum.
        has_datum = proj["datum_reach_idx"] >= 0
        qidx = wells["query_node_idx"].to_numpy("int64")
        me = pd.DataFrame(
            {
                "query_node_idx": qidx[has_datum],
                "reach_node_idx": proj["datum_reach_idx"][has_datum],
                "rel_elev_query_datum_m": (well_surf_m - proj["datum_elev"])[has_datum],
                "log1p_ds_datum_dist_m": np.log1p(proj["lat0"] + proj["datum_dist"])[
                    has_datum
                ],
                "datum_order_rel": (proj["datum_order"] - proj["basin_max_order"])[
                    has_datum
                ],
                "datum_is_self": proj["datum_is_self"][has_datum],
                "log1p_datum_da_km2": np.log1p(np.clip(proj["datum_da"], 0, None))[
                    has_datum
                ],
            }
        )
        me[["query_node_idx", "reach_node_idx", *MS_EDGE_FEATURE_COLS]].to_parquet(
            gdir / "mainstem_edges.parquet"
        )
        cov = float(has_datum.mean())
        log.info(
            "mainstem-read ON (order_band=%d): %d read edges, coverage %.3f "
            "(one edge per well whose downstream walk reached a datum)",
            order_band,
            int(has_datum.sum()),
            cov,
        )
        mainstem_read_block = {
            "order_band": order_band,
            "edge_count": int(has_datum.sum()),
            "coverage_fraction": cov,
            "ms_edge_feature_cols": MS_EDGE_FEATURE_COLS,
            "leakage_note": (
                "mainstem read edges carry relative-elevation + topological attrs only "
                "(query-vs-datum rel-elev, along-network distance, order/self flags, "
                "datum drainage). No absolute elevation / head / target -> leak-free. "
                "The edge lets the query attend to the datum reach's LEARNED state."
            ),
        }

    # --- reach-side augmentation (materialized reach_nodes) --------------------------
    # Shared rep-point coords + GSW occurrence (sampled ONCE for item-5 wet-prop, 6A.1
    # covariates, 6A.2 upstream wet fraction, and the 6B wet reference site). Any block that
    # mutates reach_nodes flips reach_augmented so the augmented parquet is materialized once.
    reach_feature_cols_out = list(reach_manifest["reach_feature_cols"])
    reach_augmented = False
    rn_idx = reach_nodes["reach_node_idx"].to_numpy("int64")
    rcx = rcy = reach_occ = None
    n_no_rep = 0
    if (
        args.wet_propagation_features
        or args.reach_covariate_features
        or args.portfolio_read
    ):
        rcx, rcy = reach_reppoint_coords(reach_nodes, geom)  # reach_node_idx-indexed
        reach_occ = sample_reach_gsw_occ(rcx, rcy)  # 0-100, reach_node_idx-indexed
        n_no_rep = int((~(np.isfinite(rcx) & np.isfinite(rcy))).sum())
        log.info(
            "reach rep-points: %d/%d reaches lack a rep-point (raster samples -> NaN there)",
            n_no_rep,
            len(reach_nodes),
        )

    # item 5 (frozen; NO-GO, kept for reproducibility) -------------------------------------
    wet_propagation_block = None
    if args.wet_propagation_features:
        wet = np.isfinite(reach_occ) & (reach_occ >= args.gsw_wet_threshold)
        # helper is reach_node_idx-indexed (0..n-1); align back to reach_nodes' row order.
        wetfeat = wet_propagation_reach_features(reach_nodes, channel_edges, wet)
        for c in WET_PROP_REACH_FEATURE_COLS:
            reach_nodes[c] = wetfeat[c].reindex(rn_idx).to_numpy()
        reach_feature_cols_out = reach_feature_cols_out + WET_PROP_REACH_FEATURE_COLS
        reach_augmented = True

        # §5.2 gate projection: rank-0 attached-reach value onto each query (net-dist adds
        # the well's own lateral distance; fraction is a straight gather). Carried for the
        # tabular gate ONLY -- NOT added to query_feature_cols (the item is reach-side).
        r0 = (
            lat[lat["rank"] == 0]
            .drop_duplicates("query_node_idx")
            .set_index("query_node_idx")
        )
        nq = len(wells)
        reach0 = np.full(nq, -1, dtype="int64")
        lat0 = np.full(nq, np.nan)
        qi = r0.index.to_numpy("int64")
        reach0[qi] = r0["reach_node_idx"].to_numpy("int64")
        lat0[qi] = r0["lateral_dist_m"].to_numpy("float64")
        have = reach0 >= 0
        net_raw = np.expm1(wetfeat["log1p_net_dist_wet_m"].to_numpy("float64"))
        upfrac = wetfeat["upstream_wet_fraction"].to_numpy("float64")
        q_net = np.full(nq, np.nan)
        q_upfrac = np.full(nq, np.nan)
        q_net[have] = lat0[have] + net_raw[reach0[have]]
        q_upfrac[have] = upfrac[reach0[have]]
        wells["q_log1p_net_dist_wet_m"] = np.log1p(q_net)
        wells["q_upstream_wet_fraction"] = q_upfrac
        wet_propagation_block = {
            "gsw_wet_threshold_pct": args.gsw_wet_threshold,
            "gsw_occurrence_dir": str(GSW_OCC_DIR),
            "wet_reach_fraction": float(wet.mean()),
            "no_wet_component_fraction": float(
                reach_nodes["no_wet_in_component"].mean()
            ),
            "reaches_without_reppoint": n_no_rep,
            "reach_feature_cols_added": WET_PROP_REACH_FEATURE_COLS,
            "query_gate_cols": WET_PROP_QUERY_GATE_COLS,
            "reach_nodes_materialized": True,
            "leakage_note": (
                "REACH-side + NETWORK-metric surface-water evidence (vs the rejected "
                "query-side + Euclidean evidence bank): a per-reach GSW wet flag "
                "propagated along the FAC channel graph. All features are distances/"
                "fractions/flags of an OBSERVABLE surface-water layer (target-blind, no "
                "head/elevation) -> leak-free. The two q_* columns are the rank-0 "
                "projection for the tabular gate only (not model features)."
            ),
        }

    # Phase 6A.1: covariate bank at reach rep-points --------------------------------------
    reach_covariates_block = None
    reach_cov = None  # raw covariate arrays (reach_node_idx-indexed), reused by 6A.2
    if args.reach_covariate_features:
        reach_cov = sample_reach_covariates(rcx, rcy, reach_occ)
        nan_frac = {c: float(np.isnan(v).mean()) for c, v in reach_cov.items()}
        log.info(
            "6A.1 reach-covariate NaN fractions: %s",
            " ".join(f"{c}={f:.3f}" for c, f in nan_frac.items()),
        )
        # Finiteness rule (the Phase-5 lesson): only raster nodata at a rep-point is
        # tolerated; a column >2% NaN is DROPPED, never shipped as a Phase-5-style hole.
        dropped = [c for c, f in nan_frac.items() if f > 0.02]
        for c in dropped:
            log.warning(
                "6A.1: dropping reach covariate %s (%.3f NaN > 2%%)", c, nan_frac[c]
            )
        kept_cols = [c for c in REACH_COVARIATE_FEATURE_COLS if c not in dropped]
        for c in kept_cols:
            reach_nodes[c] = reach_cov[c][
                rn_idx
            ]  # reach_node_idx -> reach_nodes row order
        reach_feature_cols_out = reach_feature_cols_out + kept_cols
        reach_augmented = True
        reach_covariates_block = {
            "cols": kept_cols,
            "dropped_cols": dropped,
            "nan_fraction_by_col": nan_frac,
            "samplers": "sample_coarse (5070 rasters) + gridMET 4326 transform + GSW tiles",
            "reaches_without_reppoint": n_no_rep,
            "leakage_note": (
                "all rep-point covariate rasters are target-blind (terrain position, ETRM "
                "flux, gridMET climate, GLHYMPS perm + Pelletier sediment, MODIS NDVI, GSW "
                "occurrence). No head/elevation/target -> leak-free (mirrors the query bank)."
            ),
        }

    # Phase 6A.2: length-weighted upstream-catchment means of the 6A.1 locals -------------
    upstream_accum_block = None
    if args.upstream_accumulation_features:
        rn_sorted = reach_nodes.sort_values(
            "reach_node_idx"
        )  # reach_node_idx order (0..n-1)
        down_ptr, _ = build_down_ptr(rn_sorted, channel_edges)
        seg_len = np.expm1(rn_sorted["log1p_length_m"].to_numpy("float64"))
        seg_len = np.where(np.isfinite(seg_len) & (seg_len > 0.0), seg_len, 0.0)
        wet_local = (
            np.isfinite(reach_occ) & (reach_occ >= args.gsw_wet_threshold)
        ).astype("float64")
        # accumulate the raw 6A.1 locals (nan-skipping); recharge upstream drives the table.
        local_stack = np.column_stack(
            [
                reach_cov["r_etrm_recharge_mm"],
                reach_cov["r_precip_mm"],
                reach_cov["r_ndvi_jja"],
                wet_local,
            ]
        )
        up = accumulate_upstream(
            local_stack, seg_len, down_ptr
        )  # (n, 4) reach_node_idx-idx
        up_cols = dict(zip(UPSTREAM_ACCUM_FEATURE_COLS, up.T))
        nan_frac_up = {c: float(np.isnan(v).mean()) for c, v in up_cols.items()}
        log.info(
            "6A.2 upstream-accum NaN fractions: %s",
            " ".join(f"{c}={f:.3f}" for c, f in nan_frac_up.items()),
        )
        dropped_up = [c for c, f in nan_frac_up.items() if f > 0.02]
        for c in dropped_up:
            log.warning("6A.2: dropping %s (%.3f NaN > 2%%)", c, nan_frac_up[c])
        kept_up = [c for c in UPSTREAM_ACCUM_FEATURE_COLS if c not in dropped_up]
        for c in kept_up:
            reach_nodes[c] = up_cols[c][rn_idx]
        reach_feature_cols_out = reach_feature_cols_out + kept_up
        reach_augmented = True
        upstream_accum_block = {
            "cols": kept_up,
            "dropped_cols": dropped_up,
            "nan_fraction_by_col": nan_frac_up,
            "gsw_wet_threshold_pct": args.gsw_wet_threshold,
            "note": (
                "length-weighted upstream-catchment means (Kahn accumulation over the "
                "max-drainage down-pointer, nan-skipping) of the 6A.1 recharge/precip/ndvi "
                "locals + upstream_wet_fraction (always finite: 0.0 on dry catchments). "
                "Phase-5 NaN-distance columns are excluded (adjudicated missingness killer)."
            ),
        }

    # Centralized materialization: if any reach block (item-5 / 6A.1 / 6A.2) added columns,
    # write the augmented reach_nodes ONCE. The bundle's reach_nodes.parquet may be a SYMLINK
    # to the shared statewide reach graph -- unlink FIRST so we replace the link with a
    # bundle-local file instead of clobbering shared data through it.
    if reach_augmented:
        reach_feature_cols_out = list(dict.fromkeys(reach_feature_cols_out))  # dedup
        rn_path = gdir / "reach_nodes.parquet"
        if rn_path.is_symlink():
            log.info(
                "materializing augmented reach_nodes (was a symlink -> %s)",
                rn_path.readlink(),
            )
            rn_path.unlink()
        reach_nodes.to_parquet(rn_path)

    # Phase 6B: reference-site portfolio read -- per-query typed edges DIRECT to <=4 -------
    # heterogeneous reference reaches (ds_datum / up_head / wet / ho_any). A missing site is
    # an ABSENT edge the trainer's segment-softmax renormalizes over; per-type indicators tell
    # the head an absent site from a zero read. dd/proj/order_band are from the gate above.
    portfolio_read_block = None
    if args.portfolio_read:
        qidx = wells["query_node_idx"].to_numpy("int64")
        have = (
            proj["reach0"] >= 0
        )  # every well has a rank-0 lateral attachment in practice
        reach0 = np.where(
            have, proj["reach0"], 0
        )  # safe index; always masked by `have`
        lat0 = proj["lat0"]
        basin_max_q = proj["basin_max_order"]
        rn_sorted = reach_nodes.sort_values("reach_node_idx")
        r_elev_arr = rn_sorted["reach_elev_m"].to_numpy("float64")
        r_strah_arr = rn_sorted["streamorde"].to_numpy("float64")
        r_totda_arr = rn_sorted["totdasqkm"].to_numpy("float64")

        # ds_datum: reuse the rank-0 projection (present where the downstream walk landed).
        ds_present = proj["datum_reach_idx"] >= 0
        ds_site = np.where(ds_present, proj["datum_reach_idx"], 0)
        ds_elev = proj["datum_elev"]
        ds_dist = lat0 + proj["datum_dist"]
        ds_order = proj["datum_order"]
        ds_da = proj["datum_da"]

        # up_head: farthest-upstream head on the attached flowpath (always exists for reach0).
        hd = upstream_head(reach_nodes, channel_edges)
        head_reach_arr = hd["head_reach_idx"].to_numpy("int64")
        up_present = have
        up_site = head_reach_arr[reach0]
        up_elev = hd["head_elev_m"].to_numpy("float64")[reach0]
        up_dist = lat0 + hd["head_dist_m"].to_numpy("float64")[reach0]
        up_order = hd["head_order"].to_numpy("float64")[reach0]
        up_da = hd["head_da_km2"].to_numpy("float64")[reach0]

        # wet: nearest wet reach along the network, resolved to its node identity (missingness-
        # robust return of item-5's physics as an EDGE). reach_occ/rcx from the shared block.
        wet_mask = np.isfinite(reach_occ) & (reach_occ >= args.gsw_wet_threshold)
        serving, net_dist_wet = serving_wet_source(reach_nodes, channel_edges, wet_mask)
        serv_q = np.where(have, serving[reach0], -1)
        wet_present = have & (serv_q >= 0)
        wet_site = np.where(serv_q >= 0, serv_q, 0)
        wet_elev = r_elev_arr[wet_site]
        wet_dist = lat0 + net_dist_wet[reach0]
        wet_order = r_strah_arr[wet_site]
        wet_da = r_totda_arr[wet_site]

        # ho_any: nearest high-order reach by EUCLIDEAN distance in ANY direction (deliberately
        # cross-basin -- the NV closed-basin fallback where downstream never reaches high order).
        ho_ok = dd["datum_is_self"].to_numpy(bool) & np.isfinite(rcx) & np.isfinite(rcy)
        ho_cand = np.where(ho_ok)[0]
        if len(ho_cand) == 0:
            raise SystemExit(
                "portfolio-read: no high-order reach has a rep-point (ho_any empty)"
            )
        ho_tree = cKDTree(np.column_stack([rcx[ho_cand], rcy[ho_cand]]))
        ho_dist, ho_pos = ho_tree.query(xy, k=1)
        ho_site = ho_cand[ho_pos]
        ho_present = np.isfinite(
            ho_dist
        )  # finite for every well with a rep-point candidate
        ho_elev = r_elev_arr[ho_site]
        ho_order = r_strah_arr[ho_site]
        ho_da = r_totda_arr[ho_site]

        def portfolio_rows(site_type, site, elev, dist, is_network, order, da, present):
            oh = {t: float(t == site_type) for t in PORTFOLIO_SITE_TYPES}
            m = present
            return pd.DataFrame(
                {
                    "query_node_idx": qidx[m],
                    "reach_node_idx": site[m].astype("int64"),
                    "site_type": site_type,
                    "rel_elev_query_site_m": (well_surf_m - elev)[m],
                    "log1p_site_dist_m": np.log1p(np.clip(dist, 0, None))[m],
                    "dist_is_network": np.full(int(m.sum()), float(is_network)),
                    "site_order_rel": (order - basin_max_q)[m],
                    "log1p_site_da_km2": np.log1p(np.clip(da, 0, None))[m],
                    "type_ds_datum": oh["ds_datum"],
                    "type_up_head": oh["up_head"],
                    "type_wet": oh["wet"],
                    "type_ho_any": oh["ho_any"],
                }
            )

        pf = (
            pd.concat(
                [
                    portfolio_rows(
                        "ds_datum",
                        ds_site,
                        ds_elev,
                        ds_dist,
                        1,
                        ds_order,
                        ds_da,
                        ds_present,
                    ),
                    portfolio_rows(
                        "up_head",
                        up_site,
                        up_elev,
                        up_dist,
                        1,
                        up_order,
                        up_da,
                        up_present,
                    ),
                    portfolio_rows(
                        "wet",
                        wet_site,
                        wet_elev,
                        wet_dist,
                        1,
                        wet_order,
                        wet_da,
                        wet_present,
                    ),
                    portfolio_rows(
                        "ho_any",
                        ho_site,
                        ho_elev,
                        ho_dist,
                        0,
                        ho_order,
                        ho_da,
                        ho_present,
                    ),
                ],
                ignore_index=True,
            )
            .sort_values(["query_node_idx", "site_type"])
            .reset_index(drop=True)
        )
        pf[
            [
                "query_node_idx",
                "reach_node_idx",
                "site_type",
                *PORTFOLIO_EDGE_FEATURE_COLS,
            ]
        ].to_parquet(gdir / "portfolio_edges.parquet")

        present_by_type = {
            "ds_datum": ds_present,
            "up_head": up_present,
            "wet": wet_present,
            "ho_any": ho_present,
        }
        for t in PORTFOLIO_SITE_TYPES:
            wells[f"portfolio_missing_{t}"] = (~present_by_type[t]).astype("float64")
        query_feature_cols = query_feature_cols + PORTFOLIO_MISSING_COLS
        coverage = {t: float(present_by_type[t].mean()) for t in PORTFOLIO_SITE_TYPES}
        log.info(
            "portfolio-read ON (order_band=%d): %d edges; coverage %s",
            order_band,
            len(pf),
            " ".join(f"{t}={coverage[t]:.3f}" for t in PORTFOLIO_SITE_TYPES),
        )
        portfolio_read_block = {
            "site_types": PORTFOLIO_SITE_TYPES,
            "edge_count": int(len(pf)),
            "coverage_by_type": coverage,
            "order_band": order_band,
            "gsw_wet_threshold_pct": args.gsw_wet_threshold,
            "feature_cols": PORTFOLIO_EDGE_FEATURE_COLS,
            "missing_query_cols": PORTFOLIO_MISSING_COLS,
            "leakage_note": (
                "portfolio read edges carry relative-elevation + topological/observable "
                "attrs only (query-vs-site rel-elev, network/Euclidean distance, order jump, "
                "site drainage, type one-hots). No absolute elevation / head / target -> "
                "leak-free. Each edge lets the query attend to the site reach's LEARNED "
                "state; a missing site is an absent edge (softmax renormalizes)."
            ),
        }

    # Carry the surface datum + observed WTE in both modes (cheap, enables cross-
    # mode diagnostics + the WTE identity check); mode-specific target/priors added.
    extra_keep = [SURFACE_ELEV_COL, OBS_WTE_COL, "well_class", "confinement_class"]
    if args.obs_metadata:
        extra_keep += OBS_METADATA_COLS
    if args.wet_propagation_features:
        extra_keep += WET_PROP_QUERY_GATE_COLS
    if args.target == TARGET_WTE:
        extra_keep += [REGIONAL_WTE_COL, DEEP_REGIONAL_WTE_COL, HAND_WTE_COL]
        if FAC_REM_WTE_COL in wells.columns:
            extra_keep += [FAC_REM_WTE_COL, "fac_rem_dtw_m"]
    elif args.target == TARGET_WTE_RESIDUAL:
        extra_keep += [
            REGIONAL_WTE_COL,
            DEEP_REGIONAL_WTE_COL,
            WTE_RESID_BASE_COL,
            WTE_RESIDUAL_TARGET_COL,
            "fac_rem_dtw_m",
        ]
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
        # Anchor Dirichlet BC value. In TARGET_WTE it is the absolute head_m (one
        # column). In TARGET_WTE_RESIDUAL the BC must live in residual space, so it
        # is a per-fold head-anomaly (anchor_head - R_f(anchor)) -- one leak-safe
        # column per CV fold, which the trainer selects fold-by-fold.
        anchor_bc_mode = "absolute_head"
        anchor_bc_anom_cols = None
        if args.target == TARGET_WTE_RESIDUAL:
            anom = crossfit_anchor_anomaly(
                xy,
                wte,
                fold,
                axy,
                anodes["head_m"].to_numpy("float64"),
                args.idw_k,
                args.idw_power,
            )
            anchor_bc_anom_cols = []
            for fk in sorted(anom):
                col = f"anchor_bc_anom_fold_{fk}"
                ax[col] = anom[fk]
                anchor_bc_anom_cols.append(col)
            if not all(
                np.isfinite(ax[c].to_numpy()).all() for c in anchor_bc_anom_cols
            ):
                raise SystemExit("non-finite anchor BC anomaly (anchor_head - R_f)")
            anchor_bc_mode = "head_anomaly_over_R"
            log.info(
                "anchor BC: per-fold head-anomaly over R, %d folds",
                len(anchor_bc_anom_cols),
            )
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
            "anchor_bc_mode": anchor_bc_mode,
            "anchor_bc_anom_cols": anchor_bc_anom_cols,
            "anchor_bc_units": (
                "m_residual_over_regional_wte_R"
                if anchor_bc_mode == "head_anomaly_over_R"
                else "m_same_datum_as_target_wte"
            ),
        }
        log.info(
            "anchors: %d nodes, %d->reach edges, %d->query edges (k=%d)",
            len(ax),
            len(ar),
            len(aq),
            args.knn_anchor_query,
        )
    else:
        # No anchors this build: remove any stale anchor files from a prior anchored
        # build in this dir, so the manifest ("anchors": null) and the on-disk files
        # can never disagree (which would let the tabular control read dead anchors).
        for stale in (
            "anchor_nodes.parquet",
            "anchor_to_reach_edges.parquet",
            "anchor_to_query_edges.parquet",
        ):
            p = gdir / stale
            if p.exists():
                p.unlink()
                log.info("removed stale anchor file: %s", p.name)

    # Mode-aware target metadata: the trainer/scorer read target_mode to decide how
    # to de-standardize the model's scalar output and reconstruct DTW.
    leakage_notes = [
        "Query features exclude Janssen/Ma/coords/obs DTW (carried for scoring).",
        "Reaches carry no labels; no query->query edges.",
        "Cross-fit priors (regional/deep IDW) are leave-one-CV-fold-out on the SAME "
        f"{args.cv_scheme}-blocked folds the model trains with, so a held-out unit's "
        "own DTW never informs its own prior.",
        "Deep datum: per-HUC6 deepest-quartile THRESHOLD is a global quantile (a "
        "mild relaxation under sub-HUC6 folds); leak-safety is enforced by the "
        "leave-one-fold-out deep cross-fit (held-out fold excluded from the pool).",
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
        dtw_base_col = SURFACE_ELEV_COL
        wte_features = {
            REGIONAL_WTE_COL: "cross-fit leave-one-fold-out IDW of observed WTE",
            DEEP_REGIONAL_WTE_COL: "cross-fit IDW of deep-well observed WTE (direct)",
            FAC_REM_WTE_COL: "joined_from_stacker_features"
            if FAC_REM_WTE_COL in query_feature_cols
            else False,
            HAND_WTE_COL: "z_surf_well - hand_m",
        }
        leakage_notes += [
            "WTE mode: target = observed water-table elevation; loss/scoring on "
            "reconstructed DTW (z_surf_well - wte_hat). |WTE err| == |DTW err|.",
            "Absolute land-surface elevation (z_surf_well_m) IS a feature in WTE "
            "mode; the blocked CV is the overfit monitor (watch train/test gap).",
        ]
    elif args.target == TARGET_WTE_RESIDUAL:
        target_mode = TARGET_WTE_RESIDUAL
        target_units = "m (head residual above the regional WTE prior R)"
        # the model's native output is the residual; we emit the reconstructed head
        # wte_hat = R + residual_hat (what the scorer's head-space block reads).
        native_prediction_col = "gnn_wte_hat_m"
        target_definition = f"{OBS_WTE_COL} - {REGIONAL_WTE_COL} (head residual over R)"
        final_dtw_definition = f"z_surf_well_m - ({REGIONAL_WTE_COL} + residual_hat)"
        dtw_reconstruction = "wte_resid_base_col - native_prediction"
        dtw_base_col = WTE_RESID_BASE_COL  # z_surf - R; dtw = base - resid_hat
        if args.residual_base == "fac_rem":
            wte_features = {
                STR_TOP2_WTE_ANOM_COL: "str_top2 regional WTE - R (regional-context "
                "anomaly; FAC is the base here so str_top2 is the feature)",
                DEEP_REGIONAL_WTE_ANOM_COL: "deep-well WTE IDW - R (deep-regime anomaly)",
            }
            leakage_notes += [
                "wte_residual / residual_base=fac_rem (the FAC-residual approach): R = "
                "the FAC-REM water surface (z_surf - fac_rem_dtw) from fac_rem_registry. "
                "FAC is the residual BASE, so dtw = (z_surf - R) - resid_hat = "
                "fac_rem_dtw - resid_hat -- the model predicts the CORRECTION to FAC's "
                "DTW. str_top2 (well-free streams-Strahler WTE) enters as the regional-"
                "context anomaly feature (str_top2 - R). --fac-skip is moot here (the FAC "
                "anomaly is identically 0) and must NOT be passed.",
                "NOTE: only the head-space WTE diagnostic regional_wte_idw_oof_m (= R) "
                "carries the FAC surface in this mode. The scorer's PANEL 'regional' "
                "predictor is regional_idw_dtw_oof_m (the cross-fit well-IDW DTW prior, "
                "computed in every mode) and is UNAFFECTED. The panel 'fac_rem' predictor "
                "equals the base R here. str_top2-standalone is not in the panel (feature, "
                "not base).",
                "Both R sources are well-free / leakage-free and need NO cross-fit. The "
                "deep WTE prior stays a leave-one-fold-out deep-well IDW.",
            ]
        elif args.residual_base == "relief_idw":
            wte_features = {
                FAC_REM_WTE_ANOM_COL: "(z_surf - fac_rem_dtw) - R, from the FAC raster "
                "registry; NaN outside the built basins (NaN+indicator)",
                DEEP_REGIONAL_WTE_ANOM_COL: "deep-well WTE IDW - R (deep-regime anomaly)",
            }
            leakage_notes += [
                "wte_residual / residual_base=relief_idw: R = the relief-aware well-IDW "
                f"WTE surface (crossfit_idw, vw={args.r_relief_vw:g}) -- our most-accurate "
                "regional R. Unlike str_top2/fac_rem this R IS built from wells, so it is "
                "cross-fit LEAVE-ONE-FOLD-OUT on the GNN's CV folds (a held-out fold's "
                "wells never inform their own R) -- leak-free w.r.t. evaluation. The model "
                "predicts the graph-structured head residual over R; FAC + deep enter as "
                "anomalies-from-R. NO str_top2 is used, so the off-trunk drop is skipped "
                "and the build keeps the full FAC footprint (not the trunk-only subset).",
                "HAND features removed; gridMET aridity KEPT. FAC-REM sourced from "
                "fac_rem_registry (same as the inference grid), not the stacker shard.",
                "Anchors not supported in this mode yet (anchor BC would need a "
                "head-anomaly anchor_head - R(anchor)); the build errors if requested.",
            ]
        elif args.residual_base == "ensemble_median":
            wte_features = {
                FAC_REM_WTE_ANOM_COL: "(z_surf - fac_rem_dtw) - R, from the FAC raster "
                "registry; NaN outside the built basins (NaN+indicator)",
                DEEP_REGIONAL_WTE_ANOM_COL: "deep-well WTE IDW - R (deep-regime anomaly)",
            }
            leakage_notes += [
                "wte_residual / residual_base=ensemble_median: R = the per-well MEDIAN of "
                "three level-0 members -- relief-aware well-IDW WTE (crossfit_idw, "
                f"vw={args.r_relief_vw:g}), plain well-IDW WTE (crossfit_idw, vw=0), and the "
                "FAC-REM water surface (z_surf - fac_rem_dtw). The median damps the relief-"
                "lift over-mounding on benches while retaining the shared valley-floor "
                "level. Both well-IDW members are cross-fit LEAVE-ONE-FOLD-OUT on the GNN's "
                "CV folds (leak-free w.r.t. evaluation); FAC-REM is well-free. Same full FAC "
                "footprint as relief_idw (no str_top2). The model predicts the graph-"
                "structured head residual over R; FAC + deep enter as anomalies-from-R.",
                "HAND features removed; gridMET aridity KEPT. FAC-REM sourced from "
                "fac_rem_registry (same as the inference grid), not the stacker shard.",
                "Anchors not supported in this mode yet (anchor BC would need a "
                "head-anomaly anchor_head - R(anchor)); the build errors if requested.",
            ]
        else:
            wte_features = {
                FAC_REM_WTE_ANOM_COL: "(z_surf - fac_rem_dtw) - R, from the FAC raster "
                "registry; NaN outside the built basins (NaN+indicator)",
                DEEP_REGIONAL_WTE_ANOM_COL: "deep-well WTE IDW - R (deep-regime anomaly)",
            }
            leakage_notes += [
                "wte_residual: target = obs_wte - R (small head residual); features are "
                "anomalies-from-R (translation-invariant -- no absolute elevation fed). "
                "Reconstruct dtw = (z_surf - R) - resid_hat; |WTE err| == |DTW err|.",
                "R = str_top2 streams-Strahler IDW WTE (build_str7_idw_raster, top-2 "
                "orders), sampled from huc8/{basin}/str_top2_idw_wte_100m.tif via "
                "fac_rem_registry.sample_str_top2_wte. Well-free (zero well labels) -> "
                "leakage-free, so R needs NO cross-fit. Off-trunk wells (no top-2 anchor) "
                "are dropped up front; R is finite for every retained well.",
                "HAND features removed; gridMET aridity KEPT (a tabular non-result is "
                "not a GNN non-result). FAC-REM sourced from fac_rem_registry (same as "
                "the inference grid), not the stacker shard table.",
                "Anchors not supported in this mode yet (anchor BC would need a "
                "head-anomaly anchor_head - R(anchor)); the build errors if requested.",
            ]
        if args.evidence_features:
            leakage_notes.append(
                "Evidence bank (--evidence-features): observable target-blind signals "
                "only -- MODIS JJA NDVI + JJA-DJF amplitude (phreatophyte greenness) and "
                "GSW wet-reach distance (occ>=%g%% within %g km). NO FAC-REM head "
                "estimate is a feature; FAC enters only as the fac-skip anchor."
                % (args.gsw_wet_threshold, args.wet_search_km)
            )
        if args.irrigation_features:
            leakage_notes.append(
                "Irrigation features (--irrigation-features): IrrMapper irrigation "
                "frequency (% of 2015-2024 years irrigated, per-state 30 m EPSG:5070 "
                "rasters) at the well. Target-blind satellite land-use -- no observed "
                "WTE enters. Unmapped (outside all state rasters) is set to 0 = 'no "
                "mapped irrigation' (exact for the MT/NV/NM footprint), count logged."
            )
        if args.fac_rem_feature:
            leakage_notes.append(
                "FAC-REM direct feature (--fac-rem-feature): fac_rem_dtw_m (the FAC-REM "
                "shallow DTW prior = properly-solved height above the FAC network) added "
                "as a query-node feature. Target-blind (terrain FAC-REM, no observed WTE) "
                "and translation-invariant (a depth, not an absolute elevation) -> RGA-"
                "safe, no leakage. Gives the query node the shallow height signal DIRECTLY "
                "rather than only entangled in fac_rem_wte_anom_m / on lateral edges. The "
                "graph-topology rep-point rel-elev (hand_fac_m) was rejected in review as "
                "a noisy ~2x-weaker proxy (50% of wells attach to Strahler-0 fingertips, "
                "24% negative, +-8m unstable across the knn; shallow rho <=0.12 vs 0.184)."
            )
        if args.ensemble_member_features:
            wte_features[SIMPLE_IDW_WTE_ANOM_COL] = (
                "plain vw=0 well-IDW WTE - R (ensemble member as a feature); in relief_idw "
                "mode R is the vw=100 member so this is the relief-lift disagreement "
                "(negative where relief over-mounds a bench)"
            )
            leakage_notes.append(
                "Ensemble members as features (--ensemble-member-features): adds "
                "simple_idw_wte_anom_m = plain vw=0 well-IDW WTE - R, so the GNN sees the "
                "relief-vs-simple DISAGREEMENT (the bench over-mound signal) directly "
                "rather than being handed a pre-blended base (ensemble_median washed out). "
                "The vw=0 member is cross-fit LEAVE-ONE-FOLD-OUT (leak-free w.r.t. eval) "
                "and the anomaly is translation-invariant (difference of two WTE surfaces). "
                "FAC-REM already enters as fac_rem_wte_anom_m; relief-IDW is the base frame."
            )
        if args.terrain_multiscale_features:
            for c in TERRAIN_MULTISCALE_FEATURE_COLS:
                wte_features[c] = (
                    "multi-scale terrain position: "
                    + {
                        "haf": "height above local floor z - focal_min(z) (m)",
                        "tpi": "elevation deviation from mean (DevFromMeanElev)",
                        "twi": "multi-scale wetness ln(SCA/tan(slope_scale))",
                    }[c.split("_")[0]]
                    + f" at {c.split('_', 1)[1]}"
                )
            leakage_notes.append(
                "Multi-scale terrain family (--terrain-multiscale-features): "
                "height-above-local-floor + TPI + TWI, 4 scales each (500 m/2 km/5 km/"
                "10 km), sampled from /nas/handily/covariates/terrain "
                "(build_terrain_covariates.py, canonical 100 m grid). The residual over R "
                "is terrain-position-organized (valley floor vs terrace/bench) but the "
                "wired elev_above_coarse_m proxy is ~dead (corr ~0); height-above-local-"
                "floor is the proven lever (GBM R2 0.06->0.13 on obs_wte-R, TPI +0.01 on "
                "top; single-scale TWI ~wash). All target-blind + translation-invariant "
                "(relative elevation / dimensionless) -> RGA-safe, no leakage."
            )
    else:
        target_mode = TARGET_DTW_RESIDUAL
        target_units = "m"
        native_prediction_col = "gnn_residual_hat_m"
        target_definition = f"mean_dtw - {regional_prior_col} (cross-fit IDW prior)"
        final_dtw_definition = f"{regional_prior_col} + residual_hat"
        dtw_reconstruction = "regional_prior_col + native_prediction"
        dtw_base_col = regional_prior_col
        wte_features = None
        leakage_notes.insert(
            0, "Query features = hand_m + cross-fit regional IDW prior (+ deep/RELIEF)."
        )

    if args.wet_propagation_features:
        leakage_notes.append(
            "Network-propagated wet evidence (--wet-propagation-features, item 5): a "
            "per-reach GSW wet flag (occ>=%g%%) propagated along the FAC channel graph "
            "into reach_feature_cols (undirected + downstream net-distance to wet, "
            "upstream wet fraction, no-wet-component flag). REACH-side + NETWORK-metric "
            "(distinct from the rejected query-side + Euclidean evidence bank). "
            "Target-blind observable surface-water layer -> leak-free. reach_nodes.parquet "
            "is MATERIALIZED (augmented copy, symlink replaced)."
            % args.gsw_wet_threshold
        )

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
        "residual_base": args.residual_base
        if target_mode == TARGET_WTE_RESIDUAL
        else None,
        "dtw_base_col": dtw_base_col,
        "wte_features": wte_features,
        "cv_fold_col": "cv_fold",
        "cv_group_col": "block_40km",
        "cv_unit_col": "cv_unit",
        "cv_scheme": f"{args.cv_scheme}-blocked, {args.folds} folds",
        "cv_block_km": args.cv_block_km if args.cv_scheme != "huc4" else None,
        "huc12_polys": args.huc12_polys if args.cv_scheme == "huc12" else None,
        "require_fac": bool(args.require_fac),
        "well_class_filter": args.well_class,
        "obs_metadata": (
            {
                "cols": OBS_METADATA_COLS,
                "source": args.gwx_wells,
                "non_null_coverage": obs_meta_coverage,
                "role": "diagnostics only -- never model features",
            }
            if args.obs_metadata
            else None
        ),
        "reach_feature_cols": reach_feature_cols_out,
        "reach_structural_nan_cols": reach_manifest["reach_structural_nan_cols"],
        "channel_edge_feature_cols": reach_manifest["channel_edge_feature_cols"],
        "query_feature_cols": query_feature_cols,
        "query_diagnostic_cols": QUERY_DIAGNOSTIC_COLS,
        "lateral_edge_feature_cols": LATERAL_EDGE_FEATURE_COLS,
        "relief_etrm_features": bool(args.relief_etrm_features),
        "relief_etrm_feature_cols": RELIEF_ETRM_FEATURE_COLS
        if args.relief_etrm_features
        else [],
        "evidence_features": EVIDENCE_FEATURE_COLS if args.evidence_features else [],
        "irrigation_features": IRRIGATION_FEATURE_COLS
        if args.irrigation_features
        else [],
        "ds_datum": ds_datum_block,
        "mainstem_read": mainstem_read_block,
        "wet_propagation": wet_propagation_block,
        "reach_covariates": reach_covariates_block,
        "upstream_accumulation": upstream_accum_block,
        "portfolio_read": portfolio_read_block,
        "evidence_params": {
            "ndvi_jja": NDVI_JJA,
            "ndvi_djf": NDVI_DJF,
            "gsw_occurrence_dir": str(GSW_OCC_DIR),
            "gsw_wet_threshold_pct": args.gsw_wet_threshold,
            "wet_search_km": args.wet_search_km,
        }
        if args.evidence_features
        else None,
        "anchors": anchor_block,
        "conductance_p": args.conductance_p,
        "lateral_topology": "octant" if args.octant_lateral else "knn",
        "knn_lateral": args.knn_lateral,
        "octant_lateral": bool(args.octant_lateral),
        "octant_params": {
            "sectors": args.octant_sectors,
            "k_search": args.octant_k_search,
            "mean_edges_per_well": float(len(lat) / max(len(wells), 1)),
        }
        if args.octant_lateral
        else None,
        "idw_k": args.idw_k,
        "idw_power": args.idw_power,
        "r_relief_vw": args.r_relief_vw,
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
        ]
        + (
            [
                "Octant lateral edges bucket the k_search nearest rep-points by "
                "azimuth and keep the nearest per sector; sector membership uses "
                "rep-point bearing (not the reach line), and sparse sectors may go "
                "empty (variable edges/well).",
            ]
            if args.octant_lateral
            else []
        ),
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
