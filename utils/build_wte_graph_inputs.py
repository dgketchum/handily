"""Assemble GNN inputs (hetero graph: reach + query nodes) for the WTE residual model.

Produces a framework-agnostic graph bundle a GNN trainer loads into a PyG
``HeteroData`` (no torch dependency here -- the bundle is plain parquet + json):

    reach_nodes.parquet    reach node features (from fac_channel_heads.fgb)
    query_nodes.parquet    query (well) node features + cross-fit regional OOF
                           + residual-WTE target + source (for the non-NWIS split)
    channel_edges.parquet  directed reach->reach edges (downstream + reverse) w/ features
    lateral_edges.parquet  query->controlling-reach edges (k nearest) w/ features
    graph_manifest.json    counts, feature/edge column lists, target, cv group, leakage notes

Target = residual WTE over the cross-fitted regional prior
(``obs_wte - regional_wte_oof``), the same quantity the tabular residual model in
``build_wte_regional_prior.py`` predicts -- so the GNN slots into the existing
ladder/scorer.

Leakage discipline mirrors ``build_wte_regional_prior.py``: model-input query
features are ``STATIC_RESIDUAL_FEATURES`` only; ``DISALLOWED_RESIDUAL_FEATURES``
(absolute coords / dem / Ma / WT-products / obs-derived) are carried for
diagnostics + scoring but flagged non-features in the manifest. Reaches carry NO
labels and query->query edges are never created, so a held-out well's label can
only reach a training node via query->reach->...->reach->query paths -- the
trainer masks those at train time using ``cv_fold`` / ``block_20km``.

Reach-state enrichment (``--config``): along-network distance to the nearest
mainstem reach (graph BFS, always added), plus reach-centroid sampling of climate
normals, physical covariates (Reitz recharge/ET, Wolock BFI, SSURGO water-table
depth), and NHD perenniality/canal class (FCode snap). WT products (zell/fan) are
deliberately excluded from reach features too.

    uv run python utils/build_wte_graph_inputs.py \\
        --reaches    .../rem/nm_rga_v5_arid_full/fac_channel_heads.fgb \\
        --covariates .../hybrid/gwx/covariates/wte_points.parquet \\
        --oof        .../hybrid/gwx/regional_prior/residual_model_oof_predictions.parquet \\
        --labels     .../evidence/gwx/gwx_wte_labels.parquet \\
        --config     configs/wte/nm_rga_hybrid_gwx.toml \\
        --out-dir    .../hybrid/gwx/graph
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tomllib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_wte_covariate_table import sample_at_points  # noqa: E402
from build_wte_regional_prior import (  # noqa: E402
    DISALLOWED_RESIDUAL_FEATURES,
    STATIC_RESIDUAL_FEATURES,
)

log = logging.getLogger("build_wte_graph_inputs")

# Reach attributes carried as node features (from fac_channel_heads.fgb + derived).
# log1p_drainage_km2 and reach_slope are derived; the rest are passthrough.
REACH_FEATURE_COLS = [
    "strahler",
    "log1p_drainage_km2",
    "length_m",
    "relief_m",
    "reach_slope",
    "up_elev_m",
    "down_elev_m",
    "bed_elev_m",
    "channel_head_m",
    "head_depth_m",
    "h_upper_m",
    "r_target_m",
    "r_max_m",
    "seed_strength",
    "seed_support_fraction",
    "topo_pin_weight",
    "topo_pin_weight_up",
    "topo_down_weight",
    "topo_dist_to_seed_m",
    "topo_gain_to_seed_m",
    "hard_pin",
]
# Reach columns that are structurally NaN on headwater reaches (no downstream
# neighbour => no bed / down-elevation / relief). Preserved as NaN; the trainer
# imputes + adds missingness flags. NOT silently filled here.
REACH_STRUCTURAL_NAN_COLS = [
    "down_elev_m",
    "relief_m",
    "reach_slope",
    "bed_elev_m",
    "channel_head_m",
    "head_depth_m",
    "h_upper_m",
]
CHANNEL_EDGE_FEATURE_COLS = [
    "direction",  # +1 downstream, -1 reverse (upstream)
    "elev_drop_m",  # elev[src] - elev[dst] (positive downstream)
    "log_drainage_ratio",  # log(drainage[dst] / drainage[src])
    "strahler_change",  # strahler[dst] - strahler[src]
    "centroid_dist_m",
]
LATERAL_EDGE_FEATURE_COLS = [
    "lateral_dist_m",
    "log1p_lateral_dist_m",
    "height_above_bed_m",  # query dem - reach bed_elev
    "height_above_head_m",  # query dem - reach channel_head
    "reach_log1p_drainage_km2",
    "reach_strahler",
    "rank",  # 0 == nearest == controlling reach
    "is_controlling",
]

# --- reach-state enrichment (added when --config is supplied) ---------------
# Graph-intrinsic: along-network distance to the nearest high-order (mainstem)
# reach -- a connectivity, not Euclidean, feature.
NET_DIST_COLS = ["net_dist_mainstem_m", "log1p_net_dist_mainstem_m"]
# Physical covariates sampled at reach centroids (WT products zell/fan excluded
# on purpose -- physical reach state, not a stack of other models' water tables).
REACH_COVARIATE_KEYS = [
    "reitz_effective_recharge",
    "reitz_total_recharge",
    "reitz_et",
    "wolock_bfi",
    "ssurgo_wtdep",
]
REACH_CLIMATE_COLS = ["precip_mm_yr", "pet_mm_yr", "aridity_index", "vpd_kpa", "tmax_c"]
# NHD perenniality / canal one-hots + snap quality.
NHD_FLAG_COLS = [
    "nhd_is_perennial",
    "nhd_is_intermittent",
    "nhd_is_ephemeral",
    "nhd_is_canal",
    "nhd_is_artificial",
    "nhd_snap_dist_m",
]
# NHD FCode -> reach-state class (HR FCodes).
NHD_FCODE_CLASS = {
    46006: "perennial",
    46000: "perennial",
    46003: "intermittent",
    46007: "ephemeral",
    33600: "canal_ditch",
    33601: "canal_ditch",
    33603: "canal_ditch",
    55800: "artificial_path",
}
# State controls the plan calls for but no snappable dataset is wired yet.
REACH_STATE_DEFERRED = [
    "diversion points / diverted-reach flags",
    "irrigation extent + source (surface/groundwater) + technology",
    "groundwater pumping / water-use intensity",
    "depth-to-bedrock / surficial geology / aquifer texture",
]


def assign_blocks(x: np.ndarray, y: np.ndarray, block_m: float) -> np.ndarray:
    """Spatial CV block id, matching build_wte_covariate_table.py exactly."""
    bx = (np.asarray(x, dtype="float64") // block_m).astype("int64")
    by = (np.asarray(y, dtype="float64") // block_m).astype("int64")
    return np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))


def build_reach_nodes(reaches: gpd.GeoDataFrame, block_m: float) -> gpd.GeoDataFrame:
    """Reach node table: stable index, derived features, centroid, CV block."""
    out = reaches.copy().reset_index(drop=True)
    out["reach_node_idx"] = np.arange(len(out), dtype="int64")
    out["log1p_drainage_km2"] = np.log1p(out["drainage_km2"].astype("float64"))
    length = out["length_m"].astype("float64")
    out["reach_slope"] = np.where(length > 0, out["relief_m"] / length, np.nan)
    out["hard_pin"] = out["hard_pin"].astype("float64")
    cent = out.geometry.representative_point()
    out["cx"] = cent.x.to_numpy()
    out["cy"] = cent.y.to_numpy()
    km = int(round(block_m / 1000))
    out[f"block_{km}km"] = assign_blocks(out["cx"], out["cy"], block_m)
    return out


def build_channel_edges(
    reach_nodes: gpd.GeoDataFrame, node_precision: int
) -> pd.DataFrame:
    """Directed reach->reach edges from oriented up/down node coords.

    Reuses the orientation already baked into fac_channel_heads.fgb (up_node /
    down_node), so no DEM reload: a node shared by reach u's downstream end and
    reach d's upstream end is a u->d (downstream) edge. Reverse edges are added
    with direction=-1 and features recomputed in the swapped direction.
    """
    n = len(reach_nodes)
    up = list(
        zip(
            np.round(reach_nodes["up_node_x"].to_numpy(), node_precision),
            np.round(reach_nodes["up_node_y"].to_numpy(), node_precision),
        )
    )
    down = list(
        zip(
            np.round(reach_nodes["down_node_x"].to_numpy(), node_precision),
            np.round(reach_nodes["down_node_y"].to_numpy(), node_precision),
        )
    )
    starts_at: dict[tuple[float, float], list[int]] = {}
    ends_at: dict[tuple[float, float], list[int]] = {}
    for i in range(n):
        starts_at.setdefault(up[i], []).append(i)
        ends_at.setdefault(down[i], []).append(i)

    pairs: list[tuple[int, int]] = []
    for node in set(starts_at) | set(ends_at):
        for u in ends_at.get(node, []):
            for d in starts_at.get(node, []):
                if u != d:
                    pairs.append((u, d))
    pairs = sorted(set(pairs))

    drain = reach_nodes["drainage_km2"].to_numpy("float64")
    strah = reach_nodes["strahler"].to_numpy("float64")
    down_e = reach_nodes["down_elev_m"].to_numpy("float64")
    up_e = reach_nodes["up_elev_m"].to_numpy("float64")
    elev_rep = np.where(np.isfinite(down_e), down_e, up_e)
    cx = reach_nodes["cx"].to_numpy("float64")
    cy = reach_nodes["cy"].to_numpy("float64")

    def feat(a: int, b: int, direction: int) -> dict:
        ratio = drain[b] / drain[a] if drain[a] > 0 else np.nan
        return {
            "src_reach_idx": a,
            "dst_reach_idx": b,
            "direction": float(direction),
            "elev_drop_m": elev_rep[a] - elev_rep[b],
            "log_drainage_ratio": np.log(ratio)
            if np.isfinite(ratio) and ratio > 0
            else np.nan,
            "strahler_change": strah[b] - strah[a],
            "centroid_dist_m": float(np.hypot(cx[a] - cx[b], cy[a] - cy[b])),
        }

    rows: list[dict] = []
    for u, d in pairs:
        rows.append(feat(u, d, +1))  # downstream
        rows.append(feat(d, u, -1))  # reverse (upstream)
    return pd.DataFrame(
        rows, columns=["src_reach_idx", "dst_reach_idx", *CHANNEL_EDGE_FEATURE_COLS]
    )


def build_query_nodes(
    cov: gpd.GeoDataFrame,
    oof: gpd.GeoDataFrame,
    labels: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, str]:
    """Query (well) node table: features + cross-fit regional OOF + residual target.

    Inner-joins the covariate table to the OOF predictions on canonical_id (the
    labeled, cross-fit set) and rejoins ``source`` from the GWX labels for the
    non-NWIS headline split. The residual-WTE target comes straight from the OOF.
    """
    method = next(c for c in oof.columns if c.startswith("regional_wte_oof__")).split(
        "__", 1
    )[1]
    oof_cols = {
        "canonical_id": "canonical_id",
        "cv_fold": "cv_fold",
        "obs_wte_m": "obs_wte_m",
        "obs_dtw_m": "obs_dtw_m",
        f"regional_wte_oof__{method}": "regional_wte_oof_m",
        f"regional_dtw_oof__{method}": "regional_dtw_oof_m",
        f"residual_wte_obs__{method}": "target_residual_wte_m",
        f"residual_wte_hat__{method}": "residual_wte_hat_m",
        "depth_bin": "depth_bin",
    }
    oof_slim = oof[list(oof_cols)].rename(columns=oof_cols)

    df = cov.merge(oof_slim, on="canonical_id", how="inner")
    df = df.merge(labels[["canonical_id", "source"]], on="canonical_id", how="left")
    if df["source"].isna().any():
        n = int(df["source"].isna().sum())
        raise SystemExit(f"{n} query nodes did not rejoin a source via canonical_id")

    df = df.reset_index(drop=True)
    df["query_node_idx"] = np.arange(len(df), dtype="int64")
    df["x_5070"] = df.geometry.x.to_numpy()
    df["y_5070"] = df.geometry.y.to_numpy()
    return df, method


def build_lateral_edges(
    query: gpd.GeoDataFrame,
    reach_nodes: gpd.GeoDataFrame,
    knn: int,
) -> pd.DataFrame:
    """k-nearest reach per query by exact point-to-line distance.

    Candidate reaches are the ``max(50, knn*10)`` nearest reach centroids
    (cKDTree); exact shapely line distance to those candidates then selects and
    ranks the k nearest. The candidate pool is wide enough that the true nearest
    reach is included (validated against brute force in the tests).
    """
    reach_geoms = reach_nodes.geometry.to_numpy()
    cent = np.c_[reach_nodes["cx"].to_numpy(), reach_nodes["cy"].to_numpy()]
    tree = cKDTree(cent)
    qxy = np.c_[query.geometry.x.to_numpy(), query.geometry.y.to_numpy()]
    k_cand = min(len(reach_nodes), max(50, knn * 10))
    _, cand = tree.query(qxy, k=k_cand)
    cand = np.atleast_2d(cand)

    qgeom = query.geometry.to_numpy()
    dem = query["dem_m"].to_numpy("float64")
    r_idx = reach_nodes["reach_node_idx"].to_numpy("int64")
    r_bed = reach_nodes["bed_elev_m"].to_numpy("float64")
    r_head = reach_nodes["channel_head_m"].to_numpy("float64")
    r_logdr = reach_nodes["log1p_drainage_km2"].to_numpy("float64")
    r_strah = reach_nodes["strahler"].to_numpy("float64")

    rows: list[dict] = []
    for qi in range(len(query)):
        cids = cand[qi]
        ld = shapely.distance(qgeom[qi], reach_geoms[cids])
        order = np.argsort(ld, kind="stable")[:knn]
        for rank, o in enumerate(order):
            ri = int(cids[o])
            rows.append(
                {
                    "query_node_idx": int(qi),
                    "reach_node_idx": int(r_idx[ri]),
                    "lateral_dist_m": float(ld[o]),
                    "log1p_lateral_dist_m": float(np.log1p(ld[o])),
                    "height_above_bed_m": dem[qi] - r_bed[ri],
                    "height_above_head_m": dem[qi] - r_head[ri],
                    "reach_log1p_drainage_km2": float(r_logdr[ri]),
                    "reach_strahler": float(r_strah[ri]),
                    "rank": int(rank),
                    "is_controlling": float(rank == 0),
                }
            )
    return pd.DataFrame(
        rows, columns=["query_node_idx", "reach_node_idx", *LATERAL_EDGE_FEATURE_COLS]
    )


def network_distance_to_order(
    reach_nodes: gpd.GeoDataFrame, channel_edges: pd.DataFrame, min_strahler: int
) -> np.ndarray:
    """Along-network distance from each reach to the nearest Strahler>=min reach.

    Undirected channel graph weighted by inter-reach centroid distance; a virtual
    super-source linked (weight 0) to every Strahler>=min reach turns the
    multi-source query into one Dijkstra pass. Reaches in a component with no
    high-order reach get NaN (genuinely disconnected from any mainstem).
    """
    n = len(reach_nodes)
    src = channel_edges["src_reach_idx"].to_numpy("int64")
    dst = channel_edges["dst_reach_idx"].to_numpy("int64")
    w = channel_edges["centroid_dist_m"].to_numpy("float64")
    strah = reach_nodes["strahler"].to_numpy("float64")
    mainstem = np.where(strah >= min_strahler)[0]
    if len(mainstem) == 0:
        return np.full(n, np.nan)
    super_idx = n
    rows = np.concatenate([src, mainstem])
    cols = np.concatenate([dst, np.full(len(mainstem), super_idx)])
    data = np.concatenate([w, np.zeros(len(mainstem))])
    g = csr_matrix((data, (rows, cols)), shape=(n + 1, n + 1))
    d = dijkstra(g, directed=False, indices=super_idx)[:n]
    d[~np.isfinite(d)] = np.nan
    return d


def classify_fcode(fcodes: np.ndarray) -> np.ndarray:
    """Map NHD FCode -> reach-state class string (unmapped -> 'other')."""
    f = np.asarray(fcodes, dtype="float64")
    out = np.full(len(f), "other", dtype=object)
    for code, cls in NHD_FCODE_CLASS.items():
        out[f == code] = cls
    return out


def snap_nhd_perenniality(reach_nodes: gpd.GeoDataFrame, nhd_path: str) -> pd.DataFrame:
    """Nearest-NHD-flowline perenniality/canal class per reach (FCode), + snap dist."""
    margin = 0.05
    b = reach_nodes.to_crs(4269).total_bounds
    bbox = (b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin)
    nhd = gpd.read_file(nhd_path, bbox=bbox)
    fcol = "fcode" if "fcode" in nhd.columns else "FCode"
    nhd = nhd[[fcol, "geometry"]].rename(columns={fcol: "fcode"}).to_crs(5070)
    nhd = nhd[nhd.geometry.notna() & ~nhd.geometry.is_empty].reset_index(drop=True)

    left = reach_nodes[["reach_node_idx", "geometry"]].reset_index(drop=True)
    j = gpd.sjoin_nearest(left, nhd, how="left", distance_col="nhd_snap_dist_m")
    j = (
        j.sort_values("nhd_snap_dist_m")
        .drop_duplicates("reach_node_idx")
        .set_index("reach_node_idx")
        .reindex(reach_nodes["reach_node_idx"].to_numpy())
    )
    cls = classify_fcode(j["fcode"].to_numpy())
    out = pd.DataFrame({"reach_node_idx": reach_nodes["reach_node_idx"].to_numpy()})
    out["nhd_perenniality"] = cls
    out["nhd_is_perennial"] = (cls == "perennial").astype("float64")
    out["nhd_is_intermittent"] = (cls == "intermittent").astype("float64")
    out["nhd_is_ephemeral"] = (cls == "ephemeral").astype("float64")
    out["nhd_is_canal"] = (cls == "canal_ditch").astype("float64")
    out["nhd_is_artificial"] = (cls == "artificial_path").astype("float64")
    out["nhd_snap_dist_m"] = j["nhd_snap_dist_m"].to_numpy()
    return out


def sample_reach_covariates(reach_nodes: gpd.GeoDataFrame, cfg: dict) -> pd.DataFrame:
    """Sample physical covariates + climate normals at reach centroids."""
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(reach_nodes["cx"], reach_nodes["cy"]), crs=5070
    )
    out = pd.DataFrame({"reach_node_idx": reach_nodes["reach_node_idx"].to_numpy()})
    covs = cfg.get("covariates", {})
    for key in REACH_COVARIATE_KEYS:
        if key in covs:
            out[key] = sample_at_points(covs[key], pts)
    clim_cfg = cfg.get("climate", {})
    if "gridmet_dir" in clim_cfg:
        from handily.climate_normals import sample_gridmet_normals

        clim = sample_gridmet_normals(pts, clim_cfg["gridmet_dir"])
        for c in REACH_CLIMATE_COLS:
            if c in clim.columns:
                out[c] = clim[c].to_numpy()
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reaches", required=True, help="fac_channel_heads.fgb")
    p.add_argument("--covariates", required=True, help="wte_points.parquet")
    p.add_argument(
        "--oof", required=True, help="residual_model_oof_predictions.parquet"
    )
    p.add_argument(
        "--labels", required=True, help="gwx_wte_labels.parquet (source key)"
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--knn-lateral", type=int, default=3)
    p.add_argument("--block-size-m", type=float, default=20000.0)
    p.add_argument("--node-precision", type=int, default=3)
    p.add_argument("--mainstem-strahler", type=int, default=5)
    p.add_argument(
        "--config",
        default=None,
        help="hybrid TOML; if given, sample reach climate + covariates + NHD state",
    )
    p.add_argument(
        "--nhd",
        default=None,
        help="NHD flowlines (overrides [reach_state].nhd_flowlines in --config)",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reaches = gpd.read_file(args.reaches).to_crs(5070)
    cov = gpd.read_parquet(args.covariates).to_crs(5070)
    oof = gpd.read_parquet(args.oof).to_crs(5070)
    labels = pd.read_parquet(args.labels, columns=["canonical_id", "source"])
    log.info(
        "loaded %d reaches, %d covariate points, %d oof rows",
        len(reaches),
        len(cov),
        len(oof),
    )

    missing = [f for f in STATIC_RESIDUAL_FEATURES if f not in cov.columns]
    if missing:
        raise SystemExit(f"covariate table missing model features: {missing}")
    leak = set(STATIC_RESIDUAL_FEATURES) & DISALLOWED_RESIDUAL_FEATURES
    assert not leak, f"leaky query feature: {leak}"

    reach_nodes = build_reach_nodes(reaches, args.block_size_m)
    channel_edges = build_channel_edges(reach_nodes, args.node_precision)
    query_nodes, method = build_query_nodes(cov, oof, labels)
    lateral_edges = build_lateral_edges(query_nodes, reach_nodes, args.knn_lateral)
    log.info(
        "covariate->oof join kept %d/%d query nodes (%d without cross-fit prediction)",
        len(query_nodes),
        len(cov),
        len(cov) - len(query_nodes),
    )

    # --- reach-state enrichment -------------------------------------------
    reach_feature_cols = list(REACH_FEATURE_COLS)
    reach_struct_nan = list(REACH_STRUCTURAL_NAN_COLS)
    reach_diag_cols: list[str] = []
    state_added: dict[str, list[str]] = {}
    nhd_path_used: str | None = None

    net = network_distance_to_order(reach_nodes, channel_edges, args.mainstem_strahler)
    reach_nodes["net_dist_mainstem_m"] = net
    reach_nodes["log1p_net_dist_mainstem_m"] = np.log1p(net)
    reach_feature_cols += NET_DIST_COLS
    reach_struct_nan += NET_DIST_COLS  # NaN == disconnected from any mainstem
    log.info(
        "net-dist to Strahler>=%d: %d/%d reaches connected",
        args.mainstem_strahler,
        int(np.isfinite(net).sum()),
        len(net),
    )

    if args.config:
        with open(args.config, "rb") as f:
            cfg = tomllib.load(f)
        state = sample_reach_covariates(reach_nodes, cfg)
        reach_nodes = reach_nodes.merge(state, on="reach_node_idx", how="left")
        cov_clim = [
            c
            for c in (*REACH_COVARIATE_KEYS, *REACH_CLIMATE_COLS)
            if c in state.columns
        ]
        reach_feature_cols += cov_clim
        state_added["covariates_climate"] = cov_clim
        # SSURGO maps a seasonal-high water table only in some soil units -> NaN
        # in arid uplands is a meaningful absence, not missing data.
        if "ssurgo_wtdep" in cov_clim:
            reach_struct_nan.append("ssurgo_wtdep")

        nhd_path_used = args.nhd or cfg.get("reach_state", {}).get("nhd_flowlines")
        if nhd_path_used:
            nhd = snap_nhd_perenniality(reach_nodes, nhd_path_used)
            reach_nodes = reach_nodes.merge(nhd, on="reach_node_idx", how="left")
            reach_feature_cols += NHD_FLAG_COLS
            reach_diag_cols.append("nhd_perenniality")
            state_added["nhd"] = NHD_FLAG_COLS
            log.info(
                "NHD perenniality snapped: %s",
                reach_nodes["nhd_perenniality"].value_counts().to_dict(),
            )

    km = int(round(args.block_size_m / 1000))
    reach_keep = [
        "reach_node_idx",
        "stream_id",
        *reach_feature_cols,
        *reach_diag_cols,
        "cx",
        "cy",
        f"block_{km}km",
        "geometry",
    ]
    query_carry = [
        "canonical_id",
        "query_node_idx",
        "source",
        "tier",
        "weight",
        "confinement_class",
        "is_water_table_label",
        "depth_bin",
        "cv_fold",
        f"block_{km}km",
        "dem_m",
        "obs_wte_m",
        "obs_dtw_m",
        "regional_wte_oof_m",
        "regional_dtw_oof_m",
        "residual_wte_hat_m",
        "ma_dtw_m",
        "x_5070",
        "y_5070",
    ]
    query_keep = [
        *query_carry,
        "target_residual_wte_m",
        *STATIC_RESIDUAL_FEATURES,
        "geometry",
    ]

    reach_nodes[reach_keep].to_parquet(out_dir / "reach_nodes.parquet")
    query_nodes[query_keep].to_parquet(out_dir / "query_nodes.parquet")
    channel_edges.to_parquet(out_dir / "channel_edges.parquet")
    lateral_edges.to_parquet(out_dir / "lateral_edges.parquet")

    manifest = {
        "crs": "EPSG:5070",
        "residual_method": method,
        "counts": {
            "reach_nodes": int(len(reach_nodes)),
            "query_nodes": int(len(query_nodes)),
            "channel_edges": int(len(channel_edges)),
            "lateral_edges": int(len(lateral_edges)),
            "query_nodes_non_nwis": int(
                (~query_nodes["source"].isin(["nwis", "ngwmn"])).sum()
            ),
            "query_nodes_nwis": int(
                query_nodes["source"].isin(["nwis", "ngwmn"]).sum()
            ),
        },
        "target_col": "target_residual_wte_m",
        "target_definition": "obs_wte_m - regional_wte_oof_m (cross-fit regional prior)",
        "cv_group_col": f"block_{km}km",
        "cv_fold_col": "cv_fold",
        "reach_feature_cols": reach_feature_cols,
        "reach_structural_nan_cols": reach_struct_nan,
        "reach_diagnostic_cols": reach_diag_cols,
        "query_feature_cols": list(STATIC_RESIDUAL_FEATURES),
        "query_carry_cols_non_features": query_carry,
        "channel_edge_feature_cols": CHANNEL_EDGE_FEATURE_COLS,
        "lateral_edge_feature_cols": LATERAL_EDGE_FEATURE_COLS,
        "knn_lateral": args.knn_lateral,
        "block_size_m": args.block_size_m,
        "node_precision": args.node_precision,
        "edge_schema": {
            "channel": "directed reach->reach; +1 downstream, -1 reverse",
            "lateral": "directed query->reach; rank 0 == controlling reach; no query->query edges",
        },
        "leakage_notes": [
            "Query model features = query_feature_cols only (STATIC_RESIDUAL_FEATURES).",
            "DISALLOWED_RESIDUAL_FEATURES carried as diagnostics/scoring, never model inputs.",
            "Reaches carry no labels; query->query edges never created.",
            "Regional prior is cross-fit OOF; trainer must mask held-out query labels "
            "via cv_fold / block for inductive (non-transductive) training.",
        ],
        "reach_state_added": state_added,
        "reach_state_deferred": REACH_STATE_DEFERRED,
        "mainstem_strahler": args.mainstem_strahler,
        "sources": {
            "reaches": args.reaches,
            "covariates": args.covariates,
            "oof": args.oof,
            "labels": args.labels,
            "config": args.config,
            "nhd": nhd_path_used,
        },
    }
    with open(out_dir / "graph_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(
        "wrote graph bundle -> %s  (R=%d nodes, Q=%d nodes, channel=%d, lateral=%d)",
        out_dir,
        len(reach_nodes),
        len(query_nodes),
        len(channel_edges),
        len(lateral_edges),
    )


if __name__ == "__main__":
    main()
