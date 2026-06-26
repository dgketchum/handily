"""Step 2 (FAC variant): reach graph from the FAC channel-head networks.

Drop-in replacement for ``build_conus_reach_graph.py`` -- it writes the SAME four
products (``reach_nodes.parquet``, ``channel_edges.parquet``,
``reach_graph_manifest.json``, and a FAC flowline-geom parquet) with the SAME column
NAMES (``streamorde``, ``totdasqkm``, ``log1p_totda_km2``, ``reach_elev_m``,
``comid``) so ``build_conus_graph_inputs.py`` + ``train_conus_gnn.py`` consume it
unchanged. The ONLY thing that changes between an NHD bundle and a FAC bundle is the
reach substrate: nodes (FAC morphology + seed/evidence features), channel edges
(FAC topology), the geom rep-points lateral edges attach to, and the per-reach
features the reach encoder sees.

Source: per-basin ``fac_channel_heads.fgb`` (the persisted FAC channel network, one
LineString per reach in EPSG:5070) for the ``--require-fac`` basins in
``utils/fac_rem_registry.py``. The basins are concatenated into ONE national-style
reach table (disjoint components); ``reach_node_idx`` is global, topology is matched
WITHIN a basin (node-coordinate join keyed by basin so geographically-disjoint
basins never link).

Reach NODE features are FAC **morphology + seed/evidence ONLY** -- the channel-head
SOLVE outputs (``channel_head_m``, ``head_depth_m``, ``bed_elev_m``, ``h_upper_m``,
``r_target_m`` ...) are FAC's "answer" and are EXCLUDED. FAC enters the GNN only as
the fac-skip anchor on its DTW target-estimate (``build_conus_graph_inputs.py``), so
the graph learns the shallow correction from observables + topology, not from FAC's
rough longitudinal solve. ``strahler`` here is a clean drainage-ordered field
(0 = smallest headwaters ~1 km2, monotone to the basin trunk) -- 0 is a REAL low
order, NOT an NHD-style sentinel, so it is passed through unchanged.

    uv run python utils/build_conus_fac_reach_graph.py \\
        --out-dir /data/ssd2/handily/conus/wte_gnn/graph_fac \\
        --geom-out /data/ssd2/handily/conus/wte_gnn/fac_flowline_geom.parquet
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fac_rem_registry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_conus_fac_reach_graph")

# FAC morphology + seed/evidence node features (NO head-solve outputs). The user-
# specified set; log1p_net_dist_mainstem_m is the structural-NaN companion of
# net_dist_mainstem_m (mirrors the NHD builder's pairing).
REACH_FEATURE_COLS = [
    "streamorde",  # FAC strahler (0..6; 0 is a real low order, not a sentinel)
    "log1p_totda_km2",  # log1p(drainage_km2)
    "log1p_length_m",
    "relief_m",
    "slope",  # relief_m / length_m
    "net_dist_mainstem_m",
    "log1p_net_dist_mainstem_m",
    "seed_strength",
    "seed_ndvi_q",
    "seed_support_fraction",
    "topo_pin_weight",
    "topo_dist_to_seed_m",
    "topo_down_weight",
]
# Structurally-NaN (impute + missingness flag downstream, never an error): a reach
# with no upstream mainstem path / no computable relief is genuinely missing, not bad.
REACH_STRUCTURAL_NAN_COLS = [
    "net_dist_mainstem_m",
    "log1p_net_dist_mainstem_m",
    "relief_m",
    "slope",
]
# Same Darcy edge attrs + names as the NHD builder so the edge encoder + directional-
# edge path (which keys on `direction`) are framework-agnostic.
CHANNEL_EDGE_FEATURE_COLS = [
    "direction",
    "log_drainage_ratio",
    "strahler_change",
    "log1p_length_km",
    "slope",
    "rel_elev_grad",
    "conductance",
]

# Source columns read from each fac_channel_heads.fgb.
SRC_COLS = [
    "stream_id",
    "reach_id",
    "strahler",
    "length_m",
    "up_elev_m",
    "down_elev_m",
    "relief_m",
    "up_node_x",
    "up_node_y",
    "down_node_x",
    "down_node_y",
    "drainage_km2",
    "seed_ndvi_q",
    "seed_support_fraction",
    "seed_strength",
    "topo_pin_weight",
    "topo_dist_to_seed_m",
    "topo_down_weight",
]


def _qkey(basin: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-basin node key: (basin, x, y) quantized to 0.1 m, as one string.

    Quantizing guards float-repr jitter; keying by basin keeps geographically-
    disjoint basins from accidentally linking on a shared coordinate value.
    """
    xr = np.round(np.asarray(x, "float64"), 1)
    yr = np.round(np.asarray(y, "float64"), 1)
    return np.array([f"{basin}|{a:.1f}|{b:.1f}" for a, b in zip(xr, yr)], dtype=object)


def _basin_of(fac_path: str) -> str:
    """Basin name from a registry FAC-REM raster path (.../{basin}/rem/...)."""
    return Path(fac_path).parents[2].name


def load_reaches(registry_paths: list[str]) -> pd.DataFrame:
    """Concatenate the per-basin FAC channel networks into one reach table."""
    frames = []
    for fac_path in registry_paths:
        basin = _basin_of(fac_path)
        # fac_channel_heads.fgb lives beside the FAC-REM depth raster in the registry.
        fgb = Path(fac_path).parent / "fac_channel_heads.fgb"
        if not fgb.exists():
            raise SystemExit(f"missing FAC channel network for {basin}: {fgb}")
        g = gpd.read_file(fgb)
        if g.crs is None or g.crs.to_epsg() != 5070:
            raise SystemExit(f"{basin}: expected EPSG:5070, got {g.crs}")
        keep = [c for c in SRC_COLS if c in g.columns]
        missing = set(SRC_COLS) - set(keep)
        if missing:
            raise SystemExit(f"{basin}: fac_channel_heads.fgb lacks {sorted(missing)}")
        sub = g[keep].copy()
        sub["basin"] = basin
        # on-line representative point (midpoint along the LineString) for lateral
        # attachment -- truer to a sinuous reach than the straight node midpoint.
        rep = g.geometry.interpolate(0.5, normalized=True)
        sub["cx"] = rep.x.to_numpy("float64")
        sub["cy"] = rep.y.to_numpy("float64")
        frames.append(sub)
        log.info("  %-22s %6d reaches", basin, len(sub))
    df = pd.concat(frames, ignore_index=True)
    df["reach_node_idx"] = np.arange(len(df), dtype="int64")
    df["comid"] = df["reach_node_idx"]  # synthetic, globally-unique, internal-only
    return df


def build_reach_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """Stable-indexed reach node table with FAC morphology + seed/evidence features."""
    out = pd.DataFrame()
    out["comid"] = df["comid"].astype("int64")
    out["reach_node_idx"] = df["reach_node_idx"].astype("int64")
    out["basin"] = df["basin"].astype(str)
    out["stream_id"] = df["stream_id"].astype("int64")
    out["reach_id"] = df["reach_id"].astype("int64")

    # strahler is already a clean drainage-ordered integer (0..6); pass through.
    out["streamorde"] = df["strahler"].astype("float64")
    out["totdasqkm"] = df["drainage_km2"].astype("float64")
    out["log1p_totda_km2"] = np.log1p(out["totdasqkm"].clip(lower=0))
    length_m = df["length_m"].astype("float64")
    out["log1p_length_m"] = np.log1p(length_m.clip(lower=0))
    out["relief_m"] = df["relief_m"].astype("float64")
    # slope = relief / length (dimensionless rise/run); NaN where relief NaN (rare,
    # structural). Guard a zero-length reach so it does not divide to inf.
    out["slope"] = out["relief_m"] / length_m.where(length_m > 0, np.nan)

    # reach elevation: the carried NON-feature datum for rel-elev edge/lateral attrs.
    # mean of the two endpoint DEM samples; fall back to the one finite endpoint when
    # the other fell on DEM nodata (~2.5% of down nodes), NaN only when both are gone.
    up = df["up_elev_m"].to_numpy("float64")
    dn = df["down_elev_m"].to_numpy("float64")
    both = np.isfinite(up) & np.isfinite(dn)
    out["reach_elev_m"] = np.where(
        both, (up + dn) / 2.0, np.where(np.isfinite(up), up, dn)
    )

    # seed / topology priors (observable evidence; NOT the head solve).
    out["seed_strength"] = df["seed_strength"].astype("float64")
    out["seed_ndvi_q"] = df["seed_ndvi_q"].astype("float64")
    out["seed_support_fraction"] = df["seed_support_fraction"].astype("float64")
    out["topo_pin_weight"] = df["topo_pin_weight"].astype("float64")
    out["topo_dist_to_seed_m"] = df["topo_dist_to_seed_m"].astype("float64")
    out["topo_down_weight"] = df["topo_down_weight"].astype("float64")
    return out


def build_channel_edges(
    df: pd.DataFrame, reach_nodes: pd.DataFrame, conductance_p: float
) -> pd.DataFrame:
    """Directed reach->reach edges where one reach's DOWN node == another's UP node.

    A coordinate JOIN (not a dict) so confluences (many src -> one dst) AND
    diffluences/braids (one src -> many dst, the ~handful of non-unique up-nodes)
    both emit the correct edge set. Keyed per basin so disjoint basins never link.
    """
    up_key = _qkey(df["basin"].to_numpy(), df["up_node_x"], df["up_node_y"])
    dn_key = _qkey(df["basin"].to_numpy(), df["down_node_x"], df["down_node_y"])
    idx = df["reach_node_idx"].to_numpy("int64")

    up_map = pd.DataFrame({"up_key": up_key, "dst_reach_idx": idx})
    src_map = pd.DataFrame({"up_key": dn_key, "src_reach_idx": idx})
    # src.down == dst.up  =>  src flows into dst.
    pairs = src_map.merge(up_map, on="up_key", how="inner")
    pairs = pairs[pairs["src_reach_idx"] != pairs["dst_reach_idx"]]
    src = pairs["src_reach_idx"].to_numpy("int64")
    dst = pairs["dst_reach_idx"].to_numpy("int64")
    log.info(
        "downstream edges: %d (of %d reaches; %d have a downstream reach)",
        len(src),
        len(df),
        len(np.unique(src)),
    )

    drain = reach_nodes["totdasqkm"].to_numpy("float64")
    strah = reach_nodes["streamorde"].to_numpy("float64")
    length_km = df["length_m"].to_numpy("float64") / 1000.0
    slope = reach_nodes["slope"].to_numpy("float64")
    elev = reach_nodes["reach_elev_m"].to_numpy("float64")

    def frame(a: np.ndarray, b: np.ndarray, direction: int) -> pd.DataFrame:
        ratio = np.where(
            drain[a] > 0, drain[b] / np.where(drain[a] > 0, drain[a], 1), np.nan
        )
        carry_len = np.where(direction == 1, length_km[a], length_km[b])
        carry_drain = np.where(direction == 1, drain[a], drain[b])
        return pd.DataFrame(
            {
                "src_reach_idx": a,
                "dst_reach_idx": b,
                "direction": float(direction),
                "log_drainage_ratio": np.where(
                    np.isfinite(ratio) & (ratio > 0),
                    np.log(np.where(ratio > 0, ratio, 1)),
                    np.nan,
                ),
                "strahler_change": strah[b] - strah[a],
                "log1p_length_km": np.log1p(np.clip(carry_len, 0, None)),
                "slope": np.where(direction == 1, slope[a], slope[b]),
                "rel_elev_grad": elev[b] - elev[a],
                "conductance": np.log1p(np.clip(carry_drain, 0, None))
                - conductance_p * np.log1p(np.clip(carry_len, 0, None)),
            }
        )

    down = frame(src, dst, +1)
    up = frame(dst, src, -1)
    return pd.concat([down, up], ignore_index=True)


def network_distance_to_mainstem(
    df: pd.DataFrame,
    reach_nodes: pd.DataFrame,
    channel_edges: pd.DataFrame,
    order_band: int,
) -> np.ndarray:
    """Along-network distance (m of channel) to the nearest mainstem reach.

    Mainstem is defined PER BASIN as the top ``order_band+1`` Strahler orders (band=0
    => only the basin's max order, guaranteeing >=1 mainstem reach per basin even when
    basins have different max orders). Undirected channel graph weighted by reach
    length; one virtual super-source (weight-0 link to every mainstem reach) turns the
    multi-source shortest path into a single Dijkstra. Basins are disjoint components,
    so distances stay within-basin automatically.
    """
    n = len(reach_nodes)
    down = channel_edges[channel_edges["direction"] == 1]
    s = down["src_reach_idx"].to_numpy("int64")
    d = down["dst_reach_idx"].to_numpy("int64")
    length_m = df["length_m"].to_numpy("float64")[s]
    length_m = np.where(np.isfinite(length_m) & (length_m > 0), length_m, 1.0)

    # The FAC channel network is fragmented (WBT stream extraction yields many
    # disconnected segments, unlike NHD's one connected national network); report it
    # so the net_dist NaN fraction is read as a substrate property, not a bug.
    ncomp, _ = connected_components(
        csr_matrix((np.ones(len(s)), (s, d)), shape=(n, n)), directed=False
    )
    log.info("channel graph: %d connected components over %d reaches", ncomp, n)

    strah = reach_nodes["streamorde"].to_numpy("float64")
    basin = df["basin"].to_numpy()
    mainstem = np.zeros(n, dtype=bool)
    for b in np.unique(basin):
        m = basin == b
        bmax = np.nanmax(strah[m])
        mainstem |= m & (strah >= bmax - order_band)
    mask_idx = np.where(mainstem)[0]
    if len(mask_idx) == 0:
        return np.full(n, np.nan)

    super_idx = n
    rows = np.concatenate([s, mask_idx])
    cols = np.concatenate([d, np.full(len(mask_idx), super_idx)])
    data = np.concatenate([length_m, np.zeros(len(mask_idx))])
    g = csr_matrix((data, (rows, cols)), shape=(n + 1, n + 1))
    dist = dijkstra(g, directed=False, indices=super_idx)[:n]
    dist[~np.isfinite(dist)] = np.nan
    return dist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/data/ssd2/handily/conus/wte_gnn/graph_fac")
    ap.add_argument(
        "--geom-out",
        default="/data/ssd2/handily/conus/wte_gnn/fac_flowline_geom.parquet",
        help="FAC flowline rep-points (comid, cx, cy, geometry) for lateral edges",
    )
    ap.add_argument(
        "--mainstem-order-band",
        type=int,
        default=2,
        help="mainstem = top (band+1) Strahler orders per basin (0 = basin max only). "
        "Default 2 (top-3 orders) lifts net_dist connectivity to ~80-90%% given the "
        "fragmented FAC channel network; the rest stay structural-NaN.",
    )
    ap.add_argument("--conductance-p", type=float, default=0.5)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = fac_rem_registry.existing_registry()
    basins = [_basin_of(p) for p in registry]
    log.info("loading %d FAC basins: %s", len(basins), basins)
    df = load_reaches(registry)
    log.info("concatenated reaches: %d (%d basins)", len(df), df["basin"].nunique())

    reach_nodes = build_reach_nodes(df)
    channel_edges = build_channel_edges(df, reach_nodes, args.conductance_p)
    net = network_distance_to_mainstem(
        df, reach_nodes, channel_edges, args.mainstem_order_band
    )
    reach_nodes["net_dist_mainstem_m"] = net
    reach_nodes["log1p_net_dist_mainstem_m"] = np.log1p(net)
    log.info(
        "net-dist to mainstem (top %d order/basin): %d/%d connected (%.1f%%)",
        args.mainstem_order_band + 1,
        int(np.isfinite(net).sum()),
        len(net),
        100 * np.isfinite(net).mean(),
    )
    for c in ("relief_m", "slope", "reach_elev_m"):
        log.info(
            "  %-24s finite frac %.4f", c, float(np.isfinite(reach_nodes[c]).mean())
        )

    carry = [
        "comid",
        "reach_node_idx",
        "basin",
        "stream_id",
        "reach_id",
        "totdasqkm",
        "reach_elev_m",
    ]
    cols = carry + REACH_FEATURE_COLS
    reach_nodes[cols].to_parquet(out_dir / "reach_nodes.parquet")
    channel_edges[
        ["src_reach_idx", "dst_reach_idx", *CHANNEL_EDGE_FEATURE_COLS]
    ].to_parquet(out_dir / "channel_edges.parquet")

    # FAC flowline geom: rep-points lateral edges attach to (geopandas-readable, with a
    # geometry column). cx/cy are the on-line midpoints; geometry is the same point.
    geom = gpd.GeoDataFrame(
        {
            "comid": reach_nodes["comid"].to_numpy("int64"),
            "cx": df["cx"],
            "cy": df["cy"],
        },
        geometry=gpd.points_from_xy(df["cx"], df["cy"]),
        crs="EPSG:5070",
    )
    geom.to_parquet(args.geom_out)
    log.info("wrote FAC flowline geom (%d rep-points) -> %s", len(geom), args.geom_out)

    manifest = {
        "stage": "reach_graph",
        "substrate": "fac_channels",
        "crs": "EPSG:5070",
        "counts": {
            "reach_nodes": int(len(reach_nodes)),
            "channel_edges": int(len(channel_edges)),
            "channel_edges_downstream": int((channel_edges["direction"] == 1).sum()),
            "basins": int(df["basin"].nunique()),
            "net_dist_connected_frac": float(np.isfinite(net).mean()),
        },
        "basins": basins,
        "reach_feature_cols": REACH_FEATURE_COLS,
        "reach_structural_nan_cols": REACH_STRUCTURAL_NAN_COLS,
        "channel_edge_feature_cols": CHANNEL_EDGE_FEATURE_COLS,
        "reach_carry_cols": carry,
        "mainstem_order_band": args.mainstem_order_band,
        "conductance_p": args.conductance_p,
        "edge_schema": "directed reach->reach where src.down_node == dst.up_node "
        "(coordinate join, per basin); +1 downstream, -1 reverse",
        "geom_path": args.geom_out,
        "source": "per-basin fac_channel_heads.fgb (FAC channel-head networks)",
        "notes": [
            "Reach NODE features are FAC morphology + seed/evidence ONLY -- the head-"
            "solve outputs (channel_head_m/head_depth_m/bed_elev_m/h_upper_m/r_target_m) "
            "are EXCLUDED. FAC enters the GNN only as the fac-skip anchor on its DTW "
            "target-estimate (build_conus_graph_inputs.py), so the graph learns the "
            "correction from observables + topology, not FAC's longitudinal solve.",
            "streamorde = FAC strahler, a clean drainage-ordered field (0 = smallest "
            "headwaters, a REAL low order, NOT an NHD-style sentinel) -- passed through.",
            "comid is a synthetic globally-unique id (= reach_node_idx); used only for "
            "the comid->idx map + geom join, both internal. No NHD COMID semantics.",
            "reach_elev_m is a carried NON-feature: forms the rel-elev edge/lateral "
            "attrs only (nanmean of up/down node DEM samples). Absolute elev is never "
            "a node feature.",
            "Column NAMES match build_conus_reach_graph.py so build_conus_graph_inputs."
            "py + train_conus_gnn.py are framework-agnostic (drop-in NHD<->FAC swap).",
        ],
    }
    (out_dir / "reach_graph_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        "wrote FAC reach graph -> %s (R=%d nodes, channel=%d edges)",
        out_dir,
        len(reach_nodes),
        len(channel_edges),
    )


if __name__ == "__main__":
    main()
