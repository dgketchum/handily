"""Step 2 of the CONUS WTE/DTW GNN: national reach graph from NHDPlus V2 VAA.

Builds the framework-agnostic *reach* side of the hetero graph from the cached
national VAA table (``pynhd.nhdplus_vaa()``, ~2.69M COMIDs) -- topology +
attributes only, no raster work:

    reach_nodes.parquet    one row per network COMID: VAA-derived node features
                           (Strahler, drainage, slope, arbolate, FCode class) +
                           along-network distance to the nearest mainstem reach.
    channel_edges.parquet  directed reach->reach edges from hydroseq->dnhydroseq
                           (+1 downstream, -1 reverse) with edge features.
    reach_graph_manifest.json

StreamCat covariates and the well (query) side are added downstream
(``build_conus_graph_inputs.py``) where the graph is restricted to well-relevant
reaches -- fetching StreamCat for all 2.69M COMIDs is wasteful when only the
reaches near wells (+ their channel neighborhood) ever carry messages.

    uv run python utils/build_conus_reach_graph.py \\
        --out-dir /data/ssd2/handily/conus/wte_gnn/graph
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pynhd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_conus_reach_graph")

# VAA passthrough + derived node features (model inputs).
REACH_FEATURE_COLS = [
    "streamorde",
    "log1p_totda_km2",
    "log1p_arbolatesum",
    "lengthkm",
    "pathlength",
    "slope",
    "slopelenkm",
    "divergence",
    "startflag",
    "nhd_is_perennial",
    "nhd_is_intermittent",
    "nhd_is_ephemeral",
    "nhd_is_canal",
    "nhd_is_artificial",
    "net_dist_mainstem_m",
    "log1p_net_dist_mainstem_m",
]
# Structurally-NaN: a reach with no upstream mainstem path is genuinely
# disconnected, not missing data. The trainer imputes + flags these.
REACH_STRUCTURAL_NAN_COLS = ["net_dist_mainstem_m", "log1p_net_dist_mainstem_m"]
CHANNEL_EDGE_FEATURE_COLS = [
    "direction",
    "log_drainage_ratio",
    "strahler_change",
    "log1p_length_km",
    "slope",
]
# NHDPlus V2 FCode -> reach-state class.
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


def build_reach_nodes(vaa: pd.DataFrame) -> pd.DataFrame:
    """Stable-indexed reach node table with derived + FCode-class features."""
    out = pd.DataFrame()
    out["comid"] = vaa["comid"].astype("int64")
    out["reach_node_idx"] = np.arange(len(out), dtype="int64")
    out["hydroseq"] = vaa["hydroseq"].astype("int64")
    out["dnhydroseq"] = vaa["dnhydroseq"].astype("int64")
    out["levelpathi"] = vaa["levelpathi"].astype("int64")
    out["terminalpa"] = vaa["terminalpa"].astype("int64")
    out["reachcode"] = vaa["reachcode"].astype(str).str.zfill(14)
    out["huc8"] = out["reachcode"].str[:8]
    out["huc4"] = out["reachcode"].str[:4]
    out["huc2"] = out["reachcode"].str[:2]

    # streamorde uses -9 (coastal/no-flow) and 0 (diverted minor path) sentinels;
    # treat as missing so they don't corrupt the order feature / strahler_change.
    orde = vaa["streamorde"].astype("float64")
    out["streamorde"] = np.where(orde <= 0, np.nan, orde)
    out["totdasqkm"] = vaa["totdasqkm"].astype("float64")
    out["log1p_totda_km2"] = np.log1p(out["totdasqkm"].clip(lower=0))
    out["log1p_arbolatesum"] = np.log1p(
        vaa["arbolatesu"].astype("float64").clip(lower=0)
    )
    out["lengthkm"] = vaa["lengthkm"].astype("float64")
    out["pathlength"] = vaa["pathlength"].astype("float64")
    # NHDPlus uses -9998 as a slope nodata sentinel; treat as missing.
    slope = vaa["slope"].astype("float64")
    out["slope"] = np.where(slope <= -9998, np.nan, slope)
    out["slopelenkm"] = vaa["slopelenkm"].astype("float64")
    out["divergence"] = vaa["divergence"].astype("float64")
    out["startflag"] = vaa["startflag"].astype("float64")

    fcode = vaa["fcode"].astype("float64").to_numpy()
    cls = np.full(len(fcode), "other", dtype=object)
    for code, name in NHD_FCODE_CLASS.items():
        cls[fcode == code] = name
    out["nhd_class"] = cls
    out["nhd_is_perennial"] = (cls == "perennial").astype("float64")
    out["nhd_is_intermittent"] = (cls == "intermittent").astype("float64")
    out["nhd_is_ephemeral"] = (cls == "ephemeral").astype("float64")
    out["nhd_is_canal"] = (cls == "canal_ditch").astype("float64")
    out["nhd_is_artificial"] = (cls == "artificial_path").astype("float64")
    return out


def build_channel_edges(reach_nodes: pd.DataFrame) -> pd.DataFrame:
    """Directed reach->reach edges from hydroseq -> dnhydroseq (+ reverse).

    NHDPlus hydroseq is unique per network reach; a reach flows to the reach whose
    hydroseq equals this reach's dnhydroseq. dnhydroseq==0 (or pointing outside the
    network table, e.g. coastal terminals) yields no downstream edge.
    """
    hs_to_idx = dict(
        zip(
            reach_nodes["hydroseq"].to_numpy(), reach_nodes["reach_node_idx"].to_numpy()
        )
    )
    src = reach_nodes["reach_node_idx"].to_numpy("int64")
    dn = reach_nodes["dnhydroseq"].to_numpy("int64")
    dst = np.array([hs_to_idx.get(d, -1) for d in dn], dtype="int64")
    valid = dst >= 0
    src, dst = src[valid], dst[valid]
    log.info("downstream edges: %d (of %d reaches)", valid.sum(), len(reach_nodes))

    drain = reach_nodes["totdasqkm"].to_numpy("float64")
    strah = reach_nodes["streamorde"].to_numpy("float64")
    length = reach_nodes["lengthkm"].to_numpy("float64")
    slope = reach_nodes["slope"].to_numpy("float64")

    def frame(a: np.ndarray, b: np.ndarray, direction: int) -> pd.DataFrame:
        ratio = np.where(
            drain[a] > 0, drain[b] / np.where(drain[a] > 0, drain[a], 1), np.nan
        )
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
                # feature describes the reach physically carrying flow on this edge
                "log1p_length_km": np.log1p(
                    np.where(direction == 1, length[a], length[b])
                ),
                "slope": np.where(direction == 1, slope[a], slope[b]),
            }
        )

    down = frame(src, dst, +1)
    up = frame(dst, src, -1)
    return pd.concat([down, up], ignore_index=True)


def network_distance_to_mainstem(
    reach_nodes: pd.DataFrame, channel_edges: pd.DataFrame, min_strahler: int
) -> np.ndarray:
    """Along-network distance (km of channel) to the nearest Strahler>=min reach.

    Undirected channel graph weighted by reach length; a virtual super-source
    linked (weight 0) to every mainstem reach turns the multi-source shortest path
    into one Dijkstra pass. Reaches in a component with no mainstem reach -> NaN.
    """
    n = len(reach_nodes)
    down = channel_edges[channel_edges["direction"] == 1]
    s = down["src_reach_idx"].to_numpy("int64")
    d = down["dst_reach_idx"].to_numpy("int64")
    length_m = reach_nodes["lengthkm"].to_numpy("float64")[s] * 1000.0
    length_m = np.where(np.isfinite(length_m) & (length_m > 0), length_m, 1.0)
    strah = reach_nodes["streamorde"].to_numpy("float64")
    mainstem = np.where(strah >= min_strahler)[0]
    if len(mainstem) == 0:
        return np.full(n, np.nan)
    super_idx = n
    rows = np.concatenate([s, mainstem])
    cols = np.concatenate([d, np.full(len(mainstem), super_idx)])
    data = np.concatenate([length_m, np.zeros(len(mainstem))])
    g = csr_matrix((data, (rows, cols)), shape=(n + 1, n + 1))
    dist = dijkstra(g, directed=False, indices=super_idx)[:n]
    dist[~np.isfinite(dist)] = np.nan
    return dist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/data/ssd2/handily/conus/wte_gnn/graph")
    ap.add_argument("--vaa-parquet", default=None, help="override cached VAA path")
    ap.add_argument("--mainstem-strahler", type=int, default=5)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vaa = pynhd.nhdplus_vaa(args.vaa_parquet)
    log.info("loaded VAA: %d reaches", len(vaa))

    reach_nodes = build_reach_nodes(vaa)
    channel_edges = build_channel_edges(reach_nodes)
    net = network_distance_to_mainstem(
        reach_nodes, channel_edges, args.mainstem_strahler
    )
    reach_nodes["net_dist_mainstem_m"] = net
    reach_nodes["log1p_net_dist_mainstem_m"] = np.log1p(net)
    log.info(
        "net-dist to Strahler>=%d: %d/%d reaches connected (%.1f%%)",
        args.mainstem_strahler,
        int(np.isfinite(net).sum()),
        len(net),
        100 * np.isfinite(net).mean(),
    )

    carry = [
        "comid",
        "reach_node_idx",
        "hydroseq",
        "dnhydroseq",
        "levelpathi",
        "terminalpa",
        "reachcode",
        "huc8",
        "huc4",
        "huc2",
        "totdasqkm",
        "nhd_class",
    ]
    reach_nodes[carry + REACH_FEATURE_COLS].to_parquet(out_dir / "reach_nodes.parquet")
    channel_edges[
        ["src_reach_idx", "dst_reach_idx", *CHANNEL_EDGE_FEATURE_COLS]
    ].to_parquet(out_dir / "channel_edges.parquet")

    manifest = {
        "stage": "reach_graph",
        "crs": "EPSG:5070 (query side); reach nodes are topological (no geometry)",
        "counts": {
            "reach_nodes": int(len(reach_nodes)),
            "channel_edges": int(len(channel_edges)),
            "channel_edges_downstream": int((channel_edges["direction"] == 1).sum()),
        },
        "reach_feature_cols": REACH_FEATURE_COLS,
        "reach_structural_nan_cols": REACH_STRUCTURAL_NAN_COLS,
        "channel_edge_feature_cols": CHANNEL_EDGE_FEATURE_COLS,
        "reach_carry_cols": carry,
        "mainstem_strahler": args.mainstem_strahler,
        "edge_schema": "directed reach->reach from hydroseq->dnhydroseq; +1 downstream, -1 reverse",
        "fcode_classes": NHD_FCODE_CLASS,
        "source": "pynhd.nhdplus_vaa() national NHDPlus V2 (cached)",
        "notes": [
            "Reaches carry NO labels (leak-free).",
            "StreamCat covariates + well/query side + lateral edges added by "
            "build_conus_graph_inputs.py on the well-relevant reach subset.",
        ],
    }
    (out_dir / "reach_graph_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(
        "wrote reach graph -> %s (R=%d nodes, channel=%d edges)",
        out_dir,
        len(reach_nodes),
        len(channel_edges),
    )


if __name__ == "__main__":
    main()
