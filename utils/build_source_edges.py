"""Build spatial source-well edges for the CONUS water-table GNN (assimilation rung 1).

Each query node (dest -- every row, wells and pseudo alike) is connected to its
``k`` nearest ELIGIBLE source wells (real, finite target) in the relief-lifted
coordinate space the production regional prior R uses (vw = the bundle's
``r_relief_vw``), so the GNN can read a nearby well's observed residual directly
("learned IDW") instead of through the diluted writeback -> reach -> lateral path
that rung 0 showed transmits nothing (SOURCE_WELL_ASSIMILATION_PLAN.md section 8).
Emitted as an ADDITIVE ``source_edges.parquet`` next to the bundle -- never
mutates existing bundle files.

Unlike the E3 analog edges (nonlocal by construction), these edges are LOCAL on
purpose -- the leakage guard is NOT topology but the trainer's masked-label
protocol: per epoch only drawn train-pool sources keep their edges, drawn
sources contribute no loss, and a well never keeps an edge from its own site
(same-site edges are dropped here, where "site" is exact-coordinate identity --
collocated obs would leak the ~4-5 m label-ambiguity dispersion straight in).
``src_cv_fold`` is carried for the fold-level guard (`analog_fold_keep`).

Edge features (target-blind, standardized once globally in the trainer):
  log1p_geo_dist_km    = log1p(planar EPSG:5070 distance, km)
  log1p_relief_dist_km = log1p(relief-lifted kNN metric distance, km)
  rel_elev_m           = z_surf(dest) - z_surf(src)  (m, DEM land surface)
  same_basin           = 1.0 if dest and src controlling reaches share a FAC
                         basin (obs should flow along valleys, not across
                         divides -- the Mesilla stream-anchor lesson), else 0.0

Usage:
    uv run python utils/build_source_edges.py \\
        --bundle /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2 \\
        --k 16
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_source_edges")

EDGE_COLS = ["log1p_geo_dist_km", "log1p_relief_dist_km", "rel_elev_m", "same_basin"]


def relief_coords(xy: np.ndarray, z: np.ndarray, vw: float) -> np.ndarray:
    """Relief-lifted 3D metric (x, y, vw*z) -- the production R's neighbourhood."""
    return np.column_stack([xy, vw * np.nan_to_num(z, nan=0.0)])


def controlling_basin(qn: pd.DataFrame, gdir: Path) -> np.ndarray:
    """FAC basin id of each query's controlling reach (-1 where unmapped)."""
    lat = pd.read_parquet(gdir / "lateral_edges.parquet")
    rn = pd.read_parquet(gdir / "reach_nodes.parquet").sort_values("reach_node_idx")
    ctrl = lat[lat["is_controlling"].astype(bool)].drop_duplicates("query_node_idx")
    reach_of = ctrl.set_index("query_node_idx")["reach_node_idx"]
    basin_of_reach = rn.set_index("reach_node_idx")["basin"]
    basin = qn["query_node_idx"].map(reach_of).map(basin_of_reach).fillna(-1).to_numpy()
    codes, _ = pd.factorize(basin)
    codes = codes.astype("int64")
    codes[np.asarray(basin) == -1] = -1
    return codes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--k", type=int, default=16, help="source wells per dest node")
    ap.add_argument("--out", default=None, help="default <bundle>/source_edges.parquet")
    args = ap.parse_args()
    gdir = Path(args.bundle)
    out = Path(args.out) if args.out else gdir / "source_edges.parquet"
    man = json.loads((gdir / "graph_manifest.json").read_text())
    vw = float(man["r_relief_vw"])

    qn = (
        pd.read_parquet(gdir / "query_nodes.parquet")
        .sort_values("query_node_idx")
        .reset_index(drop=True)
    )
    if not (qn["query_node_idx"].to_numpy() == np.arange(len(qn))).all():
        raise SystemExit("query_node_idx is not contiguous positional -- rebuild")
    water = (
        qn["is_water_pseudo"].to_numpy(bool)
        if "is_water_pseudo" in qn.columns
        else np.zeros(len(qn), bool)
    )
    resid = qn["wte_residual_m"].to_numpy("float64")
    eligible = ~water & np.isfinite(resid)
    xy = qn[["x5070", "y5070"]].to_numpy("float64")
    z = qn["z_surf_well_m"].to_numpy("float64")
    basin = controlling_basin(qn, gdir)
    cv_fold = qn["cv_fold"].to_numpy("int64")
    src_idx = np.flatnonzero(eligible)
    log.info(
        "bundle %s: %d query nodes, %d eligible sources, vw=%.0f, k=%d",
        gdir.name,
        len(qn),
        len(src_idx),
        vw,
        args.k,
    )

    # k+pad nearest, then drop self/same-site edges and cut back to k. The pad
    # covers the self hit plus any exactly-collocated eligible rows.
    pad = 4
    tree = cKDTree(relief_coords(xy[src_idx], z[src_idx], vw))
    rd, loc = tree.query(relief_coords(xy, z, vw), k=args.k + pad, workers=-1)
    dest = np.repeat(np.arange(len(qn), dtype="int64"), args.k + pad)
    src = src_idx[loc.ravel()]
    relief_km = rd.ravel() / 1000.0
    geo_km = np.sqrt(((xy[dest] - xy[src]) ** 2).sum(axis=1)) / 1000.0
    same_site = (xy[dest] == xy[src]).all(axis=1)
    keep = (dest != src) & ~same_site
    n_same_site = int((same_site & (dest != src)).sum())
    ed = pd.DataFrame(
        {
            "query_node_idx": dest[keep],
            "src_query_node_idx": src[keep],
            "log1p_geo_dist_km": np.log1p(geo_km[keep]),
            "log1p_relief_dist_km": np.log1p(relief_km[keep]),
            "rel_elev_m": z[dest[keep]] - z[src[keep]],
            "same_basin": (
                (basin[dest[keep]] == basin[src[keep]]) & (basin[dest[keep]] >= 0)
            ).astype("float64"),
            "src_cv_fold": cv_fold[src[keep]],
        }
    )
    # kd-tree output is dest-contiguous ascending-distance -> cumcount = rank; cut to k
    ed["rank"] = ed.groupby("query_node_idx").cumcount().astype("int64")
    ed = ed[ed["rank"] < args.k].reset_index(drop=True)

    assert (ed["query_node_idx"] != ed["src_query_node_idx"]).all(), "self edge"
    assert eligible[ed["src_query_node_idx"].to_numpy()].all(), "ineligible source"
    assert np.isfinite(ed[EDGE_COLS].to_numpy()).all(), "non-finite edge feature"
    deg = ed.groupby("query_node_idx").size()
    stats = {
        "bundle": str(gdir),
        "k": int(args.k),
        "relief_vw": vw,
        "n_query_nodes": int(len(qn)),
        "n_eligible_sources": int(len(src_idx)),
        "n_edges": int(len(ed)),
        "n_dest_with_edges": int(deg.size),
        "n_same_site_dropped": n_same_site,
        "mean_degree": float(deg.mean()),
        "median_geo_dist_km_rank0": float(
            np.expm1(ed.loc[ed["rank"] == 0, "log1p_geo_dist_km"]).median()
        ),
        "frac_same_basin": float(ed["same_basin"].mean()),
        "edge_cols": EDGE_COLS,
    }
    ed.to_parquet(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    log.info("wrote %s (%d edges) + %s", out, len(ed), out.with_suffix(".json"))
    log.info("%s", json.dumps(stats))


if __name__ == "__main__":
    main()
