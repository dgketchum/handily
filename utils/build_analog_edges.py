"""Build embedding-similarity analog edges for the CONUS water-table GNN (E3).

Each real monitoring well (dest) is connected to its ``k`` nearest NONLOCAL analog
wells (source) in the 192-dim AEF+MAE embedding space, so the GNN can import a distant,
embedding-similar well's observed water table into anchor-sparse basins (the E3
hypothesis, notes/E3_ANALOG_EDGES.md). Emitted as an ADDITIVE ``analog_edges.parquet``
next to the bundle -- never mutates existing bundle files.

Similarity metric: the AEF (64) and MAE (128) blocks live on different scales, so a raw
cosine over the concat is dominated by whichever block has the larger norm. We
per-dimension z-score across ALL query nodes, then per-node L2-normalise; cosine
similarity is then the dot product of the normalised vectors (each standardized
dimension contributes equally).

Nonlocality (a RECEPTIVE-FIELD constraint, not the leakage guard): a candidate source
must be in a DIFFERENT HUC4 and >= ``--min-dist-km`` away (default 50 km, > the 40 km
CV block edge), so the analog carries information the query's spatial neighbours and its
regional IDW prior do not already contain. Fold safety is enforced downstream in the
trainer by a per-fold source mask (``src_cv_fold`` is carried here for that), NOT by this
topology -- cv_fold is not nested in HUC4 or the 40 km block, so no spatial cutoff can
guarantee a different fold.

Edge features (target-blind, standardized once globally in the trainer):
  cos_dist    = 1 - cosine similarity            (dimensionless, [0,2])
  geo_dist_km = planar EPSG:5070 distance / 1000 (km)
  rel_elev_m  = z_surf(dest) - z_surf(src)       (m; DEM land-surface difference)

Usage:
    uv run python utils/build_analog_edges.py \\
        --bundle /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water \\
        --embeddings .../aef_mae_embeddings_allq.parquet \\
        --k 8 --min-dist-km 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("build_analog_edges")


def analog_fold_keep(src_fold: np.ndarray, fold: int | None) -> np.ndarray:
    """Boolean edge mask: keep analog edges whose SOURCE well is NOT in the held-out
    fold. ``fold=None`` keeps every edge (memory-probe sizing only).

    This is the E3 leakage guard (notes/E3_ANALOG_EDGES.md, docs/inference_leakage_
    prevention.md): during fold f's forward pass -- both the train-loss forward and the
    OOF-prediction forward -- every edge sourced from a fold-f well is dropped, so a
    test-fold query never reads a test-fold well's label. The same leave-fold-out
    discipline the regional prior R and the anchor-BC anomaly already use.
    """
    src_fold = np.asarray(src_fold)
    if fold is None:
        return np.ones(len(src_fold), dtype=bool)
    return src_fold != int(fold)


def zscore_l2(emb: np.ndarray) -> np.ndarray:
    """Per-dimension z-score (over all rows) then per-row L2-normalise -> unit vectors
    whose dot product is a balanced cosine similarity."""
    mu = emb.mean(axis=0, keepdims=True)
    sd = emb.std(axis=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    z = (emb - mu) / sd
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return (z / norm).astype("float32")


def knn_analog_edges(
    unit: np.ndarray,
    xy: np.ndarray,
    huc4: np.ndarray,
    k: int,
    min_dist_km: float,
    chunk: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k nonlocal cosine analogs per row (dest and source pools are the SAME real
    wells). Returns (dest_local_idx, src_local_idx, cos_sim), edge-major.

    Nonlocality mask: candidate excluded if same HUC4 OR planar distance < min_dist_km.
    Self is same-HUC4 (and distance 0) so it is always excluded.
    """
    n = unit.shape[0]
    min_d2 = (min_dist_km * 1000.0) ** 2
    sq = (xy**2).sum(axis=1)  # |s|^2 per node (m^2)
    dest_list: list[np.ndarray] = []
    src_list: list[np.ndarray] = []
    sim_list: list[np.ndarray] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        c = stop - start
        sims = unit[start:stop] @ unit.T  # (c, n) cosine similarity
        # squared planar distance (c, n): |d|^2 + |s|^2 - 2 d.s
        d2 = sq[start:stop, None] + sq[None, :] - 2.0 * (xy[start:stop] @ xy.T)
        near = d2 < min_d2
        same_huc4 = huc4[start:stop, None] == huc4[None, :]
        block = near | same_huc4
        sims = np.where(block, -np.inf, sims)
        kk = min(k, n - 1)
        # top-kk per row by similarity
        part = np.argpartition(-sims, kth=kk - 1, axis=1)[:, :kk]
        rows = np.arange(c)[:, None]
        part_sims = sims[rows, part]
        order = np.argsort(-part_sims, axis=1)
        nbr = part[rows, order]
        nbr_sims = part_sims[rows, order]
        for i in range(c):
            valid = np.isfinite(nbr_sims[i])
            m = int(valid.sum())
            if m == 0:
                continue
            dest_list.append(np.full(m, start + i, dtype="int64"))
            src_list.append(nbr[i, valid].astype("int64"))
            sim_list.append(nbr_sims[i, valid].astype("float64"))
        if (start // chunk) % 5 == 0:
            log.info("  analog kNN: %d / %d dest wells", stop, n)
    return (
        np.concatenate(dest_list),
        np.concatenate(src_list),
        np.concatenate(sim_list),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, help="graph bundle dir")
    ap.add_argument(
        "--embeddings",
        default=None,
        help="AEF+MAE embeddings parquet (default: <bundle>/aef_mae_embeddings_allq.parquet)",
    )
    ap.add_argument("--k", type=int, default=8, help="analog neighbours per dest well")
    ap.add_argument("--min-dist-km", type=float, default=50.0)
    ap.add_argument(
        "--out",
        default=None,
        help="output parquet (default: <bundle>/analog_edges.parquet)",
    )
    args = ap.parse_args()

    bundle = Path(args.bundle)
    emb_path = (
        Path(args.embeddings)
        if args.embeddings
        else (bundle / "aef_mae_embeddings_allq.parquet")
    )
    out_path = Path(args.out) if args.out else (bundle / "analog_edges.parquet")

    qn = (
        pd.read_parquet(
            bundle / "query_nodes.parquet",
            columns=[
                "query_node_idx",
                "canonical_id",
                "huc4",
                "x5070",
                "y5070",
                "z_surf_well_m",
                "is_water_pseudo",
                "cv_fold",
            ],
        )
        .sort_values("query_node_idx")
        .reset_index(drop=True)
    )
    assert (qn["query_node_idx"].to_numpy() == np.arange(len(qn))).all()

    emb = pd.read_parquet(emb_path)
    mae_cols = [c for c in emb.columns if c.startswith("mae_")]
    if not mae_cols:
        raise SystemExit(f"{emb_path} has no mae_* columns")
    emb = emb.set_index("query_node_idx").loc[qn["query_node_idx"].to_numpy()]
    emb_mat = emb[mae_cols].to_numpy("float64")
    if not np.isfinite(emb_mat).all():
        raise SystemExit(
            f"{int((~np.isfinite(emb_mat)).sum())} non-finite embedding values "
            "(investigate; do not impute)"
        )

    real = ~qn["is_water_pseudo"].to_numpy(bool)
    real_idx = np.where(real)[0]  # local -> global query_node_idx map
    log.info(
        "%d query nodes; %d real wells (analog source+dest pool); %d embed dims; k=%d "
        "min_dist=%.0f km",
        len(qn),
        real.sum(),
        len(mae_cols),
        args.k,
        args.min_dist_km,
    )

    # standardize + L2-normalise on the REAL-well pool only (the analog space).
    unit = zscore_l2(emb_mat[real_idx])
    xy = qn.loc[real_idx, ["x5070", "y5070"]].to_numpy("float64")
    z_surf = qn.loc[real_idx, "z_surf_well_m"].to_numpy("float64")
    huc4 = qn.loc[real_idx, "huc4"].to_numpy()
    cv_fold = qn.loc[real_idx, "cv_fold"].to_numpy("int64")

    d_loc, s_loc, cos_sim = knn_analog_edges(
        unit, xy, huc4, k=args.k, min_dist_km=args.min_dist_km
    )
    # map local (real-pool) indices back to global query_node_idx
    dest = real_idx[d_loc]
    src = real_idx[s_loc]
    geo_dist_km = np.sqrt(((xy[d_loc] - xy[s_loc]) ** 2).sum(axis=1)) / 1000.0
    rel_elev_m = z_surf[d_loc] - z_surf[s_loc]

    ed = pd.DataFrame(
        {
            "query_node_idx": dest.astype("int64"),
            "src_query_node_idx": src.astype("int64"),
            "cos_dist": (1.0 - cos_sim).astype("float64"),
            "geo_dist_km": geo_dist_km.astype("float64"),
            "rel_elev_m": rel_elev_m.astype("float64"),
            "src_cv_fold": cv_fold[s_loc].astype("int64"),
            "cos_sim": cos_sim.astype("float64"),
        }
    )
    # edges are emitted dest-contiguous in descending-similarity order -> rank = cumcount
    ed["rank"] = ed.groupby("query_node_idx").cumcount().astype("int64")
    # invariants: no self-edge, nonlocality respected
    assert (ed["query_node_idx"] != ed["src_query_node_idx"]).all(), "self analog edge"
    assert (ed["geo_dist_km"] >= args.min_dist_km - 1e-6).all(), "near analog leaked"
    assert (
        huc4[np.searchsorted(real_idx, ed["query_node_idx"])]
        != huc4[np.searchsorted(real_idx, ed["src_query_node_idx"])]
    ).all(), "same-HUC4 analog leaked"

    deg = ed.groupby("query_node_idx").size()
    stats = {
        "embeddings": str(emb_path),
        "k": int(args.k),
        "min_dist_km": float(args.min_dist_km),
        "n_real_wells": int(real.sum()),
        "n_dest_with_edges": int(deg.size),
        "n_dest_no_edges": int(real.sum() - deg.size),
        "n_edges": int(len(ed)),
        "degree_mean": float(deg.mean()),
        "degree_min": int(deg.min()),
        "degree_max": int(deg.max()),
        "geo_dist_km_median": float(ed["geo_dist_km"].median()),
        "geo_dist_km_p90": float(ed["geo_dist_km"].quantile(0.90)),
        "cos_sim_median": float(ed["cos_sim"].median()),
        "edge_feature_cols": ["cos_dist", "geo_dist_km", "rel_elev_m"],
    }
    ed.to_parquet(out_path, index=False)
    (out_path.parent / "analog_edges_manifest.json").write_text(
        json.dumps(stats, indent=2)
    )
    log.info("wrote %s (%d edges)", out_path, len(ed))
    log.info("stats: %s", json.dumps(stats))


if __name__ == "__main__":
    main()
