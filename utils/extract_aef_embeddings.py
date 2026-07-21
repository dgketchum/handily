"""AlphaEarth Foundations (AEF) satellite-embedding features at graph-bundle nodes.

Samples the Google/DeepMind Satellite Embedding dataset
(``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL``; 64-dim, 10 m, annual, unit-normalized,
target-blind) at every query node of the production bundle, emitting an ADDITIVE parquet
next to the bundle (``aef_embeddings_allq.parquet``) that mirrors the MAE embedding parquet
convention (utils/extract_mae_embeddings.py): 100% query-node coverage, keyed by
``query_node_idx``, columns ``mae_aef_00 … mae_aef_63`` so the width-agnostic
``train_conus_gnn.py --mae-embeddings`` loader and ``probe_mae_embeddings.py`` pick them up
with zero trainer edits.

Acquisition route (see notes/MAE_NEIGHBORHOOD_EMBEDDING.md): a direct GCS COG route exists
(``gs://alphaearth_foundations``) but point-sampling 69,535 continental nodes from the COGs
is ~0.4-2 TB of requester-pays egress (measured: 1,063 tiles / 23,888 512-px blocks) plus a
nonlinear-dequant reimplementation risk. EE serves de-quantized floats, mosaics UTM zones,
and reduces per cell server-side for a tens-of-MB table -- the right tool for scattered
point sampling.

Sampling semantics:
- SPATIAL: mean over the node's exact EPSG:5070 100 m lattice cell (snap_to_lattice, the
  same lattice the MAE patches use), reduceRegions(mean, scale=10) -> arithmetic mean of the
  ~100 de-quantized 10 m AEF pixels in the cell (matches lattice semantics; less noisy than
  a single 10 m pixel).
- TEMPORAL: 2020-2024 five-year mean (wells carry long-term median DTW; a multi-year mean is
  a stable "typical surface" descriptor, not a single year's moisture state).

    uv run python utils/extract_aef_embeddings.py \
        --bundle /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water \
        --out-dir /data/ssd2/handily/conus/mae/aef \
        --mae-parquet /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water/mae_embeddings_pyr_wide_allq.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mae_patches import LATTICE_ORIGIN, RES_M, snap_to_lattice  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract_aef_embeddings")

AEF_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
AEF_BANDS = [f"A{j:02d}" for j in range(64)]
AEF_DIM = 64


def aef_columns() -> list[str]:
    """Additive-parquet column names -- ``mae_`` prefix so the width-agnostic
    --mae-embeddings loader / probe consume them without any edit."""
    return [f"mae_aef_{j:02d}" for j in range(AEF_DIM)]


def cell_bounds_5070(x5070: np.ndarray, y5070: np.ndarray) -> np.ndarray:
    """Exact EPSG:5070 100 m lattice-cell rectangle [xlo, ylo, xhi, yhi] per node.

    Uses the same snap_to_lattice as the MAE patch builder, so an AEF cell and an MAE
    patch center refer to the identical lattice cell."""
    col, row = snap_to_lattice(x5070, y5070)
    x0, y0 = LATTICE_ORIGIN
    xlo = x0 + col * RES_M
    xhi = xlo + RES_M
    ytop = y0 - row * RES_M
    ylo = ytop - RES_M
    return np.stack([xlo, ylo, xhi, ytop], axis=1)


def chunk_slices(n: int, size: int) -> list[slice]:
    """Contiguous [0,n) slices of at most ``size`` rows."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def spatial_order(x5070: np.ndarray, y5070: np.ndarray, bin_m: float = 100_000.0):
    """Row-major order over a coarse (bin_m) grid so sequential chunks are geographically
    local -- keeps each EE reduceRegions request's filterBounds footprint small."""
    bx = np.floor(x5070 / bin_m).astype("int64")
    by = np.floor(y5070 / bin_m).astype("int64")
    key = by * 100_000 + bx
    return np.lexsort((x5070, key))


def _reduce_chunk(ee, image, rects, qids, retries: int):
    """reduceRegions(mean) over one chunk of 5070 cell rectangles; returns {qid: 64-vec or
    None}. filterBounds is applied by the caller (image already bounded)."""
    feats = [
        ee.Feature(
            ee.Geometry.Rectangle(
                [float(r[0]), float(r[1]), float(r[2]), float(r[3])], "EPSG:5070", False
            ),
            {"qid": int(q)},
        )
        for r, q in zip(rects, qids)
    ]
    fc = ee.FeatureCollection(feats)
    reduced = image.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4
    )
    last = None
    for attempt in range(retries):
        try:
            info = reduced.getInfo()
            break
        except Exception as e:  # noqa: BLE001 -- transient EE/network; retry with backoff
            last = e
            wait = 5 * (attempt + 1)
            log.warning(
                "getInfo failed (attempt %d/%d): %s; sleep %ds",
                attempt + 1,
                retries,
                str(e)[:160],
                wait,
            )
            time.sleep(wait)
    else:
        raise RuntimeError(
            f"reduceRegions getInfo failed after {retries} attempts: {last}"
        )

    out: dict[int, np.ndarray | None] = {}
    for f in info["features"]:
        p = f["properties"]
        qid = int(p["qid"])
        vals = [p.get(b) for b in AEF_BANDS]
        if any(v is None for v in vals):
            out[qid] = None
        else:
            out[qid] = np.asarray(vals, "float64")
    return out


def extract(
    bundle: Path,
    out_dir: Path,
    years: tuple[int, int],
    chunk: int,
    ee_project: str,
    retries: int,
    limit: int | None,
) -> pd.DataFrame:
    import ee

    ee.Initialize(project=ee_project)
    col = ee.ImageCollection(AEF_COLLECTION).filterDate(
        f"{years[0]}-01-01", f"{years[1] + 1}-01-01"
    )
    log.info(
        "AEF collection %s filtered %d-%d: %d images",
        AEF_COLLECTION,
        years[0],
        years[1],
        col.size().getInfo(),
    )

    qn = pd.read_parquet(
        bundle / "query_nodes.parquet",
        columns=["query_node_idx", "canonical_id", "x5070", "y5070", "is_water_pseudo"],
    ).reset_index(drop=True)
    if limit:
        qn = qn.iloc[:limit].reset_index(drop=True)
    log.info(
        "query nodes: %d (water_pseudo=%d)",
        len(qn),
        int(qn["is_water_pseudo"].astype(bool).sum()),
    )

    x = qn["x5070"].to_numpy("float64")
    y = qn["y5070"].to_numpy("float64")
    rects = cell_bounds_5070(x, y)
    qids = qn["query_node_idx"].to_numpy("int64")
    to4326 = pyproj.Transformer.from_crs(5070, 4326, always_xy=True)
    lon, lat = to4326.transform(x, y)

    # Primary-pass cache: the 28-chunk server-side reduce is the ~35 min cost. Cache its
    # result (NaN for masked cells) so a rerun -- e.g. to widen the null-cell fallback --
    # skips it and only re-does the handful of nulls. Keyed on year range + node identity.
    cache_p = out_dir / f"aef_primary_cache_{years[0]}_{years[1]}.npz"
    result: dict[int, np.ndarray | None] = {}
    if cache_p.exists():
        z = np.load(cache_p)
        if np.array_equal(z["qids"], qids):
            mat = z["mat"]
            for j, q in enumerate(qids):
                v = mat[j]
                result[int(q)] = (
                    None if not np.isfinite(v).all() else v.astype("float64")
                )
            log.info(
                "loaded primary-pass cache %s (%d nodes, %d null)",
                cache_p,
                len(qids),
                sum(1 for q in qids if result.get(q) is None),
            )
        else:
            log.warning("cache %s node-mismatch -- recomputing primary pass", cache_p)

    if not result:
        order = spatial_order(x, y)
        slices = chunk_slices(len(order), chunk)
        for i, sl in enumerate(slices):
            oidx = order[sl]
            blon, blat = lon[oidx], lat[oidx]
            bbox = ee.Geometry.Rectangle(
                [
                    float(blon.min()) - 0.02,
                    float(blat.min()) - 0.02,
                    float(blon.max()) + 0.02,
                    float(blat.max()) + 0.02,
                ],
                "EPSG:4326",
                False,
            )
            image = col.filterBounds(bbox).mean()
            got = _reduce_chunk(ee, image, rects[oidx], qids[oidx], retries)
            result.update(got)
            n_null = sum(1 for q in qids[oidx] if result.get(q) is None)
            log.info(
                "chunk %d/%d (%d nodes) done; nulls so far in chunk %d",
                i + 1,
                len(slices),
                len(oidx),
                n_null,
            )
        mat = np.stack(
            [
                result[int(q)]
                if result.get(q) is not None
                else np.full(AEF_DIM, np.nan)
                for q in qids
            ],
            axis=0,
        ).astype("float32")
        np.savez(cache_p, qids=qids, mat=mat)
        log.info("saved primary-pass cache -> %s", cache_p)

    null_qids = [int(q) for q in qids if result.get(q) is None]
    log.info(
        "primary pass: %d/%d nodes valid, %d null (masked cells)",
        len(qids) - len(null_qids),
        len(qids),
        len(null_qids),
    )

    # Fallback for null cells (fully-masked 100 m cell): expand the sampled region around
    # the node until a valid AEF neighborhood is found. Principled spatial fill (nearest
    # valid neighborhood), NOT a silent zero -- radius used is logged per node.
    fallback_used = {}
    if null_qids:
        qid_to_pos = {int(q): p for p, q in enumerate(qids)}
        for radius_m in (300.0, 1000.0, 3000.0, 10000.0, 30000.0):
            remaining = [q for q in null_qids if result.get(q) is None]
            if not remaining:
                break
            log.info(
                "fallback pass radius=%.0f m for %d null nodes",
                radius_m,
                len(remaining),
            )
            pos = np.array([qid_to_pos[q] for q in remaining])
            for sl in chunk_slices(len(pos), chunk):
                spos = pos[sl]
                blon, blat = lon[spos], lat[spos]
                # square cell centered on node, side = 2*radius, built in 5070
                cx = (rects[spos, 0] + rects[spos, 2]) / 2.0
                cy = (rects[spos, 1] + rects[spos, 3]) / 2.0
                frects = np.stack(
                    [cx - radius_m, cy - radius_m, cx + radius_m, cy + radius_m], axis=1
                )
                bbox = ee.Geometry.Rectangle(
                    [
                        float(blon.min()) - 0.05,
                        float(blat.min()) - 0.05,
                        float(blon.max()) + 0.05,
                        float(blat.max()) + 0.05,
                    ],
                    "EPSG:4326",
                    False,
                )
                image = col.filterBounds(bbox).mean()
                got = _reduce_chunk(ee, image, frects, qids[spos], retries)
                for q, v in got.items():
                    if v is not None and result.get(q) is None:
                        result[q] = v
                        fallback_used[int(q)] = radius_m
        still_null = [q for q in null_qids if result.get(q) is None]
        if still_null:
            raise SystemExit(
                f"{len(still_null)} nodes have NO valid AEF neighborhood within 30 km "
                f"(qids e.g. {still_null[:10]}); investigate before use -- do NOT impute."
            )
        log.info(
            "fallback resolved all nulls: %s",
            {
                r: sum(1 for v in fallback_used.values() if v == r)
                for r in sorted(set(fallback_used.values()))
            },
        )

    mat = np.stack([result[int(q)] for q in qids], axis=0).astype("float32")
    cols = aef_columns()
    df = pd.concat(
        [
            qn[["query_node_idx", "canonical_id"]].reset_index(drop=True),
            pd.DataFrame(mat, columns=cols),
        ],
        axis=1,
    )
    meta = {
        "collection": AEF_COLLECTION,
        "bands": AEF_BANDS,
        "dim": AEF_DIM,
        "years": list(years),
        "spatial": "mean over exact EPSG:5070 100 m lattice cell (snap_to_lattice), scale=10 m",
        "temporal": f"{years[0]}-{years[1]} arithmetic mean of de-quantized annual embeddings",
        "n_nodes": int(len(df)),
        "n_fallback": int(len(fallback_used)),
        "fallback_radius_m_counts": {
            str(r): sum(1 for v in fallback_used.values() if v == r)
            for r in sorted(set(fallback_used.values()))
        },
        "route": "EE server-side reduceRegions table sampling (GCS COG route rejected: "
        "~0.4-2 TB requester-pays egress for 69,535 scattered nodes)",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aef_extract_meta.json").write_text(json.dumps(meta, indent=2))
    return df


def write_concat(aef_df: pd.DataFrame, mae_parquet: Path, out: Path) -> None:
    """Concatenated AEF (64) + MAE (128) per-query embeddings for the 'with' GNN/probe arms.
    Both keyed on query_node_idx; requires identical coverage."""
    mae = pd.read_parquet(mae_parquet)
    mae_cols = [c for c in mae.columns if c.startswith("mae_")]
    if not mae_cols:
        raise SystemExit(f"{mae_parquet} has no mae_* columns")
    merged = aef_df.merge(
        mae[["query_node_idx"] + mae_cols],
        on="query_node_idx",
        how="left",
        validate="one_to_one",
    )
    if merged[mae_cols].isna().any().any():
        n = int(merged[mae_cols].isna().any(axis=1).sum())
        raise SystemExit(
            f"concat: {n} query nodes missing MAE embedding -- AEF and MAE parquets must "
            "share query-node coverage (both --all-query-nodes)."
        )
    merged.to_parquet(out)
    log.info(
        "wrote concat %d x (%d AEF + %d MAE) -> %s",
        len(merged),
        AEF_DIM,
        len(mae_cols),
        out,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bundle", required=True, help="graph bundle dir (query_nodes.parquet)"
    )
    ap.add_argument("--out-dir", required=True, help="artifacts dir (meta/logs)")
    ap.add_argument(
        "--years", default="2020-2024", help="inclusive year range, e.g. 2020-2024"
    )
    ap.add_argument(
        "--chunk", type=int, default=2500, help="nodes per reduceRegions request"
    )
    ap.add_argument("--ee-project", default="ee-dgketchum")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument(
        "--limit", type=int, default=None, help="first N nodes (smoke test)"
    )
    ap.add_argument(
        "--mae-parquet",
        default=None,
        help="if given, also write aef_mae_embeddings_allq.parquet = AEF (64) + MAE cols",
    )
    args = ap.parse_args()

    y0, y1 = (int(v) for v in args.years.split("-"))
    bundle = Path(args.bundle)
    out_dir = Path(args.out_dir)

    df = extract(
        bundle, out_dir, (y0, y1), args.chunk, args.ee_project, args.retries, args.limit
    )

    aef_out = bundle / "aef_embeddings_allq.parquet"
    df.to_parquet(aef_out)
    df.to_parquet(out_dir / "aef_embeddings_allq.parquet")
    log.info("wrote %d x %d AEF embeddings -> %s", len(df), AEF_DIM, aef_out)

    if args.mae_parquet:
        write_concat(
            df, Path(args.mae_parquet), bundle / "aef_mae_embeddings_allq.parquet"
        )
        write_concat(
            df, Path(args.mae_parquet), out_dir / "aef_mae_embeddings_allq.parquet"
        )


if __name__ == "__main__":
    main()
