"""Per-well (and optional per-lattice) MAE neighborhood embeddings.

Loads a train_neighborhood_mae.py checkpoint and emits a mean-pooled encoder embedding
for every real monitoring well in the production bundle, snapped to the canonical
lattice and windowed by the SAME primitives + norm_stats used to build the training
patches. Output is an ADDITIVE parquet next to the bundle
(``mae_embeddings_<arm>.parquet``), following utils/augment_spatial_context.py: it is
invisible to GNN arms that do not opt in, so baselines are untouched.

The normalized multi-scale well stack (all scales/channels) is extracted once and cached
under the embeddings dir, so encoding multiple arms that share a patch set is cheap.

    uv run python utils/extract_mae_embeddings.py \
        --ckpt /data/ssd2/handily/conus/mae/checkpoints/s100.pt \
        --bundle /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water \
        --out-dir /data/ssd2/handily/conus/mae/embeddings
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# module import (not from-import) so a --manifest rebind of the channel roster is seen
import build_mae_patches as bmp  # noqa: E402
from build_mae_patches import (  # noqa: E402
    SCALES,
    WIN,
    extract_stack,
    snap_to_lattice,
)
from train_neighborhood_mae import ARMS, MAE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract_mae_embeddings")


def out_basename(
    arm: str, tag: str | None, all_query_nodes: bool, coords_stem: str | None
) -> str:
    """Output parquet name. A tag keeps runs from different checkpoints of the SAME
    arm (e.g. v1 vs v2 pyr_wide) from clobbering each other in the bundle."""
    t = f"_{tag}" if tag else ""
    if coords_stem:
        return f"mae_embeddings_{arm}{t}_{coords_stem}.parquet"
    return f"mae_embeddings_{arm}{t}{'_allq' if all_query_nodes else ''}.parquet"


def build_well_stack(x5070, y5070, norm_stats: dict) -> np.ndarray:
    """Normalized multi-scale stack for points: [N, S, C, WIN, WIN] float16.

    Uses the SAME extract_stack path (pool-full -> gather -> normalize) as
    build_mae_patches, so wells and lattice cells are encoded identically to the
    training distribution."""
    col, row = snap_to_lattice(x5070, y5070)
    n, ns, nc = len(col), len(SCALES), len(bmp.CHANNEL_SPECS)
    stack = np.zeros((n, ns, nc, WIN, WIN), "float32")
    extract_stack(col, row, stack, mask_out=None, norm=norm_stats)
    return stack.astype("float16")


def encode_stack(ckpt_path: str, stack: np.ndarray, batch: int = 512) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MAE(**ck["config"]).to(device).eval()
    model.load_state_dict(ck["model"])
    sidx = ARMS[ck["arm"]]
    n = stack.shape[0]
    embs = []
    with torch.no_grad():
        for b0 in range(0, n, batch):
            p = np.asarray(stack[b0 : b0 + batch])  # [B,S,C,H,W]
            x = np.concatenate([p[:, s] for s in sidx], axis=1).astype("float32")
            e = model.encode(torch.from_numpy(x).to(device)).cpu().numpy()
            embs.append(e)
    return np.concatenate(embs, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--bundle", default=None, help="graph bundle dir (query_nodes.parquet)"
    )
    ap.add_argument(
        "--coords",
        default=None,
        help="coords parquet (x5070, y5070 [, query_node_idx, canonical_id]) to "
        "extract for INSTEAD of a bundle -- the target-blind path for cluster runs "
        "where no bundle exists. Output goes to --out-dir only, named by the coords "
        "file stem.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--manifest",
        default=None,
        help="channel-manifest JSON rebinding the roster/paths "
        "(build_mae_patches.apply_manifest); required where the default zoran "
        "raster paths are absent",
    )
    ap.add_argument(
        "--stack-cache",
        default=None,
        help="path prefix for the cached normalized well stack (.npy + .parquet); "
        "reused across arms sharing a patch set. Default: <out-dir>/well_stack",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="suffix for the output filename (mae_embeddings_<arm>_<tag>*.parquet) so "
        "checkpoints sharing an arm name (v1 vs v2 pyr_wide) never clobber each "
        "other's artifacts",
    )
    ap.add_argument(
        "--all-query-nodes",
        action="store_true",
        help="extract for EVERY query node (incl. water pseudo-rows), not just real "
        "wells. Needed to feed the GNN as a per-query head slot (the forward runs over "
        "all query nodes, so coverage must be 100%% -- no imputation). Writes "
        "mae_embeddings_<arm>_allq.parquet. Default (off) keeps the real-wells-only "
        "mae_embeddings_<arm>.parquet used by the probe.",
    )
    args = ap.parse_args()
    if bool(args.bundle) == bool(args.coords):
        raise SystemExit("exactly one of --bundle / --coords is required")
    if args.manifest:
        bmp.apply_manifest(json.loads(Path(args.manifest).read_text()))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arm = ck["arm"]
    norm_stats = json.loads(Path(ck["norm_stats"]).read_text())

    if args.coords:
        wells = pd.read_parquet(args.coords)
        for c in ("x5070", "y5070"):
            if c not in wells.columns:
                raise SystemExit(f"--coords {args.coords}: missing column {c}")
        wells = wells.reset_index(drop=True)
        if "query_node_idx" not in wells.columns:
            wells["query_node_idx"] = np.arange(len(wells), dtype="int64")
        if "canonical_id" not in wells.columns:
            wells["canonical_id"] = wells["query_node_idx"].astype(str)
        log.info("coords rows: %d (%s)", len(wells), args.coords)
    else:
        qn = pd.read_parquet(
            Path(args.bundle) / "query_nodes.parquet",
            columns=[
                "query_node_idx",
                "canonical_id",
                "x5070",
                "y5070",
                "is_water_pseudo",
            ],
        )
        if args.all_query_nodes:
            wells = qn.reset_index(drop=True)
            log.info("all query nodes (incl. water pseudo-rows): %d", len(wells))
        else:
            wells = qn[~qn["is_water_pseudo"].astype(bool)].reset_index(drop=True)
            log.info("real wells: %d", len(wells))

    cache = Path(args.stack_cache) if args.stack_cache else out_dir / "well_stack"
    stack_npy = cache.with_suffix(".f16.npy")
    keys_pq = cache.with_suffix(".keys.parquet")
    if stack_npy.exists() and keys_pq.exists():
        log.info("reusing cached well stack %s", stack_npy)
        stack = np.load(stack_npy, mmap_mode="r")
        keys = pd.read_parquet(keys_pq)
        if (
            len(keys) != len(wells)
            or not (
                keys["query_node_idx"].to_numpy() == wells["query_node_idx"].to_numpy()
            ).all()
        ):
            raise SystemExit("cached stack keys mismatch bundle wells -- delete cache")
    else:
        stack = build_well_stack(
            wells["x5070"].to_numpy("float64"),
            wells["y5070"].to_numpy("float64"),
            norm_stats,
        )
        np.save(stack_npy, stack)
        wells[["query_node_idx", "canonical_id", "x5070", "y5070"]].to_parquet(keys_pq)
        log.info("cached well stack -> %s", stack_npy)

    embs = encode_stack(args.ckpt, stack)
    dim = embs.shape[1]
    emb_df = pd.DataFrame(
        embs.astype("float32"),
        columns=[f"mae_{arm}_{j}" for j in range(dim)],
        index=wells.index,
    )
    df = pd.concat([wells[["query_node_idx", "canonical_id"]], emb_df], axis=1)
    if args.coords:
        fname = out_basename(arm, args.tag, False, Path(args.coords).stem)
        out = out_dir / fname
        df.to_parquet(out)
    else:
        fname = out_basename(arm, args.tag, args.all_query_nodes, None)
        out = Path(args.bundle) / fname
        df.to_parquet(out)
        # also a copy in the mae embeddings dir (probe reads the real-wells-only one)
        df.to_parquet(out_dir / fname)
    log.info("wrote %d x %d embeddings (arm=%s) -> %s", len(df), dim, arm, out)


if __name__ == "__main__":
    main()
