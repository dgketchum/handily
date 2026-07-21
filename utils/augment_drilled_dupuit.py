"""Augment a v0.2 graph bundle with the drilled-depth + Dupuit-hang QUERY features.

The proven deep-tail lever (statewide monitoring+dd lineage, NWIS MAD 2.54 -> 2.28 m)
is a set of four target-blind QUERY features:

  drilled_depth_idw_m, drilled_depth_p90_m  -- kNN-IDW mean + p90 of neighbour well
      CONSTRUCTION depths (build_drilled_depth_points.py pool). Construction metadata,
      NOT an observed water level, so it is deployment-available everywhere and needs
      no fold cross-fit; the well's own record + co-located nest siblings are removed
      by a self-exclusion radius (leak-safe per docs/inference_leakage_prevention.md).
  dupuit_hang_dtw_m, log1p_dupuit_d_m       -- well-free boundary-conditioned hang
      surface: z_surf - kNN-IDW of the top-2-Strahler reach elevations, plus distance
      to that boundary set. Stream elevations only -> target-blind, no cross-fit.

These are BUILD-SIDE features: the CONUS trainer reads query features solely from
``query_nodes.parquet[man["query_feature_cols"]]`` (train_conus_gnn.py: fit_stats/
apply_stats). It has NO additive-query-feature flag, so the ONLY way to feed dd/dup
without editing the trainer is to have them present as query_nodes columns listed in
query_feature_cols. To do that WITHOUT mutating the shared base bundle (its
query_nodes/manifest are read byte-identically by the frozen baseline, the AEF+MAE
arm, the E1 write-back leader and the concurrent E3 analog-edge work), this writes a
DERIVED bundle: every heavy base file is symlinked, and only a fresh query_nodes.parquet
(base + 4 dd/dup columns) and graph_manifest.json (query_feature_cols extended + the
drilled_depth / dupuit_hang manifest blocks populated) are written new. This is the
same derivation the statewide ``graph_conus_monitoring_dd_dup`` bundle used; it reuses
the exact builder functions so the feature semantics + leak-safety are identical.

Usage:
    uv run python utils/augment_drilled_dupuit.py \
        --base-dir /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water \
        --out-dir  /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_dddup
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("augment_drilled_dupuit")

WTE_GNN = "/data/ssd2/handily/conus/wte_gnn"
DEFAULT_DD_POINTS = f"{WTE_GNN}/drilled_depth_points.parquet"

# Files the CONUS trainer reads from the bundle dir; everything except query_nodes +
# graph_manifest is shared read-only and symlinked into the derived bundle.
SYMLINK_FILES = (
    "reach_nodes.parquet",
    "channel_edges.parquet",
    "lateral_edges.parquet",
    "permanent_water_blocks.parquet",
    "reach_graph_manifest.json",
)


def build_augmented_manifest(
    base_man: dict,
    dd_block: dict | None,
    dup_block: dict | None,
) -> dict:
    """Pure: return a NEW manifest with dd/dup query features + blocks added.

    Appends the drilled-depth then Dupuit feature columns to query_feature_cols
    (matching the builder's assembly order) and populates the drilled_depth /
    dupuit_hang blocks. Never mutates ``base_man``. Idempotent w.r.t. already-present
    columns (a column is never added twice).
    """
    man = copy.deepcopy(base_man)
    add_cols: list[str] = []
    if dd_block is not None:
        man["drilled_depth"] = dd_block
        add_cols += list(dd_block["feature_cols"])
    if dup_block is not None:
        man["dupuit_hang"] = dup_block
        add_cols += list(dup_block["feature_cols"])
    qfc = list(man["query_feature_cols"])
    for c in add_cols:
        if c not in qfc:
            qfc.append(c)
    man["query_feature_cols"] = qfc
    prov = man.get("dd_dup_augment", {})
    prov.update(
        {
            "augmented_from_existing_bundle": True,
            "added_query_feature_cols": add_cols,
            "note": (
                "dd/dup query features attached to a DERIVED bundle; base bundle "
                "query_nodes/manifest untouched. dd = construction-depth kNN-IDW "
                "(target-blind, self-excluded); dup = top-2-Strahler stream-elevation "
                "hang surface (well-free). No water-level leakage; no fold cross-fit."
            ),
        }
    )
    man["dd_dup_augment"] = prov
    return man


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", required=True, help="source v0.2 bundle (unchanged)")
    ap.add_argument("--out-dir", required=True, help="derived bundle to write")
    ap.add_argument("--drilled-depth-points", default=DEFAULT_DD_POINTS)
    ap.add_argument(
        "--idw-k", type=int, default=32, help="dd kNN neighbours (builder default 32)"
    )
    ap.add_argument("--idw-power", type=float, default=2.0)
    ap.add_argument(
        "--self-exclude-m",
        type=float,
        default=100.0,
        help="dd self/nest exclusion radius (builder default 100 m); never lower it",
    )
    ap.add_argument("--dupuit-top-orders", type=int, default=2)
    ap.add_argument("--dupuit-idw-k", type=int, default=8)
    ap.add_argument("--dupuit-idw-power", type=float, default=2.0)
    ap.add_argument(
        "--geom",
        default=None,
        help="flowline geometry parquet (comid/cx/cy); defaults to the base "
        "manifest's sources.geom",
    )
    ap.add_argument("--no-drilled-depth", action="store_true")
    ap.add_argument("--no-dupuit-hang", action="store_true")
    args = ap.parse_args()

    # Heavy builder imports are lazy so the pure manifest logic stays importable
    # (and unit-testable) without pulling in rasterio/scipy at module load.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_conus_graph_inputs import (  # noqa: PLC0415
        DRILLED_DEPTH_FEATURE_COLS,
        DUPUIT_FEATURE_COLS,
        sample_drilled_depth,
    )
    from build_dupuit_wte import build_boundaries, hang_interp  # noqa: PLC0415

    base = Path(args.base_dir)
    out = Path(args.out_dir)
    if out.resolve() == base.resolve():
        raise SystemExit("--out-dir must differ from --base-dir (no in-place mutation)")
    man = json.loads((base / "graph_manifest.json").read_text())
    surf_col = man["surface_elev_col"]

    qn = (
        pd.read_parquet(base / "query_nodes.parquet")
        .sort_values("query_node_idx")
        .reset_index(drop=True)
    )
    if not (qn["query_node_idx"].to_numpy() == np.arange(len(qn))).all():
        raise SystemExit("query_node_idx is not a dense 0..n-1 range -- refusing")
    for c in (surf_col, "x5070", "y5070", "canonical_id"):
        if c not in qn.columns:
            raise SystemExit(f"query_nodes.parquet lacks required column {c!r}")
    xy = qn[["x5070", "y5070"]].to_numpy("float64")
    log.info("base bundle %s: %d query nodes", base.name, len(qn))

    dd_block = dup_block = None

    if not args.no_drilled_depth:
        dd = sample_drilled_depth(
            args.drilled_depth_points,
            xy,
            args.idw_k,
            args.idw_power,
            query_ids=qn["canonical_id"].to_numpy(),
            self_exclude_m=args.self_exclude_m,
        )
        for col, vals in dd.items():
            qn[col] = vals
            log.info(
                "  %s: %.3f finite frac, median %.1f m",
                col,
                float(np.isfinite(vals).mean()),
                float(np.nanmedian(vals)),
            )
        dd_block = {
            "points": args.drilled_depth_points,
            "pool_n": int(
                len(
                    pd.read_parquet(
                        args.drilled_depth_points, columns=["drilled_depth_m"]
                    )
                )
            ),
            "k": args.idw_k,
            "power": args.idw_power,
            "self_exclude_m": args.self_exclude_m,
            "feature_cols": DRILLED_DEPTH_FEATURE_COLS,
        }

    if not args.no_dupuit_hang:
        geom = args.geom or man["sources"]["geom"]
        bnd = build_boundaries(
            str(base / "reach_nodes.parquet"), geom, top_orders=args.dupuit_top_orders
        )
        hang_wte, d_bnd = hang_interp(
            bnd[["cx", "cy"]].to_numpy("float64"),
            bnd["reach_elev_m"].to_numpy("float64"),
            xy,
            k=args.dupuit_idw_k,
            power=args.dupuit_idw_power,
        )
        qn["dupuit_hang_dtw_m"] = qn[surf_col].to_numpy("float64") - hang_wte
        qn["log1p_dupuit_d_m"] = np.log1p(d_bnd)
        log.info(
            "  dupuit hang: %d boundary reaches; dupuit_hang_dtw_m median %.1f m, "
            "d_m median %.0f m (finite frac %.3f)",
            len(bnd),
            float(np.nanmedian(qn["dupuit_hang_dtw_m"])),
            float(np.median(d_bnd)),
            float(np.isfinite(qn["dupuit_hang_dtw_m"]).mean()),
        )
        dup_block = {
            "boundary_reaches": int(len(bnd)),
            "top_orders": args.dupuit_top_orders,
            "idw_k": args.dupuit_idw_k,
            "idw_power": args.dupuit_idw_power,
            "feature_cols": DUPUIT_FEATURE_COLS,
        }

    out.mkdir(parents=True, exist_ok=True)
    for name in SYMLINK_FILES:
        src = (base / name).resolve()
        dst = out / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if src.exists():
            dst.symlink_to(src)
        else:
            log.warning("base bundle has no %s -- skipping symlink", name)

    new_man = build_augmented_manifest(man, dd_block, dup_block)
    new_man.setdefault("sources", {})["derived_from"] = str(base)
    (out / "graph_manifest.json").write_text(json.dumps(new_man, indent=2))
    qn.to_parquet(out / "query_nodes.parquet", index=False)

    log.info(
        "wrote derived bundle -> %s (query_feature_cols %d -> %d; base untouched)",
        out,
        len(man["query_feature_cols"]),
        len(new_man["query_feature_cols"]),
    )


if __name__ == "__main__":
    main()
