"""Augment an EXISTING graph bundle with the spatial-context pieces.

The spatial-context nodes/edges depend only on the query coordinates, the query
land-surface elevation (already carried in query_nodes.parquet) and the covariate
rasters -- none of the bundle's cross-fit priors or graph topology. So a bundle can
be upgraded in place (two new parquets + one manifest key) instead of re-running the
full build_conus_graph_inputs.py chain, whose exact original argv the bundle does not
record. The additive files are invisible to trainer runs that do not pass
--spatial-context, so existing baseline arms are untouched.

Usage:
    uv run python utils/augment_spatial_context.py \
        --graph-dir /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_dd_dup
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conus_graph_inputs import (  # noqa: E402
    DADS_HTD_BANK,
    DEM,
    build_spatial_context,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("augment_spatial_context")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument(
        "--dem",
        default=DEM,
        help="land-surface DEM; MUST be the DEM the bundle's z_surf came from so "
        "sc_rel_elev_m is datum-consistent",
    )
    ap.add_argument(
        "--dads-covariate-bank",
        default=DADS_HTD_BANK,
        help="dads HTD 1 km stack for the SC payload v2 cols; 'none' -> v1 payload "
        "(6A.1 bank only). See notes/SC_COVARIATE_UPGRADE.md.",
    )
    args = ap.parse_args()
    gdir = Path(args.graph_dir)
    man_path = gdir / "graph_manifest.json"
    man = json.loads(man_path.read_text())
    qn = pd.read_parquet(gdir / "query_nodes.parquet")
    surf_col = man["surface_elev_col"]
    if surf_col not in qn.columns:
        raise SystemExit(
            f"query_nodes.parquet lacks {surf_col} -- cannot compute sc_rel_elev_m "
            "on the bundle's shared land-surface datum"
        )
    block = build_spatial_context(
        qn[["x5070", "y5070"]].to_numpy("float64"),
        qn["query_node_idx"].to_numpy("int64"),
        qn[surf_col].to_numpy("float64"),
        args.dem,
        gdir,
        dads_bank=(
            None
            if str(args.dads_covariate_bank).lower() == "none"
            else args.dads_covariate_bank
        ),
    )
    block["augmented_from_existing_bundle"] = True
    man["spatial_context"] = block
    man_path.write_text(json.dumps(man, indent=2))
    log.info(
        "patched %s: spatial_context block (%d cells, %d edges)",
        man_path,
        block["node_count"],
        block["edge_count"],
    )


if __name__ == "__main__":
    main()
