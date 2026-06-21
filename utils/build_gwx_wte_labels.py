"""Adapter: GWX national well index -> WTE labels parquet for the hybrid builder.

Emits the well-label schema ``build_wte_covariate_table.py`` expects
(``canonical_id, source, well_class, confinement_class, dtw_label_m, tier,
weight, geometry``) from the GWX index, so the regional-prior / residual-fusion
pipeline can run on the large independent GWX well set instead of the thin
NM-specific label parquet.

All unconfined / unconfined_marginal wells in the reference-raster window are
kept, INCLUDING nwis/ngwmn -- those are legitimate water-table observations and
belong in the blocked-CV training pool (the rule is never tune to Ma, not never
use NWIS). The ``source`` column is preserved so the downstream scorer can split
the independent (non-NWIS) headline from the NWIS panel. ``canonical_id`` is a
stable integer key that flows through the covariate table into the OOF
predictions, letting the scorer rejoin ``source`` without threading it through.

    uv run python utils/build_gwx_wte_labels.py \\
        --ref-raster /data/ssd2/handily/nm/regional/rio_grande_albuquerque/rem/nm_rga_v5_arid_full/fac_head_depth_rem_10m.tif \\
        --out /data/ssd2/handily/nm/regional/rio_grande_albuquerque/evidence/gwx/gwx_wte_labels.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwx_wells import GWX_INDEX, WT_CLASSES, load_window_wells  # noqa: E402

log = logging.getLogger("build_gwx_wte_labels")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ref-raster", required=True, help="raster whose bounds set the window"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--gwx-index", default=GWX_INDEX)
    p.add_argument("--confinement", default=",".join(WT_CLASSES))
    args = p.parse_args()

    conf = tuple(c for c in args.confinement.split(",") if c)
    with rasterio.open(args.ref_raster) as src:
        b = src.bounds
    bbox = (b.left, b.bottom, b.right, b.top)

    # include_sources=None + empty exclude => keep ALL sources (NWIS included).
    wells = load_window_wells(args.gwx_index, bbox, conf, set(), None)
    wells = wells.reset_index(drop=True)
    wells["canonical_id"] = [f"gwx_{i}" for i in range(len(wells))]
    wells["dtw_label_m"] = wells["mean_dtw"].astype("float64")
    wells["tier"] = "primary"  # default; only used by the builder's training mode
    wells["weight"] = 1.0  # unweighted water-table wells (no Ma-style weighting)

    keep = [
        "canonical_id",
        "source",
        "well_class",
        "confinement_class",
        "dtw_label_m",
        "tier",
        "weight",
        "geometry",
    ]
    out = wells[keep].copy()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)

    n_nwis = int(out["source"].isin(["nwis", "ngwmn"]).sum())
    log.info(
        "wrote %d GWX labels (%d non-NWIS, %d NWIS) -> %s",
        len(out),
        len(out) - n_nwis,
        n_nwis,
        out_path,
    )


if __name__ == "__main__":
    main()
