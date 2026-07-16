"""Build FAC-REM-footprint fair CSVs for the shallow scatter intercomparison.

FAC-REM (the raw near-network shallow prior behind handily) is nodata away from
the drainage network by design -- it covers only ~45% of the NM fair-panel
wells. Any comparison against Ma / Janssen / handily therefore has to happen on
a FAC-conditional common footprint: sample the FAC-REM 10 m mosaic at every
fair-panel well, keep only the wells where it is finite, and re-emit ALL
products' fair CSVs restricted to that identical subset (tagged ``*_facfp_*``,
non-clobbering). The FAC-REM CSV mirrors the fair-CSV schema (``pred_dtw_m`` /
``pred_ft`` / ``resid_ft`` recomputed; well-level columns inherited), so the
existing plotter runs unchanged on the tagged CSVs:

    uv run python utils/plot_shallow_scatter_intercomp.py \
        --products "Ma=ma_facfp,Janssen V2=janssen_v2_facfp,FAC-REM=fac_rem_facfp" \
        --panel non_nwis --sources nmbgmr_amp --out-tag facfp

Never compare ``*_facfp_*`` aggregates against full-footprint runs -- the
near-network footprint is structurally shallower than the full panel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwx_wells import sample_raster  # noqa: E402

FT_PER_M = 1.0 / 0.3048
FAC_REM_VRT = (
    "/data/ssd2/handily/huc8/mosaics/"
    "gnn_conus_monitoring_water_gate_mirror_sigma_w25_prod/fac_rem_dtw_nm.vrt"
)
# Fair CSVs share one well set; well-level columns are product-independent, so
# the FAC-REM CSV inherits them from any one of the inputs.
FAIR_TAGS = ("ma_fair", "janssen_v2_fair", "handily_fair")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir", default="/data/ssd2/handily/nm/regional/nm_scatter_intercomp"
    )
    ap.add_argument("--fac-vrt", default=FAC_REM_VRT)
    args = ap.parse_args()
    in_dir = Path(args.in_dir)

    dfs = {t: pd.read_csv(in_dir / f"{t}_vs_nm_gwx_wells.csv") for t in FAIR_TAGS}
    ns = {t: len(d) for t, d in dfs.items()}
    if len(set(ns.values())) != 1:
        raise SystemExit(f"fair CSV well counts differ -- not a common footprint: {ns}")

    base = dfs["handily_fair"]
    fac = base.copy()
    fac["pred_dtw_m"] = sample_raster(
        args.fac_vrt, base["longitude"].to_numpy(), base["latitude"].to_numpy()
    )
    fac["pred_ft"] = fac["pred_dtw_m"] * FT_PER_M
    fac["resid_ft"] = fac["pred_ft"] - fac["obs_ft"]

    finite = np.isfinite(fac["pred_dtw_m"].to_numpy())
    print(f"FAC-REM finite at {finite.sum():,}/{len(fac):,} fair-panel wells")
    for src, grp in fac.groupby("source"):
        cov = np.isfinite(grp["pred_dtw_m"].to_numpy()).mean()
        print(f"  {src}: {cov * 100:.1f}% of {len(grp):,}")

    out = dict(dfs, fac_rem=fac)
    for tag, df in out.items():
        out_tag = tag.replace("_fair", "") + "_facfp"
        path = in_dir / f"{out_tag}_vs_nm_gwx_wells.csv"
        df.loc[finite].to_csv(path, index=False)
        print(f"wrote {path} (n={finite.sum():,})")


if __name__ == "__main__":
    main()
