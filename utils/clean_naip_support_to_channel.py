"""Restrict a NAIP open-water support mask to the high-order channel corridor.

The multi-year NAIP persistent-water mask (``build_basin_naip_water_multiyear.py``)
registers any persistent open water in the river-corridor decode window, which in
arid New Mexico is dominated by off-channel flood-irrigation tailwater, canals, and
ponds rather than the perennial mainstem (measured: ~86% of NM 3-yr NAIP open-water
support is off-channel). Feeding that raw mask to the FAC head solve as
``support_path`` pins large irrigated off-channel areas shallow.

This post-step keeps only support cells within ``buffer_m`` of a Strahler
>= ``min_strahler`` reach — i.e. the actual mainstem/high-order channels — and drops
the off-channel irrigation contamination. Output is the channel-clean support raster
used as the REM ``support_path``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regional-dir", required=True)
    ap.add_argument(
        "--support",
        default="naip_ndwi_support_3yr_5m.tif",
        help="support raster name under <regional-dir>/evidence/naip",
    )
    ap.add_argument("--buffer-m", type=float, default=200.0)
    ap.add_argument("--min-strahler", type=int, default=5)
    ap.add_argument("--out", default="naip_ndwi_support_3yr_5m_channelclean.tif")
    args = ap.parse_args()

    rd = Path(args.regional_dir)
    naip_dir = rd / "evidence" / "naip"
    support_path = naip_dir / args.support
    out_path = naip_dir / args.out

    with rasterio.open(support_path) as src:
        support = src.read(1)
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)
        profile = src.profile

    streams = gpd.read_file(rd / "streams_regional.fgb").to_crs(crs)
    high = streams[streams["strahler"] >= args.min_strahler]
    if high.empty:
        raise ValueError(
            f"no streams with strahler >= {args.min_strahler} in {rd}; cannot define corridor"
        )
    corridor = high.buffer(args.buffer_m).union_all()
    channel_mask = rasterize(
        [(corridor, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    clean = np.where(channel_mask, support, 0).astype(np.uint8)
    profile.update(dtype="uint8", count=1, nodata=None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(clean, 1)

    raw_cells = int((support > 0).sum())
    clean_cells = int((clean > 0).sum())
    summary = {
        "support_in": str(support_path),
        "output": str(out_path),
        "min_strahler": args.min_strahler,
        "buffer_m": args.buffer_m,
        "raw_water_cells": raw_cells,
        "channel_water_cells": clean_cells,
        "retained_fraction": round(clean_cells / raw_cells, 4) if raw_cells else 0.0,
    }
    with open(out_path.with_name(out_path.stem + "_build.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
