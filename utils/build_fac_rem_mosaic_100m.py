"""Build the 100 m CONUS FAC-REM depth mosaic that the parity model reads.

The parity graph bundle samples ``fac_rem_dtw_m`` from 992 per-basin 10 m
``fac_head_depth_rem_10m.tif`` rasters listed in its ``fac_rem_manifest.json``
with the rule "list order; first finite sample wins at overlaps". No 100 m
mosaic of that set exists on disk (``wte_dtw_100m_5070.tif`` is the WTE-IDW
regional-prior DTW, not FAC-REM), so this builds one on the common 100 m
EPSG:5070 grid with the same precedence: entries are written in reverse list
order so earlier entries overwrite later ones wherever both are finite.

Downsampling is block-average over the finite 10 m cells (halo included, as
the sampler sees it). Output is Float32, nodata -9999, tiled LZW, with a JSON
sidecar recording the manifest and counts. Used to stage the EE asset
``projects/ee-dgketchum/assets/handily/covariates/fac_rem_dtw_100m_5070``.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, transform as win_transform

from build_conus_fac_rem import (
    MOSAIC_CRS,
    MOSAIC_HEIGHT,
    MOSAIC_NODATA,
    MOSAIC_TRANSFORM,
    MOSAIC_WIDTH,
    _ensure_mosaic,
    _grid_window,
)

MANIFEST = (
    "/data/ssd2/handily/conus/wte_gnn/"
    "graph_conus_monitoring_water_v2_ndwr_ufold_ose_parity/fac_rem_manifest.json"
)
OUT = "/data/ssd2/handily/conus/fac_rem/fac_rem_dtw_100m_5070_handily.tif"


def _downsample(
    entry: dict,
) -> tuple[int, tuple[int, int, int, int] | None, np.ndarray | None]:
    """Average a 10 m REM raster onto its 100 m window of the common grid."""
    win = _grid_window(entry["bounds"], MOSAIC_TRANSFORM, MOSAIC_WIDTH, MOSAIC_HEIGHT)
    if win is None:
        return entry["idx"], None, None
    h, w = int(win.height), int(win.width)
    out = np.full((h, w), MOSAIC_NODATA, dtype="float32")
    with rasterio.open(entry["path"]) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            # 10 m REMs carry NaN outside the watershed with no nodata tag; naming it
            # keeps GDAL's average from propagating NaN into partly covered 100 m cells.
            src_nodata=src.nodata if src.nodata is not None else np.nan,
            dst_transform=win_transform(win, MOSAIC_TRANSFORM),
            dst_crs=MOSAIC_CRS,
            dst_nodata=MOSAIC_NODATA,
            resampling=Resampling.average,
        )
    out[~np.isfinite(out)] = MOSAIC_NODATA
    return entry["idx"], (int(win.col_off), int(win.row_off), w, h), out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--out", default=OUT)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args(argv)

    man = json.load(open(a.manifest))
    entries = man["entries"]
    for i, e in enumerate(entries):
        e["idx"] = i
    # Reverse list order: the LAST write wins in the raster, so writing the
    # manifest's first entry last reproduces "first finite wins".
    order = list(reversed(entries))
    out = Path(a.out)
    if out.exists():
        out.unlink()
    _ensure_mosaic(out)
    print(f"mosaic {out} from {len(entries)} rasters, {a.workers} workers", flush=True)

    t0 = time.time()
    done: dict[int, tuple] = {}
    written = 0
    next_idx = 0  # position in `order` that must be written next
    with rasterio.open(out, "r+") as dst, ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_downsample, e): e["idx"] for e in order}
        pos_of = {e["idx"]: k for k, e in enumerate(order)}
        for f in as_completed(futs):
            idx, win, arr = f.result()
            done[pos_of[idx]] = (win, arr)
            # Writes must happen in `order`, so flush the contiguous prefix.
            while next_idx in done:
                win, arr = done.pop(next_idx)
                if win is not None:
                    c0, r0, w, h = win
                    window = Window(c0, r0, w, h)
                    block = dst.read(1, window=window)
                    new = arr != MOSAIC_NODATA
                    block[new] = arr[new]
                    dst.write(block, 1, window=window)
                    written += int(new.sum())
                next_idx += 1
                if next_idx % 50 == 0:
                    print(
                        f"  {next_idx}/{len(order)} written, {time.time() - t0:.0f}s",
                        flush=True,
                    )

    side = {
        "manifest": a.manifest,
        "n_rasters": len(entries),
        "precedence": man.get("precedence"),
        "recipes": man.get("n_by_recipe"),
        "cells_written": written,
        "resampling": "average over finite 10 m cells, halo included",
        "nodata": MOSAIC_NODATA,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.with_suffix(".json").write_text(json.dumps(side, indent=2))
    print(f"done: {written} finite cells in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
