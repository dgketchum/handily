"""Window a basin-wide regional FAC product down to a single HUC8 valley.

The Rio Grande mainstem FAC (``build_regional_fac.py --huc6 ...``) is built ONCE
over the whole connected basin so flow accumulation and Strahler order are
coherent top-to-bottom. The NAIP open-water support build and the REM solve,
however, are local per-pixel operations with NO basin-wide benefit, and at the
full-basin 5m grid they OOM (``rem_fac.sample_dem_to_grid`` materializes the
entire grid in RAM). This tool clips the mainstem rasters + streams to one HUC8
(plus a buffer) so those steps run at a tractable per-valley size.

Output dir mirrors a regional dir, carrying exactly what the downstream steps
read:
- ``dem_10m.tif``            (NAIP grid + REM DEM)
- ``flow_accumulation.tif``  (REM)
- ``streams_regional.fgb``   (NAIP corridor + REM channel network)
- ``basin_boundary.fgb``     (the HUC8 polygon; NAIP coverage stats / REM clip)

Rasters are read with a windowed read (never the whole 8.6GB mainstem DEM into
RAM). The requested bounds are snapped to the DEM pixel grid so every clipped
raster stays mutually pixel-aligned.

Usage:
    uv run python utils/window_regional_to_huc8.py \
        --mainstem-dir /data/ssd2/handily/nm/regional/rio_grande_mainstem \
        --huc8 13030102 \
        --out-dir /data/ssd2/handily/nm/regional/mesilla \
        --buffer-m 5000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds

log = logging.getLogger("window_regional_to_huc8")

WBD8_DEFAULT = (
    "/mnt/mco_nas1/dgketchum/boundaries/wbd/"
    "NHD_H_New_Mexico_State_Shape/Shape/WBDHU8.shp"
)


def _snap_bounds_to_grid(bounds, transform):
    """Snap (minx, miny, maxx, maxy) outward to the raster pixel grid."""
    win = from_bounds(*bounds, transform=transform)
    col0 = int(win.col_off // 1)
    row0 = int(win.row_off // 1)
    col1 = int(-(-(win.col_off + win.width) // 1))  # ceil
    row1 = int(-(-(win.row_off + win.height) // 1))
    snapped = rasterio.windows.bounds(
        Window(col0, row0, col1 - col0, row1 - row0), transform
    )
    return snapped


def _clip_raster(src_path: Path, out_path: Path, bounds) -> dict:
    with rasterio.open(src_path) as src:
        full = Window(0, 0, src.width, src.height)
        win = (
            from_bounds(*bounds, transform=src.transform)
            .round_offsets()
            .round_lengths()
        )
        win = win.intersection(full)
        if win.width <= 0 or win.height <= 0:
            raise ValueError(f"{src_path.name}: window does not intersect raster")
        data = src.read(window=win)
        profile = src.profile.copy()
        profile.update(
            height=int(win.height),
            width=int(win.width),
            transform=src.window_transform(win),
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
    return {"width": int(win.width), "height": int(win.height)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mainstem-dir", required=True)
    ap.add_argument("--huc8", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wbd8", default=WBD8_DEFAULT)
    ap.add_argument("--buffer-m", type=float, default=5000.0)
    ap.add_argument(
        "--rasters",
        nargs="+",
        default=["dem_10m.tif", "flow_accumulation.tif"],
        help="Raster filenames under <mainstem-dir> to clip",
    )
    args = ap.parse_args()

    src_dir = Path(args.mainstem_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # HUC8 polygon, projected to the DEM CRS, buffered, snapped to the grid.
    dem_path = src_dir / "dem_10m.tif"
    with rasterio.open(dem_path) as dem:
        dem_crs, dem_transform = dem.crs, dem.transform

    huc = gpd.read_file(args.wbd8)
    hcol = "huc8" if "huc8" in huc.columns else "HUC8"
    ncol = "name" if "name" in huc.columns else "NAME"
    poly = huc[huc[hcol] == args.huc8]
    if poly.empty:
        raise SystemExit(f"HUC8 {args.huc8} not found in {args.wbd8}")
    name = str(poly.iloc[0][ncol])
    poly_dem = poly.to_crs(dem_crs)

    buffered = poly_dem.buffer(args.buffer_m).total_bounds
    bounds = _snap_bounds_to_grid(buffered, dem_transform)
    log.info(
        "HUC8 %s (%s): clip bounds (snapped) %s",
        args.huc8,
        name,
        tuple(round(v) for v in bounds),
    )

    clipped = {}
    for fname in args.rasters:
        src_path = src_dir / fname
        if not src_path.exists():
            log.warning("%s not found in %s -> skip", fname, src_dir)
            continue
        info = _clip_raster(src_path, out_dir / fname, bounds)
        clipped[fname] = info
        log.info("clipped %s -> %dx%d", fname, info["width"], info["height"])

    # Streams: keep whole reaches intersecting the buffered bbox.
    streams_src = src_dir / "streams_regional.fgb"
    streams = gpd.read_file(streams_src, bbox=tuple(bounds))
    streams = streams.to_crs(dem_crs)
    streams.to_file(out_dir / "streams_regional.fgb", driver="FlatGeobuf")
    log.info("clipped streams -> %d reaches", len(streams))

    # HUC8 polygon as the (unbuffered) basin boundary, in the DEM CRS.
    poly_dem[[hcol, ncol, "geometry"]].to_file(
        out_dir / "basin_boundary.fgb", driver="FlatGeobuf"
    )

    summary = {
        "huc8": args.huc8,
        "name": name,
        "mainstem_dir": str(src_dir),
        "out_dir": str(out_dir),
        "buffer_m": args.buffer_m,
        "clip_bounds_5070": [round(v) for v in bounds],
        "rasters": clipped,
        "n_stream_reaches": len(streams),
    }
    with open(out_dir / "window_build.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", out_dir / "window_build.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
