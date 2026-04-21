"""Build a regional FAC stream network from USGS 3DEP 10m.

Usage:
    uv run python utils/build_regional_fac.py \
        --wbd-path /nas/boundaries/wbd/NHD_H_Nevada_State_Shape/Shape/WBDHU6.shp \
        --huc6 160401 \
        --out-dir /data/ssd2/handily/nv/regional/humboldt \
        --threshold 5000 \
        --max-procs 32
"""

import argparse
import logging

import geopandas as gpd

from handily.regional_fac import (
    build_regional_dem,
    compute_regional_fac,
    download_3dep_10m_tiles,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Build regional FAC stream network")
    parser.add_argument("--wbd-path", required=True, help="Path to WBDHU6 shapefile")
    parser.add_argument("--huc6", required=True, help="HUC6 code (e.g. 160401)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--threshold", type=int, default=5000, help="FAC cell threshold"
    )
    parser.add_argument("--max-procs", type=int, default=32, help="WhiteboxTools cores")
    parser.add_argument(
        "--buffer-m", type=float, default=5000.0, help="Basin buffer (m)"
    )
    args = parser.parse_args()

    # Load basin boundary
    huc6_gdf = gpd.read_file(args.wbd_path)
    basin = huc6_gdf[huc6_gdf["huc6"] == args.huc6]
    if basin.empty:
        raise ValueError(f"HUC6 {args.huc6} not found in {args.wbd_path}")
    print(f"Basin: {basin['name'].iloc[0]} ({args.huc6})")

    # Bbox in WGS84 for tile selection
    basin_4326 = basin.to_crs(epsg=4326)
    bbox = tuple(basin_4326.total_bounds)
    print(f"Bbox (WGS84): {bbox}")

    # 1. Download tiles
    tiles_dir = f"{args.out_dir}/tiles"
    tile_paths = download_3dep_10m_tiles(bbox, tiles_dir)
    print(f"Downloaded {len(tile_paths)} tiles")

    # 2. Merge DEM
    dem_path = f"{args.out_dir}/dem_10m.tif"
    build_regional_dem(
        tile_paths,
        basin,
        dem_path,
        target_crs_epsg=5070,
        buffer_m=args.buffer_m,
    )

    # 3. FAC + stream extraction
    streams_path = compute_regional_fac(
        dem_path,
        args.out_dir,
        threshold=args.threshold,
        max_procs=args.max_procs,
    )
    print(f"Done: {streams_path}")


if __name__ == "__main__":
    main()
