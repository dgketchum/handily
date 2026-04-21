"""Batch-clip regional FAC streams to per-AOI streams_fac.fgb.

Usage:
    uv run python utils/clip_regional_to_aois.py \
        --regional-streams /data/ssd2/handily/nv/regional/humboldt/streams_regional.fgb \
        --aoi-dir /data/ssd2/handily/nv \
        --buffer-m 500 \
        --overwrite
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union

from handily.regional_fac import clip_streams_to_aoi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Clip regional streams to AOIs")
    parser.add_argument("--regional-streams", required=True, help="Regional FGB")
    parser.add_argument("--aoi-dir", required=True, help="Root with aoi_XXXX dirs")
    parser.add_argument("--buffer-m", type=float, default=500.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    regional = gpd.read_file(args.regional_streams)
    log.info("Loaded %d regional streams", len(regional))

    aoi_root = Path(args.aoi_dir)
    aoi_dirs = sorted(aoi_root.glob("aoi_*"))
    log.info("Found %d AOI directories", len(aoi_dirs))

    clipped = 0
    skipped = 0
    for aoi_dir in aoi_dirs:
        out_path = aoi_dir / "streams_fac.fgb"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        # Get AOI geometry from DEM bounds or existing boundary file.
        dem_path = aoi_dir / "dem_bounds_1m.tif"
        bounds_path = aoi_dir / "fields_bounds.fgb"
        if bounds_path.exists():
            aoi_gdf = gpd.read_file(bounds_path)
            aoi_geom = unary_union(aoi_gdf.to_crs(regional.crs).geometry)
        elif dem_path.exists():
            import rasterio

            with rasterio.open(dem_path) as src:
                from shapely.geometry import box

                aoi_geom = box(*src.bounds)
                if src.crs != regional.crs:
                    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geom], crs=src.crs).to_crs(
                        regional.crs
                    )
                    aoi_geom = aoi_gdf.geometry.iloc[0]
        else:
            log.warning("  %s: no DEM or boundary, skipping", aoi_dir.name)
            skipped += 1
            continue

        result = clip_streams_to_aoi(regional, aoi_geom, out_path, args.buffer_m)
        if len(result) > 0:
            clipped += 1
            log.info("  %s: %d streams", aoi_dir.name, len(result))
        else:
            log.info("  %s: no streams in extent", aoi_dir.name)
            skipped += 1

    log.info("Done: %d clipped, %d skipped", clipped, skipped)


if __name__ == "__main__":
    main()
