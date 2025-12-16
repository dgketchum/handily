import os
import sys
import logging
from pathlib import Path

import yaml
import geopandas as gpd
import rioxarray as rxr

from handily.config import HandilyConfig
from handily.compute import compute_field_rem_stats
from handily.io import aoi_from_bounds, ensure_dir
from handily.pipeline import REMWorkflow
from handily.viz import write_interactive_map


def configure_logging() -> None:
    level_name = os.environ.get("HANDILY_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def dev_test_bounds_rem(config: HandilyConfig, bounds_wsen, ndwi_threshold: float):
    ensure_dir(config.out_dir)
    aoi = aoi_from_bounds(bounds_wsen)
    workflow = REMWorkflow(config=config, aoi=aoi)
    result = workflow.run(ndwi_threshold=float(ndwi_threshold), stats=("mean",), cache_flowlines=True)
    return result


def _subset_results_to_aoi(results: dict, aoi_gdf: gpd.GeoDataFrame) -> dict:
    out = dict(results)
    out["aoi"] = aoi_gdf

    for key in ("flowlines", "fields_stats"):
        gdf = out.get(key)
        if gdf is None or len(gdf) == 0:
            continue
        aoi_in_crs = aoi_gdf.to_crs(gdf.crs)
        try:
            out[key] = gpd.clip(gdf, aoi_in_crs)
        except Exception:
            out[key] = gpd.overlay(gdf, aoi_in_crs, how="intersection")

    for key in ("rem", "dem", "streams"):
        da = out.get(key)
        if da is None:
            continue
        shapes = [aoi_gdf.to_crs(da.rio.crs).geometry.unary_union.__geo_interface__]
        out[key] = da.rio.clip(shapes, all_touched=True)

    return out


def main(argv=None) -> int:
    configure_logging()
    argv = sys.argv[1:] if argv is None else argv
    config_path = Path(argv[0]) if argv else Path(__file__).with_name("beaverhead_config.yaml")

    aoi_select = None # [30]

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must parse to a mapping, got {type(cfg).__name__}")

    config = HandilyConfig.from_dict(cfg)
    bounds_wsen = tuple(cfg["bounds"])
    ndwi_threshold = float(cfg.get("ndwi_threshold", 0.15))
    overwrite_outputs = bool(cfg.get("overwrite_outputs", False))
    viz_only = bool(cfg.get("viz_only", False))
    bake_tiles = bool(cfg.get("bake_tiles", False))

    rem_path = os.path.join(config.out_dir, "rem_bounds.tif")
    streams_path = os.path.join(config.out_dir, "streams_bounds.tif")
    ndwi_path = os.path.join(config.out_dir, "ndwi_bounds.tif")
    fields_fgb = os.path.join(config.out_dir, "fields_bounds.fgb")
    dem_path = os.path.join(config.out_dir, "dem_bounds_1m.tif")

    if viz_only:
        aoi = aoi_from_bounds(bounds_wsen)
        flowlines = gpd.read_file(os.path.join(config.out_dir, "flowlines_bounds.fgb"))

        rem_da = rxr.open_rasterio(rem_path)
        if "band" in rem_da.dims:
            rem_da = rem_da.squeeze("band", drop=True)
        dem_da = rxr.open_rasterio(dem_path)
        if "band" in dem_da.dims:
            dem_da = dem_da.squeeze("band", drop=True)

        fields = gpd.read_file(fields_fgb)
        fields_stats = compute_field_rem_stats(fields, rem_da, stats=("mean",))

        if not bake_tiles:
            rem_da = None
            dem_da = None

        results = {
            "aoi": aoi,
            "flowlines": flowlines,
            "rem": rem_da,
            "dem": dem_da,
            "fields": fields,
            "fields_stats": fields_stats,
            "summary": {
                "total_fields": len(fields_stats),
                "ndwi_threshold": float(ndwi_threshold),
            },
        }
    else:
        results = dev_test_bounds_rem(config=config, bounds_wsen=bounds_wsen, ndwi_threshold=ndwi_threshold)

        if overwrite_outputs and os.path.exists(rem_path):
            os.remove(rem_path)
        if overwrite_outputs and os.path.exists(streams_path):
            os.remove(streams_path)
        if overwrite_outputs and os.path.exists(ndwi_path):
            os.remove(ndwi_path)
        logging.getLogger("handily.beaverhead").info(
            "Writing rasters (overwrite_outputs=%s): rem=%s streams=%s ndwi=%s",
            overwrite_outputs,
            rem_path,
            streams_path,
            ndwi_path,
        )
        results["rem"].rio.to_raster(rem_path, overwrite=overwrite_outputs)
        results["streams"].rio.to_raster(streams_path, overwrite=overwrite_outputs)
        if results.get("ndwi") is not None:
            results["ndwi"].rio.to_raster(ndwi_path, overwrite=overwrite_outputs)

        if overwrite_outputs and os.path.exists(fields_fgb):
            os.remove(fields_fgb)
        fields = results["fields"]
        if "FID" in fields.columns:
            fields = fields.rename(columns={"FID": "FID_"})
        fields.to_file(fields_fgb, driver="FlatGeobuf")

    out_html = os.path.join(config.out_dir, "debug_map.html")
    write_interactive_map(results, out_html, initial_threshold=2.0)

    print("--- Beaverhead Bounds REM Summary ---")
    print(f"Fields total: {results['summary'].get('total_fields')}")
    print(f"NDWI threshold: {results['summary'].get('ndwi_threshold')}")
    print(f"Map: {out_html}")
    print(f"REM GeoTIFF: {rem_path}")
    print(f"Streams mask GeoTIFF: {streams_path}")
    print(f"NDWI mosaic GeoTIFF: {ndwi_path}")
    print(f"Fields FGB: {fields_fgb}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "12")
    os.environ.setdefault("OMP_THREAD_LIMIT", "12")
    raise SystemExit(main())
