import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["OMP_THREAD_LIMIT"] = "12"
import geopandas as gpd
from handily.core import (
    ensure_dir,
    aoi_from_bounds,
    tiles_for_bounds,
    ndwi_files_for_tiles,
    open_ndwi_mosaic,
    get_flowlines_within_aoi,
    get_dem_for_aoi_via_stac,
    build_streams_mask_from_nhd_ndwi,
    compute_rem_quick,
    load_and_clip_fields,
    compute_field_rem_stats,
)
from handily.viz import write_interactive_map


def dev_test_bounds_rem(fields_path, ndwi_dir, out_dir,
                        stac_dir,
                        flowlines_local_dir=None,
                        bounds=None,
                        ndwi_threshold=0.15,
                        mgrs_shp_path="~/data/IrrigationGIS/boundaries/mgrs/mgrs_aea.shp"):
    ensure_dir(out_dir)

    aoi = aoi_from_bounds(bounds)
    flowlines = get_flowlines_within_aoi(aoi, local_flowlines_dir=flowlines_local_dir)
    tiles, tiles_gdf = tiles_for_bounds(bounds, mgrs_shp_path)
    present_map, missing = ndwi_files_for_tiles(ndwi_dir, tiles)
    if missing:
        raise ValueError(f"Missing NDWI tiles: {missing}")
    ndwi_clip = open_ndwi_mosaic(present_map, bounds)
    dem_cache = os.path.join(out_dir, "dem_bounds_1m.tif")

    dem = get_dem_for_aoi_via_stac(
        aoi_gdf=aoi,
        stac_dir=os.path.expanduser(stac_dir),
        target_crs_epsg=5070,
        cache_path=dem_cache,
        overwrite=False,
        stac_download_cache_dir=os.path.join(out_dir, 'stac_cache'),
        stac_collection_id="usgs-3dep-1m-opr",
    )
    dem_crs = dem.rio.crs
    flowlines_dem = flowlines.to_crs(dem_crs)
    streams = build_streams_mask_from_nhd_ndwi(flowlines_dem, dem, ndwi_da=ndwi_clip, ndwi_threshold=float(ndwi_threshold))
    rem = compute_rem_quick(dem, streams)
    fields = load_and_clip_fields(fields_path, aoi, dem_crs)
    fields_stats = compute_field_rem_stats(fields, rem, stats=("mean",))
    results = {
        "aoi": aoi,
        "flowlines": flowlines,
        "ndwi": ndwi_clip,
        "streams": streams,
        "rem": rem,
        "mgrs_tiles": tiles_gdf,
        "fields": fields,
        "fields_stats": fields_stats,
        "summary": {
            "total_fields": len(fields_stats),
            "ndwi_threshold": float(ndwi_threshold),
        },
    }
    return results


if __name__ == '__main__':
    fields = os.path.expanduser(
        "~/data/IrrigationGIS/Montana/statewide_irrigation_dataset/statewide_irrigation_dataset_15FEB2024.shp")
    out_dir_ = "/home/dgketchum/data/IrrigationGIS/handily/outputs/"
    flowlines_local_dir_ = os.path.expanduser("~/data/IrrigationGIS/boundaries/wbd/NHD_H_Montana_State_Shape/Shape")
    ndwi_dir_ = os.path.expanduser("~/data/IrrigationGIS/surface_water/NDWI/")
    stac_dir_ = os.path.expanduser("~/data/IrrigationGIS/handily/stac/3dep_1m/")
    bounds_ = (-112.5, 45.4, -112.27, 45.6)

    results = dev_test_bounds_rem(
        fields_path=fields,
        ndwi_dir=ndwi_dir_,
        out_dir=out_dir_,
        stac_dir=stac_dir_,
        flowlines_local_dir=flowlines_local_dir_,
        bounds=bounds_,
        ndwi_threshold=0.15,
    )
    # out_html = os.path.join(out_dir_, "debug_map.html")
    # write_interactive_map(results, out_html, initial_threshold=2.0)
# ========================= EOF ====================================================================
