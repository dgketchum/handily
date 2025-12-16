import logging
import os

import rioxarray as rxr

from .io import ensure_dir
from .stac_3dep import mosaic_from_stac

LOGGER = logging.getLogger("handily.dem")


def get_dem_for_aoi_via_stac(
    aoi_gdf,
    stac_dir: str,
    target_crs_epsg: int = 5070,
    cache_path: str | None = None,
    overwrite: bool = False,
    stac_download_cache_dir: str | None = None,
    stac_collection_id: str = "usgs-3dep-1m-opr",
):
    """Build a DEM mosaic from a local 3DEP STAC for the AOI."""
    target_crs = f"EPSG:{int(target_crs_epsg)}"

    if cache_path and (not overwrite) and os.path.exists(cache_path):
        LOGGER.info("Loading cached DEM (STAC): %s", cache_path)
        dem_cached = rxr.open_rasterio(cache_path)
        if "band" in dem_cached.dims:
            dem_cached = dem_cached.squeeze("band", drop=True)
        return dem_cached

    if stac_download_cache_dir is None:
        if cache_path:
            base = os.path.dirname(os.path.abspath(cache_path))
            stac_download_cache_dir = os.path.join(base, "stac_cache")
        else:
            stac_download_cache_dir = os.path.join(os.getcwd(), "stac_cache")

    LOGGER.info("Mosaicking DEM from STAC at %s (tiles cached in %s)", stac_dir, stac_download_cache_dir)
    dem = mosaic_from_stac(
        stac_dir=stac_dir,
        aoi_gdf=aoi_gdf,
        cache_dir=stac_download_cache_dir,
        collection_id=stac_collection_id,
        target_crs_epsg=int(target_crs_epsg),
    )

    if str(dem.rio.crs) != target_crs:
        dem = dem.rio.reproject(target_crs)

    if cache_path:
        ensure_dir(os.path.dirname(os.path.abspath(cache_path)))
        LOGGER.info("Saving DEM mosaic to cache: %s", cache_path)
        dem.rio.to_raster(cache_path)
    return dem
