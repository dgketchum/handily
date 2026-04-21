"""
Build a topologically connected stream network from a DEM via flow accumulation.

Uses WhiteboxTools (Rust + rayon) for parallel processing across all available cores.

Steps:
  1. Breach depressions (least-cost paths — better than filling for connectivity)
  2. D8 flow pointer
  3. D8 flow accumulation (parallel)
  4. Extract streams at threshold (500,000 cells)
  5. Vectorize streams following flow direction (topologically connected by construction)
  6. Strahler stream order
  7. Merge degree-2 junctions, snap endpoints, write FlatGeoBuf

Usage:
    uv run python utils/flow_accumulation_network.py
"""

import logging
import os
import time

import geopandas as gpd
import rasterio
import whitebox
from shapely.geometry import MultiLineString
from shapely.ops import linemerge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DEM_PATH = "/data/ssd2/handily/nv/aoi_0773/dem_bounds_1m.tif"
OUT_DIR = "/data/ssd2/handily/nv/aoi_0773"
WORK_DIR = "/tmp/flow_acc_0773"
ACC_THRESHOLD = 50_000
N_PROCS = 32


def main():
    t0 = time.time()
    os.makedirs(WORK_DIR, exist_ok=True)

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(True)
    wbt.set_max_procs(N_PROCS)
    wbt.set_compress_rasters(True)

    # Working paths (WBT writes intermediate rasters)
    dem_conditioned = f"{WORK_DIR}/dem_breached.tif"
    fdir_path = f"{WORK_DIR}/d8_pointer.tif"
    acc_path = f"{WORK_DIR}/d8_accumulation.tif"
    streams_ras_path = f"{WORK_DIR}/streams.tif"
    streams_vec_path = f"{WORK_DIR}/streams.shp"
    strahler_path = f"{WORK_DIR}/strahler.tif"

    # ── 1. Fill depressions ─────────────────────────────────────────────
    if os.path.exists(dem_conditioned):
        log.info("Skipping fill — %s exists", dem_conditioned)
    else:
        log.info("Filling depressions (Wang & Liu)...")
        ret = wbt.fill_depressions_wang_and_liu(
            DEM_PATH,
            dem_conditioned,
            fix_flats=True,
        )
        log.info("Fill done in %.1fs (ret=%s)", time.time() - t0, ret)

    # ── 2. D8 flow pointer ───────────────────────────────────────────────
    if os.path.exists(fdir_path):
        log.info("Skipping D8 pointer — %s exists", fdir_path)
    else:
        log.info("Computing D8 flow pointer...")
        ret = wbt.d8_pointer(dem_conditioned, fdir_path)
        log.info("D8 pointer done in %.1fs (ret=%s)", time.time() - t0, ret)

    # ── 3. D8 flow accumulation ──────────────────────────────────────────
    if os.path.exists(acc_path):
        log.info("Skipping accumulation — %s exists", acc_path)
    else:
        log.info("Computing D8 flow accumulation...")
        ret = wbt.d8_flow_accumulation(
            fdir_path,
            acc_path,
            out_type="cells",
            pntr=True,
            log=False,
        )
        log.info("Flow accumulation done in %.1fs (ret=%s)", time.time() - t0, ret)

        acc_out = f"{OUT_DIR}/flow_accumulation.tif"
        log.info("Copying accumulation raster to %s", acc_out)
        _copy_raster(acc_path, acc_out)

    # ── 4. Extract streams ───────────────────────────────────────────────
    log.info("Extracting streams (threshold=%d cells)...", ACC_THRESHOLD)
    ret = wbt.extract_streams(acc_path, streams_ras_path, threshold=ACC_THRESHOLD)
    log.info("Stream extraction done in %.1fs (ret=%s)", time.time() - t0, ret)

    # Copy stream raster
    _copy_raster(streams_ras_path, f"{OUT_DIR}/streams_fac_mask.tif")

    # ── 5. Strahler stream order ─────────────────────────────────────────
    log.info("Computing Strahler stream order...")
    ret = wbt.strahler_stream_order(fdir_path, streams_ras_path, strahler_path)
    log.info("Stream order done in %.1fs (ret=%s)", time.time() - t0, ret)

    _copy_raster(strahler_path, f"{OUT_DIR}/stream_order.tif")

    # ── 6. Vectorize streams ─────────────────────────────────────────────
    # raster_streams_to_vector traces connected stream cells following the
    # D8 pointer — output segments are topologically connected at junctions.
    log.info("Vectorizing streams (following D8 pointer)...")
    ret = wbt.raster_streams_to_vector(streams_ras_path, fdir_path, streams_vec_path)
    log.info("Vectorization done in %.1fs (ret=%s)", time.time() - t0, ret)

    # ── 7. Read vectorized streams, merge, snap, write FGB ───────────────
    log.info("Reading vectorized streams...")
    raw_gdf = gpd.read_file(streams_vec_path)
    log.info("Raw segments: %d", len(raw_gdf))

    # Get CRS from DEM
    with rasterio.open(DEM_PATH) as src:
        dem_crs = src.crs

    # WBT may output in pixel coords or DEM CRS — set CRS from DEM
    if raw_gdf.crs is None:
        raw_gdf = raw_gdf.set_crs(dem_crs)
    else:
        raw_gdf = raw_gdf.to_crs(dem_crs)

    # Filter invalid/empty
    valid = raw_gdf[
        raw_gdf.geometry.notna()
        & raw_gdf.geometry.is_valid
        & ~raw_gdf.geometry.is_empty
    ]
    geoms = list(valid.geometry)
    log.info("Valid segments: %d", len(geoms))

    # Merge at degree-2 junctions
    log.info("Merging at degree-2 junctions...")
    merged = linemerge(MultiLineString(geoms))
    if merged.geom_type == "LineString":
        final_geoms = [merged]
    elif merged.geom_type == "MultiLineString":
        final_geoms = list(merged.geoms)
    else:
        final_geoms = [merged]
    log.info("After merge: %d linestrings", len(final_geoms))

    # Snap dangling endpoints (fixes sub-pixel gaps from rasterization)
    log.info("Snapping dangling endpoints...")
    final_geoms = _snap_dangling_ends(final_geoms, snap_tol=2.0)

    # Sample Strahler order per segment
    log.info("Sampling Strahler order per segment...")
    strahler_out = f"{OUT_DIR}/stream_order.tif"
    orders = _sample_stream_order(final_geoms, strahler_out)

    # Write FlatGeoBuf
    out_path = f"{OUT_DIR}/streams_fac.fgb"
    gdf = gpd.GeoDataFrame(
        {"stream_id": range(len(final_geoms)), "strahler": orders},
        geometry=final_geoms,
        crs=dem_crs,
    )
    gdf["length_m"] = gdf.geometry.length
    gdf.to_file(out_path, driver="FlatGeobuf")
    log.info(
        "Wrote %d lines (total %.1f km) to %s",
        len(gdf),
        gdf.length_m.sum() / 1000,
        out_path,
    )
    log.info("Total time: %.1fs", time.time() - t0)


def _copy_raster(src_path, dst_path):
    """Copy a raster file preserving geotransform and CRS from original DEM."""
    with rasterio.open(src_path) as src:
        data = src.read()
        profile = src.profile.copy()
    # If CRS is missing, grab from DEM
    if profile.get("crs") is None:
        with rasterio.open(DEM_PATH) as dem:
            profile["crs"] = dem.crs
            profile["transform"] = dem.transform
    profile["driver"] = "GTiff"
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)


def _snap_dangling_ends(geoms, snap_tol=2.0):
    """Thin wrapper — implementation moved to handily.regional_fac."""
    from handily.regional_fac import snap_dangling_ends

    return snap_dangling_ends(geoms, snap_tol=snap_tol)


def _sample_stream_order(geoms, strahler_path):
    """Thin wrapper — implementation moved to handily.regional_fac."""
    from handily.regional_fac import sample_stream_order

    return sample_stream_order(geoms, strahler_path)


if __name__ == "__main__":
    main()
