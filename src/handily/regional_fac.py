"""Regional flow accumulation from 10m DEM.

Downloads USGS 3DEP 1/3 arc-second tiles, merges into a basin-wide DEM,
runs WhiteboxTools D8 flow accumulation, extracts a topologically connected
stream network, and clips per-AOI.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
import whitebox
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from tqdm import tqdm

log = logging.getLogger(__name__)

# USGS 3DEP 1/3 arc-second (~10m) staged products on S3.
_TILE_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/"
    "TIFF/current/{name}/USGS_13_{name}.tif"
)


# ---------------------------------------------------------------------------
# Tile download
# ---------------------------------------------------------------------------


def _tiles_for_bbox(bbox_wgs84: tuple[float, float, float, float]) -> list[str]:
    """Return 1-degree tile names (e.g. 'n41w117') covering a WGS84 bbox.

    bbox_wgs84 is (west, south, east, north) in degrees.
    """
    west, south, east, north = bbox_wgs84
    lat_min = int(math.floor(south))
    lat_max = int(math.ceil(north))
    # Longitudes are negative in the western hemisphere; tile naming uses
    # positive values with a 'w' prefix.
    lon_min = int(math.floor(abs(east)))
    lon_max = int(math.ceil(abs(west)))
    names = []
    for lat in range(lat_min + 1, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            names.append(f"n{lat:02d}w{lon:03d}")
    return sorted(names)


def download_3dep_10m_tiles(
    bbox_wgs84: tuple[float, float, float, float],
    out_dir: str | Path,
) -> list[Path]:
    """Download USGS 1/3 arc-second tiles covering *bbox_wgs84*.

    Skips tiles already on disk. Returns list of local paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = _tiles_for_bbox(bbox_wgs84)
    log.info("Need %d tiles for bbox %s", len(names), bbox_wgs84)

    paths: list[Path] = []
    for name in names:
        url = _TILE_URL.format(name=name)
        local = out_dir / f"USGS_13_{name}.tif"
        if local.exists():
            log.info("  %s exists, skipping", local.name)
            paths.append(local)
            continue
        log.info("  downloading %s ...", name)
        tmp = local.with_suffix(".tif.part")
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with (
                open(tmp, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc=name) as pbar,
            ):
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        tmp.rename(local)
        paths.append(local)
    return paths


# ---------------------------------------------------------------------------
# DEM merge
# ---------------------------------------------------------------------------


def build_regional_dem(
    tile_paths: list[Path],
    basin_gdf: gpd.GeoDataFrame,
    out_path: str | Path,
    target_crs_epsg: int = 5070,
    buffer_m: float = 5000.0,
) -> Path:
    """Merge tiles into a single DEM clipped to *basin_gdf* + buffer.

    Uses GDAL CLI (gdalbuildvrt + gdalwarp) to avoid loading all tiles
    into Python memory.
    """
    out_path = Path(out_path)
    if out_path.exists():
        log.info("DEM already exists: %s", out_path)
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write basin boundary as a cutline shapefile for gdalwarp.
    basin_proj = basin_gdf.to_crs(epsg=target_crs_epsg)
    basin_buffered = basin_proj.copy()
    basin_buffered["geometry"] = basin_proj.geometry.buffer(buffer_m)
    cutline_path = out_path.parent / "basin_cutline.fgb"
    basin_buffered.to_file(cutline_path, driver="FlatGeobuf")

    # Save unbuffered boundary for reference.
    basin_proj.to_file(out_path.parent / "basin_boundary.fgb", driver="FlatGeobuf")

    # Build VRT from tiles.
    vrt_path = out_path.with_suffix(".vrt")
    cmd_vrt = [
        "gdalbuildvrt",
        str(vrt_path),
        *[str(p) for p in tile_paths],
    ]
    log.info("Building VRT from %d tiles ...", len(tile_paths))
    subprocess.run(cmd_vrt, check=True, capture_output=True)

    # Warp to target CRS, clip to buffered basin, 10m resolution.
    cmd_warp = [
        "gdalwarp",
        "-t_srs",
        f"EPSG:{target_crs_epsg}",
        "-tr",
        "10",
        "10",
        "-r",
        "bilinear",
        "-cutline",
        str(cutline_path),
        "-crop_to_cutline",
        "-co",
        "COMPRESS=LZW",
        "-co",
        "TILED=YES",
        "-co",
        "BIGTIFF=YES",
        "-multi",
        "-overwrite",
        str(vrt_path),
        str(out_path),
    ]
    log.info("Warping to EPSG:%d at 10m ...", target_crs_epsg)
    subprocess.run(cmd_warp, check=True, capture_output=True)
    log.info("Wrote %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# WhiteboxTools FAC pipeline
# ---------------------------------------------------------------------------


def compute_regional_fac(
    dem_path: str | Path,
    out_dir: str | Path,
    threshold: int = 5000,
    max_procs: int = 32,
) -> Path:
    """Run D8 flow accumulation and stream extraction on a regional DEM.

    Returns the path to the final ``streams_regional.fgb``.
    """
    dem_path = str(dem_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(True)
    wbt.set_max_procs(max_procs)
    wbt.set_compress_rasters(True)

    filled = str(out_dir / "dem_10m_filled.tif")
    fdir = str(out_dir / "d8_pointer.tif")
    acc = str(out_dir / "flow_accumulation.tif")
    streams_ras = str(out_dir / "streams_10m.tif")
    strahler = str(out_dir / "stream_order.tif")
    streams_shp = str(out_dir / "streams_raw.shp")
    streams_fgb = str(out_dir / "streams_regional.fgb")

    t0 = time.time()

    # 1. Fill depressions
    if os.path.exists(filled):
        log.info("Skipping fill — %s exists", filled)
    else:
        log.info("Filling depressions (Wang & Liu) ...")
        wbt.fill_depressions_wang_and_liu(dem_path, filled, fix_flats=True)
        log.info("  done in %.0fs", time.time() - t0)

    # 2. D8 flow pointer
    if os.path.exists(fdir):
        log.info("Skipping D8 pointer — %s exists", fdir)
    else:
        log.info("Computing D8 flow pointer ...")
        wbt.d8_pointer(filled, fdir)
        log.info("  done in %.0fs", time.time() - t0)

    # 3. D8 flow accumulation
    if os.path.exists(acc):
        log.info("Skipping accumulation — %s exists", acc)
    else:
        log.info("Computing D8 flow accumulation ...")
        wbt.d8_flow_accumulation(fdir, acc, out_type="cells", pntr=True, log=False)
        log.info("  done in %.0fs", time.time() - t0)

    # 4. Extract streams
    log.info("Extracting streams (threshold=%d cells) ...", threshold)
    wbt.extract_streams(acc, streams_ras, threshold=threshold)
    log.info("  done in %.0fs", time.time() - t0)

    # 5. Strahler stream order
    log.info("Computing Strahler stream order ...")
    wbt.strahler_stream_order(fdir, streams_ras, strahler)
    log.info("  done in %.0fs", time.time() - t0)

    # 6. Vectorize streams
    log.info("Vectorizing streams (following D8 pointer) ...")
    wbt.raster_streams_to_vector(streams_ras, fdir, streams_shp)
    log.info("  done in %.0fs", time.time() - t0)

    # 7. Post-process: merge, snap, order, write FGB
    log.info("Post-processing vector streams ...")
    raw_gdf = gpd.read_file(streams_shp)
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
    if raw_gdf.crs is None:
        raw_gdf = raw_gdf.set_crs(dem_crs)
    else:
        raw_gdf = raw_gdf.to_crs(dem_crs)

    valid = raw_gdf[
        raw_gdf.geometry.notna()
        & raw_gdf.geometry.is_valid
        & ~raw_gdf.geometry.is_empty
    ]
    geoms = list(valid.geometry)
    log.info("  raw segments: %d, valid: %d", len(raw_gdf), len(geoms))

    merged = linemerge(MultiLineString(geoms))
    if merged.geom_type == "LineString":
        final_geoms = [merged]
    elif merged.geom_type == "MultiLineString":
        final_geoms = list(merged.geoms)
    else:
        final_geoms = [merged]
    log.info("  after merge: %d linestrings", len(final_geoms))

    final_geoms = snap_dangling_ends(final_geoms, snap_tol=20.0)
    orders = sample_stream_order(final_geoms, strahler)

    gdf = gpd.GeoDataFrame(
        {"stream_id": range(len(final_geoms)), "strahler": orders},
        geometry=final_geoms,
        crs=dem_crs,
    )
    gdf["length_m"] = gdf.geometry.length
    gdf.to_file(streams_fgb, driver="FlatGeobuf")
    log.info(
        "  wrote %d streams (%.0f km) to %s in %.0fs",
        len(gdf),
        gdf.length_m.sum() / 1000,
        streams_fgb,
        time.time() - t0,
    )
    return Path(streams_fgb)


# ---------------------------------------------------------------------------
# Per-AOI clip
# ---------------------------------------------------------------------------


def clip_streams_to_aoi(
    regional_streams: gpd.GeoDataFrame | str | Path,
    aoi_geom,
    out_path: str | Path,
    buffer_m: float = 500.0,
) -> gpd.GeoDataFrame:
    """Clip regional streams to an AOI geometry and write ``streams_fac.fgb``."""
    if isinstance(regional_streams, (str, Path)):
        regional_streams = gpd.read_file(regional_streams)

    clipped = gpd.clip(regional_streams, aoi_geom.buffer(buffer_m))
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
    clipped["stream_id"] = np.arange(len(clipped), dtype=np.int64)
    clipped = clipped.reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_file(out_path, driver="FlatGeobuf")
    log.info("Wrote %d streams to %s", len(clipped), out_path)
    return clipped


# ---------------------------------------------------------------------------
# Shared helpers (moved from utils/flow_accumulation_network.py)
# ---------------------------------------------------------------------------


def snap_dangling_ends(
    geoms: list[LineString], snap_tol: float = 2.0
) -> list[LineString]:
    """Snap dangling line endpoints to the nearest other line within tolerance."""
    from shapely import STRtree
    from shapely.geometry import Point

    if len(geoms) <= 1:
        return geoms

    tree = STRtree(geoms)
    snapped = list(geoms)

    for i, g in enumerate(geoms):
        coords = list(g.coords)
        for end_idx in (0, -1):
            pt = Point(coords[end_idx])
            candidates = tree.query(pt.buffer(snap_tol))
            touching = any(j != i and geoms[j].distance(pt) < 0.01 for j in candidates)
            if not touching:
                min_dist = float("inf")
                nearest_pt = None
                for j in candidates:
                    if j == i:
                        continue
                    d = geoms[j].distance(pt)
                    if d < min_dist and d <= snap_tol:
                        min_dist = d
                        nearest_pt = geoms[j].interpolate(geoms[j].project(pt))
                if nearest_pt is not None:
                    new_coords = list(coords)
                    idx = 0 if end_idx == 0 else len(new_coords) - 1
                    new_coords[idx] = (nearest_pt.x, nearest_pt.y)
                    new_line = LineString(new_coords)
                    if new_line.is_valid and new_line.length > 0:
                        snapped[i] = new_line
                        coords = new_coords
    return snapped


def sample_stream_order(
    geoms: list[LineString], strahler_path: str | Path
) -> list[int]:
    """Sample max Strahler order along each linestring from the order raster."""
    with rasterio.open(str(strahler_path)) as src:
        strahler_arr = src.read(1)
        transform = src.transform
    inv = ~transform

    orders = []
    for g in geoms:
        coords = list(g.coords)
        samples = [coords[0], coords[len(coords) // 2], coords[-1]]
        max_order = 0
        for x, y in samples:
            col, row = inv * (x, y)
            r, c = int(round(row)), int(round(col))
            if 0 <= r < strahler_arr.shape[0] and 0 <= c < strahler_arr.shape[1]:
                val = int(strahler_arr[r, c])
                if val > max_order:
                    max_order = val
        orders.append(max_order)
    return orders
