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
from rasterio.features import rasterize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
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
        # Per-process temp name so concurrent workers sharing this cache dir
        # cannot clobber each other's partial download of the same tile.
        tmp = local.with_suffix(f".tif.part.{os.getpid()}")
        with requests.get(url, stream=True, timeout=300) as r:
            if r.status_code == 404:
                # No 3DEP coverage for this 1-degree cell (border/Mexico/ocean);
                # normal for a basin whose bbox overhangs the US edge -> skip.
                log.warning("  %s: no 3DEP coverage (404) -> skip", name)
                continue
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
        # Another worker may have finished the same tile while we downloaded;
        # the atomic rename makes last-writer-wins safe (identical content).
        if local.exists():
            tmp.unlink(missing_ok=True)
        else:
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


def d8_stream_cell_components(
    streams_arr: np.ndarray,
    d8_arr: np.ndarray,
    stream_value=1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Connected components of the D8 stream-cell graph.

    Two stream cells are connected iff one is the other's D8 downstream neighbor
    (treated undirected), so every connected component is the contributing area of
    a single window outlet — its most-downstream cell either has ``d8 == 0`` or
    drains off the valid-data window. Returns ``(rows, cols, labels, sizes)``:
    ``labels[i]`` is the component index of stream cell ``(rows[i], cols[i])`` and
    ``sizes[k]`` is the number of stream cells in component ``k``.

    Stepping along :data:`_D8_OFFSETS` follows flow downstream (FAC increases), so
    the component partition is the authoritative D8 routing — independent of the
    coordinate node-matching that the vector reach graph relies on.
    """
    ny, nx = streams_arr.shape
    mask = streams_arr == stream_value
    rows, cols = np.where(mask)
    n = len(rows)
    if n == 0:
        empty = np.array([], dtype=np.int64)
        return rows, cols, empty, empty

    cell_idx = -np.ones((ny, nx), dtype=np.int64)
    cell_idx[rows, cols] = np.arange(n)

    v = d8_arr[rows, cols].astype(np.int64)
    dr = np.zeros(n, np.int64)
    dc = np.zeros(n, np.int64)
    has_dir = np.zeros(n, bool)
    for val, (off_r, off_c) in _D8_OFFSETS.items():
        m = v == val
        dr[m], dc[m], has_dir[m] = off_r, off_c, True
    tr, tc = rows + dr, cols + dc
    in_bounds = has_dir & (tr >= 0) & (tr < ny) & (tc >= 0) & (tc < nx)
    dst = -np.ones(n, np.int64)
    dst[in_bounds] = cell_idx[tr[in_bounds], tc[in_bounds]]
    edge = dst >= 0
    src_e = np.arange(n)[edge]
    dst_e = dst[edge]
    graph = coo_matrix((np.ones(len(src_e)), (src_e, dst_e)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return rows, cols, labels, sizes


def clip_streams_to_fac_watershed(
    streams_gdf: gpd.GeoDataFrame,
    d8_path: str | Path,
    streams_raster_path: str | Path,
    fac_path: str | Path,
    basin_poly=None,
    min_outlet_fraction: float = 0.5,
    min_in_basin_fraction: float = 0.9,
    min_secondary_fraction: float = 0.05,
) -> gpd.GeoDataFrame:
    """Clip reaches to the FAC-delineated watershed(s) of the dominant outlet(s).

    The FAC network is extracted over ``basin + halo`` so accumulation is correct
    at the basin edges. The right unit to keep is **not** the HUC polygon: a HUC8
    pour point generally sits *upstream* of where the FAC mainstem finishes
    assembling (the mainstem keeps gathering tributaries below the pour point, out
    in the halo, before exiting the window). Clipping to the polygon therefore
    severs every tributary whose confluence with the trunk lies below the pour
    point, leaving the network fragmented. Instead, keep the contributing area of
    the dominant outlet — the largest connected D8 stream-cell component, which
    spans the halo trunk and all below-pour-point confluences and is one connected
    network by construction.

    Components are ranked by **outlet flow accumulation** (max accumulation over
    the component's cells, i.e. its drainage area at the window edge). A component
    is kept when **either**:

    - its outlet FAC is ``>= min_outlet_fraction`` of the largest — so a genuinely
      bifurcated window (two comparable rivers) retains both outlets; **or**
    - it is a substantial, unambiguously in-HUC tributary: ``>=
      min_in_basin_fraction`` of its cells fall inside ``basin_poly`` **and** its
      outlet FAC is ``>= min_secondary_fraction`` of the largest. This retains an
      in-HUC tributary whose confluence with the mainstem falls just outside the
      window (so it is a separate D8 component here) instead of discarding it.
      Such a component stays a disconnected subgraph — it is kept for coverage, not
      connectivity. Requires ``basin_poly``; with no polygon only the outlet-FAC
      rule applies.

    Reaches are assigned to the kept watershed(s) by majority vote of their
    vertices over the kept-cell mask. ``stream_id`` is reset to a contiguous range
    over the kept reaches.
    """
    with rasterio.open(streams_raster_path) as s:
        streams_arr = s.read(1)
        transform = s.transform
        shp = s.shape
    with rasterio.open(d8_path) as d:
        d8_arr = d.read(1)

    rows, cols, labels, sizes = d8_stream_cell_components(streams_arr, d8_arr)
    if len(sizes) == 0:
        log.warning("  fac-watershed clip: no stream cells; returning input unchanged")
        return streams_gdf

    with rasterio.open(fac_path) as f:
        fac_arr = f.read(1)
    cell_fac = fac_arr[rows, cols].astype(np.float64)
    ncomp = len(sizes)
    outlet_fac = np.zeros(ncomp)
    np.maximum.at(outlet_fac, labels, cell_fac)

    order = np.argsort(outlet_fac)[::-1]
    max_fac = outlet_fac[order[0]]

    in_basin_frac: dict[int, float] = {}
    if basin_poly is not None:
        bpoly = basin_poly
        if isinstance(bpoly, gpd.GeoDataFrame):
            bpoly = bpoly.geometry
        if isinstance(bpoly, gpd.GeoSeries):
            bpoly = bpoly.union_all()
        bmask = rasterize(
            [(bpoly, 1)], out_shape=shp, transform=transform, fill=0, dtype="uint8"
        ).astype(bool)
        cell_in = bmask[rows, cols]
        for k in order:
            if outlet_fac[k] < min(0.01, min_secondary_fraction) * max_fac:
                break
            in_basin_frac[int(k)] = float(cell_in[labels == k].mean())

    # Keep the dominant outlet (and any comparable outlet), plus any substantial,
    # unambiguously in-HUC tributary whose out-of-window confluence made it a
    # separate component here.
    keep_labels: set[int] = set()
    for k in order:
        k = int(k)
        is_dominant = outlet_fac[k] >= min_outlet_fraction * max_fac
        is_in_basin_trib = (
            in_basin_frac.get(k, 0.0) >= min_in_basin_fraction
            and outlet_fac[k] >= min_secondary_fraction * max_fac
        )
        if is_dominant or is_in_basin_trib:
            keep_labels.add(k)

    px, py = abs(transform.a), abs(transform.e)
    cell_km2 = px * py / 1e6
    log.info(
        "  fac-watershed clip: %d components; keeping %d "
        "(outlet >= %.0f%% of max, or >= %.0f%% in-basin and >= %.0f%% of max)",
        ncomp,
        len(keep_labels),
        100 * min_outlet_fraction,
        100 * min_in_basin_fraction,
        100 * min_secondary_fraction,
    )
    for rank, k in enumerate(order):
        if outlet_fac[k] < 0.01 * max_fac:
            break
        sub = labels == k
        oi = int(np.argmax(cell_fac[sub]))
        orow, ocol = rows[sub][oi], cols[sub][oi]
        ox, oy = transform * (ocol + 0.5, orow + 0.5)
        ib = f", in_basin={in_basin_frac[int(k)] * 100:.0f}%" if in_basin_frac else ""
        log.info(
            "    rank %d: %d cells, outlet_fac=%.0f km2 at (%.0f,%.0f)%s -> %s",
            rank,
            int(sizes[k]),
            outlet_fac[k] * cell_km2,
            ox,
            oy,
            ib,
            "KEEP" if int(k) in keep_labels else "drop",
        )
    if len(keep_labels) > 1:
        log.warning(
            "  fac-watershed clip: %d outlets kept (multi-outlet window) -- review",
            len(keep_labels),
        )
    dropped = [k for k in order if int(k) not in keep_labels]
    if dropped and outlet_fac[dropped[0]] >= 0.1 * max_fac:
        log.warning(
            "  fac-watershed clip: largest DROPPED outlet is %.0f%% of max"
            " -- possible multi-outlet HUC, review",
            100 * outlet_fac[dropped[0]] / max_fac,
        )
    if in_basin_frac and in_basin_frac.get(int(order[0]), 1.0) < 0.5:
        log.warning(
            "  fac-watershed clip: dominant kept component is only %.0f%% in-basin"
            " -- a foreign river may be clipping through the window; review",
            100 * in_basin_frac[int(order[0])],
        )

    keep_cell = np.zeros(shp, dtype=bool)
    keep_idx = np.isin(labels, list(keep_labels))
    keep_cell[rows[keep_idx], cols[keep_idx]] = True

    inv = ~transform

    def _on_keep(geom) -> bool:
        xy = np.asarray(geom.coords)
        cc, rr = inv * (xy[:, 0], xy[:, 1])
        rr = rr.astype(int)
        cc = cc.astype(int)
        good = (rr >= 0) & (rr < shp[0]) & (cc >= 0) & (cc < shp[1])
        if not good.any():
            return False
        return keep_cell[rr[good], cc[good]].mean() >= 0.5

    n_before = len(streams_gdf)
    kept = streams_gdf[streams_gdf.geometry.apply(_on_keep)].reset_index(drop=True)
    kept["stream_id"] = range(len(kept))
    log.info(
        "  fac-watershed clip: kept %d of %d reaches (dropped %d off-watershed)",
        len(kept),
        n_before,
        n_before - len(kept),
    )
    return kept


def compute_regional_fac(
    dem_path: str | Path,
    out_dir: str | Path,
    threshold: int = 5000,
    max_procs: int = 32,
    basin_poly=None,
) -> Path:
    """Run D8 flow accumulation and stream extraction on a regional DEM.

    When ``basin_poly`` (the unbuffered basin polygon, same CRS as the DEM) is
    supplied, the extracted network is clipped to the FAC-delineated watershed of
    the dominant window outlet via :func:`clip_streams_to_fac_watershed` — keeping
    the basin's own connected drainage (including its trunk where it assembles
    below the HUC pour point, out in the halo) and dropping foreign catchments
    that exit a different window edge. ``basin_poly`` is also used for the
    in-basin diagnostic log only. It defaults to ``None`` (no clip), preserving
    behavior for callers with no associated basin polygon.

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
    if basin_poly is not None:
        gdf = clip_streams_to_fac_watershed(
            gdf, fdir, streams_ras, acc, basin_poly=basin_poly
        )
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
# 1m FAC with 10m inflow injection
# ---------------------------------------------------------------------------

# WhiteboxTools D8 pointer encoding (clockwise from NE): value → (row, col)
# offset, with row increasing downward. This matches WBT's d8_pointer output and
# is NOT the ESRI scheme — stepping along it follows flow downstream.
_D8_OFFSETS = {
    1: (-1, 1),  # NE
    2: (0, 1),  # E
    4: (1, 1),  # SE
    8: (1, 0),  # S
    16: (1, -1),  # SW
    32: (0, -1),  # W
    64: (-1, -1),  # NW
    128: (-1, 0),  # N
}


def find_inflow_points(
    regional_fac_path: str | Path,
    regional_d8_path: str | Path,
    aoi_bounds: tuple[float, float, float, float],
    aoi_crs,
    min_acc: int = 1000,
) -> list[dict]:
    """Find cells where regional flow enters the AOI boundary.

    Returns list of dicts with keys: x, y, acc_10m, row_10m, col_10m.
    Only returns points with accumulation >= min_acc (10m cells).
    """
    from rasterio.windows import from_bounds

    left, bottom, right, top = aoi_bounds

    with (
        rasterio.open(str(regional_fac_path)) as fac_src,
        rasterio.open(str(regional_d8_path)) as d8_src,
    ):
        # Read a strip around the AOI boundary (one 10m cell wider)
        res = fac_src.transform[0]  # 10m
        pad = res * 2
        window = from_bounds(
            left - pad,
            bottom - pad,
            right + pad,
            top + pad,
            fac_src.transform,
        )
        # Clamp to valid raster extent
        window = window.intersection(
            rasterio.windows.Window(0, 0, fac_src.width, fac_src.height)
        )
        fac_arr = fac_src.read(1, window=window)
        d8_arr = d8_src.read(1, window=window)
        win_transform = fac_src.window_transform(window)

    ny, nx = fac_arr.shape
    inflows = []

    # Scan boundary of the AOI extent in the 10m grid
    # For each 10m cell just outside the AOI, check if its D8 direction
    # points to a cell inside the AOI
    for r in range(ny):
        for c in range(nx):
            d8_val = int(d8_arr[r, c])
            if d8_val not in _D8_OFFSETS:
                continue
            acc = float(fac_arr[r, c])
            if acc < min_acc:
                continue

            # Map coords of this cell
            x, y = win_transform * (c + 0.5, r + 0.5)
            outside = x < left or x > right or y < bottom or y > top

            if not outside:
                continue

            # Check if D8 flows into the AOI
            dr, dc = _D8_OFFSETS[d8_val]
            tr, tc = r + dr, c + dc
            if 0 <= tr < ny and 0 <= tc < nx:
                tx, ty = win_transform * (tc + 0.5, tr + 0.5)
                inside = left <= tx <= right and bottom <= ty <= top
                if inside:
                    inflows.append(
                        {
                            "x": tx,  # target cell (inside AOI)
                            "y": ty,
                            "acc_10m": acc,
                        }
                    )

    # Deduplicate: keep highest accumulation per unique target cell
    seen: dict[tuple[float, float], dict] = {}
    for pt in inflows:
        key = (round(pt["x"], 1), round(pt["y"], 1))
        if key not in seen or pt["acc_10m"] > seen[key]["acc_10m"]:
            seen[key] = pt
    inflows = sorted(seen.values(), key=lambda p: -p["acc_10m"])
    log.info(
        "Found %d inflow points (max %.0f 10m-cells = %.0f km²)",
        len(inflows),
        inflows[0]["acc_10m"] if inflows else 0,
        inflows[0]["acc_10m"] * 100 / 1e6 if inflows else 0,
    )
    return inflows


def augment_accumulation(
    acc_path: str | Path,
    d8_path: str | Path,
    regional_d8_path: str | Path,
    inflow_points: list[dict],
    out_path: str | Path,
    scale_factor: float = 100.0,
) -> Path:
    """Trace downstream from each inflow point, adding scaled accumulation.

    scale_factor converts 10m cell counts to 1m equivalents (10² = 100).
    When the initial target cell is nodata in the 1m grid (common at DEM
    boundary fringes), the 10m D8 pointer is followed inward until a valid
    1m cell is found.
    """
    out_path = Path(out_path)

    with rasterio.open(str(acc_path)) as src:
        acc = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        transform = src.transform
    with rasterio.open(str(d8_path)) as src:
        d8 = src.read(1)
    with rasterio.open(str(regional_d8_path)) as src:
        reg_d8 = src.read(1)
        reg_transform = src.transform

    inv = ~transform
    reg_inv = ~reg_transform
    ny, nx = acc.shape
    reg_ny, reg_nx = reg_d8.shape
    total_augmented = 0
    _MAX_REGIONAL_HOPS = 200

    for pt in inflow_points:
        inflow_val = pt["acc_10m"] * scale_factor
        # Find the 1m cell closest to the inflow target
        fc, fr = inv * (pt["x"], pt["y"])
        r, c = int(round(fr)), int(round(fc))

        # If target cell is nodata, follow the 10m D8 pointer inward
        if r < 0 or r >= ny or c < 0 or c >= nx or int(d8[r, c]) not in _D8_OFFSETS:
            # Start from the inflow target in the 10m grid and trace forward
            rfc, rfr = reg_inv * (pt["x"], pt["y"])
            rr10, rc10 = int(round(rfr)), int(round(rfc))
            found = False
            for _ in range(_MAX_REGIONAL_HOPS):
                d8_10 = (
                    int(reg_d8[rr10, rc10])
                    if (0 <= rr10 < reg_ny and 0 <= rc10 < reg_nx)
                    else 0
                )
                if d8_10 not in _D8_OFFSETS:
                    break
                dr10, dc10 = _D8_OFFSETS[d8_10]
                rr10, rc10 = rr10 + dr10, rc10 + dc10
                # Map this 10m cell back to 1m
                x10, y10 = reg_transform * (rc10 + 0.5, rr10 + 0.5)
                fc1, fr1 = inv * (x10, y10)
                r1, c1 = int(round(fr1)), int(round(fc1))
                if 0 <= r1 < ny and 0 <= c1 < nx and int(d8[r1, c1]) in _D8_OFFSETS:
                    r, c = r1, c1
                    found = True
                    break
            if not found:
                log.warning(
                    "  inflow %.0f km² at (%.1f, %.1f): no valid 1m cell found",
                    pt["acc_10m"] * 100 / 1e6,
                    pt["x"],
                    pt["y"],
                )
                continue

        # Trace downstream, adding inflow at every cell
        visited = set()
        steps = 0
        while 0 <= r < ny and 0 <= c < nx:
            if (r, c) in visited:
                break
            visited.add((r, c))
            acc[r, c] += inflow_val
            steps += 1

            d8_val = int(d8[r, c])
            if d8_val not in _D8_OFFSETS:
                break
            dr, dc = _D8_OFFSETS[d8_val]
            r, c = r + dr, c + dc

        total_augmented += steps
        log.info(
            "  inflow %.0f km² traced %d cells downstream",
            pt["acc_10m"] * 100 / 1e6,
            steps,
        )

    # Write augmented accumulation
    profile.update(dtype="float64")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(acc, 1)

    log.info("Augmented %d cells total, wrote %s", total_augmented, out_path)
    return out_path


def compute_aoi_fac_with_inflow(
    dem_path: str | Path,
    regional_fac_path: str | Path,
    regional_d8_path: str | Path,
    out_dir: str | Path,
    threshold: int = 50_000,
    max_procs: int = 32,
    min_inflow_acc: int = 1000,
    regional_res_m: float = 10.0,
    strahler_path: str | Path | None = None,
) -> Path:
    """Run 1m FAC on an AOI with inflow injection from the 10m regional grid.

    Returns the path to ``streams_fac.fgb``.
    """
    dem_path = str(dem_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as src:
        aoi_bounds = src.bounds
        aoi_crs = src.crs
        dem_res = src.transform[0]

    scale_factor = (regional_res_m / dem_res) ** 2

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_max_procs(max_procs)
    wbt.set_compress_rasters(True)

    filled = str(out_dir / "dem_filled.tif")
    fdir = str(out_dir / "d8_pointer.tif")
    acc = str(out_dir / "flow_accumulation.tif")
    acc_aug = str(out_dir / "flow_accumulation_augmented.tif")
    streams_ras = str(out_dir / "streams.tif")
    streams_shp = str(out_dir / "streams_raw.shp")
    strahler_out = str(out_dir / "stream_order.tif")
    streams_fgb = str(out_dir / "streams_fac.fgb")

    t0 = time.time()

    # 1. Fill depressions
    if not os.path.exists(filled):
        log.info("Filling depressions ...")
        wbt.fill_depressions_wang_and_liu(dem_path, filled, fix_flats=True)
        log.info("  done in %.0fs", time.time() - t0)
    else:
        log.info("Skipping fill — %s exists", filled)

    # 2. D8 pointer
    if not os.path.exists(fdir):
        log.info("Computing D8 pointer ...")
        wbt.d8_pointer(filled, fdir)
        log.info("  done in %.0fs", time.time() - t0)
    else:
        log.info("Skipping D8 pointer — %s exists", fdir)

    # 3. D8 flow accumulation (raw, without inflow)
    if not os.path.exists(acc):
        log.info("Computing D8 flow accumulation ...")
        wbt.d8_flow_accumulation(fdir, acc, out_type="cells", pntr=True, log=False)
        log.info("  done in %.0fs", time.time() - t0)
    else:
        log.info("Skipping accumulation — %s exists", acc)

    # 4. Find inflow points and augment
    log.info("Finding inflow points from regional grid ...")
    inflows = find_inflow_points(
        regional_fac_path,
        regional_d8_path,
        (aoi_bounds.left, aoi_bounds.bottom, aoi_bounds.right, aoi_bounds.top),
        aoi_crs,
        min_acc=min_inflow_acc,
    )
    if inflows:
        log.info("Augmenting accumulation with %d inflow points ...", len(inflows))
        augment_accumulation(
            acc,
            fdir,
            str(regional_d8_path),
            inflows,
            acc_aug,
            scale_factor=scale_factor,
        )
        extract_acc = acc_aug
    else:
        log.info("No significant inflow — using raw accumulation")
        extract_acc = acc

    # 5. Extract streams from (augmented) accumulation
    log.info("Extracting streams (threshold=%d) ...", threshold)
    wbt.extract_streams(extract_acc, streams_ras, threshold=threshold)
    log.info("  done in %.0fs", time.time() - t0)

    # 6. Strahler stream order
    log.info("Computing Strahler stream order ...")
    wbt.strahler_stream_order(fdir, streams_ras, strahler_out)
    log.info("  done in %.0fs", time.time() - t0)

    # 7. Vectorize
    log.info("Vectorizing streams ...")
    wbt.raster_streams_to_vector(streams_ras, fdir, streams_shp)
    log.info("  done in %.0fs", time.time() - t0)

    # 8. Post-process
    log.info("Post-processing ...")
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
    merged = linemerge(MultiLineString(geoms))
    if merged.geom_type == "LineString":
        final_geoms = [merged]
    elif merged.geom_type == "MultiLineString":
        final_geoms = list(merged.geoms)
    else:
        final_geoms = [merged]

    final_geoms = snap_dangling_ends(final_geoms, snap_tol=2.0)

    # Sample Strahler from per-AOI order raster (not regional)
    orders = sample_stream_order(final_geoms, strahler_out)

    gdf = gpd.GeoDataFrame(
        {"stream_id": range(len(final_geoms)), "strahler": orders},
        geometry=final_geoms,
        crs=dem_crs,
    )
    gdf["length_m"] = gdf.geometry.length
    gdf.to_file(streams_fgb, driver="FlatGeobuf")
    log.info(
        "Wrote %d streams (%.0f km) in %.0fs → %s",
        len(gdf),
        gdf.length_m.sum() / 1000,
        time.time() - t0,
        streams_fgb,
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
