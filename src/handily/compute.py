import logging
from collections import deque

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS as _CRS
from rasterio import features
from rasterstats import zonal_stats
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

LOGGER = logging.getLogger("handily.compute")


def rasterize_lines_to_grid(lines_gdf, template_da, burn_value=1):
    """Rasterize line features to match a template DataArray grid."""
    transform = template_da.rio.transform()
    shape = template_da.shape

    shapes = [(geom, burn_value) for geom in lines_gdf.geometry if geom is not None]
    if len(shapes) == 0:
        raise ValueError("No valid geometries found to rasterize.")

    stream_arr = features.rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    stream_da = xr.DataArray(
        stream_arr,
        dims=template_da.dims,
        coords=template_da.coords,
        name="streams",
    )
    return stream_da.rio.write_crs(template_da.rio.crs, inplace=False)


def build_streams_mask_from_nhd_ndwi(
    flowlines_gdf, dem_da, ndwi_da=None, ndwi_threshold=None, flowlines_buffer_m=None
):
    """Build a streams/water mask on the DEM grid by combining NHD flowlines with NDWI.

    Parameters
    ----------
    flowlines_gdf : GeoDataFrame
        NHD flowlines vector data.
    dem_da : xarray.DataArray
        DEM raster with CRS (assumed to be in meters for projected systems).
    ndwi_da : xarray.DataArray, optional
        NDWI raster for water detection.
    ndwi_threshold : float, optional
        Threshold for NDWI water mask (pixels > threshold are water).
    flowlines_buffer_m : float, optional
        Buffer distance in meters to apply to flowlines before rasterization.
        This helps capture NDWI water detections when NHD flowlines are
        misaligned with actual channel positions.
    """
    if dem_da.rio.crs is None:
        raise ValueError("DEM must have a valid CRS.")
    if flowlines_gdf.crs is None or str(flowlines_gdf.crs) != str(dem_da.rio.crs):
        flowlines_gdf = flowlines_gdf.to_crs(dem_da.rio.crs)

    # Optionally buffer flowlines to capture nearby NDWI detections
    if flowlines_buffer_m is not None and float(flowlines_buffer_m) > 0:
        LOGGER.info("Buffering flowlines by %.1f meters", flowlines_buffer_m)
        flowlines_gdf = flowlines_gdf.copy()
        flowlines_gdf["geometry"] = flowlines_gdf.geometry.buffer(
            float(flowlines_buffer_m)
        )

    streams_mask = rasterize_lines_to_grid(flowlines_gdf, dem_da, burn_value=1)

    if ndwi_da is not None and ndwi_threshold is not None:
        ndwi_match = ndwi_da
        if str(ndwi_da.rio.crs) != str(dem_da.rio.crs) or ndwi_da.shape != dem_da.shape:
            ndwi_match = ndwi_da.rio.reproject_match(dem_da)
        water = (ndwi_match > float(ndwi_threshold)).astype("uint8")
        water = water.rio.write_crs(dem_da.rio.crs, inplace=False)
        combo = xr.DataArray(
            np.asarray(streams_mask.data, dtype="uint8")
            * np.asarray(water.data, dtype="uint8"),
            dims=dem_da.dims,
            coords=dem_da.coords,
            name="streams",
        )
        streams_mask = combo.rio.write_crs(dem_da.rio.crs, inplace=False)

    return streams_mask


def _build_nhd_adjacency(
    flowlines_gdf: gpd.GeoDataFrame,
    snap_tolerance: float = 1.0,
) -> dict[int, list[int]]:
    """Build adjacency graph from shared endpoints of LineString geometries.

    Parameters
    ----------
    flowlines_gdf : GeoDataFrame
        Flowlines in a projected CRS (meters). Index must be RangeIndex 0..n-1.
    snap_tolerance : float
        Max distance (meters) between endpoints to consider them connected.

    Returns
    -------
    dict mapping feature index → list of neighbor feature indices.
    """
    from collections import defaultdict

    n = len(flowlines_gdf)
    if n == 0:
        return {}

    endpoints = np.empty((2 * n, 2), dtype=np.float64)
    owner = np.empty(2 * n, dtype=np.intp)

    for i, geom in enumerate(flowlines_gdf.geometry):
        if geom is None or geom.is_empty:
            endpoints[2 * i] = [np.inf, np.inf]
            endpoints[2 * i + 1] = [np.inf, np.inf]
        elif geom.geom_type == "LineString":
            endpoints[2 * i] = geom.coords[0][:2]
            endpoints[2 * i + 1] = geom.coords[-1][:2]
        elif geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
            endpoints[2 * i] = parts[0].coords[0][:2]
            endpoints[2 * i + 1] = parts[-1].coords[-1][:2]
        else:
            endpoints[2 * i] = [np.inf, np.inf]
            endpoints[2 * i + 1] = [np.inf, np.inf]
        owner[2 * i] = i
        owner[2 * i + 1] = i

    tree = cKDTree(endpoints)
    pairs = tree.query_pairs(r=snap_tolerance)

    adj = defaultdict(set)
    for p, q in pairs:
        fi, fj = int(owner[p]), int(owner[q])
        if fi != fj:
            adj[fi].add(fj)
            adj[fj].add(fi)

    return {i: list(adj.get(i, set())) for i in range(n)}


def filter_disconnected_flowlines(
    flowlines_gdf: gpd.GeoDataFrame,
    min_component_km: float = 5.0,
    snap_tolerance: float = 1.0,
) -> gpd.GeoDataFrame:
    """Drop flowline segments that belong to small disconnected network components.

    Builds an adjacency graph across ALL flowlines (regardless of FCODE) and
    finds connected components.  Components whose total length is below
    *min_component_km* are dropped.  This removes isolated pond crossings,
    disconnected intermittent tributaries, and other fragments that don't
    connect to the main channel network.

    Parameters
    ----------
    flowlines_gdf : GeoDataFrame
        Flowlines in a projected CRS (meters).
    min_component_km : float
        Minimum total length (km) for a connected component to be kept.
    snap_tolerance : float
        Endpoint snap tolerance (meters) for adjacency.
    """
    n = len(flowlines_gdf)
    if n == 0:
        return flowlines_gdf

    flowlines_gdf = flowlines_gdf.reset_index(drop=True)
    adj = _build_nhd_adjacency(flowlines_gdf, snap_tolerance=snap_tolerance)

    visited: set[int] = set()
    keep_idx: set[int] = set()
    for start in range(n):
        if start in visited:
            continue
        comp: list[int] = []
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for nb in adj.get(cur, []):
                if nb not in visited:
                    queue.append(nb)
        total_km = float(flowlines_gdf.iloc[comp]["lengthkm"].sum())
        if total_km >= min_component_km:
            keep_idx.update(comp)

    n_dropped = n - len(keep_idx)
    if n_dropped:
        LOGGER.info(
            "Dropped %d disconnected flowline segments (< %.1f km components)",
            n_dropped,
            min_component_km,
        )

    return flowlines_gdf.iloc[sorted(keep_idx)].reset_index(drop=True)


def build_network_propagated_mask(
    flowlines_gdf,
    dem_da,
    ndwi_da,
    ndwi_threshold: float,
    flowlines_buffer_m: float,
    max_hops: int | None = None,
    snap_tolerance: float = 1.0,
):
    """Build a stream mask using NDWI-seeded network propagation.

    If any pixel of an NHD segment overlaps the NDWI mask, the entire segment
    is confirmed.  Neighboring segments are also confirmed via BFS up to
    *max_hops* (None = unlimited).  Returns a binary mask identical in format
    to ``build_streams_mask_from_nhd_ndwi``.
    """
    if dem_da.rio.crs is None:
        raise ValueError("DEM must have a valid CRS.")
    if flowlines_gdf.crs is None or str(flowlines_gdf.crs) != str(dem_da.rio.crs):
        flowlines_gdf = flowlines_gdf.to_crs(dem_da.rio.crs)
    flowlines_gdf = flowlines_gdf.reset_index(drop=True)
    n_features = len(flowlines_gdf)
    if n_features == 0:
        raise ValueError("No flowline features to rasterize.")

    transform = dem_da.rio.transform()
    shape = dem_da.shape

    # Step 1: rasterize each feature with a unique ID (1-based, 0 = background)
    id_shapes = [
        (geom, i + 1)
        for i, geom in enumerate(flowlines_gdf.geometry)
        if geom is not None
    ]
    id_raster = features.rasterize(
        shapes=id_shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint16",
    )

    # Step 2: build NDWI-confirmed mask (buffered NHD AND NDWI > threshold)
    buffered = flowlines_gdf.copy()
    buffered["geometry"] = buffered.geometry.buffer(float(flowlines_buffer_m))
    buf_raster = features.rasterize(
        shapes=[(g, 1) for g in buffered.geometry if g is not None],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    ndwi_match = ndwi_da
    if str(ndwi_da.rio.crs) != str(dem_da.rio.crs) or ndwi_da.shape != dem_da.shape:
        ndwi_match = ndwi_da.rio.reproject_match(dem_da)
    water = np.asarray((ndwi_match > float(ndwi_threshold)).data, dtype="uint8")
    ndwi_mask = (buf_raster > 0) & (water > 0)

    # Step 3: determine seed features (any cell of feature overlaps NDWI mask)
    confirmed_ids = set(np.unique(id_raster[ndwi_mask]).tolist()) - {0}
    LOGGER.info(
        "NDWI seed features: %d / %d (%.0f%%)",
        len(confirmed_ids),
        n_features,
        100.0 * len(confirmed_ids) / n_features if n_features else 0,
    )

    # Step 4: build adjacency graph and BFS propagation
    adjacency = _build_nhd_adjacency(flowlines_gdf, snap_tolerance=snap_tolerance)

    seeds_0 = {fid - 1 for fid in confirmed_ids}
    visited = set(seeds_0)
    queue = deque((fidx, 0) for fidx in seeds_0)

    while queue:
        current, hops = queue.popleft()
        if max_hops is not None and hops >= max_hops:
            continue
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, hops + 1))

    confirmed_raster_ids = {fidx + 1 for fidx in visited}
    n_propagated = len(confirmed_raster_ids) - len(confirmed_ids)
    LOGGER.info(
        "After BFS (max_hops=%s): %d confirmed (%d propagated from %d seeds)",
        max_hops,
        len(confirmed_raster_ids),
        n_propagated,
        len(confirmed_ids),
    )

    # Step 5: burn confirmed features as binary mask (unbuffered)
    confirmed_shapes = [
        (geom, 1)
        for i, geom in enumerate(flowlines_gdf.geometry)
        if geom is not None and (i + 1) in confirmed_raster_ids
    ]
    if not confirmed_shapes:
        raise ValueError("No features confirmed after network propagation.")

    mask_arr = features.rasterize(
        shapes=confirmed_shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    mask_da = xr.DataArray(
        mask_arr,
        dims=dem_da.dims,
        coords=dem_da.coords,
        name="streams",
    )
    mask_da = mask_da.rio.write_crs(dem_da.rio.crs, inplace=False)
    mask_da.attrs["propagation_n_seeds"] = len(confirmed_ids)
    mask_da.attrs["propagation_n_confirmed"] = len(confirmed_raster_ids)
    mask_da.attrs["propagation_n_propagated"] = n_propagated
    return mask_da


def compute_rem_quick(dem_da, streams_da, radius: int = 1000):
    """Compute a quick Relative Elevation Model using a local mean water-surface base elevation."""
    if dem_da.rio.crs is None or streams_da.rio.crs is None:
        raise ValueError("Both DEM and streams rasters must have a valid CRS.")
    if (
        str(dem_da.rio.crs) != str(streams_da.rio.crs)
        or dem_da.shape != streams_da.shape
    ):
        raise ValueError("DEM and streams must share grid shape and CRS.")

    if int(radius) < 1:
        raise ValueError("radius must be >= 1")

    dem_np = np.asarray(dem_da.data, dtype="float32")
    streams_np = np.asarray(streams_da.data).astype(bool)
    if streams_np.sum() == 0:
        raise ValueError("Stream mask has no active cells after combination.")

    streams_f = streams_np.astype("float32")
    sigma = float(radius) / 3.0
    mean_elev_masked = ndi.gaussian_filter(
        dem_np * streams_f, sigma=sigma, mode="nearest", truncate=3.0
    )
    mean_mask = ndi.gaussian_filter(
        streams_f, sigma=sigma, mode="nearest", truncate=3.0
    )
    base_fallback = float(np.nanmean(dem_np[streams_np]))
    base_elev = np.divide(
        mean_elev_masked,
        mean_mask,
        out=np.full_like(mean_elev_masked, base_fallback, dtype="float32"),
        where=mean_mask > 0,
    )

    rem_np = dem_np - base_elev
    rem_np = np.where(rem_np < 0, 0, rem_np)

    rem_da = xr.DataArray(
        rem_np,
        dims=dem_da.dims,
        coords=dem_da.coords,
        name="REM",
        attrs={"description": "Quick REM (local mean water base; no sink filling)"},
    )
    return rem_da.rio.write_crs(dem_da.rio.crs, inplace=False)


def compute_rem_edt_smooth(dem_da, streams_da, sigma: float = 50.0):
    """Compute REM via EDT nearest-cell allocation then Gaussian-smooth the water surface.

    1. For each pixel, assign the DEM elevation of the nearest stream cell (EDT).
       This is direction-invariant but produces Voronoi-like discontinuities.
    2. Gaussian-smooth the resulting water surface to remove tessellation edges.
       Because the input is already approximately correct, the smooth blends only
       nearby values and does not reintroduce the upstream/downstream mixing
       that plagues a direct Gaussian on sparse stream cells.
    3. REM = DEM − smoothed water surface, clipped to 0.

    Parameters
    ----------
    dem_da : DataArray
        DEM raster in a projected CRS (units: meters).
    streams_da : DataArray
        Binary stream mask (1 = stream cell).
    sigma : float
        Gaussian sigma in pixels (= meters at 1 m resolution) for smoothing
        the EDT water surface. Default 50.
    """
    if dem_da.rio.crs is None or streams_da.rio.crs is None:
        raise ValueError("Both DEM and streams rasters must have a valid CRS.")
    if (
        str(dem_da.rio.crs) != str(streams_da.rio.crs)
        or dem_da.shape != streams_da.shape
    ):
        raise ValueError("DEM and streams must share grid shape and CRS.")

    dem_np = np.asarray(dem_da.data, dtype="float32")
    streams_np = np.asarray(streams_da.data).astype(bool)
    n_stream = int(streams_np.sum())
    if n_stream == 0:
        raise ValueError("Stream mask has no active cells after combination.")
    LOGGER.info("Stream mask: %d cells, EDT sigma=%.0f", n_stream, sigma)

    # Step 1: nearest-cell allocation — piecewise-constant water surface
    indices = ndi.distance_transform_edt(
        ~streams_np, return_distances=False, return_indices=True
    )
    base_elev = dem_np[indices[0], indices[1]]

    # Step 2: cap water surface at terrain — prevents high tributary stream
    # cells from injecting elevated values into the smoothing field.  Pixels
    # mis-assigned by EDT to a higher stream get capped to their own DEM
    # elevation, so the Gaussian can pull them toward nearby lower streams.
    base_elev = np.minimum(base_elev, dem_np)

    # Step 3: Gaussian smooth to remove Voronoi edges
    base_elev = ndi.gaussian_filter(base_elev, sigma=sigma, mode="nearest")

    # Step 4: post-cap — the smooth can blend capped values from higher
    # neighbors, pushing the surface above terrain; clamp it back down.
    base_elev = np.minimum(base_elev, dem_np)

    rem_np = dem_np - base_elev
    rem_np = np.where(rem_np < 0, 0, rem_np)

    rem_da = xr.DataArray(
        rem_np,
        dims=dem_da.dims,
        coords=dem_da.coords,
        name="REM",
        attrs={"description": f"REM (EDT + Gaussian smooth, sigma={sigma})"},
    )
    return rem_da.rio.write_crs(dem_da.rio.crs, inplace=False)


def compute_field_rem_stats(fields_gdf, rem_da, stats=("mean",)):
    """Compute zonal statistics of REM over polygon features."""
    LOGGER.info("Computing zonal statistics over fields (stats: %s)", ",".join(stats))

    rem_crs = rem_da.rio.crs
    if rem_crs is None:
        raise ValueError("REM raster has no CRS; cannot compute zonal stats.")
    if fields_gdf.crs is None:
        raise ValueError("Fields GeoDataFrame has no CRS; cannot compute zonal stats.")
    try:
        if not _CRS.from_user_input(fields_gdf.crs).equals(
            _CRS.from_user_input(rem_crs)
        ):
            fields_gdf = fields_gdf.to_crs(rem_crs)
    except Exception:
        if str(fields_gdf.crs) != str(rem_crs):
            fields_gdf = fields_gdf.to_crs(rem_crs)

    affine = rem_da.rio.transform()
    raster = np.asarray(rem_da.data)
    raster = np.ma.array(raster, mask=~np.isfinite(raster))

    zs = zonal_stats(
        vectors=fields_gdf.geometry,
        raster=raster,
        affine=affine,
        stats=list(stats),
        nodata=None,
        all_touched=True,
        geojson_out=False,
    )

    df_stats = pd.DataFrame(zs)
    df_stats = df_stats.rename(columns={c: f"rem_{c}" for c in df_stats.columns})
    return fields_gdf.reset_index(drop=True).join(df_stats)


def stratify_fields_by_rem(fields_with_stats_gdf, threshold_m=2.0):
    """Add boolean column `partitioned` where mean REM < threshold."""
    if "rem_mean" not in fields_with_stats_gdf.columns:
        raise ValueError("'rem_mean' column not found; compute stats first.")
    out = fields_with_stats_gdf.copy()
    out["partitioned"] = out["rem_mean"] < float(threshold_m)
    return out
