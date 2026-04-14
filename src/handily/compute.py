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


def propagate_flowline_confirmation(
    flowlines_gdf,
    dem_da,
    ndwi_da,
    ndwi_threshold: float,
    flowlines_buffer_m: float,
    max_hops: int | None = None,
    snap_tolerance: float = 1.0,
) -> gpd.GeoDataFrame:
    """Annotate flowlines with water-seeding evidence and BFS reachability.

    Returns a copy of *flowlines_gdf* (projected to the DEM CRS) with columns:

    - ``flowline_id``: 0-based index matching the input row order
    - ``water_seeded``: True if the segment directly overlaps water pixels
    - ``water_seed_pixels``: count of water pixels overlapping the segment
    - ``water_seed_fraction``: fraction of segment raster cells with water
    - ``reachable_from_seed``: True if connected to a seeded segment via BFS
    - ``seed_hops``: BFS distance from nearest seeded segment (-1 if unreachable)
    - ``propagation_component_id``: connected-component label

    Legacy aliases (for backward compatibility with EDT pipeline):

    - ``propagation_seeded``: alias for ``water_seeded``
    - ``propagation_confirmed``: alias for ``reachable_from_seed``
    - ``propagation_hops``: alias for ``seed_hops``
    """
    if dem_da.rio.crs is None:
        raise ValueError("DEM must have a valid CRS.")
    if flowlines_gdf.crs is None or str(flowlines_gdf.crs) != str(dem_da.rio.crs):
        flowlines_gdf = flowlines_gdf.to_crs(dem_da.rio.crs)
    flowlines_gdf = flowlines_gdf.reset_index(drop=True)
    n_features = len(flowlines_gdf)
    if n_features == 0:
        raise ValueError("No flowline features to propagate.")

    transform = dem_da.rio.transform()
    shape = dem_da.shape

    # Rasterize each feature with a unique ID (1-based, 0 = background)
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

    # Water mask (buffered NHD AND water > threshold)
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

    # Per-feature water pixel counts and total raster cell counts
    water_seed_pixels = np.zeros(n_features, dtype=np.int64)
    segment_total_pixels = np.zeros(n_features, dtype=np.int64)
    for fid in range(1, n_features + 1):
        seg_mask = id_raster == fid
        segment_total_pixels[fid - 1] = int(seg_mask.sum())
        water_seed_pixels[fid - 1] = int((seg_mask & ndwi_mask).sum())

    seed_indices = {i for i in range(n_features) if water_seed_pixels[i] > 0}
    water_seed_fraction = np.where(
        segment_total_pixels > 0,
        water_seed_pixels / segment_total_pixels,
        0.0,
    )

    LOGGER.info(
        "Water-seeded features: %d / %d (%.0f%%)",
        len(seed_indices),
        n_features,
        100.0 * len(seed_indices) / n_features if n_features else 0,
    )

    # Adjacency graph and BFS
    adjacency = _build_nhd_adjacency(flowlines_gdf, snap_tolerance=snap_tolerance)

    # Connected components (for component_id labelling)
    comp_id = np.full(n_features, -1, dtype=np.intp)
    current_comp = 0
    for start in range(n_features):
        if comp_id[start] >= 0:
            continue
        stack = [start]
        while stack:
            node = stack.pop()
            if comp_id[node] >= 0:
                continue
            comp_id[node] = current_comp
            stack.extend(adjacency.get(node, []))
        current_comp += 1

    # BFS from seeds, tracking hop distance
    hop_dist = np.full(n_features, -1, dtype=np.intp)
    for s in seed_indices:
        hop_dist[s] = 0
    queue = deque((fidx, 0) for fidx in seed_indices)

    while queue:
        current, hops = queue.popleft()
        if max_hops is not None and hops >= max_hops:
            continue
        for neighbor in adjacency.get(current, []):
            if hop_dist[neighbor] < 0:
                hop_dist[neighbor] = hops + 1
                queue.append((neighbor, hops + 1))

    reachable_mask = hop_dist >= 0
    n_reachable = int(reachable_mask.sum())
    n_reachable_only = n_reachable - len(seed_indices)
    LOGGER.info(
        "After BFS (max_hops=%s): %d seeded, %d reachable-only, %d unreachable",
        max_hops,
        len(seed_indices),
        n_reachable_only,
        n_features - n_reachable,
    )

    out = flowlines_gdf.copy()
    out["flowline_id"] = np.arange(n_features)
    out["water_seeded"] = np.isin(np.arange(n_features), list(seed_indices))
    out["water_seed_pixels"] = water_seed_pixels
    out["water_seed_fraction"] = water_seed_fraction.astype(np.float32)
    out["reachable_from_seed"] = reachable_mask
    out["seed_hops"] = hop_dist
    out["propagation_component_id"] = comp_id
    # Legacy aliases
    out["propagation_seeded"] = out["water_seeded"]
    out["propagation_confirmed"] = out["reachable_from_seed"]
    out["propagation_hops"] = out["seed_hops"]
    return out


def rasterize_confirmed_flowlines(
    confirmed_flowlines_gdf: gpd.GeoDataFrame,
    dem_da,
) -> xr.DataArray:
    """Rasterize confirmed flowlines to a binary mask on the DEM grid.

    Parameters
    ----------
    confirmed_flowlines_gdf : GeoDataFrame
        Subset of flowlines where ``propagation_confirmed == True``.
    dem_da : xr.DataArray
        DEM used as the rasterization template.
    """
    transform = dem_da.rio.transform()
    shape = dem_da.shape
    confirmed_shapes = [
        (geom, 1) for geom in confirmed_flowlines_gdf.geometry if geom is not None
    ]
    if not confirmed_shapes:
        raise ValueError("No confirmed features to rasterize.")

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
    return mask_da.rio.write_crs(dem_da.rio.crs, inplace=False)


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

    Thin wrapper around :func:`propagate_flowline_confirmation` and
    :func:`rasterize_confirmed_flowlines`.
    """
    annotated = propagate_flowline_confirmation(
        flowlines_gdf,
        dem_da,
        ndwi_da,
        ndwi_threshold=ndwi_threshold,
        flowlines_buffer_m=flowlines_buffer_m,
        max_hops=max_hops,
        snap_tolerance=snap_tolerance,
    )
    confirmed = annotated[annotated["propagation_confirmed"]].copy()
    mask_da = rasterize_confirmed_flowlines(confirmed, dem_da)
    mask_da.attrs["propagation_n_seeds"] = int(annotated["propagation_seeded"].sum())
    mask_da.attrs["propagation_n_confirmed"] = len(confirmed)
    mask_da.attrs["propagation_n_propagated"] = int(
        len(confirmed) - annotated["propagation_seeded"].sum()
    )
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


def compute_rem_edt_smooth(
    dem_da, streams_da, sigma: float = 50.0, max_dist: float | None = None
):
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
    max_dist : float, optional
        Maximum distance (in pixels/meters at 1 m resolution) from the nearest
        stream cell. Pixels beyond this distance get NaN in the output REM.
        Default None (no limit).
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
    dist, indices = ndi.distance_transform_edt(
        ~streams_np, return_distances=True, return_indices=True
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

    if max_dist is not None:
        rem_np = np.where(dist > max_dist, np.nan, rem_np)
        n_masked = int((dist > max_dist).sum())
        pct = n_masked / dist.size * 100
        LOGGER.info("max_dist=%.0f: masked %d pixels (%.1f%%)", max_dist, n_masked, pct)

    rem_da = xr.DataArray(
        rem_np.astype("float32"),
        dims=dem_da.dims,
        coords=dem_da.coords,
        name="REM",
        attrs={
            "description": f"REM (EDT + Gaussian smooth, sigma={sigma}, max_dist={max_dist})"
        },
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
