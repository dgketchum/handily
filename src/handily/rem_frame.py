"""Anisotropic REM via thalweg-frame cross-section interpolation.

This module implements an experimental REM workflow that:

1. Takes confirmed NHD flowlines (from network-propagated NDWI seeding)
2. Decomposes them into simple reaches
3. Snaps each reach to the DEM thalweg via dynamic programming
4. Builds a smoothed curvilinear frame for stable cross-section normals
5. Samples cross-sections with ridge-stop logic
6. Rasterizes an anisotropic water surface from section interpolation
7. Computes REM = DEM - water_surface
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge

from .config import HandilyConfig

LOGGER = logging.getLogger("handily.rem_frame")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ReachMetrics:
    """Interpretable metrics for a snapped reach."""

    station_water_hit_fraction: float
    station_water_support_mean: float
    mean_snap_offset_m: float
    max_snap_offset_m: float
    max_consecutive_no_water_m: float
    n_stations: int
    n_supported_stations: int
    seeded_fraction: float = 0.0


@dataclass
class SnappedReach:
    reach_id: int
    prior_geom: LineString
    snapped_geom: LineString
    stations: gpd.GeoDataFrame
    confidence: float
    metrics: ReachMetrics | None = None


@dataclass
class FrameReach:
    reach_id: int
    frame_geom: LineString
    frame_stations: gpd.GeoDataFrame
    snapped_stations: gpd.GeoDataFrame


@dataclass
class CrossSectionSet:
    sections: gpd.GeoDataFrame
    support_polygons: gpd.GeoDataFrame


@dataclass
class AnisotropicREMResult:
    rem_da: xr.DataArray
    water_surface_da: xr.DataArray
    confirmed_flowlines: gpd.GeoDataFrame
    snapped_flowlines: gpd.GeoDataFrame
    frame_flowlines: gpd.GeoDataFrame
    cross_sections: gpd.GeoDataFrame
    support_polygons: gpd.GeoDataFrame
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 4: Reach decomposition
# ---------------------------------------------------------------------------


def split_confirmed_flowlines_into_reaches(
    confirmed_flowlines: gpd.GeoDataFrame,
    snap_tolerance: float = 1.0,
) -> gpd.GeoDataFrame:
    """Decompose confirmed flowlines into simple, non-branching reaches.

    Each outgoing branch from a junction becomes its own reach.  Chains are
    walked from junction features and dead-ends (degree != 2) along degree-2
    interior nodes.

    Returns a GeoDataFrame with columns:
        reach_id, component_id, n_flowlines, is_simple_chain,
        seeded_fraction, geometry
    """
    from collections import defaultdict
    from scipy.spatial import cKDTree

    gdf = confirmed_flowlines.reset_index(drop=True)
    n = len(gdf)
    empty_cols = [
        "reach_id",
        "component_id",
        "n_flowlines",
        "is_simple_chain",
        "seeded_fraction",
        "geometry",
    ]
    if n == 0:
        return gpd.GeoDataFrame(
            columns=empty_cols,
            geometry="geometry",
            crs=gdf.crs,
        )

    # Per-feature seeded status (carried from Phase 1 annotation)
    has_seeded_col = "water_seeded" in gdf.columns
    seeded = gdf["water_seeded"].values if has_seeded_col else np.ones(n, dtype=bool)

    # --- Build endpoint array and adjacency ---
    endpoints = np.empty((2 * n, 2), dtype=np.float64)
    owner = np.empty(2 * n, dtype=np.intp)
    for i, geom in enumerate(gdf.geometry):
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

    # --- Connected components ---
    comp_id = np.full(n, -1, dtype=np.intp)
    current_comp = 0
    for start in range(n):
        if comp_id[start] >= 0:
            continue
        stack = [start]
        while stack:
            node = stack.pop()
            if comp_id[node] >= 0:
                continue
            comp_id[node] = current_comp
            for nb in adj.get(node, set()):
                if comp_id[nb] < 0:
                    stack.append(nb)
        current_comp += 1

    # --- Identify junction nodes (endpoint clusters with degree != 2) ---
    groups = tree.query_ball_tree(tree, r=snap_tolerance)
    visited_ep = set()
    ep_cluster = np.full(2 * n, -1, dtype=np.intp)
    cluster_id = 0
    for ep_idx in range(2 * n):
        if ep_idx in visited_ep:
            continue
        cluster = set()
        for member in groups[ep_idx]:
            if member not in visited_ep:
                cluster.add(member)
                visited_ep.add(member)
        for m in cluster:
            ep_cluster[m] = cluster_id
        cluster_id += 1

    cluster_features = defaultdict(set)
    for ep_idx in range(2 * n):
        cid = ep_cluster[ep_idx]
        if cid >= 0:
            cluster_features[cid].add(int(owner[ep_idx]))

    junction_clusters = {
        cid for cid, feats in cluster_features.items() if len(feats) > 2
    }
    junction_features = set()
    for cid in junction_clusters:
        junction_features.update(cluster_features[cid])

    # --- Walk chains: one branch per direction from each junction/dead-end ---
    used = set()
    reaches = []

    def _walk_one_direction(start_feat, next_feat):
        """Walk a single branch from start_feat through next_feat."""
        chain = [start_feat]
        prev, cur = start_feat, next_feat
        while cur not in used:
            used.add(cur)
            chain.append(cur)
            if cur in junction_features:
                break
            nbs = adj.get(cur, set()) - {prev}
            if len(nbs) != 1:
                break
            prev, cur = cur, nbs.pop()
        return chain

    # Starters: junction features and dead-ends (degree != 2)
    starters = set()
    for fidx in range(n):
        degree = len(adj.get(fidx, set()))
        if degree != 2 or fidx in junction_features:
            starters.add(fidx)

    for start in sorted(starters):
        if start in used:
            continue
        neighbors = adj.get(start, set())
        if not neighbors:
            # Isolated feature
            used.add(start)
            reaches.append([start])
            continue
        # Walk each branch direction separately
        branches_started = 0
        emitted_solo = False
        for nb in sorted(neighbors):
            if nb in used and nb not in junction_features:
                continue
            chain = _walk_one_direction(start, nb)
            if len(chain) == 1:
                # Walk went nowhere (neighbor already used); emit at most once
                if emitted_solo:
                    continue
                emitted_solo = True
            reaches.append(chain)
            branches_started += 1
        if branches_started > 0:
            used.add(start)
        elif start not in used:
            used.add(start)
            reaches.append([start])

    # Catch remaining isolated loops (all degree-2, no junction)
    for fidx in range(n):
        if fidx not in used:
            used.add(fidx)
            chain = [fidx]
            for nb in adj.get(fidx, set()):
                prev_inner, cur_inner = fidx, nb
                while cur_inner not in used:
                    used.add(cur_inner)
                    chain.append(cur_inner)
                    nbs = adj.get(cur_inner, set()) - {prev_inner}
                    if len(nbs) != 1:
                        break
                    prev_inner, cur_inner = cur_inner, nbs.pop()
            reaches.append(chain)

    # --- Build output GeoDataFrame ---
    rows = []
    for rid, chain in enumerate(reaches):
        # Track which features contribute valid geometry
        valid_features = []
        geoms = []
        for fidx in chain:
            g = gdf.geometry.iloc[fidx]
            if g is None or g.is_empty:
                continue
            if g.geom_type == "MultiLineString":
                geoms.extend(g.geoms)
            else:
                geoms.append(g)
            valid_features.append(fidx)
        if not geoms:
            continue
        merged = linemerge(geoms)
        if isinstance(merged, MultiLineString):
            # Merge failed: keep only the longest part and recount metadata
            # to reflect only the features whose geometry is retained.
            longest = max(merged.geoms, key=lambda g: g.length)
            # Identify which input features intersect the retained geometry
            contributing = [
                fidx
                for fidx in valid_features
                if gdf.geometry.iloc[fidx].intersects(longest.buffer(1.0))
            ]
            if not contributing:
                contributing = valid_features
            merged = longest
            n_in_reach = len(contributing)
            n_seeded = sum(1 for f in contributing if seeded[f])
            LOGGER.debug(
                "Reach %d: linemerge partial — kept %.0f m of %.0f m (%d/%d features)",
                rid,
                longest.length,
                sum(g.length for g in geoms),
                n_in_reach,
                len(valid_features),
            )
        else:
            n_in_reach = len(valid_features)
            n_seeded = sum(1 for f in valid_features if seeded[f])
        cid = int(comp_id[chain[0]])
        sf = n_seeded / n_in_reach if n_in_reach else 0.0
        rows.append(
            {
                "reach_id": rid,
                "component_id": cid,
                "n_flowlines": n_in_reach,
                "is_simple_chain": not any(f in junction_features for f in chain)
                or len(chain) == 1,
                "seeded_fraction": float(sf),
                "geometry": merged,
            }
        )

    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)
    LOGGER.info(
        "Reach decomposition: %d reaches from %d flowlines "
        "(%d components, %d junction clusters)",
        len(result),
        n,
        current_comp,
        len(junction_clusters),
    )
    return result


# ---------------------------------------------------------------------------
# Phase 5: Stationization and snapping
# ---------------------------------------------------------------------------


def stationize_reach(
    reach_geom: LineString,
    spacing_m: float,
) -> gpd.GeoDataFrame:
    """Place evenly-spaced station points along a reach geometry.

    Returns a GeoDataFrame with columns:
        station_id, s_m (distance along reach), geometry (Point)
    """
    total = reach_geom.length
    if total == 0:
        return gpd.GeoDataFrame(columns=["station_id", "s_m", "geometry"])
    distances = np.arange(0, total + spacing_m * 0.5, spacing_m)
    if distances[-1] < total:
        distances = np.append(distances, total)
    rows = []
    for i, d in enumerate(distances):
        pt = reach_geom.interpolate(d)
        rows.append({"station_id": i, "s_m": float(d), "geometry": pt})
    return gpd.GeoDataFrame(rows, geometry="geometry")


def _build_dem_interpolator(dem_da: xr.DataArray) -> RegularGridInterpolator:
    """Build a fast bilinear interpolator from a DEM DataArray."""
    y_vals = dem_da.y.values.astype(np.float64)
    x_vals = dem_da.x.values.astype(np.float64)
    # RegularGridInterpolator needs monotonically increasing axes
    y_flip = y_vals[0] > y_vals[-1]
    data = np.asarray(dem_da.values, dtype=np.float64)
    if y_flip:
        y_vals = y_vals[::-1]
        data = data[::-1, :]
    return RegularGridInterpolator(
        (y_vals, x_vals),
        data,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def _sample_raster_at_points(
    interp: RegularGridInterpolator, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """Sample a raster interpolator at (x, y) coordinates. Returns 1-D array."""
    pts = np.column_stack([ys, xs])  # RegularGridInterpolator expects (y, x)
    return interp(pts)


def _batch_corridor_support(
    centers_xy: np.ndarray,
    normals: np.ndarray,
    tangents: np.ndarray,
    water_interp: RegularGridInterpolator,
    corridor_half_width_m: float,
    corridor_half_length_m: float,
    step_m: float,
    mode: str,
    water_hit_threshold: float = 0.0,
    return_raw_mean: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Vectorized corridor water support for a batch of candidate positions.

    Parameters
    ----------
    centers_xy : (N, 2) array of candidate center coordinates
    normals : (N, 2) array of unit normal vectors (one per candidate)
    tangents : (N, 2) array of unit tangent vectors (one per candidate)
    return_raw_mean : if True, also return mean of raw (un-thresholded) values

    Returns
    -------
    water_support_frac : (N,) array of corridor water hit fractions
    raw_mean : (N,) array of mean raw values (only if return_raw_mean=True)
    """
    n = len(centers_xy)
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return (empty, empty) if return_raw_mean else empty

    # Build corridor offset grid (shared across all candidates)
    across = np.arange(
        -corridor_half_width_m, corridor_half_width_m + step_m * 0.5, step_m
    )
    along = np.arange(
        -corridor_half_length_m, corridor_half_length_m + step_m * 0.5, step_m
    )
    da, dc = np.meshgrid(along, across, indexing="ij")
    da_flat = da.ravel()  # (K,)
    dc_flat = dc.ravel()  # (K,)
    k = len(da_flat)

    # Broadcast offsets to all candidates: (N, K, 2)
    offset_xy = (
        da_flat[np.newaxis, :, np.newaxis] * tangents[:, np.newaxis, :]
        + dc_flat[np.newaxis, :, np.newaxis] * normals[:, np.newaxis, :]
    )
    pts_xy = centers_xy[:, np.newaxis, :] + offset_xy

    # Flatten for single interpolation call
    flat_x = pts_xy[:, :, 0].ravel()
    flat_y = pts_xy[:, :, 1].ravel()
    flat_vals = _sample_raster_at_points(water_interp, flat_x, flat_y)
    vals = flat_vals.reshape(n, k)

    # Compute support fraction per candidate
    if mode == "binary_mask":
        hits = vals >= 0.5
    else:
        hits = vals > water_hit_threshold
    hits[np.isnan(vals)] = False
    support_frac = np.mean(hits, axis=1)

    if return_raw_mean:
        safe_vals = np.where(np.isnan(vals), 0.0, vals)
        raw_mean = np.mean(safe_vals, axis=1)
        return support_frac, raw_mean
    return support_frac


def snap_reach_to_thalweg(
    reach_geom: LineString,
    dem_da: xr.DataArray,
    ndwi_da: xr.DataArray,
    config: HandilyConfig,
) -> SnappedReach:
    """Snap a reach to the DEM thalweg via dynamic-programming offset optimization.

    At each station along the prior reach, a search line is cast normal to the
    local tangent.  Candidate offsets are evaluated with explicit cost weights.

    Cost terms (unary, per candidate):
        - DEM elevation (normalized) — prefer low ground
        - Water support (corridor-based) — prefer water corridors
        - |offset| / max_offset — prefer staying near the prior

    Transition cost:
        - |offset_change| between adjacent stations — prefer smooth paths

    Returns a SnappedReach with station-level corridor support metrics.
    """
    station_spacing_m = config.rem_frame_station_spacing_m
    snap_max_offset_m = config.rem_snap_max_offset_m
    snap_search_spacing_m = config.rem_snap_search_spacing_m
    w_elev = config.rem_snap_w_elev
    w_water = config.rem_snap_w_water
    w_prior = config.rem_snap_w_prior
    w_transition = config.rem_snap_w_transition
    corridor_hw = config.rem_support_corridor_half_width_m
    corridor_hl = config.rem_support_corridor_half_length_m
    support_mode = config.rem_water_support_mode
    # For continuous NDWI, use the same threshold as seeding
    support_kwargs = {}
    if support_mode == "continuous_index":
        support_kwargs["water_hit_threshold"] = config.ndwi_threshold

    stations = stationize_reach(reach_geom, station_spacing_m)
    n_stations = len(stations)
    if n_stations < 2:
        return SnappedReach(
            reach_id=-1,
            prior_geom=reach_geom,
            snapped_geom=reach_geom,
            stations=stations,
            confidence=0.0,
        )

    # Build interpolators
    dem_interp = _build_dem_interpolator(dem_da)
    ndwi_interp = _build_dem_interpolator(ndwi_da)

    # Candidate offsets
    offsets = np.arange(
        -snap_max_offset_m,
        snap_max_offset_m + snap_search_spacing_m * 0.5,
        snap_search_spacing_m,
    )
    n_offsets = len(offsets)

    # Compute tangent and normal at each station
    s_vals = stations["s_m"].values
    coords = np.array([(g.x, g.y) for g in stations.geometry])

    tangent = np.zeros((n_stations, 2), dtype=np.float64)
    for i in range(n_stations):
        if i == 0:
            tangent[i] = coords[1] - coords[0]
        elif i == n_stations - 1:
            tangent[i] = coords[-1] - coords[-2]
        else:
            tangent[i] = coords[i + 1] - coords[i - 1]
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangent /= norms
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    # Build candidate positions: (n_stations, n_offsets, 2)
    cand_xy = (
        coords[:, np.newaxis, :]
        + offsets[np.newaxis, :, np.newaxis] * normal[:, np.newaxis, :]
    )

    # Sample DEM at all candidates
    cand_x = cand_xy[:, :, 0].ravel()
    cand_y = cand_xy[:, :, 1].ravel()
    dem_vals = _sample_raster_at_points(dem_interp, cand_x, cand_y).reshape(
        n_stations, n_offsets
    )

    # --- Water cost: vectorized corridor support for all candidates ---
    # Flatten (n_stations, n_offsets) → (N,) with matching normals/tangents
    n_cand = n_stations * n_offsets
    flat_centers = cand_xy.reshape(n_cand, 2)
    flat_normals = np.repeat(normal, n_offsets, axis=0)  # (N, 2)
    flat_tangents = np.repeat(tangent, n_offsets, axis=0)  # (N, 2)
    hit_threshold = support_kwargs.get("water_hit_threshold", 0.0)
    flat_frac = _batch_corridor_support(
        flat_centers,
        flat_normals,
        flat_tangents,
        ndwi_interp,
        corridor_hw,
        corridor_hl,
        step_m=snap_search_spacing_m,
        mode=support_mode,
        water_hit_threshold=hit_threshold,
    )
    water_cost = 1.0 - flat_frac.reshape(n_stations, n_offsets)

    # --- Build cost matrix ---
    dem_min = np.nanmin(dem_vals, axis=1, keepdims=True)
    dem_range = np.nanmax(dem_vals, axis=1, keepdims=True) - dem_min
    dem_range[dem_range == 0] = 1.0
    elev_cost = (dem_vals - dem_min) / dem_range

    abs_offsets = np.abs(offsets)[np.newaxis, :]
    prior_cost = abs_offsets / snap_max_offset_m

    unary = w_elev * elev_cost + w_water * water_cost + w_prior * prior_cost
    unary[np.isnan(dem_vals)] = 10.0

    # --- DP forward pass ---
    dp_cost = np.full((n_stations, n_offsets), np.inf, dtype=np.float64)
    dp_prev = np.full((n_stations, n_offsets), -1, dtype=np.intp)
    dp_cost[0] = unary[0]

    offset_diff = np.abs(offsets[:, np.newaxis] - offsets[np.newaxis, :])
    transition = w_transition * (offset_diff / snap_max_offset_m)

    for s in range(1, n_stations):
        total = dp_cost[s - 1, :, np.newaxis] + transition
        best_prev = np.argmin(total, axis=0)
        dp_cost[s] = unary[s] + total[best_prev, np.arange(n_offsets)]
        dp_prev[s] = best_prev

    # --- DP backtrack ---
    chosen = np.empty(n_stations, dtype=np.intp)
    chosen[-1] = int(np.argmin(dp_cost[-1]))
    for s in range(n_stations - 2, -1, -1):
        chosen[s] = dp_prev[s + 1, chosen[s + 1]]

    # --- Corridor support at snapped positions (vectorized) ---
    snap_offsets = offsets[chosen]
    snap_xy = coords + snap_offsets[:, np.newaxis] * normal
    snap_elev = np.array([dem_vals[s, chosen[s]] for s in range(n_stations)])

    snap_frac, snap_mean = _batch_corridor_support(
        snap_xy,
        normal,
        tangent,
        ndwi_interp,
        corridor_hw,
        corridor_hl,
        step_m=snap_search_spacing_m,
        mode=support_mode,
        water_hit_threshold=hit_threshold,
        return_raw_mean=True,
    )
    snap_hit = snap_frac > 0

    station_rows = []
    for s in range(n_stations):
        station_rows.append(
            {
                "station_id": s,
                "s_m": float(s_vals[s]),
                "x_prior": float(coords[s, 0]),
                "y_prior": float(coords[s, 1]),
                "x_snap": float(snap_xy[s, 0]),
                "y_snap": float(snap_xy[s, 1]),
                "snap_offset_m": float(snap_offsets[s]),
                "thalweg_elev_m": float(snap_elev[s]),
                "water_support_frac": float(snap_frac[s]),
                "water_support_mean": float(snap_mean[s]),
                "water_hit": bool(snap_hit[s]),
                "geometry": Point(snap_xy[s, 0], snap_xy[s, 1]),
            }
        )

    station_gdf = gpd.GeoDataFrame(
        station_rows, geometry="geometry", crs=dem_da.rio.crs
    )

    snapped_line = LineString(snap_xy.tolist())
    mean_offset = float(np.mean(np.abs(snap_offsets)))
    water_hit_frac = float(station_gdf["water_hit"].mean())

    LOGGER.info(
        "Snapped reach: %d stations, mean_offset=%.1f m, water_hit=%.0f%%",
        n_stations,
        mean_offset,
        water_hit_frac * 100,
    )

    return SnappedReach(
        reach_id=-1,  # caller assigns
        prior_geom=reach_geom,
        snapped_geom=snapped_line,
        stations=station_gdf,
        confidence=water_hit_frac,
        metrics=_compute_reach_metrics(station_gdf, snap_max_offset_m),
    )


def _compute_reach_metrics(
    station_gdf: gpd.GeoDataFrame,
    snap_max_offset_m: float,
    seeded_fraction: float = 0.0,
) -> ReachMetrics:
    """Compute interpretable metrics from a snapped station GeoDataFrame."""
    n = len(station_gdf)
    if n == 0:
        return ReachMetrics(
            station_water_hit_fraction=0.0,
            station_water_support_mean=0.0,
            mean_snap_offset_m=0.0,
            max_snap_offset_m=0.0,
            max_consecutive_no_water_m=0.0,
            n_stations=0,
            n_supported_stations=0,
            seeded_fraction=seeded_fraction,
        )

    hits = station_gdf["water_hit"].values.astype(bool)
    support_mean = station_gdf["water_support_mean"].values
    abs_offsets = station_gdf["snap_offset_m"].abs().values
    s_m = station_gdf["s_m"].values

    # Max consecutive gap without water hit
    max_gap = 0.0
    gap_start = 0.0
    in_gap = True
    for i in range(n):
        if hits[i]:
            if in_gap:
                gap_len = s_m[i] - gap_start
                if gap_len > max_gap:
                    max_gap = gap_len
            in_gap = False
        else:
            if not in_gap:
                gap_start = s_m[i]
                in_gap = True
    if in_gap and n > 0:
        gap_len = s_m[-1] - gap_start
        if gap_len > max_gap:
            max_gap = gap_len

    return ReachMetrics(
        station_water_hit_fraction=float(hits.mean()),
        station_water_support_mean=float(support_mean.mean()),
        mean_snap_offset_m=float(abs_offsets.mean()),
        max_snap_offset_m=float(abs_offsets.max()),
        max_consecutive_no_water_m=float(max_gap),
        n_stations=n,
        n_supported_stations=int(hits.sum()),
        seeded_fraction=seeded_fraction,
    )


def evaluate_reach_acceptance(
    snapped: SnappedReach,
    config: HandilyConfig,
) -> str | None:
    """Check whether a snapped reach meets acceptance thresholds.

    Returns None if accepted, or a short reason string if rejected.
    """
    m = snapped.metrics
    if m is None:
        return "no metrics"

    if m.station_water_hit_fraction < config.rem_min_station_water_hit_fraction:
        return (
            f"low water_hit {m.station_water_hit_fraction:.2f} "
            f"< {config.rem_min_station_water_hit_fraction:.2f}"
        )

    if m.max_consecutive_no_water_m > config.rem_max_consecutive_no_water_m:
        return (
            f"water gap {m.max_consecutive_no_water_m:.0f} m "
            f"> {config.rem_max_consecutive_no_water_m:.0f} m"
        )

    if m.mean_snap_offset_m > config.rem_max_mean_snap_offset_m:
        return (
            f"mean offset {m.mean_snap_offset_m:.1f} m "
            f"> {config.rem_max_mean_snap_offset_m:.0f} m"
        )

    return None


# ---------------------------------------------------------------------------
# Phase 6: Frame smoothing and cross-sections
# ---------------------------------------------------------------------------


def build_smoothed_frame(
    snapped_reach: SnappedReach,
    smoothing_m: float,
) -> FrameReach:
    """Build a smoothed curvilinear frame from a snapped thalweg.

    The frame is used only for computing stable transect normals — the
    snapped thalweg coordinates are retained for elevation sampling.

    Smooths x(s) and y(s) independently with a Gaussian kernel whose sigma
    is *smoothing_m / station_spacing*.
    """
    st = snapped_reach.stations
    if len(st) < 3:
        return FrameReach(
            reach_id=snapped_reach.reach_id,
            frame_geom=snapped_reach.snapped_geom,
            frame_stations=st.copy(),
            snapped_stations=st,
        )

    xs = st["x_snap"].values.astype(np.float64)
    ys = st["y_snap"].values.astype(np.float64)

    # Infer station spacing from s_m
    s_vals = st["s_m"].values
    spacing = np.median(np.diff(s_vals))
    if spacing <= 0:
        spacing = 1.0
    sigma_px = smoothing_m / spacing

    xs_smooth = ndi.gaussian_filter1d(xs, sigma=sigma_px, mode="nearest")
    ys_smooth = ndi.gaussian_filter1d(ys, sigma=sigma_px, mode="nearest")

    frame_rows = []
    for i in range(len(st)):
        frame_rows.append(
            {
                "station_id": i,
                "s_m": float(s_vals[i]),
                "x_frame": float(xs_smooth[i]),
                "y_frame": float(ys_smooth[i]),
                "geometry": Point(xs_smooth[i], ys_smooth[i]),
            }
        )
    frame_gdf = gpd.GeoDataFrame(frame_rows, geometry="geometry")

    frame_line = LineString(np.column_stack([xs_smooth, ys_smooth]).tolist())

    return FrameReach(
        reach_id=snapped_reach.reach_id,
        frame_geom=frame_line,
        frame_stations=frame_gdf,
        snapped_stations=st,
    )


def _find_support_limit(
    profile: np.ndarray, ridge_prominence_m: float, descend_stop_m: float
) -> int:
    """Find the index along a 1-D elevation profile where support ends.

    Walk outward from the thalweg (index 0). Track the running maximum.
    When the running max exceeds the base by *ridge_prominence_m*, start
    watching for a descent of *descend_stop_m* from that peak. Return the
    index of the descent stop, or the full length if none found.
    """
    n = len(profile)
    if n == 0:
        return 0
    base = profile[0]
    running_max = base
    ridge_found = False
    peak_val = base

    for i in range(1, n):
        v = profile[i]
        if np.isnan(v):
            return i
        if v > running_max:
            running_max = v
        # Check for ridge
        if not ridge_found and (running_max - base) >= ridge_prominence_m:
            ridge_found = True
            peak_val = running_max
        # After ridge, check for descent
        if ridge_found and (peak_val - v) >= descend_stop_m:
            return i
        if ridge_found and v > peak_val:
            peak_val = v

    return n


def sample_cross_sections(
    frame_reach: FrameReach,
    dem_da: xr.DataArray,
    ndwi_da: xr.DataArray,
    cross_max_dist_m: float,
    cross_step_m: float,
    ridge_prominence_m: float,
    descend_stop_m: float,
    zero_mode: str,
) -> CrossSectionSet:
    """Sample cross-sections normal to the frame with ridge-stop logic.

    For each station, casts left/right rays along the frame normal.  Support
    terminates when a crest rises more than *ridge_prominence_m* above the
    section base and then descends by *descend_stop_m*, or at *cross_max_dist_m*.

    Parameters
    ----------
    zero_mode : str
        ``"thalweg"`` — base elevation from snapped thalweg station
        ``"section_min"`` — base elevation from accepted section minimum
    """
    fs = frame_reach.frame_stations
    ss = frame_reach.snapped_stations
    n_stations = len(fs)

    dem_interp = _build_dem_interpolator(dem_da)

    # Compute frame tangent and normals
    frame_xy = np.column_stack([fs["x_frame"].values, fs["y_frame"].values]).astype(
        np.float64
    )

    tangent = np.zeros_like(frame_xy)
    for i in range(n_stations):
        if i == 0:
            tangent[i] = frame_xy[1] - frame_xy[0]
        elif i == n_stations - 1:
            tangent[i] = frame_xy[-1] - frame_xy[-2]
        else:
            tangent[i] = frame_xy[i + 1] - frame_xy[i - 1]
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangent /= norms
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    n_side_pts = int(cross_max_dist_m / cross_step_m) + 1
    side_dists = np.arange(0, n_side_pts) * cross_step_m

    section_rows = []
    support_rows = []

    for s in range(n_stations):
        base_xy = np.array(
            [
                float(ss.iloc[s]["x_snap"]),
                float(ss.iloc[s]["y_snap"]),
            ]
        )
        thalweg_elev = float(ss.iloc[s]["thalweg_elev_m"])

        # Cast left (negative normal) and right (positive normal)
        limits = {}
        profiles = {}
        for side, sign in [("left", -1.0), ("right", 1.0)]:
            ray_xy = (
                base_xy[np.newaxis, :]
                + (sign * side_dists[:, np.newaxis]) * normal[s][np.newaxis, :]
            )
            ray_elev = _sample_raster_at_points(dem_interp, ray_xy[:, 0], ray_xy[:, 1])
            profiles[side] = ray_elev
            limit_idx = _find_support_limit(
                ray_elev, ridge_prominence_m, descend_stop_m
            )
            limits[side] = limit_idx

        left_dist = float(side_dists[min(limits["left"], n_side_pts - 1)])
        right_dist = float(side_dists[min(limits["right"], n_side_pts - 1)])

        # Section base elevation
        if zero_mode == "section_min":
            all_elev = np.concatenate(
                [
                    profiles["left"][: limits["left"]],
                    profiles["right"][: limits["right"]],
                ]
            )
            valid = all_elev[np.isfinite(all_elev)]
            base_elev = float(np.min(valid)) if len(valid) > 0 else thalweg_elev
        else:
            base_elev = thalweg_elev

        # Section line geometry: left end → base → right end
        left_end = base_xy - left_dist * normal[s]
        right_end = base_xy + right_dist * normal[s]
        section_line = LineString(
            [left_end.tolist(), base_xy.tolist(), right_end.tolist()]
        )

        section_rows.append(
            {
                "station_id": s,
                "s_m": float(fs.iloc[s]["s_m"]),
                "base_elev_m": base_elev,
                "thalweg_elev_m": thalweg_elev,
                "left_dist_m": left_dist,
                "right_dist_m": right_dist,
                "total_width_m": left_dist + right_dist,
                "geometry": section_line,
            }
        )

        # Support polygon: rectangle from this section to next (built later)
        support_rows.append(
            {
                "station_id": s,
                "left_dist_m": left_dist,
                "right_dist_m": right_dist,
                "geometry": section_line.buffer(cross_step_m * 0.5),
            }
        )

    sections_gdf = gpd.GeoDataFrame(
        section_rows, geometry="geometry", crs=dem_da.rio.crs
    )
    support_gdf = gpd.GeoDataFrame(
        support_rows, geometry="geometry", crs=dem_da.rio.crs
    )

    median_width = float(sections_gdf["total_width_m"].median())
    LOGGER.info(
        "Cross-sections: %d stations, median_width=%.0f m, zero_mode=%s",
        n_stations,
        median_width,
        zero_mode,
    )

    return CrossSectionSet(sections=sections_gdf, support_polygons=support_gdf)


# ---------------------------------------------------------------------------
# Phases 7-8: Water-surface rasterization
# ---------------------------------------------------------------------------


def rasterize_anisotropic_water_surface(
    frame_reaches: list[FrameReach],
    cross_sections: CrossSectionSet,
    dem_da: xr.DataArray,
    zero_mode: str,
    min_confidence: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Rasterize the anisotropic water surface from frame/section geometry.

    For each pair of adjacent sections along a reach, builds a strip polygon
    from the section endpoints.  Pixels inside the strip are projected onto the
    local along-reach axis and the base elevation is linearly interpolated
    between the two bracketing stations.

    Returns (water_surface_da, support_confidence_da).
    """
    from rasterio import features as rio_features
    from shapely.geometry import Polygon

    shape = dem_da.shape
    transform = dem_da.rio.transform()
    y_coords = dem_da.y.values.astype(np.float64)
    x_coords = dem_da.x.values.astype(np.float64)
    res_x = abs(float(transform.a))
    res_y = abs(float(transform.e))

    ws = np.full(shape, np.nan, dtype=np.float64)
    conf = np.full(shape, np.nan, dtype=np.float64)

    secs = cross_sections.sections
    if secs.empty:
        ws_da = xr.DataArray(
            ws.astype(np.float32),
            dims=dem_da.dims,
            coords=dem_da.coords,
            name="water_surface",
        )
        conf_da = xr.DataArray(
            conf.astype(np.float32),
            dims=dem_da.dims,
            coords=dem_da.coords,
            name="confidence",
        )
        return ws_da.rio.write_crs(dem_da.rio.crs), conf_da.rio.write_crs(
            dem_da.rio.crs
        )

    n_strips = 0

    # Process each reach's sections in sequence
    for frame in frame_reaches:
        reach_secs = (
            secs[secs["reach_id"] == frame.reach_id]
            .sort_values("s_m")
            .reset_index(drop=True)
        )

        if len(reach_secs) < 2:
            continue

        for i in range(len(reach_secs) - 1):
            sec_a = reach_secs.iloc[i]
            sec_b = reach_secs.iloc[i + 1]

            # Section endpoints: each section geometry is left_end → base → right_end
            coords_a = np.array(sec_a.geometry.coords)
            coords_b = np.array(sec_b.geometry.coords)
            left_a, right_a = coords_a[0], coords_a[-1]
            left_b, right_b = coords_b[0], coords_b[-1]

            # Build strip polygon: left_a → right_a → right_b → left_b → close
            strip_coords = [
                tuple(left_a[:2]),
                tuple(right_a[:2]),
                tuple(right_b[:2]),
                tuple(left_b[:2]),
                tuple(left_a[:2]),
            ]
            strip = Polygon(strip_coords)
            if not strip.is_valid or strip.area == 0:
                continue

            # Compute bounding box in pixel space — only rasterize the subwindow
            bx = np.array([c[0] for c in strip_coords])
            by = np.array([c[1] for c in strip_coords])
            pad = 2  # pixel padding
            col_min = max(0, int(np.searchsorted(x_coords, bx.min()) - pad))
            col_max = min(shape[1], int(np.searchsorted(x_coords, bx.max()) + pad))
            if y_coords[0] > y_coords[-1]:  # y decreasing (typical)
                row_min = max(0, int(np.searchsorted(-y_coords, -by.max()) - pad))
                row_max = min(
                    shape[0], int(np.searchsorted(-y_coords, -by.min()) + pad)
                )
            else:
                row_min = max(0, int(np.searchsorted(y_coords, by.min()) - pad))
                row_max = min(shape[0], int(np.searchsorted(y_coords, by.max()) + pad))

            sub_h = row_max - row_min
            sub_w = col_max - col_min
            if sub_h <= 0 or sub_w <= 0:
                continue

            # Build sub-window transform
            sub_x_origin = x_coords[col_min] - res_x * 0.5
            sub_y_origin = y_coords[row_min] + res_y * 0.5  # top-left
            from rasterio.transform import from_origin

            sub_transform = from_origin(sub_x_origin, sub_y_origin, res_x, res_y)

            strip_mask = rio_features.rasterize(
                [(strip, 1)],
                out_shape=(sub_h, sub_w),
                transform=sub_transform,
                fill=0,
                all_touched=True,
                dtype="uint8",
            )
            local_ys, local_xs = np.where(strip_mask > 0)
            if len(local_ys) == 0:
                continue

            # Map back to full-grid indices
            ys_idx = local_ys + row_min
            xs_idx = local_xs + col_min

            # Pixel coordinates
            px_x = x_coords[xs_idx]
            px_y = y_coords[ys_idx]

            # Project pixels onto the along-reach axis between stations a and b
            base_a = np.array(
                [float(sec_a.geometry.coords[1][0]), float(sec_a.geometry.coords[1][1])]
            )
            base_b = np.array(
                [float(sec_b.geometry.coords[1][0]), float(sec_b.geometry.coords[1][1])]
            )
            axis = base_b - base_a
            axis_len = np.linalg.norm(axis)
            if axis_len == 0:
                continue
            axis_unit = axis / axis_len

            # Fractional position along the strip [0, 1]
            px_vec = np.column_stack([px_x - base_a[0], px_y - base_a[1]])
            t = np.clip(px_vec @ axis_unit / axis_len, 0, 1)

            # Interpolate base elevation
            elev_a = float(sec_a["base_elev_m"])
            elev_b = float(sec_b["base_elev_m"])
            interp_elev = elev_a + t * (elev_b - elev_a)

            # Write — only where not already written
            mask = np.isnan(ws[ys_idx, xs_idx])
            ws[ys_idx[mask], xs_idx[mask]] = interp_elev[mask]
            conf[ys_idx[mask], xs_idx[mask]] = 1.0
            n_strips += 1

    LOGGER.info("Rasterized %d section strips", n_strips)

    ws_da = xr.DataArray(
        ws.astype(np.float32),
        dims=dem_da.dims,
        coords=dem_da.coords,
        name="water_surface",
    )
    conf_da = xr.DataArray(
        conf.astype(np.float32),
        dims=dem_da.dims,
        coords=dem_da.coords,
        name="confidence",
    )
    return (
        ws_da.rio.write_crs(dem_da.rio.crs, inplace=False),
        conf_da.rio.write_crs(dem_da.rio.crs, inplace=False),
    )


def _dedup_overlapping_results(
    results: list,
    metrics: dict,
    buf_m: float = 10.0,
    overlap_frac: float = 0.5,
) -> list:
    """Drop shorter reaches whose snapped geometry is mostly covered by a longer one.

    Two reaches that follow the same physical channel (from different NHD features)
    produce overlapping snapped geometries.  Keep the longer reach, drop the shorter
    if more than *overlap_frac* of its length falls within *buf_m* of the longer.
    """
    # Collect (index, snapped_geom_length) for non-None results
    valid = []
    for i, r in enumerate(results):
        if r is not None:
            valid.append((i, r[0].snapped_geom.length))
    if len(valid) < 2:
        return results

    # Sort longest first so longer reaches are checked as "keepers" first
    valid.sort(key=lambda x: x[1], reverse=True)

    drop = set()
    kept_geoms = []  # (index, buffered_geom)

    for idx, length in valid:
        geom = results[idx][0].snapped_geom
        # Check if this reach is mostly covered by an already-kept longer reach
        dominated = False
        for kept_idx, kept_buf in kept_geoms:
            overlap = geom.intersection(kept_buf).length
            if length > 0 and overlap / length > overlap_frac:
                drop.add(idx)
                dominated = True
                rid = results[idx][0].reach_id
                kept_rid = results[kept_idx][0].reach_id
                LOGGER.info(
                    "Dedup: dropping reach %d (%.0f m, %.0f%% covered by reach %d)",
                    rid,
                    length,
                    100 * overlap / length,
                    kept_rid,
                )
                break
        if not dominated:
            kept_geoms.append((idx, geom.buffer(buf_m)))

    n_deduped = len(drop)
    if n_deduped:
        metrics["n_reaches_deduped"] = n_deduped
    filtered = [None if i in drop else r for i, r in enumerate(results)]
    return filtered


# ---------------------------------------------------------------------------
# Phase 9: End-to-end entry point
# ---------------------------------------------------------------------------


def compute_rem_anisotropic_frame(
    dem_da: xr.DataArray,
    ndwi_da: xr.DataArray,
    confirmed_flowlines: gpd.GeoDataFrame,
    config: HandilyConfig,
) -> AnisotropicREMResult:
    """Compute anisotropic-frame REM from confirmed flowlines.

    Orchestrates the full pipeline: reach decomposition → snapping →
    frame smoothing → cross-sections → water-surface rasterization → REM.
    """
    # Reproject NDWI to match DEM grid if needed
    if str(ndwi_da.rio.crs) != str(dem_da.rio.crs) or ndwi_da.shape != dem_da.shape:
        LOGGER.info("Reprojecting NDWI from %s to %s", ndwi_da.rio.crs, dem_da.rio.crs)
        ndwi_da = ndwi_da.rio.reproject_match(dem_da)

    # Phase 4: reach decomposition
    reaches = split_confirmed_flowlines_into_reaches(confirmed_flowlines)

    # Filter reaches by seeded fraction
    n_all = len(reaches)
    min_seeded = config.rem_min_seeded_fraction
    eligible = reaches[reaches["seeded_fraction"] >= min_seeded].copy()
    n_skipped_seed = n_all - len(eligible)
    if n_skipped_seed:
        LOGGER.info(
            "Reach eligibility: %d/%d reaches have seeded_fraction >= %.2f",
            len(eligible),
            n_all,
            min_seeded,
        )
    LOGGER.info("Processing %d eligible reaches", len(eligible))

    def _process_one_reach(row):
        """Snap + smooth + cross-section for a single reach. Thread-safe."""
        rid = int(row["reach_id"])
        geom = row["geometry"]
        if (
            geom is None
            or geom.is_empty
            or geom.length < config.rem_min_support_width_m
        ):
            LOGGER.info(
                "Skipping reach %d: too short (%.0f m)",
                rid,
                geom.length if geom else 0,
            )
            return None

        snapped = snap_reach_to_thalweg(geom, dem_da, ndwi_da, config)
        snapped.reach_id = rid
        if snapped.metrics is not None:
            snapped.metrics.seeded_fraction = float(row.get("seeded_fraction", 0.0))

        if snapped.metrics is None:
            LOGGER.info("Skipping reach %d: no metrics (too few stations)", rid)
            return None

        reject_reason = evaluate_reach_acceptance(snapped, config)
        if reject_reason:
            LOGGER.info(
                "Reach %d: %s (keeping — propagation-confirmed)", rid, reject_reason
            )

        frame = build_smoothed_frame(snapped, smoothing_m=config.rem_frame_smoothing_m)
        frame.reach_id = rid

        xsec = sample_cross_sections(
            frame,
            dem_da,
            ndwi_da,
            cross_max_dist_m=config.rem_cross_max_dist_m,
            cross_step_m=config.rem_cross_step_m,
            ridge_prominence_m=config.rem_cross_ridge_prominence_m,
            descend_stop_m=config.rem_cross_descend_stop_m,
            zero_mode=config.rem_zero_mode,
        )
        sec_df = xsec.sections.copy()
        sec_df["reach_id"] = rid
        sup_df = xsec.support_polygons.copy()
        sup_df["reach_id"] = rid
        return (snapped, frame, sec_df, sup_df, reject_reason)

    # Run reaches in parallel (ThreadPool — numpy/scipy release the GIL)
    max_workers = getattr(config, "rem_max_workers", None)
    reach_rows = [eligible.iloc[i] for i in range(len(eligible))]

    snapped_reaches = []
    frame_reaches = []
    all_sections = []
    all_support = []
    metrics = {
        "n_reaches_total": n_all,
        "n_reaches_eligible": len(eligible),
        "n_reaches_skipped_no_seed": n_skipped_seed,
        "n_reaches_processed": 0,
        "n_reaches_warned": 0,
        "n_reaches_skipped": 0,
        "snap_offsets": [],
    }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_process_one_reach, reach_rows))

    # Deduplicate overlapping snapped reaches (keep the longer one)
    results = _dedup_overlapping_results(results, metrics)

    for result in results:
        if result is None:
            metrics["n_reaches_skipped"] += 1
            continue
        snapped, frame, sec_df, sup_df, reject_reason = result
        snapped_reaches.append((snapped, reject_reason))
        frame_reaches.append(frame)
        all_sections.append(sec_df)
        all_support.append(sup_df)
        metrics["n_reaches_processed"] += 1
        if reject_reason:
            metrics["n_reaches_warned"] += 1
        metrics["snap_offsets"].extend(snapped.stations["snap_offset_m"].abs().tolist())

    # Aggregate geometry artifacts
    if snapped_reaches:
        snapped_rows = []
        for s, reject_reason in snapped_reaches:
            row_dict = {
                "reach_id": s.reach_id,
                "confidence": s.confidence,
                "reject_reason": reject_reason,
                "geometry": s.snapped_geom,
            }
            if s.metrics is not None:
                row_dict.update(
                    {
                        "water_hit_frac": s.metrics.station_water_hit_fraction,
                        "water_support_mean": s.metrics.station_water_support_mean,
                        "mean_offset_m": s.metrics.mean_snap_offset_m,
                        "max_offset_m": s.metrics.max_snap_offset_m,
                        "max_gap_m": s.metrics.max_consecutive_no_water_m,
                        "n_stations": s.metrics.n_stations,
                        "n_supported": s.metrics.n_supported_stations,
                        "seeded_fraction": s.metrics.seeded_fraction,
                    }
                )
            snapped_rows.append(row_dict)
        snapped_gdf = gpd.GeoDataFrame(
            snapped_rows,
            geometry="geometry",
            crs=dem_da.rio.crs,
        )
        frame_gdf = gpd.GeoDataFrame(
            [{"reach_id": f.reach_id, "geometry": f.frame_geom} for f in frame_reaches],
            geometry="geometry",
            crs=dem_da.rio.crs,
        )
        sections_gdf = gpd.GeoDataFrame(
            pd.concat(all_sections, ignore_index=True),
            crs=dem_da.rio.crs,
        )
        support_gdf = gpd.GeoDataFrame(
            pd.concat(all_support, ignore_index=True),
            crs=dem_da.rio.crs,
        )
    else:
        snapped_gdf = gpd.GeoDataFrame(columns=["reach_id", "confidence", "geometry"])
        frame_gdf = gpd.GeoDataFrame(columns=["reach_id", "geometry"])
        sections_gdf = gpd.GeoDataFrame()
        support_gdf = gpd.GeoDataFrame()

    # Summary metrics
    if metrics["snap_offsets"]:
        offsets_arr = np.array(metrics["snap_offsets"])
        metrics["mean_snap_offset_m"] = float(np.mean(offsets_arr))
        metrics["max_snap_offset_m"] = float(np.max(offsets_arr))
    del metrics["snap_offsets"]

    LOGGER.info(
        "Anisotropic frame: %d/%d reaches processed (%d warned), %d skipped",
        metrics["n_reaches_processed"],
        metrics["n_reaches_total"],
        metrics["n_reaches_warned"],
        metrics["n_reaches_skipped"],
    )

    # Phase 8: rasterize water surface (placeholder — strips interpolation)
    if sections_gdf.empty:
        water_surface = xr.full_like(dem_da, np.nan, dtype=np.float32)
        rem = xr.full_like(dem_da, np.nan, dtype=np.float32)
    else:
        water_surface, _ = rasterize_anisotropic_water_surface(
            frame_reaches,
            CrossSectionSet(sections_gdf, support_gdf),
            dem_da,
            config.rem_zero_mode,
            config.rem_min_confidence,
        )
        dem_np = np.asarray(dem_da.values, dtype=np.float64)
        ws_np = np.asarray(water_surface.values, dtype=np.float64)
        rem_np = np.maximum(dem_np - ws_np, 0.0).astype(np.float32)
        rem_np[np.isnan(ws_np)] = np.nan
        rem = xr.DataArray(
            rem_np,
            dims=dem_da.dims,
            coords=dem_da.coords,
            name="REM",
        )
        rem = rem.rio.write_crs(dem_da.rio.crs, inplace=False)

    n_valid = int(np.isfinite(np.asarray(rem.values)).sum())
    n_total = rem.size
    metrics["rem_valid_pixel_fraction"] = n_valid / n_total if n_total else 0
    LOGGER.info(
        "REM valid pixels: %d / %d (%.1f%%)",
        n_valid,
        n_total,
        n_valid / n_total * 100 if n_total else 0,
    )

    return AnisotropicREMResult(
        rem_da=rem,
        water_surface_da=water_surface,
        confirmed_flowlines=confirmed_flowlines,
        snapped_flowlines=snapped_gdf,
        frame_flowlines=frame_gdf,
        cross_sections=sections_gdf,
        support_polygons=support_gdf,
        metrics=metrics,
    )
