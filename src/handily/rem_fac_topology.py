"""Topology-derived pin weights for FAC-based REM relaxation.

This module builds a directed graph from the dense FAC stream network, seeds
wet reaches from NDVI / hard support evidence, and propagates that wet
influence upstream with decay in network distance and elevation gain. The
result is a reach-scale pin-weight field that can be rasterized and used by the
water-surface relaxation solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.transform import Affine
from scipy.interpolate import RegularGridInterpolator


@dataclass
class FacTopologyResult:
    streams: gpd.GeoDataFrame
    downstream: dict[int, tuple[int, ...]]
    upstream: dict[int, tuple[int, ...]]


def _match_transform(match_da: xr.DataArray) -> Affine:
    x = np.asarray(match_da.x.values, dtype=np.float64)
    y = np.asarray(match_da.y.values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size < 2 or y.size < 2:
        raise ValueError("match_da must have 1D x/y coordinates with at least 2 cells")
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    if not np.isfinite(dx) or not np.isfinite(dy):
        raise ValueError("invalid match_da coordinate spacing")
    return Affine.translation(
        float(x[0] - dx / 2.0), float(y[0] - dy / 2.0)
    ) * Affine.scale(dx, dy)


def _build_raster_interp(da: xr.DataArray) -> RegularGridInterpolator:
    vals = np.asarray(da.values, dtype=np.float64)
    x = np.asarray(da.x.values, dtype=np.float64)
    y = np.asarray(da.y.values, dtype=np.float64)
    if y[0] > y[-1]:
        y = y[::-1]
        vals = vals[::-1, :]
    return RegularGridInterpolator((y, x), vals, bounds_error=False, fill_value=np.nan)


def _line_s_values(total_m: float, spacing_m: float) -> np.ndarray:
    if total_m <= spacing_m:
        return (
            np.array([0.0, total_m], dtype=np.float64)
            if total_m > 1e-6
            else np.array([0.0], dtype=np.float64)
        )
    vals = np.arange(0.0, total_m, spacing_m, dtype=np.float64)
    if vals.size == 0 or vals[-1] < total_m - 1e-6:
        vals = np.append(vals, total_m)
    return vals


def _quantize_node(x: float, y: float, precision: int) -> tuple[float, float]:
    return (round(float(x), precision), round(float(y), precision))


def _sample_line_values(
    line, interp: RegularGridInterpolator, spacing_m: float
) -> np.ndarray:
    s_vals = _line_s_values(float(line.length), float(spacing_m))
    pts = np.array(
        [[line.interpolate(float(s)).y, line.interpolate(float(s)).x] for s in s_vals],
        dtype=np.float64,
    )
    return np.asarray(interp(pts), dtype=np.float64)


def build_fac_topology(
    streams_gdf: gpd.GeoDataFrame,
    elev_da: xr.DataArray,
    *,
    node_precision: int = 3,
) -> FacTopologyResult:
    """Orient FAC reaches by elevation and derive upstream/downstream adjacency."""
    if "stream_id" not in streams_gdf.columns:
        raise ValueError("streams_gdf must contain stream_id")
    elev_interp = _build_raster_interp(elev_da)

    rows: list[dict] = []
    starts_at: dict[tuple[float, float], list[int]] = {}
    ends_at: dict[tuple[float, float], list[int]] = {}

    for row in streams_gdf.itertuples(index=False):
        geom = row.geometry
        coords = list(geom.coords)
        z0 = float(
            elev_interp(np.array([[coords[0][1], coords[0][0]]], dtype=np.float64))[0]
        )
        z1 = float(
            elev_interp(np.array([[coords[-1][1], coords[-1][0]]], dtype=np.float64))[0]
        )
        reverse = np.isfinite(z0) and np.isfinite(z1) and (z1 > z0)
        if reverse:
            geom = type(geom)(coords[::-1])
            z_up, z_down = z1, z0
        else:
            z_up, z_down = z0, z1
        oriented = list(geom.coords)
        up_node = _quantize_node(oriented[0][0], oriented[0][1], node_precision)
        down_node = _quantize_node(oriented[-1][0], oriented[-1][1], node_precision)
        stream_id = int(row.stream_id)
        starts_at.setdefault(up_node, []).append(stream_id)
        ends_at.setdefault(down_node, []).append(stream_id)
        length_m = float(getattr(row, "length_m", geom.length))
        relief_m = (
            float(max(z_up - z_down, 0.0))
            if np.isfinite(z_up) and np.isfinite(z_down)
            else np.nan
        )
        rows.append(
            {
                "stream_id": stream_id,
                "strahler": int(getattr(row, "strahler", 0))
                if getattr(row, "strahler", None) is not None
                else 0,
                "length_m": length_m,
                "up_elev_m": z_up,
                "down_elev_m": z_down,
                "relief_m": relief_m,
                "up_node_x": up_node[0],
                "up_node_y": up_node[1],
                "down_node_x": down_node[0],
                "down_node_y": down_node[1],
                "geometry": geom,
            }
        )

    downstream: dict[int, list[int]] = {int(r["stream_id"]): [] for r in rows}
    upstream: dict[int, list[int]] = {int(r["stream_id"]): [] for r in rows}
    for node in set(starts_at) | set(ends_at):
        ups = ends_at.get(node, [])
        downs = starts_at.get(node, [])
        for u in ups:
            for d in downs:
                if u == d:
                    continue
                downstream[u].append(d)
                upstream[d].append(u)

    streams = (
        gpd.GeoDataFrame(rows, geometry="geometry", crs=streams_gdf.crs)
        .sort_values("stream_id")
        .reset_index(drop=True)
    )
    downstream_t = {int(k): tuple(sorted(set(v))) for k, v in downstream.items()}
    upstream_t = {int(k): tuple(sorted(set(v))) for k, v in upstream.items()}
    return FacTopologyResult(
        streams=streams, downstream=downstream_t, upstream=upstream_t
    )


def estimate_reach_seed_strength(
    streams_gdf: gpd.GeoDataFrame,
    ndvi_da: xr.DataArray,
    support_da: xr.DataArray | None = None,
    *,
    sample_spacing_m: float = 20.0,
    ndvi_quantile: float = 0.9,
    ndvi_mid: float = 0.35,
    ndvi_scale: float = 0.06,
    support_override: float = 1.0,
) -> gpd.GeoDataFrame:
    """Estimate wet seed strength per FAC reach from raw NDVI and hard support."""
    if not (0.0 < ndvi_quantile <= 1.0):
        raise ValueError("ndvi_quantile must be in (0, 1]")
    if ndvi_scale <= 0.0:
        raise ValueError("ndvi_scale must be > 0")
    ndvi_interp = _build_raster_interp(ndvi_da)
    support_interp = (
        _build_raster_interp(support_da) if support_da is not None else None
    )

    out = streams_gdf.copy()
    ndvi_p = np.full(len(out), np.nan, dtype=np.float64)
    support_hit = np.zeros(len(out), dtype=bool)
    support_fraction = np.zeros(len(out), dtype=np.float64)
    seed = np.zeros(len(out), dtype=np.float64)

    for i, row in enumerate(out.itertuples(index=False)):
        vals = _sample_line_values(row.geometry, ndvi_interp, sample_spacing_m)
        good = np.isfinite(vals)
        if np.any(good):
            q = float(np.nanquantile(vals[good], ndvi_quantile))
            ndvi_p[i] = q
            seed_ndvi = 1.0 / (1.0 + np.exp(-(q - float(ndvi_mid)) / float(ndvi_scale)))
        else:
            seed_ndvi = 0.0
        seed_val = float(seed_ndvi)
        if support_interp is not None:
            svals = _sample_line_values(row.geometry, support_interp, sample_spacing_m)
            # Drop first and last samples (endpoints) to avoid leaking
            # support across shared confluence/reach-break vertices.
            # Reaches with ≤ 2 samples (shorter than sample_spacing) have
            # no true interior points, so seed_support_fraction stays 0 —
            # they are too short to assess support independently and must
            # not be hard-pinned. They can still get a soft anchor via
            # seed_support_hit from the full-line fallback.
            svals_interior = (
                svals[1:-1] if len(svals) > 2 else np.array([], dtype=svals.dtype)
            )
            svals_valid = svals_interior[np.isfinite(svals_interior)]
            svals_all = svals[np.isfinite(svals)]
            if len(svals_valid) > 0:
                frac = float(np.sum(svals_valid > 0.5)) / len(svals_valid)
                support_fraction[i] = frac
                hit = frac > 0.0
            elif len(svals_all) > 0:
                hit = bool(np.any(svals_all > 0.5))
            else:
                hit = False
            support_hit[i] = hit
            if hit:
                seed_val = max(seed_val, float(support_override))
        seed[i] = min(max(seed_val, 0.0), 1.0)

    out["seed_ndvi_q"] = ndvi_p
    out["seed_support_hit"] = support_hit
    out["seed_support_fraction"] = support_fraction
    out["seed_strength"] = seed
    return out


def propagate_upstream_wet_influence(
    topology: FacTopologyResult,
    *,
    distance_scale_m: float = 1500.0,
    elevation_scale_m: float = 25.0,
    strahler_distance_scale: float = 0.5,
) -> gpd.GeoDataFrame:
    """Propagate wet seed influence upstream with exponential decay."""
    if distance_scale_m <= 0.0:
        raise ValueError("distance_scale_m must be > 0")
    if elevation_scale_m <= 0.0:
        raise ValueError("elevation_scale_m must be > 0")

    streams = topology.streams.copy()
    by_id = {int(r.stream_id): r for r in streams.itertuples(index=False)}
    seed_strength = {
        int(r.stream_id): float(getattr(r, "seed_strength", 0.0))
        for r in streams.itertuples(index=False)
    }
    memo: dict[int, tuple[float, int, float, float]] = {}
    visiting: set[int] = set()

    def solve(stream_id: int) -> tuple[float, int, float, float]:
        if stream_id in memo:
            return memo[stream_id]
        if stream_id in visiting:
            return (
                seed_strength.get(stream_id, 0.0),
                stream_id if seed_strength.get(stream_id, 0.0) > 0 else -1,
                0.0,
                0.0,
            )
        visiting.add(stream_id)
        row = by_id[stream_id]
        best_weight = seed_strength.get(stream_id, 0.0)
        best_seed = stream_id if best_weight > 0.0 else -1
        best_dist = 0.0 if best_seed >= 0 else np.nan
        best_gain = 0.0 if best_seed >= 0 else np.nan

        Ld_eff = float(distance_scale_m) * (
            1.0
            + float(strahler_distance_scale)
            * max(int(getattr(row, "strahler", 0)) - 1, 0)
        )
        relief = float(getattr(row, "relief_m", np.nan))
        if not np.isfinite(relief):
            relief = 0.0
        decay = np.exp(
            -float(getattr(row, "length_m", 0.0)) / Ld_eff
            - relief / float(elevation_scale_m)
        )

        for down_id in topology.downstream.get(stream_id, ()):
            down_w, down_seed, down_dist, down_gain = solve(int(down_id))
            cand = float(down_w) * float(decay)
            if cand > best_weight:
                best_weight = cand
                best_seed = int(down_seed)
                best_dist = float(down_dist) + float(getattr(row, "length_m", 0.0))
                best_gain = float(down_gain) + relief

        visiting.remove(stream_id)
        memo[stream_id] = (best_weight, best_seed, best_dist, best_gain)
        return memo[stream_id]

    topo_weight = np.zeros(len(streams), dtype=np.float64)
    source_seed = np.full(len(streams), -1, dtype=np.int64)
    dist_to_seed = np.full(len(streams), np.nan, dtype=np.float64)
    gain_to_seed = np.full(len(streams), np.nan, dtype=np.float64)
    for i, row in enumerate(streams.itertuples(index=False)):
        w, s, d, g = solve(int(row.stream_id))
        topo_weight[i] = w
        source_seed[i] = int(s)
        dist_to_seed[i] = d
        gain_to_seed[i] = g

    streams["topo_pin_weight"] = topo_weight
    streams["topo_seed_stream_id"] = source_seed
    streams["topo_dist_to_seed_m"] = dist_to_seed
    streams["topo_gain_to_seed_m"] = gain_to_seed
    return streams


def rasterize_reach_weights_max(
    streams_gdf: gpd.GeoDataFrame,
    match_da: xr.DataArray,
    *,
    weight_col: str = "topo_pin_weight",
    sample_step_m: float | None = None,
) -> xr.DataArray:
    """Burn reach weights to the match grid using per-cell max across sampled lines."""
    x = np.asarray(match_da.x.values, dtype=np.float64)
    y = np.asarray(match_da.y.values, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        raise ValueError("match_da must have at least 2 cells in x and y")
    dx = abs(float(np.median(np.diff(x))))
    dy = abs(float(np.median(np.diff(y))))
    step = float(sample_step_m) if sample_step_m is not None else 0.5 * min(dx, dy)

    out_arr = np.full(match_da.shape, np.nan, dtype=np.float64)
    for row in streams_gdf.itertuples(index=False):
        w = float(getattr(row, weight_col))
        if not np.isfinite(w) or w <= 0.0:
            continue
        line = row.geometry
        s_vals = _line_s_values(float(line.length), step)
        pts = np.array(
            [
                [line.interpolate(float(s)).x, line.interpolate(float(s)).y]
                for s in s_vals
            ],
            dtype=np.float64,
        )
        cols = np.round((pts[:, 0] - x[0]) / dx).astype(np.intp)
        rows = np.round((y[0] - pts[:, 1]) / dy).astype(np.intp)
        valid = (
            (rows >= 0)
            & (rows < out_arr.shape[0])
            & (cols >= 0)
            & (cols < out_arr.shape[1])
        )
        if not np.any(valid):
            continue
        rr = rows[valid]
        cc = cols[valid]
        cur = out_arr[rr, cc]
        repl = np.where(np.isfinite(cur), np.maximum(cur, w), w)
        out_arr[rr, cc] = repl

    out = xr.DataArray(
        out_arr,
        coords=match_da.coords,
        dims=match_da.dims,
        name="fac_topology_pin_weight",
        attrs={"long_name": "fac_topology_pin_weight", "units": "1"},
    )
    if hasattr(match_da, "rio"):
        out = out.rio.write_crs(match_da.rio.crs)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
        out = out.rio.write_nodata(np.nan)
    return out


def build_fac_topology_pin_weights(
    streams_gdf: gpd.GeoDataFrame,
    elev_da: xr.DataArray,
    ndvi_da: xr.DataArray,
    match_da: xr.DataArray,
    *,
    support_da: xr.DataArray | None = None,
    node_precision: int = 3,
    ndvi_sample_spacing_m: float = 20.0,
    ndvi_quantile: float = 0.9,
    ndvi_mid: float = 0.35,
    ndvi_scale: float = 0.06,
    support_override: float = 1.0,
    distance_scale_m: float = 1500.0,
    elevation_scale_m: float = 25.0,
    strahler_distance_scale: float = 0.5,
    raster_sample_step_m: float | None = None,
) -> tuple[gpd.GeoDataFrame, xr.DataArray]:
    """Build per-reach and raster topology-derived pin weights."""
    topo = build_fac_topology(streams_gdf, elev_da, node_precision=node_precision)
    seeded = estimate_reach_seed_strength(
        topo.streams,
        ndvi_da,
        support_da=support_da,
        sample_spacing_m=ndvi_sample_spacing_m,
        ndvi_quantile=ndvi_quantile,
        ndvi_mid=ndvi_mid,
        ndvi_scale=ndvi_scale,
        support_override=support_override,
    )
    topo.streams = seeded
    weighted = propagate_upstream_wet_influence(
        topo,
        distance_scale_m=distance_scale_m,
        elevation_scale_m=elevation_scale_m,
        strahler_distance_scale=strahler_distance_scale,
    )
    pin_da = rasterize_reach_weights_max(
        weighted,
        match_da,
        weight_col="topo_pin_weight",
        sample_step_m=raster_sample_step_m,
    )
    return weighted, pin_da
