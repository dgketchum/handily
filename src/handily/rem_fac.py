"""FAC-based aspect-normal strip prototype.

This path is intentionally simpler than :mod:`handily.rem_experimental`:

- use the dense flow-accumulation stream network directly
- do not resnap reaches
- do not require water support
- derive strip orientation from a valley-scale smoothed DEM
- terminate strips at the first hit on another stream or the AOI boundary

The output is a strip geometry prototype for debugging, not a full REM
ownership/rasterization workflow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import rioxarray
import xarray as xr
from rasterio.enums import Resampling as RioResampling
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.signal import fftconvolve
from shapely import STRtree
from shapely import from_wkb
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import unary_union

DEFAULT_DEM = Path("/data/ssd2/handily/nv/aoi_0773/dem_bounds_1m.tif")
DEFAULT_STREAMS = Path("/data/ssd2/handily/nv/aoi_0773/streams_fac.fgb")
DEFAULT_OUT_DIR = Path("/data/ssd2/handily/nv/aoi_0773/experimental_full")

DEFAULT_COARSE_RES_M = 20.0
DEFAULT_SMOOTH_SIGMA_M = 500.0
DEFAULT_STATION_SPACING_M = 200.0
DEFAULT_TANGENT_STEP_M = 10.0
DEFAULT_MIN_HIT_DIST_M = 5.0
DEFAULT_BURN_RES_M = 20.0
DEFAULT_GAUSSIAN_SIGMA_PX = 3.0
DEFAULT_IDW_RADIUS_M = 200.0
DEFAULT_IDW_POWER = 1.0
DEFAULT_WORKERS = 1


@dataclass
class FacOrientationField:
    smoothed_dem: xr.DataArray
    aoi_polygon: Polygon
    x_vals: np.ndarray
    y_vals: np.ndarray
    down_x_arr: np.ndarray
    down_y_arr: np.ndarray
    slope_arr: np.ndarray
    down_x_interp: RegularGridInterpolator
    down_y_interp: RegularGridInterpolator
    slope_interp: RegularGridInterpolator


_WORKER_STATE: dict | None = None


def _normalize(vec: np.ndarray) -> np.ndarray:
    nrm = float(np.hypot(vec[0], vec[1]))
    if nrm <= 1e-12:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.array([vec[0] / nrm, vec[1] / nrm], dtype=np.float64)


def _coord_transform(x_vals: np.ndarray, y_vals: np.ndarray) -> Affine:
    dx = float(abs(x_vals[1] - x_vals[0]))
    dy = float(abs(y_vals[1] - y_vals[0]))
    x0 = float(x_vals[0] - dx / 2.0)
    if y_vals[1] > y_vals[0]:
        y0 = float(y_vals[0] - dy / 2.0)
        return Affine(dx, 0.0, x0, 0.0, dy, y0)
    y0 = float(y_vals[0] + dy / 2.0)
    return Affine(dx, 0.0, x0, 0.0, -dy, y0)


def _axes_from_bounds(
    bounds: tuple[float, float, float, float],
    res_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    nx = max(1, int(np.ceil((maxx - minx) / res_m)))
    ny = max(1, int(np.ceil((maxy - miny) / res_m)))
    x = minx + (np.arange(nx, dtype=np.float64) + 0.5) * res_m
    y = maxy - (np.arange(ny, dtype=np.float64) + 0.5) * res_m
    return x, y


def _build_raster_interp(arr: np.ndarray, x_vals: np.ndarray, y_vals: np.ndarray):
    yy = y_vals.astype(np.float64)
    aa = np.asarray(arr, dtype=np.float64)
    if yy[0] > yy[-1]:
        yy = yy[::-1]
        aa = aa[::-1, :]
    return RegularGridInterpolator(
        (yy, x_vals.astype(np.float64)), aa, bounds_error=False, fill_value=np.nan
    )


def _sample_section_pixels(
    section_geom: LineString,
    base_elev: float,
    endpoint_elev: float,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (rows, cols, elevs) for pixels along a strip."""
    line_coords = np.array(section_geom.coords, dtype=np.float64)[:, :2]
    n_verts = len(line_coords)
    if n_verts < 2:
        return None
    seg_dx = np.diff(line_coords[:, 0])
    seg_dy = np.diff(line_coords[:, 1])
    seg_len = np.sqrt(seg_dx**2 + seg_dy**2)
    cum_len = np.empty(n_verts, dtype=np.float64)
    cum_len[0] = 0.0
    np.cumsum(seg_len, out=cum_len[1:])
    total = float(cum_len[-1])
    if total < 1e-6:
        return None

    res_x = abs(float(x_vals[1] - x_vals[0]))
    res_y = abs(float(y_vals[1] - y_vals[0]))
    step = min(res_x, res_y) * 0.5
    n_samples = max(2, int(np.ceil(total / step)) + 1)
    ts = np.linspace(0.0, total, n_samples)

    seg_idx = np.searchsorted(cum_len, ts, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, n_verts - 2)
    local_t = np.where(
        seg_len[seg_idx] > 1e-12,
        (ts - cum_len[seg_idx]) / seg_len[seg_idx],
        0.0,
    )
    sx = line_coords[seg_idx, 0] + local_t * seg_dx[seg_idx]
    sy = line_coords[seg_idx, 1] + local_t * seg_dy[seg_idx]
    t_frac = ts / total
    elevs = base_elev + t_frac * (endpoint_elev - base_elev)

    cols = np.round((sx - x_vals[0]) / res_x).astype(np.intp)
    rows = np.round((y_vals[0] - sy) / res_y).astype(np.intp)
    ny = len(y_vals)
    nx = len(x_vals)
    valid = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx) & np.isfinite(elevs)
    if not np.any(valid):
        return None
    return rows[valid], cols[valid], elevs[valid]


def _burn_section_to_accumulators(
    section_geom: LineString,
    base_elev: float,
    endpoint_elev: float,
    sum_wz: np.ndarray,
    sum_w: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> None:
    result = _sample_section_pixels(
        section_geom, base_elev, endpoint_elev, x_vals, y_vals
    )
    if result is None:
        return
    rows, cols, elevs = result
    np.add.at(sum_wz, (rows, cols), elevs)
    np.add.at(sum_w, (rows, cols), 1.0)


def _nan_gaussian_2d(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    if sigma_px <= 0.0:
        return data.copy()
    valid = np.isfinite(data)
    if not valid.any():
        return data.copy()
    vals = np.where(valid, data, 0.0)
    wts = valid.astype(np.float64)
    vals_smooth = gaussian_filter(vals, sigma=sigma_px, mode="nearest")
    wts_smooth = gaussian_filter(wts, sigma=sigma_px, mode="nearest")
    out = np.full_like(data, np.nan)
    ok = wts_smooth > 1e-12
    out[ok] = vals_smooth[ok] / wts_smooth[ok]
    return out


def _coarsen_dem(dem_da: xr.DataArray, coarse_res_m: float) -> xr.DataArray:
    xres = float(abs(dem_da.x.values[1] - dem_da.x.values[0]))
    yres = float(abs(dem_da.y.values[1] - dem_da.y.values[0]))
    fx = max(1, int(round(coarse_res_m / xres)))
    fy = max(1, int(round(coarse_res_m / yres)))
    coarse = dem_da.coarsen(x=fx, y=fy, boundary="trim").mean(skipna=True)
    coarse = coarse.rio.write_crs(dem_da.rio.crs)
    coarse = coarse.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return coarse


def _derive_aoi_polygon_from_dem(dem_da: xr.DataArray, coarse_res_m: float) -> Polygon:
    coarse = _coarsen_dem(dem_da.notnull().astype(np.uint8), coarse_res_m)
    arr = np.asarray(coarse.values, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("AOI mask must be 2D")
    x_vals = coarse.x.values.astype(np.float64)
    y_vals = coarse.y.values.astype(np.float64)
    transform = _coord_transform(x_vals, y_vals)
    polys = [
        shape(g)
        for g, val in shapes(arr, mask=arr > 0, transform=transform)
        if int(val) == 1
    ]
    if not polys:
        raise ValueError("Failed to derive AOI polygon from DEM footprint")
    aoi = unary_union(polys)
    if aoi.geom_type == "MultiPolygon":
        aoi = max(aoi.geoms, key=lambda g: g.area)
    return aoi.buffer(0)


def build_orientation_field(
    dem_da: xr.DataArray,
    coarse_res_m: float = DEFAULT_COARSE_RES_M,
    smooth_sigma_m: float = DEFAULT_SMOOTH_SIGMA_M,
) -> FacOrientationField:
    coarse = _coarsen_dem(dem_da, coarse_res_m)
    arr = np.asarray(coarse.values, dtype=np.float64)
    sigma_px = max(float(smooth_sigma_m / coarse_res_m), 0.0)
    smooth = _nan_gaussian_2d(arr, sigma_px)

    smoothed_da = xr.DataArray(
        smooth,
        coords={"y": coarse.y.values, "x": coarse.x.values},
        dims=("y", "x"),
        name="fac_smoothed_dem",
        attrs=coarse.attrs.copy(),
    )
    smoothed_da = smoothed_da.rio.write_crs(dem_da.rio.crs)
    smoothed_da = smoothed_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    x_vals = smoothed_da.x.values.astype(np.float64)
    y_vals = smoothed_da.y.values.astype(np.float64)
    work = np.asarray(smoothed_da.values, dtype=np.float64)
    y_for_grad = y_vals.copy()
    if y_for_grad[0] > y_for_grad[-1]:
        y_for_grad = y_for_grad[::-1]
        work = work[::-1, :]

    dy = float(abs(y_for_grad[1] - y_for_grad[0]))
    dx = float(abs(x_vals[1] - x_vals[0]))
    dz_dy, dz_dx = np.gradient(work, dy, dx)
    down_x = -dz_dx
    down_y = -dz_dy
    slope = np.hypot(down_x, down_y)
    good = slope > 1e-12
    down_x[good] /= slope[good]
    down_y[good] /= slope[good]
    down_x[~good] = np.nan
    down_y[~good] = np.nan

    if y_vals[0] > y_vals[-1]:
        down_x = down_x[::-1, :]
        down_y = down_y[::-1, :]
        slope = slope[::-1, :]

    return FacOrientationField(
        smoothed_dem=smoothed_da,
        aoi_polygon=_derive_aoi_polygon_from_dem(dem_da, coarse_res_m),
        x_vals=x_vals,
        y_vals=y_vals,
        down_x_arr=down_x,
        down_y_arr=down_y,
        slope_arr=slope,
        down_x_interp=_build_raster_interp(down_x, x_vals, y_vals),
        down_y_interp=_build_raster_interp(down_y, x_vals, y_vals),
        slope_interp=_build_raster_interp(slope, x_vals, y_vals),
    )


def _intersection_points(geom) -> list[tuple[float, float]]:
    if geom is None or geom.is_empty:
        return []
    gt = geom.geom_type
    if gt == "Point":
        return [(float(geom.x), float(geom.y))]
    if gt == "MultiPoint":
        return [(float(p.x), float(p.y)) for p in geom.geoms]
    if gt == "LineString":
        return [(float(c[0]), float(c[1])) for c in geom.coords]
    if gt in ("MultiLineString", "GeometryCollection"):
        out: list[tuple[float, float]] = []
        for part in geom.geoms:
            out.extend(_intersection_points(part))
        return out
    return []


def _station_s_values(line: LineString, spacing_m: float) -> np.ndarray:
    total = float(line.length)
    if total <= spacing_m:
        return (
            np.array([0.0, total], dtype=np.float64)
            if total > 1e-6
            else np.array([0.0], dtype=np.float64)
        )
    vals = np.arange(0.0, total, spacing_m, dtype=np.float64)
    if vals.size == 0 or vals[-1] < total - 1e-6:
        vals = np.append(vals, total)
    return vals


def _line_tangent(line: LineString, s_m: float, step_m: float) -> np.ndarray:
    total = float(line.length)
    if total <= 1e-6:
        return np.array([np.nan, np.nan], dtype=np.float64)
    half = min(step_m, max(total / 4.0, 1.0))
    s0 = max(0.0, s_m - half)
    s1 = min(total, s_m + half)
    if s1 <= s0 + 1e-6:
        s0 = max(0.0, s_m - 1.0)
        s1 = min(total, s_m + 1.0)
    p0 = line.interpolate(s0)
    p1 = line.interpolate(s1)
    return _normalize(np.array([p1.x - p0.x, p1.y - p0.y], dtype=np.float64))


def _sample_aspect_normal(
    field: FacOrientationField,
    base_xy: np.ndarray,
    tangent: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    sample = np.array([[base_xy[1], base_xy[0]]], dtype=np.float64)
    down_x = float(field.down_x_interp(sample)[0])
    down_y = float(field.down_y_interp(sample)[0])
    slope = float(field.slope_interp(sample)[0])
    if np.isfinite(down_x) and np.isfinite(down_y):
        normal = _normalize(np.array([-down_y, down_x], dtype=np.float64))
        if np.isfinite(normal[0]):
            return normal, slope, "aspect"
    left = _normalize(np.array([-tangent[1], tangent[0]], dtype=np.float64))
    return left, np.nan, "tangent_fallback"


def _first_boundary_hit(
    anchor_xy: np.ndarray,
    direction: np.ndarray,
    aoi_boundary,
    max_ray_dist_m: float,
    min_hit_dist_m: float,
) -> tuple[float, np.ndarray] | None:
    ray_end = anchor_xy + max_ray_dist_m * direction
    ray = LineString([tuple(anchor_xy), tuple(ray_end)])
    ix = ray.intersection(aoi_boundary)
    best_d = float("inf")
    best_xy: np.ndarray | None = None
    for px, py in _intersection_points(ix):
        vec = np.array([px - anchor_xy[0], py - anchor_xy[1]], dtype=np.float64)
        d = float(np.hypot(vec[0], vec[1]))
        if d < min_hit_dist_m or float(vec @ direction) <= 0.0:
            continue
        if d < best_d:
            best_d = d
            best_xy = np.array([px, py], dtype=np.float64)
    if best_xy is None:
        return None
    return best_d, best_xy


def _first_stream_hit(
    anchor_xy: np.ndarray,
    direction: np.ndarray,
    source_reach_id: int,
    ray_geom: LineString,
    stream_tree: STRtree,
    stream_geoms: list[LineString],
    stream_ids: list[int],
    min_hit_dist_m: float,
) -> tuple[float, np.ndarray, int] | None:
    best_d = float("inf")
    best_xy: np.ndarray | None = None
    best_rid = -1
    for idx in stream_tree.query(ray_geom):
        target_rid = int(stream_ids[idx])
        if target_rid == source_reach_id:
            continue
        ix = ray_geom.intersection(stream_geoms[idx])
        if ix.is_empty:
            continue
        for px, py in _intersection_points(ix):
            vec = np.array([px - anchor_xy[0], py - anchor_xy[1]], dtype=np.float64)
            d = float(np.hypot(vec[0], vec[1]))
            if d < min_hit_dist_m or float(vec @ direction) <= 0.0:
                continue
            if d < best_d:
                best_d = d
                best_xy = np.array([px, py], dtype=np.float64)
                best_rid = target_rid
    if best_xy is None:
        return None
    return best_d, best_xy, best_rid


def _init_strip_worker(
    stream_wkbs: list[bytes],
    stream_ids: list[int],
    aoi_boundary_wkb: bytes,
    max_ray_dist_m: float,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    down_x_arr: np.ndarray,
    down_y_arr: np.ndarray,
    slope_arr: np.ndarray,
) -> None:
    global _WORKER_STATE
    stream_geoms = [from_wkb(wkb) for wkb in stream_wkbs]
    _WORKER_STATE = {
        "stream_geoms": stream_geoms,
        "stream_ids": stream_ids,
        "stream_tree": STRtree(stream_geoms),
        "aoi_boundary": from_wkb(aoi_boundary_wkb),
        "max_ray_dist_m": max_ray_dist_m,
        "down_x_interp": _build_raster_interp(down_x_arr, x_vals, y_vals),
        "down_y_interp": _build_raster_interp(down_y_arr, x_vals, y_vals),
        "slope_interp": _build_raster_interp(slope_arr, x_vals, y_vals),
    }


def _generate_strip_rows_for_chunk(
    records: list[dict],
    station_spacing_m: float,
    tangent_step_m: float,
    min_hit_dist_m: float,
    halo_n: int = 0,
) -> list[dict]:
    assert _WORKER_STATE is not None
    out: list[dict] = []
    down_x_interp = _WORKER_STATE["down_x_interp"]
    down_y_interp = _WORKER_STATE["down_y_interp"]
    slope_interp = _WORKER_STATE["slope_interp"]
    aoi_boundary = _WORKER_STATE["aoi_boundary"]
    max_ray_dist_m = _WORKER_STATE["max_ray_dist_m"]
    stream_tree = _WORKER_STATE["stream_tree"]
    stream_geoms = _WORKER_STATE["stream_geoms"]
    stream_ids = _WORKER_STATE["stream_ids"]

    halo_half_len = station_spacing_m * 8.0

    for rec in records:
        line = rec["geometry"]
        if line is None or line.is_empty or float(line.length) <= 1e-6:
            continue
        s_vals = _station_s_values(line, station_spacing_m)
        n_on_stream = len(s_vals)
        for station_id, s_m in enumerate(s_vals):
            base_pt = line.interpolate(float(s_m))
            base_xy = np.array([float(base_pt.x), float(base_pt.y)], dtype=np.float64)
            tangent = _line_tangent(line, float(s_m), tangent_step_m)
            if not np.isfinite(tangent[0]):
                continue

            sample = np.array([[base_xy[1], base_xy[0]]], dtype=np.float64)
            down_x = float(down_x_interp(sample)[0])
            down_y = float(down_y_interp(sample)[0])
            slope_mag = float(slope_interp(sample)[0])
            if np.isfinite(down_x) and np.isfinite(down_y):
                normal = _normalize(np.array([-down_y, down_x], dtype=np.float64))
                orient_src = "aspect" if np.isfinite(normal[0]) else "tangent_fallback"
            else:
                normal = np.array([np.nan, np.nan], dtype=np.float64)
                orient_src = "tangent_fallback"
            if not np.isfinite(normal[0]):
                normal = _normalize(
                    np.array([-tangent[1], tangent[0]], dtype=np.float64)
                )
                slope_mag = np.nan
            if not np.isfinite(normal[0]):
                continue

            for base_dir in (normal, -normal):
                cross = float(tangent[0] * base_dir[1] - tangent[1] * base_dir[0])
                side = "left" if cross > 0.0 else "right"
                boundary_hit = _first_boundary_hit(
                    base_xy,
                    base_dir,
                    aoi_boundary,
                    max_ray_dist_m,
                    min_hit_dist_m,
                )
                if boundary_hit is None:
                    continue
                boundary_d, boundary_xy = boundary_hit
                ray_geom = LineString([tuple(base_xy), tuple(boundary_xy)])
                stream_hit = _first_stream_hit(
                    base_xy,
                    base_dir,
                    int(rec["reach_id"]),
                    ray_geom,
                    stream_tree,
                    stream_geoms,
                    stream_ids,
                    min_hit_dist_m,
                )

                if stream_hit is not None and stream_hit[0] < boundary_d:
                    hit_type = "interreach"
                    dist_m = float(stream_hit[0])
                    endpoint_xy = stream_hit[1]
                    target_reach_id = int(stream_hit[2])
                else:
                    hit_type = "edge"
                    dist_m = float(boundary_d)
                    endpoint_xy = boundary_xy
                    target_reach_id = -1

                angle_deg = float(
                    (np.degrees(np.arctan2(base_dir[1], base_dir[0])) + 360.0) % 360.0
                )
                out.append(
                    {
                        "reach_id": int(rec["reach_id"]),
                        "stream_id": int(rec["stream_id"]),
                        "strahler": rec["strahler"],
                        "station_id": int(station_id),
                        "s_m": float(s_m),
                        "side": side,
                        "hit_type": hit_type,
                        "target_reach_id": int(target_reach_id),
                        "dist_m": dist_m,
                        "base_x": float(base_xy[0]),
                        "base_y": float(base_xy[1]),
                        "endpoint_x": float(endpoint_xy[0]),
                        "endpoint_y": float(endpoint_xy[1]),
                        "angle_deg": angle_deg,
                        "orientation_source": orient_src,
                        "slope_mag": slope_mag,
                        "geometry": LineString([tuple(base_xy), tuple(endpoint_xy)]),
                    }
                )

        # --- Halo: dandelion fan of spokes from terminal stream endpoints ---
        if halo_n <= 0:
            continue
        terminal_start = rec.get("terminal_start", False)
        terminal_end = rec.get("terminal_end", False)
        if not terminal_start and not terminal_end:
            continue
        total_len = float(line.length)
        tan_start = -_line_tangent(line, 0.0, tangent_step_m)
        tan_end = _line_tangent(line, total_len, tangent_step_m)
        endpoints = []
        if terminal_start:
            endpoints.append((0, tan_start, 0.0))
        if terminal_end:
            endpoints.append((1, tan_end, total_len))
        for end_idx, tan_vec, ref_s in endpoints:
            if not np.isfinite(tan_vec[0]):
                continue
            ref_pt = line.interpolate(ref_s)
            base_xy = np.array([float(ref_pt.x), float(ref_pt.y)], dtype=np.float64)
            # Fan halo_n spokes over a 180° arc centered on the outward tangent
            center_angle = float(np.arctan2(tan_vec[1], tan_vec[0]))
            for h_i in range(halo_n):
                theta = center_angle - np.pi / 2 + np.pi * (h_i + 0.5) / halo_n
                spoke_dir = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
                ep_xy = base_xy + spoke_dir * halo_half_len
                angle_deg = float((np.degrees(theta) + 360.0) % 360.0)
                cross = float(tan_vec[0] * spoke_dir[1] - tan_vec[1] * spoke_dir[0])
                side = "left" if cross > 0.0 else "right"
                sid = n_on_stream + end_idx * halo_n + h_i
                out.append(
                    {
                        "reach_id": int(rec["reach_id"]),
                        "stream_id": int(rec["stream_id"]),
                        "strahler": rec["strahler"],
                        "station_id": int(sid),
                        "s_m": float(ref_s),
                        "side": side,
                        "hit_type": "halo",
                        "target_reach_id": -1,
                        "dist_m": float(halo_half_len),
                        "base_x": float(base_xy[0]),
                        "base_y": float(base_xy[1]),
                        "endpoint_x": float(ep_xy[0]),
                        "endpoint_y": float(ep_xy[1]),
                        "angle_deg": angle_deg,
                        "orientation_source": "halo",
                        "slope_mag": np.nan,
                        "geometry": LineString([tuple(base_xy), tuple(ep_xy)]),
                    }
                )
    return out


def _attach_fac_strip_elevations(
    strips: gpd.GeoDataFrame,
    dem_da: xr.DataArray,
) -> gpd.GeoDataFrame:
    if strips.empty:
        strips["base_elev_m"] = np.array([], dtype=np.float64)
        strips["endpoint_elev_m"] = np.array([], dtype=np.float64)
        return strips

    dem_interp = _build_raster_interp(
        np.asarray(dem_da.values, dtype=np.float64),
        dem_da.x.values.astype(np.float64),
        dem_da.y.values.astype(np.float64),
    )
    base_pts = np.column_stack(
        [
            strips["base_y"].to_numpy(dtype=np.float64),
            strips["base_x"].to_numpy(dtype=np.float64),
        ]
    )
    base_elev = dem_interp(base_pts)
    endpoint_elev = base_elev.copy()

    interreach = strips["hit_type"].eq("interreach").to_numpy()
    if interreach.any():
        end_pts = np.column_stack(
            [
                strips.loc[interreach, "endpoint_y"].to_numpy(dtype=np.float64),
                strips.loc[interreach, "endpoint_x"].to_numpy(dtype=np.float64),
            ]
        )
        endpoint_elev[interreach] = dem_interp(end_pts)

    strips = strips.copy()
    strips["base_elev_m"] = base_elev
    strips["endpoint_elev_m"] = endpoint_elev
    return strips


def _attach_fac_strip_head_elevations(
    strips: gpd.GeoDataFrame,
    heads_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Override strip base/endpoint elevations with solved channel heads.

    Strips must already have DEM-based elevations (from generate_fac_strips).
    Values are only overridden where the head lookup succeeds; DEM values
    are preserved as fallback.
    """
    if strips.empty:
        return strips

    strips = strips.copy()

    head_by_stream = dict(
        zip(
            heads_gdf["stream_id"].astype(int),
            heads_gdf["channel_head_m"].astype(np.float64),
        )
    )
    head_by_reach: dict[int, float] = {}
    if "reach_id" in heads_gdf.columns:
        head_by_reach = dict(
            zip(
                heads_gdf["reach_id"].astype(int),
                heads_gdf["channel_head_m"].astype(np.float64),
            )
        )

    # Override base elevation with source reach's solved head
    base_heads = np.array(
        [head_by_stream.get(int(sid), np.nan) for sid in strips["stream_id"]],
        dtype=np.float64,
    )
    has_base = np.isfinite(base_heads)
    if has_base.any():
        strips.loc[has_base, "base_elev_m"] = base_heads[has_base]

    # Override interreach endpoint with target reach's solved head
    interreach = strips["hit_type"].eq("interreach").to_numpy()
    if interreach.any() and head_by_reach:
        ep_heads = np.array(
            [
                head_by_reach.get(int(rid), np.nan)
                for rid in strips.loc[interreach, "target_reach_id"]
            ],
            dtype=np.float64,
        )
        has_ep = np.isfinite(ep_heads)
        idx_ir = strips.index[interreach]
        if has_ep.any():
            strips.loc[idx_ir[has_ep], "endpoint_elev_m"] = ep_heads[has_ep]

    # Edge and halo strips: endpoint = base
    edge_or_halo = (
        strips["hit_type"].eq("edge") | strips["hit_type"].eq("halo")
    ).to_numpy()
    if edge_or_halo.any():
        strips.loc[edge_or_halo, "endpoint_elev_m"] = strips.loc[
            edge_or_halo, "base_elev_m"
        ].values

    return strips


def _attach_fac_strip_head_depth_offset(
    strips: gpd.GeoDataFrame,
    heads_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Lower strip base/endpoint elevations by solved head_depth_m.

    Instead of overriding DEM elevations with an absolute channel_head_m
    (which causes sawtooth REM on long reaches), this applies head_depth_m
    as a constant depth offset below the original DEM-sampled base.  The
    water surface then follows the bed slope at constant depth per reach.
    """
    if strips.empty:
        return strips

    strips = strips.copy()

    depth_by_stream = dict(
        zip(
            heads_gdf["stream_id"].astype(int),
            heads_gdf["head_depth_m"].astype(np.float64),
        )
    )
    depth_by_reach: dict[int, float] = {}
    if "reach_id" in heads_gdf.columns:
        depth_by_reach = dict(
            zip(
                heads_gdf["reach_id"].astype(int),
                heads_gdf["head_depth_m"].astype(np.float64),
            )
        )

    # Lower base elevation by source reach's depth
    base_depths = np.array(
        [depth_by_stream.get(int(sid), 0.0) for sid in strips["stream_id"]],
        dtype=np.float64,
    )
    strips["base_elev_m"] = strips["base_elev_m"].values - base_depths

    # Lower interreach endpoint by target reach's depth
    interreach = strips["hit_type"].eq("interreach").to_numpy()
    if interreach.any() and depth_by_reach:
        ep_depths = np.array(
            [
                depth_by_reach.get(int(rid), 0.0)
                for rid in strips.loc[interreach, "target_reach_id"]
            ],
            dtype=np.float64,
        )
        idx_ir = strips.index[interreach]
        strips.loc[idx_ir, "endpoint_elev_m"] = (
            strips.loc[idx_ir, "endpoint_elev_m"].values - ep_depths
        )

    # Edge and halo strips: endpoint = base (same reach depth)
    edge_or_halo = (
        strips["hit_type"].eq("edge") | strips["hit_type"].eq("halo")
    ).to_numpy()
    if edge_or_halo.any():
        strips.loc[edge_or_halo, "endpoint_elev_m"] = strips.loc[
            edge_or_halo, "base_elev_m"
        ].values

    return strips


def _remove_long_crossing_strips(
    strips: gpd.GeoDataFrame,
    max_length_m: float,
) -> gpd.GeoDataFrame:
    """Drop strips longer than *max_length_m* that intersect any other strip."""
    long_mask = strips["dist_m"].values > max_length_m
    if not long_mask.any():
        return strips

    long_idx = strips.index[long_mask]

    tree = STRtree(strips["geometry"].values)

    drop = set()
    for idx in long_idx:
        geom = strips.at[idx, "geometry"]
        if geom is None or geom.is_empty:
            continue
        candidates = tree.query(geom, predicate="intersects")
        for c in candidates:
            other_idx = strips.index[c]
            if other_idx != idx:
                drop.add(idx)
                break

    return strips.drop(index=list(drop)).reset_index(drop=True)


def generate_fac_strips(
    streams_gdf: gpd.GeoDataFrame,
    dem_da: xr.DataArray,
    field: FacOrientationField | None = None,
    station_spacing_m: float = DEFAULT_STATION_SPACING_M,
    tangent_step_m: float = DEFAULT_TANGENT_STEP_M,
    min_hit_dist_m: float = DEFAULT_MIN_HIT_DIST_M,
    halo_n: int = 0,
    workers: int = DEFAULT_WORKERS,
) -> gpd.GeoDataFrame:
    if field is None:
        field = build_orientation_field(dem_da)

    work = streams_gdf.copy()
    if "reach_id" not in work.columns:
        work["reach_id"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values("reach_id").reset_index(drop=True)

    stream_geoms: list[LineString] = []
    stream_ids: list[int] = []
    for row in work.itertuples():
        if row.geometry is None or row.geometry.is_empty:
            continue
        stream_geoms.append(row.geometry)
        stream_ids.append(int(row.reach_id))
    stream_tree = STRtree(stream_geoms)

    bounds = field.aoi_polygon.bounds
    max_ray_dist_m = float(np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])) * 1.5
    aoi_boundary = field.aoi_polygon.boundary

    # Identify terminal endpoints (headwaters/outlets not shared with another reach)
    _JUNCTION_TOL = 1.0  # metres
    all_starts = []
    all_ends = []
    all_rids = []
    for geom, rid in zip(stream_geoms, stream_ids):
        coords = np.array(geom.coords)
        all_starts.append(coords[0, :2])
        all_ends.append(coords[-1, :2])
        all_rids.append(rid)
    all_starts = np.array(all_starts, dtype=np.float64)
    all_ends = np.array(all_ends, dtype=np.float64)
    n_streams = len(all_starts)

    terminal_start_set: set[int] = set()
    terminal_end_set: set[int] = set()
    for i in range(n_streams):
        # Check if start[i] is near any other stream's start or end
        dists_to_starts = np.hypot(
            all_starts[:, 0] - all_starts[i, 0], all_starts[:, 1] - all_starts[i, 1]
        )
        dists_to_ends = np.hypot(
            all_ends[:, 0] - all_starts[i, 0], all_ends[:, 1] - all_starts[i, 1]
        )
        dists_to_starts[i] = np.inf  # exclude self
        if (
            dists_to_starts.min() > _JUNCTION_TOL
            and dists_to_ends.min() > _JUNCTION_TOL
        ):
            terminal_start_set.add(all_rids[i])

        # Check if end[i] is near any other stream's start or end
        dists_to_starts2 = np.hypot(
            all_starts[:, 0] - all_ends[i, 0], all_starts[:, 1] - all_ends[i, 1]
        )
        dists_to_ends2 = np.hypot(
            all_ends[:, 0] - all_ends[i, 0], all_ends[:, 1] - all_ends[i, 1]
        )
        dists_to_ends2[i] = np.inf  # exclude self
        if (
            dists_to_starts2.min() > _JUNCTION_TOL
            and dists_to_ends2.min() > _JUNCTION_TOL
        ):
            terminal_end_set.add(all_rids[i])

    records = [
        {
            "reach_id": int(row.reach_id),
            "stream_id": int(getattr(row, "stream_id", int(row.reach_id))),
            "strahler": getattr(row, "strahler", np.nan),
            "geometry": row.geometry,
            "terminal_start": int(row.reach_id) in terminal_start_set,
            "terminal_end": int(row.reach_id) in terminal_end_set,
        }
        for row in work.itertuples()
        if row.geometry is not None and not row.geometry.is_empty
    ]

    rows: list[dict] = []
    if workers <= 1:
        global _WORKER_STATE
        _WORKER_STATE = {
            "stream_geoms": stream_geoms,
            "stream_ids": stream_ids,
            "stream_tree": stream_tree,
            "aoi_boundary": aoi_boundary,
            "max_ray_dist_m": max_ray_dist_m,
            "down_x_interp": field.down_x_interp,
            "down_y_interp": field.down_y_interp,
            "slope_interp": field.slope_interp,
        }
        rows = _generate_strip_rows_for_chunk(
            records,
            station_spacing_m,
            tangent_step_m,
            min_hit_dist_m,
            halo_n=halo_n,
        )
    else:
        chunk_size = max(1, int(np.ceil(len(records) / (workers * 4))))
        chunks = [
            records[i : i + chunk_size] for i in range(0, len(records), chunk_size)
        ]
        stream_wkbs = [geom.wkb for geom in stream_geoms]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_strip_worker,
            initargs=(
                stream_wkbs,
                stream_ids,
                aoi_boundary.wkb,
                max_ray_dist_m,
                field.x_vals,
                field.y_vals,
                field.down_x_arr,
                field.down_y_arr,
                field.slope_arr,
            ),
        ) as ex:
            futs = {
                ex.submit(
                    _generate_strip_rows_for_chunk,
                    chunk,
                    station_spacing_m,
                    tangent_step_m,
                    min_hit_dist_m,
                    halo_n,
                ): len(chunk)
                for chunk in chunks
            }
            completed = 0
            for fut in as_completed(futs):
                chunk_rows = fut.result()
                rows.extend(chunk_rows)
                completed += futs[fut]
                print(
                    f"  processed {completed}/{len(records)} reaches, "
                    f"{len(rows)} strips",
                    flush=True,
                )

    strips = gpd.GeoDataFrame(rows, geometry="geometry", crs=work.crs)
    if not strips.empty:
        strips = strips.sort_values(["reach_id", "side", "station_id"]).reset_index(
            drop=True
        )
    return _attach_fac_strip_elevations(strips, dem_da)


def build_fac_wedges(strips: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows: list[dict] = []
    if strips.empty:
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=strips.crs)

    for (reach_id, side), grp in strips.groupby(["reach_id", "side"], sort=False):
        grp = grp.sort_values("station_id").reset_index(drop=True)
        for i in range(len(grp) - 1):
            s0 = grp.iloc[i]
            s1 = grp.iloc[i + 1]
            if str(s0["hit_type"]) != str(s1["hit_type"]):
                continue
            if str(s0["hit_type"]) == "interreach" and int(
                s0["target_reach_id"]
            ) != int(s1["target_reach_id"]):
                continue

            quad = Polygon(
                [
                    (float(s0["base_x"]), float(s0["base_y"])),
                    (float(s0["endpoint_x"]), float(s0["endpoint_y"])),
                    (float(s1["endpoint_x"]), float(s1["endpoint_y"])),
                    (float(s1["base_x"]), float(s1["base_y"])),
                ]
            )
            if quad.is_empty:
                continue
            if not quad.is_valid:
                quad = quad.buffer(0)
            if quad.is_empty or quad.area <= 1e-6:
                continue

            rows.append(
                {
                    "reach_id": int(reach_id),
                    "side": side,
                    "hit_type": str(s0["hit_type"]),
                    "target_reach_id": int(s0["target_reach_id"]),
                    "station_id_0": int(s0["station_id"]),
                    "station_id_1": int(s1["station_id"]),
                    "base_elev_0": float(s0["base_elev_m"]),
                    "base_elev_1": float(s1["base_elev_m"]),
                    "endpoint_elev_0": float(s0["endpoint_elev_m"]),
                    "endpoint_elev_1": float(s1["endpoint_elev_m"]),
                    "geometry": quad,
                }
            )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=strips.crs)


def rasterize_sparse_sections_20m(
    strips: gpd.GeoDataFrame,
    dem_da: xr.DataArray,
    res_m: float = DEFAULT_BURN_RES_M,
) -> tuple[xr.DataArray, xr.DataArray]:
    x_vals, y_vals = _axes_from_bounds(tuple(dem_da.rio.bounds()), res_m)
    ny, nx = len(y_vals), len(x_vals)
    min_arr = np.full((ny, nx), np.inf, dtype=np.float64)
    count_arr = np.zeros((ny, nx), dtype=np.float64)

    for row in strips.itertuples():
        g = row.geometry
        be = float(row.base_elev_m)
        ee = float(row.endpoint_elev_m)
        if g is None or g.is_empty or not np.isfinite(be) or not np.isfinite(ee):
            continue
        result = _sample_section_pixels(g, be, ee, x_vals, y_vals)
        if result is None:
            continue
        rows, cols, elevs = result
        np.minimum.at(min_arr, (rows, cols), elevs)
        np.add.at(count_arr, (rows, cols), 1.0)

    ws = np.full((ny, nx), np.nan, dtype=np.float64)
    mask = count_arr > 0.0
    ws[mask] = min_arr[mask]

    ws_da = xr.DataArray(
        ws,
        coords={"y": y_vals, "x": x_vals},
        dims=("y", "x"),
        name="fac_sparse_sections_ws_20m",
    )
    ws_da = ws_da.rio.write_crs(dem_da.rio.crs)
    ws_da = ws_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    count_da = xr.DataArray(
        count_arr.astype(np.float32),
        coords={"y": y_vals, "x": x_vals},
        dims=("y", "x"),
        name="fac_sparse_sections_count_20m",
    )
    count_da = count_da.rio.write_crs(dem_da.rio.crs)
    count_da = count_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return ws_da, count_da


def sample_dem_to_grid(
    dem_da: xr.DataArray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> xr.DataArray:
    dem_interp = _build_raster_interp(
        np.asarray(dem_da.values, dtype=np.float64),
        dem_da.x.values.astype(np.float64),
        dem_da.y.values.astype(np.float64),
    )
    xx, yy = np.meshgrid(x_vals, y_vals)
    pts = np.column_stack([yy.ravel(), xx.ravel()])
    vals = dem_interp(pts).reshape((len(y_vals), len(x_vals)))
    out = xr.DataArray(
        vals,
        coords={"y": y_vals, "x": x_vals},
        dims=("y", "x"),
        name="fac_dem_20m",
    )
    out = out.rio.write_crs(dem_da.rio.crs)
    out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return out


def fill_sparse_sections_nearest(
    sparse_ws_da: xr.DataArray,
    dem20_da: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    sparse = np.asarray(sparse_ws_da.values, dtype=np.float64)
    dem20 = np.asarray(dem20_da.values, dtype=np.float64)
    mask = np.isfinite(dem20)
    valid = np.isfinite(sparse) & mask
    if not np.any(valid):
        filled = np.full_like(sparse, np.nan)
        dist = np.full_like(sparse, np.nan)
    else:
        invalid = ~valid
        dist_px, nearest_idx = distance_transform_edt(invalid, return_indices=True)
        filled = sparse[nearest_idx[0], nearest_idx[1]]
        filled[~mask] = np.nan
        res_x = abs(float(sparse_ws_da.x.values[1] - sparse_ws_da.x.values[0]))
        dist = dist_px * res_x
        dist[~mask] = np.nan

    filled_da = xr.DataArray(
        filled,
        coords=sparse_ws_da.coords,
        dims=sparse_ws_da.dims,
        name="fac_nearest_fill_ws_20m",
    )
    filled_da = filled_da.rio.write_crs(sparse_ws_da.rio.crs)
    filled_da = filled_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    dist_da = xr.DataArray(
        dist,
        coords=sparse_ws_da.coords,
        dims=sparse_ws_da.dims,
        name="fac_nearest_fill_distance_20m",
    )
    dist_da = dist_da.rio.write_crs(sparse_ws_da.rio.crs)
    dist_da = dist_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    rem = np.maximum(dem20 - filled, 0.0)
    rem[~mask] = np.nan
    rem_da = xr.DataArray(
        rem,
        coords=sparse_ws_da.coords,
        dims=sparse_ws_da.dims,
        name="fac_nearest_fill_rem_20m",
    )
    rem_da = rem_da.rio.write_crs(sparse_ws_da.rio.crs)
    rem_da = rem_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    return filled_da, dist_da, rem_da


def fill_sparse_sections_gaussian(
    sparse_ws_da: xr.DataArray,
    sparse_count_da: xr.DataArray,
    dem20_da: xr.DataArray,
    sigma_px: float = DEFAULT_GAUSSIAN_SIGMA_PX,
) -> tuple[xr.DataArray, xr.DataArray]:
    sparse = np.asarray(sparse_ws_da.values, dtype=np.float64)
    counts = np.asarray(sparse_count_da.values, dtype=np.float64)
    dem20 = np.asarray(dem20_da.values, dtype=np.float64)
    mask = np.isfinite(dem20)
    valid = np.isfinite(sparse) & (counts > 0.0) & mask

    vals = np.zeros_like(sparse, dtype=np.float64)
    wts = np.zeros_like(sparse, dtype=np.float64)
    vals[valid] = sparse[valid] * counts[valid]
    wts[valid] = counts[valid]

    num = gaussian_filter(vals, sigma=sigma_px, mode="nearest")
    den = gaussian_filter(wts, sigma=sigma_px, mode="nearest")

    filled = np.full_like(sparse, np.nan)
    ok = (den > 1e-12) & mask
    filled[ok] = num[ok] / den[ok]

    filled_da = xr.DataArray(
        filled,
        coords=sparse_ws_da.coords,
        dims=sparse_ws_da.dims,
        name="fac_gaussian_fill_ws_20m",
    )
    filled_da = filled_da.rio.write_crs(sparse_ws_da.rio.crs)
    filled_da = filled_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    rem = np.maximum(dem20 - filled, 0.0)
    rem[~mask] = np.nan
    rem_da = xr.DataArray(
        rem,
        coords=sparse_ws_da.coords,
        dims=sparse_ws_da.dims,
        name="fac_gaussian_fill_rem_20m",
    )
    rem_da = rem_da.rio.write_crs(sparse_ws_da.rio.crs)
    rem_da = rem_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return filled_da, rem_da


def _idw_kernel(radius_px: float, power: float, eps_px: float = 0.5) -> np.ndarray:
    rad = max(int(np.ceil(radius_px)), 1)
    yy, xx = np.mgrid[-rad : rad + 1, -rad : rad + 1]
    rr = np.hypot(xx.astype(np.float64), yy.astype(np.float64))
    kernel = np.zeros_like(rr, dtype=np.float64)
    inside = rr <= radius_px
    denom = np.maximum(rr[inside], eps_px)
    kernel[inside] = 1.0 / np.power(denom, power)
    return kernel


def fill_sparse_sections_idw(
    sparse_ws_da: xr.DataArray,
    sparse_count_da: xr.DataArray,
    dem20_da: xr.DataArray,
    radius_m: float = DEFAULT_IDW_RADIUS_M,
    power: float = DEFAULT_IDW_POWER,
    post_smooth_m: float = 0.0,
    dem_min_da: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    sparse = np.asarray(sparse_ws_da.values, dtype=np.float64)
    counts = np.asarray(sparse_count_da.values, dtype=np.float64)
    dem20 = np.asarray(dem20_da.values, dtype=np.float64)
    mask = np.isfinite(dem20)
    valid = np.isfinite(sparse) & (counts > 0.0) & mask

    res_x = abs(float(sparse_ws_da.x.values[1] - sparse_ws_da.x.values[0]))
    radius_px = max(float(radius_m / res_x), 1.0)
    kernel = _idw_kernel(radius_px=radius_px, power=power)

    vals = np.zeros_like(sparse, dtype=np.float64)
    wts = np.zeros_like(sparse, dtype=np.float64)
    vals[valid] = sparse[valid] * counts[valid]
    wts[valid] = counts[valid]

    num = fftconvolve(vals, kernel, mode="same")
    den = fftconvolve(wts, kernel, mode="same")

    filled = np.full_like(sparse, np.nan)
    ok = (den > 1e-12) & mask
    filled[ok] = num[ok] / den[ok]

    # Post-IDW Gaussian smooth to feather strip edges
    if post_smooth_m > 0.0:
        sigma_px = post_smooth_m / res_x
        filled = _nan_gaussian_2d(filled, sigma_px)
        filled[~mask] = np.nan

    # Clamp water surface to DEM — WS cannot exceed ground elevation.
    # Use min-resampled DEM if available (preserves narrow channel bottoms).
    clamp_surface = (
        np.asarray(dem_min_da.values, dtype=np.float64)
        if dem_min_da is not None
        else dem20
    )
    clamped = np.isfinite(filled) & np.isfinite(clamp_surface)
    filled[clamped] = np.minimum(filled[clamped], clamp_surface[clamped])

    filled_da = xr.DataArray(
        filled,
        coords=sparse_ws_da.coords,
        dims=sparse_ws_da.dims,
        name="fac_idw_fill_ws_20m",
    )
    filled_da = filled_da.rio.write_crs(sparse_ws_da.rio.crs)
    filled_da = filled_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    rem = np.maximum(dem20 - filled, 0.0)
    rem[~mask] = np.nan
    rem_da = xr.DataArray(
        rem,
        coords=sparse_ws_da.coords,
        dims=sparse_ws_da.dims,
        name="fac_idw_fill_rem_20m",
    )
    rem_da = rem_da.rio.write_crs(sparse_ws_da.rio.crs)
    rem_da = rem_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return filled_da, rem_da


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FAC-based aspect-normal strip prototype"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML config file (sets all defaults; CLI flags override)",
    )
    parser.add_argument("--dem-path", type=Path, default=None)
    parser.add_argument("--streams-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--coarse-res-m", type=float, default=None)
    parser.add_argument("--smooth-sigma-m", type=float, default=None)
    parser.add_argument("--station-spacing-m", type=float, default=None)
    parser.add_argument("--tangent-step-m", type=float, default=None)
    parser.add_argument("--min-hit-dist-m", type=float, default=None)
    parser.add_argument("--min-strahler", type=int, default=None)
    parser.add_argument("--max-crossing-strip-m", type=float, default=None)
    parser.add_argument("--halo-n", type=int, default=None)
    parser.add_argument("--burn-res-m", type=float, default=None)
    parser.add_argument("--idw-radius-m", type=float, default=None)
    parser.add_argument("--idw-power", type=float, default=None)
    parser.add_argument("--post-smooth-m", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--naip-path", type=Path, default=None)
    parser.add_argument("--support-path", type=Path, default=None)
    cli = parser.parse_args(argv)

    if cli.config is not None:
        from handily.rem_fac_config import FacRemConfig

        cfg = FacRemConfig.from_toml(cli.config)
        # Config provides defaults; CLI flags override.
        if cli.dem_path is None:
            cli.dem_path = Path(cfg.dem_path)
        if cli.streams_path is None:
            cli.streams_path = Path(cfg.streams_path)
        if cli.out_dir is None:
            cli.out_dir = Path(cfg.out_dir)
        if cli.naip_path is None and cfg.naip_path is not None:
            cli.naip_path = Path(cfg.naip_path)
        if cli.support_path is None and cfg.support_path is not None:
            cli.support_path = Path(cfg.support_path)
        if cli.coarse_res_m is None:
            cli.coarse_res_m = cfg.coarse_res_m
        if cli.smooth_sigma_m is None:
            cli.smooth_sigma_m = cfg.smooth_sigma_m
        if cli.station_spacing_m is None:
            cli.station_spacing_m = cfg.station_spacing_m
        if cli.tangent_step_m is None:
            cli.tangent_step_m = cfg.tangent_step_m
        if cli.min_hit_dist_m is None:
            cli.min_hit_dist_m = cfg.min_hit_dist_m
        if cli.min_strahler is None:
            cli.min_strahler = cfg.min_strahler
        if cli.max_crossing_strip_m is None:
            cli.max_crossing_strip_m = cfg.max_crossing_strip_m
        if cli.halo_n is None:
            cli.halo_n = cfg.halo_n
        if cli.burn_res_m is None:
            cli.burn_res_m = cfg.burn_res_m
        if cli.idw_radius_m is None:
            cli.idw_radius_m = cfg.idw_radius_m
        if cli.idw_power is None:
            cli.idw_power = cfg.idw_power
        if cli.post_smooth_m is None:
            cli.post_smooth_m = cfg.post_smooth_m
        if cli.workers is None:
            cli.workers = cfg.workers
        # Stash head-solve kwargs from config for main() to use.
        cli.head_solve_kwargs = cfg.head_solve_kwargs()
    else:
        # No config — use hardcoded module defaults for paths/geometry.
        if cli.dem_path is None:
            cli.dem_path = DEFAULT_DEM
        if cli.streams_path is None:
            cli.streams_path = DEFAULT_STREAMS
        if cli.out_dir is None:
            cli.out_dir = DEFAULT_OUT_DIR
        if cli.coarse_res_m is None:
            cli.coarse_res_m = DEFAULT_COARSE_RES_M
        if cli.smooth_sigma_m is None:
            cli.smooth_sigma_m = DEFAULT_SMOOTH_SIGMA_M
        if cli.station_spacing_m is None:
            cli.station_spacing_m = DEFAULT_STATION_SPACING_M
        if cli.tangent_step_m is None:
            cli.tangent_step_m = DEFAULT_TANGENT_STEP_M
        if cli.min_hit_dist_m is None:
            cli.min_hit_dist_m = DEFAULT_MIN_HIT_DIST_M
        if cli.min_strahler is None:
            cli.min_strahler = 0
        if cli.max_crossing_strip_m is None:
            cli.max_crossing_strip_m = 0.0
        if cli.halo_n is None:
            cli.halo_n = 0
        if cli.burn_res_m is None:
            cli.burn_res_m = DEFAULT_BURN_RES_M
        if cli.idw_radius_m is None:
            cli.idw_radius_m = DEFAULT_IDW_RADIUS_M
        if cli.idw_power is None:
            cli.idw_power = DEFAULT_IDW_POWER
        if cli.post_smooth_m is None:
            cli.post_smooth_m = 0.0
        if cli.workers is None:
            cli.workers = DEFAULT_WORKERS
        cli.head_solve_kwargs = {}

    return cli


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DEM: {args.dem_path}")
    dem_da = rioxarray.open_rasterio(args.dem_path).squeeze("band", drop=True)
    dem_da = dem_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    # Mask fill-value sentinels (e.g. -999999) that lack nodata metadata.
    dem_da = dem_da.where(dem_da > -1e5)

    print(f"Loading FAC streams: {args.streams_path}")
    streams = gpd.read_file(args.streams_path)
    if args.min_strahler > 0 and "strahler" in streams.columns:
        streams = streams.loc[streams["strahler"] >= args.min_strahler].copy()
    if "reach_id" not in streams.columns:
        streams["reach_id"] = np.arange(len(streams), dtype=np.int64)
    streams = streams.loc[
        streams.geometry.notnull() & ~streams.geometry.is_empty
    ].copy()
    streams = streams.sort_values("reach_id").reset_index(drop=True)
    print(f"  streams: {len(streams)}")

    print("Building smoothed DEM orientation field")
    field = build_orientation_field(
        dem_da,
        coarse_res_m=args.coarse_res_m,
        smooth_sigma_m=args.smooth_sigma_m,
    )
    streams.to_file(args.out_dir / "fac_normals_streams.fgb", driver="FlatGeobuf")
    gpd.GeoDataFrame(
        [{"geometry": field.aoi_polygon}], geometry="geometry", crs=streams.crs
    ).to_file(
        args.out_dir / "fac_normals_aoi.fgb",
        driver="FlatGeobuf",
    )
    field.smoothed_dem.rio.to_raster(args.out_dir / "fac_normals_smoothed_dem.tif")

    print("Generating aspect-normal strips")
    strips = generate_fac_strips(
        streams,
        dem_da,
        field=field,
        station_spacing_m=args.station_spacing_m,
        tangent_step_m=args.tangent_step_m,
        min_hit_dist_m=args.min_hit_dist_m,
        halo_n=args.halo_n,
        workers=args.workers,
    )
    print(
        f"  strips: {len(strips)} "
        f"({int((strips['hit_type'] == 'interreach').sum())} interreach, "
        f"{int((strips['hit_type'] == 'edge').sum())} edge)"
    )
    if args.max_crossing_strip_m > 0:
        n_before = len(strips)
        strips = _remove_long_crossing_strips(strips, args.max_crossing_strip_m)
        print(
            f"  crossing filter (>{args.max_crossing_strip_m:.0f} m): "
            f"removed {n_before - len(strips)}, kept {len(strips)}"
        )

    strips.to_file(args.out_dir / "fac_normals_cross_sections.fgb", driver="FlatGeobuf")

    if args.naip_path is not None:
        from handily.rem_fac_head import build_channel_heads
        from handily.rem_surface_relax import compute_naip_ndvi_match

        print("Running channel-head longitudinal solve")
        t0 = perf_counter()
        _x, _y = _axes_from_bounds(tuple(dem_da.rio.bounds()), args.burn_res_m)
        _dem20 = sample_dem_to_grid(dem_da, _x, _y)
        # Compute NDVI at native NAIP resolution, then resample to 20m with max
        # so the greenest pixel in each cell drives the seed strength.
        _x20, _y20 = _axes_from_bounds(tuple(dem_da.rio.bounds()), 20.0)
        _dem_20m = sample_dem_to_grid(dem_da, _x20, _y20)
        from rasterio.enums import Resampling as _Resamp

        ndvi_native = compute_naip_ndvi_match(
            str(args.naip_path),
            _dem20,
            resampling=_Resamp.nearest,
        )
        ndvi_da = ndvi_native.rio.reproject_match(_dem_20m, resampling=_Resamp.max)
        print(
            f"  NDVI: native {ndvi_native.shape} -> 20m {ndvi_da.shape} (max resample)"
        )

        support_da = None
        if args.support_path is not None and args.support_path.exists():
            support_da = rioxarray.open_rasterio(args.support_path).squeeze(
                "band", drop=True
            )
            support_da = support_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

        heads = build_channel_heads(
            streams,
            _dem20,
            ndvi_da,
            support_da=support_da,
            **args.head_solve_kwargs,
        )
        heads.to_file(args.out_dir / "fac_channel_heads.fgb", driver="FlatGeobuf")
        strips_depth = _attach_fac_strip_head_depth_offset(strips, heads)
        dt = perf_counter() - t0
        depth = heads["head_depth_m"]
        print(
            f"  {len(heads)} reaches solved in {dt:.1f}s, "
            f"head depth: min={depth.min():.2f} mean={depth.mean():.2f} "
            f"max={depth.max():.2f} m"
        )
    else:
        strips_depth = None

    # --- Depth-offset REM (head_depth as constant offset below local bed) ---
    if strips_depth is not None:
        print(f"Burning depth-offset sparse raster at {args.burn_res_m:.1f} m")
        t0 = perf_counter()
        sparse_hd_da, sparse_hd_count = rasterize_sparse_sections_20m(
            strips_depth,
            dem_da,
            res_m=args.burn_res_m,
        )
        dt = perf_counter() - t0
        burned = int(np.isfinite(sparse_hd_da.values).sum())
        total = int(sparse_hd_da.values.size)
        print(
            f"  sparse burn: {burned}/{total} pixels "
            f"({100.0 * burned / max(total, 1):.1f}%), {dt:.2f}s"
        )

        print("IDW-filling depth-offset raster")
        t0 = perf_counter()
        dem20_da = sample_dem_to_grid(
            dem_da,
            sparse_hd_da.x.values.astype(np.float64),
            sparse_hd_da.y.values.astype(np.float64),
        )
        # Min-resampled DEM to capture narrow channel bottoms for WS clamping
        dem_min_da = dem_da.rio.reproject_match(dem20_da, resampling=RioResampling.min)
        hd_idw_ws_da, hd_idw_rem_da = fill_sparse_sections_idw(
            sparse_hd_da,
            sparse_hd_count,
            dem20_da,
            radius_m=args.idw_radius_m,
            power=args.idw_power,
            post_smooth_m=args.post_smooth_m,
            dem_min_da=dem_min_da,
        )
        dt = perf_counter() - t0
        hd_filled = int(np.isfinite(hd_idw_ws_da.values).sum())
        print(f"  idw fill: {hd_filled}/{total} pixels, {dt:.2f}s")

        res_tag = f"{int(args.burn_res_m)}m"
        sparse_hd_da.rio.to_raster(
            args.out_dir / f"fac_head_depth_sparse_{res_tag}.tif"
        )
        hd_idw_ws_da.rio.to_raster(
            args.out_dir / f"fac_head_depth_idw_fill_{res_tag}.tif"
        )
        hd_idw_rem_da.rio.to_raster(args.out_dir / f"fac_head_depth_rem_{res_tag}.tif")
        print(f"Wrote fac_head_depth_rem_{res_tag}.tif")

    print(f"Done — outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
