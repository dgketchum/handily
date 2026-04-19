"""Upward-only water-surface relaxation from hard support.

This module treats an existing water-surface raster as a lower bound and lets
it relax upward from hard support pixels where water is known to be at the
ground surface. It never lowers the prior water surface and never rises above
the DEM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter

from handily.rem_sag import _boundary_mask


@dataclass
class WaterSurfaceRelaxInfo:
    iterations: int
    max_change: float
    n_support: int
    n_boundary_fixed: int


def _neighbor_sum_and_weight(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nbr_sum = np.zeros_like(values, dtype=np.float64)
    nbr_w = np.zeros_like(values, dtype=np.float64)

    pair = valid[1:, :] & valid[:-1, :]
    nbr_sum[1:, :][pair] += values[:-1, :][pair]
    nbr_w[1:, :][pair] += 1.0
    nbr_sum[:-1, :][pair] += values[1:, :][pair]
    nbr_w[:-1, :][pair] += 1.0

    pair = valid[:, 1:] & valid[:, :-1]
    nbr_sum[:, 1:][pair] += values[:, :-1][pair]
    nbr_w[:, 1:][pair] += 1.0
    nbr_sum[:, :-1][pair] += values[:, 1:][pair]
    nbr_w[:, :-1][pair] += 1.0

    return nbr_sum, nbr_w


def _match_transform(match_da: xr.DataArray) -> Affine:
    x = np.asarray(match_da.x.values, dtype=np.float64)
    y = np.asarray(match_da.y.values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size < 2 or y.size < 2:
        raise ValueError("match_da must have 1D x/y coordinates with at least 2 cells")
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    if not np.isfinite(dx) or not np.isfinite(dy):
        raise ValueError("invalid match_da coordinate spacing")
    return Affine.translation(float(x[0] - dx / 2.0), float(y[0] - dy / 2.0)) * Affine.scale(dx, dy)


def compute_naip_ndvi_match(
    naip_path: str,
    match_da: xr.DataArray,
    *,
    resampling: Resampling = Resampling.average,
) -> xr.DataArray:
    """Resample NAIP red/NIR bands directly to a match grid and compute NDVI."""
    match_transform = _match_transform(match_da)
    match_shape = tuple(int(v) for v in match_da.shape)
    dst_crs = match_da.rio.crs
    if dst_crs is None:
        raise ValueError("match_da must have a CRS")

    red = np.full(match_shape, np.nan, dtype=np.float32)
    nir = np.full(match_shape, np.nan, dtype=np.float32)

    with rasterio.open(naip_path) as src:
        if src.count < 4:
            raise ValueError("NAIP raster must have at least 4 bands (R,G,B,NIR)")
        reproject(
            source=rasterio.band(src, 1),
            destination=red,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=match_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        reproject(
            source=rasterio.band(src, 4),
            destination=nir,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=match_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    denom = nir + red
    ndvi = np.full(match_shape, np.nan, dtype=np.float32)
    ok = np.isfinite(red) & np.isfinite(nir) & (np.abs(denom) > 1e-12)
    ndvi[ok] = (nir[ok] - red[ok]) / denom[ok]
    ndvi = np.clip(ndvi, -1.0, 1.0)

    out = xr.DataArray(
        ndvi,
        coords=match_da.coords,
        dims=match_da.dims,
        name="ndvi",
        attrs={"long_name": "naip_ndvi", "units": "1"},
    )
    out = out.rio.write_crs(dst_crs)
    out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
    out = out.rio.write_nodata(np.nan)
    return out


def ndvi_to_clearance(
    ndvi_da: xr.DataArray,
    *,
    min_clearance: float = 0.1,
    max_clearance: float = 10.0,
    ndvi_dense: float = 0.6,
    ndvi_sparse: float = 0.1,
    gamma: float = 1.0,
) -> xr.DataArray:
    """Map NDVI to a minimum REM clearance raster.

    Low NDVI gets larger required clearance; high NDVI approaches min_clearance.
    """
    if min_clearance < 0.0:
        raise ValueError("min_clearance must be >= 0")
    if max_clearance < min_clearance:
        raise ValueError("max_clearance must be >= min_clearance")
    if ndvi_dense <= ndvi_sparse:
        raise ValueError("ndvi_dense must be > ndvi_sparse")
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")

    ndvi = np.asarray(ndvi_da.values, dtype=np.float64)
    clearance = np.full(ndvi.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ndvi)
    frac = np.clip((ndvi_dense - ndvi) / (ndvi_dense - ndvi_sparse), 0.0, 1.0)
    frac = frac**gamma
    clearance[valid] = float(min_clearance) + (
        float(max_clearance) - float(min_clearance)
    ) * frac[valid]

    out = xr.DataArray(
        clearance,
        coords=ndvi_da.coords,
        dims=ndvi_da.dims,
        name="ndvi_clearance",
        attrs={"long_name": "ndvi_based_minimum_rem_clearance", "units": "m"},
    )
    if hasattr(ndvi_da, "rio"):
        out = out.rio.write_crs(ndvi_da.rio.crs)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
        out = out.rio.write_nodata(np.nan)
    return out


def _normalized_gaussian(values: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0.0:
        return values.astype(np.float64, copy=True)
    valid = np.isfinite(values)
    vals = np.where(valid, values, 0.0).astype(np.float64, copy=False)
    wts = valid.astype(np.float64, copy=False)
    num = gaussian_filter(vals, sigma=float(sigma_px), mode="nearest")
    den = gaussian_filter(wts, sigma=float(sigma_px), mode="nearest")
    out = np.full(values.shape, np.nan, dtype=np.float64)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def smooth_ndvi_gaussian(ndvi_da: xr.DataArray, *, sigma_px: float = 2.0) -> xr.DataArray:
    vals = np.asarray(ndvi_da.values, dtype=np.float64)
    smoothed = _normalized_gaussian(vals, sigma_px=float(sigma_px))
    smoothed = np.clip(smoothed, -1.0, 1.0)
    out = xr.DataArray(
        smoothed,
        coords=ndvi_da.coords,
        dims=ndvi_da.dims,
        name="ndvi_smoothed",
        attrs={"long_name": "smoothed_naip_ndvi", "units": "1"},
    )
    if hasattr(ndvi_da, "rio"):
        out = out.rio.write_crs(ndvi_da.rio.crs)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
        out = out.rio.write_nodata(np.nan)
    return out


def ndvi_to_clearance_logistic(
    ndvi_da: xr.DataArray,
    *,
    min_clearance: float = 0.1,
    max_clearance: float = 10.0,
    ndvi_mid: float = 0.22,
    ndvi_scale: float = 0.08,
) -> xr.DataArray:
    """Map NDVI to clearance with a smooth logistic transition.

    Low NDVI approaches ``max_clearance`` and high NDVI approaches ``min_clearance``.
    """
    if min_clearance < 0.0:
        raise ValueError("min_clearance must be >= 0")
    if max_clearance < min_clearance:
        raise ValueError("max_clearance must be >= min_clearance")
    if ndvi_scale <= 0.0:
        raise ValueError("ndvi_scale must be > 0")

    ndvi = np.asarray(ndvi_da.values, dtype=np.float64)
    clearance = np.full(ndvi.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ndvi)
    logistic = 1.0 / (1.0 + np.exp((ndvi - float(ndvi_mid)) / float(ndvi_scale)))
    clearance[valid] = float(min_clearance) + (
        float(max_clearance) - float(min_clearance)
    ) * logistic[valid]

    out = xr.DataArray(
        clearance,
        coords=ndvi_da.coords,
        dims=ndvi_da.dims,
        name="ndvi_clearance_logistic",
        attrs={"long_name": "ndvi_logistic_minimum_rem_clearance", "units": "m"},
    )
    if hasattr(ndvi_da, "rio"):
        out = out.rio.write_crs(ndvi_da.rio.crs)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
        out = out.rio.write_nodata(np.nan)
    return out


def shallow_rem_pin_weight(
    rem_prior_da: xr.DataArray,
    clearance_target_da: xr.DataArray,
    *,
    tolerance_m: float = 1.0,
    max_weight: float = 2.0,
    exceedance_scale_m: float = 2.0,
) -> xr.DataArray:
    """Convert shallow-REM exceedance into a soft pin weight.

    Pixels only get a penalty when the prior REM is shallower than the target
    clearance by more than ``tolerance_m``. The penalty ramps up smoothly with
    exceedance instead of forcing a hard discontinuity.
    """
    if tolerance_m < 0.0:
        raise ValueError("tolerance_m must be >= 0")
    if max_weight < 0.0:
        raise ValueError("max_weight must be >= 0")
    if exceedance_scale_m <= 0.0:
        raise ValueError("exceedance_scale_m must be > 0")

    rem_prior_da, clearance_target_da = xr.align(rem_prior_da, clearance_target_da, join="exact")
    rem_prior = np.asarray(rem_prior_da.values, dtype=np.float64)
    clearance = np.asarray(clearance_target_da.values, dtype=np.float64)
    weight = np.full(rem_prior.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(rem_prior) & np.isfinite(clearance)
    exceed = np.maximum(clearance - rem_prior - float(tolerance_m), 0.0)
    weight[valid] = float(max_weight) * (
        1.0 - np.exp(-exceed[valid] / float(exceedance_scale_m))
    )

    out = xr.DataArray(
        weight,
        coords=rem_prior_da.coords,
        dims=rem_prior_da.dims,
        name="shallow_rem_pin_weight",
        attrs={"long_name": "soft_weight_for_implausibly_shallow_rem", "units": "1"},
    )
    if hasattr(rem_prior_da, "rio"):
        out = out.rio.write_crs(rem_prior_da.rio.crs)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
        out = out.rio.write_nodata(np.nan)
    return out


def ndvi_to_prior_pin_weight(
    ndvi_da: xr.DataArray,
    *,
    min_weight: float = 0.0,
    max_weight: float = 2.0,
    ndvi_mid: float = 0.22,
    ndvi_scale: float = 0.08,
) -> xr.DataArray:
    """Map raw NDVI to a soft prior-pin weight.

    High NDVI gives stronger pinning to the prior water surface. Low NDVI
    relaxes that pinning and allows the membrane solve to sag more between the
    hard thalweg supports and the weakly pinned dry areas.
    """
    if min_weight < 0.0:
        raise ValueError("min_weight must be >= 0")
    if max_weight < min_weight:
        raise ValueError("max_weight must be >= min_weight")
    if ndvi_scale <= 0.0:
        raise ValueError("ndvi_scale must be > 0")

    ndvi = np.asarray(ndvi_da.values, dtype=np.float64)
    weight = np.full(ndvi.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ndvi)
    logistic = 1.0 / (1.0 + np.exp(-(ndvi - float(ndvi_mid)) / float(ndvi_scale)))
    weight[valid] = float(min_weight) + (float(max_weight) - float(min_weight)) * logistic[valid]

    out = xr.DataArray(
        weight,
        coords=ndvi_da.coords,
        dims=ndvi_da.dims,
        name="ndvi_prior_pin_weight",
        attrs={"long_name": "ndvi_based_soft_prior_pin_weight", "units": "1"},
    )
    if hasattr(ndvi_da, "rio"):
        out = out.rio.write_crs(ndvi_da.rio.crs)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
        out = out.rio.write_nodata(np.nan)
    return out


def relax_water_surface_upward(
    ws_prior_da: xr.DataArray,
    dem_da: xr.DataArray,
    support_mask_da: xr.DataArray,
    fac_hint_da: xr.DataArray | None = None,
    *,
    min_clearance_off_support: float | xr.DataArray = 0.0,
    base_fidelity: float = 0.25,
    fac_hint_scale: float = 4.0,
    fix_boundary_to_prior: bool = True,
    max_iter: int = 500,
    tol: float = 1e-3,
    omega: float = 1.0,
    return_info: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, WaterSurfaceRelaxInfo]:
    """Projected upward-only membrane relaxation in water-surface space.

    Constraints:
    - pinned to ``dem`` on support pixels
    - ``ws >= ws_prior_clipped`` everywhere
    - ``ws <= dem`` on support pixels
    - ``ws <= dem - min_clearance_off_support`` off support
    """
    if not (0.0 < omega <= 1.0):
        raise ValueError("omega must be in (0, 1]")
    aligned = [ws_prior_da, dem_da, support_mask_da]
    clearance_idx: int | None = None
    if fac_hint_da is not None:
        aligned.append(fac_hint_da)
    if isinstance(min_clearance_off_support, xr.DataArray):
        clearance_idx = len(aligned)
        aligned.append(min_clearance_off_support)
    aligned = xr.align(*aligned, join="exact")

    ws_prior_da = aligned[0]
    dem_da = aligned[1]
    support_da = aligned[2]
    next_idx = 3
    hint_da = aligned[next_idx] if fac_hint_da is not None else None
    if fac_hint_da is not None:
        next_idx += 1
    clearance_da = aligned[clearance_idx] if clearance_idx is not None else None

    ws_prior = np.asarray(ws_prior_da.values, dtype=np.float64)
    dem = np.asarray(dem_da.values, dtype=np.float64)
    if ws_prior.ndim != 2 or dem.ndim != 2:
        raise ValueError("ws_prior_da and dem_da must be 2D")

    valid = np.isfinite(ws_prior) & np.isfinite(dem)
    support = np.asarray(support_da.values).astype(bool) & valid
    if clearance_da is None:
        if float(min_clearance_off_support) < 0.0:
            raise ValueError("min_clearance_off_support must be >= 0")
        clearance = np.full(ws_prior.shape, float(min_clearance_off_support), dtype=np.float64)
    else:
        clearance = np.asarray(clearance_da.values, dtype=np.float64)
        if clearance.shape != ws_prior.shape:
            raise ValueError("clearance raster shape mismatch")
        if np.nanmin(clearance) < 0.0:
            raise ValueError("clearance raster values must be >= 0")
        if np.any(valid & ~np.isfinite(clearance)):
            raise ValueError("clearance raster contains NaN/inf values on valid cells")
    upper = np.where(
        valid,
        np.where(support, dem, dem - clearance),
        np.nan,
    )
    lower = np.where(valid, np.minimum(ws_prior, upper), np.nan)
    boundary = _boundary_mask(valid) if fix_boundary_to_prior else np.zeros_like(valid, dtype=bool)
    fixed = support | boundary

    fidelity = np.full(lower.shape, float(base_fidelity), dtype=np.float64)
    if hint_da is not None:
        hint = np.asarray(hint_da.values, dtype=np.float64)
        hint = np.where(np.isfinite(hint) & valid, np.maximum(hint, 0.0), 0.0)
        hint_max = float(np.nanmax(hint)) if np.isfinite(hint).any() else 0.0
        if hint_max > 0.0:
            hint = hint / hint_max
        fidelity = fidelity / (1.0 + float(fac_hint_scale) * hint)

    current = lower.copy()
    current[support] = upper[support]

    max_change = np.inf
    updatable = valid & ~fixed
    for it in range(1, int(max_iter) + 1):
        nbr_sum, nbr_w = _neighbor_sum_and_weight(current, valid)
        denom = fidelity + nbr_w
        target = current.copy()
        ok = updatable & (denom > 1e-12)
        target[ok] = (fidelity[ok] * lower[ok] + nbr_sum[ok]) / denom[ok]
        candidate = current.copy()
        candidate[ok] = (1.0 - omega) * current[ok] + omega * target[ok]
        candidate[ok] = np.clip(candidate[ok], lower[ok], upper[ok])
        candidate[support] = upper[support]
        if fix_boundary_to_prior:
            candidate[boundary] = lower[boundary]
        candidate[~valid] = np.nan

        if np.any(ok):
            max_change = float(np.nanmax(np.abs(candidate[ok] - current[ok])))
        else:
            max_change = 0.0
        current = candidate
        if max_change <= tol:
            break

    out = xr.DataArray(
        current,
        coords=ws_prior_da.coords,
        dims=ws_prior_da.dims,
        name="relaxed_water_surface",
        attrs={
            "long_name": "upward_relaxed_water_surface_elevation",
            "units": "m",
        },
    )
    if hasattr(ws_prior_da, "rio"):
        out = out.rio.write_crs(ws_prior_da.rio.crs)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
        out = out.rio.write_nodata(np.nan)

    if not return_info:
        return out
    info = WaterSurfaceRelaxInfo(
        iterations=it,
        max_change=float(max_change),
        n_support=int(support.sum()),
        n_boundary_fixed=int(boundary.sum()),
    )
    return out, info


def rem_from_water_surface(dem_da: xr.DataArray, ws_da: xr.DataArray) -> xr.DataArray:
    dem_da, ws_da = xr.align(dem_da, ws_da, join="exact")
    dem = np.asarray(dem_da.values, dtype=np.float64)
    ws = np.asarray(ws_da.values, dtype=np.float64)
    rem = np.maximum(dem - ws, 0.0)
    rem[~(np.isfinite(dem) & np.isfinite(ws))] = np.nan
    out = xr.DataArray(
        rem,
        coords=dem_da.coords,
        dims=dem_da.dims,
        name="relaxed_rem",
        attrs={"long_name": "relative_elevation_model", "units": "m"},
    )
    if hasattr(dem_da, "rio"):
        out = out.rio.write_crs(dem_da.rio.crs)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
        out = out.rio.write_nodata(np.nan)
    return out


def relax_water_surface_soft_ceiling(
    ws_prior_da: xr.DataArray,
    dem_da: xr.DataArray,
    support_mask_da: xr.DataArray,
    soft_target_ws_da: xr.DataArray,
    pin_weight_da: xr.DataArray,
    fac_hint_da: xr.DataArray | None = None,
    *,
    min_clearance_off_support: float = 0.1,
    base_fidelity: float = 0.25,
    fac_hint_scale: float = 4.0,
    fix_boundary_to_prior: bool = True,
    max_iter: int = 500,
    tol: float = 1e-3,
    omega: float = 1.0,
    return_info: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, WaterSurfaceRelaxInfo]:
    """Relax water surface with hard thalweg support and soft shallow-REM penalties.

    This variant avoids hard local ceilings. Off support, it only enforces a
    global minimum REM clearance, while low-NDVI shallow areas are pulled
    downward smoothly through a soft target and weight.
    """
    if not (0.0 < omega <= 1.0):
        raise ValueError("omega must be in (0, 1]")
    if min_clearance_off_support < 0.0:
        raise ValueError("min_clearance_off_support must be >= 0")

    aligned = [ws_prior_da, dem_da, support_mask_da, soft_target_ws_da, pin_weight_da]
    if fac_hint_da is not None:
        aligned.append(fac_hint_da)
    aligned = xr.align(*aligned, join="exact")

    ws_prior_da = aligned[0]
    dem_da = aligned[1]
    support_da = aligned[2]
    soft_target_da = aligned[3]
    pin_weight_da = aligned[4]
    hint_da = aligned[5] if fac_hint_da is not None else None

    ws_prior = np.asarray(ws_prior_da.values, dtype=np.float64)
    dem = np.asarray(dem_da.values, dtype=np.float64)
    soft_target = np.asarray(soft_target_da.values, dtype=np.float64)
    pin_weight = np.asarray(pin_weight_da.values, dtype=np.float64)
    if ws_prior.ndim != 2 or dem.ndim != 2:
        raise ValueError("ws_prior_da and dem_da must be 2D")

    valid = np.isfinite(ws_prior) & np.isfinite(dem)
    support = np.asarray(support_da.values).astype(bool) & valid
    if np.any(valid & ~np.isfinite(soft_target)):
        raise ValueError("soft_target_ws_da contains NaN/inf values on valid cells")
    if np.any(valid & ~np.isfinite(pin_weight)):
        raise ValueError("pin_weight_da contains NaN/inf values on valid cells")
    pin_weight = np.where(valid, np.maximum(pin_weight, 0.0), 0.0)

    upper = np.where(valid, np.where(support, dem, dem - float(min_clearance_off_support)), np.nan)
    soft_target = np.where(valid, np.minimum(soft_target, upper), np.nan)
    boundary = _boundary_mask(valid) if fix_boundary_to_prior else np.zeros_like(valid, dtype=bool)
    fixed = support | boundary

    fidelity = np.full(ws_prior.shape, float(base_fidelity), dtype=np.float64)
    if hint_da is not None:
        hint = np.asarray(hint_da.values, dtype=np.float64)
        hint = np.where(np.isfinite(hint) & valid, np.maximum(hint, 0.0), 0.0)
        hint_max = float(np.nanmax(hint)) if np.isfinite(hint).any() else 0.0
        if hint_max > 0.0:
            hint = hint / hint_max
        fidelity = fidelity / (1.0 + float(fac_hint_scale) * hint)

    current = np.where(valid, np.minimum(ws_prior, upper), np.nan)
    current[support] = dem[support]

    max_change = np.inf
    updatable = valid & ~fixed
    for it in range(1, int(max_iter) + 1):
        nbr_sum, nbr_w = _neighbor_sum_and_weight(current, valid)
        denom = fidelity + nbr_w + pin_weight
        target = current.copy()
        ok = updatable & (denom > 1e-12)
        target[ok] = (
            fidelity[ok] * ws_prior[ok]
            + nbr_sum[ok]
            + pin_weight[ok] * soft_target[ok]
        ) / denom[ok]
        candidate = current.copy()
        candidate[ok] = (1.0 - omega) * current[ok] + omega * target[ok]
        candidate[ok] = np.minimum(candidate[ok], upper[ok])
        candidate[support] = dem[support]
        if fix_boundary_to_prior:
            candidate[boundary] = np.minimum(ws_prior[boundary], upper[boundary])
        candidate[~valid] = np.nan

        if np.any(ok):
            max_change = float(np.nanmax(np.abs(candidate[ok] - current[ok])))
        else:
            max_change = 0.0
        current = candidate
        if max_change <= tol:
            break

    out = xr.DataArray(
        current,
        coords=ws_prior_da.coords,
        dims=ws_prior_da.dims,
        name="relaxed_water_surface_soft_ceiling",
        attrs={"long_name": "soft_ceiling_relaxed_water_surface_elevation", "units": "m"},
    )
    if hasattr(ws_prior_da, "rio"):
        out = out.rio.write_crs(ws_prior_da.rio.crs)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
        out = out.rio.write_nodata(np.nan)

    if not return_info:
        return out
    info = WaterSurfaceRelaxInfo(
        iterations=it,
        max_change=float(max_change),
        n_support=int(support.sum()),
        n_boundary_fixed=int(boundary.sum()),
    )
    return out, info


def relax_water_surface_ndvi_pins(
    ws_prior_da: xr.DataArray,
    dem_da: xr.DataArray,
    support_mask_da: xr.DataArray,
    ndvi_pin_weight_da: xr.DataArray,
    *,
    min_clearance_off_support: float = 0.1,
    base_fidelity: float = 0.1,
    smoothness_weight: float = 2.0,
    fix_boundary_to_prior: bool = True,
    max_iter: int = 500,
    tol: float = 1e-3,
    omega: float = 1.0,
    return_info: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, WaterSurfaceRelaxInfo]:
    """Relax water surface with hard support and raw-NDVI soft pinning to prior.

    The only hard off-support constraint is the global minimum REM clearance.
    Raw NDVI controls how strongly the prior FAC-based surface is preserved:
    high NDVI keeps the prior shallow surface, while low NDVI lets the membrane
    sag more under the influence of the smoothness term.
    """
    if not (0.0 < omega <= 1.0):
        raise ValueError("omega must be in (0, 1]")
    if min_clearance_off_support < 0.0:
        raise ValueError("min_clearance_off_support must be >= 0")
    if base_fidelity < 0.0:
        raise ValueError("base_fidelity must be >= 0")
    if smoothness_weight <= 0.0:
        raise ValueError("smoothness_weight must be > 0")

    ws_prior_da, dem_da, support_da, ndvi_pin_weight_da = xr.align(
        ws_prior_da, dem_da, support_mask_da, ndvi_pin_weight_da, join="exact"
    )

    ws_prior = np.asarray(ws_prior_da.values, dtype=np.float64)
    dem = np.asarray(dem_da.values, dtype=np.float64)
    pin_weight = np.asarray(ndvi_pin_weight_da.values, dtype=np.float64)
    if ws_prior.ndim != 2 or dem.ndim != 2:
        raise ValueError("ws_prior_da and dem_da must be 2D")

    valid = np.isfinite(ws_prior) & np.isfinite(dem)
    support = np.asarray(support_da.values).astype(bool) & valid
    if np.any(valid & ~np.isfinite(pin_weight)):
        raise ValueError("ndvi_pin_weight_da contains NaN/inf values on valid cells")
    if np.nanmin(pin_weight) < 0.0:
        raise ValueError("ndvi_pin_weight_da values must be >= 0")

    upper = np.where(valid, np.where(support, dem, dem - float(min_clearance_off_support)), np.nan)
    prior_target = np.where(valid, np.minimum(ws_prior, upper), np.nan)
    boundary = _boundary_mask(valid) if fix_boundary_to_prior else np.zeros_like(valid, dtype=bool)
    fixed = support | boundary
    total_pin_weight = np.where(valid, float(base_fidelity) + pin_weight, 0.0)

    current = prior_target.copy()
    current[support] = dem[support]

    max_change = np.inf
    updatable = valid & ~fixed
    for it in range(1, int(max_iter) + 1):
        nbr_sum, nbr_w = _neighbor_sum_and_weight(current, valid)
        denom = total_pin_weight + float(smoothness_weight) * nbr_w
        target = current.copy()
        ok = updatable & (denom > 1e-12)
        target[ok] = (
            total_pin_weight[ok] * prior_target[ok]
            + float(smoothness_weight) * nbr_sum[ok]
        ) / denom[ok]
        candidate = current.copy()
        candidate[ok] = (1.0 - omega) * current[ok] + omega * target[ok]
        candidate[ok] = np.minimum(candidate[ok], upper[ok])
        candidate[support] = dem[support]
        if fix_boundary_to_prior:
            candidate[boundary] = prior_target[boundary]
        candidate[~valid] = np.nan

        if np.any(ok):
            max_change = float(np.nanmax(np.abs(candidate[ok] - current[ok])))
        else:
            max_change = 0.0
        current = candidate
        if max_change <= tol:
            break

    out = xr.DataArray(
        current,
        coords=ws_prior_da.coords,
        dims=ws_prior_da.dims,
        name="relaxed_water_surface_ndvi_pins",
        attrs={"long_name": "raw_ndvi_pin_relaxed_water_surface_elevation", "units": "m"},
    )
    if hasattr(ws_prior_da, "rio"):
        out = out.rio.write_crs(ws_prior_da.rio.crs)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
        out = out.rio.write_nodata(np.nan)

    if not return_info:
        return out
    info = WaterSurfaceRelaxInfo(
        iterations=it,
        max_change=float(max_change),
        n_support=int(support.sum()),
        n_boundary_fixed=int(boundary.sum()),
    )
    return out, info
