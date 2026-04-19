"""Downward-only REM relaxation from hard water-surface support.

This module treats an existing REM as an upper bound and lets it relax
downward from hard support pixels where water is known to be at the surface.
It does not lift the REM anywhere.

Typical usage:

1. Build a hard support mask from a lower-density snapped-thalweg network and
   an evidence-based surface water mask.
2. Solve a projected membrane relaxation in REM space:
   - pinned to zero on support pixels
   - optionally fixed to the prior on the AOI boundary
   - constrained by ``0 <= rem <= rem_prior`` everywhere
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio import features
from rasterio.transform import Affine


@dataclass
class RemSagInfo:
    iterations: int
    max_change: float
    n_support: int
    n_boundary_fixed: int


def _coord_transform(x_vals: np.ndarray, y_vals: np.ndarray) -> Affine:
    dx = float(abs(x_vals[1] - x_vals[0]))
    dy = float(abs(y_vals[1] - y_vals[0]))
    x0 = float(x_vals[0] - dx / 2.0)
    if y_vals[1] > y_vals[0]:
        y0 = float(y_vals[0] - dy / 2.0)
        return Affine(dx, 0.0, x0, 0.0, dy, y0)
    y0 = float(y_vals[0] + dy / 2.0)
    return Affine(dx, 0.0, x0, 0.0, -dy, y0)


def _bool_dataarray_like(template: xr.DataArray, values: np.ndarray, name: str) -> xr.DataArray:
    out = xr.DataArray(
        values.astype(np.uint8),
        coords=template.coords,
        dims=template.dims,
        name=name,
    )
    if hasattr(template, "rio"):
        out = out.rio.write_crs(template.rio.crs)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return out


def build_stream_evidence_support(
    snapped_gdf: gpd.GeoDataFrame,
    evidence_mask_da: xr.DataArray,
    buffer_m: float = 0.0,
) -> xr.DataArray:
    """Build hard support pixels from snapped streams intersecting evidence.

    Parameters
    ----------
    snapped_gdf:
        Lower-density stream network used as the trusted channel skeleton.
    evidence_mask_da:
        Binary or numeric mask on the target grid; positive/True means
        evidence for water at the surface.
    buffer_m:
        Optional stream buffer before rasterization.
    """
    if "x" not in evidence_mask_da.coords or "y" not in evidence_mask_da.coords:
        raise ValueError("evidence_mask_da must have x/y coordinates")
    if len(evidence_mask_da.x) < 2 or len(evidence_mask_da.y) < 2:
        raise ValueError("evidence_mask_da must have at least 2 cells in x and y")

    x_vals = evidence_mask_da.x.values.astype(np.float64)
    y_vals = evidence_mask_da.y.values.astype(np.float64)
    transform = _coord_transform(x_vals, y_vals)
    arr = np.asarray(evidence_mask_da.values)
    if arr.ndim != 2:
        raise ValueError("evidence_mask_da must be 2D")

    shapes = []
    for geom in snapped_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        shapes.append((geom.buffer(buffer_m) if buffer_m > 0.0 else geom, 1))
    if not shapes:
        support = np.zeros(arr.shape, dtype=bool)
    else:
        stream_mask = features.rasterize(
            shapes,
            out_shape=arr.shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        support = (stream_mask > 0) & np.isfinite(arr) & (arr > 0)
    return _bool_dataarray_like(evidence_mask_da, support, "stream_evidence_support")


def _boundary_mask(valid: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(valid, dtype=bool)
    if valid.size == 0:
        return boundary
    boundary[0, :] |= valid[0, :]
    boundary[-1, :] |= valid[-1, :]
    boundary[:, 0] |= valid[:, 0]
    boundary[:, -1] |= valid[:, -1]

    north_invalid = np.zeros_like(valid, dtype=bool)
    north_invalid[1:, :] = ~valid[:-1, :]
    south_invalid = np.zeros_like(valid, dtype=bool)
    south_invalid[:-1, :] = ~valid[1:, :]
    west_invalid = np.zeros_like(valid, dtype=bool)
    west_invalid[:, 1:] = ~valid[:, :-1]
    east_invalid = np.zeros_like(valid, dtype=bool)
    east_invalid[:, :-1] = ~valid[:, 1:]
    boundary |= valid & (north_invalid | south_invalid | west_invalid | east_invalid)
    return boundary


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


def relax_rem_downward(
    rem_prior_da: xr.DataArray,
    support_mask_da: xr.DataArray,
    fac_hint_da: xr.DataArray | None = None,
    *,
    base_fidelity: float = 0.25,
    fac_hint_scale: float = 4.0,
    fix_boundary_to_prior: bool = True,
    max_iter: int = 500,
    tol: float = 1e-3,
    omega: float = 1.0,
    return_info: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, RemSagInfo]:
    """Projected downward-only membrane relaxation in REM space.

    Parameters
    ----------
    rem_prior_da:
        Prior REM acting as an upper bound.
    support_mask_da:
        Hard support pixels where water is known to be at the surface;
        these are pinned to zero REM.
    fac_hint_da:
        Optional soft hint raster. Higher values reduce fidelity to the prior
        and therefore allow more downward sag.
    base_fidelity:
        Weight keeping the relaxed REM close to the prior.
    fac_hint_scale:
        How strongly the FAC hint reduces prior fidelity.
    fix_boundary_to_prior:
        Keep the valid AOI boundary fixed at the prior REM.
    max_iter, tol, omega:
        Iteration controls for projected Jacobi/SOR-style relaxation.
    """
    if not (0.0 < omega <= 1.0):
        raise ValueError("omega must be in (0, 1]")

    aligned = [rem_prior_da, support_mask_da]
    if fac_hint_da is not None:
        aligned.append(fac_hint_da)
    aligned = xr.align(*aligned, join="exact")

    prior_da = aligned[0]
    support_da = aligned[1]
    hint_da = aligned[2] if fac_hint_da is not None else None

    prior = np.asarray(prior_da.values, dtype=np.float64)
    if prior.ndim != 2:
        raise ValueError("rem_prior_da must be 2D")
    prior = np.where(np.isfinite(prior), np.maximum(prior, 0.0), np.nan)
    valid = np.isfinite(prior)

    support = np.asarray(support_da.values).astype(bool) & valid
    boundary = _boundary_mask(valid) if fix_boundary_to_prior else np.zeros_like(valid, dtype=bool)
    fixed = support | boundary

    fidelity = np.full(prior.shape, float(base_fidelity), dtype=np.float64)
    if hint_da is not None:
        hint = np.asarray(hint_da.values, dtype=np.float64)
        hint = np.where(np.isfinite(hint) & valid, np.maximum(hint, 0.0), 0.0)
        hint_max = float(np.nanmax(hint)) if np.isfinite(hint).any() else 0.0
        if hint_max > 0.0:
            hint = hint / hint_max
        fidelity = fidelity / (1.0 + float(fac_hint_scale) * hint)

    current = prior.copy()
    current[support] = 0.0

    max_change = np.inf
    updatable = valid & ~fixed
    for it in range(1, int(max_iter) + 1):
        nbr_sum, nbr_w = _neighbor_sum_and_weight(current, valid)
        denom = fidelity + nbr_w
        target = current.copy()
        ok = updatable & (denom > 1e-12)
        target[ok] = (fidelity[ok] * prior[ok] + nbr_sum[ok]) / denom[ok]
        candidate = current.copy()
        candidate[ok] = (1.0 - omega) * current[ok] + omega * target[ok]
        candidate[ok] = np.clip(candidate[ok], 0.0, prior[ok])
        candidate[support] = 0.0
        if fix_boundary_to_prior:
            candidate[boundary] = prior[boundary]
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
        coords=prior_da.coords,
        dims=prior_da.dims,
        name="relaxed_rem",
        attrs={
            "long_name": "downward_relaxed_relative_elevation_model",
            "units": "m",
        },
    )
    if hasattr(prior_da, "rio"):
        out = out.rio.write_crs(prior_da.rio.crs)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
        out = out.rio.write_nodata(np.nan)

    if not return_info:
        return out
    info = RemSagInfo(
        iterations=it,
        max_change=float(max_change),
        n_support=int(support.sum()),
        n_boundary_fixed=int(boundary.sum()),
    )
    return out, info

