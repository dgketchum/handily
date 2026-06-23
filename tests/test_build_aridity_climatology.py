"""Unit tests for utils/build_aridity_climatology.py (gridMET reader).

Covers the correctness-critical steps: daily-layer summation -> annual total, the
static-vs-sporadic NaN guard (ocean land mask propagates, per-day gaps raise), and the
multi-year mean + UNEP aridity index AI = P / |PET| with georeferenced GeoTIFF output.

gridMET arrives as per-year netCDF with var `precipitation_amount` (mm) /
`potential_evapotranspiration` (mm, +), dims (day, lat, lon), EPSG:4326. We fabricate
tiny synthetic years with xarray to_netcdf (engine netcdf4; h5netcdf is NOT installed).
"""

import importlib.util
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

_MOD = Path(__file__).resolve().parents[1] / "utils" / "build_aridity_climatology.py"
_spec = importlib.util.spec_from_file_location("build_aridity_climatology", _MOD)
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)

# 2x2 gridMET-like grid: lon ascending, lat descending (north-up).
LON = np.array([-110.0, -109.0])
LAT = np.array([40.0, 39.0])


def _write_nc(path, var, day_arrays):
    """Write a synthetic gridMET-style netCDF: var(day, lat, lon) with lon/lat coords.

    day_arrays: list of 2x2 float arrays, one per day.
    """
    stack = np.stack([np.asarray(a, dtype=np.float64) for a in day_arrays])
    days = np.arange(stack.shape[0])
    da = xr.DataArray(
        stack,
        dims=("day", "lat", "lon"),
        coords={"day": days, "lat": LAT, "lon": LON},
        name=var,
    )
    da.to_dataset().to_netcdf(path, engine="netcdf4")


def test_read_year_returns_day_lat_lon_float64(tmp_path):
    p = tmp_path / "pr.nc"
    _write_nc(p, ar.VAR_P, [np.full((2, 2), 1.0), np.full((2, 2), 2.0)])
    arr = ar.read_year(str(p), ar.VAR_P)
    assert arr.shape == (2, 2, 2)  # (day, lat, lon)
    assert arr.dtype == np.float64
    assert np.allclose(arr[0], 1.0) and np.allclose(arr[1], 2.0)


def test_annual_total_sums_daily_layers():
    stack = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 2.0), np.full((2, 2), 0.5)])
    assert np.allclose(ar.annual_total(stack), 3.5)


def test_annual_total_static_ocean_nan_ok_but_sporadic_raises():
    # One static "ocean" cell, NaN on every day -> propagates (never fabricated).
    ocean = np.array([[True, False], [False, False]])
    d0 = np.where(ocean, np.nan, 1.0)
    d1 = np.where(ocean, np.nan, 2.0)
    out = ar.annual_total(np.stack([d0, d1]))
    assert np.isnan(out[0, 0])  # ocean stays NaN
    assert out[1, 1] == 3.0  # land summed

    # A different cell NaN on a later day = per-day-varying gap -> corruption, raise.
    other = np.array([[False, True], [False, False]])
    d1_bad = np.where(other, np.nan, 2.0)
    try:
        ar.annual_total(np.stack([d0, d1_bad]))
        raise AssertionError("expected ValueError on non-static NaN mask")
    except ValueError:
        pass


def test_build_multiyear_mean_and_aridity_index(tmp_path):
    gm = tmp_path / "gridmet"
    (gm / "pr_raw").mkdir(parents=True)
    (gm / "pet_raw").mkdir(parents=True)
    # P: 2000 annual = 1+2 = 3 ; 2001 annual = 0+1 = 1 -> mean 2.0
    _write_nc(
        gm / "pr_raw" / "pr_2000.nc",
        ar.VAR_P,
        [np.full((2, 2), 1.0), np.full((2, 2), 2.0)],
    )
    _write_nc(
        gm / "pr_raw" / "pr_2001.nc",
        ar.VAR_P,
        [np.full((2, 2), 0.0), np.full((2, 2), 1.0)],
    )
    # PET (positive, gridMET): 2000 annual = 1+1 = 2 ; 2001 annual = 2+2 = 4 -> mean 3.0
    _write_nc(
        gm / "pet_raw" / "pet_2000.nc",
        ar.VAR_PET,
        [np.full((2, 2), 1.0), np.full((2, 2), 1.0)],
    )
    _write_nc(
        gm / "pet_raw" / "pet_2001.nc",
        ar.VAR_PET,
        [np.full((2, 2), 2.0), np.full((2, 2), 2.0)],
    )

    out = tmp_path / "out"
    paths = ar.build(str(out), gridmet_dir=str(gm), year0=2000, year1=2001)

    with rasterio.open(paths["gridmet_mean_annual_precip_mm.tif"]) as s:
        assert np.allclose(s.read(1), 2.0)
        assert s.crs.to_epsg() == 4326  # georeferenced, lon/lat
    with rasterio.open(paths["gridmet_mean_annual_pet_mm.tif"]) as s:
        assert np.allclose(s.read(1), 3.0)
    with rasterio.open(paths["gridmet_aridity_index.tif"]) as s:
        assert np.allclose(s.read(1), 2.0 / 3.0)  # AI = P / |PET|
