import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
import pytest

from handily.rem_surface_relax import (
    ndvi_to_prior_pin_weight,
    ndvi_to_clearance_logistic,
    ndvi_to_clearance,
    relax_water_surface_ndvi_pins,
    relax_water_surface_soft_ceiling,
    relax_water_surface_upward,
    rem_from_water_surface,
    shallow_rem_pin_weight,
    smooth_ndvi_gaussian,
)


def _grid(values: np.ndarray) -> xr.DataArray:
    ny, nx = values.shape
    x = np.arange(0.5, nx + 0.5, 1.0)
    y = np.arange(ny - 0.5, -0.5, -1.0)
    da = xr.DataArray(values, coords={"y": y, "x": x}, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:5070")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da


def test_relax_water_surface_upward_pins_support_and_never_lowers():
    dem = _grid(np.full((5, 5), 10.0, dtype=np.float64))
    prior_ws = _grid(np.full((5, 5), 2.0, dtype=np.float64))
    support = _grid(np.zeros((5, 5), dtype=np.uint8))
    support.values[2, 2] = 1
    relaxed, info = relax_water_surface_upward(prior_ws, dem, support, return_info=True, max_iter=300, tol=1e-5)
    assert relaxed.values[2, 2] == dem.values[2, 2]
    assert np.nanmin(relaxed.values - prior_ws.values) >= -1e-12
    assert np.nanmax(relaxed.values - dem.values) <= 1e-12
    assert relaxed.values[2, 1] > prior_ws.values[2, 1]
    assert info.n_support == 1


def test_relax_water_surface_respects_off_support_clearance():
    dem = _grid(np.full((5, 5), 10.0, dtype=np.float64))
    prior_ws = _grid(np.full((5, 5), 9.95, dtype=np.float64))
    support = _grid(np.zeros((5, 5), dtype=np.uint8))
    support.values[2, 2] = 1
    relaxed = relax_water_surface_upward(
        prior_ws,
        dem,
        support,
        min_clearance_off_support=0.25,
        max_iter=300,
        tol=1e-5,
    )
    off_support = ~support.values.astype(bool)
    assert np.nanmax(relaxed.values[off_support] - (dem.values[off_support] - 0.25)) <= 1e-12
    assert relaxed.values[2, 2] == dem.values[2, 2]


def test_relax_water_surface_accepts_clearance_raster():
    dem = _grid(np.full((5, 5), 10.0, dtype=np.float64))
    prior_ws = _grid(np.full((5, 5), 9.95, dtype=np.float64))
    support = _grid(np.zeros((5, 5), dtype=np.uint8))
    support.values[2, 2] = 1
    clearance = _grid(np.full((5, 5), 0.5, dtype=np.float64))
    clearance.values[:, :2] = 2.0
    relaxed = relax_water_surface_upward(
        prior_ws,
        dem,
        support,
        min_clearance_off_support=clearance,
        max_iter=300,
        tol=1e-5,
    )
    off_support = ~support.values.astype(bool)
    assert np.nanmax(relaxed.values[:, :2][off_support[:, :2]] - 8.0) <= 1e-12
    assert np.nanmax(relaxed.values[:, 2:][off_support[:, 2:]] - 9.5) <= 1e-12
    assert relaxed.values[2, 2] == 10.0


def test_relax_water_surface_rejects_nan_clearance_on_valid_cells():
    dem = _grid(np.full((4, 4), 10.0, dtype=np.float64))
    prior_ws = _grid(np.full((4, 4), 9.0, dtype=np.float64))
    support = _grid(np.zeros((4, 4), dtype=np.uint8))
    clearance = _grid(np.full((4, 4), 0.5, dtype=np.float64))
    clearance.values[1, 1] = np.nan
    with pytest.raises(ValueError, match="clearance raster contains NaN/inf values on valid cells"):
        relax_water_surface_upward(prior_ws, dem, support, min_clearance_off_support=clearance)


def test_relax_water_surface_keeps_boundary_when_requested():
    dem = _grid(np.full((6, 6), 20.0, dtype=np.float64))
    prior_ws = _grid(np.full((6, 6), 5.0, dtype=np.float64))
    support = _grid(np.zeros((6, 6), dtype=np.uint8))
    support.values[3, 3] = 1
    relaxed = relax_water_surface_upward(prior_ws, dem, support, fix_boundary_to_prior=True, max_iter=300, tol=1e-5)
    assert np.allclose(relaxed.values[0, :], 5.0)
    assert np.allclose(relaxed.values[-1, :], 5.0)
    assert np.allclose(relaxed.values[:, 0], 5.0)
    assert np.allclose(relaxed.values[:, -1], 5.0)


def test_fac_hint_allows_more_upward_relaxation():
    dem = _grid(np.full((7, 7), 12.0, dtype=np.float64))
    prior_ws = _grid(np.full((7, 7), 4.0, dtype=np.float64))
    support = _grid(np.zeros((7, 7), dtype=np.uint8))
    support.values[3, 3] = 1
    hint = _grid(np.zeros((7, 7), dtype=np.float64))
    hint.values[3, 1:6] = 1.0
    no_hint = relax_water_surface_upward(
        prior_ws,
        dem,
        support,
        min_clearance_off_support=0.25,
        max_iter=300,
        tol=1e-5,
    )
    with_hint = relax_water_surface_upward(
        prior_ws,
        dem,
        support,
        fac_hint_da=hint,
        min_clearance_off_support=0.25,
        fac_hint_scale=8.0,
        max_iter=300,
        tol=1e-5,
    )
    assert with_hint.values[3, 1] >= no_hint.values[3, 1] - 1e-12
    assert with_hint.values[3, 5] >= no_hint.values[3, 5] - 1e-12


def test_ndvi_to_clearance_maps_sparse_to_large_clearance():
    ndvi = _grid(np.array([[0.8, 0.6, 0.35, 0.1, -0.1]], dtype=np.float64))
    clearance = ndvi_to_clearance(
        ndvi,
        min_clearance=0.1,
        max_clearance=10.0,
        ndvi_dense=0.6,
        ndvi_sparse=0.1,
        gamma=1.0,
    )
    vals = clearance.values[0]
    assert np.isclose(vals[0], 0.1)
    assert np.isclose(vals[1], 0.1)
    assert vals[2] > vals[1]
    assert np.isclose(vals[3], 10.0)
    assert np.isclose(vals[4], 10.0)


def test_ndvi_to_clearance_logistic_is_monotone():
    ndvi = _grid(np.array([[-0.1, 0.1, 0.22, 0.4, 0.8]], dtype=np.float64))
    clearance = ndvi_to_clearance_logistic(
        ndvi,
        min_clearance=0.1,
        max_clearance=10.0,
        ndvi_mid=0.22,
        ndvi_scale=0.08,
    )
    vals = clearance.values[0]
    assert vals[0] > vals[1] > vals[2] > vals[3] > vals[4]
    assert vals[0] <= 10.0 + 1e-12
    assert vals[4] >= 0.1 - 1e-12


def test_ndvi_to_prior_pin_weight_increases_with_ndvi():
    ndvi = _grid(np.array([[-0.1, 0.1, 0.22, 0.4, 0.8]], dtype=np.float64))
    weights = ndvi_to_prior_pin_weight(
        ndvi,
        min_weight=0.0,
        max_weight=2.0,
        ndvi_mid=0.22,
        ndvi_scale=0.08,
    )
    vals = weights.values[0]
    assert vals[0] < vals[1] < vals[2] < vals[3] < vals[4]
    assert vals[0] >= 0.0
    assert vals[4] <= 2.0 + 1e-12


def test_smooth_ndvi_gaussian_preserves_nan_edges_with_weighted_blur():
    ndvi = _grid(np.array([[np.nan, 0.0, 1.0, np.nan]], dtype=np.float64))
    smoothed = smooth_ndvi_gaussian(ndvi, sigma_px=1.0)
    vals = smoothed.values[0]
    assert np.isfinite(vals[1])
    assert np.isfinite(vals[2])
    assert 0.0 < vals[1] < 1.0
    assert 0.0 < vals[2] < 1.0


def test_shallow_rem_pin_weight_only_targets_shallow_exceedance():
    rem_prior = _grid(np.array([[0.1, 1.0, 5.0, 12.0]], dtype=np.float64))
    clearance = _grid(np.array([[10.0, 5.0, 5.0, 5.0]], dtype=np.float64))
    weights = shallow_rem_pin_weight(
        rem_prior,
        clearance,
        tolerance_m=1.0,
        max_weight=2.0,
        exceedance_scale_m=2.0,
    )
    vals = weights.values[0]
    assert vals[0] > vals[1] > 0.0
    assert vals[2] == 0.0
    assert vals[3] == 0.0


def test_relax_water_surface_soft_ceiling_sags_shallow_pixels_without_zeroing_off_support():
    dem = _grid(np.full((5, 5), 10.0, dtype=np.float64))
    prior_ws = _grid(np.full((5, 5), 9.7, dtype=np.float64))
    support = _grid(np.zeros((5, 5), dtype=np.uint8))
    support.values[2, 2] = 1
    soft_target_ws = _grid(np.full((5, 5), 7.0, dtype=np.float64))
    pin_weight = _grid(np.zeros((5, 5), dtype=np.float64))
    pin_weight.values[:, :2] = 2.0
    relaxed = relax_water_surface_soft_ceiling(
        prior_ws,
        dem,
        support,
        soft_target_ws,
        pin_weight,
        min_clearance_off_support=0.1,
        max_iter=300,
        tol=1e-5,
    )
    rem = rem_from_water_surface(dem, relaxed)
    off_support = ~support.values.astype(bool)
    assert np.nanmin(rem.values[off_support]) >= 0.1 - 1e-9
    assert np.nanmean(relaxed.values[:, :2]) < np.nanmean(relaxed.values[:, 3:])
    assert relaxed.values[2, 2] == 10.0


def test_relax_water_surface_ndvi_pins_sags_more_where_pin_weight_is_low():
    dem = _grid(np.full((5, 5), 10.0, dtype=np.float64))
    prior_ws = _grid(np.array([
        [9.0, 9.0, 8.0, 7.0, 7.0],
        [9.0, 9.0, 8.0, 7.0, 7.0],
        [9.0, 9.0, 8.0, 7.0, 7.0],
        [9.0, 9.0, 8.0, 7.0, 7.0],
        [9.0, 9.0, 8.0, 7.0, 7.0],
    ], dtype=np.float64))
    support = _grid(np.zeros((5, 5), dtype=np.uint8))
    support.values[2, 2] = 1
    pin = _grid(np.array([
        [2.0, 2.0, 0.5, 0.0, 0.0],
        [2.0, 2.0, 0.5, 0.0, 0.0],
        [2.0, 2.0, 0.5, 0.0, 0.0],
        [2.0, 2.0, 0.5, 0.0, 0.0],
        [2.0, 2.0, 0.5, 0.0, 0.0],
    ], dtype=np.float64))
    relaxed = relax_water_surface_ndvi_pins(
        prior_ws,
        dem,
        support,
        pin,
        min_clearance_off_support=0.1,
        base_fidelity=0.1,
        smoothness_weight=2.0,
        max_iter=300,
        tol=1e-5,
    )
    rem = rem_from_water_surface(dem, relaxed)
    off_support = ~support.values.astype(bool)
    assert np.nanmin(rem.values[off_support]) >= 0.1 - 1e-9
    assert np.nanmean(relaxed.values[:, :2]) > np.nanmean(relaxed.values[:, 3:])
    assert relaxed.values[2, 2] == 10.0


def test_rem_from_water_surface():
    dem = _grid(np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float64))
    ws = _grid(np.array([[9.0, 8.0], [12.5, np.nan]], dtype=np.float64))
    rem = rem_from_water_surface(dem, ws)
    assert np.isclose(rem.values[0, 0], 1.0)
    assert np.isclose(rem.values[0, 1], 3.0)
    assert np.isclose(rem.values[1, 0], 0.0)
    assert np.isnan(rem.values[1, 1])
