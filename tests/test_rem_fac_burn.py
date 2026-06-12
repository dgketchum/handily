"""Burn-stage base-elevation sampling: FAC snap and along-reach smoothing."""

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString

from handily.rem_fac import _attach_fac_strip_elevations, _snap_points_to_fac_max


def _grid_da(arr, res=10.0):
    ny, nx = arr.shape
    x = np.arange(nx) * res + res / 2
    y = (np.arange(ny) * res + res / 2)[::-1]
    da = xr.DataArray(arr.astype(np.float64), coords={"y": y, "x": x}, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:5070")
    return da.rio.set_spatial_dims(x_dim="x", y_dim="y")


def _strips(base_pts, stream_id=1):
    rows = []
    for i, (x, y) in enumerate(base_pts):
        rows.append(
            {
                "stream_id": stream_id,
                "s_m": float(i * 50.0),
                "hit_type": "edge",
                "base_x": x,
                "base_y": y,
                "endpoint_x": x + 100.0,
                "endpoint_y": y,
                "geometry": LineString([(x, y), (x + 100.0, y)]),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:5070")


@pytest.fixture
def channel_grids():
    # 7x7 10m grids: channel runs down column 3 (x=35), banks 5m higher.
    dem = np.full((7, 7), 105.0)
    dem[:, 3] = 100.0
    fac = np.ones((7, 7))
    fac[:, 3] = 1000.0
    return _grid_da(dem), _grid_da(fac)


def test_snap_moves_offset_point_to_channel_cell(channel_grids):
    _, fac_da = channel_grids
    # point one cell off-channel (column 4) snaps to column 3 center
    snapped = _snap_points_to_fac_max(np.array([[45.0, 35.0]]), fac_da, 2)
    assert snapped[0, 0] == pytest.approx(35.0)


def test_snap_finds_max_anywhere_in_window():
    # single FAC peak at the window's bottom-right corner — its flattened
    # index exceeds one window row, which a (N, k) reshape would miss
    fac = np.ones((7, 7))
    fac[5, 5] = 1000.0
    # point at cell (3, 3) = (35, 35); peak cell (5, 5) = (55, 15)
    snapped = _snap_points_to_fac_max(np.array([[35.0, 35.0]]), _grid_da(fac), 2)
    assert snapped[0].tolist() == pytest.approx([55.0, 15.0])


def test_snap_outside_grid_returns_point_unchanged(channel_grids):
    _, fac_da = channel_grids
    pt = np.array([[1e6, 1e6]])
    snapped = _snap_points_to_fac_max(pt, fac_da, 2)
    np.testing.assert_allclose(snapped, pt)


def test_base_elev_snaps_to_channel_bottom(channel_grids):
    dem_da, fac_da = channel_grids
    # stations one cell off-channel sample bank (105) without snapping,
    # channel (100) with snapping
    strips = _strips([(45.0, 55.0), (45.0, 45.0), (45.0, 35.0)])
    plain = _attach_fac_strip_elevations(strips, dem_da)
    assert plain["base_elev_m"].to_numpy() == pytest.approx([105.0] * 3)
    snapped = _attach_fac_strip_elevations(
        strips, dem_da, fac_da=fac_da, base_fac_snap_cells=2
    )
    assert snapped["base_elev_m"].to_numpy() == pytest.approx([100.0] * 3)


def test_station_smoothing_suppresses_spike(channel_grids):
    dem_da, fac_da = channel_grids
    # five on-channel stations; spike the middle one by raising its DEM cell
    dem = dem_da.values.copy()
    dem[3, 3] = 104.0
    dem_da = _grid_da(dem)
    pts = [(35.0, 65.0 - 10.0 * i) for i in range(5)]
    strips = _strips(pts)
    raw = _attach_fac_strip_elevations(
        strips, dem_da, fac_da=fac_da, base_fac_snap_cells=1
    )
    assert raw["base_elev_m"].max() == pytest.approx(104.0)
    smooth = _attach_fac_strip_elevations(
        strips,
        dem_da,
        fac_da=fac_da,
        base_fac_snap_cells=1,
        base_smooth_stations=5,
    )
    assert smooth["base_elev_m"].max() == pytest.approx(100.0)
