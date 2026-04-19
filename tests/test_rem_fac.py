import geopandas as gpd
import numpy as np
import geopandas as gpd
import rioxarray  # noqa: F401
import xarray as xr

from handily.rem_fac import (
    build_fac_wedges,
    build_orientation_field,
    fill_sparse_sections_gaussian,
    fill_sparse_sections_idw,
    fill_sparse_sections_nearest,
    generate_fac_strips,
    rasterize_sparse_sections_20m,
    sample_dem_to_grid,
)


def _synthetic_dem() -> xr.DataArray:
    x = np.arange(0.5, 100.0, 1.0)
    y = np.arange(99.5, -0.5, -1.0)
    xx, yy = np.meshgrid(x, y)
    arr = yy.copy()
    da = xr.DataArray(arr, coords={"y": y, "x": x}, dims=("y", "x"), name="dem")
    da = da.rio.write_crs("EPSG:5070")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da


def _synthetic_streams() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "reach_id": [0, 1],
            "stream_id": [0, 1],
            "strahler": [2, 2],
            "geometry": [
                gpd.GeoSeries.from_wkt(["LINESTRING (30 10, 30 90)"]).iloc[0],
                gpd.GeoSeries.from_wkt(["LINESTRING (70 10, 70 90)"]).iloc[0],
            ],
        },
        geometry="geometry",
        crs="EPSG:5070",
    )


def test_generate_fac_strips_hits_stream_and_edge():
    dem_da = _synthetic_dem()
    streams = _synthetic_streams()
    field = build_orientation_field(dem_da, coarse_res_m=5.0, smooth_sigma_m=20.0)
    strips = generate_fac_strips(
        streams,
        dem_da,
        field=field,
        station_spacing_m=20.0,
        tangent_step_m=5.0,
        min_hit_dist_m=1.0,
    )
    assert not strips.empty

    reach0 = strips.loc[strips["reach_id"] == 0]
    assert (reach0["hit_type"] == "interreach").any()
    assert (reach0["hit_type"] == "edge").any()
    assert 1 in set(reach0.loc[reach0["hit_type"] == "interreach", "target_reach_id"])


def test_build_fac_wedges_keeps_same_target_runs():
    dem_da = _synthetic_dem()
    streams = _synthetic_streams()
    field = build_orientation_field(dem_da, coarse_res_m=5.0, smooth_sigma_m=20.0)
    strips = generate_fac_strips(
        streams,
        dem_da,
        field=field,
        station_spacing_m=20.0,
        tangent_step_m=5.0,
        min_hit_dist_m=1.0,
    )
    wedges = build_fac_wedges(strips)
    assert not wedges.empty
    assert set(wedges["hit_type"]).issubset({"interreach", "edge"})


def test_rasterize_sparse_sections_20m_burns_line_pixels():
    dem_da = _synthetic_dem()
    strips = gpd.GeoDataFrame(
        {
            "base_elev_m": [100.0],
            "endpoint_elev_m": [120.0],
            "geometry": [gpd.GeoSeries.from_wkt(["LINESTRING (10 50, 90 50)"]).iloc[0]],
        },
        geometry="geometry",
        crs="EPSG:5070",
    )
    ws_da, count_da = rasterize_sparse_sections_20m(strips, dem_da, res_m=20.0)
    assert int(np.isfinite(ws_da.values).sum()) > 0
    assert float(np.nanmax(count_da.values)) >= 1.0


def test_fill_sparse_sections_nearest_fills_grid():
    dem_da = _synthetic_dem()
    strips = gpd.GeoDataFrame(
        {
            "base_elev_m": [100.0],
            "endpoint_elev_m": [120.0],
            "geometry": [gpd.GeoSeries.from_wkt(["LINESTRING (10 50, 90 50)"]).iloc[0]],
        },
        geometry="geometry",
        crs="EPSG:5070",
    )
    sparse_da, _ = rasterize_sparse_sections_20m(strips, dem_da, res_m=20.0)
    dem20_da = sample_dem_to_grid(dem_da, sparse_da.x.values, sparse_da.y.values)
    filled_da, dist_da, rem_da = fill_sparse_sections_nearest(sparse_da, dem20_da)
    assert int(np.isfinite(filled_da.values).sum()) == filled_da.values.size
    assert float(np.nanmax(dist_da.values)) > 0.0
    assert int(np.isfinite(rem_da.values).sum()) == rem_da.values.size


def test_fill_sparse_sections_gaussian_fills_grid():
    dem_da = _synthetic_dem()
    strips = gpd.GeoDataFrame(
        {
            "base_elev_m": [100.0],
            "endpoint_elev_m": [120.0],
            "geometry": [gpd.GeoSeries.from_wkt(["LINESTRING (10 50, 90 50)"]).iloc[0]],
        },
        geometry="geometry",
        crs="EPSG:5070",
    )
    sparse_da, count_da = rasterize_sparse_sections_20m(strips, dem_da, res_m=20.0)
    dem20_da = sample_dem_to_grid(dem_da, sparse_da.x.values, sparse_da.y.values)
    filled_da, rem_da = fill_sparse_sections_gaussian(
        sparse_da,
        count_da,
        dem20_da,
        sigma_px=2.0,
    )
    assert int(np.isfinite(filled_da.values).sum()) == filled_da.values.size
    assert int(np.isfinite(rem_da.values).sum()) == rem_da.values.size


def test_fill_sparse_sections_idw_fills_within_radius():
    dem_da = _synthetic_dem()
    strips = gpd.GeoDataFrame(
        {
            "base_elev_m": [100.0],
            "endpoint_elev_m": [120.0],
            "geometry": [gpd.GeoSeries.from_wkt(["LINESTRING (10 50, 90 50)"]).iloc[0]],
        },
        geometry="geometry",
        crs="EPSG:5070",
    )
    sparse_da, count_da = rasterize_sparse_sections_20m(strips, dem_da, res_m=20.0)
    dem20_da = sample_dem_to_grid(dem_da, sparse_da.x.values, sparse_da.y.values)
    filled_da, rem_da = fill_sparse_sections_idw(
        sparse_da,
        count_da,
        dem20_da,
        radius_m=60.0,
        power=1.0,
    )
    assert int(np.isfinite(filled_da.values).sum()) > int(np.isfinite(sparse_da.values).sum())
    assert int(np.isfinite(rem_da.values).sum()) == int(np.isfinite(filled_da.values).sum())
