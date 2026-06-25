import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import LineString

from handily.rem_fac import (
    _JUNCTION_TOL,
    _axes_from_bounds,
    _rasterize_sparse_sections_from_arrays,
    _sample_section_pixels,
    _strip_endpoint_arrays,
    _terminal_endpoints,
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


def test_terminal_endpoints_subcell_tolerance():
    # Two reaches whose meeting endpoints are ~0.57 m apart (< _JUNCTION_TOL = 1 m)
    # but round to adjacent grid cells: the cKDTree euclidean test must still
    # join them, so neither shared endpoint is flagged terminal.
    assert _JUNCTION_TOL >= 0.5
    g = [LineString([(0, 0), (0, 100)]), LineString([(0.4, 100.4), (50, 200)])]
    ids = [10, 20]
    term_start, term_end = _terminal_endpoints(g, ids)
    assert 10 not in term_end  # reach 10's end meets reach 20's start
    assert 20 not in term_start
    # the genuinely free endpoints remain terminal
    assert 10 in term_start
    assert 20 in term_end


def test_generate_fac_strips_worker_equivalence():
    dem_da = _synthetic_dem()
    streams = _synthetic_streams()
    field = build_orientation_field(dem_da, coarse_res_m=5.0, smooth_sigma_m=20.0)
    params = dict(
        station_spacing_m=10.0,
        tangent_step_m=5.0,
        min_hit_dist_m=1.0,
        halo_n=4,
        max_crossing_strip_m=50.0,
    )
    s1 = generate_fac_strips(streams, dem_da, field=field, workers=1, **params)
    s3 = generate_fac_strips(streams, dem_da, field=field, workers=3, **params)
    cols = [
        "reach_id",
        "station_id",
        "side",
        "hit_type",
        "target_reach_id",
        "dist_m",
        "base_x",
        "base_y",
        "endpoint_x",
        "endpoint_y",
    ]
    a = s1[cols].round(6).sort_values(cols).reset_index(drop=True)
    b = s3[cols].round(6).sort_values(cols).reset_index(drop=True)
    assert a.equals(b)


def test_generate_fac_strips_naked_mode_bounded():
    dem_da = _synthetic_dem()
    streams = _synthetic_streams()
    field = build_orientation_field(dem_da, coarse_res_m=5.0, smooth_sigma_m=20.0)
    strips = generate_fac_strips(
        streams,
        dem_da,
        field=field,
        station_spacing_m=10.0,
        tangent_step_m=5.0,
        min_hit_dist_m=1.0,
        naked_fill_m=20.0,
    )
    # naked mode disables edge strips and emits bounded flat anchors instead
    assert (strips["hit_type"] == "edge").sum() == 0
    naked = strips.loc[strips["hit_type"] == "naked"]
    assert not naked.empty
    # every naked strip is clamped to <= naked_fill_m (shorter only near the edge)
    assert float(naked["dist_m"].max()) <= 20.0 + 1e-6


def test_rasterize_sparse_sections_array_matches_per_strip():
    # The fused array burn must be bit-identical to sampling each equivalent
    # 2-vertex LineString through the per-strip reference path.
    dem_da = _synthetic_dem()
    x_vals = dem_da.x.values.astype(np.float64)
    y_vals = dem_da.y.values.astype(np.float64)
    rng = np.random.default_rng(11)
    n = 200
    bx = rng.uniform(5, 95, n)
    by = rng.uniform(5, 95, n)
    ang = rng.uniform(0, 2 * np.pi, n)
    length = rng.uniform(1, 40, n)
    ex = bx + np.cos(ang) * length
    ey = by + np.sin(ang) * length
    be = rng.uniform(100, 120, n)
    ee = be + rng.uniform(-5, 5, n)
    # exercise the keep mask: one NaN elevation, one degenerate zero-length
    ee[3] = np.nan
    ex[7], ey[7] = bx[7], by[7]

    ny, nx = len(y_vals), len(x_vals)
    ref_min = np.full((ny, nx), np.inf)
    ref_cnt = np.zeros((ny, nx))
    for i in range(n):
        if not (np.isfinite(be[i]) and np.isfinite(ee[i])):
            continue
        geom = LineString([(bx[i], by[i]), (ex[i], ey[i])])
        res = _sample_section_pixels(geom, be[i], ee[i], x_vals, y_vals)
        if res is None:
            continue
        r, c, e = res
        np.minimum.at(ref_min, (r, c), e)
        np.add.at(ref_cnt, (r, c), 1.0)
    ref_ws = np.where(ref_cnt > 0, ref_min, np.nan)

    min_arr, cnt = _rasterize_sparse_sections_from_arrays(
        bx, by, ex, ey, be, ee, x_vals, y_vals
    )
    ws = np.where(cnt > 0, min_arr, np.nan)
    assert np.array_equal(ref_cnt, cnt)
    assert np.array_equal(np.isfinite(ref_ws), np.isfinite(ws))
    finite = np.isfinite(ref_ws)
    assert float(np.max(np.abs(ref_ws[finite] - ws[finite]))) == 0.0


def test_rasterize_sparse_sections_array_batch_invariant():
    # The sample-budget batching must not change the result.
    x_vals = np.arange(0.5, 100.0, 1.0)
    y_vals = np.arange(99.5, -0.5, -1.0)
    rng = np.random.default_rng(3)
    n = 150
    bx = rng.uniform(5, 95, n)
    by = rng.uniform(5, 95, n)
    ex = bx + rng.uniform(-30, 30, n)
    ey = by + rng.uniform(-30, 30, n)
    be = rng.uniform(100, 120, n)
    ee = be + rng.uniform(-5, 5, n)
    m1, c1 = _rasterize_sparse_sections_from_arrays(
        bx, by, ex, ey, be, ee, x_vals, y_vals
    )
    m2, c2 = _rasterize_sparse_sections_from_arrays(
        bx, by, ex, ey, be, ee, x_vals, y_vals, sample_budget=64
    )
    assert np.array_equal(c1, c2)
    assert np.array_equal(np.isnan(m1), np.isnan(m2))
    assert np.array_equal(m1[np.isfinite(m1)], m2[np.isfinite(m2)])


def test_strip_endpoint_arrays_reads_columns():
    # The fast path reads the explicit straight base->endpoint columns directly.
    strips = gpd.GeoDataFrame(
        {
            "base_x": [1.0, 5.0],
            "base_y": [2.0, 6.0],
            "endpoint_x": [3.0, 9.0],
            "endpoint_y": [4.0, 10.0],
            "geometry": [LineString([(1, 2), (3, 4)]), LineString([(5, 6), (9, 10)])],
        },
        geometry="geometry",
        crs="EPSG:5070",
    )
    bx, by, ex, ey = _strip_endpoint_arrays(strips)
    assert np.array_equal(bx, [1.0, 5.0])
    assert np.array_equal(by, [2.0, 6.0])
    assert np.array_equal(ex, [3.0, 9.0])
    assert np.array_equal(ey, [4.0, 10.0])


def test_rasterize_sparse_sections_geometry_only_walks_bent_path():
    # Geometry-only callers (no endpoint columns) must walk each LineString's
    # full path -- a bent strip burns along every segment, not just the
    # base->endpoint chord -- and skip empty geometries without raising.
    dem_da = _synthetic_dem()
    bent = LineString([(10, 10), (10, 80), (80, 80)])  # right-angle dogleg
    strips = gpd.GeoDataFrame(
        {
            "base_elev_m": [100.0, 100.0],
            "endpoint_elev_m": [120.0, 120.0],
            "geometry": [bent, LineString()],  # second geometry empty -> skipped
        },
        geometry="geometry",
        crs="EPSG:5070",
    )
    ws_da, count_da = rasterize_sparse_sections_20m(strips, dem_da, res_m=20.0)

    # Reference: only the bent strip, sampled through the per-geometry path on the
    # same 20 m burn axes. The empty geometry contributes nothing.
    x20, y20 = _axes_from_bounds(tuple(dem_da.rio.bounds()), 20.0)
    rows, cols, elevs = _sample_section_pixels(bent, 100.0, 120.0, x20, y20)
    ref_min = np.full((len(y20), len(x20)), np.inf)
    ref_cnt = np.zeros((len(y20), len(x20)))
    np.minimum.at(ref_min, (rows, cols), elevs)
    np.add.at(ref_cnt, (rows, cols), 1.0)
    ref_ws = np.where(ref_cnt > 0, ref_min, np.nan)

    assert np.array_equal(ref_cnt, count_da.values)
    assert np.array_equal(np.isfinite(ref_ws), np.isfinite(ws_da.values))
    finite = np.isfinite(ref_ws)
    assert float(np.max(np.abs(ref_ws[finite] - ws_da.values[finite]))) == 0.0
    # The dogleg's vertical leg burns cells the straight chord would miss.
    chord = LineString([(10, 10), (80, 80)])
    cr, cc, _ = _sample_section_pixels(chord, 100.0, 120.0, x20, y20)
    chord_cells = set(zip(cr.tolist(), cc.tolist()))
    bent_cells = set(zip(rows.tolist(), cols.tolist()))
    assert bent_cells - chord_cells


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
    assert int(np.isfinite(filled_da.values).sum()) > int(
        np.isfinite(sparse_da.values).sum()
    )
    assert int(np.isfinite(rem_da.values).sum()) == int(
        np.isfinite(filled_da.values).sum()
    )
