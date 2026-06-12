import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import LineString

from handily.rem_fac_topology import (
    build_fac_topology,
    estimate_reach_seed_strength,
    propagate_downstream_wet_influence,
    propagate_upstream_wet_influence,
    compute_strahler_from_topology,
    sample_reach_drainage_km2,
)


def _grid(values: np.ndarray) -> xr.DataArray:
    ny, nx = values.shape
    x = np.arange(0.5, nx + 0.5, 1.0)
    y = np.arange(ny - 0.5, -0.5, -1.0)
    da = xr.DataArray(values, coords={"y": y, "x": x}, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:5070")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da


def _toy_streams() -> gpd.GeoDataFrame:
    # Simple chain: 1 -> 2 -> 3 downstream, all endpoints shared exactly.
    geoms = [
        LineString([(0.5, 0.5), (1.5, 0.5)]),
        LineString([(1.5, 0.5), (2.5, 0.5)]),
        LineString([(2.5, 0.5), (3.5, 0.5)]),
    ]
    return gpd.GeoDataFrame(
        {
            "stream_id": [1, 2, 3],
            "strahler": [1, 2, 3],
            "length_m": [1.0, 1.0, 1.0],
            "geometry": geoms,
        },
        geometry="geometry",
        crs="EPSG:5070",
    )


def test_estimate_reach_seed_strength_uses_support_override():
    streams = _toy_streams()
    ndvi = _grid(np.full((3, 5), -0.1, dtype=np.float64))
    support = _grid(np.zeros((3, 5), dtype=np.float64))
    # Fill both cols so interior samples (excluding endpoints) exceed 0.5
    support.values[2, 0] = 1.0
    support.values[2, 1] = 1.0
    seeded = estimate_reach_seed_strength(
        streams,
        ndvi,
        support_da=support,
        sample_spacing_m=0.5,
        support_override=1.0,
    )
    vals = dict(zip(seeded["stream_id"], seeded["seed_strength"]))
    assert vals[1] == 1.0
    assert vals[2] < 0.5
    assert vals[3] < 0.5


def test_propagate_upstream_wet_influence_decays_along_chain():
    streams = _toy_streams()
    elev = _grid(np.tile(np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64), (3, 1)))
    topo = build_fac_topology(streams, elev)
    topo.streams["seed_strength"] = [0.0, 0.0, 1.0]
    weighted = propagate_upstream_wet_influence(
        topo,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
        strahler_distance_scale=0.0,
    )
    by_id = weighted.set_index("stream_id")
    assert np.isclose(by_id.loc[3, "topo_pin_weight"], 1.0)
    assert 0.0 < by_id.loc[2, "topo_pin_weight"] < by_id.loc[3, "topo_pin_weight"]
    assert 0.0 < by_id.loc[1, "topo_pin_weight"] < by_id.loc[2, "topo_pin_weight"]
    assert np.isclose(by_id.loc[2, "topo_dist_to_seed_m"], 1.0)
    assert np.isclose(by_id.loc[1, "topo_dist_to_seed_m"], 2.0)


def test_propagate_downstream_wet_influence_decays_along_chain():
    streams = _toy_streams()
    elev = _grid(np.tile(np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64), (3, 1)))
    topo = build_fac_topology(streams, elev)
    # Wet seed at the upstream-most reach; wetness persists downstream.
    topo.streams["seed_strength"] = [1.0, 0.0, 0.0]
    weighted = propagate_downstream_wet_influence(
        topo,
        distance_scale_m=2.0,
        strahler_distance_scale=0.0,
    )
    by_id = weighted.set_index("stream_id")
    assert np.isclose(by_id.loc[1, "topo_down_weight"], 1.0)
    assert 0.0 < by_id.loc[2, "topo_down_weight"] < by_id.loc[1, "topo_down_weight"]
    assert 0.0 < by_id.loc[3, "topo_down_weight"] < by_id.loc[2, "topo_down_weight"]
    assert np.isclose(by_id.loc[2, "topo_down_dist_m"], 1.0)
    assert np.isclose(by_id.loc[3, "topo_down_dist_m"], 2.0)
    assert int(by_id.loc[3, "topo_down_seed_stream_id"]) == 1


def test_build_fac_topology_orients_by_fac_on_flat_water():
    # Reach 1 is drawn downstream-first; on dead-flat elevation the
    # elevation rule keeps it backwards and the chain severs. FAC
    # strictly increases downstream, so it recovers the link.
    geoms = [
        LineString([(1.5, 0.5), (0.5, 0.5)]),
        LineString([(1.5, 0.5), (2.5, 0.5)]),
    ]
    streams = gpd.GeoDataFrame(
        {"stream_id": [1, 2], "geometry": geoms}, geometry="geometry", crs="EPSG:5070"
    )
    elev = _grid(np.ones((3, 5), dtype=np.float64))
    fac = _grid(np.zeros((3, 5), dtype=np.float64))
    fac.values[2, :] = [1.0, 2.0, 3.0, 4.0, 5.0]
    topo_elev = build_fac_topology(streams, elev)
    assert topo_elev.upstream[2] == ()
    topo = build_fac_topology(streams, elev, fac)
    assert topo.upstream[2] == (1,)
    assert topo.downstream[1] == (2,)


def test_compute_strahler_from_topology_increments_only_at_like_order_meet():
    # Y network: headwaters 1, 2 meet -> 3 increments to order 1; reach 5
    # continues at order 1 (single max contributor) despite headwater
    # tributary 4 joining at the same node. WBT labels (all 9, junk by
    # construction) are preserved as strahler_raw and otherwise ignored.
    geoms = [
        LineString([(0.5, 2.5), (1.5, 1.5)]),
        LineString([(0.5, 0.5), (1.5, 1.5)]),
        LineString([(1.5, 1.5), (3.5, 1.5)]),
        LineString([(2.5, 2.5), (3.5, 1.5)]),
        LineString([(3.5, 1.5), (4.5, 1.5)]),
    ]
    streams = gpd.GeoDataFrame(
        {"stream_id": [1, 2, 3, 4, 5], "strahler": [9, 9, 9, 9, 9], "geometry": geoms},
        geometry="geometry",
        crs="EPSG:5070",
    )
    elev = _grid(np.tile(np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64), (3, 1)))
    topo = build_fac_topology(streams, elev)
    out = compute_strahler_from_topology(topo)
    by_id = out.set_index("stream_id")
    assert list(by_id.loc[[1, 2, 4], "strahler"]) == [0, 0, 0]
    assert int(by_id.loc[3, "strahler"]) == 1
    assert int(by_id.loc[5, "strahler"]) == 1
    assert list(by_id["strahler_raw"]) == [9, 9, 9, 9, 9]


def test_sample_reach_drainage_km2_uses_line_max():
    streams = _toy_streams()
    fac = _grid(np.zeros((3, 5), dtype=np.float64))
    fac.values[2, :] = [1.0, 2.0, 3.0, 4.0, 5.0]
    area = sample_reach_drainage_km2(
        streams, fac, sample_spacing_m=0.5, junction_trim_cells=0.0
    )
    # 1 m2 cells: km2 = cells * 1e-6; each reach picks the max-FAC cell
    # under its line (reach 1 spans x 0.5-1.5 -> 2; reach 3 spans 2.5-3.5 -> 4)
    assert np.isclose(area[0], 2.0e-6)
    assert np.isclose(area[2], 4.0e-6)


def test_sample_reach_drainage_km2_trims_junction_cells():
    # Tributary whose downstream vertex lands on a junction cell carrying
    # the receiving mainstem's accumulation (100 cells); the trimmed max
    # must come from the reach's own cells instead.
    streams = gpd.GeoDataFrame(
        {"stream_id": [1], "geometry": [LineString([(0.5, 0.5), (4.5, 0.5)])]},
        geometry="geometry",
        crs="EPSG:5070",
    )
    fac = _grid(np.zeros((3, 5), dtype=np.float64))
    fac.values[2, :] = [1.0, 2.0, 3.0, 4.0, 100.0]
    contaminated = sample_reach_drainage_km2(
        streams, fac, sample_spacing_m=0.5, junction_trim_cells=0.0
    )
    assert np.isclose(contaminated[0], 100.0e-6)
    trimmed = sample_reach_drainage_km2(
        streams, fac, sample_spacing_m=0.5, junction_trim_cells=1.5
    )
    assert trimmed[0] < 100.0e-6
    assert 3.0e-6 - 1e-12 <= trimmed[0] <= 4.0e-6 + 1e-12


def test_sample_reach_drainage_km2_short_reach_falls_back_to_midpoint():
    # Length 2 m <= 2 * 1.5 m trim -> single midpoint sample at x=1.5 (FAC 2)
    streams = gpd.GeoDataFrame(
        {"stream_id": [1], "geometry": [LineString([(0.5, 0.5), (2.5, 0.5)])]},
        geometry="geometry",
        crs="EPSG:5070",
    )
    fac = _grid(np.zeros((3, 5), dtype=np.float64))
    fac.values[2, :] = [1.0, 2.0, 3.0, 4.0, 100.0]
    area = sample_reach_drainage_km2(
        streams, fac, sample_spacing_m=0.5, junction_trim_cells=1.5
    )
    assert np.isclose(area[0], 2.0e-6)
