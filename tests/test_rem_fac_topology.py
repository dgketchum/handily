import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import LineString

from handily.rem_fac_topology import (
    build_fac_topology,
    build_fac_topology_pin_weights,
    estimate_reach_seed_strength,
    propagate_upstream_wet_influence,
    rasterize_reach_weights_max,
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


def test_build_fac_topology_pin_weights_rasterizes_nonzero_weights():
    streams = _toy_streams()
    elev = _grid(np.tile(np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64), (3, 1)))
    ndvi = _grid(np.array([
        [0.0, 0.0, 0.8, 0.8, 0.8],
        [0.0, 0.0, 0.8, 0.8, 0.8],
        [0.0, 0.0, 0.8, 0.8, 0.8],
    ], dtype=np.float64))
    reaches, pin = build_fac_topology_pin_weights(
        streams,
        elev,
        ndvi,
        ndvi,
        support_da=None,
        ndvi_sample_spacing_m=0.5,
        ndvi_mid=0.35,
        ndvi_scale=0.06,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
        strahler_distance_scale=0.0,
        raster_sample_step_m=0.25,
    )
    assert float(np.nanmax(reaches["topo_pin_weight"])) > 0.9
    assert np.isfinite(pin.values).any()
    assert float(np.nanmax(pin.values)) > 0.9
