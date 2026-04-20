import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import LineString

from handily.rem_fac_head import build_channel_heads, solve_channel_heads
from handily.rem_fac_topology import (
    build_fac_topology,
    estimate_reach_seed_strength,
    propagate_upstream_wet_influence,
)
from handily.rem_fac import _attach_fac_strip_head_elevations


def _grid(values: np.ndarray) -> xr.DataArray:
    ny, nx = values.shape
    x = np.arange(0.5, nx + 0.5, 1.0)
    y = np.arange(ny - 0.5, -0.5, -1.0)
    da = xr.DataArray(values, coords={"y": y, "x": x}, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:5070")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da


def _chain_streams() -> gpd.GeoDataFrame:
    """Three-reach chain: 1 -> 2 -> 3, descending elevation left to right."""
    geoms = [
        LineString([(0.5, 0.5), (1.5, 0.5)]),
        LineString([(1.5, 0.5), (2.5, 0.5)]),
        LineString([(2.5, 0.5), (3.5, 0.5)]),
    ]
    return gpd.GeoDataFrame(
        {
            "stream_id": [1, 2, 3],
            "reach_id": [10, 20, 30],
            "strahler": [1, 1, 2],
            "length_m": [1.0, 1.0, 1.0],
            "geometry": geoms,
        },
        geometry="geometry",
        crs="EPSG:5070",
    )


def _sloped_elev() -> xr.DataArray:
    """Elevation grid sloping from left (high) to right (low)."""
    return _grid(np.tile(np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64), (3, 1)))


def _seeded_topo(support_reach=None):
    """Build topology with seed estimation and propagation.

    Optionally place hard support on a specific reach.
    """
    streams = _chain_streams()
    elev = _sloped_elev()
    topo = build_fac_topology(streams, elev)

    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    support = None
    if support_reach is not None:
        support = _grid(np.zeros((3, 5), dtype=np.float64))
        # Fill support across interior of the target reach so that
        # interior-only sampling (excluding shared endpoints) still
        # yields seed_support_fraction > threshold.
        col_ranges = {1: [0, 1], 2: [1, 2, 3], 3: [2, 3, 4]}
        for c in col_ranges[support_reach]:
            support.values[2, c] = 1.0

    topo.streams = estimate_reach_seed_strength(
        topo.streams, ndvi, support_da=support, sample_spacing_m=0.5
    )
    topo.streams = propagate_upstream_wet_influence(
        topo,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
        strahler_distance_scale=0.0,
    )
    return topo


def test_solve_channel_heads_support_hard_pinned():
    """Reaches with sufficient support fraction are hard-pinned at bed."""
    topo = _seeded_topo(support_reach=3)
    heads = solve_channel_heads(topo, d_min_off_support_m=0.5)
    by_id = heads.set_index("stream_id")
    # Reach 3 has support_fraction > 0.25 -> hard pin -> h = z_mid exactly
    assert np.isclose(by_id.loc[3, "head_depth_m"], 0.0)


def test_support_at_shared_endpoint_does_not_hard_pin_neighbor():
    """Support pixel at a shared endpoint must not hard-pin the adjacent reach."""
    streams = _chain_streams()
    elev = _sloped_elev()
    topo = build_fac_topology(streams, elev)

    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    # Place support only at the shared vertex between reaches 2 and 3
    # (col 2 = x=2.5, which is reach 2 downstream / reach 3 upstream)
    support = _grid(np.zeros((3, 5), dtype=np.float64))
    support.values[2, 2] = 1.0

    topo.streams = estimate_reach_seed_strength(
        topo.streams, ndvi, support_da=support, sample_spacing_m=0.5
    )
    # Neither reach should have seed_support_fraction > 0 because the
    # only support pixel is at an endpoint excluded from interior sampling.
    fracs = topo.streams.set_index("stream_id")["seed_support_fraction"]
    assert fracs.loc[2] == 0.0, f"reach 2 fraction {fracs.loc[2]} should be 0"
    assert fracs.loc[3] == 0.0, f"reach 3 fraction {fracs.loc[3]} should be 0"


def test_short_reach_gets_soft_anchor_not_hard_pin():
    """A sub-spacing reach cannot be hard-pinned — fraction stays 0, hit is soft."""
    geoms = [
        LineString([(0.5, 0.5), (1.5, 0.5)]),
        LineString([(1.5, 0.5), (2.5, 0.5)]),
        # Reach 3 is very short (0.3 m) — only 2 endpoint samples
        LineString([(2.5, 0.5), (2.8, 0.5)]),
    ]
    streams = gpd.GeoDataFrame(
        {
            "stream_id": [1, 2, 3],
            "reach_id": [10, 20, 30],
            "strahler": [1, 1, 2],
            "length_m": [1.0, 1.0, 0.3],
            "geometry": geoms,
        },
        geometry="geometry",
        crs="EPSG:5070",
    )
    elev = _sloped_elev()
    topo = build_fac_topology(streams, elev)

    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    # Fill support across cols 2-3 covering reach 3's endpoints
    support = _grid(np.zeros((3, 5), dtype=np.float64))
    support.values[2, 2] = 1.0
    support.values[2, 3] = 1.0

    topo.streams = estimate_reach_seed_strength(
        topo.streams, ndvi, support_da=support, sample_spacing_m=0.5
    )
    by_id = topo.streams.set_index("stream_id")
    # No interior samples → fraction stays 0 → not hard-pinnable
    assert by_id.loc[3, "seed_support_fraction"] == 0.0
    # But full-line hit still fires → soft anchor via seed_strength override
    assert bool(by_id.loc[3, "seed_support_hit"]) is True
    assert by_id.loc[3, "seed_strength"] == 1.0


def test_solve_channel_heads_hard_pin_resists_neighbor_smoothing():
    """Hard-pinned reaches must stay at bed even with strong smoothing."""
    topo = _seeded_topo(support_reach=3)
    # Extreme smoothness that would pull a soft reach well below bed
    heads = solve_channel_heads(
        topo,
        d_min_off_support_m=0.5,
        smoothness_weight=100.0,
        wet_anchor_strength=0.01,
    )
    by_id = heads.set_index("stream_id")
    assert np.isclose(by_id.loc[3, "head_depth_m"], 0.0)


def test_solve_channel_heads_dry_headwaters_detach():
    """Dry upstream reaches should have h well below bed."""
    topo = _seeded_topo(support_reach=None)
    heads = solve_channel_heads(topo, d_min_off_support_m=0.5)
    by_id = heads.set_index("stream_id")
    # All dry -> topo_pin_weight near zero -> clearance near d_min
    # (tiny residual sigmoid weight reduces clearance slightly from 0.5)
    assert (by_id["head_depth_m"] >= 0.45).all()


def test_solve_channel_heads_max_slope():
    """Head rise should not exceed max_hydraulic_slope * connection_length."""
    streams = _chain_streams()
    steep = _grid(
        np.tile(
            np.array([400.0, 300.0, 200.0, 100.0, 0.0], dtype=np.float64),
            (3, 1),
        )
    )
    topo = build_fac_topology(streams, steep)
    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    topo.streams = estimate_reach_seed_strength(
        topo.streams, ndvi, sample_spacing_m=0.5
    )
    topo.streams = propagate_upstream_wet_influence(
        topo,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
        strahler_distance_scale=0.0,
    )
    s_max = 0.5
    heads = solve_channel_heads(
        topo, d_min_off_support_m=0.5, max_hydraulic_slope=s_max
    )
    by_id = heads.set_index("stream_id")
    h1 = by_id.loc[1, "channel_head_m"]
    h2 = by_id.loc[2, "channel_head_m"]
    h3 = by_id.loc[3, "channel_head_m"]
    L12 = (by_id.loc[1, "length_m"] + by_id.loc[2, "length_m"]) / 2.0
    L23 = (by_id.loc[2, "length_m"] + by_id.loc[3, "length_m"]) / 2.0
    assert h1 - h2 <= s_max * L12 + 1e-6
    assert h2 - h3 <= s_max * L23 + 1e-6


def test_solve_channel_heads_propagated_weight_decays_upstream():
    """Upstream reaches should detach more where propagated weight is low."""
    topo = _seeded_topo(support_reach=3)
    heads = solve_channel_heads(topo, d_min_off_support_m=0.5)
    by_id = heads.set_index("stream_id")
    # Reach 3 (wet) should be shallower than reach 1 (dry headwater)
    assert by_id.loc[1, "head_depth_m"] > by_id.loc[3, "head_depth_m"]


def test_build_channel_heads_preserves_reach_id():
    """build_channel_heads should carry reach_id from input streams."""
    streams = _chain_streams()
    elev = _sloped_elev()
    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    heads = build_channel_heads(
        streams,
        elev,
        ndvi,
        sample_spacing_m=0.5,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
    )
    assert "reach_id" in heads.columns
    assert set(heads["reach_id"].astype(int)) == {10, 20, 30}


def test_build_channel_heads_includes_propagated_weight():
    """build_channel_heads should produce topo_pin_weight from propagation."""
    streams = _chain_streams()
    elev = _sloped_elev()
    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    heads = build_channel_heads(
        streams,
        elev,
        ndvi,
        sample_spacing_m=0.5,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
    )
    assert "topo_pin_weight" in heads.columns


def test_attach_fac_strip_head_elevations_overrides_base():
    """Head-solved base elevation should replace DEM-based values."""
    strips = gpd.GeoDataFrame(
        {
            "stream_id": [1, 1, 2],
            "reach_id": [10, 10, 20],
            "hit_type": ["edge", "interreach", "edge"],
            "target_reach_id": [-1, 20, -1],
            "base_elev_m": [100.0, 100.0, 90.0],
            "endpoint_elev_m": [100.0, 90.0, 90.0],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 0), (2, 0)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:5070",
    )
    heads = gpd.GeoDataFrame(
        {
            "stream_id": [1, 2],
            "reach_id": [10, 20],
            "channel_head_m": [95.0, 88.0],
            "geometry": [
                LineString([(0.5, 0.5), (1.5, 0.5)]),
                LineString([(1.5, 0.5), (2.5, 0.5)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:5070",
    )
    result = _attach_fac_strip_head_elevations(strips, heads)
    assert np.isclose(result.iloc[0]["base_elev_m"], 95.0)
    assert np.isclose(result.iloc[1]["base_elev_m"], 95.0)
    # Interreach endpoint (target reach 20) should be 88.0
    assert np.isclose(result.iloc[1]["endpoint_elev_m"], 88.0)
    # Edge endpoint should equal base
    assert np.isclose(result.iloc[0]["endpoint_elev_m"], 95.0)
    assert np.isclose(result.iloc[2]["endpoint_elev_m"], 88.0)
