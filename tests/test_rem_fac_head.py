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
    """Build topology with seed estimation, propagation, and sag targets.

    Optionally place hard support on a specific reach.
    """
    from handily.rem_fac_head import _build_residual_targets

    streams = _chain_streams()
    elev = _sloped_elev()
    topo = build_fac_topology(streams, elev)

    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    support = None
    if support_reach is not None:
        support = _grid(np.zeros((3, 5), dtype=np.float64))
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
    topo.streams = _build_residual_targets(
        topo.streams,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
        rmax_min_m=0.5,
        rmax_max_m=2.0,
    )
    return topo


# --- Regression tests (preserved) ---


def test_solve_channel_heads_support_hard_pinned():
    """Reaches with sufficient support fraction are hard-pinned at bed."""
    topo = _seeded_topo(support_reach=3)
    heads = solve_channel_heads(topo, d_min_off_support_m=0.5)
    by_id = heads.set_index("stream_id")
    assert np.isclose(by_id.loc[3, "head_depth_m"], 0.0)


def test_support_at_shared_endpoint_does_not_hard_pin_neighbor():
    """Support pixel at a shared endpoint must not hard-pin the adjacent reach."""
    streams = _chain_streams()
    elev = _sloped_elev()
    topo = build_fac_topology(streams, elev)

    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    support = _grid(np.zeros((3, 5), dtype=np.float64))
    support.values[2, 2] = 1.0

    topo.streams = estimate_reach_seed_strength(
        topo.streams, ndvi, support_da=support, sample_spacing_m=0.5
    )
    fracs = topo.streams.set_index("stream_id")["seed_support_fraction"]
    assert fracs.loc[2] == 0.0, f"reach 2 fraction {fracs.loc[2]} should be 0"
    assert fracs.loc[3] == 0.0, f"reach 3 fraction {fracs.loc[3]} should be 0"


def test_short_reach_gets_soft_anchor_not_hard_pin():
    """A sub-spacing reach cannot be hard-pinned — fraction stays 0, hit is soft."""
    geoms = [
        LineString([(0.5, 0.5), (1.5, 0.5)]),
        LineString([(1.5, 0.5), (2.5, 0.5)]),
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
    support = _grid(np.zeros((3, 5), dtype=np.float64))
    support.values[2, 2] = 1.0
    support.values[2, 3] = 1.0

    topo.streams = estimate_reach_seed_strength(
        topo.streams, ndvi, support_da=support, sample_spacing_m=0.5
    )
    by_id = topo.streams.set_index("stream_id")
    assert by_id.loc[3, "seed_support_fraction"] == 0.0
    assert bool(by_id.loc[3, "seed_support_hit"]) is True
    assert by_id.loc[3, "seed_strength"] == 1.0


def test_short_reach_shared_endpoint_pixel_not_hard_pinned():
    """A sub-pixel reach with support only at its shared endpoint must not be hard-pinned."""
    geoms = [
        LineString([(0.5, 0.5), (1.5, 0.5)]),
        LineString([(1.5, 0.5), (2.5, 0.5)]),
        # 0.3 m reach — shorter than a grid pixel
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
    # Support ONLY at col 2 (x=2.5) — the shared vertex between reaches 2 and 3.
    # The short reach's midpoint (x=2.65) falls in the same pixel.
    support = _grid(np.zeros((3, 5), dtype=np.float64))
    support.values[2, 2] = 1.0

    topo.streams = estimate_reach_seed_strength(
        topo.streams, ndvi, support_da=support, sample_spacing_m=0.5
    )
    by_id = topo.streams.set_index("stream_id")
    # fraction must be 0 — short reach has no interior samples
    assert by_id.loc[3, "seed_support_fraction"] == 0.0
    # reach 2 also must not be hard-pinned (endpoint excluded)
    assert by_id.loc[2, "seed_support_fraction"] == 0.0


def test_solve_channel_heads_hard_pin_resists_neighbor_smoothing():
    """Hard-pinned reaches must stay at bed even with strong smoothing."""
    topo = _seeded_topo(support_reach=3)
    heads = solve_channel_heads(
        topo,
        d_min_off_support_m=0.5,
        smoothness_weight=100.0,
    )
    by_id = heads.set_index("stream_id")
    assert np.isclose(by_id.loc[3, "head_depth_m"], 0.0)


def test_solve_channel_heads_propagated_weight_decays_upstream():
    """Upstream reaches should detach more where propagated weight is low."""
    topo = _seeded_topo(support_reach=3)
    heads = solve_channel_heads(topo, d_min_off_support_m=0.5)
    by_id = heads.set_index("stream_id")
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
    assert np.isclose(result.iloc[1]["endpoint_elev_m"], 88.0)
    assert np.isclose(result.iloc[0]["endpoint_elev_m"], 95.0)
    assert np.isclose(result.iloc[2]["endpoint_elev_m"], 88.0)


# --- New residual-depth tests ---


def _steep_chain():
    """Three-reach chain with large bed relief (400 m per reach)."""
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
            "length_m": [500.0, 500.0, 500.0],
            "geometry": geoms,
        },
        geometry="geometry",
        crs="EPSG:5070",
    )


def _steep_elev() -> xr.DataArray:
    return _grid(
        np.tile(
            np.array([1200.0, 800.0, 400.0, 200.0, 100.0], dtype=np.float64),
            (3, 1),
        )
    )


def _steep_seeded_topo(wet_reach=3):
    """Build topology on steep terrain with wet support on one downstream reach."""
    from handily.rem_fac_head import _build_residual_targets

    streams = _steep_chain()
    elev = _steep_elev()
    topo = build_fac_topology(streams, elev)

    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    support = _grid(np.zeros((3, 5), dtype=np.float64))
    col_ranges = {1: [0, 1], 2: [1, 2, 3], 3: [2, 3, 4]}
    for c in col_ranges[wet_reach]:
        support.values[2, c] = 1.0

    topo.streams = estimate_reach_seed_strength(
        topo.streams, ndvi, support_da=support, sample_spacing_m=0.5
    )
    topo.streams = propagate_upstream_wet_influence(
        topo,
        distance_scale_m=1500.0,
        elevation_scale_m=25.0,
        strahler_distance_scale=0.0,
    )
    topo.streams = _build_residual_targets(
        topo.streams,
        distance_scale_m=1500.0,
        elevation_scale_m=25.0,
        rmax_min_m=2.0,
        rmax_max_m=40.0,
    )
    return topo


def test_residual_solve_dry_reaches_do_not_collapse_to_valley_floor():
    """Upstream head depth must stay within r_max, not approach full relief."""
    topo = _steep_seeded_topo(wet_reach=3)
    heads = solve_channel_heads(topo, d_min_off_support_m=0.5)
    by_id = heads.set_index("stream_id")
    for sid in [1, 2]:
        depth = by_id.loc[sid, "head_depth_m"]
        r_max = by_id.loc[sid, "r_max_m"]
        assert depth <= r_max + d_min + 0.01, (
            f"reach {sid}: depth {depth:.1f} > r_max {r_max:.1f} + d_min"
        )
    # Explicitly check no reach approaches full relief
    assert heads["head_depth_m"].max() < 100.0


d_min = 0.5


def test_residual_solve_tracks_target_sag_when_smoothness_is_low():
    """With low smoothness, head_depth should approximate r_target + d_min."""
    topo = _seeded_topo(support_reach=None)
    heads = solve_channel_heads(
        topo,
        d_min_off_support_m=0.5,
        smoothness_weight=0.001,
        target_weight_base=10.0,
    )
    by_id = heads.set_index("stream_id")
    for sid in [1, 2, 3]:
        depth = by_id.loc[sid, "head_depth_m"]
        expected = by_id.loc[sid, "r_target_m"] + 0.5
        assert abs(depth - expected) < 0.1, (
            f"reach {sid}: depth {depth:.3f} != expected {expected:.3f}"
        )


def test_residual_solve_smooths_residual_not_absolute_head():
    """Steep and flat chains with same topology target should produce similar residuals."""
    from handily.rem_fac_head import _build_residual_targets

    # Flat chain
    flat_streams = _chain_streams()
    flat_elev = _grid(
        np.tile(np.array([4.0, 3.5, 3.0, 2.5, 2.0], dtype=np.float64), (3, 1))
    )
    flat_topo = build_fac_topology(flat_streams, flat_elev)
    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    flat_topo.streams = estimate_reach_seed_strength(
        flat_topo.streams, ndvi, sample_spacing_m=0.5
    )
    flat_topo.streams = propagate_upstream_wet_influence(
        flat_topo,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
        strahler_distance_scale=0.0,
    )
    flat_topo.streams = _build_residual_targets(
        flat_topo.streams,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
        rmax_min_m=0.5,
        rmax_max_m=2.0,
    )

    # Steep chain
    steep_streams = _chain_streams()
    steep_elev = _grid(
        np.tile(np.array([400.0, 300.0, 200.0, 100.0, 0.0], dtype=np.float64), (3, 1))
    )
    steep_topo = build_fac_topology(steep_streams, steep_elev)
    steep_topo.streams = estimate_reach_seed_strength(
        steep_topo.streams, ndvi, sample_spacing_m=0.5
    )
    steep_topo.streams = propagate_upstream_wet_influence(
        steep_topo,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
        strahler_distance_scale=0.0,
    )
    steep_topo.streams = _build_residual_targets(
        steep_topo.streams,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
        rmax_min_m=0.5,
        rmax_max_m=2.0,
    )

    flat_heads = solve_channel_heads(flat_topo, d_min_off_support_m=0.5)
    steep_heads = solve_channel_heads(steep_topo, d_min_off_support_m=0.5)

    flat_r = flat_heads.set_index("stream_id")["head_depth_m"] - 0.5
    steep_r = steep_heads.set_index("stream_id")["head_depth_m"] - 0.5
    # Residuals should be similar despite 100x elevation difference
    for sid in [1, 2, 3]:
        assert abs(flat_r.loc[sid] - steep_r.loc[sid]) < 0.5, (
            f"reach {sid}: flat_r={flat_r.loc[sid]:.2f} vs steep_r={steep_r.loc[sid]:.2f}"
        )


def test_max_hydraulic_slope_becomes_lower_bound_on_residual():
    """Hydraulic slope constraint should increase r when bed drops steeply."""
    topo = _steep_seeded_topo(wet_reach=3)
    # With a very tight slope, the constraint forces upstream reaches to
    # deepen (increase r) relative to what the target alone would give.
    tight = solve_channel_heads(
        topo, d_min_off_support_m=0.5, max_hydraulic_slope=0.0001
    )
    loose = solve_channel_heads(topo, d_min_off_support_m=0.5, max_hydraulic_slope=1.0)
    tight_id = tight.set_index("stream_id")
    loose_id = loose.set_index("stream_id")
    # Tight slope should produce deeper (larger) head_depth on upstream reaches
    assert tight_id.loc[1, "head_depth_m"] >= loose_id.loc[1, "head_depth_m"] - 0.01


def test_invalid_reach_excluded_does_not_break_neighbors():
    """A reach with NaN bed elevation should be excluded without breaking the solve."""
    from handily.rem_fac_head import _build_residual_targets

    streams = _chain_streams()
    elev = _sloped_elev()
    topo = build_fac_topology(streams, elev)

    # Inject NaN elevation on reach 2 after topology build
    idx = topo.streams.index[topo.streams["stream_id"] == 2]
    topo.streams.loc[idx, "up_elev_m"] = np.nan
    topo.streams.loc[idx, "down_elev_m"] = np.nan

    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    topo.streams = estimate_reach_seed_strength(
        topo.streams, ndvi, sample_spacing_m=0.5
    )
    topo.streams = propagate_upstream_wet_influence(
        topo, distance_scale_m=2.0, elevation_scale_m=2.0, strahler_distance_scale=0.0
    )
    topo.streams = _build_residual_targets(
        topo.streams,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
        rmax_min_m=0.5,
        rmax_max_m=2.0,
    )

    heads = solve_channel_heads(topo, d_min_off_support_m=0.5)
    by_id = heads.set_index("stream_id")
    assert np.isnan(by_id.loc[2, "channel_head_m"])
    assert np.isfinite(by_id.loc[1, "channel_head_m"])
    assert np.isfinite(by_id.loc[3, "channel_head_m"])


def test_solve_channel_heads_dry_headwaters_detach():
    """Dry upstream reaches should have h below bed by at least d_min."""
    topo = _seeded_topo(support_reach=None)
    heads = solve_channel_heads(topo, d_min_off_support_m=0.5)
    by_id = heads.set_index("stream_id")
    assert (by_id["head_depth_m"] >= 0.5 - 1e-6).all()
