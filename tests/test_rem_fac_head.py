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


def test_order_area_hard_pin_pins_evidence_free_mainstem():
    """High-order, high-drainage reach pins to the surface without mask support."""
    topo = _seeded_topo()
    topo.streams["strahler"] = [0, 1, 5]
    topo.streams["drainage_km2"] = [1.0, 10.0, 600.0]
    heads = solve_channel_heads(
        topo, d_min_off_support_m=0.5, strahler_pin_min=5, area_pin_km2=250.0
    )
    by_id = heads.set_index("stream_id")
    assert bool(by_id.loc[3, "hard_pin"])
    assert np.isclose(by_id.loc[3, "head_depth_m"], 0.0)
    assert not bool(by_id.loc[1, "hard_pin"])
    assert by_id.loc[1, "head_depth_m"] >= 0.5 - 1e-9


def test_order_pin_respects_area_guard():
    """Order qualifies but junction-corrected drainage is tiny: no pin."""
    topo = _seeded_topo()
    topo.streams["strahler"] = [0, 1, 5]
    topo.streams["drainage_km2"] = [1.0, 10.0, 2.0]
    heads = solve_channel_heads(
        topo, d_min_off_support_m=0.5, strahler_pin_min=5, area_pin_km2=250.0
    )
    assert not heads.set_index("stream_id")["hard_pin"].any()


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


def test_drainage_area_prior_keeps_big_river_at_floor():
    """High-drainage reaches with zero imagery evidence stay at the wet floor.

    A river draining >= area_sag_hi_km2 never runs dry — the FAC-derived
    area prior must hold it near the bed even when NDVI/support see nothing.
    """
    streams = _chain_streams()
    elev = _sloped_elev()
    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    # 1e9 cells of 1 m2 = 1000 km2 — far above area_sag_hi_km2.
    fac = _grid(np.full((3, 5), 1.0e9, dtype=np.float64))
    heads = build_channel_heads(
        streams,
        elev,
        ndvi,
        fac_da=fac,
        sample_spacing_m=0.5,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
    )
    by_id = heads.set_index("stream_id")
    for sid in [1, 2, 3]:
        assert by_id.loc[sid, "head_depth_m"] < 0.6, (
            f"reach {sid}: depth {by_id.loc[sid, 'head_depth_m']:.2f} m — "
            "area prior failed to hold a big river near the bed"
        )
    # Control: the same bone-dry chain without FAC detaches.
    heads_dry = build_channel_heads(
        streams,
        elev,
        ndvi,
        sample_spacing_m=0.5,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
    )
    assert heads_dry.set_index("stream_id").loc[1, "head_depth_m"] > 1.0


def test_hydraulic_slope_floor_does_not_drag_wet_reaches_off_bed():
    """Observed-wet reaches above a steep wet-to-wet bed drop stay near bed.

    With a tight max_hydraulic_slope and steep bed, the slope floor would
    push an unpinned wet reach to its r_max cap (depth = d_min + rmax_min).
    Wet evidence must relax the allowance so depth stays at the d_min floor.
    """
    streams = _steep_chain()
    elev = _steep_elev()
    # High NDVI everywhere: every reach is a strong seed (w_wet ~ 1) but
    # nothing is hard-pinned (no support raster).
    ndvi = _grid(np.full((3, 5), 0.8, dtype=np.float64))
    heads = build_channel_heads(
        streams,
        elev,
        ndvi,
        sample_spacing_m=0.5,
        max_hydraulic_slope=0.05,
    )
    by_id = heads.set_index("stream_id")
    for sid in [1, 2, 3]:
        assert by_id.loc[sid, "head_depth_m"] < 0.6, (
            f"reach {sid}: depth {by_id.loc[sid, 'head_depth_m']:.2f} m — "
            "slope floor dragged a wet reach off the bed"
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


def test_downstream_wet_propagation_keeps_mainstem_wet():
    """Reaches below a wet headwater seed must not sag toward r_max.

    Wet support on the upstream-most reach; the two downstream reaches
    have no local evidence. Without downstream propagation they would sag
    toward r_target (tens of m with default rmax) — with it they stay
    near the bed.
    """
    streams = _chain_streams()
    elev = _sloped_elev()
    ndvi = _grid(np.full((3, 5), 0.0, dtype=np.float64))
    support = _grid(np.zeros((3, 5), dtype=np.float64))
    for c in [0, 1]:
        support.values[2, c] = 1.0

    heads = build_channel_heads(
        streams,
        elev,
        ndvi,
        support_da=support,
        sample_spacing_m=0.5,
        distance_scale_m=2.0,
        elevation_scale_m=2.0,
        down_distance_scale_m=100.0,
        # Toy bed drops 1 m per 1 m reach; loosen the hydraulic-slope floor
        # so it doesn't force detachment regardless of wetness.
        max_hydraulic_slope=2.0,
    )
    by_id = heads.set_index("stream_id")
    assert np.isclose(by_id.loc[1, "head_depth_m"], 0.0)
    for sid in [2, 3]:
        assert by_id.loc[sid, "head_depth_m"] < 1.0, (
            f"reach {sid}: depth {by_id.loc[sid, 'head_depth_m']:.2f} m "
            "should stay near bed below a wet headwater"
        )


def test_solve_channel_heads_dry_headwaters_detach():
    """Dry upstream reaches should have h below bed by at least d_min."""
    topo = _seeded_topo(support_reach=None)
    heads = solve_channel_heads(topo, d_min_off_support_m=0.5)
    by_id = heads.set_index("stream_id")
    assert (by_id["head_depth_m"] >= 0.5 - 1e-6).all()
