"""Tests for anisotropic REM support alignment (Phases 1-6)."""

import numpy as np
import geopandas as gpd
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import LineString

from handily import compute, rem_frame
from handily.config import HandilyConfig


# ---------------------------------------------------------------------------
# Helpers — synthetic rasters and flowlines
# ---------------------------------------------------------------------------


def _make_dem_da(
    nrow=100, ncol=100, res=1.0, origin_x=0.0, origin_y=100.0, crs="EPSG:5070"
):
    """Flat DEM at 100 m elevation with a 2 m deep channel at col 50."""
    elev = np.full((nrow, ncol), 100.0, dtype=np.float32)
    elev[:, 48:53] = 98.0  # 5-pixel-wide channel
    transform = from_origin(origin_x, origin_y, res, res)
    xs = np.arange(ncol) * res + origin_x + res / 2
    ys = origin_y - np.arange(nrow) * res - res / 2
    da = xr.DataArray(elev, dims=["y", "x"], coords={"y": ys, "x": xs})
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform)
    return da


def _make_water_da(dem_da, water_cols=range(49, 52)):
    """Binary water mask matching the DEM grid — water in given columns."""
    arr = np.zeros(dem_da.shape, dtype=np.float32)
    arr[:, list(water_cols)] = 1.0
    da = xr.DataArray(arr, dims=dem_da.dims, coords=dem_da.coords)
    da = da.rio.write_crs(dem_da.rio.crs)
    da = da.rio.write_transform(dem_da.rio.transform())
    return da


def _make_flowlines(n_lines=3, x_offset=50.5, crs="EPSG:5070"):
    """Vertical flowlines at x_offset, each covering a third of the domain."""
    lines = []
    for i in range(n_lines):
        y_start = 99.5 - i * 33
        y_end = y_start - 30
        lines.append(LineString([(x_offset, y_start), (x_offset, y_end)]))
    gdf = gpd.GeoDataFrame({"lengthkm": [0.03] * n_lines}, geometry=lines, crs=crs)
    return gdf


def _default_config(**overrides):
    kw = dict(
        out_dir="/tmp/test_handily",
        flowlines_local_dir="",
        ndwi_dir="",
        stac_dir="",
        fields_path="",
        rem_method="anisotropic_frame",
        rem_frame_station_spacing_m=5.0,
        rem_snap_max_offset_m=20.0,
        rem_snap_search_spacing_m=1.0,
        rem_frame_smoothing_m=10.0,
        rem_cross_max_dist_m=40.0,
        rem_cross_step_m=1.0,
        rem_cross_ridge_prominence_m=1.5,
        rem_cross_descend_stop_m=0.5,
        rem_min_support_width_m=5.0,
        rem_min_confidence=0.05,
        rem_snap_w_elev=0.3,
        rem_snap_w_water=0.4,
        rem_snap_w_prior=0.3,
        rem_snap_w_transition=1.0,
        rem_support_corridor_half_width_m=3.0,
        rem_support_corridor_half_length_m=3.0,
        rem_water_support_mode="binary_mask",
        rem_min_station_water_hit_fraction=0.05,
        rem_max_consecutive_no_water_m=50.0,
        rem_max_mean_snap_offset_m=15.0,
        rem_min_seeded_fraction=0.0,
    )
    kw.update(overrides)
    return HandilyConfig(**kw)


# ---------------------------------------------------------------------------
# Phase 1: seeded vs reachable
# ---------------------------------------------------------------------------


class TestFlowlineStatus:
    def test_seeded_vs_reachable(self):
        """A seeded segment + a dry neighbor: both reachable, only one seeded."""
        dem = _make_dem_da()
        water = _make_water_da(dem, water_cols=range(49, 52))
        # Line 0: over water.  Line 1: off to the side, no water, but connected
        lines = gpd.GeoDataFrame(
            {"lengthkm": [0.03, 0.03]},
            geometry=[
                LineString([(50.5, 90), (50.5, 60)]),
                LineString([(50.5, 60), (70.5, 30)]),
            ],
            crs="EPSG:5070",
        )
        ann = compute.propagate_flowline_confirmation(
            lines,
            dem,
            water,
            ndwi_threshold=0.5,
            flowlines_buffer_m=5.0,
        )
        assert ann.iloc[0]["water_seeded"] is True or ann.iloc[0]["water_seeded"]
        assert ann.iloc[0]["water_seed_pixels"] > 0
        # Line 1 is reachable via BFS but NOT seeded
        assert ann.iloc[1]["reachable_from_seed"]
        # Legacy aliases still work
        assert ann.iloc[0]["propagation_confirmed"]
        assert ann.iloc[1]["propagation_confirmed"]

    def test_seed_fraction(self):
        """water_seed_fraction is nonzero for a line overlapping water."""
        dem = _make_dem_da()
        water = _make_water_da(dem)
        lines = _make_flowlines(n_lines=1)
        ann = compute.propagate_flowline_confirmation(
            lines,
            dem,
            water,
            ndwi_threshold=0.5,
            flowlines_buffer_m=5.0,
        )
        assert ann.iloc[0]["water_seed_fraction"] > 0


# ---------------------------------------------------------------------------
# Phase 2: branch splitting
# ---------------------------------------------------------------------------


class TestReachSplitting:
    def test_junction_produces_separate_branches(self):
        """A Y-junction (3 lines meeting) should yield separate branch reaches,
        not one merged chain.  With 5 features (2-1-junction-1-2 topology),
        each branch is its own reach."""
        # Build a clear Y: one trunk of 2 features, two branches of 1 each
        lines = gpd.GeoDataFrame(
            {"lengthkm": [0.05] * 4, "water_seeded": [True] * 4},
            geometry=[
                LineString([(0, 0), (50, 0)]),  # trunk seg 1
                LineString([(50, 0), (100, 0)]),  # trunk seg 2
                LineString([(100, 0), (150, 50)]),  # branch A
                LineString([(100, 0), (150, -50)]),  # branch B
            ],
            crs="EPSG:5070",
        )
        reaches = rem_frame.split_confirmed_flowlines_into_reaches(lines)
        # Should produce at least 2 separate reaches (trunk + each branch)
        assert len(reaches) >= 2
        # No single reach should contain all 4 features
        for _, r in reaches.iterrows():
            assert r["n_flowlines"] < 4

    def test_seeded_fraction_propagates(self):
        """Each NHD feature becomes its own reach; seeded_fraction is per-feature."""
        lines = gpd.GeoDataFrame(
            {"lengthkm": [0.03, 0.03], "water_seeded": [True, False]},
            geometry=[
                LineString([(0, 0), (50, 0)]),
                LineString([(50, 0), (100, 0)]),
            ],
            crs="EPSG:5070",
        )
        reaches = rem_frame.split_confirmed_flowlines_into_reaches(lines)
        # Each feature is its own reach — no feature duplication across reaches
        assert len(reaches) == 2
        fractions = sorted(reaches["seeded_fraction"].tolist())
        assert fractions == [0.0, 1.0]


# ---------------------------------------------------------------------------
# Phase 3: corridor support
# ---------------------------------------------------------------------------


class TestCorridorSupport:
    def test_narrow_channel_detected_by_corridor(self):
        """A 3-pixel water stripe should be detected by corridor but could
        be missed by single-point sampling."""
        dem = _make_dem_da()
        water = _make_water_da(dem, water_cols=[50])  # 1-pixel stripe
        from scipy.interpolate import RegularGridInterpolator

        y_vals = water.y.values.astype(np.float64)
        x_vals = water.x.values.astype(np.float64)
        data = water.values.astype(np.float64)
        y_flip = y_vals[0] > y_vals[-1]
        if y_flip:
            y_vals = y_vals[::-1]
            data = data[::-1, :]
        interp = RegularGridInterpolator(
            (y_vals, x_vals),
            data,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        center = np.array([[50.5, 50.5]])
        normal = np.array([[1.0, 0.0]])
        tangent_vec = np.array([[0.0, 1.0]])
        frac = rem_frame._batch_corridor_support(
            center,
            normal,
            tangent_vec,
            interp,
            corridor_half_width_m=3.0,
            corridor_half_length_m=3.0,
            step_m=1.0,
            mode="binary_mask",
        )
        assert frac[0] > 0


# ---------------------------------------------------------------------------
# Phase 4: snap cost behavior
# ---------------------------------------------------------------------------


class TestSnapCost:
    def test_water_pulls_snapper_toward_wet_corridor(self):
        """When water and elevation disagree, higher water weight
        should pull the snapper toward the wet side."""
        dem = _make_dem_da()
        # Put the lowest ground at col 30 instead of 50
        elev = np.full((100, 100), 100.0, dtype=np.float32)
        elev[:, 28:33] = 97.0  # dry swale at col 30
        elev[:, 48:53] = 99.0  # slightly higher wet channel at col 50
        dem.values[:] = elev
        water = _make_water_da(dem, water_cols=range(49, 52))

        # Prior line at col 40 (between swale and channel)
        line = LineString([(40.5, 90), (40.5, 10)])
        cfg = _default_config(rem_snap_w_elev=0.2, rem_snap_w_water=0.6)
        snapped = rem_frame.snap_reach_to_thalweg(line, dem, water, cfg)

        # With water weight dominating, mean snap position should be closer
        # to col 50 (water) than col 30 (elevation low)
        mean_x = snapped.stations["x_snap"].mean()
        assert mean_x > 40, (
            f"Snapper should pull toward water (col ~50), got mean_x={mean_x:.1f}"
        )


# ---------------------------------------------------------------------------
# Phase 5: reach acceptance
# ---------------------------------------------------------------------------


class TestReachAcceptance:
    def test_accept_good_reach(self):
        dem = _make_dem_da()
        water = _make_water_da(dem)
        line = LineString([(50.5, 90), (50.5, 10)])
        cfg = _default_config()
        snapped = rem_frame.snap_reach_to_thalweg(line, dem, water, cfg)
        reason = rem_frame.evaluate_reach_acceptance(snapped, cfg)
        assert reason is None, f"Expected acceptance, got: {reason}"

    def test_reject_dry_reach(self):
        dem = _make_dem_da()
        water = _make_water_da(dem, water_cols=[])  # no water
        line = LineString([(50.5, 90), (50.5, 10)])
        cfg = _default_config()
        snapped = rem_frame.snap_reach_to_thalweg(line, dem, water, cfg)
        reason = rem_frame.evaluate_reach_acceptance(snapped, cfg)
        assert reason is not None
        assert "water_hit" in reason

    def test_reject_high_offset(self):
        dem = _make_dem_da()
        water = _make_water_da(dem)
        # Prior line far from the water — at col 5
        line = LineString([(5.5, 90), (5.5, 10)])
        cfg = _default_config(
            rem_max_mean_snap_offset_m=3.0, rem_snap_max_offset_m=50.0
        )
        snapped = rem_frame.snap_reach_to_thalweg(line, dem, water, cfg)
        reason = rem_frame.evaluate_reach_acceptance(snapped, cfg)
        # Should either accept (if it snapped close enough) or reject on offset
        if snapped.metrics.mean_snap_offset_m > 3.0:
            assert reason is not None
            assert "offset" in reason

    def test_reach_metrics_populated(self):
        dem = _make_dem_da()
        water = _make_water_da(dem)
        line = LineString([(50.5, 90), (50.5, 10)])
        cfg = _default_config()
        snapped = rem_frame.snap_reach_to_thalweg(line, dem, water, cfg)
        m = snapped.metrics
        assert m is not None
        assert m.n_stations > 0
        assert 0.0 <= m.station_water_hit_fraction <= 1.0
        assert m.mean_snap_offset_m >= 0.0
