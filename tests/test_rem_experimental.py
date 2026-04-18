from __future__ import annotations

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import LineString, Polygon

from handily.rem_experimental import (
    NetworkFaces,
    _OpenEdgeAnchorCandidate,
    _best_family_chain_from_anchor,
    _find_open_edge_polygon_ids,
    _is_two_sided_paired_polygon,
    _protected_interreach_polygon_ids,
    _wedge_strip_type,
    attach_strip_elevations,
)


def test_find_open_edge_polygon_ids_marks_boundary_and_closure_faces() -> None:
    dem = xr.DataArray(
        np.zeros((11, 11), dtype=np.float64),
        coords={"y": np.arange(11, dtype=np.float64), "x": np.arange(11, dtype=np.float64)},
        dims=("y", "x"),
    )
    faces = NetworkFaces(
        polygons=[
            Polygon([(0.0, 2.0), (4.0, 2.0), (4.0, 6.0), (0.0, 6.0)]),
            Polygon([(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)]),
            Polygon([(7.0, 7.0), (9.0, 7.0), (9.0, 9.0), (7.0, 9.0)]),
        ],
        closure_edges=[LineString([(6.0, 2.0), (6.0, 6.0)])],
        reach_side_map={},
        polygon_aspects={},
        min_area_m2=100.0,
    )

    open_ids = _find_open_edge_polygon_ids(faces, dem, touch_tol_m=0.01)

    assert open_ids == {0, 1}


def test_best_family_chain_from_anchor_prefers_longest_continuation_path() -> None:
    frame_by_rid = {
        4: LineString([(0.0, 0.0), (10.0, 0.0)]),
        11: LineString([(10.0, 0.0), (20.0, 0.0)]),
        25: LineString([(20.0, 0.0), (30.0, 0.0)]),
        26: LineString([(30.0, 0.0), (40.0, 0.0)]),
        28: LineString([(20.0, 0.0), (20.0, 8.0)]),
    }
    snap_by_rid = frame_by_rid
    graph = {
        4: {"start": [], "end": [(11, "start")]},
        11: {"start": [(4, "end")], "end": [(25, "start"), (28, "start")]},
        25: {"start": [(11, "end")], "end": [(26, "start")]},
        26: {"start": [(25, "end")], "end": []},
        28: {"start": [(11, "end")], "end": []},
    }
    candidate = _OpenEdgeAnchorCandidate(
        poly_id=10,
        side_label="left",
        reach_id=4,
        station_id=43,
        open_end="end",
        unstable_run_len=3,
        row={},
    )

    chain_order, score = _best_family_chain_from_anchor(
        candidate,
        {4, 11, 25, 26, 28},
        graph,
        frame_by_rid,
        snap_by_rid,
    )

    assert [rid for rid, _ in chain_order] == [4, 11, 25, 26]
    assert score[0] == 4


def test_is_two_sided_paired_polygon_requires_both_labels() -> None:
    assert _is_two_sided_paired_polygon({"left": {4, 11}, "right": {15}})
    assert not _is_two_sided_paired_polygon({"left": {4, 11}})
    assert not _is_two_sided_paired_polygon({"right": {15}})


def test_protected_interreach_polygon_ids_only_keeps_two_sided_faces() -> None:
    faces = NetworkFaces(
        polygons=[],
        closure_edges=[],
        reach_side_map={
            (4, "left"): 10,
            (11, "left"): 10,
            (15, "right"): 10,
            (18, "left"): 1,
            (19, "right"): 1,
            (25, "left"): 22,
        },
        polygon_aspects={},
        min_area_m2=100.0,
    )

    assert _protected_interreach_polygon_ids(faces) == {1, 10}


def test_wedge_strip_type_marks_open_mixed_pairs_as_transition() -> None:
    s0 = {"strip_type": "interreach", "poly_id": 10}
    s1 = {"strip_type": "side", "poly_id": -1}

    assert _wedge_strip_type(s0, s1, protected_poly_ids={1}) == ("transition", -1)
    assert _wedge_strip_type(s0, s1, protected_poly_ids={10}) == ("interreach", 10)


def test_wedge_strip_type_uses_transition_span_for_interreach_pairs() -> None:
    s0 = {
        "reach_id": 4,
        "side": "left",
        "station_id": 18,
        "strip_type": "interreach",
        "poly_id": 10,
    }
    s1 = {
        "reach_id": 4,
        "side": "left",
        "station_id": 19,
        "strip_type": "interreach",
        "poly_id": 10,
    }

    assert _wedge_strip_type(
        s0,
        s1,
        protected_poly_ids={1},
        transition_spans={(4, "left"): (18, 44)},
    ) == ("transition", -1)


def test_attach_strip_elevations_smooths_base_elevations_without_nan_bleed() -> None:
    dem = xr.DataArray(
        np.array(
            [
                [10.0, 11.0, 12.0],
                [10.0, 11.0, 12.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=np.float64,
        ),
        coords={"y": np.array([0.0, 1.0, 2.0]), "x": np.array([0.0, 1.0, 2.0])},
        dims=("y", "x"),
    )
    strips = gpd.GeoDataFrame(
        [
            {
                "reach_id": 4,
                "station_id": 0,
                "side": "left",
                "strip_type": "side",
                "poly_id": -1,
                "anchor_x": 0.5,
                "anchor_y": 1.0,
                "geometry": LineString([(0.5, 1.0), (0.5, 2.0)]),
            },
            {
                "reach_id": 4,
                "station_id": 1,
                "side": "left",
                "strip_type": "side",
                "poly_id": -1,
                "anchor_x": 10.0,
                "anchor_y": 1.0,
                "geometry": LineString([(10.0, 1.0), (10.0, 2.0)]),
            },
            {
                "reach_id": 4,
                "station_id": 2,
                "side": "left",
                "strip_type": "side",
                "poly_id": -1,
                "anchor_x": 1.5,
                "anchor_y": 1.0,
                "geometry": LineString([(1.5, 1.0), (1.5, 2.0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:5070",
    )
    snapped = gpd.GeoDataFrame(columns=["reach_id", "geometry"], geometry="geometry", crs="EPSG:5070")

    out = attach_strip_elevations(strips, snapped, dem).sort_values("station_id")

    assert np.isfinite(out.iloc[0]["base_elev_m"])
    assert np.isfinite(out.iloc[2]["base_elev_m"])
    assert np.isfinite(out.iloc[1]["base_elev_m"])
