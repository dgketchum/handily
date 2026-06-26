"""Tests for regional_fac: D8 pointer offsets and FAC-watershed stream clipping."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from handily.regional_fac import (
    _D8_OFFSETS,
    clip_streams_to_fac_watershed,
    d8_stream_cell_components,
)


def test_d8_offsets_follow_flow_downstream():
    """Stepping along the WBT offset must go downstream (FAC increases).

    A 3-cell SE-flowing chain (0,0)->(1,1)->(2,2): flow accumulation increases
    downstream. The corrected table maps the SE value (4) to (+1, +1); the old
    (wrong) table mapped 4 to (-1, 0), which would step to a zero-FAC cell.
    """
    fac = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    # d8_pointer: (0,0) and (1,1) both flow SE (WBT value 4).
    d8 = np.array(
        [
            [4, 0, 0],
            [0, 4, 0],
            [0, 0, 0],
        ]
    )
    for r, c in [(0, 0), (1, 1)]:
        dr, dc = _D8_OFFSETS[int(d8[r, c])]
        tr, tc = r + dr, c + dc
        assert fac[tr, tc] > fac[r, c]


def test_d8_offsets_match_wbt_convention():
    """Pin the exact WBT clockwise-from-NE encoding (row increases downward)."""
    assert _D8_OFFSETS == {
        1: (-1, 1),  # NE
        2: (0, 1),  # E
        4: (1, 1),  # SE
        8: (1, 0),  # S
        16: (1, -1),  # SW
        32: (0, -1),  # W
        64: (-1, -1),  # NW
        128: (-1, 0),  # N
    }


def _two_chain_rasters():
    """A 6x6 grid with two disjoint SE-flowing stream chains.

    - Chain A (big): (1,1)->(2,2)->(3,3), accumulation 5/8/10, outlet (3,3).
    - Chain B (small): (1,4)->(2,5), accumulation 1/2, outlet (2,5).

    The two chains never touch, so the D8 cell graph has exactly two components;
    A's outlet FAC (10) dominates B's (2).
    """
    streams = np.zeros((6, 6), dtype=np.int16)
    d8 = np.zeros((6, 6), dtype=np.int16)
    fac = np.zeros((6, 6), dtype=np.float32)

    chain_a = [(1, 1), (2, 2), (3, 3)]
    for (r, c), a in zip(chain_a, (5, 8, 10)):
        streams[r, c] = 1
        fac[r, c] = a
    d8[1, 1] = 4  # SE -> (2,2)
    d8[2, 2] = 4  # SE -> (3,3)
    d8[3, 3] = 0  # outlet

    chain_b = [(1, 4), (2, 5)]
    for (r, c), a in zip(chain_b, (1, 2)):
        streams[r, c] = 1
        fac[r, c] = a
    d8[1, 4] = 4  # SE -> (2,5)
    d8[2, 5] = 0  # outlet
    return streams, d8, fac


def test_d8_stream_cell_components_partitions_disjoint_chains():
    streams, d8, _ = _two_chain_rasters()
    rows, cols, labels, sizes = d8_stream_cell_components(streams, d8)
    assert len(rows) == 5  # 3 + 2 stream cells
    assert len(sizes) == 2  # two disconnected chains
    assert sorted(sizes.tolist()) == [2, 3]


def _write_rasters(tmp_path, streams, d8, fac):
    # 1 m cells, origin at (0, 6) so cell (row, col) center is (col+0.5, 6-row-0.5).
    transform = from_origin(0, 6, 1, 1)
    paths = {}
    for name, arr, dtype, nodata in [
        ("streams_10m.tif", streams, "int16", 0),
        ("d8_pointer.tif", d8, "int16", 0),
        ("flow_accumulation.tif", fac, "float32", -1),
    ]:
        p = tmp_path / name
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=dtype,
            crs="EPSG:5070",
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(arr.astype(dtype), 1)
        paths[name] = p
    return paths, transform


def _reach_on_cells(cells, transform):
    """A LineString through the centers of the given (row, col) cells."""
    pts = [transform * (c + 0.5, r + 0.5) for r, c in cells]
    return LineString(pts)


def test_clip_streams_to_fac_watershed_keeps_dominant_drops_minor(tmp_path):
    streams, d8, fac = _two_chain_rasters()
    paths, transform = _write_rasters(tmp_path, streams, d8, fac)

    gdf = gpd.GeoDataFrame(
        {"stream_id": [0, 1], "strahler": [2, 1]},
        geometry=[
            _reach_on_cells([(1, 1), (2, 2), (3, 3)], transform),  # chain A (FAC 10)
            _reach_on_cells([(1, 4), (2, 5)], transform),  # chain B (FAC 2)
        ],
        crs="EPSG:5070",
    )
    kept = clip_streams_to_fac_watershed(
        gdf,
        paths["d8_pointer.tif"],
        paths["streams_10m.tif"],
        paths["flow_accumulation.tif"],
    )
    # only the dominant chain A survives; stream_id reset to a contiguous range.
    assert len(kept) == 1
    assert list(kept["stream_id"]) == [0]
    assert kept.geometry.iloc[0].equals(gdf.geometry.iloc[0])


def test_clip_streams_to_fac_watershed_keeps_comparable_second_outlet(tmp_path):
    """A bifurcated window (two comparable outlets) keeps both, not just one."""
    streams, d8, fac = _two_chain_rasters()
    fac[2, 5] = 8  # lift chain B's outlet to 8 (>= 0.5 * 10) -> kept
    paths, transform = _write_rasters(tmp_path, streams, d8, fac)

    gdf = gpd.GeoDataFrame(
        {"stream_id": [0, 1], "strahler": [2, 1]},
        geometry=[
            _reach_on_cells([(1, 1), (2, 2), (3, 3)], transform),
            _reach_on_cells([(1, 4), (2, 5)], transform),
        ],
        crs="EPSG:5070",
    )
    kept = clip_streams_to_fac_watershed(
        gdf,
        paths["d8_pointer.tif"],
        paths["streams_10m.tif"],
        paths["flow_accumulation.tif"],
    )
    assert len(kept) == 2


def test_clip_keeps_in_basin_tributary_with_polygon(tmp_path):
    """A minor (non-comparable) but unambiguously in-HUC tributary is kept.

    Chain B's outlet (2) is only 20% of the dominant (10) -- below
    ``min_outlet_fraction`` (0.5), so it would be dropped on the outlet rule
    alone. But it is >= 5% of the max and lies fully inside ``basin_poly``, so the
    in-HUC tributary rule retains it (coverage), mirroring the Rio Grande rank-1
    tributary whose confluence falls outside the window. Without the polygon it is
    dropped (see ``..._keeps_dominant_drops_minor``).
    """
    streams, d8, fac = _two_chain_rasters()
    paths, transform = _write_rasters(tmp_path, streams, d8, fac)
    # polygon covering the whole 6x6 / 1 m grid -> both chains are 100% in-basin.
    basin_poly = box(0, 0, 6, 6)

    gdf = gpd.GeoDataFrame(
        {"stream_id": [0, 1], "strahler": [2, 1]},
        geometry=[
            _reach_on_cells([(1, 1), (2, 2), (3, 3)], transform),
            _reach_on_cells([(1, 4), (2, 5)], transform),
        ],
        crs="EPSG:5070",
    )
    kept = clip_streams_to_fac_watershed(
        gdf,
        paths["d8_pointer.tif"],
        paths["streams_10m.tif"],
        paths["flow_accumulation.tif"],
        basin_poly=basin_poly,
    )
    assert len(kept) == 2
