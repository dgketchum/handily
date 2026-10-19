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


# --- closed-basin clip ------------------------------------------------------

_CODE_FOR_OFFSET = {v: k for k, v in _D8_OFFSETS.items()}


def _closed_window(open_basin: bool = False):
    """A 12 x 24 window (1 m cells) holding a HUC box plus a halo.

    - Chain A (dominant, outlet FAC 1000): runs east along row 5 inside the HUC
      down to a sink, then climbs NE over a spill and leaves the HUC through the
      halo to the window edge (in-basin share 15/17 = 0.88). ``open_basin=True``
      makes the raw DEM descend monotonically to the exit instead (no sink).
    - Chain B (small internally drained sub-basin, outlet FAC 30 = 3 % of max):
      row 8 inside the HUC, spill path leaves south through the halo (6/8 = 0.75
      in-basin) -- dropped by the open-basin rules (< 5 % of max, < 90 % in).
    - Chain C (neighbour clipping the halo, outlet FAC 100): fully outside.
    """
    shape = (12, 24)
    streams = np.zeros(shape, dtype=np.int16)
    d8 = np.zeros(shape, dtype=np.int16)
    fac = np.zeros(shape, dtype=np.float32)
    dem = np.full(shape, 200.0, dtype=np.float32)

    chain_a = [(5, c) for c in range(3, 15)] + [
        (4, 15),
        (3, 16),
        (2, 17),
        (1, 18),
        (0, 19),
    ]
    if open_basin:
        z_a = np.linspace(150.0, 20.0, len(chain_a))
    else:
        # descend to the sink at col 10 (z 10), then climb to the spill (65 at
        # the last in-basin cell (2,17)), then drop off the divide in the halo.
        z_a = [100, 90, 80, 60, 40, 20, 15, 10, 20, 30, 40, 50, 55, 60, 65, 40, 30]
    fac_a = np.linspace(10.0, 1000.0, len(chain_a))
    chain_b = [(8, c) for c in range(5, 10)] + [(9, 10), (10, 11), (11, 12)]
    z_b = [60, 50, 40, 30, 20, 25, 22, 18]
    fac_b = np.linspace(2.0, 30.0, len(chain_b))
    chain_c = [(11, c) for c in range(15, 21)]
    z_c = [90, 80, 70, 60, 50, 40]
    fac_c = np.linspace(20.0, 100.0, len(chain_c))

    for chain, zs, fs in (
        (chain_a, z_a, fac_a),
        (chain_b, z_b, fac_b),
        (chain_c, z_c, fac_c),
    ):
        for i, ((r, c), z, f) in enumerate(zip(chain, zs, fs)):
            streams[r, c] = 1
            dem[r, c] = z
            fac[r, c] = f
            if i + 1 < len(chain):
                nr, nc = chain[i + 1]
                d8[r, c] = _CODE_FOR_OFFSET[(nr - r, nc - c)]
    basin_poly = box(2, 2, 22, 10)  # cells rows 2..9, cols 2..21
    return streams, d8, fac, dem, basin_poly, (chain_a, chain_b, chain_c)


def _write_window(tmp_path, streams, d8, fac, dem):
    transform = from_origin(0, streams.shape[0], 1, 1)
    paths = {}
    for name, arr, dtype, nodata in [
        ("streams_10m.tif", streams, "int16", 0),
        ("d8_pointer.tif", d8, "int16", 0),
        ("flow_accumulation.tif", fac, "float32", -1),
        ("dem_10m.tif", dem, "float32", -9999),
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


def _window_reaches(chains, transform):
    return gpd.GeoDataFrame(
        {"stream_id": [0, 1, 2], "strahler": [3, 1, 2]},
        geometry=[_reach_on_cells(ch, transform) for ch in chains],
        crs="EPSG:5070",
    )


def _clip(paths, gdf, basin_poly, with_dem):
    return clip_streams_to_fac_watershed(
        gdf,
        paths["d8_pointer.tif"],
        paths["streams_10m.tif"],
        paths["flow_accumulation.tif"],
        basin_poly=basin_poly,
        dem_path=paths["dem_10m.tif"] if with_dem else None,
    )


def test_clip_closed_basin_keeps_small_internal_subbasin(tmp_path, caplog):
    """Closed HUC: the small sub-basin (3 % of max FAC, 75 % in-basin) is dropped
    by the open-basin rules but kept once closure is detected from the DEM; the
    halo-only neighbour is still dropped."""
    streams, d8, fac, dem, basin_poly, chains = _closed_window()
    paths, transform = _write_window(tmp_path, streams, d8, fac, dem)
    gdf = _window_reaches(chains, transform)

    legacy = _clip(paths, gdf, basin_poly, with_dem=False)
    assert len(legacy) == 1  # old rule: sub-basin B dropped

    with caplog.at_level("INFO", logger="handily.regional_fac"):
        kept = _clip(paths, gdf, basin_poly, with_dem=True)
    assert "basin CLOSED" in caplog.text
    assert len(kept) == 2
    assert kept.geometry.iloc[0].equals(gdf.geometry.iloc[0])  # A
    assert kept.geometry.iloc[1].equals(gdf.geometry.iloc[1])  # B
    assert list(kept["stream_id"]) == [0, 1]


def test_clip_open_basin_unchanged_by_closure_check(tmp_path, caplog):
    """Open HUC (trunk descends to its exit): the closure check is a no-op and
    the kept set is identical to the legacy open-basin rules."""
    streams, d8, fac, dem, basin_poly, chains = _closed_window(open_basin=True)
    paths, transform = _write_window(tmp_path, streams, d8, fac, dem)
    gdf = _window_reaches(chains, transform)

    legacy = _clip(paths, gdf, basin_poly, with_dem=False)
    with caplog.at_level("INFO", logger="handily.regional_fac"):
        new = _clip(paths, gdf, basin_poly, with_dem=True)
    assert "basin OPEN" in caplog.text
    assert len(new) == len(legacy) == 1
    assert new.geometry.iloc[0].equals(legacy.geometry.iloc[0])
    assert new.to_wkb().equals(legacy.to_wkb())
