"""Tests for the canonical WBD accessor handily.io.load_huc."""

from __future__ import annotations

import os

import pytest

from handily.io import WBD_NATIONAL_DIR, get_huc10_boundary, load_huc

# A HUC8 confirmed present in the national WBD (Flint, region 04).
KNOWN_HUC8 = "04080204"

_HU8_PARQUET = os.path.join(WBD_NATIONAL_DIR, "wbdhu8_5070.parquet")
requires_wbd = pytest.mark.skipif(
    not os.path.exists(_HU8_PARQUET),
    reason=f"canonical WBD not present at {_HU8_PARQUET}",
)


def test_load_huc_invalid_level_raises():
    """An out-of-range level fails fast before any file read."""
    with pytest.raises(ValueError):
        load_huc(7)


@requires_wbd
def test_load_huc8_exact_code():
    gdf = load_huc(8, huc=KNOWN_HUC8)
    assert len(gdf) == 1
    assert gdf.crs is not None and gdf.crs.to_epsg() == 5070
    assert "huc8" in gdf.columns
    assert gdf["huc8"].iloc[0] == KNOWN_HUC8


@requires_wbd
def test_load_huc8_columns_keeps_key_and_geometry():
    gdf = load_huc(8, huc=KNOWN_HUC8, columns=["name"])
    # key + requested + geometry, deduplicated
    assert set(gdf.columns) == {"huc8", "name", "geometry"}


@requires_wbd
def test_get_huc10_boundary_wrapper_signature():
    """get_huc10_boundary is a thin one-arg wrapper (no wbd_local_dir)."""
    import inspect

    params = list(inspect.signature(get_huc10_boundary).parameters)
    assert params == ["huc10"]
