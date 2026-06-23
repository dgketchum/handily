"""Unit tests for the MODIS NDVI climatology helpers (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from utils.build_modis_ndvi_climatology import (
    NDVI_FILL,
    granule_month,
    granule_platform,
    scaled_ndvi,
    season_of,
    seasonal_mean,
    tile_id,
    valid_mask,
)


def test_granule_platform():
    assert granule_platform("MOD13Q1.A2015209.h10v04.061.x") == "terra"
    assert granule_platform("MYD13Q1.A2015209.h10v04.061.x") == "aqua"


def test_granule_platform_missing():
    with pytest.raises(ValueError):
        granule_platform("MCD43A4.A2015209.h10v04.061.x")


def test_tile_id():
    assert tile_id("MOD13Q1.A2015209.h10v04.061.2021329192357") == "h10v04"
    assert tile_id("MOD13Q1.A2000049.h08v06.061.x") == "h08v06"


def test_tile_id_missing():
    with pytest.raises(ValueError):
        tile_id("MOD13Q1.A2015209.061.no_tile")


def test_granule_month():
    # A2015209 = 2015 DOY 209 = 28 July; A2000049 = 2000 DOY 49 = 18 Feb
    assert granule_month("MOD13Q1.A2015209.h10v04.061.2021329192357") == 7
    assert granule_month("MOD13Q1.A2000049.h08v06.061.x") == 2
    assert granule_month("MOD13Q1.A2016001.h10v04.061.x") == 1  # leap-year DOY 1


def test_granule_month_missing():
    with pytest.raises(ValueError):
        granule_month("MOD13Q1.h10v04.061.no_date")


def test_season_of():
    assert season_of(1) == "djf"
    assert season_of(12) == "djf"
    assert season_of(4) == "mam"
    assert season_of(7) == "jja"
    assert season_of(10) == "son"


def test_season_of_invalid():
    with pytest.raises(ValueError):
        season_of(0)
    with pytest.raises(ValueError):
        season_of(13)


def test_valid_mask_drops_cloud_snow_fill_and_outofrange():
    # reliability: 0 good, 1 marginal, 2 snow, 3 cloud, 255 fill
    rel = np.array([0, 1, 2, 3, 255, 0, 0], dtype=np.uint8)
    ndvi = np.array([5000, 6000, 7000, 8000, 9000, NDVI_FILL, 20000], dtype=np.int16)
    m = valid_mask(ndvi, rel)
    # keep good+marginal in-range; drop snow, cloud, fill-reliability, NDVI fill,
    # and the out-of-range 20000
    assert m.tolist() == [True, True, False, False, False, False, False]


def test_scaled_ndvi():
    ndvi = np.array([10000, 5000, -2000], dtype=np.int16)
    np.testing.assert_allclose(scaled_ndvi(ndvi), [1.0, 0.5, -0.2], rtol=1e-6)
    assert scaled_ndvi(ndvi).dtype == np.float32


def test_seasonal_mean_counts_and_nan():
    sums = np.array([[1.0, 4.0], [0.0, 9.0]], dtype=np.float32)
    counts = np.array([[2, 4], [0, 3]], dtype=np.uint16)
    out = seasonal_mean(sums, counts)
    assert out[0, 0] == pytest.approx(0.5)
    assert out[0, 1] == pytest.approx(1.0)
    assert np.isnan(out[1, 0])  # never-clear pixel -> NaN, not a fabricated 0
    assert out[1, 1] == pytest.approx(3.0)


def test_accumulation_matches_masked_mean():
    """End-to-end accumulation arithmetic (the loop body in accumulate_tile)."""
    rng = np.random.default_rng(0)
    shape = (4, 4)
    s = np.zeros(shape, np.float32)
    c = np.zeros(shape, np.uint16)
    kept = [[] for _ in range(16)]
    for _ in range(10):
        ndvi = rng.integers(-2000, 10000, shape).astype(np.int16)
        rel = rng.integers(0, 4, shape).astype(np.uint8)
        m = valid_mask(ndvi, rel)
        s += np.where(m, scaled_ndvi(ndvi), np.float32(0))
        c += m.astype(np.uint16)
        for i, (mm, vv) in enumerate(zip(m.ravel(), scaled_ndvi(ndvi).ravel())):
            if mm:
                kept[i].append(vv)
    out = seasonal_mean(s, c).ravel()
    for i in range(16):
        if kept[i]:
            assert out[i] == pytest.approx(np.mean(kept[i]), rel=1e-5)
        else:
            assert np.isnan(out[i])
