"""Unit tests for the 2 km moving-window coarse surface (build_coarse_surface_mw2km.py).

* ``moving_mean`` is the nodata-aware centred window mean: sum of valid over
  count of valid, zero-padded at the array edge, NaN where no valid neighbour.
* ``edge_enrichment`` (the 2 km-line column-jump statistic) is >> 1 on a 2 km
  block raster nearest-sampled to 100 m and ~1 on a smooth moving-window field.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_UTILS = Path(__file__).resolve().parents[1] / "utils"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _UTILS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mw = _load("build_coarse_surface_mw2km")
ND = 32767


def test_moving_mean_nodata_aware():
    dem = np.array(
        [
            [10, 20, ND, ND, ND],
            [30, 40, ND, ND, ND],
            [ND, ND, ND, ND, ND],
            [ND, ND, ND, ND, 7],
        ],
        dtype="int16",
    )
    m = mw.moving_mean(dem, ND, 3)
    # corner (0, 0): window holds 10, 20, 30, 40 (rest out-of-array / nodata)
    np.testing.assert_allclose(m[0, 0], 25.0)
    # (1, 2): neighbours 20, 40 valid only
    np.testing.assert_allclose(m[1, 2], 30.0)
    # (2, 2): window rows 1-3, cols 1-3 -> only 40
    np.testing.assert_allclose(m[2, 2], 40.0)
    # (0, 4): no valid neighbour -> NaN (stays nodata)
    assert np.isnan(m[0, 4])
    # (3, 4) and (2, 3): see the lone 7
    np.testing.assert_allclose(m[3, 4], 7.0)
    np.testing.assert_allclose(m[2, 3], 7.0)


def test_moving_mean_matches_bruteforce():
    rng = np.random.default_rng(1)
    dem = rng.integers(0, 3000, (40, 37)).astype("int16")
    dem[rng.random(dem.shape) < 0.2] = ND
    m = mw.moving_mean(dem, ND, 5)
    for r, c in [(0, 0), (10, 10), (39, 36), (20, 0), (5, 33)]:
        w = dem[max(r - 2, 0) : r + 3, max(c - 2, 0) : c + 3]
        v = w[w != ND].astype(float)
        if len(v):
            np.testing.assert_allclose(m[r, c], v.mean(), rtol=1e-9)
        else:
            assert np.isnan(m[r, c])


def test_edge_enrichment_block_vs_smooth():
    # 100 m cell centres on the canonical origin, 8 km x 8 km
    xs = -2_540_000.0 + 50.0 + 100.0 * np.arange(80)
    ys = 3_258_000.0 - 50.0 - 100.0 * np.arange(80)
    xx, yy = np.meshgrid(xs, ys)
    smooth = 0.01 * (xx - xs[0]) + 0.005 * (yy - ys[0]) + 1000.0
    rng = np.random.default_rng(0)
    smooth = smooth + rng.normal(0, 0.5, smooth.shape)
    # 2 km block mean, nearest-sampled back to 100 m
    bx = np.floor((xx - (-2_540_000.0)) / 2000.0)
    by = np.floor((3_258_000.0 - yy) / 2000.0)
    block = np.zeros_like(smooth)
    for i in np.unique(by):
        for j in np.unique(bx):
            sel = (by == i) & (bx == j)
            block[sel] = smooth[sel].mean() + 30.0 * ((i + j) % 2)
    e_block = mw.edge_enrichment(block + rng.normal(0, 0.5, block.shape), xs, ys)
    e_smooth = mw.edge_enrichment(smooth, xs, ys)
    assert e_block["col"]["enrich_mean"] > 5.0
    assert e_block["row"]["enrich_mean"] > 5.0
    assert abs(e_smooth["col"]["enrich_mean"] - 1.0) < 0.15
    assert abs(e_smooth["col"]["enrich_mean_1km_not_2km"] - 1.0) < 0.15
