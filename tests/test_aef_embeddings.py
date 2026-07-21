"""Unit tests for the pure logic of utils/extract_aef_embeddings.py: AEF column naming,
exact EPSG:5070 lattice-cell bounds (must agree with build_mae_patches.snap_to_lattice),
chunk slicing, and spatial ordering. EE / network paths are not unit-tested.

Loads the module by path (repo test convention).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


aef = _load("extract_aef_embeddings")
bmp = _load("build_mae_patches")


def test_aef_columns_are_mae_prefixed_and_64():
    cols = aef.aef_columns()
    assert len(cols) == 64
    assert cols[0] == "mae_aef_00" and cols[-1] == "mae_aef_63"
    # mae_ prefix is REQUIRED: the width-agnostic --mae-embeddings loader / probe select
    # columns by that prefix, so AEF rides in with zero trainer edits.
    assert all(c.startswith("mae_aef_") for c in cols)


def test_cell_bounds_match_snap_to_lattice_and_contain_point():
    x0, y0 = bmp.LATTICE_ORIGIN
    res = bmp.RES_M
    # three arbitrary points, incl. one off-center within its cell
    # interior points only (a coord exactly on a 100 m gridline lands on a half-open
    # cell edge; floor-snap is still consistent, but containment at the excluded top edge
    # is a measure-zero case real well coords never hit).
    x = np.array([x0 + 250.0, x0 + 1005.0, 1_000_037.0])
    y = np.array([y0 - 250.0, y0 - 3333.0, 900_051.0])
    b = aef.cell_bounds_5070(x, y)  # [xlo, ylo, xhi, ytop]
    col, row = bmp.snap_to_lattice(x, y)
    exp_xlo = x0 + col * res
    exp_ytop = y0 - row * res
    assert np.allclose(b[:, 0], exp_xlo)
    assert np.allclose(b[:, 2], exp_xlo + res)
    assert np.allclose(b[:, 3], exp_ytop)
    assert np.allclose(b[:, 1], exp_ytop - res)
    # cells are exactly 100 m and the point lies within [xlo,xhi) x [ylo,ytop)
    assert np.allclose(b[:, 2] - b[:, 0], res)
    assert np.allclose(b[:, 3] - b[:, 1], res)
    assert np.all((x >= b[:, 0]) & (x < b[:, 2]))
    assert np.all((y >= b[:, 1]) & (y < b[:, 3]))


def test_chunk_slices_cover_all_rows():
    sl = aef.chunk_slices(10, 4)
    assert [(s.start, s.stop) for s in sl] == [(0, 4), (4, 8), (8, 10)]
    idx = np.concatenate([np.arange(s.start, s.stop) for s in sl])
    assert np.array_equal(idx, np.arange(10))
    assert aef.chunk_slices(0, 5) == []


def test_spatial_order_is_a_permutation():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2e6, 2e6, 500)
    y = rng.uniform(4e5, 3e6, 500)
    order = aef.spatial_order(x, y)
    assert np.array_equal(np.sort(order), np.arange(500))
    # nearby points cluster: consecutive coarse bins should be monotone non-decreasing
    bx = np.floor(x[order] / 100_000.0)
    by = np.floor(y[order] / 100_000.0)
    key = by * 100_000 + bx
    assert np.all(np.diff(key) >= 0)
