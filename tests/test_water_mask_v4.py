"""Unit tests for the V4 surface-water mask (build_water_mask_v4.py).

* ``rasterize_any_touch`` masks every 100 m cell a polygon touches (all_touched),
  including a cell only clipped by a sliver, on the lattice sub-window.
* ``lattice_window`` covers the bounds with a one-cell margin and clips at the
  lattice edge.
"""

import importlib.util
import sys
from pathlib import Path

from rasterio.transform import from_origin
from shapely.geometry import box

_UTILS = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(_UTILS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _UTILS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v4 = _load("build_water_mask_v4")
T = from_origin(0.0, 1000.0, 100.0, 100.0)  # 10 x 10 lattice of 100 m cells


def test_any_touch_includes_sliver_cells():
    win = v4.lattice_window((0, 0, 1000, 1000), T, 10, 10)
    # polygon covers cell (row 2, col 2) fully and clips 10 m into col 3
    poly = box(200.0, 700.0, 310.0, 800.0)
    a = v4.rasterize_any_touch([poly], T, 10, 10, win)
    assert a.dtype.name == "uint8"
    assert a[2, 2] == 1 and a[2, 3] == 1
    assert a.sum() == 2


def test_lattice_window_margin_and_clip():
    w = v4.lattice_window((250.0, 450.0, 350.0, 550.0), T, 10, 10)
    assert (w.col_off, w.row_off, w.width, w.height) == (1, 3, 4, 4)
    w = v4.lattice_window((-500.0, -500.0, 2000.0, 2000.0), T, 10, 10)
    assert (w.col_off, w.row_off, w.width, w.height) == (0, 0, 10, 10)


def test_playa_excluded_and_targets_included():
    assert 361 not in v4.FTYPES
    assert set(v4.FTYPES) == {390, 436, 466}
