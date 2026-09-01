"""Unit tests for the V3 screened-water mask and its train/inference replay.

Three properties the water_v3 bundles depend on:
  * ``mask_boundary`` keeps exactly the masked cells that touch unmasked ground
    (array edges included), and the resulting pool gives IDENTICAL nearest-water
    distances to the full mask for every off-mask query point -- the claim that
    lets a CONUS pool that includes the ocean be a tractable kD-tree.
  * ``sample_water_mask`` reads the mask windowed and returns False outside the
    lattice.
  * ``water_v3_inference_block`` replays the bundle's treatment: the FAC flatten
    and the R pin are applied on the mask and nowhere else, and a bundle with no
    ``water_v3`` block is a strict no-op.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

_UTILS = Path(__file__).resolve().parents[1] / "utils"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _UTILS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wm = _load("build_water_mask_v3")
bc = _load("build_conus_graph_inputs")

RES = 100.0
X0, Y0 = -2_540_000.0, 3_258_000.0


def _write_mask(path: Path, mask: np.ndarray) -> Path:
    h, w = mask.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="uint8",
        crs="EPSG:5070",
        transform=from_origin(X0, Y0, RES, RES),
        nodata=0,
    ) as dst:
        dst.write(mask.astype("uint8"), 1)
    return path


def _centers(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.c_[X0 + (cols + 0.5) * RES, Y0 - (rows + 0.5) * RES]


def test_mask_boundary_drops_interior_keeps_edge_touching():
    m = np.zeros((7, 7), bool)
    m[2:5, 2:5] = True  # 3x3 block -> only the centre is interior
    b = wm.mask_boundary(m)
    assert b.sum() == 8
    assert not b[3, 3]
    assert b[2, 2] and b[4, 4]
    # a block flush against the array edge: the edge cell is still boundary (the
    # out-of-array neighbourhood counts as unmasked), while (1,1) is interior
    m2 = np.zeros((5, 5), bool)
    m2[0:3, 0:3] = True
    b2 = wm.mask_boundary(m2)
    assert b2[0, 0] and not b2[1, 1]
    # a fully masked array: the perimeter is boundary (padding outside is unmasked),
    # the centre is not
    assert wm.mask_boundary(np.ones((3, 3), bool)).sum() == 8


def test_boundary_pool_preserves_nearest_water_distance_off_mask():
    rng = np.random.default_rng(0)
    m = np.zeros((60, 60), bool)
    for _ in range(6):  # a few blobs, some wide enough to have interiors
        r, c = rng.integers(5, 50, 2)
        s = int(rng.integers(3, 12))
        m[r : r + s, c : c + s] = True
    b = wm.mask_boundary(m)
    assert b.sum() < m.sum()  # interiors actually exist in this fixture
    full = _centers(*np.where(m))
    shore = _centers(*np.where(b))
    qr, qc = np.where(~m)
    q = _centers(qr, qc)
    d_full = cKDTree(full).query(q, k=1)[0]
    d_shore = cKDTree(shore).query(q, k=1)[0]
    assert np.allclose(d_full, d_shore)


def test_sample_water_mask_windowed_and_outside_lattice(tmp_path):
    m = np.zeros((10, 10), bool)
    m[3, 4] = True
    p = _write_mask(tmp_path / "mask.tif", m)
    on = _centers(np.array([3]), np.array([4]))
    off = _centers(np.array([3]), np.array([5]))
    outside = np.array([[X0 - 10 * RES, Y0 + 10 * RES]])
    got = wm.sample_water_mask(
        np.r_[on[:, 0], off[:, 0], outside[:, 0]],
        np.r_[on[:, 1], off[:, 1], outside[:, 1]],
        p,
    )
    assert got.tolist() == [True, False, False]


def test_water_v3_inference_block_flatten_and_pin(tmp_path):
    m = np.zeros((10, 10), bool)
    m[5, 5] = True
    p = _write_mask(tmp_path / "mask.tif", m)
    cells = pd.DataFrame(
        dict(
            zip(
                ("x5070", "y5070"),
                _centers(np.array([5]), np.array([5])).T,
                strict=True,
            )
        )
    )
    cp = tmp_path / "cells.parquet"
    cells.to_parquet(cp)
    qxy = np.r_[
        _centers(np.array([5]), np.array([5])), _centers(np.array([1]), np.array([1]))
    ]
    z = np.array([100.0, 200.0])
    fac = np.array([3.0, 4.0])
    r = np.array([90.0, 190.0])
    bman = {
        "water_v3": {
            "mask_raster": str(p),
            "cells_parquet": str(cp),
            "features_enabled": True,
            "fac_flatten": True,
            "pin_r_at_water_rows": True,
        }
    }
    cols, fac2, r2, on = bc.water_v3_inference_block(
        bman, qxy, z, fac, r, str(p), sampler=lambda _p, x, y: np.zeros(len(x))
    )
    assert on.tolist() == [True, False]
    assert fac2.tolist() == [0.0, 4.0]  # flattened on the mask only
    assert r2.tolist() == [100.0, 190.0]  # pinned to z_surf on the mask only
    assert set(cols) == set(bc.WATER_V3_FEATURE_COLS)
    assert cols["on_water_v3"].tolist() == [1.0, 0.0]


def test_water_v3_inference_block_is_noop_without_block():
    qxy = np.zeros((3, 2))
    z, fac, r = np.ones(3), np.full(3, 5.0), np.full(3, 2.0)
    cols, fac2, r2, on = bc.water_v3_inference_block(None, qxy, z, fac, r, "unused")
    assert cols == {}
    assert fac2 is fac and r2 is r
    assert not on.any()


@pytest.mark.parametrize("flatten,pin", [(False, True), (True, False)])
def test_water_v3_inference_block_applies_only_what_the_bundle_declares(
    tmp_path, flatten, pin
):
    m = np.zeros((6, 6), bool)
    m[2, 2] = True
    p = _write_mask(tmp_path / f"mask_{flatten}{pin}.tif", m)
    qxy = _centers(np.array([2]), np.array([2]))
    z, fac, r = np.array([50.0]), np.array([7.0]), np.array([40.0])
    bman = {
        "water_v3": {
            "mask_raster": str(p),
            "features_enabled": False,
            "fac_flatten": flatten,
            "pin_r_at_water_rows": pin,
        }
    }
    _, fac2, r2, _ = bc.water_v3_inference_block(bman, qxy, z, fac, r, str(p))
    assert fac2[0] == (0.0 if flatten else 7.0)
    assert r2[0] == (50.0 if pin else 40.0)


def test_water_flatten_factor_hard_is_mask_only(tmp_path):
    m = np.zeros((8, 8), bool)
    m[4, 4] = True
    p = _write_mask(tmp_path / "hard.tif", m)
    rows = np.array([4, 4, 3, 0])
    cols = np.array([4, 5, 3, 0])
    q = _centers(rows, cols)
    f = wm.water_flatten_factor(q[:, 0], q[:, 1], p)
    # on the mask -> 0; every off-mask cell (8-neighbour included) untouched
    assert f.tolist() == [0.0, 1.0, 1.0, 1.0]
    fac = np.array([3.0, 3.0, 3.0, np.nan])
    out = fac * f
    assert out[0] == 0.0 and out[1] == 3.0 and np.isnan(out[3])


def test_water_flatten_factor_ramp_tapers_one_cell_ring(tmp_path):
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    p = _write_mask(tmp_path / "ramp.tif", m)
    # centre (on mask), the 4 rook + 4 diagonal neighbours, and a 2-cell-away cell
    rows = np.array([4, 3, 5, 4, 4, 3, 3, 5, 5, 4, 2])
    cols = np.array([4, 4, 4, 3, 5, 3, 5, 3, 5, 6, 2])
    q = _centers(rows, cols)
    f = wm.water_flatten_factor(q[:, 0], q[:, 1], p, ramp=True)
    assert f[0] == 0.0
    assert (f[1:9] == wm.RAMP_FACTOR).all()  # the whole 8-ring tapers
    assert (f[9:] == 1.0).all()  # two cells out is untouched


def test_water_flatten_factor_ramp_needs_a_masked_neighbour(tmp_path):
    # a wide blob: interior cells stay 0, only the shell outside it tapers
    m = np.zeros((12, 12), bool)
    m[4:8, 4:8] = True
    p = _write_mask(tmp_path / "blob.tif", m)
    rows, cols = np.meshgrid(np.arange(12), np.arange(12), indexing="ij")
    q = _centers(rows.ravel(), cols.ravel())
    f = wm.water_flatten_factor(q[:, 0], q[:, 1], p, ramp=True).reshape(12, 12)
    assert (f[m] == 0.0).all()
    ring = np.zeros((12, 12), bool)
    ring[3:9, 3:9] = True
    ring &= ~m
    assert (f[ring] == wm.RAMP_FACTOR).all()
    assert (f[~(m | ring)] == 1.0).all()
    assert ring.sum() == 20  # 6x6 shell minus the 4x4 blob


def test_water_flatten_factor_outside_lattice_and_window_edge(tmp_path):
    m = np.zeros((6, 6), bool)
    m[0, 0] = True  # flush against the array corner
    p = _write_mask(tmp_path / "edge.tif", m)
    q = np.r_[
        _centers(np.array([0]), np.array([0])),
        _centers(np.array([0]), np.array([1])),
        np.array([[X0 - 10 * RES, Y0 + 10 * RES]]),  # off the lattice
    ]
    f = wm.water_flatten_factor(q[:, 0], q[:, 1], p, ramp=True)
    assert f.tolist() == [0.0, wm.RAMP_FACTOR, 1.0]
