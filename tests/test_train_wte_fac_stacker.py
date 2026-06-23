"""Unit tests for utils/train_wte_fac_stacker.py.

Covers the two pieces that must be right independent of the model: the per-state Ma
coalesce (first finite value, in raster order) and the leave-one-block-out OOF
bookkeeping (eligible blocks only; every eligible well predicted by a model not
trained on its block; tiny blocks dropped from train and eval).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

_MOD = Path(__file__).resolve().parents[1] / "utils" / "train_wte_fac_stacker.py"
_spec = importlib.util.spec_from_file_location("train_wte_fac_stacker", _MOD)
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)


def _const_raster(path, left, val):
    # 4x4 deg grid, 1-deg cells, EPSG:4326, constant value, nodata -9999.
    arr = np.full((4, 4), val, dtype=np.float32)
    t = Affine(1.0, 0.0, left, 0.0, -1.0, 34.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
        crs="EPSG:4326", transform=t, nodata=-9999.0,
    ) as dst:
        dst.write(arr, 1)


def test_sample_ma_coalesces_first_finite(tmp_path):
    a = tmp_path / "a.tif"  # covers lon [-110,-106]
    b = tmp_path / "b.tif"  # covers lon [-108,-104]; overlap [-108,-106]
    _const_raster(a, -110.0, 10.0)
    _const_raster(b, -108.0, 20.0)
    lon = np.array([-109.0, -105.0, -100.0, -107.0])  # A-only, B-only, neither, both
    lat = np.array([32.0, 32.0, 32.0, 32.0])
    out = tr.sample_ma(lon, lat, [str(a), str(b)])
    assert out[0] == 10.0  # A only
    assert out[1] == 20.0  # B only
    assert np.isnan(out[2])  # neither
    assert out[3] == 10.0  # both -> first raster (A) wins


def test_oof_blocked_bookkeeping():
    rng = np.random.default_rng(0)
    # Folds large enough for HGB to split past min_samples_leaf (real folds ~21k).
    blocks = np.array(["A"] * 500 + ["B"] * 500 + ["C"] * 5)  # C is tiny
    n = len(blocks)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    df = pd.DataFrame({"huc4": blocks, "f1": f1, "f2": f2, "y": 2 * f1 - f2})
    oof, keep, eligible = tr.oof_blocked(df, ("f1", "f2"), "y", "huc4", min_n=50)

    assert eligible == ["A", "B"]  # C dropped (below min_n)
    assert keep.sum() == 1000
    assert np.isfinite(oof[keep]).all()  # every eligible well predicted
    assert np.isnan(oof[~keep]).all()  # tiny-block wells never predicted
    # The blend transfers across the two blocks -> OOF correlates with truth.
    yk = df.y.to_numpy()[keep]
    assert np.corrcoef(oof[keep], yk)[0, 1] > 0.3
