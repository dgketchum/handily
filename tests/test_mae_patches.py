"""Unit tests for the neighborhood-MAE pure logic: lattice snapping, patch geometry,
multi-scale pooling, nodata handling, and robust normalization stats.

Loads utils/build_mae_patches.py by path (repo test convention) so the module's
rasterio import at load time is exercised but no raster is opened by these tests.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bmp = _load("build_mae_patches")


def test_snap_to_lattice_origin_and_offsets():
    x0, y0 = bmp.LATTICE_ORIGIN
    col, row = bmp.snap_to_lattice(np.array([x0]), np.array([y0]))
    assert col[0] == 0 and row[0] == 0
    # +150 m east, 250 m south of origin -> col 1, row 2 (100 m cells, y decreases south)
    col, row = bmp.snap_to_lattice(np.array([x0 + 150.0]), np.array([y0 - 250.0]))
    assert col[0] == 1 and row[0] == 2


def test_snap_matches_sample_coarse_convention():
    x0, y0 = bmp.LATTICE_ORIGIN
    x = np.array([x0 + 99.9, x0 + 100.1])
    y = np.array([y0 - 99.9, y0 - 100.1])
    col, row = bmp.snap_to_lattice(x, y)
    assert list(col) == [0, 1]
    assert list(row) == [0, 1]


def test_lattice_center_roundtrip():
    col = np.array([0, 5, 4999])
    row = np.array([0, 7, 3000])
    x, y = bmp.lattice_center_xy(col, row)
    c2, r2 = bmp.snap_to_lattice(x, y)
    assert np.array_equal(c2, col) and np.array_equal(r2, row)


def test_pool_full_basic_and_nan():
    a = np.arange(16, dtype="float32").reshape(4, 4)
    out = bmp.pool_full(a, 2)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(2.5)  # mean of [0,1,4,5]
    a[0, 0] = np.nan  # nan ignored -> mean of [1,4,5]
    out = bmp.pool_full(a, 2)
    assert out[0, 0] == pytest.approx((1 + 4 + 5) / 3)


def test_pool_full_all_nan_block_stays_nan():
    a = np.full((2, 2), np.nan, "float32")
    out = bmp.pool_full(a, 2)
    assert np.isnan(out[0, 0])


def test_pool_full_factor_one_identity():
    a = np.arange(9, dtype="float32").reshape(3, 3)
    assert bmp.pool_full(a, 1) is a


def test_gather_windows_shape_center_and_padding():
    full = np.arange(100 * 100, dtype="float32").reshape(100, 100)
    # center at (10,10): a half=32 window runs to row/col -22 -> upper-left corner pads
    win = bmp.gather_windows(full, np.array([10]), np.array([10]))  # default half=32
    assert win.shape == (1, 2 * bmp.HALF, 2 * bmp.HALF)
    c = bmp.HALF
    assert win[0, c, c] == pytest.approx(full[10, 10])  # center pixel exact
    assert np.isnan(win[0, 0, 0])  # offset (-32,-32) -> row/col -22 OOB -> NaN pad
    # a fully-interior center (50,50) has no padding
    win2 = bmp.gather_windows(full, np.array([50]), np.array([50]))
    assert np.isfinite(win2[0]).all()


def test_gather_windows_no_wraparound():
    full = np.arange(200 * 200, dtype="float32").reshape(200, 200)
    win = bmp.gather_windows(full, np.array([0]), np.array([100]))
    c = bmp.HALF
    assert np.isnan(win[0, c, 0])  # one col left of col 0 -> OOB, not wrapped
    assert win[0, c, c] == pytest.approx(full[100, 0])


def test_gather_windows_custom_half():
    full = np.arange(50 * 50, dtype="float32").reshape(50, 50)
    win = bmp.gather_windows(full, np.array([25]), np.array([25]), half=4)
    assert win.shape == (1, 8, 8)
    assert win[0, 4, 4] == pytest.approx(full[25, 25])


def test_coarse_center_mapping_consistency():
    # a fine cell and its coarse-grid center map by integer division (build + extract
    # both use col//f, row//f), so the same cell lands on the same coarse window center.
    fine_col, fine_row, f = np.array([1234]), np.array([5678]), 10
    assert (fine_col // f)[0] == 123 and (fine_row // f)[0] == 567


def test_robust_stats():
    v = np.array([1.0, 2, 3, 4, 5, np.nan])
    med, iqr = bmp.robust_stats(v)
    assert med == pytest.approx(3.0)
    assert iqr == pytest.approx(2.0)  # p75-p25 of 1..5 = 4-2
    assert bmp.robust_stats(np.array([np.nan])) == (0.0, 1.0)
    _, iqr0 = bmp.robust_stats(np.full(5, 7.0))
    assert iqr0 > 0


def test_apply_norm_clip_and_nan_impute():
    a = np.array([0.0, 10.0, -10.0, np.nan])
    out = bmp.apply_norm(a, med=0.0, iqr=1.0, clip=5.0)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(5.0)  # clipped
    assert out[2] == pytest.approx(-5.0)
    assert out[3] == pytest.approx(0.0)  # nan imputed to 0 after normalization


# ---------------------------------------------------------------------- manifest
@pytest.fixture()
def _manifest_globals():
    # apply_manifest rebinds module globals; snapshot/restore so tests stay isolated
    saved = (
        bmp.CHANNEL_SPECS,
        bmp.CHANNEL_NAMES,
        bmp._DEM,
        bmp._COV_CACHE,
        bmp.FOOTPRINT_CHANNEL,
    )
    yield
    (
        bmp.CHANNEL_SPECS,
        bmp.CHANNEL_NAMES,
        bmp._DEM,
        bmp._COV_CACHE,
        bmp.FOOTPRINT_CHANNEL,
    ) = saved


def test_apply_manifest_rebinds_roster(_manifest_globals):
    bmp.apply_manifest(
        {
            "dem": "/mirror/dem.tif",
            "cache_dir": "/mirror/cache",
            "channels": [
                {"name": "dem_rel", "path": "/mirror/dem.tif", "transform": "dem_rel"},
                {"name": "slope_deg", "path": "/mirror/slope.tif"},
                {"name": "perm_logk_m2", "path": "/mirror/perm.tif", "transform": 0.01},
            ],
        }
    )
    assert bmp.CHANNEL_NAMES == ["dem_rel", "slope_deg", "perm_logk_m2"]
    assert bmp.CHANNEL_SPECS[1] == ("slope_deg", "/mirror/slope.tif", None)
    assert bmp.CHANNEL_SPECS[2][2] == pytest.approx(0.01)
    assert isinstance(bmp.CHANNEL_SPECS[2][2], float)
    assert bmp._DEM == "/mirror/dem.tif"
    assert str(bmp._COV_CACHE) == "/mirror/cache"


def test_apply_manifest_requires_one_dem_rel(_manifest_globals):
    with pytest.raises(SystemExit, match="dem_rel"):
        bmp.apply_manifest(
            {"dem": "/d.tif", "channels": [{"name": "slope", "path": "/s.tif"}]}
        )
    with pytest.raises(SystemExit, match="dem_rel"):
        bmp.apply_manifest(
            {
                "dem": "/d.tif",
                "channels": [
                    {"name": "a", "path": "/a.tif", "transform": "dem_rel"},
                    {"name": "b", "path": "/b.tif", "transform": "dem_rel"},
                ],
            }
        )


def test_apply_manifest_rejects_duplicate_names(_manifest_globals):
    with pytest.raises(SystemExit, match="duplicate"):
        bmp.apply_manifest(
            {
                "dem": "/d.tif",
                "channels": [
                    {"name": "dem_rel", "path": "/d.tif", "transform": "dem_rel"},
                    {"name": "x", "path": "/1.tif"},
                    {"name": "x", "path": "/2.tif"},
                ],
            }
        )


def test_apply_manifest_footprint_validation(_manifest_globals):
    chans = [
        {"name": "dem_rel", "path": "/d.tif", "transform": "dem_rel"},
        {"name": "slope", "path": "/s.tif"},
    ]
    with pytest.raises(SystemExit, match="footprint_channel"):
        bmp.apply_manifest(
            {"dem": "/d.tif", "channels": chans, "footprint_channel": "nope"}
        )
    # footprint before dem_rel would AND into an uninitialized mask
    rev = [chans[1], chans[0]]
    with pytest.raises(SystemExit, match="after the dem_rel"):
        bmp.apply_manifest(
            {"dem": "/d.tif", "channels": rev, "footprint_channel": "slope"}
        )
    bmp.apply_manifest(
        {"dem": "/d.tif", "channels": chans, "footprint_channel": "slope"}
    )
    assert bmp.FOOTPRINT_CHANNEL == "slope"


def _write_raster(path, arr, nodata):
    import rasterio
    from rasterio.transform import Affine

    t = Affine(100.0, 0, bmp.LATTICE_ORIGIN[0], 0, -100.0, bmp.LATTICE_ORIGIN[1])
    prof = dict(
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:5070",
        transform=t,
        nodata=nodata,
    )
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr.astype("float32"), 1)


def test_sample_centers_respects_footprint(tmp_path, monkeypatch, _manifest_globals):
    # DEM valid everywhere (the elev48i0100a failure mode: ocean/cross-border cells
    # carry real values); footprint valid only for col >= 50 -> no center left of it
    W, H = 200, 150
    rng = np.random.default_rng(1)
    dem = rng.normal(1500, 100, (H, W))
    _write_raster(tmp_path / "dem.tif", dem, 32767)
    fp = rng.normal(800, 50, (H, W))
    fp[:, :50] = -9999
    _write_raster(tmp_path / "fp.tif", fp, -9999)
    monkeypatch.setattr(bmp, "REF_W", W)
    monkeypatch.setattr(bmp, "REF_H", H)
    bmp.apply_manifest(
        {
            "dem": str(tmp_path / "dem.tif"),
            "cache_dir": str(tmp_path / "nocache"),
            "channels": [
                {
                    "name": "dem_rel",
                    "path": str(tmp_path / "dem.tif"),
                    "transform": "dem_rel",
                },
                {"name": "fp", "path": str(tmp_path / "fp.tif")},
            ],
            "footprint_channel": "fp",
        }
    )
    df = bmp.sample_centers(20, seed=0)
    assert len(df) == 20
    assert (df["col"] >= 50).all()


def test_extract_stack_mask_is_dem_and_footprint(
    tmp_path, monkeypatch, _manifest_globals
):
    # fine-scale mask must zero pixels where the footprint is nodata even though the
    # DEM is valid there (window straddles the footprint boundary at col 50)
    W, H = 200, 150
    rng = np.random.default_rng(2)
    dem = rng.normal(1500, 100, (H, W))
    _write_raster(tmp_path / "dem.tif", dem, 32767)
    fp = rng.normal(800, 50, (H, W))
    fp[:, :50] = -9999
    _write_raster(tmp_path / "fp.tif", fp, -9999)
    monkeypatch.setattr(bmp, "REF_W", W)
    monkeypatch.setattr(bmp, "REF_H", H)
    bmp.apply_manifest(
        {
            "dem": str(tmp_path / "dem.tif"),
            "cache_dir": str(tmp_path / "nocache"),
            "channels": [
                {
                    "name": "dem_rel",
                    "path": str(tmp_path / "dem.tif"),
                    "transform": "dem_rel",
                },
                {"name": "fp", "path": str(tmp_path / "fp.tif")},
            ],
            "footprint_channel": "fp",
        }
    )
    col, row = np.array([70]), np.array([75])  # window cols 38..101 straddle 50
    out = np.zeros((1, len(bmp.SCALES), 2, bmp.WIN, bmp.WIN), "float32")
    mask = np.zeros((1, len(bmp.SCALES), bmp.WIN, bmp.WIN), "uint8")
    bmp.extract_stack(col, row, out, mask_out=mask, norm=None)
    fine = mask[0, 0]  # window x -> raster col = 70 - 32 + x
    assert (fine[:, : 50 - 38] == 0).all()  # footprint nodata -> masked out
    assert (fine[:, 50 - 38 :] == 1).all()  # dem+footprint valid -> kept


def test_build_end_to_end_small(tmp_path, monkeypatch, _manifest_globals):
    # exercises the streaming build path (raw f16 pass -> subsample stats ->
    # blockwise in-place normalize) on tiny on-lattice synthetic rasters
    import json

    import rasterio
    from rasterio.transform import Affine

    W, H = 200, 150
    t = Affine(100.0, 0, bmp.LATTICE_ORIGIN[0], 0, -100.0, bmp.LATTICE_ORIGIN[1])
    rng = np.random.default_rng(3)
    prof = dict(
        driver="GTiff",
        width=W,
        height=H,
        count=1,
        dtype="float32",
        crs="EPSG:5070",
        transform=t,
        nodata=32767,
    )
    dem_path = tmp_path / "dem.tif"
    dem = rng.normal(1500, 100, (H, W)).astype("float32")
    dem[:5, :5] = 32767  # off-CONUS corner
    with rasterio.open(dem_path, "w", **prof) as d:
        d.write(dem, 1)
    ch_path = tmp_path / "slope.tif"
    slope = np.abs(rng.normal(5, 2, (H, W))).astype("float32")
    with rasterio.open(ch_path, "w", **dict(prof, nodata=-9999)) as d:
        d.write(slope, 1)

    monkeypatch.setattr(bmp, "REF_W", W)
    monkeypatch.setattr(bmp, "REF_H", H)
    bmp.apply_manifest(
        {
            "dem": str(dem_path),
            "cache_dir": str(tmp_path / "nocache"),
            "channels": [
                {"name": "dem_rel", "path": str(dem_path), "transform": "dem_rel"},
                {"name": "slope", "path": str(ch_path)},
            ],
        }
    )
    out = tmp_path / "patches"
    bmp.build(out, 40, seed=0)

    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_patches"] == 40 and meta["n_channels"] == 2
    assert meta["norm_subsample"] == 40
    p = np.asarray(
        np.memmap(
            out / "patches.f16", "float16", "r", shape=tuple(meta["patches_shape"])
        ),
        "float32",
    )
    assert np.isfinite(p).all()  # nodata imputed to 0 post-norm
    assert np.abs(p).max() <= 5.0 + 1e-2  # +-5 IQR clip held
    stats = json.loads((out / "norm_stats.json").read_text())
    med = stats["per_channel_scale"]["slope"][0]["median"]
    assert abs(med - 5.0) < 1.0  # ~|N(5,2)| median recovered from subsample
    m = np.memmap(out / "mask.u8", "uint8", "r", shape=(40, 4, bmp.WIN, bmp.WIN))
    assert 0 < np.asarray(m).mean() <= 1.0
