"""100 m REM retention mosaic in build_conus_fac_rem.py.

Covers the two properties the CONUS-scale retention patch depends on:
  * The mosaic grid is pixel-for-pixel aligned with the DTW stacker grid
    (wte_dtw_100m_5070.tif) -- the whole point of retaining REM tiles is that
    they co-register with the surface the stacker scores on.
  * _append_to_mosaic downsamples a 10 m raster to 100 m by averaging, lands the
    values on the correct mosaic cells, and writes ONLY the unbuffered HUC8
    footprint (the halo is masked out), leaving everything else nodata.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio import Affine
from shapely.geometry import Polygon, box

_MOD = Path(__file__).resolve().parents[1] / "utils" / "build_conus_fac_rem.py"
_spec = importlib.util.spec_from_file_location("build_conus_fac_rem", _MOD)
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)


def _write_raster(path, arr, transform, nodata):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:5070",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype("float32"), 1)


def test_mosaic_constants_match_common_grid():
    """The hard-coded mosaic grid must equal wte_dtw_100m_5070.tif exactly."""
    if not Path(bc.COMMON_GRID).exists():
        pytest.skip("common grid raster not present in this environment")
    with rasterio.open(bc.COMMON_GRID) as g:
        assert (g.width, g.height) == (bc.MOSAIC_WIDTH, bc.MOSAIC_HEIGHT)
        assert g.transform == bc.MOSAIC_TRANSFORM
        assert g.crs.to_epsg() == 5070


def test_ensure_mosaic_initialises_to_nodata(tmp_path, monkeypatch):
    """An un-built region must read back as nodata, never a spurious 0 depth."""
    # shrink the grid so the test does not allocate the full CONUS mosaic
    monkeypatch.setattr(bc, "MOSAIC_WIDTH", 8)
    monkeypatch.setattr(bc, "MOSAIC_HEIGHT", 8)
    monkeypatch.setattr(bc, "MOSAIC_TRANSFORM", Affine(100.0, 0, 0, 0, -100.0, 1000.0))
    path = tmp_path / "mosaic.tif"
    bc._ensure_mosaic(path)
    with rasterio.open(path) as r:
        assert r.nodata == bc.MOSAIC_NODATA
        assert (r.width, r.height) == (8, 8)
        assert np.all(r.read(1) == bc.MOSAIC_NODATA)


def test_append_downsamples_aligns_and_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "MOSAIC_WIDTH", 8)
    monkeypatch.setattr(bc, "MOSAIC_HEIGHT", 8)
    monkeypatch.setattr(bc, "MOSAIC_TRANSFORM", Affine(100.0, 0, 0, 0, -100.0, 1000.0))
    mosaic = tmp_path / "mosaic.tif"

    # 10 m REM: 20x20 px over x[300,500], y[500,700], constant depth 5.0.
    rem = tmp_path / "rem_10m.tif"
    _write_raster(
        rem,
        np.full((20, 20), 5.0),
        Affine(10.0, 0, 300.0, 0, -10.0, 700.0),
        nodata=bc.MOSAIC_NODATA,
    )

    # Unbuffered footprint = L-shape covering 3 of the 4 mosaic cells in the
    # window; cell (row3,col4) center (450,650) is outside it (the masked halo).
    poly = Polygon(
        [(300, 500), (500, 500), (500, 600), (400, 600), (400, 700), (300, 700)]
    )

    n = bc._append_to_mosaic(rem, poly, mosaic)
    assert n == 3

    with rasterio.open(mosaic) as r:
        out = r.read(1)
    nd = bc.MOSAIC_NODATA
    # downsample (constant 5.0 -> 5.0) landed on the right cells (alignment) ...
    assert out[3, 3] == pytest.approx(5.0)
    assert out[4, 3] == pytest.approx(5.0)
    assert out[4, 4] == pytest.approx(5.0)
    # ... the halo cell outside the polygon stayed nodata (masking) ...
    assert out[3, 4] == nd
    # ... and nothing spilled outside the HUC8 window.
    touched = np.zeros_like(out, dtype=bool)
    touched[3:5, 3:5] = True
    assert np.all(out[~touched] == nd)


def test_append_is_idempotent_on_rebuild(tmp_path, monkeypatch):
    """A --force rebuild of a HUC8 overwrites its own cells, not neighbours'."""
    monkeypatch.setattr(bc, "MOSAIC_WIDTH", 8)
    monkeypatch.setattr(bc, "MOSAIC_HEIGHT", 8)
    monkeypatch.setattr(bc, "MOSAIC_TRANSFORM", Affine(100.0, 0, 0, 0, -100.0, 1000.0))
    mosaic = tmp_path / "mosaic.tif"
    rem = tmp_path / "rem_10m.tif"
    _write_raster(
        rem,
        np.full((10, 10), 7.0),
        Affine(10.0, 0, 300.0, 0, -10.0, 700.0),
        nodata=bc.MOSAIC_NODATA,
    )
    poly = box(300, 600, 400, 700)
    bc._append_to_mosaic(rem, poly, mosaic)
    with rasterio.open(mosaic) as r:
        first = r.read(1)
    bc._append_to_mosaic(rem, poly, mosaic)
    with rasterio.open(mosaic) as r:
        second = r.read(1)
    np.testing.assert_array_equal(first, second)
    assert second[3, 3] == pytest.approx(7.0)


def test_append_clears_stale_value_when_rebuild_tile_is_nodata(tmp_path, monkeypatch):
    """A rebuild whose new tile is nodata over a previously-finite cell must clear
    the stale depth, not leave the old value behind."""
    monkeypatch.setattr(bc, "MOSAIC_WIDTH", 8)
    monkeypatch.setattr(bc, "MOSAIC_HEIGHT", 8)
    monkeypatch.setattr(bc, "MOSAIC_TRANSFORM", Affine(100.0, 0, 0, 0, -100.0, 1000.0))
    mosaic = tmp_path / "mosaic.tif"
    poly = box(300, 600, 400, 700)
    transform = Affine(10.0, 0, 300.0, 0, -10.0, 700.0)

    # First build: finite depth lands in the footprint cell.
    rem1 = tmp_path / "rem1_10m.tif"
    _write_raster(rem1, np.full((10, 10), 5.0), transform, nodata=bc.MOSAIC_NODATA)
    bc._append_to_mosaic(rem1, poly, mosaic)
    with rasterio.open(mosaic) as r:
        assert r.read(1)[3, 3] == pytest.approx(5.0)

    # Rebuild: the new tile is all-nodata over the same footprint.
    rem2 = tmp_path / "rem2_10m.tif"
    _write_raster(
        rem2,
        np.full((10, 10), bc.MOSAIC_NODATA),
        transform,
        nodata=bc.MOSAIC_NODATA,
    )
    n = bc._append_to_mosaic(rem2, poly, mosaic)
    assert n == 0
    with rasterio.open(mosaic) as r:
        assert r.read(1)[3, 3] == bc.MOSAIC_NODATA


def _backfill_args(out_root, keep_intermediates):
    return SimpleNamespace(
        out_root=str(out_root), force=False, keep_intermediates=keep_intermediates
    )


def _small_grid(monkeypatch):
    monkeypatch.setattr(bc, "MOSAIC_WIDTH", 8)
    monkeypatch.setattr(bc, "MOSAIC_HEIGHT", 8)
    monkeypatch.setattr(bc, "MOSAIC_TRANSFORM", Affine(100.0, 0, 0, 0, -100.0, 1000.0))


def _seed_shard_and_rasters(out_root, huc8, *, rem=5.0, ws_name=None, ws=9.0):
    """Pre-mosaic shard + 10 m REM (and optional WS under *ws_name*) in workdir."""
    (out_root / "fac_rem_shards").mkdir(parents=True, exist_ok=True)
    (out_root / "fac_rem_shards" / f"{huc8}.parquet").write_bytes(b"")
    workdir = out_root / huc8
    workdir.mkdir(exist_ok=True)
    tr = Affine(10.0, 0, 300.0, 0, -10.0, 700.0)
    if rem is not None:
        _write_raster(
            workdir / bc.REM_RASTER, np.full((10, 10), rem), tr, nodata=bc.MOSAIC_NODATA
        )
    if ws_name is not None:
        _write_raster(
            workdir / ws_name, np.full((10, 10), ws), tr, nodata=bc.MOSAIC_NODATA
        )
    return workdir


def test_resume_backfills_both_mosaics_when_shard_predates_marker(
    tmp_path, monkeypatch
):
    """A shard with no marker but surviving 10 m REM + WS backfills BOTH mosaics,
    marks done, and cleans up the 10 m rasters -- never a silent skip."""
    _small_grid(monkeypatch)
    huc8 = "13030102"
    workdir = _seed_shard_and_rasters(tmp_path, huc8, ws_name=bc.WS_RASTER)
    poly = box(300, 600, 400, 700)

    status = bc.build_one_huc8(
        huc8, "label", poly, None, None, _backfill_args(tmp_path, False)
    )
    assert status == bc.STATUS_BUILT
    assert bc._mosaic_marker(tmp_path, huc8).exists()
    with rasterio.open(tmp_path / bc.REM_MOSAIC) as r:
        assert r.read(1)[3, 3] == pytest.approx(5.0)
    with rasterio.open(tmp_path / bc.WS_MOSAIC) as r:
        assert r.read(1)[3, 3] == pytest.approx(9.0)
    # default cleanup removed both 10 m rasters once the mosaics captured them
    assert not (workdir / bc.REM_RASTER).exists()
    assert not (workdir / bc.WS_RASTER).exists()


def test_resume_backfill_honours_legacy_ws_filename(tmp_path, monkeypatch):
    """A pre-rename run whose WS file is fac_head_depth_idw_fill_10m.tif must still
    populate the WS mosaic and mark complete."""
    _small_grid(monkeypatch)
    huc8 = "13030102"
    _seed_shard_and_rasters(tmp_path, huc8, ws_name=bc.WS_RASTER_LEGACY)

    status = bc.build_one_huc8(
        huc8,
        "label",
        box(300, 600, 400, 700),
        None,
        None,
        _backfill_args(tmp_path, True),
    )
    assert status == bc.STATUS_BUILT
    assert bc._mosaic_marker(tmp_path, huc8).exists()
    with rasterio.open(tmp_path / bc.WS_MOSAIC) as r:
        assert r.read(1)[3, 3] == pytest.approx(9.0)


def test_resume_needs_force_when_ws_missing(tmp_path, monkeypatch):
    """REM present but NO water-surface raster -> retention is incomplete: the WS
    mosaic and marker are not written, and the status is NEEDS_FORCE."""
    _small_grid(monkeypatch)
    huc8 = "13030102"
    workdir = _seed_shard_and_rasters(tmp_path, huc8, ws_name=None)

    status = bc.build_one_huc8(
        huc8,
        "label",
        box(300, 600, 400, 700),
        None,
        None,
        _backfill_args(tmp_path, False),
    )
    assert status == bc.STATUS_NEEDS_FORCE
    assert not bc._mosaic_marker(tmp_path, huc8).exists()
    assert not (tmp_path / bc.WS_MOSAIC).exists()
    # the surviving 10 m REM is kept so a --force rebuild can regenerate the WS
    assert (workdir / bc.REM_RASTER).exists()


def test_resume_skips_when_shard_and_marker_present(tmp_path, monkeypatch):
    """Shard + marker -> a clean skip, no mosaic write attempted."""
    huc8 = "13030102"
    (tmp_path / "fac_rem_shards").mkdir(parents=True)
    (tmp_path / "fac_rem_shards" / f"{huc8}.parquet").write_bytes(b"")
    marker = bc._mosaic_marker(tmp_path, huc8)
    marker.parent.mkdir(parents=True)
    marker.touch()

    status = bc.build_one_huc8(
        huc8,
        "label",
        box(300, 600, 400, 700),
        None,
        None,
        _backfill_args(tmp_path, False),
    )
    assert status == bc.STATUS_SKIPPED
    assert not (tmp_path / bc.REM_MOSAIC).exists()


def test_resume_needs_force_when_rem_gone_and_no_marker(tmp_path, monkeypatch):
    """Shard without a marker and no surviving 10 m REM -> no silent success: the
    mosaic is not fabricated, no marker is written, status is NEEDS_FORCE."""
    huc8 = "13030102"
    (tmp_path / "fac_rem_shards").mkdir(parents=True)
    (tmp_path / "fac_rem_shards" / f"{huc8}.parquet").write_bytes(b"")

    status = bc.build_one_huc8(
        huc8,
        "label",
        box(300, 600, 400, 700),
        None,
        None,
        _backfill_args(tmp_path, False),
    )
    assert status == bc.STATUS_NEEDS_FORCE
    assert not bc._mosaic_marker(tmp_path, huc8).exists()
    assert not (tmp_path / bc.REM_MOSAIC).exists()
