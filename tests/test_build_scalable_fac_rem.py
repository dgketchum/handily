"""Canal-aware support in build_scalable_fac_rem.py.

Covers the two properties the canal-support variant depends on:
  * gate_canal_by_irrigation keeps canal cells only within the irrigation buffer
    (and returns nothing when either input is empty).
  * build_canal_support rasterizes canal lines onto the DEM grid, gates them by
    the irrigation mask (MIrAD nodata reads as not-irrigated), and ORs the result
    with the GSW permanent-water support.
"""

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

_MOD = Path(__file__).resolve().parents[1] / "utils" / "build_scalable_fac_rem.py"
_spec = importlib.util.spec_from_file_location("build_scalable_fac_rem", _MOD)
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)

TRANSFORM = from_origin(0.0, 100.0, 10.0, 10.0)  # 10x10 grid, 10 m cells


def _write_raster(path, arr, dtype, nodata=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs="EPSG:5070",
        transform=TRANSFORM,
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def test_gate_keeps_only_canal_near_irrigation():
    canal = np.zeros((10, 10), dtype=bool)
    canal[4, :] = True  # canal across the whole row
    irr = np.zeros((10, 10), dtype=bool)
    irr[4, 0:3] = True  # irrigated only on the left
    out = bs.gate_canal_by_irrigation(canal, irr, buffer_cells=2)
    assert out[4, 0:5].all()  # on + within 2 cells of irrigation
    assert not out[4, 6:].any()  # beyond the buffer
    assert not out[5, :].any()  # gate never adds non-canal cells


def test_gate_empty_inputs():
    canal = np.zeros((5, 5), dtype=bool)
    irr = np.zeros((5, 5), dtype=bool)
    assert not bs.gate_canal_by_irrigation(canal, irr, 3).any()
    canal[2, 2] = True
    assert not bs.gate_canal_by_irrigation(canal, irr, 3).any()  # no irrigation
    assert not bs.gate_canal_by_irrigation(np.zeros((5, 5), bool), ~irr, 3).any()


def test_build_canal_support_composes_gsw_and_gated_canal(tmp_path):
    dem = np.zeros((10, 10), dtype="float32")
    _write_raster(tmp_path / "dem.tif", dem, "float32")

    gsw = np.zeros((10, 10), dtype="uint8")
    gsw[0, 0] = 1
    _write_raster(tmp_path / "gsw.tif", gsw, "uint8")

    # Irrigated on the left half; a 255-nodata block on the right must read as
    # NOT irrigated (MIrAD nodata = unmapped, gates as dry).
    irr = np.zeros((10, 10), dtype="uint8")
    irr[:, 0:5] = 1
    irr[:, 8:] = 255
    _write_raster(tmp_path / "irr.tif", irr, "uint8", nodata=255)

    # One canal across row y=55 (row 4), one canal far from irrigation impossible
    # here (grid fully within buffer reach), so gate with a small buffer instead.
    canals = gpd.GeoDataFrame(
        {"fcode": [33600]},
        geometry=[LineString([(1.0, 55.0), (99.0, 55.0)])],
        crs="EPSG:5070",
    )
    canals.to_file(tmp_path / "canals.fgb", driver="FlatGeobuf")

    out = bs.build_canal_support(
        tmp_path / "dem.tif",
        tmp_path / "gsw.tif",
        tmp_path / "support.tif",
        tmp_path / "canals.fgb",
        tmp_path / "irr.tif",
        buffer_m=20.0,  # 2 cells: canal kept over cols 0-6 (irr cols 0-4 + 2)
        force=False,
    )
    with rasterio.open(out) as ds:
        sup = ds.read(1)
    assert sup[0, 0] == 1  # GSW carried through
    assert sup[4, 0:7].all()  # canal cells within 2 cells of irrigation
    assert not sup[4, 8:].any()  # canal beyond the buffer (incl. nodata zone) dropped
    assert sup[np.arange(10) != 4][1:].sum() == 0  # nothing invented off-canal

    # Idempotent skip: existing output is reused unless force.
    before = out.stat().st_mtime_ns
    bs.build_canal_support(
        tmp_path / "dem.tif",
        tmp_path / "gsw.tif",
        tmp_path / "support.tif",
        tmp_path / "canals.fgb",
        tmp_path / "irr.tif",
        buffer_m=20.0,
        force=False,
    )
    assert out.stat().st_mtime_ns == before


def test_build_canal_support_no_canals_in_window(tmp_path):
    dem = np.zeros((10, 10), dtype="float32")
    _write_raster(tmp_path / "dem.tif", dem, "float32")
    gsw = np.zeros((10, 10), dtype="uint8")
    gsw[2, 2] = 1
    _write_raster(tmp_path / "gsw.tif", gsw, "uint8")
    irr = np.ones((10, 10), dtype="uint8")
    _write_raster(tmp_path / "irr.tif", irr, "uint8", nodata=255)
    # Canal entirely outside the DEM window.
    canals = gpd.GeoDataFrame(
        {"fcode": [33600]},
        geometry=[LineString([(1000.0, 1000.0), (1100.0, 1000.0)])],
        crs="EPSG:5070",
    )
    canals.to_file(tmp_path / "canals.fgb", driver="FlatGeobuf")
    out = bs.build_canal_support(
        tmp_path / "dem.tif",
        tmp_path / "gsw.tif",
        tmp_path / "support.tif",
        tmp_path / "canals.fgb",
        tmp_path / "irr.tif",
        buffer_m=300.0,
        force=False,
    )
    with rasterio.open(out) as ds:
        sup = ds.read(1)
    # Support degrades to exactly the GSW layer.
    assert (sup == gsw).all()
