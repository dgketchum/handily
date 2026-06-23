"""Unit tests for utils/build_stacker_features.py.

Covers the two correctness-critical pure functions: the coarse-raster sampler
(nearest cell, NaN off-grid/nodata -- never fabricated) and load_clean_wells
(the leakage firewall: unconfined-only, dedup by canonical_id, hard-exclude the
retirement manifest).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

_MOD = Path(__file__).resolve().parents[1] / "utils" / "build_stacker_features.py"
_spec = importlib.util.spec_from_file_location("build_stacker_features", _MOD)
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)


def test_sample_coarse_nearest_and_nodata(tmp_path):
    # 2x2 grid, 1000 m cells, origin (0, 2000); rows go down (e = -1000).
    arr = np.array([[10.0, 20.0], [30.0, -9999.0]], dtype=np.float32)
    tr = Affine(1000.0, 0.0, 0.0, 0.0, -1000.0, 2000.0)
    p = tmp_path / "c.tif"
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:5070",
        transform=tr,
        nodata=-9999.0,
    ) as dst:
        dst.write(arr, 1)
    # Cell centers: (500,1500)->10, (1500,1500)->20, (500,500)->30, (1500,500)->nodata.
    x = np.array([500.0, 1500.0, 500.0, 1500.0, -10.0])
    y = np.array([1500.0, 1500.0, 500.0, 500.0, 1500.0])
    v = bs.sample_coarse(str(p), x, y)
    assert v[0] == 10.0 and v[1] == 20.0 and v[2] == 30.0
    assert np.isnan(v[3])  # nodata cell -> NaN, not -9999
    assert np.isnan(v[4])  # off-grid -> NaN


def _wells_parquet(tmp_path):
    df = pd.DataFrame(
        {
            "canonical_id": ["a", "a", "b", "c", "d", "e"],
            "source": ["nwis", "state", "state", "state", "state", "state"],
            "confinement_class": [
                "unconfined",
                "unconfined",
                "unconfined_marginal",
                "confined",  # must be dropped
                "unconfined",
                "unconfined",
            ],
            "obs_count": [1, 1, 1, 1, 1, 1],
            "longitude": [-106.0, -106.0, -107.0, -108.0, -109.0, -110.0],
            "latitude": [32.0, 32.0, 33.0, 34.0, 35.0, 36.0],
            "mean_dtw": [5.0, 5.0, 8.0, 3.0, 12.0, np.nan],  # e dropped (no target)
            "land_surface_elev_m": [1200.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0],
        }
    )
    p = tmp_path / "wells.parquet"
    df.to_parquet(p, index=False)
    return p


def _const_raster(path, val, nodata=-9999.0):
    # Same 2x2 / 1000 m / origin (0,2000) grid as the sampler test.
    arr = np.full((2, 2), val, dtype=np.float32)
    tr = Affine(1000.0, 0.0, 0.0, 0.0, -1000.0, 2000.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:5070",
        transform=tr,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


def _latlon_raster(path, val, nodata=-9999.0):
    # 2x2 / 1 deg EPSG:4326 grid, origin (-110, 35), rows go down (lat descending).
    # Cell centers: lon -109.5/-108.5, lat 34.5/33.5.
    arr = np.full((2, 2), val, dtype=np.float32)
    tr = Affine(1.0, 0.0, -110.0, 0.0, -1.0, 35.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=tr,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


def test_add_covariates_arithmetic_and_nan_passthrough(tmp_path):
    coarse = tmp_path / "coarse.tif"
    accum = tmp_path / "accum.tif"
    other = tmp_path / "other.tif"  # stands in for slope/tri/dist
    ai = tmp_path / "ai.tif"  # gridMET aridity index (EPSG:4326)
    precip = tmp_path / "precip.tif"  # gridMET mean annual precip (EPSG:4326)
    _const_raster(coarse, 1000.0)
    _const_raster(accum, 99.0)  # log1p(99) = log(100)
    _const_raster(other, 7.0)
    _latlon_raster(ai, 0.15)  # arid
    _latlon_raster(precip, 250.0)
    # One in-grid well (5070 cell center, lon/lat inside the 4326 raster) and one
    # off-both-grids well -> NaN everywhere.
    df = pd.DataFrame(
        {
            "x5070": [500.0, -10.0],
            "y5070": [1500.0, 1500.0],
            "longitude": [-109.5, -200.0],  # 2nd lon off the 4326 grid -> NaN aridity
            "latitude": [34.5, 34.5],
            "land_surface_elev_m": [1250.0, 1250.0],
        }
    )
    out = bs.add_covariates(
        df.copy(),
        coarse_surface=str(coarse),
        slope=str(other),
        tri=str(other),
        dist_stream=str(other),
        accum=str(accum),
        gridmet_ai=str(ai),
        gridmet_p=str(precip),
    )
    assert out.elev_above_coarse_m.iloc[0] == 250.0  # 1250 - 1000
    assert out.slope_deg.iloc[0] == 7.0
    assert np.isclose(out.log_drainage_area.iloc[0], np.log(100.0))  # log1p(99)
    # gridMET sampled at lon/lat (EPSG:4326), NOT at x5070/y5070.
    assert np.isclose(out.aridity_index.iloc[0], 0.15)  # float32 raster storage
    assert out.mean_annual_precip_mm.iloc[0] == 250.0
    # Off-grid well: every covariate NaN, not fabricated.
    for c in bs.COVARIATES:
        assert np.isnan(out[c].iloc[1])


def test_load_clean_wells_firewall(tmp_path):
    wells = _wells_parquet(tmp_path)
    manifest = tmp_path / "retired.parquet"
    pd.DataFrame({"canonical_id": ["d"]}).to_parquet(manifest, index=False)  # retire d

    clean = bs.load_clean_wells(str(wells), str(manifest))
    ids = set(clean.canonical_id)
    assert "a" in ids  # nwis kept (source-agnostic)
    assert "b" in ids
    assert "c" not in ids  # confined dropped
    assert "d" not in ids  # retired (manifest) dropped
    assert "e" not in ids  # no mean_dtw dropped
    assert len(clean) == len(clean.drop_duplicates("canonical_id"))  # 'a' deduped
    assert (clean.canonical_id == "a").sum() == 1
