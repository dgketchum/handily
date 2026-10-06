"""Build the Nevada NWI data packet: statewide handily mosaics plus accuracy tables.

Source is the 72-basin 100 m ramp render of 2026-09-16, arm
``gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold`` (model of record,
``notes/NV_NWI_TODO.md`` item 2). Each basin render is clipped to its WBD HUC8
polygon and pasted into one statewide 100 m EPSG:5070 grid, one Cloud Optimized
GeoTIFF per layer. After writing, every basin is re-read and compared against
the mosaic inside its polygon (max abs diff must be 0), and no valid cell may be
claimed by two basins.

The accuracy tables are re-cut from the frozen statewide eval
(``/data/ssd2/handily/nv/regional/statewide_eval_ndwrlbl_ufold/``): handily
(the ndwrlbl_ufold raster) and Ma (native tiles sampled at the site) only. The
report's Panel A is the held-out water-table set; its Panel B is all NDWR sites,
with the pumping-only subset as a separate row group. Sign convention:
``residual = predicted - observed`` (m), positive = predicted too deep.

File names are fixed by ``reports/nwi/nwi_handily_tech_report.md`` ("Data
Package Contents").

    uv run --directory /home/dgketchum/code/handily python \
        /home/dgketchum/code/handily/utils/package_nv_nwi_rasters.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.shutil
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill
from openpyxl.utils import get_column_letter
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin, rowcol

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "utils"))
sys.path.insert(0, str(_REPO / "figures" / "nwi"))

from build_share_workbook import (  # noqa: E402
    BAND_TINT,
    RULE,
    WIN,
    _header,
    _title,
)
from nwi_common import HUC8_POLYS, rendered_basins  # noqa: E402

ARM = "gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold"
RENDER_DATE = "2026-09-16"
HUC8_ROOT = Path("/data/ssd2/handily/huc8")
EVAL_DIR = Path("/data/ssd2/handily/nv/regional/statewide_eval_ndwrlbl_ufold")
OUT_DIR = Path("/data/ssd2/handily/share/nwi_packet")
STAGE_DIR = Path("/data/ssd2/handily/share/nwi_packet_staging")
CELL = 100.0
NODATA = -9999.0
N_PRIMARY = 8877

#: Shipped layer -> (source file, band names, unit, description).
LAYERS = {
    "handily_dtw_nv_100m": (
        "gnn_dtw_100m.tif",
        ("dtw_m",),
        "m",
        "handily depth to water, m below land surface (positive down; negative = "
        "water table above the surface, unclamped)",
    ),
    "handily_wte_nv_100m": (
        "gnn_wte_100m.tif",
        ("wte_m",),
        "m",
        "handily water-table elevation, m above sea level (NAVD88, per the 100 m "
        "DEM used for dtw = z_surf - wte)",
    ),
    "handily_sigma_nv_100m": (
        "gnn_sigma_100m.tif",
        ("sigma_laplace_b_m",),
        "m",
        "handily uncertainty, Laplace scale b in m (not a standard deviation; "
        "nominal 63.2% of residuals within +-1 b)",
    ),
    "handily_p_shallow_nv_100m": (
        "gnn_p_dtw_lt_100m.tif",
        ("p_dtw_lt_2m", "p_dtw_lt_5m", "p_dtw_lt_10m"),
        "probability (0-1, dimensionless)",
        "handily probability that depth to water is less than 2, 5 and 10 m "
        "(bands 1-3)",
    ),
}

#: Eval set -> label used in the shipped tables (the report's panel names).
SETS = {
    "A_primary_eligible": "A_heldout_water_table",
    "D_context_all_targets": "B_all_ndwr_sites",
    "B_secondary_pumping": "B_pumping_wells_only",
}
PREDICTORS = {"ndwrlbl_ufold raster": "handily", "Ma raster (resampled here)": "Ma"}
PANEL_COLS = [
    "set",
    "stratum_kind",
    "stratum",
    "n",
    "predictor",
    "MAD_m",
    "median_resid_m",
    "bias_m",
    "RMSE_m",
    "p95_abs_resid_m",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------
def check_render(basins: list[str]) -> dict:
    """Every basin carries the arm, one model, and tifs written on the render date."""
    models, dates = set(), set()
    for b in basins:
        d = HUC8_ROOT / b / "gnn" / ARM
        run = json.loads((d / "infer_run.json").read_text())
        if run["basin"] != b:
            raise ValueError(f"{b}: infer_run.json names basin {run['basin']}")
        models.add(run["model"])
        for src, *_ in LAYERS.values():
            ts = (d / src).stat().st_mtime
            dates.add(dt.date.fromtimestamp(ts).isoformat())
    if len(models) != 1:
        raise ValueError(f"basins disagree on model: {models}")
    if dates != {RENDER_DATE}:
        raise ValueError(f"render tif dates {sorted(dates)} != {RENDER_DATE}")
    run0 = json.loads(
        (HUC8_ROOT / basins[0] / "gnn" / ARM / "infer_run.json").read_text()
    )
    log(
        f"render check: {len(basins)} basins, model {models.pop()}, tifs dated {RENDER_DATE}"
    )
    return run0


def grid_and_polys(basins: list[str]):
    bounds = []
    for b in basins:
        with rasterio.open(HUC8_ROOT / b / "gnn" / ARM / "gnn_dtw_100m.tif") as ds:
            if ds.transform.a != CELL or ds.transform.e != -CELL:
                raise ValueError(f"{b}: not a 100 m north-up grid")
            if ds.crs.to_epsg() != 5070:
                raise ValueError(f"{b}: CRS is {ds.crs}, expected EPSG:5070")
            if ds.nodata != NODATA:
                raise ValueError(f"{b}: nodata {ds.nodata}")
            bounds.append(ds.bounds)
    left = min(b.left for b in bounds)
    top = max(b.top for b in bounds)
    right = max(b.right for b in bounds)
    bottom = min(b.bottom for b in bounds)
    if (left % CELL) or (top % CELL):
        raise ValueError("mosaic origin is not on the 100 m grid")
    nrow = int(round((top - bottom) / CELL))
    ncol = int(round((right - left) / CELL))
    hucs = gpd.read_parquet(HUC8_POLYS, columns=["huc8", "name", "geometry"])
    hucs["huc8"] = hucs["huc8"].astype(str).str.zfill(8)
    hucs = hucs[hucs["huc8"].isin(basins)].set_index("huc8")
    if len(hucs) != len(basins):
        raise ValueError(f"{len(hucs)} HUC8 polygons for {len(basins)} basins")
    if hucs.crs.to_epsg() != 5070:
        raise ValueError("HUC8 polygons are not EPSG:5070")
    log(f"grid {nrow} x {ncol} cells, origin ({left:.0f}, {top:.0f})")
    return from_origin(left, top, CELL, CELL), nrow, ncol, hucs


def basin_window(ds, transform) -> tuple[slice, slice]:
    r0 = int(round((transform.f - ds.bounds.top) / CELL))
    c0 = int(round((ds.bounds.left - transform.c) / CELL))
    return slice(r0, r0 + ds.height), slice(c0, c0 + ds.width)


def inside_mask(geom, ds) -> np.ndarray:
    return (
        rasterize(
            [(geom, 1)],
            out_shape=(ds.height, ds.width),
            transform=ds.transform,
            fill=0,
            dtype="uint8",
        )
        == 1
    )


def mosaic_layer(
    name: str, basins, transform, nrow, ncol, hucs
) -> tuple[np.ndarray, dict]:
    src, bands, *_ = LAYERS[name]
    nb = len(bands)
    out = np.full((nb, nrow, ncol), NODATA, np.float32)
    owner = np.full((nrow, ncol), -1, np.int16)
    n_pre = n_drop = n_overlap = n_band_mismatch = 0
    for k, b in enumerate(basins):
        with rasterio.open(HUC8_ROOT / b / "gnn" / ARM / src) as ds:
            if ds.count != nb:
                raise ValueError(f"{b}/{src}: {ds.count} bands, expected {nb}")
            if nb > 1 and tuple(ds.descriptions) != bands:
                raise ValueError(f"{b}/{src}: band names {ds.descriptions}")
            arr = ds.read().astype(np.float32)
            sl = basin_window(ds, transform)
            inside = inside_mask(hucs.loc[b, "geometry"], ds)
        if np.isnan(arr).any():
            raise ValueError(f"{b}/{src}: NaN cells (nodata is -9999)")
        valid_b = arr != NODATA
        valid = valid_b[0]
        n_band_mismatch += int((valid_b != valid[None]).sum())
        n_pre += int(valid.sum())
        n_drop += int((valid & ~inside).sum())
        keep = valid & inside
        n_overlap += int((keep & (owner[sl] >= 0)).sum())
        owner[sl] = np.where(keep, k, owner[sl])
        for i in range(nb):
            out[i][sl] = np.where(keep, arr[i], out[i][sl])
    if n_overlap:
        raise ValueError(f"{name}: {n_overlap} valid cells claimed by two basins")
    if n_band_mismatch:
        raise ValueError(
            f"{name}: band validity masks disagree on {n_band_mismatch} cells"
        )
    meta = {
        "n_preclip_valid_cells": n_pre,
        "n_cells_dropped_by_clip": n_drop,
        "n_valid_cells": int((owner >= 0).sum()),
        "overlap_after_clip": n_overlap,
    }
    log(f"{name}: {meta}")
    return out, meta


def write_cog(name: str, arr: np.ndarray, transform, run0: dict) -> Path:
    src, bands, unit, desc = LAYERS[name]
    tmp = STAGE_DIR / f"{name}.tmp.tif"
    dst = STAGE_DIR / f"{name}.tif"
    prof = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": NODATA,
        "crs": "EPSG:5070",
        "transform": transform,
        "width": arr.shape[2],
        "height": arr.shape[1],
        "count": arr.shape[0],
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(tmp, "w", **prof) as ds:
        ds.write(arr)
        ds.update_tags(
            layer=name,
            description=desc,
            units=unit,
            model=run0["model"],
            surface=run0["surface"],
            source_arm=ARM,
            render_date=RENDER_DATE,
            resolution_m="100",
            nodata=str(NODATA),
        )
        ds.units = tuple(unit for _ in bands)
        for i, bn in enumerate(bands, start=1):
            ds.set_band_description(i, bn)
            ds.update_tags(i, units=unit, description=bn if len(bands) > 1 else desc)
    if dst.exists():
        dst.unlink()
    rasterio.shutil.copy(
        tmp,
        dst,
        driver="COG",
        COMPRESS="DEFLATE",
        PREDICTOR="2",
        BLOCKSIZE="512",
        OVERVIEWS="AUTO",
        OVERVIEW_RESAMPLING="AVERAGE",
        BIGTIFF="IF_SAFER",
        NUM_THREADS="ALL_CPUS",
    )
    tmp.unlink()
    with rasterio.open(dst) as ds:
        if ds.descriptions != bands or ds.nodata != NODATA or ds.crs.to_epsg() != 5070:
            raise ValueError(f"{dst}: metadata did not survive the COG copy")
        log(
            f"wrote {dst.name}: overviews {ds.overviews(1)}, block {ds.block_shapes[0]}"
        )
    return dst


def verify_cog(name: str, path: Path, basins, hucs) -> dict:
    """Mosaic inside each basin polygon == basin raster exactly; no double claims."""
    src, bands, *_ = LAYERS[name]
    with rasterio.open(path) as ds:
        mos = ds.read()
        transform = ds.transform
    claims = np.zeros(mos.shape[1:], np.uint8)
    max_diff = 0.0
    n_cmp = 0
    for b in basins:
        with rasterio.open(HUC8_ROOT / b / "gnn" / ARM / src) as ds:
            arr = ds.read()
            sl = basin_window(ds, transform)
            inside = inside_mask(hucs.loc[b, "geometry"], ds)
        m = mos[(slice(None),) + sl]
        if not np.array_equal(arr[:, inside], m[:, inside]):
            d = np.abs(arr[:, inside].astype(np.float64) - m[:, inside])
            raise AssertionError(
                f"{name}/{b}: mosaic differs inside polygon, max {d.max()}"
            )
        max_diff = max(
            max_diff, float(np.abs(arr[:, inside] - m[:, inside]).max(initial=0))
        )
        n_cmp += int(inside.sum())
        claims[sl] += (inside & (arr[0] != NODATA)).astype(np.uint8)
    valid = mos[0] != NODATA
    n_multi = int((claims > 1).sum())
    if n_multi:
        raise AssertionError(f"{name}: {n_multi} valid cells claimed by two basins")
    if int((valid & (claims == 0)).sum()):
        raise AssertionError(f"{name}: mosaic valid cells outside every basin polygon")
    res = {
        "max_abs_diff_inside_polygons": max_diff,
        "n_cells_compared": n_cmp,
        "n_cells_multi_claimed": n_multi,
        "n_valid_cells": int(valid.sum()),
        "n_total_cells": int(valid.size),
    }
    log(f"verify {name}: {res}")
    return res


def check_sites_against_dtw(dtw_path: Path, prim: pd.DataFrame) -> dict:
    """The distributed DTW mosaic reproduces the scored site values."""
    with rasterio.open(dtw_path) as ds:
        arr = ds.read(1)
        r, c = rowcol(ds.transform, prim["x5070"].to_numpy(), prim["y5070"].to_numpy())
    v = arr[np.asarray(r), np.asarray(c)].astype(np.float64)
    if (v == NODATA).any():
        raise AssertionError(
            f"{int((v == NODATA).sum())} primary sites on mosaic nodata"
        )
    diff = np.abs(v - prim["handily_dtw_m"].to_numpy())
    mad = float(np.median(np.abs(v - prim["obs_dtw_m"].to_numpy())))
    log(
        f"site check: max |mosaic - scored| {diff.max():.3g} m, MAD from mosaic {mad:.4f} m"
    )
    if diff.max() > 0:
        raise AssertionError("mosaic does not reproduce the scored site values")
    return {"max_abs_diff_m": float(diff.max()), "mad_from_mosaic_m": mad}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    df = pd.concat(
        [pd.read_csv(EVAL_DIR / f"panel_{k}.csv") for k in ("core", "depth", "dist")],
        ignore_index=True,
    )
    df = df[df["set"].isin(SETS) & df["predictor"].isin(PREDICTORS)].copy()
    df["set"] = pd.Categorical(
        df["set"].map(SETS), categories=list(SETS.values()), ordered=True
    )
    df["predictor"] = df["predictor"].map(PREDICTORS)
    kind_order = {"overall": 0, "depth": 1, "dist_src": 2}
    df["_k"] = df["stratum_kind"].map(kind_order)
    if df["_k"].isna().any():
        raise ValueError(f"unexpected stratum kinds {set(df['stratum_kind'])}")
    df = df.sort_values(["set", "_k"], kind="stable").drop(columns="_k")
    return df[PANEL_COLS].reset_index(drop=True)


def load_shallow() -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "panel_shallow.csv")
    df = df[df["set"].isin(SETS) & df["predictor"].isin(PREDICTORS)].copy()
    df["set"] = pd.Categorical(
        df["set"].map(SETS), categories=list(SETS.values()), ordered=True
    )
    df["predictor"] = df["predictor"].map(PREDICTORS)
    return df.sort_values(["set", "threshold_m"], kind="stable").reset_index(drop=True)


def load_delta() -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "delta_bootstrap.csv")
    df = df[df["set"].isin(SETS) & (df["comparison"] == "Ma - primary")].copy()
    df["set"] = pd.Categorical(
        df["set"].map(SETS), categories=list(SETS.values()), ordered=True
    )
    df = df.rename(
        columns={
            "mad_delta_m": "MAD_Ma_minus_handily_m",
            "rmse_delta_m": "RMSE_Ma_minus_handily_m",
            "mad_lo_w25r": "MAD_delta_ci95_lo_m",
            "mad_hi_w25r": "MAD_delta_ci95_hi_m",
            "rmse_lo_w25r": "RMSE_delta_ci95_lo_m",
            "rmse_hi_w25r": "RMSE_delta_ci95_hi_m",
            "n_blocks_w25r": "n_basin_blocks",
        }
    )
    cols = [
        "set",
        "stratum_kind",
        "stratum",
        "n",
        "MAD_Ma_minus_handily_m",
        "MAD_delta_ci95_lo_m",
        "MAD_delta_ci95_hi_m",
        "RMSE_Ma_minus_handily_m",
        "RMSE_delta_ci95_lo_m",
        "RMSE_delta_ci95_hi_m",
        "n_basin_blocks",
    ]
    return df.sort_values("set", kind="stable")[cols].reset_index(drop=True)


def check_report_numbers(
    panel: pd.DataFrame, shallow: pd.DataFrame, delta: pd.DataFrame
) -> None:
    """Assert the numbers quoted in the tech report against the shipped tables."""

    def get(s, kind, stratum, pred, col):
        r = panel[
            (panel.set == s)
            & (panel.stratum_kind == kind)
            & (panel.stratum == stratum)
            & (panel.predictor == pred)
        ]
        if len(r) != 1:
            raise ValueError(f"{s}/{kind}/{stratum}/{pred}: {len(r)} rows")
        return float(r[col].iloc[0])

    A, B, P = "A_heldout_water_table", "B_all_ndwr_sites", "B_pumping_wells_only"
    expect = [
        (A, "overall", "all", "handily", "MAD_m", 1.92),
        (A, "overall", "all", "Ma", "MAD_m", 4.41),
        (A, "overall", "all", "handily", "RMSE_m", 31.7),
        (A, "overall", "all", "Ma", "RMSE_m", 51.9),
        (A, "overall", "all", "handily", "bias_m", -4.45),
        (A, "overall", "all", "handily", "median_resid_m", -0.40),
        (A, "depth", "0-2 m", "handily", "MAD_m", 1.11),
        (A, "depth", "30+ m", "Ma", "RMSE_m", 117.16),
        (A, "dist_src", "0-0.3 km", "handily", "MAD_m", 1.29),
        (A, "dist_src", "20+ km", "Ma", "MAD_m", 18.90),
        (B, "overall", "all", "handily", "MAD_m", 4.93),
        (B, "overall", "all", "Ma", "MAD_m", 6.46),
        (P, "overall", "all", "handily", "MAD_m", 6.05),
        (P, "overall", "all", "Ma", "MAD_m", 7.11),
    ]
    for s, k, st, p, col, want in expect:
        got = get(s, k, st, p, col)
        nd = len(str(want).split(".")[1])
        if round(got, nd) != want:
            raise AssertionError(
                f"report says {want}, table has {got} ({s}/{k}/{st}/{p}/{col})"
            )
    n = get(A, "overall", "all", "handily", "n")
    if n != N_PRIMARY:
        raise AssertionError(f"primary n {n}")
    s2 = shallow[(shallow.set == A) & (shallow.threshold_m == 2.0)].set_index(
        "predictor"
    )
    for p, rec, prec in (("handily", 0.28, 0.49), ("Ma", 0.15, 0.14)):
        if (round(s2.loc[p, "recall"], 2), round(s2.loc[p, "precision"], 2)) != (
            rec,
            prec,
        ):
            raise AssertionError(f"shallow <2 m {p}: {s2.loc[p].to_dict()}")
    d = delta[(delta.set == A) & (delta.stratum_kind == "overall")].iloc[0]
    got = tuple(
        round(float(d[c]), 2)
        for c in (
            "MAD_Ma_minus_handily_m",
            "MAD_delta_ci95_lo_m",
            "MAD_delta_ci95_hi_m",
        )
    )
    if got != (2.49, 1.87, 4.78):
        raise AssertionError(f"MAD delta CI {got}")
    log(f"report numbers: {len(expect) + 3} quoted values reproduced")


def build_residuals() -> gpd.GeoDataFrame:
    d = pd.read_parquet(EVAL_DIR / "sites_sampled.parquet")
    prim = d[d["is_admission_eligible"] & ~d["is_pumping"]].copy()
    if len(prim) != N_PRIMARY:
        raise AssertionError(f"primary set has {len(prim)} sites, expected {N_PRIMARY}")
    if (prim["n_basin_hits"] != 1).any():
        raise AssertionError("a primary site is not in exactly one basin")
    out = pd.DataFrame(
        {
            "obs_dtw_m": prim["obs_dtw_m"],
            "handily_dtw_m": prim["ndwrlbl_ufold__dtw"],
            "handily_sigma_m": prim["ndwrlbl_ufold__sigma"],
            "handily_p_dtw_lt_2m": prim["ndwrlbl_ufold__p_dtw_lt_2m"],
            "handily_p_dtw_lt_5m": prim["ndwrlbl_ufold__p_dtw_lt_5m"],
            "handily_p_dtw_lt_10m": prim["ndwrlbl_ufold__p_dtw_lt_10m"],
            "ma_dtw_m": prim["ma_resampled_m"],
            "dist_src_km": prim["dist_src_km"],
            "dist_nwis_km": prim["dist_nwis_km"],
            "huc8": prim["basin_hit"].astype(str),
            "por_end_yr": prim["por_end_yr"].astype(int),
            "x5070": prim["x5070"],
            "y5070": prim["y5070"],
        }
    )
    if out.isna().any().any():
        raise ValueError(
            f"missing values in primary residual table:\n{out.isna().sum()}"
        )
    out["handily_resid_m"] = out["handily_dtw_m"] - out["obs_dtw_m"]
    out["ma_resid_m"] = out["ma_dtw_m"] - out["obs_dtw_m"]
    lon, lat = Transformer.from_crs(5070, 4326, always_xy=True).transform(
        out["x5070"].to_numpy(), out["y5070"].to_numpy()
    )
    out["lon"], out["lat"] = lon, lat
    gdf = gpd.GeoDataFrame(
        out.reset_index(drop=True),
        geometry=gpd.points_from_xy(out["x5070"], out["y5070"]),
        crs=5070,
    )
    mad_h = float(np.median(np.abs(gdf["handily_resid_m"])))
    mad_m = float(np.median(np.abs(gdf["ma_resid_m"])))
    log(f"residuals: {len(gdf)} sites, MAD handily {mad_h:.3f} m, Ma {mad_m:.3f} m")
    return gdf


GUIDE = [
    ("Nevada depth-to-water maps: how to read this workbook", None),
    (
        "",
        "Two maps of depth to water, each checked against Nevada Division of Water "
        "Resources well-log records. Every number is in meters unless marked. Depth "
        "is measured downward from the ground surface, so a bigger number means "
        "deeper water.",
    ),
    ("The two maps", None),
    (
        "handily",
        "Our machine-learning water-table map, 100 m pixels. Sampled at each test "
        "site from the distributed statewide GeoTIFF (handily_dtw_nv_100m.tif).",
    ),
    (
        "Ma",
        "The published national benchmark (Ma and others, 2026, Communications Earth "
        "& Environment 7, 45), the HydroFrame release on its native 24 m grid, "
        "sampled at each test site.",
    ),
    ("The panels", None),
    (
        "A_heldout_water_table",
        "Panel A. 8,877 NDWR sites that are not pumping wells and whose record ends "
        "in 1980 or later, drawn at random by site and never shown to handily in any "
        "role. This is the water-table accuracy statement.",
    ),
    (
        "B_all_ndwr_sites",
        "Panel B. All 39,581 NDWR sites with a usable static level, including 30,076 "
        "pumping wells and 628 sites with pre-1980 or undated records. Context only: "
        "a pumping well's static level is not the quantity either map predicts.",
    ),
    (
        "B_pumping_wells_only",
        "The 30,076 pumping wells inside Panel B, reported on their own.",
    ),
    ("What each column means", None),
    (
        "residual",
        "predicted minus observed depth, m. Positive = the map places water too deep.",
    ),
    ("MAD_m", "Median absolute residual, m. The typical miss. Lower is better."),
    (
        "median_resid_m",
        "Middle signed residual, m. Which way the map leans at a typical site.",
    ),
    (
        "bias_m",
        "Mean signed residual, m. When it is much larger than the median residual the "
        "map has a one-sided tail of large misses, not a uniform offset.",
    ),
    (
        "RMSE_m",
        "Root-mean-square residual, m. Weights the worst misses heavily; read it next to MAD.",
    ),
    (
        "p95_abs_resid_m",
        "95th-percentile absolute residual, m. The worst 1-in-20 miss.",
    ),
    ("n", "Number of sites in the row."),
    (
        "stratum",
        "depth = observed depth band at the site (not the prediction); dist_src = "
        "distance from the site to the nearest NDWR well handily read, km.",
    ),
    (
        "Ma minus handily",
        "On the difference tab, MAD (or RMSE) of Ma minus that of handily, m; "
        "positive = handily better. The 95% interval bounds that difference, from a "
        "paired bootstrap of 2,000 resamples that draws whole basin blocks.",
    ),
    ("Shallow detection", None),
    (
        "recall",
        "Of the sites truly shallower than the cutoff, the share the map called "
        "shallower than the cutoff (dimensionless, 0-1).",
    ),
    (
        "precision",
        "Of the sites the map called shallower than the cutoff, the share that truly "
        "were (dimensionless, 0-1).",
    ),
    ("f1", "Harmonic mean of precision and recall (dimensionless, 0-1)."),
]


def _metric_block(ws, df: pd.DataFrame, start: int, label: str) -> int:
    cols = [
        (label, "stratum", None),
        ("Sites (n)", "n", "#,##0"),
        ("Map", "predictor", None),
        ("MAD", "MAD_m", "0.00"),
        ("Median resid", "median_resid_m", "+0.00;-0.00"),
        ("Mean bias", "bias_m", "+0.00;-0.00"),
        ("RMSE", "RMSE_m", "0.00"),
        ("p95 abs resid", "p95_abs_resid_m", "0.00"),
    ]
    _header(ws, start, [c[0] for c in cols], widths=[16, 11, 11, 9, 13, 11, 9, 13])
    row, shade = start + 1, False
    for g in dict.fromkeys(df["stratum"]):
        sub = df[df["stratum"] == g].set_index("predictor").reindex(["handily", "Ma"])
        best = {m: sub[m].min() for m in ("MAD_m", "RMSE_m", "p95_abs_resid_m")}
        first = row
        for prod, rec in sub.iterrows():
            for i, (_, key, fmt) in enumerate(cols, start=1):
                if key == "stratum":
                    val = g if row == first else None
                elif key == "predictor":
                    val = prod
                elif key == "n":
                    val = int(rec["n"])
                else:
                    val = float(rec[key])
                c = ws.cell(row=row, column=i, value=val)
                if fmt and val is not None:
                    c.number_format = fmt
                c.alignment = Alignment(horizontal="center" if i > 1 else "left")
                if shade:
                    c.fill = PatternFill("solid", fgColor=BAND_TINT)
                if key in best and rec[key] == best[key]:
                    c.font = Font(bold=True, color="1E5C2E")
                    c.fill = PatternFill("solid", fgColor=WIN)
                if key == "predictor":
                    c.font = Font(bold=True, size=10)
                c.border = Border(bottom=RULE)
            row += 1
        shade = not shade
    return row


def _plain_sheet(wb, title: str, df: pd.DataFrame, widths: int = 16) -> None:
    ws = wb.create_sheet(title)
    heads = list(df.columns)
    _header(ws, 1, heads, widths=[widths] * len(heads))
    for j, rec in enumerate(df.itertuples(index=False), start=2):
        for i, v in enumerate(rec, start=1):
            v = v.item() if hasattr(v, "item") else v
            c = ws.cell(row=j, column=i, value=v)
            col = heads[i - 1]
            if col in ("n", "n_obs_shallow", "n_pred_shallow", "n_basin_blocks"):
                c.number_format = "#,##0"
            elif col.endswith("_m") or col in ("precision", "recall", "f1"):
                c.number_format = "0.00"
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(heads))}{len(df) + 1}"


def build_workbook(panel, shallow, delta, path: Path) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    ws = wb.create_sheet("How to read this")
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 108
    r = 1
    for k, v in GUIDE:
        if v is None:
            r += 1 if r > 1 else 0
            _title(ws, r, k, size=15 if r == 1 else 13)
        else:
            a = ws.cell(row=r, column=1, value=k or None)
            a.font = Font(bold=True, size=10)
            a.alignment = Alignment(vertical="top")
            b = ws.cell(row=r, column=2, value=v)
            b.alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[r].height = max(15, 13 * (1 + len(v) // 95))
        r += 1

    for kind, title, label in (
        ("depth", "Depth bands", "Observed depth"),
        ("dist_src", "Distance to wells read", "To nearest well read"),
    ):
        ws = wb.create_sheet(title)
        ws.sheet_view.showGridLines = False
        r = 1
        for s in SETS.values():
            sub = panel[(panel.set == s) & panel.stratum_kind.isin(["overall", kind])]
            n = int(sub[sub.stratum_kind == "overall"]["n"].iloc[0])
            _title(ws, r, f"{s}  (n = {n:,} sites)", size=11)
            r = _metric_block(ws, sub, r + 1, label) + 2
        ws.freeze_panes = "A2"

    _plain_sheet(wb, "Shallow detection", shallow.astype({"set": str}))
    _plain_sheet(wb, "Ma minus handily", delta.astype({"set": str}), widths=18)
    _plain_sheet(wb, "Full panel", panel.astype({"set": str}))
    wb.save(path)


# ---------------------------------------------------------------------------
def zip_one(zpath: Path, members: list[Path]) -> None:
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for m in members:
            z.write(m, arcname=m.name)
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad:
            raise IOError(f"{zpath}: corrupt member {bad}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-rasters", action="store_true", help="reuse staged COGs")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    basins = rendered_basins()
    if len(basins) != 72:
        raise ValueError(f"{len(basins)} rendered basins, expected 72")
    missing = [b for b in basins if not (HUC8_ROOT / b / "gnn" / ARM).is_dir()]
    if missing:
        raise FileNotFoundError(f"arm missing for {missing}")
    run0 = check_render(basins)
    transform, nrow, ncol, hucs = grid_and_polys(basins)

    # Tables first: cheap, and they carry the count assertions.
    panel, shallow, delta = load_panel(), load_shallow(), load_delta()
    check_report_numbers(panel, shallow, delta)
    resid = build_residuals()

    verify = {}
    for name in LAYERS:
        path = STAGE_DIR / f"{name}.tif"
        if not args.skip_rasters:
            arr, meta = mosaic_layer(name, basins, transform, nrow, ncol, hucs)
            path = write_cog(name, arr, transform, run0)
            del arr
        else:
            meta = {}
        verify[name] = {**meta, **verify_cog(name, path, basins, hucs)}
    valid_sets = {v["n_valid_cells"] for v in verify.values()}
    if len(valid_sets) != 1:
        raise AssertionError(f"layers disagree on valid cell count: {valid_sets}")
    site_check = check_sites_against_dtw(STAGE_DIR / "handily_dtw_nv_100m.tif", resid)

    tables = {
        "csv": STAGE_DIR / "nv_accuracy_panel.csv",
        "xlsx": STAGE_DIR / "nv_accuracy_panel.xlsx",
        "shallow": STAGE_DIR / "nv_shallow_detection.csv",
        "gpkg": STAGE_DIR / "nv_heldout_residuals.gpkg",
    }
    panel.to_csv(tables["csv"], index=False)
    shallow.to_csv(tables["shallow"], index=False)
    build_workbook(panel, shallow, delta, tables["xlsx"])
    if tables["gpkg"].exists():
        tables["gpkg"].unlink()
    resid.to_file(tables["gpkg"], layer="nv_heldout_residuals", driver="GPKG")

    entries = []
    common = {
        "crs": "EPSG:5070",
        "nodata": NODATA,
        "source_arm": ARM,
        "render_date": RENDER_DATE,
        "model": run0["model"],
    }
    for name, (_, bands, unit, desc) in LAYERS.items():
        tif = STAGE_DIR / f"{name}.tif"
        z = OUT_DIR / f"{name}.zip"
        log(f"zipping {z.name}")
        zip_one(z, [tif])
        entries.append(
            {
                "file": z.name,
                "contents": [
                    {
                        "file": tif.name,
                        "bytes": tif.stat().st_size,
                        "sha256": sha256(tif),
                    }
                ],
                "layer": name,
                "bands": list(bands),
                "units": unit,
                "description": desc,
                "resolution_m": CELL,
                **common,
            }
        )
    z = OUT_DIR / "nv_accuracy_tables.zip"
    members = [tables["xlsx"], tables["csv"], tables["shallow"], tables["gpkg"]]
    zip_one(z, members)
    entries.append(
        {
            "file": z.name,
            "contents": [
                {"file": m.name, "bytes": m.stat().st_size, "sha256": sha256(m)}
                for m in members
            ],
            "layer": "accuracy_tables",
            "units": "m (residual metrics); dimensionless 0-1 "
            "(precision, recall, f1, probabilities); km (distances)",
            "description": "Panel A/B accuracy panel, shallow detection, held-out residuals "
            "(EPSG:5070 points with lon/lat)",
            **common,
        }
    )
    for e in entries:
        p = OUT_DIR / e["file"]
        e["bytes"] = p.stat().st_size
        e["sha256"] = sha256(p)
    entries = [
        {
            k: e[k]
            for k in (
                "file",
                "bytes",
                "sha256",
                *[k for k in e if k not in ("file", "bytes", "sha256")],
            )
        }
        for e in entries
    ]

    km2 = verify["handily_dtw_nv_100m"]["n_valid_cells"] * (CELL / 1000.0) ** 2
    manifest = {
        "packet": "handily Nevada NWI data packet",
        "built": dt.datetime.now().isoformat(timespec="seconds"),
        "model": run0["model"],
        "surface": run0["surface"],
        "source_arm": ARM,
        "render_date": RENDER_DATE,
        "basins": basins,
        "grid": {
            "crs": "EPSG:5070",
            "cell_m": CELL,
            "nrow": nrow,
            "ncol": ncol,
            "transform": list(transform)[:6],
            "nodata": NODATA,
        },
        "valid_cells": verify["handily_dtw_nv_100m"]["n_valid_cells"],
        "valid_area_km2": km2,
        "verification": verify,
        "site_check": site_check,
        "residual_sign": "predicted - observed (m), positive = predicted too deep",
        "files": entries,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_DIR / "SHA256SUMS").write_text(
        "".join(f"{e['sha256']}  {e['file']}\n" for e in entries)
    )
    shutil.copy(OUT_DIR / "manifest.json", STAGE_DIR / "manifest.json")

    print("\narchive                              size")
    for e in entries:
        print(f"{e['file']:<36} {human(e['bytes'])}")
    v = verify["handily_dtw_nv_100m"]
    print(
        f"\ntotal cells {v['n_total_cells']:,}; valid cells {v['n_valid_cells']:,}; "
        f"valid area {km2:,.0f} km2"
    )


if __name__ == "__main__":
    main()
