"""Surface-water mask V4 = V3 ∪ NHD-HR lake / reservoir / marsh polygons, 100 m lattice.

V3 (``build_water_mask_v3.py``) is JRC occurrence >= 50 % AND a 1 km density
screen. Intermittent lakes, terminal sinks and marshes never reach 50 %
occurrence (JRC p50 Humboldt Lake 33 %, Humboldt Salt Marsh 23 %, Big Water 8 %,
Dixie Meadows 0 %; notes/NV_RASTER_DEBUG_2026-10-07.md section 4), so the
renderer's inference-time water flatten never touches them. V4 adds the mapped
NHD waterbody polygons:

    water_v4 = water_v3  OR  rasterize(NHDWaterbody where FType in {390, 436, 466})

* FType 390 LakePond, 436 Reservoir, 466 SwampMarsh -- every FCode under those
  FTypes (perennial AND intermittent; no hydrographic-category filter).
* FType 361 Playa (FCode 36100) is EXCLUDED: a dry playa is not surface water,
  and the Carson Sink playa alone is ~1,100 km^2. FType 378 Ice Mass excluded.
* Any-overlap semantics (``all_touched=True``): a 100 m cell is masked if the
  polygon touches it -- same convention as V3's any-overlap of 30 m JRC pixels.
* No density screen on the NHD part (the polygons are mapped features, not a
  noisy per-pixel classification).

Sources: ``/nas/hydrography/nhd/hr_state/NHD_H_<State>_State_Shape/Shape/
NHDWaterbody.shp`` (EPSG:4269), reprojected to EPSG:5070. ``--states`` picks the
state packages (default: Nevada + its neighbours, so NV basins that straddle a
border are covered; add New_Mexico etc. later and rebuild).

Outputs (``--out-dir``, default /nas/handily/covariates/water_mask_v4):
  water_mask_v4_100m_5070.tif   uint8 0/1, identical grid / dtype to V3, so
                                ``sample_water_mask`` / ``water_flatten_factor``
                                read it unchanged (``--water-flatten-mask``)
  build_manifest.json           recipe, FTypes, states, cell counts by source

No shoreline-cells parquet is written: the renderer consumes only the mask
raster (``--water-flatten-mask``); the V3 cells parquet feeds the bundle
builder's ``water_v3`` block, which the production bundles do not carry.

Usage:
    uv run python utils/build_water_mask_v4.py
    uv run python utils/build_water_mask_v4.py --states Nevada New_Mexico
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely import force_2d

from build_water_mask_v3 import (
    DEM_100M,
    MASK_TIF as V3_MASK_NAME,
    OUT_DIR_DEFAULT as V3_DIR,
    RES,
    canonical_grid,
    write_uint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_water_mask_v4")

NHD_STATE_DIR = Path("/nas/hydrography/nhd/hr_state")
OUT_DIR_DEFAULT = Path("/nas/handily/covariates/water_mask_v4")
MASK_TIF = "water_mask_v4_100m_5070.tif"
V3_MASK = V3_DIR / V3_MASK_NAME
FTYPES = {390: "LakePond", 436: "Reservoir", 466: "SwampMarsh"}
EXCLUDED_FTYPES = {361: "Playa", 378: "Ice Mass"}
DEFAULT_STATES = ("Nevada", "California", "Oregon", "Idaho", "Utah", "Arizona")
CHECK_NAMES = ("Dixie Meadows", "Humboldt Salt Marsh", "Big Water", "Humboldt Lake")


def waterbody_path(state: str) -> Path:
    return NHD_STATE_DIR / f"NHD_H_{state}_State_Shape" / "Shape" / "NHDWaterbody.shp"


def load_waterbodies(state: str) -> gpd.GeoDataFrame:
    """Selected-FType NHDWaterbody polygons of one state package, EPSG:5070, 2-D."""
    p = waterbody_path(state)
    if not p.exists():
        raise SystemExit(f"no NHDWaterbody for {state}: {p}")
    g = gpd.read_file(p, columns=["gnis_name", "ftype", "fcode", "areasqkm"])
    g = g[g["ftype"].isin(list(FTYPES))].to_crs(5070)
    g["geometry"] = force_2d(g.geometry.values)
    g = g[~g.geometry.is_empty & g.geometry.notna()]
    g["state"] = state
    return g


def lattice_window(bounds: tuple, transform, width: int, height: int) -> Window:
    """Lattice window (col_off, row_off, w, h) covering ``bounds``, clipped, +1 cell."""
    xmin, ymin, xmax, ymax = bounds
    c0 = max(0, int(np.floor((xmin - transform.c) / RES)) - 1)
    c1 = min(width, int(np.ceil((xmax - transform.c) / RES)) + 1)
    r0 = max(0, int(np.floor((transform.f - ymax) / RES)) - 1)
    r1 = min(height, int(np.ceil((transform.f - ymin) / RES)) + 1)
    return Window(c0, r0, c1 - c0, r1 - r0)


def rasterize_any_touch(
    geoms, transform, width: int, height: int, win: Window
) -> np.ndarray:
    """uint8 0/1 of ``geoms`` on the lattice sub-window, any-overlap (all_touched)."""
    sub_t = rasterio.windows.transform(win, transform)
    return rasterize(
        ((g, 1) for g in geoms),
        out_shape=(int(win.height), int(win.width)),
        transform=sub_t,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )


def check_named(
    wb: gpd.GeoDataFrame, mask_path: Path, v3_path: Path, names=CHECK_NAMES
) -> list[dict]:
    """Share of each named waterbody's cells (centre-in-polygon) masked in V3 / V4."""
    out = []
    with rasterio.open(mask_path) as d4, rasterio.open(v3_path) as d3:
        t, w, h = d4.transform, d4.width, d4.height
        for name in names:
            sel = wb[wb["gnis_name"] == name]
            if sel.empty:
                raise SystemExit(f"named check: {name!r} not in the selected NHD set")
            for _, r in sel.iterrows():
                win = lattice_window(r.geometry.bounds, t, w, h)
                inside = rasterize(
                    [(r.geometry, 1)],
                    out_shape=(int(win.height), int(win.width)),
                    transform=rasterio.windows.transform(win, t),
                    fill=0,
                    all_touched=False,
                    dtype="uint8",
                ).astype(bool)
                m4 = d4.read(1, window=win).astype(bool)
                m3 = d3.read(1, window=win).astype(bool)
                n = int(inside.sum())
                rec = {
                    "gnis_name": name,
                    "state": r["state"],
                    "ftype": int(r["ftype"]),
                    "fcode": int(r["fcode"]),
                    "areasqkm": float(r["areasqkm"]),
                    "n_cells_centre_in": n,
                    "frac_masked_v3": float(m3[inside].mean()) if n else None,
                    "frac_masked_v4": float(m4[inside].mean()) if n else None,
                }
                log.info("named check: %s", rec)
                out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--v3-mask", default=str(V3_MASK))
    ap.add_argument("--dem", default=DEM_100M)
    ap.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    args = ap.parse_args()
    t_start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / MASK_TIF

    transform, width, height = canonical_grid(args.dem)
    with rasterio.open(args.v3_mask) as ds:
        if (ds.transform, ds.width, ds.height) != (transform, width, height):
            raise SystemExit(f"V3 mask is not on the canonical lattice: {args.v3_mask}")
        v3 = ds.read(1).astype(bool)
    log.info("V3 mask: %d cells", int(v3.sum()))

    nhd = np.zeros((height, width), dtype=bool)
    per_state, per_ftype = {}, {f: 0 for f in FTYPES}
    wbs = []
    for state in args.states:
        wb = load_waterbodies(state)
        wbs.append(wb)
        win = lattice_window(tuple(wb.total_bounds), transform, width, height)
        rs = slice(int(win.row_off), int(win.row_off + win.height))
        cs = slice(int(win.col_off), int(win.col_off + win.width))
        st = np.zeros((int(win.height), int(win.width)), dtype=bool)
        for ft in FTYPES:
            geoms = wb.geometry[wb["ftype"] == ft].values
            if not len(geoms):
                continue
            a = rasterize_any_touch(geoms, transform, width, height, win).astype(bool)
            per_ftype[ft] += int(a.sum())
            st |= a
        nhd[rs, cs] |= st
        per_state[state] = {
            "n_polygons": int(len(wb)),
            "n_polygons_by_ftype": {
                FTYPES[k]: int(v) for k, v in wb["ftype"].value_counts().items()
            },
            "n_cells": int(st.sum()),
        }
        log.info("%s: %d polygons -> %d cells", state, len(wb), int(st.sum()))

    v4 = v3 | nhd
    n_both = int((v3 & nhd).sum())
    n_v3_only = int((v3 & ~nhd).sum())
    n_nhd_only = int((nhd & ~v3).sum())
    km2 = RES * RES / 1e6
    write_uint(mask_path, v4, transform, "uint8")
    log.info(
        "V4: %d cells (V3-only %d, NHD-only %d, both %d) -> %s",
        int(v4.sum()),
        n_v3_only,
        n_nhd_only,
        n_both,
        mask_path,
    )
    del v3, nhd, v4

    wb_all = gpd.GeoDataFrame(pd.concat(wbs, ignore_index=True), crs=5070)
    named = check_named(wb_all, mask_path, Path(args.v3_mask))

    (out_dir / "build_manifest.json").write_text(
        json.dumps(
            {
                "recipe": "V4 = V3 mask OR rasterized NHD-HR NHDWaterbody polygons "
                "with FType in {390 LakePond, 436 Reservoir, 466 SwampMarsh} (all "
                "FCodes, perennial and intermittent), any-overlap (all_touched); "
                "no density screen on the NHD part",
                "v3_mask": str(args.v3_mask),
                "v3_manifest": str(Path(args.v3_mask).with_name("build_manifest.json")),
                "nhd_source": str(NHD_STATE_DIR)
                + "/NHD_H_<State>_State_Shape/Shape/NHDWaterbody.shp",
                "nhd_crs_in": "EPSG:4269 (reprojected to EPSG:5070)",
                "ftypes_included": {str(k): v for k, v in FTYPES.items()},
                "ftypes_excluded": {str(k): v for k, v in EXCLUDED_FTYPES.items()},
                "states": list(args.states),
                "per_state": per_state,
                "nhd_cells_by_ftype": {FTYPES[k]: v for k, v in per_ftype.items()},
                "nhd_cells_by_ftype_note": "per-FType rasterizations counted "
                "separately (a cell touched by two FTypes counts in both)",
                "cell_semantics": "a 100 m cell is masked when the polygon touches "
                "it (rasterio all_touched=True)",
                "grid": {
                    "crs": "EPSG:5070",
                    "res_m": RES,
                    "width": width,
                    "height": height,
                    "origin_x": transform.c,
                    "origin_y": transform.f,
                    "reference": args.dem,
                },
                "n_cells_v3_only": n_v3_only,
                "n_cells_nhd_only": n_nhd_only,
                "n_cells_both": n_both,
                "n_cells_masked": n_v3_only + n_nhd_only + n_both,
                "area_km2_v3_only": n_v3_only * km2,
                "area_km2_nhd_only": n_nhd_only * km2,
                "area_km2_both": n_both * km2,
                "area_km2_masked": (n_v3_only + n_nhd_only + n_both) * km2,
                "named_check": named,
                "cells_parquet": "not written: the renderer reads only the mask "
                "raster; the V3 cells parquet feeds the bundle builder's water_v3 "
                "block, which production bundles do not carry",
                "runtime_s": time.time() - t_start,
                "outputs": {"mask": str(mask_path)},
            },
            indent=2,
        )
    )
    log.info("wrote %s", out_dir / "build_manifest.json")


if __name__ == "__main__":
    main()
