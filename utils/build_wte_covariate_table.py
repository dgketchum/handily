"""Phase 1 of the REM-regional WTE hybrid: build the point covariate table.

One reproducible table (one row per GWX well, optionally one per NHD spring) that
feeds the gate diagnostics, the regional WTE model, and the blend-gate fit. See
``notes/plans/nm_rio_grande_wte_hybrid_code_implementation_plan.md``.

Covariate groups:
  labels      observed DTW, WTE, confinement, tier, weight (GWX), well attrs
  fac         FAC REM DTW/WTE + residuals (the riparian expert)
  benchmark   Ma DTW (kept for reference only; NEVER a training label)
  topo-pos    distance + height-above-drainage to Strahler floors {1,5,7},
              drainage area, slope, local relief  (the gate variables)
  climate     gridMET aridity index P/PET, precip, PET, VPD, Tmax
              (the transferable driver for statewide -> CONUS)
  regional    professional gridded WTD/recharge/ET/BFI/SSURGO products

Sampling mirrors ``compare_fac_ma.sample_raster_at_points``: out-of-bounds and
nodata become NaN (never a spurious 0). Per-covariate coverage is reported so
layers that do not cover the basin are visible rather than silently zero-filled.
"""

import argparse
import json
import tomllib
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

UNCONFINED = {"unconfined", "unconfined_marginal"}
TRAIN_TIERS = {"primary", "secondary"}


def sample_xy(raster_path, xs, ys, raster_crs_xy=True):
    """Sample band 1 at projected (xs, ys); nodata / OOB / non-finite -> NaN."""
    xs = np.asarray(xs, dtype="float64")
    ys = np.asarray(ys, dtype="float64")
    with rasterio.open(raster_path) as src:
        vals = np.array(
            [v[0] for v in src.sample(zip(xs, ys), indexes=1)], dtype="float64"
        )
        nod = src.nodata
        b = src.bounds
    vals[(xs < b.left) | (xs > b.right) | (ys < b.bottom) | (ys > b.top)] = np.nan
    if nod is not None and np.isfinite(nod):
        vals[vals == nod] = np.nan
    vals[~np.isfinite(vals)] = np.nan
    return vals


def sample_at_points(raster_path, pts_gdf):
    """Sample a raster at point geometries (reprojecting points to raster CRS)."""
    with rasterio.open(raster_path) as src:
        rc = src.crs
    p = pts_gdf.to_crs(rc)
    return sample_xy(raster_path, [g.x for g in p.geometry], [g.y for g in p.geometry])


def dist_and_height_above(pts_5070, streams_sub, dem_path):
    """Distance to nearest reach and DEM height above the nearest reach point."""
    streams_sub = streams_sub.reset_index(drop=True)[["geometry"]]
    pts = pts_5070.reset_index()[["index", "geometry"]]
    joined = gpd.sjoin_nearest(
        pts, streams_sub, how="left", distance_col="_dist"
    ).drop_duplicates("index")
    joined = joined.set_index("index").reindex(pts_5070.index)
    cx, cy = [], []
    for idx, pt in pts_5070.geometry.items():
        ri = joined.loc[idx, "index_right"]
        if pd.isna(ri):
            cx.append(np.nan)
            cy.append(np.nan)
            continue
        line = streams_sub.geometry.iloc[int(ri)]
        npt = line.interpolate(line.project(pt))
        cx.append(npt.x)
        cy.append(npt.y)
    dem_pt = sample_xy(
        dem_path, [g.x for g in pts_5070.geometry], [g.y for g in pts_5070.geometry]
    )
    dem_ch = sample_xy(dem_path, cx, cy)
    return joined["_dist"].to_numpy(dtype="float64"), dem_pt - dem_ch


def windowed_slope_relief(dem_path, pts_5070, radius_m):
    """Per-point local relief (max-min) and slope magnitude from a DEM window."""
    relief = np.full(len(pts_5070), np.nan)
    slope = np.full(len(pts_5070), np.nan)
    with rasterio.open(dem_path) as src:
        res = abs(src.transform.a)
        rad = max(1, int(round(radius_m / res)))
        nod = src.nodata
        H, W = src.height, src.width
        for i, pt in enumerate(pts_5070.geometry):
            row, col = src.index(pt.x, pt.y)
            r0, r1 = max(0, row - rad), min(H, row + rad + 1)
            c0, c1 = max(0, col - rad), min(W, col + rad + 1)
            if r1 <= r0 or c1 <= c0:
                continue
            win = src.read(1, window=((r0, r1), (c0, c1))).astype("float64")
            if nod is not None and np.isfinite(nod):
                win[win == nod] = np.nan
            if np.all(~np.isfinite(win)):
                continue
            relief[i] = np.nanmax(win) - np.nanmin(win)
            if win.shape[0] >= 3 and win.shape[1] >= 3:
                gy, gx = np.gradient(win, res)
                slope[i] = float(np.nanmedian(np.hypot(gx, gy)))
    return slope, relief


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    t0 = perf_counter()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    paths = cfg["paths"]
    ct = cfg.get("covariate_table", {})
    out_path = Path(ct["out"])
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- points: wells (+ springs) -------------------------------------------
    wells = gpd.read_parquet(paths["wells"]).to_crs(5070)
    keep = [
        "canonical_id",
        "tier",
        "weight",
        "confinement_class",
        "dtw_label_m",
        "observation_label_type",
        "period_represented",
        "well_depth",
        "screen_top",
        "screen_bottom",
        "aquifer",
        "suspect",
        "exclusion_reason",
    ]
    keep = [c for c in keep if c in wells.columns]
    pts = wells[keep + ["geometry"]].copy()
    pts["point_type"] = "well"
    pts["target_dtw_m"] = pts["dtw_label_m"].astype("float64")

    if ct.get("include_springs") and paths.get("springs"):
        sp = gpd.read_file(paths["springs"]).to_crs(5070)
        sp = sp[["geometry"]].copy()
        sp["point_type"] = "spring"
        sp["target_dtw_m"] = 0.0
        sp["tier"] = "spring"
        sp["weight"] = float(ct.get("spring_weight", 0.25))
        sp["confinement_class"] = "spring"
        pts = pd.concat([pts, sp], ignore_index=True)
    pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=5070)

    pts["is_water_table_label"] = pts["confinement_class"].isin(UNCONFINED)
    pts["is_training_label"] = pts["is_water_table_label"] & pts["tier"].isin(
        TRAIN_TIERS
    )

    # --- DEM + FAC + Ma --------------------------------------------------------
    coverage = {}

    def add(col, vals):
        pts[col] = vals
        coverage[col] = float(np.isfinite(vals).mean())

    add("dem_m", sample_at_points(paths["dem"], pts))
    pts["target_wte_m"] = pts["dem_m"] - pts["target_dtw_m"]
    add("fac_dtw_m", sample_at_points(paths["fac_dtw"], pts))
    pts["fac_wte_m"] = pts["dem_m"] - pts["fac_dtw_m"]
    pts["fac_residual_dtw_m"] = pts["fac_dtw_m"] - pts["target_dtw_m"]
    pts["fac_residual_wte_m"] = pts["fac_wte_m"] - pts["target_wte_m"]
    if paths.get("ma_benchmark"):
        add("ma_dtw_m", sample_at_points(paths["ma_benchmark"], pts))

    # --- drainage area ---------------------------------------------------------
    with rasterio.open(paths["flow_accumulation"]) as src:
        cell_km2 = abs(src.transform.a * src.transform.e) / 1e6
    accum = sample_at_points(paths["flow_accumulation"], pts)
    pts["drainage_km2"] = accum * cell_km2
    pts["log1p_drainage_km2"] = np.log1p(pts["drainage_km2"])

    # --- distance + height-above-drainage to Strahler floors -------------------
    streams = gpd.read_file(paths["streams"]).to_crs(5070)
    for floor in ct.get("distance_strahler_floors", [1, 5, 7]):
        sub = streams[streams["strahler"] >= floor]
        if sub.empty:
            continue
        dist, hgt = dist_and_height_above(pts, sub, paths["dem"])
        pts[f"dist_str{floor}_m"] = dist
        pts[f"log1p_dist_str{floor}_m"] = np.log1p(dist)
        pts[f"elev_above_str{floor}_m"] = hgt

    # --- slope + relief --------------------------------------------------------
    slope, relief = windowed_slope_relief(
        paths["dem"], pts, float(ct.get("relief_radius_m", 300.0))
    )
    pts["slope_m_m"] = slope
    rr = int(float(ct.get("relief_radius_m", 300.0)))
    pts[f"local_relief_{rr}m_m"] = relief

    # --- climate ---------------------------------------------------------------
    from handily.climate_normals import sample_gridmet_normals

    clim = sample_gridmet_normals(pts, cfg["climate"]["gridmet_dir"])
    for c in clim.columns:
        add(c, clim[c].to_numpy())

    # --- professional gridded covariates --------------------------------------
    for name, path in cfg.get("covariates", {}).items():
        if not Path(path).exists():
            coverage[name] = 0.0
            continue
        add(name, sample_at_points(path, pts))

    # --- spatial CV blocks -----------------------------------------------------
    pts["x_5070"] = pts.geometry.x
    pts["y_5070"] = pts.geometry.y
    ll = pts.to_crs(4326)
    pts["lon"] = ll.geometry.x
    pts["lat"] = ll.geometry.y
    for b in ct.get("block_sizes_m", [10000, 20000, 50000]):
        pts[f"block_{b // 1000}km"] = (
            (pts["x_5070"] // b).astype("int64").astype(str)
            + "_"
            + (pts["y_5070"] // b).astype("int64").astype(str)
        )

    # --- write -----------------------------------------------------------------
    pts.to_parquet(out_path)
    run = {
        "config": str(args.config),
        "out": str(out_path),
        "n_points": int(len(pts)),
        "n_wells": int((pts.point_type == "well").sum()),
        "n_springs": int((pts.point_type == "spring").sum()),
        "n_water_table": int(pts.is_water_table_label.sum()),
        "n_training": int(pts.is_training_label.sum()),
        "coverage_fraction": coverage,
        "runtime_s": round(perf_counter() - t0, 1),
    }
    with open(out_path.with_suffix(".run.json"), "w") as f:
        json.dump(run, f, indent=2)

    print(f"wrote {out_path}  ({len(pts)} pts, {pts.shape[1]} cols)")
    print(json.dumps(run, indent=2))
    low = {k: round(v, 2) for k, v in coverage.items() if v < 0.9}
    if low:
        print("LOW-COVERAGE covariates (<90% finite):", low)


if __name__ == "__main__":
    main()
