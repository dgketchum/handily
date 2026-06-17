"""Score Ma WTD against the curated, source-tagged NWIS-INDEPENDENT study wells
downloaded to /nas/gwx/studies/nm_independent_wells/ (federal ScienceBase recon).

Each release carries an explicit source column, so we isolate the non-NWIS
records and score Ma only on those. These are the deep Albuquerque-basin /
Mesilla regional water-table wells -- the regime where Ma and the FAC prior fail.

Residual = Ma - observed DTW (positive => Ma too deep). All depths positive-down.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

FT_TO_M = 0.3048
# Mesilla / RGTIHM wells straddle the NM-TX line, so sample both state Ma tiles
# (identical CRS, each clipped to its state) and coalesce.
MA_TILES = [
    "/nas/gwx/wtd_states/wtd_new_mexico.tif",
    "/nas/gwx/wtd_states/wtd_texas.tif",
]
BASE = Path("/nas/gwx/studies/nm_independent_wells")
OUT = Path("/data/ssd2/handily/nm/regional/ma_nwis_statewide")


def sample_ma(gdf):
    with rasterio.open(MA_TILES[0]) as s0:
        pts = gdf.to_crs(s0.crs)
    xs = np.array([g.x for g in pts.geometry])
    ys = np.array([g.y for g in pts.geometry])
    out = np.full(len(xs), np.nan)
    for path in MA_TILES:
        with rasterio.open(path) as src:
            vals = np.array(
                [v[0] for v in src.sample(zip(xs, ys), indexes=1)], dtype="float64"
            )
            nod, b = src.nodata, src.bounds
        vals[(xs < b.left) | (xs > b.right) | (ys < b.bottom) | (ys > b.top)] = np.nan
        if nod is not None and np.isfinite(nod):
            vals[vals == nod] = np.nan
        vals[(~np.isfinite(vals)) | (vals > 1e30)] = np.nan
        take = np.isnan(out) & np.isfinite(vals)
        out[take] = vals[take]
    return out


def metrics(obs, pred):
    valid = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[valid], pred[valid]
    if o.size == 0:
        return {"n": 0}
    r = p - o
    return {
        "n": int(o.size),
        "obs_med_m": round(float(np.median(o)), 1),
        "pred_med_m": round(float(np.median(p)), 1),
        "MAD_m": round(float(np.median(np.abs(r))), 2),
        "bias_m": round(float(np.mean(r)), 2),
        "med_resid_m": round(float(np.median(r)), 2),
        "rmse_m": round(float(np.sqrt(np.mean(r**2))), 1),
        "corr": round(float(np.corrcoef(o, p)[0, 1]), 3) if o.size > 2 else None,
    }


def score(gdf, dtw_ft_col, label, report, csv_name):
    gdf = gdf.copy()
    gdf["dtw_m"] = pd.to_numeric(gdf[dtw_ft_col], errors="coerce") * FT_TO_M
    gdf = gdf[gdf["dtw_m"].between(-30, 600)]  # drop nulls / impossible
    gdf["ma_dtw_m"] = sample_ma(gdf)
    gin = gdf[gdf["ma_dtw_m"].notna()].copy()
    report[label] = {
        "n_independent": len(gdf),
        "n_in_ma": len(gin),
        "metrics": metrics(gin["dtw_m"].to_numpy(), gin["ma_dtw_m"].to_numpy()),
    }
    gin.drop(columns="geometry").to_csv(OUT / csv_name, index=False)
    print(f"{label:32s} {report[label]['metrics']}")


def main():
    report = {}

    # --- Eastern Albuquerque WT (Lat/Long WGS84, DTW_ft, Source) ---
    for yr, sep in (("2016", "\t"), ("2008", "\t")):
        df = pd.read_csv(
            BASE / f"eastern_albuquerque_wt/Welldata{yr}.txt", sep=sep, engine="python"
        )
        ind = df[~df["Source"].isin(["NWIS"]) & df["Source"].notna()].copy()
        g = gpd.GeoDataFrame(
            ind,
            geometry=gpd.points_from_xy(
                pd.to_numeric(ind["Long"], errors="coerce"),
                pd.to_numeric(ind["Lat"], errors="coerce"),
            ),
            crs="EPSG:4326",
        )
        score(
            g,
            "DTW_ft",
            f"eastern_abq_{yr}_indep",
            report,
            f"ma_vs_eastern_abq_{yr}.csv",
        )

    # --- Santa Fe Group 2016 (NM State Plane Central ftUS = EPSG:2903, DTW_FT, SOURCE_WL) ---
    sf = pd.read_csv(
        BASE / "albuquerque_sfgas_2016/Wells_and_wls_WY2016_SFGAS.txt",
        sep="\t",
        engine="python",
    )
    sfi = sf[~sf["SOURCE_WL"].str.contains("USGS", na=False)].copy()
    g = gpd.GeoDataFrame(
        sfi,
        geometry=gpd.points_from_xy(
            pd.to_numeric(sfi["X_N83SPCFT"], errors="coerce"),
            pd.to_numeric(sfi["Y_N83SPCFT"], errors="coerce"),
        ),
        crs="EPSG:2903",
    )
    ll = g.to_crs(4326)
    print(
        "SFGAS reprojected lon/lat sample:",
        round(ll.geometry.x.median(), 3),
        round(ll.geometry.y.median(), 3),
    )
    score(g, "DTW_FT", "santa_fe_group_2016_indep", report, "ma_vs_sfgas_2016.csv")

    # --- RGTIHM IBWC (independent) head obs: join HOB table to locations shp ---
    hob = pd.read_csv(BASE / "rgtihm_hob/RGTIHM_HOB.csv")
    ibwc = hob[hob["WL_source"].str.contains("International Boundary", na=False)]
    perwell = ibwc.groupby("Site_ID").agg(dtw_ft=("DTW_ft", "median")).reset_index()
    loc = gpd.read_file(f"zip://{BASE}/rgtihm_hob/RGTIHM_HOB_Locations.zip")
    idcol = next(
        (c for c in loc.columns if c.lower() in ("site_id", "siteid", "site_no")), None
    )
    print(
        "RGTIHM loc cols:",
        list(loc.columns)[:12],
        "| id match:",
        idcol,
        "| crs:",
        loc.crs,
    )
    if idcol:
        m = loc.merge(perwell, left_on=idcol, right_on="Site_ID", how="inner")
        g = gpd.GeoDataFrame(m, geometry=m.geometry, crs=loc.crs)
        score(g, "dtw_ft", "rgtihm_ibwc_indep", report, "ma_vs_rgtihm_ibwc.csv")

    with open(OUT / "ma_vs_study_wells_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
