"""Score Ma WTD against NWIS-independent wells in the TAAP Fort Bliss/WSMR
geodatabase (/nas/gwx/studies/nm_independent_wells/taap_fortbliss_wlwq/).

Independent point depth-to-water sources (all non-NWIS), DTW in feet, positive-down:
  - tbl_TWDB_DiscreteWL  : TWDB discrete measurements, inline LatitudeDD/LongitudeDD,
                           WaterLevel = DTW ft. MeasuringAgency != USGS kept only.
  - tbl_SDRDB_WellLevels : Texas state driller-report levels (Measurement ft),
                           joined to tbl_SDRDB_welllocs for coordinates.

The Hueco/Mesilla bolson straddles the NM-TX line, so we sample BOTH state Ma
tiles (`wtd_new_mexico.tif` + `wtd_texas.tif`, identical CRS, each clipped to its
state) and coalesce -- this lets the deep Texas-side bolson wells finally score.
Results split by tile. Residual = Ma - observed DTW (positive => Ma too deep).
"""

from __future__ import annotations

import json
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

FT_TO_M = 0.3048
MA_TILES = {
    "nm": "/nas/gwx/wtd_states/wtd_new_mexico.tif",
    "tx": "/nas/gwx/wtd_states/wtd_texas.tif",
}
GDB = "zip:///nas/gwx/studies/nm_independent_wells/taap_fortbliss_wlwq/TAAPGIS_WLWQ.gdb.zip"
OUT = Path("/data/ssd2/handily/nm/regional/ma_nwis_statewide")


def layer_df(layer):
    with fiona.open(GDB, layer=layer) as src:
        return pd.DataFrame([r["properties"] for r in src])


def sample_tiles(gdf):
    """Sample each state Ma tile and coalesce; tiles share CRS and are state-clipped
    so each well is valid in exactly one. Returns (values, tile_name array)."""
    names, paths = list(MA_TILES), list(MA_TILES.values())
    with rasterio.open(paths[0]) as s0:
        pts = gdf.to_crs(s0.crs)
    xs = np.array([g.x for g in pts.geometry])
    ys = np.array([g.y for g in pts.geometry])
    out = np.full(len(xs), np.nan)
    tile = np.array([""] * len(xs), dtype=object)
    for nm_, p in zip(names, paths):
        with rasterio.open(p) as src:
            vals = np.array(
                [v[0] for v in src.sample(zip(xs, ys), indexes=1)], dtype="float64"
            )
            nod, b = src.nodata, src.bounds
        vals[(xs < b.left) | (xs > b.right) | (ys < b.bottom) | (ys > b.top)] = np.nan
        if nod is not None and np.isfinite(nod):
            vals[vals == nod] = np.nan
        vals[(~np.isfinite(vals)) | (vals > 1e30)] = np.nan
        take = np.isnan(out) & np.isfinite(vals)
        out[take], tile[take] = vals[take], nm_
    return out, tile


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


def score(wells, label, report):
    """wells: DataFrame with lon, lat, dtw_ft."""
    wells = wells.copy()
    wells["dtw_m"] = pd.to_numeric(wells["dtw_ft"], errors="coerce") * FT_TO_M
    wells["lon"] = pd.to_numeric(wells["lon"], errors="coerce")
    wells["lat"] = pd.to_numeric(wells["lat"], errors="coerce")
    wells = wells[
        wells["dtw_m"].between(0, 600)
        & wells["lon"].between(-108, -93)
        & wells["lat"].between(25, 37)
    ]
    g = gpd.GeoDataFrame(
        wells, geometry=gpd.points_from_xy(wells["lon"], wells["lat"]), crs="EPSG:4326"
    )
    g["ma_dtw_m"], g["tile"] = sample_tiles(g)
    gin = g[g["ma_dtw_m"].notna()].copy()
    gin["resid_m"] = gin["ma_dtw_m"] - gin["dtw_m"]
    tx = gin[gin["tile"] == "tx"]
    bands = {}
    for lo, hi in [(0, 10), (10, 50), (50, 150), (150, 600)]:
        s = tx[tx["dtw_m"].between(lo, hi)]
        bands[f"{lo}-{hi}m"] = metrics(s["dtw_m"].to_numpy(), s["ma_dtw_m"].to_numpy())
    report[label] = {
        "n_wells_total": len(g),
        "n_scored": len(gin),
        "all": metrics(gin["dtw_m"].to_numpy(), gin["ma_dtw_m"].to_numpy()),
        "by_tile": {
            t: metrics(sub["dtw_m"].to_numpy(), sub["ma_dtw_m"].to_numpy())
            for t, sub in gin.groupby("tile")
        },
        "tx_by_depth_band": bands,
    }
    gin.drop(columns="geometry").to_csv(OUT / f"ma_vs_taap_{label}.csv", index=False)
    print(f"{label:22s} all={report[label]['all']}")
    for t, m in report[label]["by_tile"].items():
        print(f"  {t:4s} {m}")


def main():
    report = {}

    # TWDB discrete WL (coords inline). Drop USGS-measured rows, but NOTE: TWDB's
    # GWDB exchanges data with USGS, so even non-USGS-measured rows likely overlap
    # NWIS -- Ma fits them too well (deep band MAD ~1.5 m), so treat TWDB as
    # contaminated, NOT clean independent. SDRDB driller logs are the clean set.
    twdb = layer_df("tbl_TWDB_DiscreteWL")
    twdb["WaterLevel"] = pd.to_numeric(twdb["WaterLevel"], errors="coerce")
    twdb = twdb.dropna(subset=["WaterLevel"])
    print(
        "TWDB MeasuringAgency counts:\n",
        twdb["MeasuringAgency"].value_counts().head(12).to_string(),
    )
    twdb = twdb[
        ~twdb["MeasuringAgency"].astype(str).str.contains("USGS", case=False, na=False)
    ]
    pw = (
        twdb.groupby("StateWellNumber")
        .agg(
            dtw_ft=("WaterLevel", "median"),
            lon=("LongitudeDD", "first"),
            lat=("LatitudeDD", "first"),
            agency=("MeasuringAgency", "first"),
        )
        .reset_index()
    )
    score(pw, "twdb_discrete", report)

    # SDRDB driller levels joined to locations (Texas state driller reports, independent)
    lv = layer_df("tbl_SDRDB_WellLevels")
    lv["Measurement"] = pd.to_numeric(lv["Measurement"], errors="coerce")
    lvw = (
        lv.dropna(subset=["Measurement"])
        .groupby("WellReportTrackingNumber")
        .agg(dtw_ft=("Measurement", "median"))
        .reset_index()
    )
    loc = layer_df("tbl_SDRDB_welllocs")[
        ["WellReport", "LatitudeDD", "LongitudeD", "County", "ProposedUs"]
    ]
    sd = lvw.merge(
        loc, left_on="WellReportTrackingNumber", right_on="WellReport", how="inner"
    )
    sd = sd.rename(columns={"LatitudeDD": "lat", "LongitudeD": "lon"})
    score(sd, "sdrdb_driller", report)

    with open(OUT / "ma_vs_taap_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
