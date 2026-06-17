"""Reduce the NM Water Data SensorThings store (/nas/gwx/nm_sta/) to per-site
median depth-to-water and score Ma WTD against it.

These are NM state/local telemetry monitoring networks (BernCo, City of
Albuquerque, OSE-Roswell, PVACD, EBID, San Acacia Reach) -- NWIS-independent
and more current than the 910-well GWX `nm_sta` subset (full time series here).

We keep only the "Depth to Water Below Ground Surface" datastreams (unit ft),
reduce 1.36M observations to a per-thing median DTW, and report metrics overall
and per agency. Residual = Ma - observed (positive => Ma too deep).

CONFINEMENT CAVEAT: the Roswell-basin networks (OSE-Roswell, PVACD,
CityOfRoswell) tap the Roswell Artesian Basin -- many wells are confined
(potentiometric, often flowing/negative DTW) and must be read as diagnostic,
not water-table. The Albuquerque-basin networks (BernCo, CABQ) are unconfined
basin fill and are the clean water-table comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

FT_TO_M = 0.3048
MA = "/nas/gwx/wtd_states/wtd_new_mexico.tif"
STA = Path("/nas/gwx/nm_sta")
OUT = Path("/data/ssd2/handily/nm/regional/ma_nwis_statewide")
ABQ_BASIN = {"BernCo", "CABQ"}  # unconfined basin fill -> water-table headline


def sample_ma(gdf):
    with rasterio.open(MA) as src:
        pts = gdf.to_crs(src.crs)
        xs = np.array([g.x for g in pts.geometry])
        ys = np.array([g.y for g in pts.geometry])
        vals = np.array(
            [v[0] for v in src.sample(zip(xs, ys), indexes=1)], dtype="float64"
        )
        nod, b = src.nodata, src.bounds
    vals[(xs < b.left) | (xs > b.right) | (ys < b.bottom) | (ys > b.top)] = np.nan
    if nod is not None and np.isfinite(nod):
        vals[vals == nod] = np.nan
    vals[(~np.isfinite(vals)) | (vals > 1e30)] = np.nan
    return vals


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


def main():
    ds = pd.read_parquet(STA / "datastreams.parquet")
    dtw_ds = ds[ds["observed_property_name"] == "Depth to Water Below Ground Surface"]
    ds2thing = dict(zip(dtw_ds["datastream_id"], dtw_ds["thing_id"]))

    obs = pd.read_parquet(
        STA / "observations.parquet",
        columns=["result", "_datastream_id", "phenomenonTime"],
    )
    obs = obs[obs["_datastream_id"].isin(ds2thing)].copy()
    obs["thing_id"] = obs["_datastream_id"].map(ds2thing)
    obs["result"] = pd.to_numeric(obs["result"], errors="coerce")
    obs = obs.dropna(subset=["result"])

    per = (
        obs.groupby("thing_id")
        .agg(
            dtw_ft=("result", "median"),
            n_obs=("result", "size"),
            first_obs=("phenomenonTime", "min"),
            last_obs=("phenomenonTime", "max"),
        )
        .reset_index()
    )

    th = pd.read_parquet(STA / "things.parquet")[
        [
            "thing_id",
            "longitude",
            "latitude",
            "prop_agency",
            "prop_nmbgmr_id",
            "prop_aquifer",
            "prop_well_depth",
        ]
    ]
    w = per.merge(th, on="thing_id", how="left")
    w["dtw_m"] = w["dtw_ft"] * FT_TO_M
    w["lon"] = pd.to_numeric(w["longitude"], errors="coerce")
    w["lat"] = pd.to_numeric(w["latitude"], errors="coerce")
    w = w[w["lon"].between(-110, -102) & w["lat"].between(31, 38)]

    n_artesian = int((w["dtw_m"] < 0).sum())
    g = gpd.GeoDataFrame(
        w, geometry=gpd.points_from_xy(w["lon"], w["lat"]), crs="EPSG:4326"
    )
    g["ma_dtw_m"] = sample_ma(g)
    g = g[g["ma_dtw_m"].notna()].copy()
    g.drop(columns="geometry").to_csv(OUT / "ma_vs_nm_sensorthings.csv", index=False)

    wt = g[g["dtw_m"].between(0, 600)]  # drop artesian/flowing for the table metric
    report = {
        "n_dtw_datastreams": len(dtw_ds),
        "n_sites_reduced": len(per),
        "n_in_nm_ma_raster": len(g),
        "n_flowing_artesian_negative": n_artesian,
        "all_water_table_0_600m": metrics(
            wt["dtw_m"].to_numpy(), wt["ma_dtw_m"].to_numpy()
        ),
        "albuquerque_basin_unconfined": metrics(
            wt[wt["prop_agency"].isin(ABQ_BASIN)]["dtw_m"].to_numpy(),
            wt[wt["prop_agency"].isin(ABQ_BASIN)]["ma_dtw_m"].to_numpy(),
        ),
        "by_agency": {},
    }
    for ag, sub in wt.groupby("prop_agency"):
        report["by_agency"][str(ag)] = metrics(
            sub["dtw_m"].to_numpy(), sub["ma_dtw_m"].to_numpy()
        )

    with open(OUT / "ma_vs_nm_sensorthings_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
