"""Consolidate every NWIS-independent NM water-table observation we have into one
validation layer, score Ma against it, and write it ready for FAC-prior scoring.

Sources unioned (all NWIS-independent, DTW positive-down in meters):
  - ose_driller     : NM OSE POD driller logs, depth_wate (genuine drill-time DTW;
                      static_lev is a redundant dup, dropped). 100k+ wells, no
                      confinement typing -> confinement='unknown'.
  - nm_sensorthings : NM Water Data telemetry (BernCo/CABQ/OSE-Roswell/PVACD/EBID/
                      SanAcacia), per-site median DTW reduced from 1.36M obs.
                      ABQ-basin = unconfined; Roswell basin = artesian (confined).
  - eastern_abq_*   : USGS ScienceBase deep ABQ-basin water table (KAFB/SNL/CABQ/
                      AECOM source rows only), unconfined.
  - sfgas_2016      : Santa Fe Group deep production-zone wells, unconfined.

Two national WTD baselines are sampled at every well and stored as columns:
  ma_dtw_m / resid_m              : Ma et al. WTD (resid = ma - observed)
  janssen_v1_dtw_m / janssen_v1_resid_m : Janssen et al. (UBC) V1 (real-obs-only
      XGBoost; the deep-basin-best of V1/V2/V3, see score_janssen_vs_ma.py).
Both trained on USGS NWIS, so these non-NWIS wells are independent of both.
Residual = predicted - observed (positive => product too deep). Output:
  nm_independent_validation_wells.{geoparquet,csv} + _summary.json
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
JANSSEN_V1 = "/nas/gwx/janssen/V1_140.tif"
BASELINES = ("ma_dtw_m", "janssen_v1_dtw_m")
STUD = Path("/nas/gwx/studies/nm_independent_wells")
STA_CSV = Path(
    "/data/ssd2/handily/nm/regional/ma_nwis_statewide/ma_vs_nm_sensorthings.csv"
)
OUT = Path("/data/ssd2/handily/nm/regional/ma_nwis_statewide")
NM = dict(lon=(-109.1, -102.9), lat=(31.2, 37.1))

UNCONF = {"BernCo", "CABQ"}  # Albuquerque basin fill
ARTESIAN = {"OSE-Roswell", "PVACD", "CityOfRoswell"}  # Roswell Artesian Basin
VALLEY = {
    "EBID",
    "SanAcaciaReach",
    "EBWPC",
}  # Rio Grande / Mesilla valley, shallow unconfined


def sample_raster(path, gdf):
    """Sample a WTD raster at well points, reprojecting to its CRS; out-of-bounds
    and nodata (either sign of ±3.4e38) -> NaN, never a fill value."""
    with rasterio.open(path) as src:
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
    vals[(~np.isfinite(vals)) | (np.abs(vals) > 1e30)] = np.nan
    return vals


def metrics(df, pred_col="ma_dtw_m"):
    o, p = df["dtw_m"].to_numpy(), df[pred_col].to_numpy()
    valid = np.isfinite(o) & np.isfinite(p)
    o, p = o[valid], p[valid]
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


def ose_driller():
    o = pd.read_parquet(
        "/nas/gwx/nm_ose/pod.parquet",
        columns=["_longitude", "_latitude", "depth_wate", "finish_dat"],
    )
    lon = pd.to_numeric(o["_longitude"], errors="coerce")
    lat = pd.to_numeric(o["_latitude"], errors="coerce")
    dtw = pd.to_numeric(o["depth_wate"], errors="coerce")
    m = (
        lon.between(*NM["lon"])
        & lat.between(*NM["lat"])
        & dtw.between(0, 3000, inclusive="neither")
    )
    o = pd.DataFrame(
        {
            "lon": lon[m],
            "lat": lat[m],
            "dtw_m": dtw[m] * FT_TO_M,
            "date": o["finish_dat"][m],
        }
    )
    o["cell"] = list(zip(o["lon"].round(4), o["lat"].round(4)))
    w = (
        o.groupby("cell")
        .agg(
            lon=("lon", "first"),
            lat=("lat", "first"),
            dtw_m=("dtw_m", "median"),
            n_obs=("dtw_m", "size"),
            date=("date", "max"),
        )
        .reset_index(drop=True)
    )
    w["source"], w["agency"], w["confinement"] = "ose_driller", "NM_OSE", "unknown"
    return w


def sensorthings():
    s = pd.read_csv(STA_CSV)
    s = s.rename(columns={"prop_agency": "agency", "last_obs": "date"})
    s = s[s["dtw_m"].between(0, 600)].copy()
    s["source"] = "nm_sensorthings"
    s["confinement"] = np.where(
        s["agency"].isin(UNCONF | VALLEY),
        "unconfined",
        np.where(s["agency"].isin(ARTESIAN), "artesian_confined", "unknown"),
    )
    return s[
        ["lon", "lat", "dtw_m", "n_obs", "date", "source", "agency", "confinement"]
    ]


def study():
    rows = []
    for yr in ("2016", "2008"):
        d = pd.read_csv(
            STUD / f"eastern_albuquerque_wt/Welldata{yr}.txt", sep="\t", engine="python"
        )
        d = d[~d["Source"].isin(["NWIS"]) & d["Source"].notna()]
        rows.append(
            pd.DataFrame(
                {
                    "lon": pd.to_numeric(d["Long"], errors="coerce"),
                    "lat": pd.to_numeric(d["Lat"], errors="coerce"),
                    "dtw_m": pd.to_numeric(d["DTW_ft"], errors="coerce") * FT_TO_M,
                    "source": f"eastern_abq_{yr}",
                    "agency": d["Source"].astype(str),
                    "confinement": "unconfined",
                }
            )
        )
    sf = pd.read_csv(
        STUD / "albuquerque_sfgas_2016/Wells_and_wls_WY2016_SFGAS.txt",
        sep="\t",
        engine="python",
    )
    sf = sf[~sf["SOURCE_WL"].str.contains("USGS", na=False)]
    g = gpd.GeoDataFrame(
        sf,
        geometry=gpd.points_from_xy(
            pd.to_numeric(sf["X_N83SPCFT"], errors="coerce"),
            pd.to_numeric(sf["Y_N83SPCFT"], errors="coerce"),
        ),
        crs="EPSG:2903",
    ).to_crs(4326)
    rows.append(
        pd.DataFrame(
            {
                "lon": g.geometry.x,
                "lat": g.geometry.y,
                "dtw_m": pd.to_numeric(sf["DTW_FT"], errors="coerce") * FT_TO_M,
                "source": "sfgas_2016",
                "agency": "SFGAS",
                "confinement": "unconfined",
            }
        )
    )
    out = pd.concat(rows, ignore_index=True)
    return out[out["dtw_m"].between(0, 600)]


def main():
    df = pd.concat([ose_driller(), sensorthings(), study()], ignore_index=True)
    df = df[df["lon"].between(*NM["lon"]) & df["lat"].between(*NM["lat"])].copy()
    df["date"] = df["date"].astype(
        "string"
    )  # OSE int + STA ISO str + study <NA> -> uniform
    g = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )
    g["ma_dtw_m"] = sample_raster(MA, g)
    g["janssen_v1_dtw_m"] = sample_raster(JANSSEN_V1, g)
    # Membership = wells with a valid Ma value (its tile is the limiting footprint).
    g = g[g["ma_dtw_m"].notna()].copy()
    g["resid_m"] = g["ma_dtw_m"] - g["dtw_m"]
    g["janssen_v1_resid_m"] = g["janssen_v1_dtw_m"] - g["dtw_m"]
    n_jv1_missing = int(g["janssen_v1_dtw_m"].isna().sum())
    print(f"Ma-valid wells missing Janssen V1 coverage: {n_jv1_missing} / {len(g)}")

    g.to_parquet(OUT / "nm_independent_validation_wells.geoparquet")
    g.drop(columns="geometry").to_csv(
        OUT / "nm_independent_validation_wells.csv", index=False
    )

    def by(subset):
        return {b: metrics(subset, b) for b in BASELINES}

    summary = {
        "n_total": len(g),
        "baselines": list(BASELINES),
        "all_independent": by(g),
        "water_table_only": by(g[g["confinement"] == "unconfined"]),
        "by_source": {s: by(sub) for s, sub in g.groupby("source")},
        "by_confinement": {c: by(sub) for c, sub in g.groupby("confinement")},
    }
    with open(OUT / "nm_independent_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
