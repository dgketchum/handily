"""Build a NWIS-independent NM well set and score the Ma WTD raster against it.

Under the assumption that Ma et al. trained on USGS NWIS, wells from sources
OTHER than NWIS are independent validation labels. This pulls two on-disk,
non-NWIS sources statewide:

  1. NM OSE points-of-diversion driller records (/nas/gwx/nm_ose/pod.parquet) --
     driller-reported water levels. Two candidate fields (static_lev, depth_wate);
     we score both and report their agreement (semantics are driller-grade, single
     measurement at drilling time, NO confinement typing -> screening quality).
  2. NM SensorThings / NMBGMR monitoring wells, taken from the curated GWX product
     (source == 'nm_sta'), which carry mean_dtw + confinement_class -> the
     project DTW rule (unconfined only) can be applied honestly here.

Residual = Ma - observed (positive => Ma too deep). All depths positive-down.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio

FT_TO_M = 0.3048
MA = "/nas/gwx/wtd_states/wtd_new_mexico.tif"
OUT = Path("/data/ssd2/handily/nm/regional/ma_nwis_statewide")
NM_BBOX = dict(lon=(-109.1, -102.9), lat=(31.2, 37.1))
SHALLOW = (5.0, 10.0)


def sample_ma(gdf: gpd.GeoDataFrame) -> np.ndarray:
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
    out = {
        "n": int(o.size),
        "obs_med_m": round(float(np.median(o)), 1),
        "pred_med_m": round(float(np.median(p)), 1),
        "MAD_m": round(float(np.median(np.abs(r))), 2),
        "bias_m": round(float(np.mean(r)), 2),
        "med_resid_m": round(float(np.median(r)), 2),
        "rmse_m": round(float(np.sqrt(np.mean(r**2))), 1),
        "corr": round(float(np.corrcoef(o, p)[0, 1]), 3) if o.size > 2 else None,
    }
    for lvl in SHALLOW:
        os_, ps_ = o < lvl, p < lvl
        tp = int((os_ & ps_).sum())
        out[f"rec<{int(lvl)}"] = round(tp / int(os_.sum()), 3) if os_.sum() else None
    return out


def numeric_clean(s, lo, hi):
    v = pd.to_numeric(s, errors="coerce")
    return v.where((v > lo) & (v < hi))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    # ---------------- NM OSE driller records ----------------
    o = pd.read_parquet(
        "/nas/gwx/nm_ose/pod.parquet",
        columns=[
            "_longitude",
            "_latitude",
            "static_lev",
            "depth_wate",
            "depth_well",
            "use_of_wel",
        ],
    )
    lon = pd.to_numeric(o["_longitude"], errors="coerce")
    lat = pd.to_numeric(o["_latitude"], errors="coerce")
    o = o[lon.between(*NM_BBOX["lon"]) & lat.between(*NM_BBOX["lat"])].copy()
    o["lon"], o["lat"] = lon[o.index], lat[o.index]
    o["static_lev_m"] = (
        numeric_clean(o["static_lev"], 0, 3000) * FT_TO_M
    )  # ft, >0, <3000ft
    o["depth_wate_m"] = numeric_clean(o["depth_wate"], 0, 3000) * FT_TO_M
    both = o["static_lev_m"].notna() & o["depth_wate_m"].notna()
    report["ose_field_agreement"] = {
        "n_both_fields": int(both.sum()),
        "median_abs_diff_m": round(
            float(
                (o.loc[both, "static_lev_m"] - o.loc[both, "depth_wate_m"])
                .abs()
                .median()
            ),
            2,
        ),
        "corr": round(
            float(o.loc[both, ["static_lev_m", "depth_wate_m"]].corr().iloc[0, 1]), 3
        ),
    }

    ose_results = {}
    for fld in ("static_lev_m", "depth_wate_m"):
        sub = o[o[fld].notna()].copy()
        # dedup co-located POD records: median water level per ~10 m cell
        sub["cell"] = list(zip(sub["lon"].round(4), sub["lat"].round(4)))
        w = (
            sub.groupby("cell")
            .agg(
                dtw_m=(fld, "median"),
                lon=("lon", "first"),
                lat=("lat", "first"),
                n_pod=(fld, "size"),
            )
            .reset_index(drop=True)
        )
        gdf = gpd.GeoDataFrame(
            w, geometry=gpd.points_from_xy(w["lon"], w["lat"]), crs="EPSG:4326"
        )
        gdf["ma_dtw_m"] = sample_ma(gdf)
        in_ma = gdf["ma_dtw_m"].notna()
        ose_results[fld] = {
            "wells_dedup": len(gdf),
            "in_ma_raster": int(in_ma.sum()),
            "metrics": metrics(gdf["dtw_m"].to_numpy(), gdf["ma_dtw_m"].to_numpy()),
        }
        gdf[in_ma].drop(columns="geometry").assign(field=fld).to_csv(
            OUT / f"ma_vs_ose_{fld}.csv", index=False
        )
    report["nm_ose"] = ose_results

    # ---------------- NMBGMR monitoring (nm_sta) from GWX product ----------------
    t = pq.read_table(
        "/nas/gwx/products/wells.geoparquet",
        columns=[
            "source",
            "longitude",
            "latitude",
            "mean_dtw",
            "confinement_class",
            "obs_count",
            "has_time_series",
        ],
    ).to_pandas()
    sta = t[
        (t.source == "nm_sta")
        & t.longitude.between(*NM_BBOX["lon"])
        & t.latitude.between(*NM_BBOX["lat"])
        & t.mean_dtw.notna()
    ].copy()
    gsta = gpd.GeoDataFrame(
        sta, geometry=gpd.points_from_xy(sta.longitude, sta.latitude), crs="EPSG:4326"
    )
    gsta["ma_dtw_m"] = sample_ma(gsta)
    gsta = gsta[gsta["ma_dtw_m"].notna()].copy()
    sta_res = {
        "n_in_ma": len(gsta),
        "all": metrics(gsta["mean_dtw"].to_numpy(), gsta["ma_dtw_m"].to_numpy()),
    }
    for bucket in (
        "unconfined",
        "unconfined_marginal",
        "confined",
        "likely_confined",
        "unknown",
    ):
        m = (gsta["confinement_class"] == bucket).to_numpy()
        sta_res[bucket] = metrics(
            gsta["mean_dtw"].to_numpy()[m], gsta["ma_dtw_m"].to_numpy()[m]
        )
    wt = (
        gsta["confinement_class"].isin(["unconfined", "unconfined_marginal"]).to_numpy()
    )
    sta_res["water_table_only"] = metrics(
        gsta["mean_dtw"].to_numpy()[wt], gsta["ma_dtw_m"].to_numpy()[wt]
    )
    report["nm_sta_nmbgmr"] = sta_res
    gsta.drop(columns="geometry").to_csv(OUT / "ma_vs_nmbgmr_nm_sta.csv", index=False)

    with open(OUT / "ma_vs_independent_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
