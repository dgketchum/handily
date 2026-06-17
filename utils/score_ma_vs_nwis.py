"""Score the Ma et al. national WTD product against NWIS depth-to-water wells.

For a single state: aggregate per-site median depth-to-water (NWIS parameter
72019, feet below land surface) from /nas/gwx/nwis/measurements/<ST>_72019.parquet,
join the NWIS site file for confinement (aquifer_type_code) and site type, sample
the Ma WTD raster at each well, and report accuracy metrics.

Both quantities are depth below land surface, positive-down, so they compare
directly. Residual = predicted (Ma) - observed (well); positive = Ma too deep.

Per the project DTW rule, only UNCONFINED wells (aquifer_type_code U or N) belong
in headline accuracy; confined (C/M) wells measure a potentiometric surface and
are reported separately, never folded into the headline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

FT_TO_M = 0.3048
SHALLOW_LEVELS = (2.0, 5.0, 10.0)
# NWIS aquifer_type_code -> confinement bucket
CONF = {
    "U": "unconfined",
    "N": "unconfined",
    "C": "confined",
    "M": "confined",
    "X": "mixed",
}


def sample_raster_at_points(raster_path: str, gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Sample band 1; out-of-bounds / nodata / non-finite -> NaN."""
    with rasterio.open(raster_path) as src:
        pts = gdf.to_crs(src.crs)
        xs = np.array([g.x for g in pts.geometry], dtype="float64")
        ys = np.array([g.y for g in pts.geometry], dtype="float64")
        vals = np.array(
            [v[0] for v in src.sample(zip(xs, ys), indexes=1)], dtype="float64"
        )
        nod, b = src.nodata, src.bounds
    vals[(xs < b.left) | (xs > b.right) | (ys < b.bottom) | (ys > b.top)] = np.nan
    if nod is not None and np.isfinite(nod):
        vals[vals == nod] = np.nan
    vals[(~np.isfinite(vals)) | (vals > 1e30)] = np.nan
    return vals


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    valid = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[valid], pred[valid]
    if o.size == 0:
        return {"n": 0}
    r = p - o
    out = {
        "n": int(o.size),
        "obs_dtw_median_m": round(float(np.median(o)), 2),
        "pred_dtw_median_m": round(float(np.median(p)), 2),
        "MAD_m": round(float(np.median(np.abs(r))), 2),
        "bias_m": round(float(np.mean(r)), 2),
        "median_resid_m": round(float(np.median(r)), 2),
        "rmse_m": round(float(np.sqrt(np.mean(r**2))), 2),
        "corr": round(float(np.corrcoef(o, p)[0, 1]), 3) if o.size > 2 else None,
    }
    for lvl in SHALLOW_LEVELS:
        os_, ps_ = o < lvl, p < lvl
        tp = int((os_ & ps_).sum())
        out[f"recall_<{int(lvl)}m"] = (
            round(tp / int(os_.sum()), 3) if os_.sum() else None
        )
        out[f"prec_<{int(lvl)}m"] = round(tp / int(ps_.sum()), 3) if ps_.sum() else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default="NM")
    ap.add_argument("--ma", default="/nas/gwx/wtd_states/wtd_new_mexico.tif")
    ap.add_argument(
        "--out-dir", default="/data/ssd2/handily/nm/regional/ma_nwis_statewide"
    )
    ap.add_argument(
        "--static-only", action="store_true", help="keep only [Static] measurements"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    st = args.state

    meas = pd.read_parquet(f"/nas/gwx/nwis/measurements/{st}_72019.parquet")
    diag = {
        "raw_measurements": len(meas),
        "unit_values": meas["unit_of_measure"].value_counts(dropna=False).to_dict(),
    }
    diag["value_null"] = int(meas["value"].isna().sum())
    diag["qualifier_top"] = (
        meas["qualifier"].astype(str).value_counts().head(6).to_dict()
    )

    meas = meas[meas["unit_of_measure"] == "ft"].copy()
    if args.static_only:
        meas = meas[meas["qualifier"].astype(str).str.contains("Static", na=False)]
    val_num = pd.to_numeric(meas["value"], errors="coerce")
    diag["value_nonnumeric"] = int(val_num.isna().sum() - meas["value"].isna().sum())
    meas["value"] = val_num
    meas = meas.dropna(subset=["value", "longitude", "latitude"])
    # Guard against obviously bad sentinels masquerading as depths.
    meas = meas[(meas["value"] > -100) & (meas["value"] < 5000)]

    g = meas.groupby("monitoring_location_id")
    wells = pd.DataFrame(
        {
            "dtw_m": g["value"].median() * FT_TO_M,
            "n_obs": g["value"].size(),
            "lon": g["longitude"].median(),
            "lat": g["latitude"].median(),
        }
    ).reset_index()
    diag["unique_sites_with_dtw"] = len(wells)

    sites = pd.read_parquet(f"/nas/gwx/nwis/sites/{st}.parquet")
    sites = sites[
        [
            "id",
            "aquifer_type_code",
            "site_type_code",
            "state_name",
            "national_aquifer_code",
        ]
    ].copy()
    wells = wells.merge(
        sites, left_on="monitoring_location_id", right_on="id", how="left"
    )
    diag["site_join_matched"] = int(wells["id"].notna().sum())
    diag["aqfr_type_dist"] = (
        wells["aquifer_type_code"].value_counts(dropna=False).to_dict()
    )
    wells["confinement"] = wells["aquifer_type_code"].map(CONF).fillna("unknown")

    gdf = gpd.GeoDataFrame(
        wells, geometry=gpd.points_from_xy(wells["lon"], wells["lat"]), crs="EPSG:4326"
    )
    gdf["ma_dtw_m"] = sample_raster_at_points(args.ma, gdf)

    in_ma = gdf["ma_dtw_m"].notna()
    diag["wells_total"] = len(gdf)
    diag["wells_in_ma_raster"] = int(in_ma.sum())
    g_in = gdf[in_ma].copy()
    g_in["residual_m"] = g_in["ma_dtw_m"] - g_in["dtw_m"]

    obs, pred = g_in["dtw_m"].to_numpy(), g_in["ma_dtw_m"].to_numpy()
    results = {"all_nwis": metrics(obs, pred)}
    for bucket in ("unconfined", "confined", "mixed", "unknown"):
        m = (g_in["confinement"] == bucket).to_numpy()
        results[bucket] = metrics(obs[m], pred[m])
    # The honest water-table headline excludes confined/mixed/unknown.
    results["water_table_only"] = results["unconfined"]

    g_in.drop(columns=["geometry"]).to_csv(
        out_dir / f"ma_vs_nwis_{st}_wells.csv", index=False
    )
    g_in.to_file(out_dir / f"ma_vs_nwis_{st}_wells.fgb", driver="FlatGeobuf")
    report = {
        "state": st,
        "ma_raster": args.ma,
        "static_only": args.static_only,
        "diagnostics": diag,
        "metrics": results,
    }
    with open(out_dir / f"ma_vs_nwis_{st}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"diagnostics": diag, "metrics": results}, indent=2))


if __name__ == "__main__":
    main()
