import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


def _water_year_index(dt_index):
    dt = pd.to_datetime(dt_index)
    wy = dt.year + (dt.month >= 10).astype(int)
    return wy


def _annual_sums(daily):
    idx = pd.to_datetime(daily.index)
    wy = _water_year_index(idx)

    out = pd.DataFrame(index=pd.Index(sorted(pd.unique(wy)), name="water_year"))
    out["aet"] = daily["aet"].groupby(wy).sum()
    out["eto"] = daily["eto"].groupby(wy).sum()
    out["prcp"] = daily["prcp"].groupby(wy).sum()
    out["pe"] = np.minimum(out["prcp"].values, out["aet"].values)
    out["et_gwsm"] = out["aet"].values - out["pe"].values
    out["etf_gwsm"] = out["et_gwsm"].values / out["eto"].values
    return out


def _monthly_disaggregation(daily):
    net = daily["aet"] - daily["prcp"]
    net_m = net.resample("MS").sum()
    net_m = net_m.clip(lower=0)

    wy_m = _water_year_index(net_m.index)
    net_a = net.groupby(_water_year_index(daily.index)).sum()
    net_a = net_a.clip(lower=0)

    net_a_aligned = pd.Series([net_a[int(w)] for w in wy_m], index=net_m.index)
    fdisag = net_m / net_a_aligned
    fdisag = fdisag.fillna(0.0)
    fdisag = fdisag.replace([np.inf, -np.inf], 0.0)
    return fdisag


def partition_et(
    fields_path,
    joined_parquet_dir,
    out_parquet_dir,
    feature_id="FID",
    strata_col="strata",
    pattern_col="pattern",
    bounds_wsen=None,
):
    fields = gpd.read_file(fields_path)
    fields = fields[[feature_id, strata_col, pattern_col, "geometry"]]
    if bounds_wsen is not None:
        w, s, e, n = bounds_wsen
        aoi = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs="EPSG:4326").to_crs(fields.crs)
        minx, miny, maxx, maxy = aoi.total_bounds
        fields = fields.cx[minx:maxx, miny:maxy]
    fields_cent = fields.copy()
    fields_cent["geometry"] = fields_cent.geometry.centroid
    fields_cent = fields_cent.to_crs("EPSG:5071")

    donors = fields_cent[fields_cent[pattern_col].astype(bool)].copy()
    recipients = fields_cent.copy()

    donor_map = {}
    for strata in pd.unique(recipients[strata_col].values):
        r = recipients[recipients[strata_col] == strata][[feature_id, "geometry"]]
        d = donors[donors[strata_col] == strata][[feature_id, "geometry"]]
        if d.empty:
            raise ValueError
        j = gpd.sjoin_nearest(r, d, how="left", distance_col="dist", lsuffix="r", rsuffix="d")
        for _, row in j.iterrows():
            donor_map[row[f"{feature_id}_r"]] = row[f"{feature_id}_d"]

    os.makedirs(out_parquet_dir, exist_ok=True)

    donor_etf = {}
    for donor_fid in pd.unique(list(donor_map.values())):
        df = pd.read_parquet(os.path.join(joined_parquet_dir, f"{donor_fid}.parquet"))
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
        annual = _annual_sums(df)
        donor_etf[donor_fid] = annual["etf_gwsm"]

    for fid, donor_fid in donor_map.items():
        df = pd.read_parquet(os.path.join(joined_parquet_dir, f"{fid}.parquet"))
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])

        annual = _annual_sums(df)
        etf = donor_etf[donor_fid].reindex(annual.index)

        annual["et_gwsm"] = etf.values * annual["eto"].values
        annual["et_irr"] = annual["aet"].values - annual["et_gwsm"].values - annual["pe"].values

        neg = annual["et_irr"].values < 0
        annual.loc[neg, "et_irr"] = 0.0
        annual.loc[neg, "et_gwsm"] = annual.loc[neg, "aet"].values - annual.loc[neg, "pe"].values
        annual["etf_gwsm_used"] = annual["et_gwsm"].values / annual["eto"].values

        fdisag = _monthly_disaggregation(df)
        wy_m = _water_year_index(fdisag.index)
        et_gwsm_a = pd.Series([annual.loc[int(w), "et_gwsm"] for w in wy_m], index=fdisag.index)
        et_irr_a = pd.Series([annual.loc[int(w), "et_irr"] for w in wy_m], index=fdisag.index)

        out = pd.DataFrame(index=fdisag.index)
        out["water_year"] = wy_m
        out[feature_id] = fid
        out["donor_fid"] = donor_fid
        out["fdisag"] = fdisag.values
        out["et_gwsm_m"] = fdisag.values * et_gwsm_a.values
        out["et_irr_m"] = fdisag.values * et_irr_a.values

        out.to_parquet(os.path.join(out_parquet_dir, f"{fid}.parquet"))

    result = None
    return result


# ========================= EOF =======================================================================================
