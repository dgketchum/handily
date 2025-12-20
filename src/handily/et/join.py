import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def join_gridmet_ptjpl(
    gridmet_parquet_dir,
    ptjpl_csv_dir,
    out_parquet_dir,
    ptjpl_csv_template=None,
    fields_path=None,
    bounds_wsen=None,
    feature_id="FID",
    eto_col="eto",
    prcp_col="prcp",
):
    os.makedirs(out_parquet_dir, exist_ok=True)

    if ptjpl_csv_template is None:
        ptjpl_csv_template = os.path.join(ptjpl_csv_dir, "{fid}.csv")

    select = None
    if fields_path is not None and bounds_wsen is not None:
        w, s, e, n = bounds_wsen
        fields = gpd.read_file(fields_path)
        aoi = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs="EPSG:4326").to_crs(fields.crs)
        minx, miny, maxx, maxy = aoi.total_bounds
        fields = fields.cx[minx:maxx, miny:maxy]
        select = set(fields[feature_id].astype(str).tolist())

    for fn in os.listdir(gridmet_parquet_dir):
        if not fn.endswith(".parquet"):
            continue

        fid = os.path.splitext(fn)[0]
        if select is not None and str(fid) not in select:
            continue

        met = pd.read_parquet(os.path.join(gridmet_parquet_dir, fn))
        if "date" in met.columns:
            met.index = pd.to_datetime(met["date"])

        pt = pd.read_csv(ptjpl_csv_template.format(fid=fid))
        if "date" in pt.columns:
            pt["date"] = pd.to_datetime(pt["date"])
        pt = pt.sort_values("date")
        pt = pt[[feature_id, "date", "et_fraction"]]

        met["ptjpl_etf"] = pd.Series(index=met.index, dtype="float64")
        obs = pd.Series(pt["et_fraction"].values, index=pt["date"].values)
        met.loc[obs.index, "ptjpl_etf"] = obs.values

        met["ptjpl_etf_interp"] = met["ptjpl_etf"].interpolate(method="time")
        met["aet"] = met["ptjpl_etf_interp"] * met[eto_col]

        out = met
        out.to_parquet(os.path.join(out_parquet_dir, f"{fid}.parquet"))

    result = None
    return result


# ========================= EOF =======================================================================================
