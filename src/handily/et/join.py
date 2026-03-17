import calendar
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def join_gridmet_openet_eta(
    gridmet_parquet_dir,
    openet_csv_path,
    out_parquet_dir,
    fields_path=None,
    bounds_wsen=None,
    feature_id="FID",
    eto_col="eto",
    prcp_col="prcp",
):
    os.makedirs(out_parquet_dir, exist_ok=True)

    openet = pd.read_csv(openet_csv_path)
    openet[feature_id] = openet[feature_id].astype(str)
    openet = openet.set_index(feature_id)

    select = None
    if fields_path is not None and bounds_wsen is not None:
        w, s, e, n = bounds_wsen
        fields = gpd.read_file(fields_path)
        aoi = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs="EPSG:4326").to_crs(
            fields.crs
        )
        minx, miny, maxx, maxy = aoi.total_bounds
        fields = fields.cx[minx:maxx, miny:maxy]
        select = set(fields[feature_id].astype(str).tolist())

    for fn in os.listdir(gridmet_parquet_dir):
        if not fn.endswith(".parquet"):
            continue

        fid = os.path.splitext(fn)[0]
        if select is not None and str(fid) not in select:
            continue
        if str(fid) not in openet.index:
            continue

        met = pd.read_parquet(os.path.join(gridmet_parquet_dir, fn))
        if "date" in met.columns:
            met.index = pd.to_datetime(met["date"])

        row = openet.loc[str(fid)]
        eta_cols = [c for c in row.index if c.startswith("eta_")]

        daily_aet = pd.Series(index=met.index, dtype="float64")
        for col in eta_cols:
            parts = col.split("_")
            year, month = int(parts[1]), int(parts[2])
            days = calendar.monthrange(year, month)[1]
            daily_val = row[col] / days
            mask = (met.index.year == year) & (met.index.month == month)
            daily_aet.loc[mask] = daily_val

        met["aet"] = daily_aet
        met.to_parquet(os.path.join(out_parquet_dir, f"{fid}.parquet"))


# ========================= EOF =======================================================================================
