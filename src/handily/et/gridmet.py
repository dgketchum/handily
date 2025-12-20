import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm

from handily.et.thredds import GridMet

CLIMATE_COLS = {
    "etr": {"col": "etr"},
    "pet": {"col": "eto"},
    "pr": {"col": "prcp"},
    "srad": {"col": "srad"},
    "tmmx": {"col": "tmax"},
    "tmmn": {"col": "tmin"},
    "vs": {"col": "u2"},  # likely not 2m wind; naming preserved from swim-rs
    "sph": {"col": "q"},
}


def air_pressure(elev, method="asce"):
    pair = np.array(elev, copy=True, ndmin=1).astype(np.float64)
    pair *= -0.0065
    if method == "asce":
        pair += 293
        pair /= 293
        np.power(pair, 5.26, out=pair)
    elif method == "refet":
        pair += 293
        pair /= 293
        np.power(pair, 9.8 / (0.0065 * 286.9), out=pair)
    pair *= 101.3
    return pair


def actual_vapor_pressure(q, pair):
    ea = np.array(q, copy=True, ndmin=1).astype(np.float64)
    ea *= 0.378
    ea += 0.622
    np.reciprocal(ea, out=ea)
    ea *= pair
    ea *= q
    return ea


def download_gridmet(
    fields,
    gridmet_parquet_dir,
    gridmet_centroids_path=None,
    gridmet_centroid_parquet_dir=None,
    bounds_wsen=None,
    start=None,
    end=None,
    overwrite=False,
    feature_id="FID",
    gridmet_id_col="GFID",
    return_df=False,
):
    if start is None:
        start = "1987-01-01"
    if end is None:
        end = "2021-12-31"

    df = gpd.read_file(fields)
    df.index = df[feature_id]
    df.index.name = None

    if bounds_wsen is not None:
        w, s, e, n = bounds_wsen
        aoi = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs="EPSG:4326")
        aoi = aoi.to_crs(df.crs)
        minx, miny, maxx, maxy = aoi.total_bounds
        df = df.cx[minx:maxx, miny:maxy]

    if gridmet_centroids_path is not None:
        join_crs = "EPSG:5071"
        df_join = df.to_crs(join_crs)
        cent = df_join.copy()
        cent["geometry"] = cent.geometry.centroid
        pts = gpd.read_file(gridmet_centroids_path)
        pts = pts.to_crs(join_crs)
        if bounds_wsen is not None:
            aoi_pts = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs="EPSG:4326").to_crs(join_crs)
            aoi_pts["geometry"] = aoi_pts.geometry.buffer(4000)
            minx, miny, maxx, maxy = aoi_pts.total_bounds
            pts = pts.cx[minx:maxx, miny:maxy]
        joined = gpd.sjoin_nearest(
            cent[[feature_id, "geometry"]],
            pts,
            how="left",
            distance_col="dist",
        )
        gfid_map = dict(zip(joined[feature_id].values, joined[gridmet_id_col].values))
        df[gridmet_id_col] = df[feature_id].map(gfid_map)

        pts_ll = pts.to_crs("EPSG:4326")
        pts_ll[gridmet_id_col] = pts_ll[gridmet_id_col].astype(int)
        pts_ll = pts_ll.set_index(gridmet_id_col)
        df[gridmet_id_col] = df[gridmet_id_col].astype(int)
    else:
        cent = df.copy()
        cent["geometry"] = cent.geometry.centroid
        wgs84 = cent.to_crs("EPSG:4326")
        df["centroid_lat"] = wgs84.geometry.y.values
        df["centroid_lon"] = wgs84.geometry.x.values

    os.makedirs(gridmet_parquet_dir, exist_ok=True)
    if gridmet_centroid_parquet_dir is not None:
        os.makedirs(gridmet_centroid_parquet_dir, exist_ok=True)

    centroid_cache = {}

    if gridmet_centroids_path is not None:
        gfids = [int(i) for i in pd.unique(df[gridmet_id_col].values)]
        for gfid in tqdm(gfids, desc="Downloading GridMET per centroid", total=len(gfids)):
            centroid_file = None
            if gridmet_centroid_parquet_dir is not None:
                centroid_file = os.path.join(gridmet_centroid_parquet_dir, f"{gfid}.parquet")

            if centroid_file is not None and os.path.exists(centroid_file) and not overwrite:
                centroid_cache[gfid] = centroid_file
                continue

            lat = float(pts_ll.loc[gfid].geometry.y)
            lon = float(pts_ll.loc[gfid].geometry.x)

            out = pd.DataFrame()
            first = True

            for thredds_var, meta in CLIMATE_COLS.items():
                if thredds_var not in ['pet', 'pr']:
                    continue
                variable = meta["col"]
                g = GridMet(thredds_var, start=start, end=end, lat=lat, lon=lon)
                s = g.get_point_timeseries()
                out[variable] = s[thredds_var]

                if first:
                    out["date"] = [i.strftime("%Y-%m-%d") for i in out.index]
                    out["year"] = [i.year for i in out.index]
                    out["month"] = [i.month for i in out.index]
                    out["day"] = [i.day for i in out.index]
                    out["centroid_lat"] = [lat for _ in range(out.shape[0])]
                    out["centroid_lon"] = [lon for _ in range(out.shape[0])]
                    elev_g = GridMet("elev", lat=lat, lon=lon)
                    elev = elev_g.get_point_elevation()
                    out["elev"] = [elev for _ in range(out.shape[0])]
                    first = False

            if centroid_file is not None:
                out.to_parquet(centroid_file)
                centroid_cache[gfid] = centroid_file

        for fid, row in tqdm(df.iterrows(), desc="Writing field GridMET from centroid cache", total=df.shape[0]):
            out_file = os.path.join(gridmet_parquet_dir, f"{fid}.parquet")
            if os.path.exists(out_file) and not overwrite:
                continue
            gfid = int(row[gridmet_id_col])
            centroid_file = centroid_cache[gfid]
            met = pd.read_parquet(centroid_file)
            met[feature_id] = fid
            met.to_parquet(out_file)
            if return_df:
                return met

    else:
        for fid, row in tqdm(df.iterrows(), desc="Downloading GridMET per field centroid", total=df.shape[0]):
            out_file = os.path.join(gridmet_parquet_dir, f"{fid}.parquet")
            if os.path.exists(out_file) and not overwrite:
                continue

            lat = float(row["centroid_lat"])
            lon = float(row["centroid_lon"])

            out = pd.DataFrame()
            first = True

            for thredds_var, meta in CLIMATE_COLS.items():
                variable = meta["col"]
                g = GridMet(thredds_var, start=start, end=end, lat=lat, lon=lon)
                s = g.get_point_timeseries()
                out[variable] = s[thredds_var]

                if first:
                    out["date"] = [i.strftime("%Y-%m-%d") for i in out.index]
                    out["year"] = [i.year for i in out.index]
                    out["month"] = [i.month for i in out.index]
                    out["day"] = [i.day for i in out.index]
                    out["centroid_lat"] = [lat for _ in range(out.shape[0])]
                    out["centroid_lon"] = [lon for _ in range(out.shape[0])]
                    elev_g = GridMet("elev", lat=lat, lon=lon)
                    elev = elev_g.get_point_elevation()
                    out["elev"] = [elev for _ in range(out.shape[0])]
                    first = False

            p_air = air_pressure(out["elev"])
            ea_kpa = actual_vapor_pressure(out["q"], p_air)
            out["ea"] = ea_kpa.copy()

            out["tmax"] = out["tmax"] - 273.15
            out["tmin"] = out["tmin"] - 273.15

            out.to_parquet(out_file)
            if return_df:
                return out

    result = None
    return result


# ========================= EOF =======================================================================================
