import os
import subprocess

import ee
import geopandas as gpd
from openet import ptjpl
from tqdm import tqdm

LANDSAT_COLLECTIONS = [
    "LANDSAT/LT04/C02/T1_L2",
    "LANDSAT/LT05/C02/T1_L2",
    "LANDSAT/LE07/C02/T1_L2",
    "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT/LC09/C02/T1_L2",
]


def list_gcs_bucket_contents(gcs_path: str, gsutil: str = "gsutil") -> list[str]:
    command = [gsutil, "ls", gcs_path]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    lines = []
    if result.returncode == 0 and result.stdout:
        lines = [line for line in result.stdout.strip().split("\n") if line]
    return lines


def export_ptjpl_et_fraction(
    shapefile,
    bucket,
    feature_id="FID",
    select=None,
    start_yr=2000,
    end_yr=2024,
    overwrite=False,
    check_dir=None,
    buffer=None,
    bounds_wsen=None,
    cloud_cover_max=70,
    landsat_collections=None,
):
    ee.Initialize()
    df = gpd.read_file(shapefile)
    df = df.set_index(feature_id, drop=False)
    df = df.sort_index(ascending=False)

    if buffer is not None:
        df.geometry = df.geometry.buffer(buffer)

    original_crs = df.crs
    if original_crs and original_crs.srs != "EPSG:4326":
        df = df.to_crs(4326)
    if bounds_wsen is not None:
        w, s, e, n = bounds_wsen
        df = df.cx[w:e, s:n]

    collections = landsat_collections
    if collections is None:
        collections = LANDSAT_COLLECTIONS

    for fid, row in tqdm(df.iterrows(), desc="Export PT-JPL capture-date zonal stats", total=df.shape[0]):
        if select is not None and fid not in select:
            continue

        if row["geometry"].geom_type != "Polygon":
            raise ValueError

        polygon = ee.Geometry(row.geometry.__geo_interface__)

        desc = f"ptjpl_etf_zonal_{fid}_{start_yr}_{end_yr}"
        fn_prefix = os.path.join("ptjpl_tables", "etf_zonal", str(fid), desc)

        if not overwrite:
            dst = os.path.join(f"gs://{bucket}", "ptjpl_tables", "etf_zonal", str(fid), f"{desc}.csv")
            existing = list_gcs_bucket_contents(dst)
            if existing:
                continue

        if check_dir is not None:
            target_file = os.path.join(check_dir, "ptjpl_tables", "etf_zonal", str(fid), f"{desc}.csv")
            if os.path.isfile(target_file) and not overwrite:
                continue

        coll = ptjpl.Collection(
            collections,
            start_date=f"{start_yr}-01-01",
            end_date=f"{end_yr}-12-31",
            geometry=polygon,
            cloud_cover_max=cloud_cover_max,
            model_args={
                "et_reference_source": "IDAHO_EPSCOR/GRIDMET",
                "et_reference_band": "eto",
                "et_reference_factor": 1.0,
                "et_reference_resample": "bilinear",
            },
        )

        # Use server-side mapping to avoid client-side payload size limits
        img_coll = coll.overpass(variables=["et_fraction"])

        def compute_zonal_stats(img):
            """Server-side function to compute zonal stats for each image."""
            etf_img = img.select("et_fraction").clip(polygon)
            time_start = img.get("system:time_start")
            date = ee.Date(time_start).format("YYYY-MM-dd")
            img_id = img.get("system:id")

            stats = etf_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=polygon,
                scale=30,
                maxPixels=1e13,
            )
            return ee.Feature(None, stats).set(feature_id, fid).set("date", date).set("img_id", img_id)

        fc = ee.FeatureCollection(img_coll.map(compute_zonal_stats))
        task = ee.batch.Export.table.toCloudStorage(
            collection=fc,
            description=desc,
            bucket=bucket,
            fileNamePrefix=fn_prefix,
            fileFormat="CSV",
            selectors=[feature_id, "date", "img_id", "et_fraction"],
        )
        task.start()


# ========================= EOF =======================================================================================
