import os
import glob
import geopandas as gpd
from shapely.geometry import box as shapely_box
import ee


def _tiles_for_bounds(bounds_wsen, mgrs_shp_path):
    w, s, e, n = bounds_wsen
    aoi_ll = gpd.GeoDataFrame([{}], geometry=[shapely_box(w, s, e, n)], crs="EPSG:4326")
    mgrs = gpd.read_file(os.path.expanduser(mgrs_shp_path))
    mgrs_aea = mgrs.to_crs("EPSG:5070")
    aoi_aea = aoi_ll.to_crs("EPSG:5070")
    hits = mgrs_aea[mgrs_aea.intersects(aoi_aea.unary_union)].copy()
    name_col = "MGRS_TILE"
    tiles = list(hits[name_col].astype(str).unique())
    hits_ll = hits.to_crs("EPSG:4326")
    return tiles, hits_ll


def _ee_geom(geom):
    gj = gpd.GeoSeries([geom], crs="EPSG:4326").__geo_interface__["features"][0]["geometry"]
    g = ee.Geometry(gj)
    return g


def _naip_ndwi_image(region_geom, start_date, end_date):
    col = ee.ImageCollection('USDA/NAIP/DOQQ').filterBounds(region_geom).filterDate(start_date, end_date)
    img = col.mosaic()
    ndwi = img.normalizedDifference(['G', 'N']).rename('ndwi')
    ndwi = ndwi.clip(region_geom)
    return ndwi


def export_ndwi_for_bounds(bounds_wsen,
                           mgrs_shp_path,
                           bucket='wudr',
                           prefix='naip_ndwi',
                           start_date='2014-01-01',
                           end_date='2024-12-31',
                           skip_if_present_dir=None):
    tiles, tiles_ll = _tiles_for_bounds(bounds_wsen, mgrs_shp_path)

    skip = set()
    if skip_if_present_dir:
        for t in tiles:
            matches = glob.glob(os.path.join(os.path.expanduser(skip_if_present_dir), f"*{t}*.tif"))
            if matches:
                skip.add(t)

    tasks = []
    for i, row in tiles_ll.iterrows():
        tile = str(row['MGRS_TILE'])
        if tile in skip:
            continue
        region = _ee_geom(row.geometry)
        ndwi = _naip_ndwi_image(region, start_date, end_date)

        fname = f"{prefix}_{tile}"
        task = ee.batch.Export.image.toCloudStorage(
            image=ndwi, bucket=bucket, fileNamePrefix=fname, description=fname,
            region=region, scale=1, maxPixels=1e13)
        task.start()
        tasks.append(task)
        print(f'exported {fname}')

        # Only export raw NDWI; thresholding occurs downstream.

    return tasks


def export_ndwi_for_polygons(aoi_gdf,
                             bucket='wudr',
                             prefix='naip_ndwi_aoi',
                             start_date='2014-01-01',
                             end_date='2024-12-31',
                             skip_if_present_dir=None):
    tasks = []
    for i, row in aoi_gdf.iterrows():
        geom = row.geometry
        region = _ee_geom(geom)
        ndwi = _naip_ndwi_image(region, start_date, end_date)
        fname = f"{prefix}_{i:04d}"
        if skip_if_present_dir:
            present = glob.glob(os.path.join(os.path.expanduser(skip_if_present_dir), f"{fname}*.tif"))
            if present:
                continue
        task = ee.batch.Export.image.toCloudStorage(
            image=ndwi, bucket=bucket, fileNamePrefix=fname, description=fname,
            region=region, scale=1, maxPixels=1e13)
        task.start()
        tasks.append(task)
    return tasks


if __name__ == '__main__':
    ee.Initialize()
    shp_path = '/home/dgketchum/data/IrrigationGIS/handily/outputs/testing/ndwi_aois.shp'
    aoi = gpd.read_file(shp_path)
    ndwi_dir = os.path.expanduser('~/data/IrrigationGIS/surface_water/NDWI/')
    tasks = export_ndwi_for_polygons(
        aoi_gdf=aoi,
        bucket='wudr',
        prefix='naip_ndwi_aoi',
        start_date='2014-01-01',
        end_date='2024-12-31',
        skip_if_present_dir=ndwi_dir,
    )
# ========================= EOF ====================================================================
