import os
import glob
import geopandas as gpd
import ee


def _ee_geom(geom):
    gj = gpd.GeoSeries([geom], crs="EPSG:4326").__geo_interface__["features"][0][
        "geometry"
    ]
    g = ee.Geometry(gj)
    return g


def _naip_ndwi_image(region_geom, start_date, end_date):
    """
    Build a single NDWI image for a region by taking the per-pixel maximum NDWI value
    across all NAIP images in the date range.

    This mirrors the Earth Engine JS workflow:
      ndwiCollection = NAIP.map(addNDWI).select('ndwi')
      maxNdwi = ndwiCollection.max()
    """
    col = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(region_geom)
        .filterDate(start_date, end_date)
    )

    def _add_ndwi(img):
        return img.normalizedDifference(["G", "N"]).rename("ndwi")

    ndwi = col.map(_add_ndwi).select("ndwi").max()
    return ndwi.clip(region_geom)


def export_ndwi_for_polygons(
    aoi_shapefile,
    bucket="wudr",
    prefix=None,
    start_date="2010-01-01",
    end_date="2024-12-31",
    skip_if_present_dir=None,
    coverage_col=None,
):
    """Submit NAIP NDWI export tasks to Earth Engine for each AOI polygon.

    Parameters
    ----------
    aoi_shapefile : str
        Path to AOI shapefile produced by ``handily aoi``.
    bucket : str
        GCS bucket name.
    prefix : str
        GCS path prefix for exported files (e.g. ``handily/mt/ndwi/naip_ndwi_aoi``).
    start_date, end_date : str
        NAIP date range for the max-NDWI composite.
    skip_if_present_dir : str, optional
        Local directory; AOIs whose output .tif already exists there are skipped.
    coverage_col : str, optional
        If set, only rows where this column == 1 are exported (e.g. ``stac_1m``).
    """
    shp_path = os.path.expanduser(aoi_shapefile)
    aoi_gdf = gpd.read_file(shp_path)

    if coverage_col and coverage_col in aoi_gdf.columns:
        n_before = len(aoi_gdf)
        aoi_gdf = aoi_gdf[aoi_gdf[coverage_col] == 1].reset_index(drop=True)
        print(f"Filtered to {coverage_col}==1: {len(aoi_gdf)}/{n_before} AOIs")

    tasks = []
    for i, row in aoi_gdf.iterrows():
        geom = row.geometry
        region = _ee_geom(geom)
        ndwi = _naip_ndwi_image(region, start_date, end_date)
        idx = int(row["aoi_id"]) if "aoi_id" in aoi_gdf.columns else int(i)
        fname = f"{prefix}/naip_ndwi_aoi_{idx:04d}"
        if skip_if_present_dir:
            present = glob.glob(
                os.path.join(skip_if_present_dir, f"{os.path.basename(fname)}*.tif")
            )
            if present:
                continue
        task = ee.batch.Export.image.toCloudStorage(
            image=ndwi,
            bucket=bucket,
            fileNamePrefix=fname,
            description=f"naip_ndwi_aoi_{idx:04d}",
            region=region,
            scale=1,
            maxPixels=1e13,
        )
        task.start()
        tasks.append(task)
        print(fname)
    return tasks


if __name__ == "__main__":
    ee.Initialize()
    shp_path = "/nas/handily/outputs/testing/ndwi_aois.shp"
    ndwi_dir = os.path.expanduser("/nas/surface_water/NDWI/")
    tasks = export_ndwi_for_polygons(
        aoi_shapefile=shp_path,
        bucket="wudr",
        prefix="naip_ndwi_aoi",
        start_date="2010-01-01",
        end_date="2024-12-31",
        skip_if_present_dir=ndwi_dir,
    )
# ========================= EOF ====================================================================
