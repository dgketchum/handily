from __future__ import annotations

import logging
import os
import re
from typing import Iterable

import ee
import geopandas as gpd
import numpy as np

from handily.config import HandilyConfig
from handily.ee.common import export_table, gdf_to_ee_feature_collection, initialize_ee

LOGGER = logging.getLogger("handily.points.ee_extract")

IRRMAPPER_ASSET = "projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp"
DEFAULT_IRRMAPPER_START_YEAR = 1987
DEFAULT_IRRMAPPER_END_YEAR = 2024

OPENET_ETA_V2 = "projects/openet/assets/ensemble/conus/gridmet/monthly/v2_0"
OPENET_ETA_PRE2000 = (
    "projects/openet/assets/ensemble/conus/gridmet/monthly/v2_0_pre2000"
)
OPENET_ETA_SPLIT_YEAR = 1999
OPENET_ETA_BAND = "et_ensemble_mad"

GRIDMET_COLLECTION = "IDAHO_EPSCOR/GRIDMET"
GRIDMET_ETO_BAND = "eto"
GRIDMET_PR_BAND = "pr"

LANDSAT_COLLECTIONS = [
    ("LANDSAT/LT04/C02/T1_L2", "SR_B3", "SR_B4"),
    ("LANDSAT/LT05/C02/T1_L2", "SR_B3", "SR_B4"),
    ("LANDSAT/LE07/C02/T1_L2", "SR_B3", "SR_B4"),
    ("LANDSAT/LC08/C02/T1_L2", "SR_B4", "SR_B5"),
    ("LANDSAT/LC09/C02/T1_L2", "SR_B4", "SR_B5"),
]

POINT_EXPORT_PROPS = [
    "point_id",
    "aoi_id",
    "sample_group",
    "sample_seed",
    "in_irrigated_lands",
    "rem_at_sample",
    "stream_context_at_sample",
    "nearest_stream_type",
    "stream_distance",
    "dist_field_edge_m",
]


def _points_out_dir(config: HandilyConfig) -> str:
    if config.points_out_dir:
        return config.points_out_dir
    return os.path.join(config.out_dir, "points")


def _points_path(config: HandilyConfig) -> str:
    out_dir = _points_out_dir(config)
    parquet_path = os.path.join(out_dir, "points.parquet")
    if os.path.exists(parquet_path):
        return parquet_path
    fgb_path = os.path.join(out_dir, "points.fgb")
    if os.path.exists(fgb_path):
        return fgb_path
    raise FileNotFoundError(f"Sample points not found under {out_dir}")


def load_points(
    points: str | gpd.GeoDataFrame | None, config: HandilyConfig
) -> gpd.GeoDataFrame:
    if isinstance(points, gpd.GeoDataFrame):
        return points
    if isinstance(points, str):
        path = os.path.expanduser(points)
    else:
        path = _points_path(config)

    if path.endswith(".parquet"):
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    return gdf


def _validate_export_properties(points_gdf: gpd.GeoDataFrame, props: list[str]) -> None:
    invalid: list[str] = []
    details: list[str] = []
    for prop in props:
        if prop not in points_gdf.columns:
            continue
        series = points_gdf[prop]
        null_count = int(series.isna().sum())
        inf_count = 0
        if np.issubdtype(series.dtype, np.number):
            values = series.to_numpy()
            inf_count = int(np.isinf(values).sum())
        if null_count or inf_count:
            invalid.append(prop)
            details.append(f"{prop}: null={null_count}, inf={inf_count}")

    if invalid:
        raise ValueError(
            "Point export properties contain invalid values. "
            "Regenerate the points dataset or adjust exported properties. "
            + "; ".join(details)
        )


def points_to_ee_feature_collection(
    points: str | gpd.GeoDataFrame | None,
    config: HandilyConfig,
    keep_props: list[str] | None = None,
) -> tuple[gpd.GeoDataFrame, ee.FeatureCollection]:
    gdf = load_points(points, config)
    props = keep_props or POINT_EXPORT_PROPS
    keep = [prop for prop in props if prop in gdf.columns]
    _validate_export_properties(gdf, keep)
    return gdf, gdf_to_ee_feature_collection(gdf, keep_props=keep)


def _sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")


def _resolve_aoi_id(
    points_gdf: gpd.GeoDataFrame, config: HandilyConfig, aoi_id: str | None = None
) -> str:
    if aoi_id:
        return _sanitize_token(aoi_id)
    if "aoi_id" in points_gdf.columns:
        unique = [str(v) for v in points_gdf["aoi_id"].dropna().unique()]
        if len(unique) == 1:
            return _sanitize_token(unique[0])
    return _sanitize_token(config.project_name)


def _resolve_year_range(
    config: HandilyConfig,
    year_start: int | None,
    year_end: int | None,
    default_start: int,
    default_end: int,
) -> tuple[int, int]:
    start = year_start or config.points_year_start or default_start
    end = year_end or config.points_year_end or default_end
    if start > end:
        raise ValueError(f"Invalid year range: {start} > {end}")
    return int(start), int(end)


def _resolve_export_dest(
    config: HandilyConfig,
    dest: str | None,
    bucket: str | None,
    drive_folder: str | None,
) -> tuple[str, str | None, str | None]:
    resolved_dest = dest or config.points_ee_dest
    resolved_bucket = bucket or config.points_ee_bucket or config.et_bucket
    resolved_drive_folder = drive_folder or config.points_ee_drive_folder
    if resolved_dest == "bucket" and not resolved_bucket:
        raise ValueError(
            "Bucket export requires points_ee_bucket or et_bucket in config"
        )
    if resolved_dest == "drive" and not resolved_drive_folder:
        raise ValueError("Drive export requires points_ee_drive_folder in config")
    return resolved_dest, resolved_bucket, resolved_drive_folder


def build_points_export_prefix(
    config: HandilyConfig,
    product: str,
    aoi_id: str,
    year_start: int,
    year_end: int,
) -> tuple[str, str]:
    filename = (
        f"{config.project_name}_{aoi_id}_{product}_points_{year_start}_{year_end}"
    )
    filename = _sanitize_token(filename)
    prefix = config.get_bucket_path(f"points/{product}", filename)
    description = _sanitize_token(filename)[:100]
    return prefix, description


def _sample_points_image(
    image: ee.Image,
    points_fc: ee.FeatureCollection,
    selectors: list[str],
    scale: int = 30,
) -> ee.FeatureCollection:
    return image.sampleRegions(
        collection=points_fc,
        properties=selectors,
        scale=scale,
        geometries=False,
        tileScale=8,
    )


def export_points_irrmapper(
    points: str | gpd.GeoDataFrame | None,
    config: HandilyConfig,
    aoi_id: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    dest: str | None = None,
    bucket: str | None = None,
    drive_folder: str | None = None,
    ee_project: str | None = None,
) -> dict[str, str]:
    initialize_ee(ee_project or config.ee_project)
    points_gdf, points_fc = points_to_ee_feature_collection(points, config)
    selectors = [prop for prop in POINT_EXPORT_PROPS if prop in points_gdf.columns]
    aoi_name = _resolve_aoi_id(points_gdf, config, aoi_id=aoi_id)
    start_year, end_year = _resolve_year_range(
        config,
        year_start,
        year_end,
        default_start=DEFAULT_IRRMAPPER_START_YEAR,
        default_end=DEFAULT_IRRMAPPER_END_YEAR,
    )
    resolved_dest, resolved_bucket, resolved_drive_folder = _resolve_export_dest(
        config,
        dest,
        bucket,
        drive_folder,
    )

    irr_coll = ee.ImageCollection(IRRMAPPER_ASSET)
    stacked = None
    band_names: list[str] = []
    for year in range(start_year, end_year + 1):
        band_name = f"irr_{year}"
        irr = (
            irr_coll.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select("classification")
            .mosaic()
        )
        band = irr.lt(1).rename(band_name).toFloat()
        stacked = band if stacked is None else stacked.addBands(band)
        band_names.append(band_name)

    if stacked is None:
        raise ValueError("No IrrMapper bands were created")

    samples = _sample_points_image(stacked, points_fc, selectors=selectors)
    prefix, description = build_points_export_prefix(
        config, "irrmapper", aoi_name, start_year, end_year
    )
    export_table(
        samples,
        desc=description,
        dest=resolved_dest,
        selectors=selectors + band_names,
        bucket=resolved_bucket,
        file_prefix=prefix if resolved_dest == "bucket" else os.path.basename(prefix),
        drive_folder=resolved_drive_folder,
    )
    return {"product": "irrmapper", "description": description, "prefix": prefix}


def _mask_landsat_sr(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 1)
        .eq(0)
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    return image.updateMask(mask)


def _landsat_ndvi_collection(
    geometry: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
) -> ee.ImageCollection:
    collections: list[ee.ImageCollection] = []
    for collection_id, red_band, nir_band in LANDSAT_COLLECTIONS:
        coll = (
            ee.ImageCollection(collection_id)
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .map(_mask_landsat_sr)
            .map(
                lambda img, red=red_band, nir=nir_band: (
                    img.select([red, nir])
                    .multiply(0.0000275)
                    .add(-0.2)
                    .rename(["red", "nir"])
                    .normalizedDifference(["nir", "red"])
                    .rename("ndvi")
                    .copyProperties(img, ["system:time_start"])
                )
            )
        )
        collections.append(coll)

    merged = collections[0]
    for coll in collections[1:]:
        merged = merged.merge(coll)
    return merged


def _add_yearly_ndvi_bands(
    years: Iterable[int],
    geometry: ee.Geometry,
    start_month: int,
    end_month: int,
) -> tuple[ee.Image, list[str]]:
    stacked = None
    band_names: list[str] = []
    for year in years:
        start_date = ee.Date.fromYMD(int(year), int(start_month), 1)
        end_date = ee.Date.fromYMD(int(year), int(end_month), 1).advance(1, "month")
        coll = _landsat_ndvi_collection(geometry, start_date, end_date)
        ndvi = coll.select("ndvi")
        stats = ee.Image.cat(
            ndvi.mean().rename(f"ndvi_mean_{year}"),
            ndvi.max().rename(f"ndvi_peak_{year}"),
            ndvi.reduce(ee.Reducer.percentile([25])).rename([f"ndvi_p25_{year}"]),
            ndvi.reduce(ee.Reducer.percentile([75])).rename([f"ndvi_p75_{year}"]),
            ndvi.count().rename(f"ndvi_n_obs_{year}"),
        )
        stacked = stats if stacked is None else stacked.addBands(stats)
        band_names.extend(
            [
                f"ndvi_mean_{year}",
                f"ndvi_peak_{year}",
                f"ndvi_p25_{year}",
                f"ndvi_p75_{year}",
                f"ndvi_n_obs_{year}",
            ]
        )

    if stacked is None:
        raise ValueError("No NDVI bands were created")
    return stacked, band_names


def export_points_ndvi(
    points: str | gpd.GeoDataFrame | None,
    config: HandilyConfig,
    aoi_id: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    dest: str | None = None,
    bucket: str | None = None,
    drive_folder: str | None = None,
    ee_project: str | None = None,
) -> dict[str, str]:
    initialize_ee(ee_project or config.ee_project)
    points_gdf, points_fc = points_to_ee_feature_collection(points, config)
    selectors = [prop for prop in POINT_EXPORT_PROPS if prop in points_gdf.columns]
    aoi_name = _resolve_aoi_id(points_gdf, config, aoi_id=aoi_id)
    start_year, end_year = _resolve_year_range(config, year_start, year_end, 1984, 2024)
    resolved_dest, resolved_bucket, resolved_drive_folder = _resolve_export_dest(
        config,
        dest,
        bucket,
        drive_folder,
    )

    geometry = points_fc.geometry()
    ndvi_image, band_names = _add_yearly_ndvi_bands(
        range(start_year, end_year + 1),
        geometry=geometry,
        start_month=int(config.points_ndvi_start_month),
        end_month=int(config.points_ndvi_end_month),
    )
    samples = _sample_points_image(ndvi_image, points_fc, selectors=selectors)
    prefix, description = build_points_export_prefix(
        config, "ndvi", aoi_name, start_year, end_year
    )
    export_table(
        samples,
        desc=description,
        dest=resolved_dest,
        selectors=selectors + band_names,
        bucket=resolved_bucket,
        file_prefix=prefix if resolved_dest == "bucket" else os.path.basename(prefix),
        drive_folder=resolved_drive_folder,
    )
    return {"product": "ndvi", "description": description, "prefix": prefix}


def _monthly_openet_images(year: int, geometry: ee.Geometry) -> list[ee.Image]:
    collection_id = (
        OPENET_ETA_V2 if year >= OPENET_ETA_SPLIT_YEAR else OPENET_ETA_PRE2000
    )
    raw = (
        ee.ImageCollection(collection_id)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(geometry)
        .select(OPENET_ETA_BAND)
    )

    images: list[ee.Image] = []
    for month in range(1, 13):
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        band_name = f"eta_{year}_{month:02d}"
        image = raw.filterDate(start, end).mosaic().rename(band_name)
        images.append(image)
    return images


def export_points_openet_eta(
    points: str | gpd.GeoDataFrame | None,
    config: HandilyConfig,
    aoi_id: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    dest: str | None = None,
    bucket: str | None = None,
    drive_folder: str | None = None,
    ee_project: str | None = None,
) -> dict[str, str]:
    initialize_ee(ee_project or config.ee_project)
    points_gdf, points_fc = points_to_ee_feature_collection(points, config)
    selectors = [prop for prop in POINT_EXPORT_PROPS if prop in points_gdf.columns]
    aoi_name = _resolve_aoi_id(points_gdf, config, aoi_id=aoi_id)
    start_year, end_year = _resolve_year_range(config, year_start, year_end, 1984, 2024)
    resolved_dest, resolved_bucket, resolved_drive_folder = _resolve_export_dest(
        config,
        dest,
        bucket,
        drive_folder,
    )

    geometry = points_fc.geometry()
    monthly_images: list[ee.Image] = []
    band_names: list[str] = []
    for year in range(start_year, end_year + 1):
        images = _monthly_openet_images(year, geometry)
        monthly_images.extend(images)
        band_names.extend([f"eta_{year}_{month:02d}" for month in range(1, 13)])

    if not monthly_images:
        raise ValueError("No OpenET monthly images were created")

    stacked = ee.Image(monthly_images[0])
    for image in monthly_images[1:]:
        stacked = stacked.addBands(image)
    # Unmask so sampleRegions returns all points (NoData → 0 instead of dropped)
    stacked = stacked.unmask(0)

    samples = _sample_points_image(stacked, points_fc, selectors=selectors)
    prefix, description = build_points_export_prefix(
        config, "openet_eta", aoi_name, start_year, end_year
    )
    export_table(
        samples,
        desc=description,
        dest=resolved_dest,
        selectors=selectors + band_names,
        bucket=resolved_bucket,
        file_prefix=prefix if resolved_dest == "bucket" else os.path.basename(prefix),
        drive_folder=resolved_drive_folder,
    )
    return {"product": "openet_eta", "description": description, "prefix": prefix}


def export_points_gridmet(
    points: str | gpd.GeoDataFrame | None,
    config: HandilyConfig,
    aoi_id: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    dest: str | None = None,
    bucket: str | None = None,
    drive_folder: str | None = None,
    ee_project: str | None = None,
) -> dict[str, str]:
    """Export annual and growing-season ETo and precipitation from GridMET."""
    initialize_ee(ee_project or config.ee_project)
    points_gdf, points_fc = points_to_ee_feature_collection(points, config)
    selectors = [prop for prop in POINT_EXPORT_PROPS if prop in points_gdf.columns]
    aoi_name = _resolve_aoi_id(points_gdf, config, aoi_id=aoi_id)
    start_year, end_year = _resolve_year_range(config, year_start, year_end, 1984, 2024)
    resolved_dest, resolved_bucket, resolved_drive_folder = _resolve_export_dest(
        config,
        dest,
        bucket,
        drive_folder,
    )

    geometry = points_fc.geometry()
    gs_start = config.points_ndvi_start_month  # default 4 (April)
    gs_end = config.points_ndvi_end_month  # default 10 (October)

    images: list[ee.Image] = []
    band_names: list[str] = []

    for year in range(start_year, end_year + 1):
        col = (
            ee.ImageCollection(GRIDMET_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filterBounds(geometry)
        )
        # Annual sums
        eto_annual = col.select(GRIDMET_ETO_BAND).sum().rename(f"eto_{year}")
        pr_annual = col.select(GRIDMET_PR_BAND).sum().rename(f"pr_{year}")

        # Growing-season sums
        gs_col = col.filterDate(
            ee.Date.fromYMD(year, gs_start, 1),
            ee.Date.fromYMD(year, gs_end + 1, 1),
        )
        eto_gs = gs_col.select(GRIDMET_ETO_BAND).sum().rename(f"eto_gs_{year}")
        pr_gs = gs_col.select(GRIDMET_PR_BAND).sum().rename(f"pr_gs_{year}")

        images.extend([eto_annual, pr_annual, eto_gs, pr_gs])
        band_names.extend(
            [
                f"eto_{year}",
                f"pr_{year}",
                f"eto_gs_{year}",
                f"pr_gs_{year}",
            ]
        )

    stacked = ee.Image(images[0])
    for image in images[1:]:
        stacked = stacked.addBands(image)

    samples = _sample_points_image(stacked, points_fc, selectors=selectors, scale=4000)
    prefix, description = build_points_export_prefix(
        config,
        "gridmet",
        aoi_name,
        start_year,
        end_year,
    )
    export_table(
        samples,
        desc=description,
        dest=resolved_dest,
        selectors=selectors + band_names,
        bucket=resolved_bucket,
        file_prefix=prefix if resolved_dest == "bucket" else os.path.basename(prefix),
        drive_folder=resolved_drive_folder,
    )
    return {"product": "gridmet", "description": description, "prefix": prefix}


def export_fields_openet_eta(
    fields: str | gpd.GeoDataFrame,
    config: HandilyConfig,
    year_start: int | None = None,
    year_end: int | None = None,
    dest: str | None = None,
    bucket: str | None = None,
    drive_folder: str | None = None,
    ee_project: str | None = None,
    feature_id: str = "FID",
) -> dict[str, str]:
    """Export monthly OpenET v2.0 ensemble ET zonal means for field polygons."""
    initialize_ee(ee_project or config.ee_project)

    if isinstance(fields, str):
        fields_gdf = gpd.read_file(os.path.expanduser(fields))
    else:
        fields_gdf = fields

    fc = gdf_to_ee_feature_collection(fields_gdf, keep_props=[feature_id])

    start_year, end_year = _resolve_year_range(config, year_start, year_end, 1984, 2024)
    resolved_dest, resolved_bucket, resolved_drive_folder = _resolve_export_dest(
        config, dest, bucket, drive_folder
    )

    monthly_images: list[ee.Image] = []
    band_names: list[str] = []
    geometry = fc.geometry()
    for year in range(start_year, end_year + 1):
        images = _monthly_openet_images(year, geometry)
        monthly_images.extend(images)
        band_names.extend([f"eta_{year}_{month:02d}" for month in range(1, 13)])

    if not monthly_images:
        raise ValueError("No OpenET monthly images were created")

    stacked = ee.Image(monthly_images[0])
    for image in monthly_images[1:]:
        stacked = stacked.addBands(image)

    reduced = stacked.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=30,
        tileScale=8,
    )

    description = _sanitize_token(
        f"{config.project_name}_openet_eta_{start_year}_{end_year}"
    )[:100]
    prefix = config.get_bucket_path("openet_eta", description)

    export_table(
        reduced,
        desc=description,
        dest=resolved_dest,
        selectors=[feature_id] + band_names,
        bucket=resolved_bucket,
        file_prefix=prefix if resolved_dest == "bucket" else os.path.basename(prefix),
        drive_folder=resolved_drive_folder,
    )
    return {
        "product": "openet_eta_fields",
        "description": description,
        "prefix": prefix,
    }


def export_points_products_from_config(
    config: HandilyConfig,
    points: str | gpd.GeoDataFrame | None = None,
    product: str = "all",
    year_start: int | None = None,
    year_end: int | None = None,
    dest: str | None = None,
) -> list[dict[str, str]]:
    requested = product.lower()
    products = (
        ["irrmapper", "ndvi", "openet_eta", "gridmet"]
        if requested == "all"
        else [requested]
    )
    actions = {
        "irrmapper": export_points_irrmapper,
        "ndvi": export_points_ndvi,
        "openet_eta": export_points_openet_eta,
        "gridmet": export_points_gridmet,
    }

    results: list[dict[str, str]] = []
    for name in products:
        if name not in actions:
            raise ValueError(f"Unsupported points export product: {name}")
        LOGGER.info("Starting points export: %s", name)
        result = actions[name](
            points=points,
            config=config,
            year_start=year_start,
            year_end=year_end,
            dest=dest,
        )
        results.append(result)
    return results
