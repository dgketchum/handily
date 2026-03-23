from __future__ import annotations

import logging
from typing import Any

import ee
import geopandas as gpd
from shapely.geometry import mapping

LOGGER = logging.getLogger("handily.ee.common")


def initialize_ee(project: str | None = None) -> None:
    if project:
        ee.Initialize(project=project)
        return
    ee.Initialize()


def _coerce_property(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    return value


def gdf_to_ee_feature_collection(
    gdf: gpd.GeoDataFrame,
    keep_props: list[str] | None = None,
) -> ee.FeatureCollection:
    if keep_props is None:
        keep_props = []

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    feats: list[ee.Feature] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        props = {}
        for key in keep_props:
            if key in row.index:
                props[key] = _coerce_property(row[key])

        geo = mapping(geom)
        feats.append(ee.Feature(ee.Geometry(geo), props))

    return ee.FeatureCollection(feats)


def export_table(
    collection: ee.FeatureCollection,
    desc: str,
    dest: str = "bucket",
    selectors: list[str] | None = None,
    bucket: str | None = None,
    file_prefix: str | None = None,
    drive_folder: str | None = None,
) -> ee.batch.Task:
    if dest == "bucket":
        if not bucket:
            raise ValueError("bucket export requires a bucket name")
        if not file_prefix:
            raise ValueError("bucket export requires file_prefix")
        task = ee.batch.Export.table.toCloudStorage(
            collection=collection,
            description=desc,
            bucket=bucket,
            fileNamePrefix=file_prefix,
            fileFormat="CSV",
            selectors=selectors,
        )
    elif dest == "drive":
        if not drive_folder:
            raise ValueError("drive export requires drive_folder")
        prefix = file_prefix or desc
        task = ee.batch.Export.table.toDrive(
            collection=collection,
            description=desc,
            folder=drive_folder,
            fileNamePrefix=prefix,
            fileFormat="CSV",
            selectors=selectors,
        )
    else:
        raise ValueError(f"Unsupported export destination: {dest}")

    task.start()
    LOGGER.info("Started EE table export: %s", desc)
    return task
