"""IrrMapper irrigation frequency export via Earth Engine.

Exports per-field annual irrigation fraction using the IrrMapper dataset.
Based on swim-rs/src/swimrs/data_extraction/ee/ee_props.py pattern.
"""

import logging
import os

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

LOGGER = logging.getLogger("handily.et.irrmapper")

# IrrMapper asset path
IRRMAPPER_ASSET = "projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp"

# Default year range
DEFAULT_START_YEAR = 1987
DEFAULT_END_YEAR = 2024


def gdf_to_ee_feature_collection(
    gdf: gpd.GeoDataFrame,
    feature_id: str = "FID",
    keep_props: list[str] | None = None,
) -> ee.FeatureCollection:
    """Convert a GeoDataFrame to an Earth Engine FeatureCollection.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame with polygon geometries.
    feature_id : str
        Name of the feature ID column to preserve.
    keep_props : list[str], optional
        Additional property names to preserve from GeoDataFrame rows.

    Returns
    -------
    ee.FeatureCollection
        EE FeatureCollection with geometries and specified properties.
    """
    if keep_props is None:
        keep_props = []

    # Always include feature_id
    if feature_id and feature_id not in keep_props:
        keep_props = [feature_id] + keep_props

    # Ensure WGS84 for EE
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Add centroid lat/lon for output
    centroids = gdf.geometry.centroid
    gdf = gdf.copy()
    gdf["_lat"] = centroids.y
    gdf["_lon"] = centroids.x

    feats = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Build properties dict
        props = {"LAT": row["_lat"], "LON": row["_lon"]}
        for k in keep_props:
            if k in row.index:
                val = row[k]
                # Convert numpy types to Python types for EE
                if hasattr(val, "item"):
                    val = val.item()
                props[k] = val

        # Convert geometry to EE
        geo = mapping(geom)
        if geo["type"] == "Polygon":
            ee_geom = ee.Geometry.Polygon(geo["coordinates"])
        elif geo["type"] == "MultiPolygon":
            ee_geom = ee.Geometry.MultiPolygon(geo["coordinates"])
        else:
            ee_geom = ee.Geometry(geo)

        feats.append(ee.Feature(ee_geom, props))

    return ee.FeatureCollection(feats)


def as_ee_feature_collection(
    fields: str | gpd.GeoDataFrame | ee.FeatureCollection,
    feature_id: str = "FID",
    keep_props: list[str] | None = None,
) -> ee.FeatureCollection:
    """Convert various inputs to an Earth Engine FeatureCollection.

    Parameters
    ----------
    fields : str, GeoDataFrame, or ee.FeatureCollection
        One of:
        - str: EE asset path ('projects/...', 'users/...') or local file path
        - GeoDataFrame: converted directly to EE FeatureCollection
        - ee.FeatureCollection: returned as-is
    feature_id : str
        Name of the feature ID column to preserve.
    keep_props : list[str], optional
        Additional property names to preserve.

    Returns
    -------
    ee.FeatureCollection
    """
    # Already an EE FeatureCollection
    if isinstance(fields, ee.FeatureCollection):
        return fields

    # GeoDataFrame - convert directly
    if isinstance(fields, gpd.GeoDataFrame):
        LOGGER.info(
            "Converting GeoDataFrame (%d features) to EE FeatureCollection", len(fields)
        )
        return gdf_to_ee_feature_collection(
            fields, feature_id=feature_id, keep_props=keep_props
        )

    # String input
    if isinstance(fields, str):
        # EE asset path
        if fields.startswith("projects/") or fields.startswith("users/"):
            LOGGER.info("Loading EE asset: %s", fields)
            return ee.FeatureCollection(fields)

        # Local file path
        if os.path.exists(fields):
            LOGGER.info("Loading local file: %s", fields)
            gdf = gpd.read_file(fields)
            return gdf_to_ee_feature_collection(
                gdf, feature_id=feature_id, keep_props=keep_props
            )

        # Assume it's an EE asset path that doesn't start with projects/users
        return ee.FeatureCollection(fields)

    raise TypeError(
        f"fields must be str, GeoDataFrame, or ee.FeatureCollection, got {type(fields)}"
    )


def export_irrigation_frequency(
    fields: str | gpd.GeoDataFrame | ee.FeatureCollection,
    desc: str,
    feature_id: str = "FID",
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    dest: str = "drive",
    bucket: str | None = None,
    drive_folder: str = "handily",
    file_prefix: str = "handily",
    select: list[str] | None = None,
    debug: bool = False,
) -> None:
    """Export per-field annual irrigation fraction using IrrMapper.

    For each year, computes the fraction of irrigated pixels (IrrMapper class < 1)
    within each field polygon.

    Parameters
    ----------
    fields : str, GeoDataFrame, or ee.FeatureCollection
        One of:
        - str: EE asset path ('projects/...', 'users/...') or local file path
        - GeoDataFrame: converted directly to EE FeatureCollection (no upload needed)
        - ee.FeatureCollection: used directly
    desc : str
        Export task description/prefix.
    feature_id : str
        Property name for feature ID (default 'FID').
    start_year : int
        First year to export (default 1987).
    end_year : int
        Last year to export (default 2024).
    dest : str
        Export destination: 'drive' or 'bucket'.
    bucket : str, optional
        GCS bucket name (required if dest='bucket').
    drive_folder : str
        Google Drive folder name (default 'handily').
    file_prefix : str
        Prefix for output files (default 'handily').
    select : list[str], optional
        Subset of feature IDs to process.
    debug : bool
        If True, print sample feature info.

    Returns
    -------
    None
        Starts an EE batch export task.
    """
    plots = as_ee_feature_collection(fields, feature_id=feature_id)

    # Optionally filter to subset of features
    if select is not None:
        plots = plots.filter(ee.Filter.inList(feature_id, select))

    irr_coll = ee.ImageCollection(IRRMAPPER_ASSET)

    selectors = [feature_id, "LAT", "LON"]
    irr_img = None
    first = True

    for year in range(start_year, end_year + 1):
        # Get IrrMapper classification for the year
        irr = (
            irr_coll.filterDate(f"{year}-01-01", f"{year}-12-31")
            .select("classification")
            .mosaic()
        )
        # Classification < 1 means irrigated (binary mask)
        # Result: 1 = irrigated, 0 = not irrigated
        irr = irr.lt(1).rename(f"irr_{year}").toInt()

        col_name = f"irr_{year}"
        selectors.append(col_name)

        if first:
            irr_img = irr.rename(col_name)
            first = False
        else:
            irr_img = irr_img.addBands(irr.rename(col_name))

    # Reduce to mean irrigation fraction per field
    means = irr_img.reduceRegions(
        collection=plots,
        reducer=ee.Reducer.mean(),
        scale=30,
    )

    if debug:
        sample = means.first().getInfo()
        LOGGER.info("Sample feature: %s", sample)

    # Export
    out_desc = f"{desc}_irr_freq"

    if dest == "bucket":
        if not bucket:
            raise ValueError("IrrMapper export dest='bucket' requires a bucket name")
        task = ee.batch.Export.table.toCloudStorage(
            means,
            description=out_desc,
            bucket=bucket,
            fileNamePrefix=f"{file_prefix}/irrmapper/{out_desc}",
            fileFormat="CSV",
            selectors=selectors,
        )
    elif dest == "drive":
        task = ee.batch.Export.table.toDrive(
            collection=means,
            description=out_desc,
            folder=drive_folder,
            fileNamePrefix=f"irrmapper/{out_desc}",
            fileFormat="CSV",
            selectors=selectors,
        )
    else:
        raise ValueError(f"dest must be 'drive' or 'bucket', got '{dest}'")

    task.start()
    LOGGER.info("Started IrrMapper export: %s", out_desc)
    print(f"IrrMapper export started: {out_desc}")


def load_irrigation_frequency(
    csv_path: str,
    feature_id: str = "FID",
) -> pd.DataFrame:
    """Load IrrMapper frequency CSV and compute summary statistics.

    Parameters
    ----------
    csv_path : str
        Path to exported IrrMapper CSV.
    feature_id : str
        Feature ID column name.

    Returns
    -------
    DataFrame
        With columns: feature_id, irr_mean, irr_count, irr_years, irr_freq
        - irr_mean: mean irrigation fraction across all years
        - irr_count: number of years with any irrigation (irr > 0.1)
        - irr_years: total years in record
        - irr_freq: irr_count / irr_years (frequency of irrigation)
    """
    df = pd.read_csv(csv_path)

    # Find irrigation columns
    irr_cols = [c for c in df.columns if c.startswith("irr_")]

    if not irr_cols:
        raise ValueError(f"No irrigation columns (irr_YYYY) found in {csv_path}")

    # Compute statistics
    irr_data = df[irr_cols]

    result = pd.DataFrame()
    result[feature_id] = df[feature_id]

    # Mean irrigation fraction across all years
    result["irr_mean"] = irr_data.mean(axis=1)

    # Number of years with significant irrigation (>10% of field)
    result["irr_count"] = (irr_data > 0.1).sum(axis=1)

    # Total years in record
    result["irr_years"] = len(irr_cols)

    # Frequency: proportion of years irrigated
    result["irr_freq"] = result["irr_count"] / result["irr_years"]

    LOGGER.info(
        "Loaded irrigation frequency: %d fields, %.1f%% mean irrigated",
        len(result),
        result["irr_mean"].mean() * 100,
    )

    return result


def identify_pattern_candidates(
    irr_stats: pd.DataFrame,
    max_irr_freq: float = 0.1,
    max_irr_mean: float = 0.05,
    feature_id: str = "FID",
) -> pd.DataFrame:
    """Identify pattern field candidates based on irrigation frequency.

    Pattern fields are those that are rarely or never irrigated, making them
    suitable donors for groundwater+soil moisture ET estimation.

    Parameters
    ----------
    irr_stats : DataFrame
        Output from load_irrigation_frequency().
    max_irr_freq : float
        Maximum irrigation frequency (default 0.1 = irrigated <10% of years).
    max_irr_mean : float
        Maximum mean irrigation fraction (default 0.05 = <5% mean coverage).
    feature_id : str
        Feature ID column name.

    Returns
    -------
    DataFrame
        With added 'pattern_candidate' (bool) and 'pattern_evidence' (str) columns.
    """
    df = irr_stats.copy()

    # Candidate criteria: low frequency AND low mean coverage
    low_freq = df["irr_freq"] <= max_irr_freq
    low_mean = df["irr_mean"] <= max_irr_mean

    df["pattern_candidate"] = low_freq & low_mean

    # Evidence string
    evidence = []
    for _, row in df.iterrows():
        parts = []
        if row["irr_freq"] <= max_irr_freq:
            parts.append(f"freq={row['irr_freq']:.2f}")
        if row["irr_mean"] <= max_irr_mean:
            parts.append(f"mean={row['irr_mean']:.2f}")
        evidence.append("; ".join(parts) if row["pattern_candidate"] else "")
    df["pattern_evidence"] = evidence

    n_candidates = df["pattern_candidate"].sum()
    LOGGER.info(
        "Pattern candidates: %d / %d (%.1f%%) with max_irr_freq=%.2f, max_irr_mean=%.2f",
        n_candidates,
        len(df),
        100 * n_candidates / len(df) if len(df) > 0 else 0,
        max_irr_freq,
        max_irr_mean,
    )

    return df
