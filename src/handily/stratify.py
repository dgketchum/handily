"""Stratification workflow for ET partitioning.

Assigns strata to fields based on:
1. REM threshold (partitioned vs non-partitioned)
2. Nearest stream type (perennial, intermittent, managed)

Output: fields GeoDataFrame with 'partitioned' (bool) and 'strata' (str) columns.
"""

import logging
from typing import Literal

import geopandas as gpd
import xarray as xr

from handily.compute import compute_field_rem_stats, stratify_fields_by_rem
from handily.nhd import (
    assign_nearest_stream_type,
    classify_flowlines,
    filter_flowlines_for_stratification,
)

LOGGER = logging.getLogger("handily.stratify")

StrataType = Literal[
    "perennial",  # partitioned, nearest stream is perennial
    "intermittent",  # partitioned, nearest stream is intermittent
    "managed",  # partitioned, nearest stream is managed (canal/ditch)
    "non_partitioned",  # REM >= threshold, outside shallow GW area
]


def assign_strata(
    fields: gpd.GeoDataFrame,
    partitioned_col: str = "partitioned",
    stream_type_col: str = "nearest_stream_type",
) -> gpd.GeoDataFrame:
    """Combine partition status and stream type into categorical strata column.

    Parameters
    ----------
    fields : GeoDataFrame
        Fields with 'partitioned' and 'nearest_stream_type' columns.
    partitioned_col : str
        Name of partition boolean column.
    stream_type_col : str
        Name of stream type column.

    Returns
    -------
    GeoDataFrame
        Fields with added 'strata' column.
    """
    if partitioned_col not in fields.columns:
        raise ValueError(
            f"'{partitioned_col}' column not found. Run REM stratification first."
        )
    if stream_type_col not in fields.columns:
        raise ValueError(
            f"'{stream_type_col}' column not found. Run stream type assignment first."
        )

    df = fields.copy()

    # Assign strata based on partition status and stream type
    strata = []
    for _, row in df.iterrows():
        if not row[partitioned_col]:
            strata.append("non_partitioned")
        else:
            stream_type = row[stream_type_col]
            if stream_type in ("perennial", "intermittent", "managed"):
                strata.append(stream_type)
            else:
                # Default to intermittent if stream type unknown
                strata.append("intermittent")

    df["strata"] = strata

    # Log strata distribution
    strata_counts = df["strata"].value_counts()
    LOGGER.info("Strata distribution: %s", strata_counts.to_dict())

    return df


def stratify(
    fields: gpd.GeoDataFrame,
    flowlines: gpd.GeoDataFrame,
    rem_da: xr.DataArray,
    rem_threshold: float = 2.0,
    max_stream_distance: float | None = None,
    rem_stats: tuple[str, ...] = ("mean",),
) -> gpd.GeoDataFrame:
    """Full stratification workflow: REM threshold + stream type assignment.

    This is the main entry point for stratification. It:
    1. Computes REM zonal statistics for each field
    2. Assigns 'partitioned' based on REM threshold
    3. Classifies flowlines by stream category
    4. Assigns nearest stream type to each field
    5. Combines into final 'strata' column

    Parameters
    ----------
    fields : GeoDataFrame
        Field polygons to stratify.
    flowlines : GeoDataFrame
        NHD flowlines with FCODE attribute.
    rem_da : DataArray
        REM raster.
    rem_threshold : float
        REM threshold in meters (default 2.0). Fields with mean REM below
        this are considered within shallow groundwater influence.
    max_stream_distance : float, optional
        Maximum distance to nearest stream (in CRS units). Fields beyond
        this distance get 'none' stream type.
    rem_stats : tuple
        Statistics to compute for REM (default: ('mean',)).

    Returns
    -------
    GeoDataFrame
        Fields with added columns:
        - rem_mean (float): mean REM value
        - partitioned (bool): True if REM < threshold
        - stream_category (str): category of nearest stream
        - nearest_stream_type (str): type of nearest stream
        - stream_distance (float): distance to nearest stream
        - strata (str): final strata assignment
    """
    LOGGER.info("Starting stratification workflow")
    LOGGER.info(
        "REM threshold: %.1f m, max_stream_distance: %s",
        rem_threshold,
        max_stream_distance,
    )

    # Step 1: Compute REM statistics (skip if already present)
    if "rem_mean" in fields.columns:
        LOGGER.info("Step 1: Using existing rem_mean column (%d fields)", len(fields))
        fields_stats = fields.copy()
    else:
        LOGGER.info("Step 1: Computing REM zonal statistics")
        fields_stats = compute_field_rem_stats(fields, rem_da, stats=rem_stats)
        LOGGER.info("Computed stats for %d fields", len(fields_stats))

    # Step 2: Assign partition status based on REM threshold
    LOGGER.info("Step 2: Applying REM threshold (%.1f m)", rem_threshold)
    fields_partitioned = stratify_fields_by_rem(fields_stats, threshold_m=rem_threshold)
    n_partitioned = fields_partitioned["partitioned"].sum()
    LOGGER.info(
        "Partitioned: %d / %d (%.1f%%)",
        n_partitioned,
        len(fields_partitioned),
        100 * n_partitioned / len(fields_partitioned)
        if len(fields_partitioned) > 0
        else 0,
    )

    # Step 3: Classify flowlines
    LOGGER.info("Step 3: Classifying flowlines by stream category")
    flowlines_classified = classify_flowlines(flowlines)
    flowlines_filtered = filter_flowlines_for_stratification(flowlines_classified)

    # Step 4: Assign nearest stream type
    LOGGER.info("Step 4: Assigning nearest stream type to fields")
    fields_with_streams = assign_nearest_stream_type(
        fields_partitioned,
        flowlines_filtered,
        max_distance=max_stream_distance,
    )

    # Step 5: Combine into strata
    LOGGER.info("Step 5: Assigning final strata")
    fields_stratified = assign_strata(fields_with_streams)

    LOGGER.info("Stratification complete")
    return fields_stratified


def stratify_from_results(
    results: dict,
    rem_threshold: float = 2.0,
    max_stream_distance: float | None = None,
) -> gpd.GeoDataFrame:
    """Convenience wrapper to stratify from REMWorkflow results dict.

    Parameters
    ----------
    results : dict
        Output from REMWorkflow.run() containing 'fields', 'flowlines', 'rem' keys.
    rem_threshold : float
        REM threshold in meters.
    max_stream_distance : float, optional
        Maximum distance to nearest stream.

    Returns
    -------
    GeoDataFrame
        Stratified fields.
    """
    required_keys = ("fields", "flowlines", "rem")
    for key in required_keys:
        if key not in results:
            raise ValueError(f"Missing required key '{key}' in results dict")

    return stratify(
        fields=results["fields"],
        flowlines=results["flowlines"],
        rem_da=results["rem"],
        rem_threshold=rem_threshold,
        max_stream_distance=max_stream_distance,
    )
