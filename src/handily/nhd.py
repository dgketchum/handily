"""NHD feature type mapping and stream classification utilities.

Maps NHD FCODE values to stream categories for stratification:
- perennial: natural perennial streams/rivers
- intermittent: intermittent or ephemeral streams
- managed: canals, ditches, artificial paths
"""

import logging
from typing import Literal

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

LOGGER = logging.getLogger("handily.nhd")

# NHD FCODE to stream category mapping
# Reference: https://nhd.usgs.gov/userGuide/Robohelpfiles/NHD_User_Guide/Feature_Catalog/Hydrography_Dataset/Complete_FCode_List.htm
FCODE_CATEGORIES = {
    # Stream/River - Perennial
    46006: "perennial",
    # Stream/River - Intermittent
    46003: "intermittent",
    # Stream/River - Ephemeral
    46007: "intermittent",
    # Stream/River - Unknown (treat as intermittent to be conservative)
    46000: "intermittent",
    # Large river centerline (NHD artificial path through NHDArea polygons)
    55800: "perennial",
    # Canal/Ditch variants — kept in FGB and stratification but excluded from
    # REM stream mask (see REM_EXCLUDED_FCODES) to prevent hillside artifacts
    33400: "managed",  # Connector
    33600: "managed",  # Canal/Ditch
    33601: "managed",  # Canal/Ditch (aqueduct) — hillside canals cause false low-REM
    33602: "managed",  # Canal/Ditch (stormwater)
    33603: "managed",  # Canal/Ditch (irrigation)
    # Artificial Path (through lakes/reservoirs)
    46800: "managed",
    # Pipeline variants - exclude
    42000: None,
    42001: None,
    42003: None,
    42800: None,
    42801: None,
    42802: None,
    42803: None,
    42805: None,
    42807: None,
    42809: None,
    42810: None,
    42811: None,
    42813: None,
    42816: None,
}

# FCODEs to exclude from the REM stream mask only (still kept in FGB and stratification).
# Hillside canals (33601) traverse slope contours and cause false near-zero REM patches
# on adjacent hillside pixels. They are retained for DTW modeling via the FGB.
# Intermittent streams (46003) are kept and filtered by network connectivity instead.
REM_EXCLUDED_FCODES: frozenset[int] = frozenset({33601})

StreamCategory = Literal["perennial", "intermittent", "managed"]


def get_fcode_column(gdf: gpd.GeoDataFrame) -> str | None:
    """Find the FCODE column in a GeoDataFrame (case-insensitive)."""
    for col in gdf.columns:
        if col.lower() == "fcode":
            return col
    return None


def classify_flowlines(
    flowlines: gpd.GeoDataFrame,
    fcode_col: str | None = None,
) -> gpd.GeoDataFrame:
    """Add stream_category column to flowlines based on FCODE.

    Parameters
    ----------
    flowlines : GeoDataFrame
        NHD flowlines with FCODE attribute
    fcode_col : str, optional
        Name of FCODE column. Auto-detected if not provided.

    Returns
    -------
    GeoDataFrame
        Flowlines with added 'stream_category' column
    """
    if len(flowlines) == 0:
        flowlines = flowlines.copy()
        flowlines["stream_category"] = None
        return flowlines

    if fcode_col is None:
        fcode_col = get_fcode_column(flowlines)
        if fcode_col is None:
            raise ValueError(
                "FCODE column not found in flowlines. Available columns: "
                f"{list(flowlines.columns)}"
            )

    df = flowlines.copy()

    # Map FCODE to category
    df["stream_category"] = df[fcode_col].map(FCODE_CATEGORIES)

    # Log unmapped codes
    unmapped = df[df["stream_category"].isna() & df[fcode_col].notna()][
        fcode_col
    ].unique()
    if len(unmapped) > 0:
        LOGGER.warning(
            "Unmapped FCODE values (will be excluded): %s", unmapped.tolist()
        )

    counts = df["stream_category"].value_counts(dropna=False)
    LOGGER.info("Flowline classification: %s", counts.to_dict())

    return df


def filter_flowlines_for_stratification(
    flowlines: gpd.GeoDataFrame,
    include_categories: list[StreamCategory] | None = None,
) -> gpd.GeoDataFrame:
    """Filter flowlines to only include specified stream categories.

    Parameters
    ----------
    flowlines : GeoDataFrame
        Flowlines with 'stream_category' column (from classify_flowlines)
    include_categories : list, optional
        Categories to include. Default: ['perennial', 'intermittent', 'managed']

    Returns
    -------
    GeoDataFrame
        Filtered flowlines
    """
    if "stream_category" not in flowlines.columns:
        flowlines = classify_flowlines(flowlines)

    if include_categories is None:
        include_categories = ["perennial", "intermittent", "managed"]

    mask = flowlines["stream_category"].isin(include_categories)
    filtered = flowlines[mask].copy()

    LOGGER.info(
        "Filtered flowlines: %d -> %d (categories: %s)",
        len(flowlines),
        len(filtered),
        include_categories,
    )

    return filtered.reset_index(drop=True)


def assign_nearest_stream_type(
    fields: gpd.GeoDataFrame,
    flowlines: gpd.GeoDataFrame,
    max_distance: float | None = None,
) -> gpd.GeoDataFrame:
    """Assign nearest stream category to each field based on centroid distance.

    Parameters
    ----------
    fields : GeoDataFrame
        Fields to classify
    flowlines : GeoDataFrame
        Flowlines with 'stream_category' column
    max_distance : float, optional
        Maximum distance in CRS units. Fields beyond this get 'none'.

    Returns
    -------
    GeoDataFrame
        Fields with added 'nearest_stream_type' and 'stream_distance' columns
    """
    if "stream_category" not in flowlines.columns:
        flowlines = classify_flowlines(flowlines)

    # Filter to valid categories only
    valid_flowlines = flowlines[flowlines["stream_category"].notna()].copy()

    if len(valid_flowlines) == 0:
        LOGGER.warning("No valid flowlines for stream type assignment")
        fields = fields.copy()
        fields["nearest_stream_type"] = "none"
        fields["stream_distance"] = np.nan
        return fields

    # Ensure same CRS
    if fields.crs != valid_flowlines.crs:
        valid_flowlines = valid_flowlines.to_crs(fields.crs)

    # Get field centroids
    field_centroids = fields.geometry.centroid
    field_coords = np.array([(p.x, p.y) for p in field_centroids])

    # Sample points along flowlines for distance calculation
    # Use interpolate to get evenly-spaced points
    flowline_points = []
    flowline_categories = []

    for idx, row in valid_flowlines.iterrows():
        geom = row.geometry
        category = row["stream_category"]

        if geom is None or geom.is_empty:
            continue

        # Sample at ~100m intervals (or finer for short segments)
        length = geom.length
        n_points = max(2, int(length / 100))

        for i in range(n_points):
            frac = i / (n_points - 1) if n_points > 1 else 0
            pt = geom.interpolate(frac, normalized=True)
            flowline_points.append((pt.x, pt.y))
            flowline_categories.append(category)

    if len(flowline_points) == 0:
        LOGGER.warning("No flowline points generated")
        fields = fields.copy()
        fields["nearest_stream_type"] = "none"
        fields["stream_distance"] = np.nan
        return fields

    flowline_coords = np.array(flowline_points)
    flowline_categories = np.array(flowline_categories)

    # Build KD-tree for fast nearest neighbor lookup
    tree = cKDTree(flowline_coords)
    distances, indices = tree.query(field_coords, k=1)

    # Assign categories
    nearest_types = flowline_categories[indices]

    # Apply max distance filter
    if max_distance is not None:
        nearest_types = np.where(distances > max_distance, "none", nearest_types)

    fields = fields.copy()
    fields["nearest_stream_type"] = nearest_types
    fields["stream_distance"] = distances

    type_counts = fields["nearest_stream_type"].value_counts()
    LOGGER.info("Nearest stream type assignment: %s", type_counts.to_dict())

    return fields
