"""Pattern field selection workflow for ET partitioning.

Pattern fields are non-irrigated fields used as donors for groundwater+soil moisture
ET estimation. This module identifies pattern candidates based on IrrMapper
irrigation frequency data.

Workflow:
1. Export IrrMapper irrigation frequency (async EE task)
2. Load exported data and compute irrigation statistics
3. Identify pattern candidates based on low irrigation frequency
4. Assign 'pattern' column to fields
"""
import logging
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

from handily.et.irrmapper import (
    export_irrigation_frequency,
    identify_pattern_candidates,
    load_irrigation_frequency,
)

LOGGER = logging.getLogger("handily.pattern")


def assign_pattern_from_irrmapper(
    fields: gpd.GeoDataFrame,
    irrmapper_csv: str,
    feature_id: str = "FID",
    max_irr_freq: float = 0.1,
    max_irr_mean: float = 0.05,
) -> gpd.GeoDataFrame:
    """Assign pattern column to fields based on IrrMapper data.

    Parameters
    ----------
    fields : GeoDataFrame
        Field polygons with feature_id column.
    irrmapper_csv : str
        Path to exported IrrMapper CSV from export_irrigation_frequency().
    feature_id : str
        Feature ID column name (default 'FID').
    max_irr_freq : float
        Maximum irrigation frequency to qualify as pattern (default 0.1).
    max_irr_mean : float
        Maximum mean irrigation fraction to qualify as pattern (default 0.05).

    Returns
    -------
    GeoDataFrame
        Fields with added columns:
        - irr_mean: mean irrigation fraction
        - irr_freq: frequency of irrigation
        - pattern: True if field is a pattern candidate
        - pattern_evidence: string describing why field was flagged
    """
    LOGGER.info("Loading IrrMapper data from %s", irrmapper_csv)

    # Load and compute irrigation statistics
    irr_stats = load_irrigation_frequency(irrmapper_csv, feature_id=feature_id)

    # Identify pattern candidates
    irr_with_patterns = identify_pattern_candidates(
        irr_stats,
        max_irr_freq=max_irr_freq,
        max_irr_mean=max_irr_mean,
        feature_id=feature_id,
    )

    # Join to fields
    fields = fields.copy()

    # Ensure feature_id types match
    if feature_id in fields.columns:
        fields[feature_id] = fields[feature_id].astype(str)
    irr_with_patterns[feature_id] = irr_with_patterns[feature_id].astype(str)

    # Select columns to join
    join_cols = [feature_id, "irr_mean", "irr_freq", "pattern_candidate", "pattern_evidence"]
    join_df = irr_with_patterns[join_cols]

    # Merge
    fields_merged = fields.merge(join_df, on=feature_id, how="left")

    # Rename pattern_candidate to pattern
    fields_merged = fields_merged.rename(columns={"pattern_candidate": "pattern"})

    # Fill missing patterns with False
    fields_merged["pattern"] = fields_merged["pattern"].fillna(False)

    n_pattern = fields_merged["pattern"].sum()
    LOGGER.info(
        "Pattern assignment complete: %d / %d fields (%.1f%%)",
        n_pattern,
        len(fields_merged),
        100 * n_pattern / len(fields_merged) if len(fields_merged) > 0 else 0,
    )

    return fields_merged


def assign_pattern_by_strata(
    fields: gpd.GeoDataFrame,
    min_per_strata: int = 1,
    strata_col: str = "strata",
    pattern_col: str = "pattern",
) -> gpd.GeoDataFrame:
    """Ensure each strata has at least min_per_strata pattern fields.

    If a strata has fewer pattern fields than required, this function
    promotes additional fields based on lowest irrigation frequency.

    Parameters
    ----------
    fields : GeoDataFrame
        Fields with strata and pattern columns.
    min_per_strata : int
        Minimum pattern fields per strata (default 1).
    strata_col : str
        Strata column name.
    pattern_col : str
        Pattern column name.

    Returns
    -------
    GeoDataFrame
        Fields with potentially updated pattern column.
    """
    if strata_col not in fields.columns:
        raise ValueError(f"'{strata_col}' column not found")
    if pattern_col not in fields.columns:
        raise ValueError(f"'{pattern_col}' column not found")
    if "irr_freq" not in fields.columns:
        LOGGER.warning("'irr_freq' column not found; cannot promote additional patterns")
        return fields

    df = fields.copy()
    promoted_count = 0

    for strata in df[strata_col].unique():
        if strata == "non_partitioned":
            continue

        strata_mask = df[strata_col] == strata
        strata_fields = df[strata_mask]

        current_patterns = strata_fields[pattern_col].sum()

        if current_patterns < min_per_strata:
            # Find non-pattern fields with lowest irrigation frequency
            non_pattern_mask = strata_mask & ~df[pattern_col]
            candidates = df[non_pattern_mask].sort_values("irr_freq")

            n_to_promote = min_per_strata - current_patterns
            promote_idx = candidates.head(n_to_promote).index

            df.loc[promote_idx, pattern_col] = True
            df.loc[promote_idx, "pattern_evidence"] = "promoted:min_per_strata"
            promoted_count += len(promote_idx)

            LOGGER.info(
                "Strata '%s': promoted %d fields to pattern (was %d, now %d)",
                strata,
                len(promote_idx),
                current_patterns,
                current_patterns + len(promote_idx),
            )

    if promoted_count > 0:
        LOGGER.info("Total fields promoted to pattern: %d", promoted_count)

    return df


def pattern_workflow(
    fields: gpd.GeoDataFrame,
    irrmapper_csv: str | None = None,
    ee_asset: str | None = None,
    feature_id: str = "FID",
    max_irr_freq: float = 0.1,
    max_irr_mean: float = 0.05,
    min_per_strata: int = 1,
    strata_col: str = "strata",
    export_desc: str = "handily",
    export_dest: str = "drive",
    export_bucket: str | None = None,
) -> gpd.GeoDataFrame:
    """Full pattern selection workflow.

    If irrmapper_csv is provided, loads existing data.
    Otherwise, if ee_asset is provided, starts EE export (async).

    Parameters
    ----------
    fields : GeoDataFrame
        Field polygons (should already have strata column).
    irrmapper_csv : str, optional
        Path to existing IrrMapper CSV export.
    ee_asset : str, optional
        EE asset path for fields (to start new export).
    feature_id : str
        Feature ID column name.
    max_irr_freq : float
        Maximum irrigation frequency for pattern candidates.
    max_irr_mean : float
        Maximum mean irrigation fraction for pattern candidates.
    min_per_strata : int
        Minimum pattern fields per strata.
    strata_col : str
        Strata column name.
    export_desc : str
        Description for EE export task.
    export_dest : str
        Export destination ('drive' or 'bucket').
    export_bucket : str, optional
        GCS bucket (required if export_dest='bucket').

    Returns
    -------
    GeoDataFrame
        Fields with pattern column, or original fields if export started.
    """
    LOGGER.info("Starting pattern selection workflow")

    if irrmapper_csv and os.path.exists(irrmapper_csv):
        LOGGER.info("Using existing IrrMapper data: %s", irrmapper_csv)

        # Assign patterns from existing data
        fields_with_pattern = assign_pattern_from_irrmapper(
            fields,
            irrmapper_csv,
            feature_id=feature_id,
            max_irr_freq=max_irr_freq,
            max_irr_mean=max_irr_mean,
        )

        # Ensure minimum patterns per strata
        if strata_col in fields_with_pattern.columns:
            fields_with_pattern = assign_pattern_by_strata(
                fields_with_pattern,
                min_per_strata=min_per_strata,
                strata_col=strata_col,
            )

        return fields_with_pattern

    elif ee_asset:
        LOGGER.info("Starting IrrMapper export from EE asset: %s", ee_asset)
        LOGGER.warning("Export is async. Re-run with irrmapper_csv path after export completes.")

        export_irrigation_frequency(
            fields=ee_asset,
            desc=export_desc,
            feature_id=feature_id,
            dest=export_dest,
            bucket=export_bucket,
        )

        # Return fields without pattern column (export pending)
        return fields

    else:
        raise ValueError(
            "Either irrmapper_csv (existing export) or ee_asset (for new export) is required"
        )
