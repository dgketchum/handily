"""Beaverhead example script for handily workflow testing.

This script exposes all module-level functions for dev testing and walkthrough.
Each step can be run independently via config flags or --step CLI argument.

Usage:
    python examples/beaverhead.py [config.toml]
    python examples/beaverhead.py --step rem          # Run only REM workflow
    python examples/beaverhead.py --step stratify     # Run only stratification
    python examples/beaverhead.py --step irrmapper    # Run only IrrMapper export
    python examples/beaverhead.py --step pattern      # Run only pattern selection
    python examples/beaverhead.py --step met          # Run only GridMET download
    python examples/beaverhead.py --step et           # Run only PT-JPL export
    python examples/beaverhead.py --step join         # Run only ET join
    python examples/beaverhead.py --step partition    # Run only ET partition
    python examples/beaverhead.py --step qgis         # Run only QGIS project update
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import tomllib

import geopandas as gpd
import rioxarray as rxr

from handily.config import HandilyConfig
from handily.et.gridmet import download_gridmet
from handily.et.irrmapper import export_irrigation_frequency, load_irrigation_frequency
from handily.et.join import join_gridmet_openet_eta
from handily.points.ee_extract import export_fields_openet_eta
from handily.et.partition import partition_et
from handily.io import aoi_from_bounds, ensure_dir
from handily.nhd import classify_flowlines
from handily.pattern import pattern_workflow
from handily.pipeline import REMWorkflow
from handily.qgis import discover_outputs, update_project
from handily.stratify import stratify, stratify_from_results

LOGGER = logging.getLogger("handily.beaverhead")


def configure_logging() -> None:
    level_name = os.environ.get("HANDILY_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


# =============================================================================
# REM Workflow
# =============================================================================


def dev_test_bounds_rem(
    config: HandilyConfig, bounds_wsen, ndwi_threshold: float
) -> dict:
    """Run REM workflow for bounds AOI.

    Returns dict with keys: aoi, flowlines, dem, rem, streams, ndwi, fields, fields_stats, summary
    """
    ensure_dir(config.out_dir)
    aoi = aoi_from_bounds(bounds_wsen)
    workflow = REMWorkflow(config=config, aoi=aoi)
    result = workflow.run(
        ndwi_threshold=float(ndwi_threshold), stats=("mean",), cache_flowlines=True
    )
    return result


# =============================================================================
# Stratification Workflow
# =============================================================================


def dev_test_stratify(
    config: HandilyConfig,
    results: dict | None = None,
    rem_threshold: float = 2.0,
    max_stream_distance: float | None = None,
) -> gpd.GeoDataFrame:
    """Run stratification workflow.

    If results dict is provided, uses those. Otherwise loads from disk.
    Returns GeoDataFrame with strata column.
    """
    if results is not None:
        LOGGER.info("Running stratification from provided results")
        return stratify_from_results(
            results,
            rem_threshold=rem_threshold,
            max_stream_distance=max_stream_distance,
        )

    # Load from disk
    rem_path = os.path.join(config.out_dir, "rem_bounds.tif")
    flowlines_path = os.path.join(config.out_dir, "flowlines_bounds.fgb")
    fields_path = os.path.join(config.out_dir, "fields_bounds.fgb")

    if not all(os.path.exists(p) for p in [rem_path, flowlines_path, fields_path]):
        raise FileNotFoundError(
            "Missing required files. Run REM workflow first (run_rem=true or --step rem)"
        )

    LOGGER.info("Loading data from disk for stratification")
    rem_da = rxr.open_rasterio(rem_path)
    if "band" in rem_da.dims:
        rem_da = rem_da.squeeze("band", drop=True)

    flowlines = gpd.read_file(flowlines_path)
    fields = gpd.read_file(fields_path)

    return stratify(
        fields=fields,
        flowlines=flowlines,
        rem_da=rem_da,
        rem_threshold=rem_threshold,
        max_stream_distance=max_stream_distance,
    )


def dev_test_classify_flowlines(
    config: HandilyConfig,
    flowlines: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Classify flowlines by stream category (for inspection).

    Useful for debugging NHD feature type mapping.
    """
    if flowlines is None:
        flowlines_path = os.path.join(config.out_dir, "flowlines_bounds.fgb")
        if not os.path.exists(flowlines_path):
            raise FileNotFoundError(f"Flowlines not found: {flowlines_path}")
        flowlines = gpd.read_file(flowlines_path)

    classified = classify_flowlines(flowlines)
    LOGGER.info("Flowline classification complete")
    LOGGER.info(
        "Categories: %s", classified["stream_category"].value_counts().to_dict()
    )
    return classified


# =============================================================================
# IrrMapper Workflow
# =============================================================================


def dev_test_irrmapper(
    config: HandilyConfig,
    fields: gpd.GeoDataFrame | str | None = None,
    start_year: int = 1987,
    end_year: int = 2024,
    dest: str = "bucket",
    desc: str | None = None,
) -> None:
    """Export IrrMapper irrigation frequency via Earth Engine.

    Can accept:
    - GeoDataFrame: converted directly to EE FeatureCollection (no upload needed)
    - str: EE asset path or local file path
    - None: loads from config.fields_path or config.ee_fields_asset

    Exports to: gs://{bucket}/{bucket_prefix}/{project_name}/irrmapper/{desc}_irr_freq.csv

    Note: This is an async operation. Check EE task manager for progress.
    After export completes, run dev_test_sync_irrmapper() to download locally.
    """
    import ee

    ee.Initialize()

    # Determine fields source
    if fields is None:
        # Try loading from disk first (preferred - no EE asset upload needed)
        fields_fgb = os.path.join(config.out_dir, "fields_bounds.fgb")
        if os.path.exists(fields_fgb):
            LOGGER.info("Loading fields from: %s", fields_fgb)
            fields = gpd.read_file(fields_fgb)
        elif config.ee_fields_asset:
            fields = config.ee_fields_asset
        else:
            raise ValueError(
                "No fields source found. Either run REM workflow first, "
                "or set ee_fields_asset in config."
            )

    if desc is None:
        desc = f"{config.project_name}_{config.feature_id}"

    # Build bucket path using config
    # Result: handily/beaverhead/irrmapper/{desc}_irr_freq
    file_prefix = f"{config.bucket_prefix}/{config.project_name}"

    export_irrigation_frequency(
        fields=fields,
        desc=desc,
        feature_id=config.feature_id,
        start_year=start_year,
        end_year=end_year,
        dest=dest,
        bucket=config.et_bucket,
        file_prefix=file_prefix,
    )

    # Log expected local path after sync
    expected_csv = f"{desc}_irr_freq.csv"
    if config.local_data_root:
        local_path = config.get_local_path("irrmapper", expected_csv)
        LOGGER.info("After sync, CSV will be at: %s", local_path)


def dev_test_sync_irrmapper(
    config: HandilyConfig,
    overwrite: bool = False,
    dry_run: bool = False,
) -> str | None:
    """Sync IrrMapper exports from bucket to local filesystem.

    Returns path to local CSV if found.
    """
    from handily.bucket import sync_bucket_to_local

    if config.local_data_root is None:
        raise ValueError("local_data_root not set in config")

    full_prefix = f"{config.bucket_prefix}/{config.project_name}"

    result = sync_bucket_to_local(
        bucket=config.et_bucket,
        bucket_prefix=full_prefix,
        local_root=config.local_data_root,
        subdir="irrmapper",
        glob_pattern="irr_freq",
        overwrite=overwrite,
        dry_run=dry_run,
    )

    if result["copied"] > 0 or result["skipped"] > 0:
        local_dir = config.get_local_path("irrmapper")
        csvs = [f for f in os.listdir(local_dir) if f.endswith(".csv")]
        if csvs:
            return os.path.join(local_dir, csvs[0])

    return None


def dev_test_load_irrmapper(
    irrmapper_csv: str,
    feature_id: str = "FID",
) -> tuple:
    """Load and inspect IrrMapper data.

    Returns (irr_stats_df, summary_dict) for inspection.
    """
    irr_stats = load_irrigation_frequency(irrmapper_csv, feature_id=feature_id)

    summary = {
        "n_fields": len(irr_stats),
        "mean_irr_freq": irr_stats["irr_freq"].mean(),
        "mean_irr_mean": irr_stats["irr_mean"].mean(),
        "n_never_irrigated": (irr_stats["irr_count"] == 0).sum(),
        "n_always_irrigated": (irr_stats["irr_freq"] > 0.9).sum(),
    }

    LOGGER.info("IrrMapper summary: %s", summary)
    return irr_stats, summary


# =============================================================================
# Pattern Selection Workflow
# =============================================================================


def dev_test_pattern(
    config: HandilyConfig,
    fields: gpd.GeoDataFrame | None = None,
    irrmapper_csv: str | None = None,
    max_irr_freq: float = 0.1,
    max_irr_mean: float = 0.05,
) -> gpd.GeoDataFrame:
    """Run pattern selection workflow.

    Requires either:
    - irrmapper_csv: path to exported IrrMapper CSV
    - Or run dev_test_irrmapper first and wait for export to complete
    """
    if fields is None:
        fields_path = os.path.join(config.out_dir, "fields_stratified.fgb")
        if os.path.exists(fields_path):
            fields = gpd.read_file(fields_path)
        else:
            fields_path = os.path.join(config.out_dir, "fields_bounds.fgb")
            if not os.path.exists(fields_path):
                raise FileNotFoundError(
                    "Fields file not found. Run REM workflow first."
                )
            fields = gpd.read_file(fields_path)

    if irrmapper_csv is None:
        irrmapper_csv = config.irrmapper_csv
        if irrmapper_csv is None or not os.path.exists(irrmapper_csv):
            raise FileNotFoundError(
                "IrrMapper CSV required. Run dev_test_irrmapper first, "
                "wait for export, then provide irrmapper_csv path."
            )

    return pattern_workflow(
        fields=fields,
        irrmapper_csv=irrmapper_csv,
        feature_id=config.feature_id,
        max_irr_freq=max_irr_freq,
        max_irr_mean=max_irr_mean,
    )


# =============================================================================
# ET Workflow Functions (existing)
# =============================================================================


def dev_test_met(config: HandilyConfig, overwrite: bool = False) -> None:
    """Download GridMET data for fields."""
    download_gridmet(
        config.fields_path,
        config.gridmet_parquet_dir,
        gridmet_centroids_path=config.gridmet_centroids_path,
        gridmet_centroid_parquet_dir=config.gridmet_centroid_parquet_dir,
        bounds_wsen=tuple(config.bounds) if config.bounds else None,
        start=config.met_start,
        end=config.met_end,
        overwrite=overwrite,
        feature_id=config.feature_id,
        gridmet_id_col=config.gridmet_id_col,
        return_df=False,
    )


def dev_test_et(config: HandilyConfig) -> None:
    """Export OpenET v2.0 ensemble ET zonal means for field polygons via Earth Engine.

    Exports to: gs://{bucket}/handily/{project}/openet_eta/{desc}.csv
    After export completes, sync with: handily sync --config ... --subdir openet_eta
    """
    export_fields_openet_eta(
        config.fields_path,
        config,
        year_start=config.openet_start_yr,
        year_end=config.openet_end_yr,
        feature_id=config.feature_id,
    )


def check_openet_data_exists(config: HandilyConfig) -> bool:
    """Check if OpenET CSV data exists locally."""
    if config.openet_csv_path is None:
        return False
    return os.path.exists(os.path.expanduser(config.openet_csv_path))


def check_irrmapper_data_exists(config: HandilyConfig) -> str | None:
    """Check if IrrMapper CSV data exists locally.

    Returns path to CSV if found, None otherwise.
    """
    # Check explicit config path first
    if config.irrmapper_csv and os.path.exists(
        os.path.expanduser(config.irrmapper_csv)
    ):
        return os.path.expanduser(config.irrmapper_csv)

    # Check expected location based on project structure
    if config.local_data_root:
        local_dir = config.get_local_path("irrmapper")
        if os.path.exists(local_dir):
            csvs = [
                f
                for f in os.listdir(local_dir)
                if f.endswith(".csv") and "irr_freq" in f
            ]
            if csvs:
                return os.path.join(local_dir, csvs[0])

    return None


def dev_test_join(config: HandilyConfig) -> None:
    """Join GridMET with OpenET monthly ET."""
    join_gridmet_openet_eta(
        config.gridmet_parquet_dir,
        config.openet_csv_path,
        config.et_join_parquet_dir,
        fields_path=config.fields_path,
        bounds_wsen=tuple(config.bounds) if config.bounds else None,
        feature_id=config.feature_id,
        eto_col="eto",
        prcp_col="prcp",
    )


def dev_test_partition(config: HandilyConfig) -> None:
    """Run ET partition workflow.

    Uses fields_pattern.fgb which has strata and pattern columns
    from the stratify and pattern steps.
    """
    # Use processed fields file with strata/pattern columns
    fields_fgb = os.path.join(config.out_dir, "fields_pattern.fgb")
    if not os.path.exists(fields_fgb):
        LOGGER.warning("Fields file not found: %s", fields_fgb)
        LOGGER.warning("Run stratify and pattern steps first")
        return

    partition_et(
        fields_fgb,
        config.partition_joined_parquet_dir,
        config.partition_out_parquet_dir,
        feature_id=config.feature_id,
        strata_col=config.partition_strata_col,
        pattern_col=config.partition_pattern_col,
    )


# =============================================================================
# QGIS Integration
# =============================================================================


def dev_test_qgis(config: HandilyConfig) -> str | None:
    """Update QGIS project with output layers.

    Returns path to updated project or None if no project configured.
    """
    if config.qgis_project is None:
        LOGGER.warning("No qgis_project configured; skipping QGIS update")
        return None

    if not os.path.exists(config.qgis_project):
        LOGGER.warning("QGIS project not found: %s", config.qgis_project)
        return None

    layers = discover_outputs(config.out_dir)
    if not layers:
        LOGGER.warning("No output layers found in %s", config.out_dir)
        return None

    update_project(config.qgis_project, layers, config.qgis_layer_group)
    LOGGER.info("Updated QGIS project: %s", config.qgis_project)
    LOGGER.info("  Group: %s, Layers: %d", config.qgis_layer_group, len(layers))
    return config.qgis_project


# =============================================================================
# Helper Functions
# =============================================================================


def _subset_results_to_aoi(results: dict, aoi_gdf: gpd.GeoDataFrame) -> dict:
    """Clip results to AOI extent."""
    out = dict(results)
    out["aoi"] = aoi_gdf

    for key in ("flowlines", "fields_stats"):
        gdf = out.get(key)
        if gdf is None or len(gdf) == 0:
            continue
        aoi_in_crs = aoi_gdf.to_crs(gdf.crs)
        try:
            out[key] = gpd.clip(gdf, aoi_in_crs)
        except Exception:
            out[key] = gpd.overlay(gdf, aoi_in_crs, how="intersection")

    for key in ("rem", "dem", "streams"):
        da = out.get(key)
        if da is None:
            continue
        shapes = [aoi_gdf.to_crs(da.rio.crs).geometry.unary_union.__geo_interface__]
        out[key] = da.rio.clip(shapes, all_touched=True)

    return out


def save_outputs(config: HandilyConfig, results: dict, overwrite: bool = False) -> dict:
    """Save workflow outputs to disk.

    Returns dict of output paths.
    """
    paths = {
        "rem": os.path.join(config.out_dir, "rem_bounds.tif"),
        "streams": os.path.join(config.out_dir, "streams_bounds.tif"),
        "ndwi": os.path.join(config.out_dir, "ndwi_bounds.tif"),
        "fields": os.path.join(config.out_dir, "fields_bounds.fgb"),
        "flowlines": os.path.join(config.out_dir, "flowlines_bounds.fgb"),
        "dem": os.path.join(config.out_dir, "dem_bounds_1m.tif"),
    }

    # Remove existing files if overwrite
    if overwrite:
        for path in paths.values():
            if os.path.exists(path):
                os.remove(path)

    # Save rasters
    for key in ("rem", "streams", "ndwi", "dem"):
        da = results.get(key)
        if da is not None:
            LOGGER.info("Writing %s: %s", key, paths[key])
            da.rio.to_raster(paths[key], overwrite=overwrite)

    # Save vectors - prefer fields_stats (has rem_mean) over raw fields
    fields_to_save = results.get("fields_stats")
    if fields_to_save is None:
        fields_to_save = results.get("fields")
    if fields_to_save is not None:
        fields = fields_to_save.copy()
        if "FID" in fields.columns:
            fields = fields.rename(columns={"FID": "FID_"})
        fields.to_file(paths["fields"], driver="FlatGeobuf")
        LOGGER.info("Writing fields: %s", paths["fields"])

    if results.get("flowlines") is not None:
        results["flowlines"].to_file(paths["flowlines"], driver="FlatGeobuf")
        LOGGER.info("Writing flowlines: %s", paths["flowlines"])

    return paths


def save_stratified_fields(config: HandilyConfig, fields: gpd.GeoDataFrame) -> str:
    """Save stratified fields to disk."""
    path = os.path.join(config.out_dir, "fields_stratified.fgb")
    if "FID" in fields.columns:
        fields = fields.rename(columns={"FID": "FID_"})
    fields.to_file(path, driver="FlatGeobuf")
    LOGGER.info("Stratified fields saved: %s", path)
    return path


# =============================================================================
# Main Entry Point
# =============================================================================


def main(argv=None) -> int:
    configure_logging()

    parser = argparse.ArgumentParser(description="Beaverhead development workflow")
    parser.add_argument("config", nargs="?", default=None, help="Config TOML path")
    parser.add_argument(
        "--step",
        choices=[
            "rem",
            "stratify",
            "irrmapper",
            "pattern",
            "met",
            "et",
            "join",
            "partition",
            "qgis",
            "all",
        ],
        help="Run specific step only",
    )

    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    config_path = (
        Path(args.config)
        if args.config
        else Path(__file__).with_name("beaverhead_config.toml")
    )

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must parse to a mapping, got {type(cfg).__name__}")

    config = HandilyConfig.from_dict(cfg)
    bounds_wsen = tuple(cfg["bounds"])
    ndwi_threshold = float(cfg.get("ndwi_threshold", 0.15))
    overwrite_outputs = bool(cfg.get("overwrite_outputs", False))

    # Config flags for each step
    run_rem = bool(cfg.get("run_rem", True))
    run_stratify = bool(cfg.get("run_stratify", False))
    run_irrmapper = bool(cfg.get("run_irrmapper", False))
    run_pattern = bool(cfg.get("run_pattern", False))
    run_met = bool(cfg.get("run_met", False))
    run_et = bool(cfg.get("run_et", False))
    run_join = bool(cfg.get("run_join", False))
    run_partition = bool(cfg.get("run_partition", False))
    run_qgis = bool(cfg.get("run_qgis", False))

    # Override with --step if provided
    if args.step:
        run_rem = args.step in ("rem", "all")
        run_stratify = args.step in ("stratify", "all")
        run_irrmapper = args.step in ("irrmapper", "all")
        run_pattern = args.step in ("pattern", "all")
        run_met = args.step in ("met", "all")
        run_et = args.step in ("et", "all")
        run_join = args.step in ("join", "all")
        run_partition = args.step in ("partition", "all")
        run_qgis = args.step in ("qgis", "all")

    ensure_dir(config.out_dir)
    results = None

    # Step 1: REM workflow
    if run_rem:
        LOGGER.info("=== Step: REM Workflow ===")
        results = dev_test_bounds_rem(
            config=config,
            bounds_wsen=bounds_wsen,
            ndwi_threshold=ndwi_threshold,
        )
        save_outputs(config, results, overwrite=overwrite_outputs)

    # Step 2: Stratification
    if run_stratify:
        LOGGER.info("=== Step: Stratification ===")
        rem_threshold = float(cfg.get("rem_threshold", 2.0))
        fields_stratified = dev_test_stratify(
            config=config,
            results=results,
            rem_threshold=rem_threshold,
        )
        save_stratified_fields(config, fields_stratified)

    # Step 3: IrrMapper export
    if run_irrmapper:
        LOGGER.info("=== Step: IrrMapper Export ===")
        # Prefer using local fields GeoDataFrame (no EE asset upload needed)
        irr_fields = None
        fields_fgb = os.path.join(config.out_dir, "fields_bounds.fgb")
        if os.path.exists(fields_fgb):
            irr_fields = gpd.read_file(fields_fgb)
            LOGGER.info(
                "Using local fields: %s (%d features)", fields_fgb, len(irr_fields)
            )
        dev_test_irrmapper(config=config, fields=irr_fields)

    # Step 4: Pattern selection
    if run_pattern:
        LOGGER.info("=== Step: Pattern Selection ===")
        irrmapper_csv = cfg.get("irrmapper_csv")
        if irrmapper_csv and os.path.exists(irrmapper_csv):
            fields_with_pattern = dev_test_pattern(
                config=config,
                irrmapper_csv=irrmapper_csv,
            )
            pattern_path = os.path.join(config.out_dir, "fields_pattern.fgb")
            fields_with_pattern.to_file(pattern_path, driver="FlatGeobuf")
            LOGGER.info("Pattern fields saved: %s", pattern_path)
        else:
            LOGGER.warning("irrmapper_csv not found; skipping pattern selection")

    # Step 5: GridMET
    if run_met:
        LOGGER.info("=== Step: GridMET Download ===")
        dev_test_met(config=config, overwrite=bool(cfg.get("overwrite_met", False)))

    # Step 6: PT-JPL ET export
    if run_et:
        LOGGER.info("=== Step: PT-JPL Export ===")
        dev_test_et(config=config)

    # Step 7: Join
    if run_join:
        LOGGER.info("=== Step: ET Join ===")
        dev_test_join(config=config)

    # Step 8: Partition
    if run_partition:
        LOGGER.info("=== Step: ET Partition ===")
        dev_test_partition(config=config)

    # Step 9: QGIS update
    if run_qgis:
        LOGGER.info("=== Step: QGIS Update ===")
        project_path = dev_test_qgis(config=config)
        if project_path:
            print(f"QGIS project updated: {project_path}")

    # Summary
    print("--- Beaverhead Workflow Summary ---")
    if results:
        print(f"Fields total: {results['summary'].get('total_fields')}")
        print(f"NDWI threshold: {results['summary'].get('ndwi_threshold')}")
    print(f"Output dir: {config.out_dir}")
    if config.qgis_project:
        print(f"QGIS project: {config.qgis_project}")
        print("Run 'handily qgis update --config <config.toml>' to update QGIS layers")

    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "12")
    os.environ.setdefault("OMP_THREAD_LIMIT", "12")
    raise SystemExit(main())
