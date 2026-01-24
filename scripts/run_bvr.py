"""Run Beaverhead workflow programmatically without CLI.

This script demonstrates how to run the handily workflow step-by-step
from a main execution block for development and testing.

Usage:
    python scripts/run_bvr.py
"""
import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import geopandas as gpd
import yaml

from handily.config import HandilyConfig
from handily.io import aoi_from_bounds, ensure_dir

# Import all dev_test functions from beaverhead
from beaverhead import (
    configure_logging,
    dev_test_bounds_rem,
    dev_test_classify_flowlines,
    dev_test_stratify,
    dev_test_irrmapper,
    dev_test_load_irrmapper,
    dev_test_pattern,
    dev_test_met,
    dev_test_et,
    dev_test_join,
    dev_test_partition,
    dev_test_viz,
    save_outputs,
    save_stratified_fields,
)


def main():
    configure_logging()

    # Load config
    config_path = Path(__file__).with_name("beaverhead_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config = HandilyConfig.from_dict(cfg)
    bounds_wsen = tuple(cfg["bounds"])
    ndwi_threshold = float(cfg.get("ndwi_threshold", 0.15))
    rem_threshold = float(cfg.get("rem_threshold", 2.0))
    overwrite = bool(cfg.get("overwrite_outputs", False))

    ensure_dir(config.out_dir)

    # =========================================================================
    # Step 1: REM Workflow
    # =========================================================================
    print("\n=== Step 1: REM Workflow ===")

    # Check if cached outputs exist
    rem_path = os.path.join(config.out_dir, "rem_bounds.tif")
    fields_path = os.path.join(config.out_dir, "fields_bounds.fgb")
    flowlines_path = os.path.join(config.out_dir, "flowlines_bounds.fgb")
    cached_exists = all(os.path.exists(p) for p in [rem_path, fields_path, flowlines_path])

    if cached_exists and not overwrite:
        print("  Loading cached outputs (set overwrite=True to rerun)")
        import rioxarray as rxr

        rem_da = rxr.open_rasterio(rem_path)
        if "band" in rem_da.dims:
            rem_da = rem_da.squeeze("band", drop=True)

        results = {
            "rem": rem_da,
            "fields": gpd.read_file(fields_path),
            "flowlines": gpd.read_file(flowlines_path),
            "aoi": aoi_from_bounds(bounds_wsen),
            "summary": {"ndwi_threshold": ndwi_threshold},
        }
        # Load optional rasters if they exist
        for key, fname in [("dem", "dem_bounds_1m.tif"), ("streams", "streams_bounds.tif")]:
            path = os.path.join(config.out_dir, fname)
            if os.path.exists(path):
                da = rxr.open_rasterio(path)
                if "band" in da.dims:
                    da = da.squeeze("band", drop=True)
                results[key] = da
    else:
        print("  Running REM workflow..." if not cached_exists else "  Overwriting cached outputs...")
        results = dev_test_bounds_rem(
            config=config,
            bounds_wsen=bounds_wsen,
            ndwi_threshold=ndwi_threshold,
        )
        save_outputs(config, results, overwrite=overwrite)

    # Inspect results
    print(f"  Fields: {len(results['fields'])}")
    print(f"  Flowlines: {len(results['flowlines'])}")
    print(f"  REM shape: {results['rem'].shape}")

    # =========================================================================
    # Step 2: Classify Flowlines (optional inspection)
    # =========================================================================
    print("\n=== Step 2: Classify Flowlines ===")
    flowlines_classified = dev_test_classify_flowlines(
        config=config,
        flowlines=results["flowlines"],
    )
    print(f"  Stream categories: {flowlines_classified['stream_category'].value_counts().to_dict()}")

    # =========================================================================
    # Step 3: Stratification
    # =========================================================================
    print("\n=== Step 3: Stratification ===")
    fields_stratified = dev_test_stratify(
        config=config,
        results=results,
        rem_threshold=rem_threshold,
    )
    save_stratified_fields(config, fields_stratified)

    # Inspect stratification
    print(f"  Strata distribution: {fields_stratified['strata'].value_counts().to_dict()}")
    print(f"  Partitioned: {fields_stratified['partitioned'].sum()} / {len(fields_stratified)}")

    # =========================================================================
    # Step 4: IrrMapper Export (async)
    # =========================================================================
    # NOTE: This starts an async EE export task. Check EE task manager for progress.
    # Uses the local fields GeoDataFrame directly - no EE asset upload needed.

    print("\n=== Step 4: IrrMapper Export ===")
    # Use the fields from REM workflow (already clipped to bounds)
    irr_fields = results["fields"]
    dev_test_irrmapper(config=config, fields=irr_fields)
    print(f"  Export started for {len(irr_fields)} fields - check EE task manager")

    # =========================================================================
    # Step 5: Pattern Selection (requires IrrMapper CSV)
    # =========================================================================
    # NOTE: Uncomment after IrrMapper export completes and CSV is downloaded.

    print("\n=== Step 5: Pattern Selection ===")
    irrmapper_csv = cfg.get("irrmapper_csv")
    if irrmapper_csv and os.path.exists(irrmapper_csv):
        # Load and inspect IrrMapper data
        irr_stats, summary = dev_test_load_irrmapper(irrmapper_csv, config.feature_id)
        print(f"  IrrMapper summary: {summary}")

        # Assign patterns
        fields_with_pattern = dev_test_pattern(
            config=config,
            fields=fields_stratified,
            irrmapper_csv=irrmapper_csv,
        )
        pattern_path = os.path.join(config.out_dir, "fields_pattern.fgb")
        fields_with_pattern.to_file(pattern_path, driver="FlatGeobuf")
        print(f"  Pattern fields: {fields_with_pattern['pattern'].sum()} / {len(fields_with_pattern)}")
    else:
        print("  Skipped - irrmapper_csv not found")

    # =========================================================================
    # Step 6: Visualization
    # =========================================================================
    print("\n=== Step 6: Visualization ===")
    out_html = dev_test_viz(
        config=config,
        results=results,
        bounds_wsen=bounds_wsen,
        ndwi_threshold=ndwi_threshold,
    )
    print(f"  Debug map: {out_html}")

    # =========================================================================
    # ET Workflow Steps (uncomment as needed)
    # =========================================================================
    # print("\n=== GridMET Download ===")
    # dev_test_met(config=config, overwrite=False)
    #
    # print("\n=== PT-JPL Export ===")
    # dev_test_et(config=config)
    #
    # print("\n=== ET Join ===")
    # dev_test_join(config=config)
    #
    # print("\n=== ET Partition ===")
    # dev_test_partition(config=config)

    print("\n=== Done ===")
    print(f"Output directory: {config.out_dir}")

    return results, fields_stratified


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "12")
    os.environ.setdefault("OMP_THREAD_LIMIT", "12")
    results, fields = main()
