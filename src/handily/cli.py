#!/usr/bin/env python3
import os
import sys
import logging
import argparse


def configure_logging(verbosity: int) -> None:
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="handily", description="Handily CLI: REM/HAND pipeline and 3DEP STAC tools"
    )
    parser.add_argument("-v", action="count", default=0, help="Increase verbosity")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # bounds subcommand (bounds + NHD + NDWI quick REM)
    p_bounds = sub.add_parser(
        "bounds", help="Build REM within bounds using NHD flowlines + NAIP NDWI"
    )
    p_bounds.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        required=True,
        metavar=("W", "S", "E", "N"),
        help="Bounds in EPSG:4326: W S E N",
    )
    p_bounds.add_argument(
        "--fields",
        required=True,
        help="Path to irrigation fields dataset (SHP/GPKG/etc.)",
    )
    p_bounds.add_argument(
        "--ndwi-dir", required=True, help="Directory containing local NDWI GeoTIFFs"
    )
    p_bounds.add_argument(
        "--flowlines-local-dir",
        required=True,
        help="Path to local NHD state shapefile folder",
    )
    p_bounds.add_argument(
        "--stac-dir", required=True, help="Path to local 3DEP STAC catalog (required)"
    )
    p_bounds.add_argument(
        "--ndwi-threshold",
        type=float,
        default=0.15,
        help="NDWI threshold for water masking (default 0.15)",
    )
    p_bounds.add_argument(
        "--flowlines-buffer",
        type=float,
        default=None,
        help="Buffer NHD flowlines by this many meters before AND with NDWI",
    )
    p_bounds.add_argument(
        "--out-dir", required=True, help="Output directory for results"
    )

    p_aoi = sub.add_parser(
        "aoi", help="Build buffered-field AOI tiles and write a shapefile"
    )
    p_aoi.add_argument(
        "--fields", required=True, help="Path to statewide irrigation dataset"
    )
    p_aoi.add_argument("--out-shp", required=True, help="Output AOI shapefile path")
    p_aoi.add_argument(
        "--max-km2",
        type=float,
        default=625.0,
        help="Maximum AOI tile area in square kilometers",
    )
    p_aoi.add_argument(
        "--buffer-m",
        type=float,
        default=1000.0,
        help="Centroid buffer radius in meters",
    )
    p_aoi.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        help="Optional bounds in EPSG:4326 (W S E N)",
    )
    p_aoi.add_argument(
        "--simplify-m",
        type=float,
        default=None,
        help="Optional simplify tolerance in meters",
    )

    # stac subcommand with nested build/extend
    p_stac = sub.add_parser("stac", help="3DEP 1 m STAC tools")
    stac_sub = p_stac.add_subparsers(dest="stac_cmd", required=True)

    p_build = stac_sub.add_parser(
        "build", help="Create a new STAC catalog from TNM S3 index"
    )
    p_build.add_argument(
        "--out-dir", required=True, help="Output directory for the STAC catalog"
    )
    p_build.add_argument(
        "--states",
        nargs="*",
        help="State abbreviations to include (e.g., MT ID). If omitted, all projects.",
    )
    p_build.add_argument(
        "--collection-id",
        default="usgs-3dep-1m-opr",
        help="Collection ID (default: usgs-3dep-1m-opr)",
    )

    p_extend = stac_sub.add_parser(
        "extend", help="Extend an existing STAC catalog with more states"
    )
    p_extend.add_argument(
        "--out-dir", required=True, help="Existing STAC catalog directory"
    )
    p_extend.add_argument(
        "--states",
        nargs="*",
        required=True,
        help="Additional state abbreviations (e.g., WA OR)",
    )
    p_extend.add_argument(
        "--collection-id",
        default="usgs-3dep-1m-opr",
        help="Collection ID (default: usgs-3dep-1m-opr)",
    )

    p_rem = sub.add_parser("rem", help="REM batch workflows")
    rem_sub = p_rem.add_subparsers(dest="rem_cmd", required=True)

    p_rem_fetch = rem_sub.add_parser(
        "fetch-dem", help="Download and cache DEMs for every AOI without computing REM"
    )
    p_rem_fetch.add_argument(
        "--aoi-shp", required=True, help="AOI shapefile produced by 'handily aoi'"
    )
    p_rem_fetch.add_argument("--config", required=True, help="TOML config path")
    p_rem_fetch.add_argument(
        "--out-root", required=True, help="Root directory for per-AOI outputs"
    )
    p_rem_fetch.add_argument(
        "--overwrite", action="store_true", help="Re-download existing DEMs"
    )
    p_rem_fetch.add_argument(
        "--coverage-col",
        default=None,
        help="Shapefile column to filter on; only rows where column == 1 are processed",
    )
    p_rem_fetch.add_argument(
        "--sort-col",
        default=None,
        help="Sort AOIs descending by this column before processing (e.g. n_fields)",
    )
    p_rem_fetch.add_argument(
        "--aoi-ids",
        nargs="+",
        type=int,
        default=None,
        help="Only process these aoi_id values (e.g. --aoi-ids 7 8 9 10)",
    )

    p_rem_batch = rem_sub.add_parser(
        "batch", help="Run REM pipeline for every AOI in a shapefile"
    )
    p_rem_batch.add_argument(
        "--aoi-shp", required=True, help="AOI shapefile produced by 'handily aoi'"
    )
    p_rem_batch.add_argument("--config", required=True, help="TOML config path")
    p_rem_batch.add_argument(
        "--out-root", required=True, help="Root directory for per-AOI outputs"
    )
    p_rem_batch.add_argument(
        "--ndwi-threshold",
        type=float,
        default=0.15,
        help="NDWI threshold for water masking (default 0.15)",
    )
    p_rem_batch.add_argument(
        "--flowlines-buffer",
        type=float,
        default=None,
        help="Buffer NHD flowlines by this many meters before AND with NDWI",
    )
    p_rem_batch.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute AOIs that already have rem_bounds.tif",
    )
    p_rem_batch.add_argument(
        "--coverage-col",
        default=None,
        help="Shapefile column to filter on; only rows where column == 1 are processed (e.g. stac_1m)",
    )
    p_rem_batch.add_argument(
        "--sort-col",
        default=None,
        help="Sort AOIs descending by this column before processing (e.g. n_fields)",
    )
    p_rem_batch.add_argument(
        "--aoi-ids",
        nargs="+",
        type=int,
        default=None,
        help="Only process these aoi_id values (e.g. --aoi-ids 7 8 9 10)",
    )

    p_nhd = sub.add_parser("nhd", help="NHD flowline preprocessing")
    nhd_sub = p_nhd.add_subparsers(dest="nhd_cmd", required=True)

    p_nhd_build = nhd_sub.add_parser(
        "build-state",
        help="Merge and filter state NHDFlowline shapefiles into a single FlatGeobuf",
    )
    p_nhd_build.add_argument(
        "--nhd-dir",
        required=True,
        help="NHD state Shape directory containing NHDFlowline*.shp files",
    )
    p_nhd_build.add_argument(
        "--out",
        default=None,
        help="Output .fgb path (default: <nhd-dir>/NHDFlowline_filtered.fgb)",
    )

    p_ndwi = sub.add_parser("ndwi", help="NAIP NDWI Earth Engine export")
    ndwi_sub = p_ndwi.add_subparsers(dest="ndwi_cmd", required=True)

    p_ndwi_export = ndwi_sub.add_parser(
        "export", help="Batch-submit NAIP NDWI export tasks to Earth Engine"
    )
    p_ndwi_export.add_argument(
        "--aoi-shp", required=True, help="AOI shapefile produced by 'handily aoi'"
    )
    p_ndwi_export.add_argument("--bucket", required=True, help="GCS bucket name")
    p_ndwi_export.add_argument(
        "--prefix",
        required=True,
        help="GCS path prefix for exported files (e.g. handily/mt/ndwi/naip_ndwi_aoi)",
    )
    p_ndwi_export.add_argument(
        "--ee-project", required=True, help="Earth Engine project ID (e.g. ee-username)"
    )
    p_ndwi_export.add_argument("--start-date", default="2010-01-01")
    p_ndwi_export.add_argument("--end-date", default="2024-12-31")
    p_ndwi_export.add_argument(
        "--skip-dir",
        default=None,
        help="Local NDWI directory; AOIs with existing .tif are skipped",
    )
    p_ndwi_export.add_argument(
        "--coverage-col",
        default=None,
        help="Shapefile column to filter on (exports only rows where column == 1)",
    )

    p_et = sub.add_parser("et", help="ET workflows (OpenET v2.0 ensemble)")
    et_sub = p_et.add_subparsers(dest="et_cmd", required=True)

    p_et_export = et_sub.add_parser(
        "export",
        help="Export OpenET v2.0 ensemble ET zonal means for field polygons to GCS",
    )
    p_et_export.add_argument("--config", required=True, help="TOML config path")

    p_et_join = et_sub.add_parser(
        "join", help="Join OpenET CSV with GridMET parquet and write joined parquet"
    )
    p_et_join.add_argument("--config", required=True, help="TOML config path")

    p_met = sub.add_parser("met", help="Meteorology workflows (GridMET)")
    met_sub = p_met.add_subparsers(dest="met_cmd", required=True)

    p_met_download = met_sub.add_parser(
        "download", help="Download GridMET time series at field centroids"
    )
    p_met_download.add_argument("--config", required=True, help="TOML config path")

    p_partition = sub.add_parser(
        "partition", help="Partition ET into subsurface and irrigation components"
    )
    p_partition.add_argument("--config", required=True, help="TOML config path")

    p_points = sub.add_parser("points", help="Points workflows for donor discovery")
    points_sub = p_points.add_subparsers(dest="points_cmd", required=True)

    p_points_sample = points_sub.add_parser(
        "sample", help="Generate AOI-scoped sample points from REM outputs"
    )
    p_points_sample.add_argument("--config", required=True, help="TOML config path")

    p_points_export = points_sub.add_parser(
        "export", help="Export point-based EE products from sampled points"
    )
    p_points_export.add_argument("--config", required=True, help="TOML config path")
    p_points_export.add_argument(
        "--product",
        default="all",
        choices=["all", "irrmapper", "ndvi", "openet_eta"],
        help="EE product to export (default: all)",
    )
    p_points_export.add_argument(
        "--year-start",
        type=int,
        default=None,
        help="Optional export start year override",
    )
    p_points_export.add_argument(
        "--year-end", type=int, default=None, help="Optional export end year override"
    )
    p_points_export.add_argument(
        "--dest",
        choices=["bucket", "drive"],
        default=None,
        help="Optional export destination override",
    )

    # Bucket sync subcommand
    p_sync = sub.add_parser(
        "sync", help="Sync EE exports from GCS bucket to local filesystem"
    )
    p_sync.add_argument("--config", required=True, help="TOML config path")
    p_sync.add_argument(
        "--subdir",
        default="irrmapper",
        help="Subdirectory to sync (default: irrmapper)",
    )
    p_sync.add_argument("--glob", default="*", help="Glob pattern to filter files")
    p_sync.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing local files"
    )
    p_sync.add_argument(
        "--dry-run", action="store_true", help="Print files without copying"
    )

    # QGIS integration subcommand
    p_qgis = sub.add_parser("qgis", help="QGIS project integration")
    qgis_sub = p_qgis.add_subparsers(dest="qgis_cmd", required=True)

    p_qgis_update = qgis_sub.add_parser(
        "update", help="Update QGIS project with handily output layers"
    )
    p_qgis_update.add_argument("--config", required=True, help="TOML config path")
    p_qgis_update.add_argument(
        "--project", help="Path to QGIS project (.qgz), overrides config"
    )
    p_qgis_update.add_argument("--group", help="Layer group name, overrides config")

    p_qgis_qlr = qgis_sub.add_parser("qlr", help="Generate QLR file for manual import")
    p_qgis_qlr.add_argument("--config", required=True, help="TOML config path")
    p_qgis_qlr.add_argument(
        "--output", help="Output QLR path (default: <out_dir>/handily.qlr)"
    )

    p_qgis_open = qgis_sub.add_parser("open", help="Open QGIS with the project")
    p_qgis_open.add_argument(
        "--project", required=True, help="Path to QGIS project (.qgz)"
    )

    args = parser.parse_args(argv)
    configure_logging(args.v)

    if args.cmd == "bounds":
        from handily.config import HandilyConfig
        from handily.io import aoi_from_bounds, ensure_dir
        from handily.pipeline import REMWorkflow

        logger = logging.getLogger("handily.cli")
        ensure_dir(args.out_dir)
        logger.info("Output directory: %s", args.out_dir)
        logger.info("Building REM within bounds: %s", args.bounds)

        # Build config from CLI args
        config = HandilyConfig(
            out_dir=args.out_dir,
            flowlines_local_dir=os.path.expanduser(args.flowlines_local_dir),
            ndwi_dir=os.path.expanduser(args.ndwi_dir),
            stac_dir=os.path.expanduser(args.stac_dir),
            fields_path=os.path.expanduser(args.fields),
            bounds=list(args.bounds),
        )

        aoi = aoi_from_bounds(tuple(args.bounds))
        workflow = REMWorkflow(config=config, aoi=aoi)
        results = workflow.run(
            ndwi_threshold=float(args.ndwi_threshold),
            stats=("mean",),
            cache_flowlines=True,
            flowlines_buffer_m=args.flowlines_buffer,
        )

        # Persist QA rasters
        rem_path = os.path.join(args.out_dir, "rem_bounds.tif")
        streams_path = os.path.join(args.out_dir, "streams_bounds.tif")
        results["rem"].rio.to_raster(rem_path)
        results["streams"].rio.to_raster(streams_path)
        # Persist QA vectors
        fields_fgb = os.path.join(args.out_dir, "fields_bounds.fgb")
        fields = results.get("fields_stats") or results["fields"]
        if "FID" in fields.columns:
            fields = fields.rename(columns={"FID": "FID_"})
        fields.to_file(fields_fgb, driver="FlatGeobuf")

        print("--- Bounds REM Summary ---")
        print(f"Fields total: {results['summary'].get('total_fields')}")
        print(f"NDWI threshold: {results['summary'].get('ndwi_threshold')}")
        print(f"REM GeoTIFF: {rem_path}")
        print(f"Streams mask GeoTIFF: {streams_path}")
        print(f"Fields FGB: {fields_fgb}")
        print("Run 'handily qgis update --config <config.toml>' to add layers to QGIS")
        return 0

    if args.cmd == "aoi":
        from handily.aoi_split import build_centroid_buffer_aois, write_aois_shapefile

        bounds = tuple(args.bounds) if args.bounds else None
        tiles = build_centroid_buffer_aois(
            fields_path=os.path.expanduser(args.fields),
            max_km2=float(args.max_km2),
            buffer_m=float(args.buffer_m),
            bounds_wsen=bounds,
            simplify_tolerance_m=args.simplify_m,
        )
        out_shp = os.path.expanduser(args.out_shp)
        write_aois_shapefile(tiles, out_shp)
        print(f"AOI tiles written: {out_shp}")
        return 0

    if args.cmd == "stac":
        from handily.stac_3dep import build_3dep_stac, extend_3dep_stac

        def parse_states(values):
            states = []
            for v in values or []:
                parts = [p for p in v.replace(",", " ").split() if p]
                states.extend(parts)
            # unique uppercase
            seen = set()
            uniq = []
            for s in [s.upper() for s in states]:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
            return uniq

        if args.stac_cmd == "build":
            states = parse_states(args.states)
            root = build_3dep_stac(
                args.out_dir, states=states or None, collection_id=args.collection_id
            )
            print(f"STAC catalog written: {root}")
            return 0
        elif args.stac_cmd == "extend":
            states = parse_states(args.states)
            root = extend_3dep_stac(
                args.out_dir, states=states, collection_id=args.collection_id
            )
            print(f"STAC catalog updated: {root}")
            return 0

    if args.cmd == "rem":
        import geopandas as gpd
        from handily.config import HandilyConfig
        from handily.pipeline import batch_fetch_dem, batch_run_rem

        config = HandilyConfig.from_toml(args.config)
        aoi_gdf = gpd.read_file(os.path.expanduser(args.aoi_shp))
        if args.coverage_col and args.coverage_col in aoi_gdf.columns:
            n_before = len(aoi_gdf)
            aoi_gdf = aoi_gdf[aoi_gdf[args.coverage_col] == 1].reset_index(drop=True)
            print(f"Filtered to {args.coverage_col}==1: {len(aoi_gdf)}/{n_before} AOIs")
        if args.aoi_ids and "aoi_id" in aoi_gdf.columns:
            n_before = len(aoi_gdf)
            aoi_gdf = aoi_gdf[aoi_gdf["aoi_id"].isin(args.aoi_ids)].reset_index(
                drop=True
            )
            print(f"Filtered to {len(aoi_gdf)}/{n_before} AOIs by aoi_id")
        if args.sort_col and args.sort_col in aoi_gdf.columns:
            aoi_gdf = aoi_gdf.sort_values(args.sort_col, ascending=False).reset_index(
                drop=True
            )
            print(f"Sorted by {args.sort_col} descending")

        if args.rem_cmd == "fetch-dem":
            results = batch_fetch_dem(
                aoi_gdf=aoi_gdf,
                config=config,
                out_root=os.path.expanduser(args.out_root),
                overwrite=args.overwrite,
            )
            done = sum(1 for r in results if r["status"] == "done")
            skipped = sum(1 for r in results if r["status"] == "skipped")
            errors = sum(1 for r in results if r["status"] == "error")
            print(f"Batch DEM fetch: done={done} skipped={skipped} errors={errors}")
            for r in results:
                if r["status"] == "error":
                    print(f"  ERROR aoi_{r['aoi_id']:04d}: {r.get('error')}")
            return 0

        results = batch_run_rem(
            aoi_gdf=aoi_gdf,
            config=config,
            out_root=os.path.expanduser(args.out_root),
            ndwi_threshold=float(args.ndwi_threshold),
            flowlines_buffer_m=args.flowlines_buffer,
            overwrite=args.overwrite,
        )
        done = sum(1 for r in results if r["status"] == "done")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        errors = sum(1 for r in results if r["status"] == "error")
        print(f"Batch complete: done={done} skipped={skipped} errors={errors}")
        for r in results:
            if r["status"] == "error":
                print(f"  ERROR aoi_{r['aoi_id']:04d}: {r.get('error')}")
        return 0

    if args.cmd == "nhd":
        from handily.io import build_state_flowlines

        if args.nhd_cmd == "build-state":
            out = build_state_flowlines(
                os.path.expanduser(args.nhd_dir),
                out_path=os.path.expanduser(args.out) if args.out else None,
            )
            print(f"Flowlines FGB written: {out}")
            return 0

    if args.cmd == "ndwi":
        import ee
        from handily.ndwi_export import export_ndwi_for_polygons

        ee.Initialize(project=args.ee_project)
        tasks = export_ndwi_for_polygons(
            aoi_shapefile=os.path.expanduser(args.aoi_shp),
            bucket=args.bucket,
            prefix=args.prefix,
            start_date=args.start_date,
            end_date=args.end_date,
            skip_if_present_dir=os.path.expanduser(args.skip_dir)
            if args.skip_dir
            else None,
            coverage_col=args.coverage_col,
        )
        print(f"Submitted {len(tasks)} EE export tasks")
        return 0

    if args.cmd == "met":
        from handily.config import HandilyConfig
        from handily.et.gridmet import download_gridmet

        config = HandilyConfig.from_toml(args.config)
        bounds_wsen = None
        if config.bounds:
            bounds_wsen = tuple(config.bounds)
        download_gridmet(
            config.fields_path,
            config.gridmet_parquet_dir,
            gridmet_centroids_path=config.gridmet_centroids_path,
            gridmet_centroid_parquet_dir=config.gridmet_centroid_parquet_dir,
            bounds_wsen=bounds_wsen,
            start=config.met_start,
            end=config.met_end,
            overwrite=False,
            feature_id=config.feature_id,
            gridmet_id_col=config.gridmet_id_col,
            return_df=False,
        )
        return 0

    if args.cmd == "et":
        from handily.config import HandilyConfig

        config = HandilyConfig.from_toml(args.config)
        bounds_wsen = None
        if config.bounds:
            bounds_wsen = tuple(config.bounds)

        if args.et_cmd == "export":
            from handily.points.ee_extract import export_fields_openet_eta

            export_fields_openet_eta(
                config.fields_path,
                config,
                year_start=config.openet_start_yr,
                year_end=config.openet_end_yr,
                feature_id=config.feature_id,
            )
            return 0

        if args.et_cmd == "join":
            from handily.et.join import join_gridmet_openet_eta

            join_gridmet_openet_eta(
                config.gridmet_parquet_dir,
                config.openet_csv_path,
                config.et_join_parquet_dir,
                fields_path=config.fields_path,
                bounds_wsen=bounds_wsen,
                feature_id=config.feature_id,
                eto_col="eto",
                prcp_col="prcp",
            )
            return 0

    if args.cmd == "partition":
        from handily.config import HandilyConfig
        from handily.et.partition import partition_et

        config = HandilyConfig.from_toml(args.config)
        bounds_wsen = None
        if config.bounds:
            bounds_wsen = tuple(config.bounds)
        partition_et(
            config.fields_path,
            config.partition_joined_parquet_dir,
            config.partition_out_parquet_dir,
            feature_id=config.feature_id,
            strata_col=config.partition_strata_col,
            pattern_col=config.partition_pattern_col,
            bounds_wsen=bounds_wsen,
        )
        return 0

    if args.cmd == "points":
        from handily.config import HandilyConfig
        from handily.points.ee_extract import export_points_products_from_config
        from handily.points.sample import sample_points_from_config

        config = HandilyConfig.from_toml(args.config)

        if args.points_cmd == "sample":
            result = sample_points_from_config(config)
            print("--- Points Sample Summary ---")
            print(f"Points total: {result['n_points']}")
            print(f"Output directory: {result['out_dir']}")
            for group, count in result["group_counts"].items():
                print(f"  {group}: {count}")
            print(f"Points FGB: {result['paths']['fgb']}")
            print(f"Points parquet: {result['paths']['parquet']}")
            return 0

        if args.points_cmd == "export":
            results = export_points_products_from_config(
                config,
                product=args.product,
                year_start=args.year_start,
                year_end=args.year_end,
                dest=args.dest,
            )
            print("--- Points EE Export Summary ---")
            for result in results:
                print(f"{result['product']}: {result['description']}")
                print(f"  prefix: {result['prefix']}")
            return 0

    if args.cmd == "sync":
        from handily.bucket import sync_bucket_to_local
        from handily.config import HandilyConfig

        config = HandilyConfig.from_toml(args.config)

        if config.local_data_root is None:
            print("Error: local_data_root not set in config")
            return 1

        full_prefix = f"{config.bucket_prefix}/{config.project_name}"

        result = sync_bucket_to_local(
            bucket=config.et_bucket,
            bucket_prefix=full_prefix,
            local_root=config.local_data_root,
            subdir=args.subdir,
            glob_pattern=args.glob,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

        print(
            f"Sync complete: copied={result['copied']}, skipped={result['skipped']}, errors={result['errors']}"
        )
        return 0

    if args.cmd == "qgis":
        from handily.qgis import (
            discover_outputs,
            generate_qlr,
            open_project,
            update_project,
        )

        if args.qgis_cmd == "update":
            from handily.config import HandilyConfig

            config = HandilyConfig.from_toml(args.config)
            project_path = args.project or config.qgis_project
            group_name = args.group or config.qgis_layer_group

            if not project_path:
                print(
                    "Error: No QGIS project specified. Use --project or set qgis_project in config."
                )
                return 1

            layers = discover_outputs(config.out_dir, view_root=config.qgis_view_root)
            if not layers:
                print(f"No output layers found in {config.out_dir}")
                return 1

            update_project(project_path, layers, group_name)
            print(f"Updated QGIS project: {project_path}")
            print(f"  Group: {group_name}")
            print(f"  Layers: {len(layers)}")
            for layer in layers:
                print(f"    - {layer['name']} ({layer['type']})")
            return 0

        if args.qgis_cmd == "qlr":
            from handily.config import HandilyConfig

            config = HandilyConfig.from_toml(args.config)
            layers = discover_outputs(config.out_dir, view_root=config.qgis_view_root)
            if not layers:
                print(f"No output layers found in {config.out_dir}")
                return 1

            output_path = args.output or os.path.join(config.out_dir, "handily.qlr")
            generate_qlr(layers, output_path)
            print(f"Generated QLR: {output_path}")
            if config.qgis_view_root:
                print(f"Paths remapped for: {config.qgis_view_root}")
            print("Drag this file into QGIS to add all layers.")
            return 0

        if args.qgis_cmd == "open":
            open_project(args.project)
            print(f"Opened QGIS with: {args.project}")
            return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logging.getLogger("handily.cli").exception("Unhandled error: %s", exc)
        sys.exit(1)
