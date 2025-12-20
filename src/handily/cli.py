#!/usr/bin/env python3
import os
import sys
import logging
import argparse


def configure_logging(verbosity: int) -> None:
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="handily", description="Handily CLI: REM/HAND pipeline and 3DEP STAC tools")
    parser.add_argument("-v", action="count", default=0, help="Increase verbosity")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # bounds subcommand (bounds + NHD + NDWI quick REM)
    p_bounds = sub.add_parser("bounds", help="Build REM within bounds using NHD flowlines + NAIP NDWI")
    p_bounds.add_argument("--bounds", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"), help="Bounds in EPSG:4326: W S E N")
    p_bounds.add_argument("--fields", required=True, help="Path to irrigation fields dataset (SHP/GPKG/etc.)")
    p_bounds.add_argument("--ndwi-dir", required=True, help="Directory containing local NDWI GeoTIFFs")
    p_bounds.add_argument("--flowlines-local-dir", required=True, help="Path to local NHD state shapefile folder")
    p_bounds.add_argument("--stac-dir", required=True, help="Path to local 3DEP STAC catalog (required)")
    p_bounds.add_argument("--ndwi-threshold", type=float, default=0.15, help="NDWI threshold for water masking (default 0.15)")
    p_bounds.add_argument("--out-dir", required=True, help="Output directory for results")

    p_aoi = sub.add_parser("aoi", help="Build buffered-field AOI tiles and write a shapefile")
    p_aoi.add_argument("--fields", required=True, help="Path to statewide irrigation dataset")
    p_aoi.add_argument("--out-shp", required=True, help="Output AOI shapefile path")
    p_aoi.add_argument("--max-km2", type=float, default=625.0, help="Maximum AOI tile area in square kilometers")
    p_aoi.add_argument("--buffer-m", type=float, default=1000.0, help="Centroid buffer radius in meters")
    p_aoi.add_argument("--bounds", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="Optional bounds in EPSG:4326 (W S E N)")
    p_aoi.add_argument("--simplify-m", type=float, default=None, help="Optional simplify tolerance in meters")

    # stac subcommand with nested build/extend
    p_stac = sub.add_parser("stac", help="3DEP 1 m STAC tools")
    stac_sub = p_stac.add_subparsers(dest="stac_cmd", required=True)

    p_build = stac_sub.add_parser("build", help="Create a new STAC catalog from TNM S3 index")
    p_build.add_argument("--out-dir", required=True, help="Output directory for the STAC catalog")
    p_build.add_argument("--states", nargs="*", help="State abbreviations to include (e.g., MT ID). If omitted, all projects.")
    p_build.add_argument("--collection-id", default="usgs-3dep-1m-opr", help="Collection ID (default: usgs-3dep-1m-opr)")

    p_extend = stac_sub.add_parser("extend", help="Extend an existing STAC catalog with more states")
    p_extend.add_argument("--out-dir", required=True, help="Existing STAC catalog directory")
    p_extend.add_argument("--states", nargs="*", required=True, help="Additional state abbreviations (e.g., WA OR)")
    p_extend.add_argument("--collection-id", default="usgs-3dep-1m-opr", help="Collection ID (default: usgs-3dep-1m-opr)")

    p_et = sub.add_parser("et", help="ET workflows (PT-JPL)")
    et_sub = p_et.add_subparsers(dest="et_cmd", required=True)

    p_et_export = et_sub.add_parser("export", help="Export PT-JPL capture-date zonal stats to GCS")
    p_et_export.add_argument("--config", required=True, help="YAML config path")

    p_et_join = et_sub.add_parser("join", help="Join local PT-JPL CSVs with GridMET parquet and write joined parquet")
    p_et_join.add_argument("--config", required=True, help="YAML config path")

    p_met = sub.add_parser("met", help="Meteorology workflows (GridMET)")
    met_sub = p_met.add_subparsers(dest="met_cmd", required=True)

    p_met_download = met_sub.add_parser("download", help="Download GridMET time series at field centroids")
    p_met_download.add_argument("--config", required=True, help="YAML config path")

    p_partition = sub.add_parser("partition", help="Partition ET into subsurface and irrigation components")
    p_partition.add_argument("--config", required=True, help="YAML config path")

    args = parser.parse_args(argv)
    configure_logging(args.v)

    if args.cmd == "bounds":
        from handily.viz import write_interactive_map

        ensure_dir(args.out_dir)  # BUG?: ensure_dir/run_bounds_rem/CORE_LOGGER not defined in this module
        CORE_LOGGER.info("Output directory: %s", args.out_dir)
        CORE_LOGGER.info("Building REM within bounds: %s", args.bounds)
        results = run_bounds_rem(
            bounds_wsen=tuple(args.bounds),
            fields_path=os.path.expanduser(args.fields),
            ndwi_dir=os.path.expanduser(args.ndwi_dir),
            stac_dir=os.path.expanduser(args.stac_dir),
            flowlines_local_dir=os.path.expanduser(args.flowlines_local_dir),
            out_dir=args.out_dir,
            ndwi_threshold=float(args.ndwi_threshold),
        )
        out_html = os.path.join(args.out_dir, "debug_map.html")
        write_interactive_map(results, out_html, initial_threshold=2.0)
        # Persist QA rasters
        rem_path = os.path.join(args.out_dir, "rem_bounds.tif")
        streams_path = os.path.join(args.out_dir, "streams_bounds.tif")
        results["rem"].rio.to_raster(rem_path)
        results["streams"].rio.to_raster(streams_path)
        # Persist QA vectors
        fields_gpkg = os.path.join(args.out_dir, "fields_bounds.gpkg")
        results["fields"].to_file(fields_gpkg, driver="GPKG")

        print("--- Bounds REM Summary ---")
        print(f"Fields total: {results['summary'].get('total_fields')}")
        print(f"NDWI threshold: {results['summary'].get('ndwi_threshold')}")
        print(f"Map: {out_html}")
        print(f"REM GeoTIFF: {rem_path}")
        print(f"Streams mask GeoTIFF: {streams_path}")
        print(f"Fields GPKG: {fields_gpkg}")
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
            root = build_3dep_stac(args.out_dir, states=states or None, collection_id=args.collection_id)
            print(f"STAC catalog written: {root}")
            return 0
        elif args.stac_cmd == "extend":
            states = parse_states(args.states)
            root = extend_3dep_stac(args.out_dir, states=states, collection_id=args.collection_id)
        print(f"STAC catalog updated: {root}")
        return 0

    if args.cmd == "met":
        from handily.config import HandilyConfig
        from handily.et.gridmet import download_gridmet

        config = HandilyConfig.from_yaml(args.config)
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

        config = HandilyConfig.from_yaml(args.config)
        bounds_wsen = None
        if config.bounds:
            bounds_wsen = tuple(config.bounds)

        if args.et_cmd == "export":
            from handily.et.image_export import export_ptjpl_et_fraction

            export_ptjpl_et_fraction(
                config.fields_path,
                config.et_bucket,
                feature_id=config.feature_id,
                select=None,
                start_yr=config.ptjpl_start_yr,
                end_yr=config.ptjpl_end_yr,
                overwrite=False,
                check_dir=config.ptjpl_check_dir,
                buffer=None,
                bounds_wsen=bounds_wsen,
                cloud_cover_max=70,
                landsat_collections=None,
            )
            return 0

        if args.et_cmd == "join":
            from handily.et.join import join_gridmet_ptjpl

            join_gridmet_ptjpl(
                config.gridmet_parquet_dir,
                config.ptjpl_csv_dir,
                config.et_join_parquet_dir,
                ptjpl_csv_template=config.ptjpl_csv_template,
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

        config = HandilyConfig.from_yaml(args.config)
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

    parser.print_help()
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logging.getLogger("handily.cli").exception("Unhandled error: %s", exc)
        sys.exit(1)
