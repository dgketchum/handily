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
    p_bounds.add_argument("--ndwi-dir", required=True, help="Directory containing local NDWI tiles (by MGRS tile ID)")
    p_bounds.add_argument("--flowlines-local-dir", required=True, help="Path to local NHD state shapefile folder")
    p_bounds.add_argument("--stac-dir", required=True, help="Path to local 3DEP STAC catalog (required)")
    p_bounds.add_argument("--mgrs-shp", default=os.path.expanduser("~/data/IrrigationGIS/boundaries/mgrs/mgrs_aea.shp"), help="Path to MGRS AEA shapefile with MGRS_TILE column")
    p_bounds.add_argument("--ndwi-threshold", type=float, default=0.15, help="NDWI threshold for water masking (default 0.15)")
    p_bounds.add_argument("--out-dir", required=True, help="Output directory for results")

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

    args = parser.parse_args(argv)
    configure_logging(args.v)

    if args.cmd == "bounds":
        from handily.core import run_bounds_rem, ensure_dir, LOGGER as CORE_LOGGER
        from handily.viz import write_interactive_map

        ensure_dir(args.out_dir)
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
            mgrs_shp_path=os.path.expanduser(args.mgrs_shp),
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
        mgrs_gpkg = os.path.join(args.out_dir, "mgrs_tiles_bounds.gpkg")
        results["fields"].to_file(fields_gpkg, driver="GPKG")
        results["mgrs_tiles"].to_file(mgrs_gpkg, driver="GPKG")

        print("--- Bounds REM Summary ---")
        print(f"Fields total: {results['summary'].get('total_fields')}")
        print(f"NDWI threshold: {results['summary'].get('ndwi_threshold')}")
        print(f"Map: {out_html}")
        print(f"REM GeoTIFF: {rem_path}")
        print(f"Streams mask GeoTIFF: {streams_path}")
        print(f"Fields GPKG: {fields_gpkg}")
        print(f"MGRS tiles GPKG: {mgrs_gpkg}")
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

    parser.print_help()
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logging.getLogger("handily.cli").exception("Unhandled error: %s", exc)
        sys.exit(1)
