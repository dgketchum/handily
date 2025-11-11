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

    # run subcommand (Klamath-aligned stratification)
    p_run = sub.add_parser("run", help="Run REM/HAND stratification and write a Leaflet debug map")
    p_run.add_argument("--huc10", required=True, help="Target HUC-10 ID (e.g., 1002000207)")
    p_run.add_argument("--fields", required=True, help="Path to irrigation fields dataset (SHP/GPKG/etc.)")
    p_run.add_argument("--out-dir", required=True, help="Output directory for results")
    # DEM cache control
    p_run.add_argument("--overwrite-dem", action="store_true", help="Overwrite cached DEM if it exists")
    # STAC (required): use local 3DEP STAC for LiDAR tiles
    p_run.add_argument("--stac-dir", required=True, help="Path to local 3DEP STAC catalog (required)")
    p_run.add_argument("--stac-collection-id", default=os.environ.get("HANDILY_STAC_COLLECTION", "usgs-3dep-1m-opr"), help="STAC Collection ID (default: usgs-3dep-1m-opr)")
    p_run.add_argument("--stac-download-cache-dir", default=os.environ.get("HANDILY_STAC_CACHE_DIR"), help="Directory to cache downloaded STAC GeoTIFF tiles")
    # Outputs
    p_run.add_argument("--no-save-rem", dest="save_rem", action="store_false", default=True, help="Do not write the REM GeoTIFF")
    p_run.add_argument("--save-intermediates", action="store_true", default=False, help="Also write AOI/flowlines/streams intermediate files")
    p_run.add_argument("--wbd-local-dir", default=os.environ.get("HANDILY_WBD_DIR"), help="Path to local WBD state shapefile folder or WBDHU10.shp (required)")
    p_run.add_argument("--flowlines-local-dir", default=os.environ.get("HANDILY_NHD_DIR"), help="Path to local NHD state shapefile folder (defaults to --wbd-local-dir)")

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

    if args.cmd == "run":
        from handily.core import run_hand_stratification, ensure_dir, LOGGER as CORE_LOGGER
        from handily.viz import write_interactive_map

        ensure_dir(args.out_dir)
        CORE_LOGGER.info("Output directory: %s", args.out_dir)
        CORE_LOGGER.info("Starting REM/HAND stratification for HUC-10 %s", args.huc10)
        results = run_hand_stratification(
            huc10=str(args.huc10),
            fields_path=os.path.expanduser(args.fields),
            out_dir=args.out_dir,
            save_rem=bool(args.save_rem),
            save_intermediates=bool(args.save_intermediates),
            overwrite_dem=bool(args.overwrite_dem),
            wbd_local_dir=args.wbd_local_dir,
            flowlines_local_dir=(args.flowlines_local_dir or args.wbd_local_dir),
            stac_dir=args.stac_dir,
            stac_collection_id=args.stac_collection_id,
            stac_download_cache_dir=args.stac_download_cache_dir,
        )
        out_html = os.path.join(args.out_dir, "debug_map.html")
        write_interactive_map(results, out_html, initial_threshold=2.0)
        summary = results.get("summary", {})
        total_fields = summary.get("total_fields")
        partitioned = summary.get("partitioned")
        threshold_m = summary.get("threshold_m")
        print("--- Stratification Summary ---")
        print(f"HUC-10: {args.huc10}")
        print(f"Fields total: {total_fields}")
        print(f"Partitioned (< {threshold_m} m): {partitioned}")
        print("Outputs:")
        if results.get("rem_path"):
            print(f"  REM: {results['rem_path']}")
        if results.get("fields_out_gpkg"):
            print(f"  Stratified fields (GPKG): {results['fields_out_gpkg']}")
        if results.get("fields_out_shp"):
            print(f"  Stratified fields (SHP): {results['fields_out_shp']}")
        print(f"  Map: {out_html}")
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
