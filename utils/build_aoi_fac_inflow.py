"""Build 1m FAC stream network for an AOI with regional inflow injection.

Usage:
    uv run python utils/build_aoi_fac_inflow.py \
        --aoi-dir /data/ssd2/handily/mt/aoi_0009 \
        --regional-dir /data/ssd2/handily/mt/regional/missouri_headwaters \
        --threshold 50000 --max-procs 32
"""

import argparse
import logging

from handily.regional_fac import compute_aoi_fac_with_inflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Build 1m FAC with regional inflow injection"
    )
    parser.add_argument(
        "--aoi-dir", required=True, help="AOI directory with dem_bounds_1m.tif"
    )
    parser.add_argument("--regional-dir", required=True, help="Regional FAC directory")
    parser.add_argument(
        "--threshold", type=int, default=50_000, help="FAC cell threshold"
    )
    parser.add_argument("--max-procs", type=int, default=32, help="WhiteboxTools cores")
    parser.add_argument(
        "--min-inflow-acc", type=int, default=1000, help="Min regional acc for inflow"
    )
    args = parser.parse_args()

    dem_path = f"{args.aoi_dir}/dem_bounds_1m.tif"
    regional_fac = f"{args.regional_dir}/flow_accumulation.tif"
    regional_d8 = f"{args.regional_dir}/d8_pointer.tif"
    out_dir = args.aoi_dir

    compute_aoi_fac_with_inflow(
        dem_path=dem_path,
        regional_fac_path=regional_fac,
        regional_d8_path=regional_d8,
        out_dir=out_dir,
        threshold=args.threshold,
        max_procs=args.max_procs,
        min_inflow_acc=args.min_inflow_acc,
    )


if __name__ == "__main__":
    main()
