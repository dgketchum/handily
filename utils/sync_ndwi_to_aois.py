#!/usr/bin/env python
"""Sync NDWI tiles from GCS to per-AOI directories.

Downloads from a flat GCS prefix, then distributes each tile into its
matching aoi_{id:04d}/ directory based on the AOI index in the filename.

Usage:
    uv run python utils/sync_ndwi_to_aois.py \
        --bucket wudr \
        --prefix handily/nv/ndwi/naip_ndwi_aoi \
        --out-root /data/ssd2/handily/nv
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(
        description="Sync NDWI tiles from GCS to per-AOI dirs"
    )
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument(
        "--prefix",
        required=True,
        help="GCS prefix (e.g. handily/nv/ndwi/naip_ndwi_aoi)",
    )
    parser.add_argument(
        "--out-root", required=True, help="Local root with aoi_XXXX dirs"
    )
    parser.add_argument(
        "--staging-dir", default=None, help="Temp staging dir (default: auto)"
    )
    args = parser.parse_args()

    out_root = os.path.expanduser(args.out_root)

    # Stage 1: gsutil rsync to a staging dir
    staging = args.staging_dir or tempfile.mkdtemp(prefix="ndwi_staging_")
    os.makedirs(staging, exist_ok=True)
    gcs_path = f"gs://{args.bucket}/{args.prefix}/"
    print(f"Syncing {gcs_path} → {staging}/")
    rc = subprocess.call(["gsutil", "-m", "rsync", gcs_path, staging + "/"])
    if rc != 0:
        print(f"gsutil rsync failed (exit {rc})", file=sys.stderr)
        sys.exit(rc)

    # Stage 2: distribute tiles into per-AOI dirs
    tifs = glob.glob(os.path.join(staging, "*.tif"))
    if not tifs:
        print("No .tif files found after sync", file=sys.stderr)
        sys.exit(1)

    moved = 0
    for tif in tifs:
        basename = os.path.basename(tif)
        # Extract AOI index from filename, e.g. naip_ndwi_aoi_0773.tif → 0773
        m = re.search(r"_(\d{4,})\.tif$", basename)
        if not m:
            m = re.search(r"(\d{4,})\.tif$", basename)
        if not m:
            print(f"  Skipping {basename} (no AOI index found)")
            continue

        aoi_id = m.group(1)
        aoi_dir = os.path.join(out_root, f"aoi_{aoi_id}")
        if not os.path.isdir(aoi_dir):
            os.makedirs(aoi_dir, exist_ok=True)

        dest = os.path.join(aoi_dir, basename)
        shutil.move(tif, dest)
        moved += 1

    print(f"Distributed {moved} NDWI tiles into {out_root}/aoi_*/")

    # Clean up staging
    if args.staging_dir is None:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
