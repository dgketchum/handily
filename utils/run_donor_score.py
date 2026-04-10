"""Batch donor scoring across all AOIs in one or more state directories."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
LOGGER = logging.getLogger("run_donor_score")


def run_aoi(points_dir: str) -> dict:
    """Score donors for a single AOI points directory."""
    from handily.points.donor_score import (
        DonorConfig,
        score_donors_for_aoi,
        write_donor_outputs,
    )

    aoi_id = os.path.basename(os.path.dirname(points_dir))

    py_path = os.path.join(points_dir, "point_year.parquet")
    ps_path = os.path.join(points_dir, "point_summary.parquet")
    pt_path = os.path.join(points_dir, "points.parquet")

    for p in (py_path, ps_path, pt_path):
        if not os.path.exists(p):
            return {
                "aoi_id": aoi_id,
                "status": "skip",
                "reason": f"missing {os.path.basename(p)}",
            }

    point_year = pd.read_parquet(py_path)
    point_summary = pd.read_parquet(ps_path)
    points_static = pd.read_parquet(pt_path)

    cfg = DonorConfig()

    try:
        scores, pairs = score_donors_for_aoi(
            point_year, point_summary, points_static, cfg
        )
    except Exception as exc:
        LOGGER.exception("Error in %s: %s", aoi_id, exc)
        return {"aoi_id": aoi_id, "status": "error", "reason": str(exc)}

    if scores.empty:
        return {"aoi_id": aoi_id, "status": "empty", "n_recipients": 0}

    write_donor_outputs(scores, pairs, points_dir)

    n_ok = (scores["status"] == "ok").sum()
    n_low = (scores["status"] == "low_confidence").sum()
    n_no = (scores["status"] == "no_donor").sum()
    return {
        "aoi_id": aoi_id,
        "status": "ok",
        "n_recipients": len(scores),
        "n_ok": int(n_ok),
        "n_low_confidence": int(n_low),
        "n_no_donor": int(n_no),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch donor scoring")
    parser.add_argument(
        "state_dirs",
        nargs="+",
        help="State-level directories, e.g. /data/ssd2/handily/mt /data/ssd2/handily/nv",
    )
    args = parser.parse_args(argv)

    all_points_dirs = []
    for state_dir in args.state_dirs:
        found = sorted(glob.glob(os.path.join(state_dir, "aoi_*", "points")))
        LOGGER.info("%s: %d AOIs with points dir", state_dir, len(found))
        all_points_dirs.extend(found)

    LOGGER.info("Total AOIs to score: %d", len(all_points_dirs))

    results = []
    for points_dir in all_points_dirs:
        aoi_id = os.path.basename(os.path.dirname(points_dir))
        LOGGER.info("Scoring %s ...", aoi_id)
        result = run_aoi(points_dir)
        results.append(result)
        LOGGER.info("  %s", result)

    # Summary
    summary = pd.DataFrame(results)
    print("\n=== Donor Scoring Summary ===")
    print(summary.to_string(index=False))

    n_ok = (summary["status"] == "ok").sum()
    n_skip = (summary["status"].isin(["skip", "empty", "error"])).sum()
    print(f"\nProcessed: {n_ok}/{len(summary)}  Skipped/error: {n_skip}")

    if n_ok > 0:
        ok = summary[summary["status"] == "ok"]
        print(f"Recipients total: {ok['n_recipients'].sum()}")
        print(f"  ok:             {ok['n_ok'].sum()}")
        print(f"  low_confidence: {ok['n_low_confidence'].sum()}")
        print(f"  no_donor:       {ok['n_no_donor'].sum()}")


if __name__ == "__main__":
    main()
