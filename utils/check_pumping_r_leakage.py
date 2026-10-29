"""Leakage test for the pumping-well R donor pool (utils/pumping_r_core.py).

The donor wells that build R must be disjoint from every well handily uses
elsewhere: training labels (all real query nodes of the bundle), assimilation
sources (the frozen OSE source table; the bundle's source edges point at query
nodes, so those are covered), and scoring wells (the OSE admission-eligible
half; NM monitoring scoring sites are bundle query nodes). Each donor is tested
against each well set three ways:

  id    donor canonical_id equals the 12-character prefix of a set well's GWX id
        (bundle canonical ids; OSE POD ids mapped through the GWX nm_ose build)
  near  donor within 10 m / 100 m of a set well
  dup   donor within 250 m of a set well whose DTW matches within 0.5 m

The donor pool is rebuilt from a raster run's build_run.json (same classes,
buffer, exclusions), read from a parquet, or built CONUS-wide from explicit
exclusion arguments (``--build``). Writes ``leak_report.json`` and
``leak_donors.parquet`` (every flagged donor, one row per donor x set). With
``--write-clean``, also writes the pool minus every flagged donor: the one pool
that both the NM raster and the bundle R (utils/build_pumping_r_query_nodes.py)
are built from, so the second pass of this test on it must come back zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pumping_r_core import DEM, HAND, exclude_donors, load_donors  # noqa: E402

BUNDLE = "/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2_ndwr_ufold_ose_parity"
SOURCES = "/data/ssd2/handily/nm/regional/wells/ose_admitted_sources_frozen.parquet"
TARGETS = "/data/ssd2/handily/nm/regional/wells/ose_admission_targets.parquet"
NV_SOURCES = "/data/ssd2/handily/nv/regional/wells/ndwr_admitted_sources_frozen.parquet"
NV_TARGETS = "/data/ssd2/handily/nv/regional/wells/ndwr_admission_targets.parquet"
OSE_GWX = (
    "/data/ssd2/handily/nm/regional/wells/nm_ose_gwx_20260919T2220Z-78ad348.parquet"
)
NEAR_M = (10.0, 100.0)
DUP_M = 250.0
DUP_DTW_M = 0.5

log = logging.getLogger("check_pumping_r_leakage")


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-json", help="build_run.json of a pumping-R raster run")
    g.add_argument("--donors", help="parquet of donors (canonical_id, xy, mean_dtw)")
    g.add_argument("--build", action="store_true", help="CONUS pool from the args")
    ap.add_argument("--pool-classes", default="pumping,unknown")
    ap.add_argument("--exclude-points", nargs="*", default=[])
    ap.add_argument("--exclude-m", type=float, default=10.0)
    ap.add_argument("--exclude-same-source", nargs="*", default=[])
    ap.add_argument("--exclude-same-source-m", type=float, default=0.0)
    ap.add_argument("--exclude-ids", nargs="*", default=[])
    ap.add_argument("--write-clean", help="parquet path for the pool minus flags")
    ap.add_argument(
        "--wbd",
        default="/nas/hydrography/HUC_Boundaries/wbd_national/wbdhu8_5070.parquet",
    )
    ap.add_argument("--donor-buffer-km", type=float, default=200.0)
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--sources", default=SOURCES)
    ap.add_argument("--targets", default=TARGETS)
    ap.add_argument("--nv-sources", default=NV_SOURCES)
    ap.add_argument("--nv-targets", default=NV_TARGETS)
    ap.add_argument("--ose-gwx", default=OSE_GWX)
    ap.add_argument("--hand", default=HAND)
    ap.add_argument("--dem", default=DEM)
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def rebuild_pool(args: argparse.Namespace) -> pd.DataFrame:
    import geopandas as gpd

    run = json.load(open(args.run_json))
    donors = load_donors(args.hand, args.dem, run["pool_classes"].split(","))
    wbd = gpd.read_parquet(args.wbd)
    minx, miny, maxx, maxy = wbd[wbd["huc8"].isin(run["huc8"])].total_bounds
    buf = args.donor_buffer_km * 1000.0
    near = donors["x5070"].between(minx - buf, maxx + buf) & donors["y5070"].between(
        miny - buf, maxy + buf
    )
    donors = donors[near].reset_index(drop=True)
    ex = run["exclusion"]
    donors, _ = exclude_donors(
        donors,
        ex["files"],
        ex["radius_m"],
        ex["same_source"],
        ex["same_source_m"],
        ex["id_files"],
    )
    if len(donors) != run.get("donors_used", len(donors)):
        raise SystemExit(
            f"rebuilt pool {len(donors)} != run donors_used {run['donors_used']}"
        )
    return donors


def build_pool(args: argparse.Namespace) -> pd.DataFrame:
    donors = load_donors(args.hand, args.dem, args.pool_classes.split(","))
    donors, _ = exclude_donors(
        donors,
        args.exclude_points,
        args.exclude_m,
        args.exclude_same_source,
        args.exclude_same_source_m,
        args.exclude_ids,
    )
    return donors


def ose_ids(pod_lists: pd.Series, gwx: pd.DataFrame) -> set[str]:
    pods = {p.strip() for s in pod_lists.dropna() for p in str(s).split(",")}
    return set(gwx.loc[gwx["source_well_id"].isin(pods), "cid12"])


def well_sets(args: argparse.Namespace) -> dict[str, dict]:
    q = pd.read_parquet(
        f"{args.bundle}/query_nodes.parquet",
        columns=[
            "canonical_id",
            "source",
            "x5070",
            "y5070",
            "mean_dtw",
            "is_water_pseudo",
        ],
    )
    q = q[~q["is_water_pseudo"].astype(bool)].reset_index(drop=True)
    gwx = pq.read_table(args.ose_gwx, columns=["source_well_id", "canonical_id"])
    gwx = gwx.to_pandas()
    gwx["source_well_id"] = gwx["source_well_id"].astype(str)
    gwx["cid12"] = gwx["canonical_id"].str[:12]
    src = pd.read_parquet(args.sources)
    tgt = pd.read_parquet(args.targets)
    tgt = tgt[tgt["is_admission_eligible"].astype(bool)].reset_index(drop=True)
    sets = {
        "train_labels": {
            "xy": q[["x5070", "y5070"]].to_numpy("float64"),
            "dtw": q["mean_dtw"].to_numpy("float64"),
            "ids": set(q["canonical_id"].str[:12]),
            "n": len(q),
            "by_source": q["source"].value_counts().to_dict(),
        },
        "assim_sources": {
            "xy": src[["x5070", "y5070"]].to_numpy("float64"),
            "dtw": src["obs_dtw_m"].to_numpy("float64"),
            "ids": ose_ids(src["source_well_ids"], gwx),
            "n": len(src),
        },
        "scoring_ose": {
            "xy": tgt[["x5070", "y5070"]].to_numpy("float64"),
            "dtw": tgt["obs_dtw_m"].to_numpy("float64"),
            "ids": ose_ids(tgt["source_well_ids"], gwx),
            "n": len(tgt),
        },
    }
    nvs = pd.read_parquet(args.nv_sources)
    nvt = pd.read_parquet(args.nv_targets)
    nvt = nvt[nvt["is_admission_eligible"].astype(bool)].reset_index(drop=True)
    for k, t in (("nv_assim_sources", nvs), ("nv_scoring", nvt)):
        sets[k] = {
            "xy": t[["x5070", "y5070"]].to_numpy("float64"),
            "dtw": t["obs_dtw_m"].to_numpy("float64"),
            "ids": set(),
            "n": len(t),
        }
    for k, s in sets.items():
        log.info("%s: %d wells, %d linked ids", k, s["n"], len(s["ids"]))
    return sets


def test(donors: pd.DataFrame, s: dict) -> tuple[dict, pd.DataFrame]:
    dxy = donors[["x5070", "y5070"]].to_numpy("float64")
    ddtw = donors["mean_dtw"].to_numpy("float64")
    tree = cKDTree(s["xy"])
    d, _ = tree.query(dxy, k=1)
    flags = {"id": donors["canonical_id"].isin(s["ids"]).to_numpy()}
    for r in NEAR_M:
        flags[f"near_{r:.0f}m"] = d <= r
    dup = np.zeros(len(donors), bool)
    for i in np.nonzero(d <= DUP_M)[0]:
        nb = tree.query_ball_point(dxy[i], DUP_M)
        dup[i] = bool(np.any(np.abs(s["dtw"][nb] - ddtw[i]) <= DUP_DTW_M))
    flags[f"dup_{DUP_M:.0f}m_{DUP_DTW_M}m"] = dup
    counts = {k: int(v.sum()) for k, v in flags.items()}
    any_flag = np.logical_or.reduce(list(flags.values()))
    counts["any"] = int(any_flag.sum())
    counts["any_by_source"] = donors.loc[any_flag, "source"].value_counts().to_dict()
    rows = donors.loc[
        any_flag, ["canonical_id", "source", "x5070", "y5070", "mean_dtw"]
    ]
    rows = rows.assign(
        nearest_m=d[any_flag], **{k: v[any_flag] for k, v in flags.items()}
    )
    return counts, rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.run_json:
        donors = rebuild_pool(args)
    elif args.build:
        donors = build_pool(args)
    else:
        donors = pd.read_parquet(args.donors)
    log.info("donor pool: %d wells", len(donors))
    report = {
        "n_donors": len(donors),
        "donors_by_source": donors["source"].value_counts().to_dict(),
        "sets": {},
    }
    flagged = []
    for name, s in well_sets(args).items():
        counts, rows = test(donors, s)
        report["sets"][name] = {"n_wells": s["n"], "n_ids": len(s["ids"]), **counts}
        if "by_source" in s:
            report["sets"][name]["wells_by_source"] = s["by_source"]
        flagged.append(rows.assign(well_set=name))
        log.info("%s: %s", name, counts)
    fl = pd.concat(flagged, ignore_index=True)
    report["n_donors_flagged_any_set"] = int(fl["canonical_id"].nunique())
    fl.to_parquet(out / "leak_donors.parquet", index=False)
    json.dump(report, open(out / "leak_report.json", "w"), indent=1)
    log.info("flagged donors (any set): %d", report["n_donors_flagged_any_set"])
    if args.write_clean:
        clean = donors[~donors["canonical_id"].isin(set(fl["canonical_id"]))]
        clean.reset_index(drop=True).to_parquet(args.write_clean, index=False)
        log.info("clean pool: %d donors -> %s", len(clean), args.write_clean)


if __name__ == "__main__":
    main()
