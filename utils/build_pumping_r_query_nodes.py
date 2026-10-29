"""Pumping-well R at every bundle query node and frozen source well.

The R a handily arm trains on must be the R it renders with. The NM render
samples the pumping-R raster (utils/build_pumping_r_raster.py) at 100 m cell
centres, so here every query node (CONUS, water pseudo-rows included) and every
row of the frozen OSE source table is evaluated at the centre of its DEM cell,
with that cell's DEM surface, from the same screened donor pool and variant.
Inside the raster footprint this reproduces the raster value exactly (checked
with ``--check-raster``).

Writes, in ``--out-dir``:

  r_replace.parquet   query_node_idx, r_wte_m (the variant's capped blend, m asl)
                      plus diagnostics; input to build_parity_bundle --r-replace
  <sources stem>_pumpr.parquet
                      the frozen source table with r_wte_m := new R and
                      wte_residual_m := (z_surf_m - obs_dtw_m) - new R, so the
                      inference reframe (--r-ground-cap-m) sees the new frame
  pumpr_query_run.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pumping_r_core import DEM, DonorIndex, Variant, compute, sample_coarse  # noqa: E402

BUNDLE = "/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2_ndwr_ufold_ose_parity"
SOURCES = "/data/ssd2/handily/nm/regional/wells/ose_admitted_sources_frozen.parquet"

log = logging.getLogger("pumping_r_query_nodes")


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--donor-pool", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--sources", default=SOURCES)
    ap.add_argument("--dem", default=DEM)
    ap.add_argument("--check-raster", help="r_blend_wte_100m.tif to reproduce")
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def cell_centres(dem: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with rasterio.open(dem) as src:
        t = src.transform
    col = np.floor((x - t.c) / t.a)
    row = np.floor((y - t.f) / t.e)
    return np.column_stack([t.c + (col + 0.5) * t.a, t.f + (row + 0.5) * t.e])


def evaluate(
    index: DonorIndex, v: Variant, dem: str, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    cxy = cell_centres(dem, x, y)
    cz = sample_coarse(dem, cxy[:, 0], cxy[:, 1])
    if not np.isfinite(cz).all():
        raise SystemExit(f"{int((~np.isfinite(cz)).sum())} points off the DEM")
    return cxy, cz, compute(index, cxy, cz, v)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    v = Variant.from_dict(
        {d["name"]: d for d in json.load(open(args.variants))}[args.variant]
    )
    donors = pd.read_parquet(args.donor_pool)
    index = DonorIndex(donors, v.vw)
    log.info("donor pool %s: %d wells", args.donor_pool, len(donors))
    run: dict = {"variant": json.loads(v.to_json()), "donor_pool": args.donor_pool}

    qn = pd.read_parquet(
        f"{args.bundle}/query_nodes.parquet",
        columns=[
            "query_node_idx",
            "source",
            "x5070",
            "y5070",
            "z_surf_well_m",
            "wte_obs_m",
            "regional_wte_idw_oof_m",
            "is_water_pseudo",
        ],
    )
    cxy, cz, res = evaluate(
        index,
        v,
        args.dem,
        qn["x5070"].to_numpy("float64"),
        qn["y5070"].to_numpy("float64"),
    )
    dz = cz - qn["z_surf_well_m"].to_numpy("float64")
    log.info(
        "query nodes %d: cell DEM - bundle z |.| p99 %.3f m, max %.3f m",
        len(qn),
        np.percentile(np.abs(dz), 99),
        np.abs(dz).max(),
    )
    r = res["r_blend"]
    if not np.isfinite(r).all():
        raise SystemExit(f"{int((~np.isfinite(r)).sum())} query nodes without R")
    rep = pd.DataFrame(
        {
            "query_node_idx": qn["query_node_idx"].to_numpy(),
            "r_wte_m": r,
            "r_median_m": res["r_median"],
            "w_blend": res["w_blend"],
            "n_eff": res["n_eff"],
            "nearest_h_km": res["nearest_h_km"],
            "dk_h_km": res["dk_h_km"],
            "cap_bound": res["cap_bound"],
        }
    )
    rep.to_parquet(out / "r_replace.parquet", index=False)
    z = qn["z_surf_well_m"].to_numpy("float64")
    real = ~qn["is_water_pseudo"].astype(bool).to_numpy()
    wte = qn["wte_obs_m"].to_numpy("float64")
    old = qn["regional_wte_idw_oof_m"].to_numpy("float64")
    grp = np.where(real, qn["source"].to_numpy(), "water_pseudo")
    run["query_nodes"] = {}
    for g in ["all", *sorted(set(grp))]:
        m = np.ones(len(qn), bool) if g == "all" else grp == g
        # residual convention of the sweep: R - obs WTE, m; positive = R too high
        d_new, d_old = r[m] - wte[m], np.minimum(old[m], z[m] - 2.0) - wte[m]
        run["query_nodes"][g] = {
            "n": int(m.sum()),
            "new_mad_m": float(np.median(np.abs(d_new))),
            "new_median_m": float(np.median(d_new)),
            "old_capped_mad_m": float(np.median(np.abs(d_old))),
            "old_capped_median_m": float(np.median(d_old)),
            "new_cap_bound_frac": float(np.mean(res["cap_bound"][m])),
            "new_nearest_km_p50": float(np.median(res["nearest_h_km"][m])),
        }
        log.info("%s: %s", g, run["query_nodes"][g])

    src = pd.read_parquet(args.sources)
    obs = src["z_surf_m"].to_numpy("float64") - src["obs_dtw_m"].to_numpy("float64")
    gap = np.abs(src["r_wte_m"] + src["wte_residual_m"] - obs).max()
    if gap > 1e-3:
        raise SystemExit(f"sources: r_wte_m + wte_residual_m != z - dtw (max {gap})")
    sxy, sz, sres = evaluate(
        index,
        v,
        args.dem,
        src["x5070"].to_numpy("float64"),
        src["y5070"].to_numpy("float64"),
    )
    szgap = np.abs(sz - src["z_surf_m"].to_numpy("float64"))
    log.info("sources %d: cell DEM - z_surf_m max %.3f m", len(src), szgap.max())
    src_new = src.copy()
    src_new["r_wte_m"] = sres["r_blend"]
    src_new["wte_residual_m"] = obs - sres["r_blend"]
    spath = out / f"{Path(args.sources).stem}_pumpr.parquet"
    src_new.to_parquet(spath, index=False)
    run["sources"] = {
        "path": str(spath),
        "n": len(src),
        "z_gap_max_m": float(szgap.max()),
        "new_mad_m": float(np.median(np.abs(sres["r_blend"] - obs))),
        "old_mad_m": float(np.median(np.abs(src["r_wte_m"].to_numpy() - obs))),
    }

    if args.check_raster:
        allxy = np.vstack([cxy, sxy])
        allr = np.concatenate([r, sres["r_blend"]])
        ras = sample_coarse(args.check_raster, allxy[:, 0], allxy[:, 1])
        inside = np.isfinite(ras)
        d = np.abs(ras[inside] - allr[inside])
        run["raster_check"] = {
            "raster": args.check_raster,
            "n_inside": int(inside.sum()),
            "max_abs_m": float(d.max()) if d.size else None,
            "p99_abs_m": float(np.percentile(d, 99)) if d.size else None,
        }
        log.info("raster check: %s", run["raster_check"])
    run["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(run, open(out / "pumpr_query_run.json", "w"), indent=1)
    log.info("wrote %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
