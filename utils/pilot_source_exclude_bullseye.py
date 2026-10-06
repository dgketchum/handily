"""Pilot scorer for the source-edge exclusion-radius sweep (bullseye artefacts).

The source-assimilation render reads each cell's k nearest wells with a
short-range learned kernel, so every admitted well sits in a dimple of roughly
1 km. ``--source-exclude-m`` drops edges from wells inside a disk around the
cell; this script quantifies, per render variant of ONE basin:

* accuracy at the held-out primary sites inside the basin (MAD / median /
  bias / RMSE, m; by distance-to-source band), Ma alongside;
* the dimple at admitted source wells: DTW at the well's cell minus the median
  DTW on a 700-1300 m annulus (m; positive = well cell deeper than its ring);
* the ring step at every tested radius: mean DTW on the annulus just inside
  the radius minus the annulus just outside (m), the artefact the exclusion
  disk can introduce where a well crosses the disk edge;
* map roughness: median and p99 of the DTW gradient magnitude (m per 100 m
  cell) and the share of negative cells.

Usage:
    uv run python utils/pilot_source_exclude_bullseye.py --basin 16060001 \\
        --arms shipped=gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold \\
               x1000=..._srcx1000 x2000=..._srcx2000 x3500=..._srcx3500 \\
        --radii-m 1000 2000 3500 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pilot_source_exclude")

HUC8_ROOT = Path("/data/ssd2/handily/huc8")
SITES = (
    "/data/ssd2/handily/nv/regional/statewide_eval_ndwrlbl_ufold/sites_sampled.parquet"
)
SOURCES = "/data/ssd2/handily/nv/regional/wells/ndwr_admitted_sources_frozen.parquet"
DIST_BANDS = [(0, 0.3), (0.3, 1), (1, 5), (5, np.inf)]
GWX = "/data/ssd2/gwx/products/current/wells.geoparquet"
RINGS = [(0, 100), (100, 300), (300, 500), (500, 700), (700, 1000), (1000, 1500)]
OUTER = (1500, 2500)


def radial_profile(a: np.ndarray, tr, pts: np.ndarray) -> np.ndarray:
    """Per point: ring means minus the OUTER ring mean (m), one column per RINGS entry."""
    out = np.full((len(pts), len(RINGS)), np.nan)
    pad = int(np.ceil(OUTER[1] / abs(tr.a))) + 1
    for i, (x, y) in enumerate(pts):
        ri, ci = int((y - tr.f) / tr.e), int((x - tr.c) / tr.a)
        rs = slice(max(ri - pad, 0), min(ri + pad + 1, a.shape[0]))
        cs = slice(max(ci - pad, 0), min(ci + pad + 1, a.shape[1]))
        win = a[rs, cs]
        rr, cc = np.mgrid[rs, cs]
        d = np.hypot((cc + 0.5) * tr.a + tr.c - x, (rr + 0.5) * tr.e + tr.f - y)
        ref_sel = (d >= OUTER[0]) & (d < OUTER[1]) & np.isfinite(win)
        if not ref_sel.any():
            continue
        ref = win[ref_sel].mean()
        for j, (lo, hi) in enumerate(RINGS):
            sel = (d >= lo) & (d < hi) & np.isfinite(win)
            if sel.any():
                out[i, j] = win[sel].mean() - ref
    return out


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(rows)


def panel(resid: np.ndarray) -> dict:
    r = resid[np.isfinite(resid)]
    return {
        "n": int(len(r)),
        "mad_m": float(np.median(np.abs(r))) if len(r) else np.nan,
        "median_resid_m": float(np.median(r)) if len(r) else np.nan,
        "bias_m": float(r.mean()) if len(r) else np.nan,
        "rmse_m": float(np.sqrt((r**2).mean())) if len(r) else np.nan,
    }


def load_dtw(path: Path):
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("float64")
        a[a == ds.nodata] = np.nan
        return a, ds.transform


def sample(a: np.ndarray, tr, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    c = np.floor((x - tr.c) / tr.a).astype(int)
    r = np.floor((y - tr.f) / tr.e).astype(int)
    ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    out = np.full(len(x), np.nan)
    out[ok] = a[r[ok], c[ok]]
    return out


def annulus_stat(
    a: np.ndarray, tr, x: float, y: float, r0: float, r1: float, fn=np.nanmedian
):
    cell = abs(tr.a)
    c0 = (x - tr.c) / tr.a
    r0_ = (y - tr.f) / tr.e
    pad = int(np.ceil(r1 / cell)) + 1
    ci, ri = int(np.floor(c0)), int(np.floor(r0_))
    rs = slice(max(ri - pad, 0), min(ri + pad + 1, a.shape[0]))
    cs = slice(max(ci - pad, 0), min(ci + pad + 1, a.shape[1]))
    win = a[rs, cs]
    rr, cc = np.mgrid[rs, cs]
    dx = (cc + 0.5) * tr.a + tr.c - x
    dy = (rr + 0.5) * tr.e + tr.f - y
    d = np.hypot(dx, dy)
    sel = (d >= r0) & (d < r1) & np.isfinite(win)
    return float(fn(win[sel])) if sel.any() else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="name=render_dir_name")
    ap.add_argument("--radii-m", nargs="+", type=float, default=[1000, 2000, 3500])
    ap.add_argument("--sites", default=SITES)
    ap.add_argument("--sources", default=SOURCES)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gwx", default=GWX, help="GWX wells geoparquet (all sources)")
    ap.add_argument(
        "--reference-arm",
        default=None,
        help="arm name to difference the others against",
    )
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arms = dict(a.split("=", 1) for a in args.arms)

    s = pd.read_parquet(args.sites)
    s = s[s["basin_hit"].astype(str) == args.basin]
    prim = s[s["is_admission_eligible"] & ~s["is_pumping"]].reset_index(drop=True)
    log.info("basin %s: %d target sites, %d primary", args.basin, len(s), len(prim))

    bb = gpd.read_file(HUC8_ROOT / args.basin / "basin_boundary.fgb").to_crs(5070)
    src = pd.read_parquet(args.sources)
    g = gpd.GeoDataFrame(
        src, geometry=gpd.points_from_xy(src.x5070, src.y5070), crs=5070
    )
    src = src[g.within(bb.union_all()).to_numpy()].reset_index(drop=True)
    log.info("%d admitted source wells inside the basin", len(src))

    acc, dimple, ring, rough = [], [], [], []
    ma_done = False
    for name, rdir in arms.items():
        a, tr = load_dtw(HUC8_ROOT / args.basin / "gnn" / rdir / "gnn_dtw_100m.tif")
        pred = sample(a, tr, prim.x5070.to_numpy(), prim.y5070.to_numpy())
        resid = pred - prim.obs_dtw_m.to_numpy()
        acc.append({"arm": name, "band": "all", **panel(resid)})
        for lo, hi in DIST_BANDS:
            m = (prim.dist_src_km >= lo) & (prim.dist_src_km < hi)
            acc.append(
                {"arm": name, "band": f"{lo}-{hi} km", **panel(resid[m.to_numpy()])}
            )
        if not ma_done:
            rm = prim.ma_dtw_m.to_numpy() - prim.obs_dtw_m.to_numpy()
            acc.append({"arm": "Ma", "band": "all", **panel(rm)})
            for lo, hi in DIST_BANDS:
                m = (prim.dist_src_km >= lo) & (prim.dist_src_km < hi)
                acc.append(
                    {"arm": "Ma", "band": f"{lo}-{hi} km", **panel(rm[m.to_numpy()])}
                )
            ma_done = True

        well = sample(a, tr, src.x5070.to_numpy(), src.y5070.to_numpy())
        ann = np.array(
            [annulus_stat(a, tr, x, y, 700, 1300) for x, y in zip(src.x5070, src.y5070)]
        )
        d = well - ann
        dimple.append(
            {
                "arm": name,
                "n_wells": int(np.isfinite(d).sum()),
                "median_abs_dimple_m": float(np.nanmedian(np.abs(d))),
                "p90_abs_dimple_m": float(np.nanpercentile(np.abs(d), 90)),
                "frac_abs_gt_2m": float(np.nanmean(np.abs(d) > 2)),
                "frac_abs_gt_5m": float(np.nanmean(np.abs(d) > 5)),
            }
        )
        for rad in args.radii_m:
            inner = np.array(
                [
                    annulus_stat(a, tr, x, y, rad - 250, rad - 50, np.nanmean)
                    for x, y in zip(src.x5070, src.y5070)
                ]
            )
            outer = np.array(
                [
                    annulus_stat(a, tr, x, y, rad + 50, rad + 250, np.nanmean)
                    for x, y in zip(src.x5070, src.y5070)
                ]
            )
            step = inner - outer
            ring.append(
                {
                    "arm": name,
                    "radius_m": rad,
                    "n_wells": int(np.isfinite(step).sum()),
                    "median_abs_step_m": float(np.nanmedian(np.abs(step))),
                    "p90_abs_step_m": float(np.nanpercentile(np.abs(step), 90)),
                }
            )
        gy, gx = np.gradient(a)
        gm = np.hypot(gx, gy)
        gm = gm[np.isfinite(gm)]
        rough.append(
            {
                "arm": name,
                "median_grad_m_per_cell": float(np.median(gm)),
                "p99_grad_m_per_cell": float(np.percentile(gm, 99)),
                "frac_negative": float(np.nanmean(a < 0)),
                "median_dtw_m": float(np.nanmedian(a)),
            }
        )
        log.info("%s: scored", name)

    # stacked radial |anomaly| profiles: ring mean minus the 1.5-2.5 km ring mean,
    # median of |.| over points; groups isolate WHICH wells the map is organised
    # around (admitted sources = source read; other GWX wells = well-pool
    # features such as drilled depth; random cells = terrain baseline)
    gw = gpd.read_parquet(args.gwx).to_crs(5070)
    gw = gw[gw.within(bb.union_all())]
    gxy = np.c_[gw.geometry.x, gw.geometry.y]
    dsrc = (
        cKDTree(np.c_[src.x5070, src.y5070]).query(gxy)[0]
        if len(src)
        else np.full(len(gw), np.inf)
    )
    groups = {
        "admitted sources": np.c_[src.x5070, src.y5070],
        "other NDWR wells": gxy[(dsrc > 500) & (gw["source"] == "nv_ndwr").to_numpy()],
        "NWIS wells": gxy[(dsrc > 500) & (gw["source"] == "nwis").to_numpy()],
    }
    rng = np.random.default_rng(1)
    prof_rows, diff_rows = [], []
    ref_name = args.reference_arm
    ref = (
        load_dtw(HUC8_ROOT / args.basin / "gnn" / arms[ref_name] / "gnn_dtw_100m.tif")[
            0
        ]
        if ref_name
        else None
    )
    qdir = out / "qgis"
    qdir.mkdir(exist_ok=True)
    for name, rdir in arms.items():
        a, tr = load_dtw(HUC8_ROOT / args.basin / "gnn" / rdir / "gnn_dtw_100m.tif")
        rr, cc = np.nonzero(np.isfinite(a))
        pick = rng.choice(len(rr), min(1500, len(rr)), replace=False)
        groups["random cells"] = np.c_[
            tr.c + (cc[pick] + 0.5) * tr.a, tr.f + (rr[pick] + 0.5) * tr.e
        ]
        for gname, pts in groups.items():
            pr = radial_profile(a, tr, pts)
            pr = pr[np.isfinite(pr).all(1)]
            row = {"arm": name, "group": gname, "n": int(len(pr))}
            for (lo, hi), v in zip(RINGS, np.median(np.abs(pr), 0)):
                row[f"abs_{lo}-{hi}m"] = float(v)
            prof_rows.append(row)
        if ref is not None and name != ref_name:
            d = ref - a
            ok = np.isfinite(d)
            diff_rows.append(
                {
                    "arm": name,
                    "median_abs_diff_m": float(np.median(np.abs(d[ok]))),
                    "p90_abs_diff_m": float(np.percentile(np.abs(d[ok]), 90)),
                    "p99_abs_diff_m": float(np.percentile(np.abs(d[ok]), 99)),
                    "frac_abs_gt_1m": float(np.mean(np.abs(d[ok]) > 1)),
                }
            )
            with rasterio.open(
                HUC8_ROOT / args.basin / "gnn" / rdir / "gnn_dtw_100m.tif"
            ) as ds:
                prof_r = ds.profile
            with rasterio.open(
                qdir / f"dtw_{ref_name}_minus_{name}.tif", "w", **prof_r
            ) as dst:
                dst.write(np.where(ok, d, prof_r["nodata"]).astype("float32"), 1)
    prof_df, diff_df = pd.DataFrame(prof_rows), pd.DataFrame(diff_rows)
    prof_df.to_csv(out / "radial_profile.csv", index=False)
    diff_df.to_csv(out / "diff_vs_reference.csv", index=False)

    acc_df, dim_df, ring_df, rough_df = map(pd.DataFrame, (acc, dimple, ring, rough))
    acc_df.to_csv(out / "accuracy_primary.csv", index=False)
    dim_df.to_csv(out / "dimple_at_sources.csv", index=False)
    ring_df.to_csv(out / "ring_step.csv", index=False)
    rough_df.to_csv(out / "roughness.csv", index=False)
    with open(out / "report.md", "w") as f:
        f.write(f"# Source-exclusion sweep, basin {args.basin}\n\n")
        f.write(
            "Residual = predicted - observed (m). Primary = held-out admission-eligible non-pumping sites.\n\n"
        )
        f.write("## Accuracy at primary sites\n\n" + md_table(acc_df.round(2)) + "\n\n")
        f.write(
            "## Dimple at admitted source wells (well cell - 700-1300 m annulus median, m)\n\n"
        )
        f.write(md_table(dim_df.round(3)) + "\n\n")
        f.write(
            "## Ring step across each radius (mean inside annulus - mean outside annulus, m)\n\n"
        )
        f.write(md_table(ring_df.round(3)) + "\n\n")
        f.write("## Map roughness\n\n" + md_table(rough_df.round(3)) + "\n\n")
        f.write(
            "## Radial |anomaly| profile (median over points of |ring mean - 1.5-2.5 km ring mean|, m)\n\n"
        )
        f.write(md_table(prof_df.round(2)) + "\n\n")
        if len(diff_df):
            f.write(
                f"## DTW difference vs reference arm `{ref_name}` (rasters in qgis/)\n\n"
            )
            f.write(md_table(diff_df.round(3)) + "\n")
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
