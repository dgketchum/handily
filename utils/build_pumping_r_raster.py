"""Pumping-well R surface as 100 m rasters over a set of HUC8s, beside the
fold-based crossfit R of a rendered GNN arm.

R is a deterministic function of position (utils/pumping_r_core.compute): at every
100 m DEM cell one ``Variant`` recipe (kernel, gate, neighbour selection, outlier
screen, share cap) turns the nearby donor WTEs into a median, an upper quartile,
and their depth-gated blend. Donors are the HAND-table pumping / class-unknown
wells minus everything that made it through a state funnel into handily (the
same exclusion flags as utils/sweep_pumping_r.py), so no label or scoring well's
own record reaches its cell. ``--donor-pool`` instead reads a pool already
screened by utils/check_pumping_r_leakage.py ``--write-clean`` (no exclusions
re-applied), the pool the bundle R is built from too.

Outputs (float32, -9999 nodata, EPSG:5070 on the 100 m DEM grid, masked to the
HUC8 union):

  r_blend_wte_100m.tif / r_blend_dtw_100m.tif   the candidate R and z - R
  r_median_wte_100m.tif, r_q75_wte_100m.tif      components (uncapped)
  r_blend_w_100m.tif                             blend weight (1 = q75)
  r_nearest_donor_km_100m.tif                    horizontal distance to donor 1
  r_dk_km_100m.tif                               horizontal distance to the farthest weighted donor
  r_iqr_m_100m.tif, r_n_eff_100m.tif             donor-WTE IQR (m), effective donor count
  r_top_share_100m.tif                           largest single-donor weight share
  r_crossfit_wte_100m.tif / r_crossfit_dtw_100m.tif
                                                 the reference arm's rendered R
  r_diff_dtw_100m.tif                            blend DTW - crossfit DTW
                                                 (positive = candidate deeper)
  gnn_dtw_reference_100m.tif                     the reference arm's DTW map
  donors_extent.fgb                              donors inside the extent
  build_run.json                                 variant, counts, diagnostics

Diagnostics (build_run.json ``diagnostics``), for the blend and the crossfit R:
single-donor control (share of cells whose top donor carries >= 0.5 of the
weight, and >= 0.25 among cells with blend weight > 0.5), cap-bound share,
far-field share (farthest weighted donor > 10 km), roughness of the R surface
(median |5-point Laplacian| of WTE, m, and p99 |WTE step| between 4-neighbour
cells, m, overall and beyond 500 m of any donor), and agreement with the handily map (share of cells
within 5 m of its DTW, and the median signed difference R DTW - map DTW, m).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.merge import merge
from rasterio.windows import Window
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pumping_r_core import (  # noqa: E402
    DEM,
    HAND,
    DonorIndex,
    Variant,
    compute,
    exclude_donors,
    load_donors,
)

log = logging.getLogger("pumping_r_raster")

WBD = "/nas/hydrography/HUC_Boundaries/wbd_national/wbdhu8_5070.parquet"
HUC8_ROOT = "/data/ssd2/handily/huc8"
NODATA = -9999.0


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--huc8", nargs="+")
    g.add_argument("--huc8-file", help="text file, one HUC8 per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--reference-arm",
        required=True,
        help="gnn/<arm> dir name under each HUC8 holding gnn_r_wte_100m.tif",
    )
    ap.add_argument("--variants", required=True, help="JSON list of Variant dicts")
    ap.add_argument("--variant", required=True, help="name of the variant to render")
    ap.add_argument("--wbd", default=WBD)
    ap.add_argument("--dem", default=DEM)
    ap.add_argument("--hand", default=HAND)
    ap.add_argument("--pool-classes", default="pumping,unknown")
    ap.add_argument("--donor-pool", help="screened donor parquet; skips exclusions")
    ap.add_argument("--exclude-points", nargs="*", default=[])
    ap.add_argument("--exclude-m", type=float, default=10.0)
    ap.add_argument("--exclude-same-source", nargs="*", default=[])
    ap.add_argument("--exclude-same-source-m", type=float, default=0.0)
    ap.add_argument("--exclude-ids", nargs="*", default=[])
    ap.add_argument("--donor-buffer-km", type=float, default=200.0)
    ap.add_argument("--chunk", type=int, default=500_000)
    return ap.parse_args()


def write(path: Path, arr: np.ndarray, profile: dict) -> None:
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(arr), arr, NODATA).astype("float32"), 1)


def mosaic_reference(
    huc8s: list[str], arm: str, name: str, like: dict, window_bounds: tuple
) -> np.ndarray:
    paths = [Path(HUC8_ROOT) / h / "gnn" / arm / name for h in huc8s]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"reference arm lacks {name} for: {missing}")
    srcs = [rasterio.open(p) for p in paths]
    arr, tr = merge(
        srcs,
        bounds=window_bounds,
        res=(like["transform"].a, -like["transform"].e),
        nodata=NODATA,
        method="first",
    )
    for s in srcs:
        s.close()
    out = arr[0].astype("float64")
    out[out == NODATA] = np.nan
    if out.shape != (like["height"], like["width"]):
        raise SystemExit(
            f"mosaic shape {out.shape} != lattice {(like['height'], like['width'])}"
        )
    if abs(tr.c - like["transform"].c) > 1e-3 or abs(tr.f - like["transform"].f) > 1e-3:
        raise SystemExit(
            f"mosaic origin {tr.c, tr.f} != lattice {like['transform'].c, like['transform'].f}"
        )
    return out


def roughness(dtw: np.ndarray, mask: np.ndarray) -> dict:
    """Median |5-point Laplacian| and p99 |4-neighbour step| of a water-table
    elevation grid, m, over cells where the cell and all four neighbours are
    finite and in mask. Taken on WTE, not DTW, so terrain does not dominate."""
    c = dtw[1:-1, 1:-1]
    n, s, w, e = dtw[:-2, 1:-1], dtw[2:, 1:-1], dtw[1:-1, :-2], dtw[1:-1, 2:]
    ok = mask[1:-1, 1:-1] & np.isfinite(c) & np.isfinite(n) & np.isfinite(s)
    ok &= np.isfinite(w) & np.isfinite(e)
    if not ok.any():
        return {"lap_p50_m": None, "step_p99_m": None, "n": 0}
    lap = np.abs(n + s + w + e - 4 * c)[ok]
    step = np.concatenate([np.abs(c - e)[ok], np.abs(c - s)[ok]])
    return {
        "lap_p50_m": float(np.median(lap)),
        "step_p99_m": float(np.percentile(step, 99)),
        "n": int(ok.sum()),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vlist = {d["name"]: d for d in json.load(open(args.variants))}
    if args.variant not in vlist:
        raise SystemExit(f"variant {args.variant} not in {sorted(vlist)}")
    v = Variant.from_dict(vlist[args.variant])
    huc8s = args.huc8 or [s.strip() for s in open(args.huc8_file) if s.strip()]

    wbd = gpd.read_parquet(args.wbd)
    sel = wbd[wbd["huc8"].isin(huc8s)]
    if len(sel) != len(huc8s):
        raise SystemExit(f"WBD lacks {set(huc8s) - set(sel['huc8'])}")
    union = unary_union(sel.geometry.values)
    minx, miny, maxx, maxy = union.bounds

    # lattice = DEM grid windowed to the union, snapped to whole cells
    with rasterio.open(args.dem) as dem:
        t = dem.transform
        c0 = int(np.floor((minx - t.c) / t.a))
        c1 = int(np.ceil((maxx - t.c) / t.a))
        r0 = int(np.floor((maxy - t.f) / t.e))
        r1 = int(np.ceil((miny - t.f) / t.e))
        win = Window(c0, r0, c1 - c0, r1 - r0)
        z = dem.read(1, window=win).astype("float64")
        z[z == dem.nodata] = np.nan
        tr = dem.window_transform(win)
        crs = dem.crs
    h, w = z.shape
    prof = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "crs": crs,
        "transform": tr,
        "height": h,
        "width": w,
        "nodata": NODATA,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }
    inside = ~features.geometry_mask(
        [union], out_shape=(h, w), transform=tr, invert=False
    )
    valid = inside & np.isfinite(z)
    log.info(
        "lattice %d x %d cells; %d inside HUC8 union with DEM", h, w, int(valid.sum())
    )
    bounds = (tr.c, tr.f + tr.e * h, tr.c + tr.a * w, tr.f)

    if args.donor_pool:
        donors = pd.read_parquet(args.donor_pool)
    else:
        donors = load_donors(args.hand, args.dem, args.pool_classes.split(","))
    buf = args.donor_buffer_km * 1000.0
    near = donors["x5070"].between(minx - buf, maxx + buf) & donors["y5070"].between(
        miny - buf, maxy + buf
    )
    donors = donors[near].reset_index(drop=True)
    log.info(
        "donors within %.0f km of the extent: %d", args.donor_buffer_km, len(donors)
    )
    if args.donor_pool:
        excl = {"donor_pool": args.donor_pool}
    else:
        donors, excl = exclude_donors(
            donors,
            args.exclude_points,
            args.exclude_m,
            args.exclude_same_source,
            args.exclude_same_source_m,
            args.exclude_ids,
        )
    index = DonorIndex(donors, v.vw)

    rows, cols = np.nonzero(valid)
    qxy = np.column_stack([tr.c + (cols + 0.5) * tr.a, tr.f + (rows + 0.5) * tr.e])
    qz = z[rows, cols]
    res = compute(index, qxy, qz, v, chunk=args.chunk)
    log.info("computed %s at %d cells (%.0f s)", v.name, len(qz), time.time() - t0)

    def grid(a: np.ndarray) -> np.ndarray:
        g = np.full((h, w), np.nan)
        g[rows, cols] = a
        return g

    blend_wte = grid(res["r_blend"])
    blend_dtw = z - blend_wte
    write(out / "r_blend_wte_100m.tif", blend_wte, prof)
    write(out / "r_blend_dtw_100m.tif", blend_dtw, prof)
    write(out / "r_median_wte_100m.tif", grid(res["r_median"]), prof)
    write(out / "r_q75_wte_100m.tif", grid(res["r_q75"]), prof)
    write(out / "r_blend_w_100m.tif", grid(res["w_blend"]), prof)
    write(out / "r_nearest_donor_km_100m.tif", grid(res["nearest_h_km"]), prof)
    write(out / "r_dk_km_100m.tif", grid(res["dk_h_km"]), prof)
    write(out / "r_iqr_m_100m.tif", grid(res["iqr_m"]), prof)
    write(out / "r_n_eff_100m.tif", grid(res["n_eff"]), prof)
    write(out / "r_top_share_100m.tif", grid(res["top_share"]), prof)

    ref_r = mosaic_reference(
        huc8s, args.reference_arm, "gnn_r_wte_100m.tif", prof, bounds
    )
    ref_dtw = mosaic_reference(
        huc8s, args.reference_arm, "gnn_dtw_100m.tif", prof, bounds
    )
    ref_r[~valid] = np.nan
    ref_dtw[~valid] = np.nan
    ref_r_dtw = z - ref_r
    write(out / "r_crossfit_wte_100m.tif", ref_r, prof)
    write(out / "r_crossfit_dtw_100m.tif", ref_r_dtw, prof)
    write(out / "gnn_dtw_reference_100m.tif", ref_dtw, prof)
    diff = blend_dtw - ref_r_dtw
    write(out / "r_diff_dtw_100m.tif", diff, prof)

    ext = donors[
        donors["x5070"].between(bounds[0], bounds[2])
        & donors["y5070"].between(bounds[1], bounds[3])
    ]
    gpd.GeoDataFrame(
        ext[
            [
                "canonical_id",
                "source",
                "well_class",
                "confinement_class",
                "mean_dtw",
                "z_surf_m",
                "wte_m",
            ]
        ],
        geometry=gpd.points_from_xy(ext["x5070"], ext["y5070"]),
        crs="EPSG:5070",
    ).to_file(out / "donors_extent.fgb", driver="FlatGeobuf")

    far500 = grid(res["nearest_h_km"]) > 0.5
    ts, wb, dk = res["top_share"], res["w_blend"], res["dk_h_km"]
    hi_w, far = wb > 0.5, dk > 10
    ref_v = ref_r_dtw[valid]
    diag = {
        "single_donor_share_ge_0.5": float(np.mean(ts >= 0.5)),
        "single_donor_share_ge_0.25_where_w_gt_0.5": (
            float(np.mean(ts[hi_w] >= 0.25)) if hi_w.any() else None
        ),
        "cap_bound_blend": float(np.mean(res["cap_bound"])),
        "cap_bound_crossfit_2m": float(
            np.mean(ref_v[np.isfinite(ref_v)] <= 2.0 + 1e-6)
        ),
        "far_field_dk_gt_10km": float(np.mean(far)),
        "cap_bound_blend_in_far_field": (
            float(np.mean(res["cap_bound"][far])) if far.any() else None
        ),
        "roughness_blend": roughness(blend_wte, valid),
        "roughness_blend_beyond_500m": roughness(blend_wte, valid & far500),
        "roughness_crossfit": roughness(ref_r, valid),
        "roughness_crossfit_beyond_500m": roughness(ref_r, valid & far500),
        "roughness_handily_map": roughness(z - ref_dtw, valid),
    }
    for lab, a in (("blend", blend_dtw), ("crossfit", ref_r_dtw)):
        ok = valid & np.isfinite(a) & np.isfinite(ref_dtw)
        dd = a[ok] - ref_dtw[ok]
        diag[f"agree_{lab}_within_5m_of_map"] = float(np.mean(np.abs(dd) <= 5))
        diag[f"median_{lab}_minus_map_m"] = float(np.median(dd))
        diag[f"dtw_p50_{lab}_m"] = float(np.median(a[ok]))
        diag[f"frac_dtw_gt_300m_{lab}"] = float(np.mean(a[ok] > 300))
    diag["dtw_p50_map_m"] = float(np.median(ref_dtw[valid & np.isfinite(ref_dtw)]))

    dv = diff[np.isfinite(diff)]
    summary = {
        "huc8": huc8s,
        "reference_arm": args.reference_arm,
        "variant": json.loads(v.to_json()),
        "pool_classes": args.pool_classes,
        "exclusion": {**excl, "files": args.exclude_points},
        "donors_used": int(len(donors)),
        "donors_in_extent": int(len(ext)),
        "cells_valid": int(len(qz)),
        "cells_with_reference": int(dv.size),
        "blend_w_mean": float(np.nanmean(wb)),
        "frac_pure_q75": float(np.mean(wb >= 1.0)),
        "frac_pure_median": float(np.mean(wb <= 0.0)),
        "nearest_donor_km_p50": float(np.nanmedian(res["nearest_h_km"])),
        "nearest_donor_km_p90": float(np.nanpercentile(res["nearest_h_km"], 90)),
        "diff_dtw_m_blend_minus_crossfit": {
            "p10": float(np.percentile(dv, 10)),
            "p50": float(np.percentile(dv, 50)),
            "p90": float(np.percentile(dv, 90)),
            "mean": float(dv.mean()),
        },
        "diagnostics": diag,
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(summary, open(out / "build_run.json", "w"), indent=2)
    log.info("diagnostics %s", json.dumps(diag))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
