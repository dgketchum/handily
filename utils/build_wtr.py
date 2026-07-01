#!/usr/bin/env python
"""Per-HUC8 Water Table Ratio (WTR), regime gate, and R/K Toth fallback.

WTR (Haitjema & Mitchell-Bruker 2005) is the ratio of a recharge-mound height
h_max = R*L^2/(m*K*H) to the available relief d:

    WTR = R * L^2 / (m * K * H * d)

Interpreted per cell on the FAC grid (the within-basin discriminator we are
testing):
    WTR > 1  -> topography-controlled (mound exceeds relief, table clipped to
                terrain) -> HAND/FAC-REM is the right prior; gate g -> 1.
    WTR < 1  -> recharge-controlled (table flat/deep/decoupled) -> lean on the
                regional prior R; gate g -> 0.

Computed in log10 space so the L^2 sensitivity is bounded and the dimensional
cancellation is auditable:
    log10 WTR = [log10 R - log10 K]            (R[m/yr]/K[m/yr], dimensionless)
              + [2 log10 L - log10 H - log10 d](L^2/(H*d),       dimensionless)
              - log10 m

Terms (all reprojected onto the canonical per-basin grid defined by the L raster
from build_wtr_L.py; L and d stay native-fine, only R/K/H are upsampled):
    R  Reitz effective recharge (m/yr; default) | ETRM conus_mean (mm/yr / 1000).
       Sampled as a WINDOWED AREAL MEAN (--r-window-km, valid-normalized) -- the
       recharge mound h_max = R*L^2/(mKH) is built by recharge over the whole
       interfluve, so the point value at a valley-floor discharge zone (R~0 by
       definition) is the WRONG sample. Diffuse areal recharge is genuinely ~0 at
       most arid-basin wells; a GENUINE zero -> WTR=0 -> g=0 (kept as valid, the
       strongest recharge-controlled signal). Only recharge-product nodata -> NaN.
    L  wtr_L_edt (or wtr_L_dd) from build_wtr_L.py, L_eff = L_scale * L
    K  permeability_logk_x100 is GLHYMPS log10(intrinsic permeability k[m^2]) * 100
       (raw ~ -1050..-1650 == k 1e-10..1e-16 m^2; NOT log10(m/day) as an earlier
       project note mislabeled it -- caught by the R/K cancellation print: a m/day
       decode gives R/K ~ 1e10, physically impossible). Darcy: K = k*(rho*g/mu) ->
       K[m/s] = k * 9.8e6; K[m/yr] = K[m/s] * 3.156e7 (water ~20 C).
    H  sediment_thickness_basinfill_m (regolith proxy; WEAKEST term) | _avg (cap 50)
    d  fac_head_depth_rem_10m (HAND), floored at --d-floor-m
    m  scalar 8 (1-D) | 16 (radial)

Any term <=0 / NaN -> WTR NaN for that cell, with per-term nodata accounting
logged (no silent fill, per CLAUDE.md). Zero recharge fabricates WTR=0 and is
never patched. The raw WTR is written unclamped so blow-ups stay visible.

R/K Toth fallback (wtr_rk = R/K, dimensionless) drops the two weakest terms
(H, exact L); kept as an independent regime feature and a cross-check -- where
WTR and R/K disagree strongly, suspect an H/L artifact.

Optional --blend writes g*FAC_dtw + (1-g)*R_dtw for a chosen R prior, for direct
scoring with validate_fac_gwx_wells.py.

Run:
    uv run python utils/build_wtr.py --res 30                          # pilots, ETRM R
    uv run python utils/build_wtr.py --res 30 --r-source reitz --m 16
    uv run python utils/build_wtr.py --res 30 --blend --r-prior str_top2 --beta 1.5 --logwtr0 0.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import scipy.ndimage as ndi
from rasterio.warp import Resampling, reproject

H = Path("/data/ssd2/handily")
HUC8_ROOT = H / "huc8"
STUDY_BASINS = [
    "huc8/mt_ruby",
    "huc8/mt_big_hole",
    "huc8/nv_upper_humboldt",
    "huc8/nm_rio_grande_abq",
]
NODATA = -9999.0

ETRM_RECHARGE = (
    "/nas/etrm/conus/recharge_250m/conus_mean_recharge_2020-2024_albers1km.tif"
)
REITZ_RECHARGE = "/nas/gwx/studies/analysis_ready/reitz_2017_recharge/reitz_effective_recharge_0013_5070.tif"
PERM_LOGK = "/nas/handily/covariates/geology/permeability_logk_x100.tif"
SED_BASINFILL = "/nas/handily/covariates/geology/sediment_thickness_basinfill_m.tif"
SED_AVG = "/nas/handily/covariates/geology/sediment_thickness_avg_m.tif"
RELIEF_WTE = "/data/ssd2/handily/conus/hydrography90m/coarse_wte_idw_relief.tif"
# GLHYMPS permeability k[m^2] -> hydraulic conductivity K. Darcy K = k*rho*g/mu.
PERM_K_M2_TO_M_S = 9.8e6  # k[m^2] -> K[m/s], water ~20 C (rho*g/mu)
SECONDS_PER_YEAR = 3.155693e7


def grid_from(path):
    with rasterio.open(path) as s:
        return s.transform, (s.height, s.width)


def read_on_grid_native(path):
    """Read a raster that is ALREADY on the canonical grid (the L rasters)."""
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32")
        nod = s.nodata
    if nod is not None:
        a[a == nod] = np.nan
    return a


def windowed_areal_mean(path, dst_tr, dst_shape, res, win_km, to_m):
    """Valid-normalized areal-mean recharge over a win_km window (m/yr).

    Genuine zeros (modeled no-recharge) contribute 0 to the mean; only product
    nodata is excluded from the normalization, so cells deep in product-nodata
    return NaN while genuine-zero neighborhoods return 0. win_km<=0 -> point value.
    """
    R = reproject_onto(path, dst_tr, dst_shape, Resampling.bilinear)  # NaN where nodata
    R = R * to_m
    R[R < 0] = np.nan  # Reitz fill (-3.4e38) and any negative are nodata, not recharge
    if win_km <= 0:
        return R
    win = max(1, int(round(win_km * 1000.0 / res)))
    valid = np.isfinite(R).astype("float32")
    num = ndi.uniform_filter(
        np.where(valid > 0, R, 0.0).astype("float32"), size=win, mode="nearest"
    )
    den = ndi.uniform_filter(valid, size=win, mode="nearest")
    return np.where(den > 1e-6, num / den, np.nan).astype("float32")


def reproject_onto(path, dst_tr, dst_shape, resampling, valid_min=None, valid_max=None):
    """Reproject band 1 of `path` onto the canonical grid; nodata/out-of-range -> NaN."""
    with rasterio.open(path) as src:
        src_nod = src.nodata
        dst = np.full(dst_shape, np.nan, "float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nod,
            dst_transform=dst_tr,
            dst_crs="EPSG:5070",
            dst_nodata=np.nan,
            resampling=resampling,
        )
    if valid_min is not None:
        dst[dst < valid_min] = np.nan
    if valid_max is not None:
        dst[dst > valid_max] = np.nan
    return dst


def write_tif(path, arr, transform, dtype="float32", nodata=NODATA):
    arr = np.where(np.isfinite(arr), arr, nodata).astype(dtype)
    prof = dict(
        driver="GTiff",
        dtype=dtype,
        count=1,
        height=arr.shape[0],
        width=arr.shape[1],
        crs="EPSG:5070",
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)


def med(a):
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else None


def pct(a):
    a = a[np.isfinite(a)]
    if not a.size:
        return None
    return {q: float(np.percentile(a, q)) for q in (10, 50, 90)}


def build_basin(rel, args):
    root = H / rel
    res = int(args.res)
    l_name = "wtr_L_edt" if args.l_estimator == "edt" else "wtr_L_dd"
    l_path = root / f"{l_name}_{res}m.tif"
    if not l_path.exists():
        raise FileNotFoundError(f"missing {l_path} -- run build_wtr_L.py first")

    transform, shape = grid_from(l_path)
    L = read_on_grid_native(l_path)
    L[L <= 0] = np.nan
    L_eff = args.l_scale * L
    inside = np.isfinite(L)  # the L raster is already clipped to the basin

    # d = HAND, floored. NaN = outside (verified: 64% NaN, corners NaN, 0=channel).
    hand_path = (
        root / "rem" / f"{rel.split('/')[-1]}_scalable" / "fac_head_depth_rem_10m.tif"
    )
    d = reproject_onto(
        str(hand_path), transform, shape, Resampling.average, valid_min=0.0
    )
    d_eff = np.where(np.isfinite(d), np.maximum(d, args.d_floor_m), np.nan)

    # R recharge -> m/yr, windowed areal mean (mound-feeding recharge over the
    # interfluve). Genuine zeros are kept (WTR=0 -> g=0); only product nodata -> NaN.
    if args.r_source == "etrm":
        R = windowed_areal_mean(
            ETRM_RECHARGE, transform, shape, res, args.r_window_km, to_m=1 / 1000.0
        )
    else:
        R = windowed_areal_mean(
            REITZ_RECHARGE, transform, shape, res, args.r_window_km, to_m=1.0
        )

    # K: reproject GLHYMPS log10(k[m^2])*100 in log space (bilinear == geometric mean),
    # decode to permeability then to hydraulic conductivity via Darcy's K = k*rho*g/mu.
    v = reproject_onto(PERM_LOGK, transform, shape, Resampling.bilinear)
    K = np.power(10.0, v / 100.0) * PERM_K_M2_TO_M_S * SECONDS_PER_YEAR  # m/yr
    K[~np.isfinite(K) | (K <= 0)] = np.nan

    # H sediment thickness (m); weakest term.
    if args.h_source == "basinfill":
        Hsed = reproject_onto(
            SED_BASINFILL, transform, shape, Resampling.bilinear, valid_min=0.0
        )
    else:
        Hsed = reproject_onto(
            SED_AVG, transform, shape, Resampling.bilinear, valid_min=0.0
        )
        Hsed = np.minimum(Hsed, 50.0)
    Hsed[Hsed <= 0] = np.nan

    # --- per-term nodata accounting within the basin (no silent fill) ---
    n_in = int(inside.sum())
    acct = {
        "n_inside": n_in,
        "lost_R": int(np.sum(inside & ~np.isfinite(R))),
        "lost_K": int(np.sum(inside & ~np.isfinite(K))),
        "lost_H": int(np.sum(inside & ~np.isfinite(Hsed))),
        "lost_d": int(np.sum(inside & ~np.isfinite(d_eff))),
        "lost_L": int(np.sum(inside & ~np.isfinite(L_eff))),
    }

    # R finite (incl. genuine 0) is valid; only recharge-product nodata drops a cell.
    valid = (
        inside
        & np.isfinite(R)
        & np.isfinite(K)
        & np.isfinite(Hsed)
        & np.isfinite(d_eff)
        & np.isfinite(L_eff)
    )
    pos = valid & (R > 0)  # genuine-zero R -> recharge-controlled floor (handled below)
    zero = valid & (R <= 0)
    acct["n_valid"] = int(valid.sum())
    acct["n_zeroR"] = int(zero.sum())

    log_rk = np.full(shape, np.nan, "float32")  # R/K, dimensionless
    log_geom = np.full(shape, np.nan, "float32")  # L^2/(H*d), dimensionless
    logwtr = np.full(shape, np.nan, "float32")
    log_geom[valid] = (
        2.0 * np.log10(L_eff[valid]) - np.log10(Hsed[valid]) - np.log10(d_eff[valid])
    )
    log_rk[pos] = np.log10(R[pos]) - np.log10(K[pos])
    logwtr[pos] = log_rk[pos] + log_geom[pos] - np.log10(args.m)
    # genuine zero recharge: no mound -> maximally recharge-controlled (floor, g->0)
    log_rk[zero] = args.logwtr_floor
    logwtr[zero] = args.logwtr_floor
    logwtr[valid] = np.maximum(logwtr[valid], args.logwtr_floor)

    wtr = np.power(10.0, logwtr)
    rk = np.power(10.0, log_rk)
    gate = np.full(shape, np.nan, "float32")
    gate[valid] = 1.0 / (1.0 + np.exp(-args.beta * (logwtr[valid] - args.logwtr0)))

    write_tif(str(root / f"wtr_{res}m.tif"), wtr, transform)
    write_tif(str(root / f"wtr_logwtr_{res}m.tif"), logwtr, transform)
    write_tif(str(root / f"wtr_gate_{res}m.tif"), gate, transform)
    write_tif(str(root / f"wtr_rk_{res}m.tif"), rk, transform)

    run = dict(
        basin=rel,
        res_m=res,
        params=dict(
            m=args.m,
            l_scale=args.l_scale,
            beta=args.beta,
            logwtr0=args.logwtr0,
            d_floor_m=args.d_floor_m,
            l_estimator=args.l_estimator,
            r_source=args.r_source,
            r_window_km=args.r_window_km,
            h_source=args.h_source,
        ),
        nodata_accounting=acct,
        medians=dict(
            R_m_yr=med(R[valid]),
            K_m_yr=med(K[valid]),
            H_m=med(Hsed[valid]),
            d_m=med(d_eff[valid]),
            L_eff_m=med(L_eff[valid]),
            log10_RK=med(log_rk[valid]),
            log10_geom=med(log_geom[valid]),
            log10_WTR=med(logwtr[valid]),
        ),
        percentiles=dict(
            logWTR=pct(logwtr[valid]), gate=pct(gate[valid]), wtr_rk=pct(rk[valid])
        ),
        gate_frac_gt_half=float(np.mean(gate[valid] > 0.5)) if valid.any() else None,
        wtr_frac_gt1=float(np.mean(logwtr[valid] > 0.0)) if valid.any() else None,
    )

    blend_note = ""
    if args.blend:
        blend_note = build_blend(root, rel, transform, shape, d, gate, args)
        run["blend"] = blend_note

    (root / f"wtr_run_{res}m.json").write_text(json.dumps(run, indent=2))

    m = run["medians"]
    print(
        f"  {rel}: valid {acct['n_valid']}/{n_in} ({acct['n_valid'] / max(n_in, 1):.0%})  "
        f"R={m['R_m_yr']:.3g} K={m['K_m_yr']:.3g} H={m['H_m']:.3g} d={m['d_m']:.1f} L={m['L_eff_m']:.0f}  "
        f"med log10[R/K]={m['log10_RK']:.2f} log10[L2/Hd]={m['log10_geom']:.2f} logWTR={m['log10_WTR']:.2f}  "
        f"WTR>1 {run['wtr_frac_gt1']:.0%}  g>.5 {run['gate_frac_gt_half']:.0%}  {blend_note}"
    )


def build_blend(root, rel, transform, shape, d_hand, gate, args):
    """g*FAC_dtw + (1-g)*R_dtw for the chosen R prior. FAC_dtw is the raw HAND depth."""
    res = int(args.res)
    fac_dtw = d_hand  # HAND depth IS the FAC DTW prior (un-floored)
    if args.r_prior == "str_top2":
        rp = root / "str_top2_idw_dtw_100m.tif"
        if not rp.exists():
            return "blend SKIP: no str_top2_idw_dtw_100m.tif"
        r_dtw = reproject_onto(str(rp), transform, shape, Resampling.bilinear)
    else:  # relief_idw: DTW = ground - WTE(elevation)
        wte = reproject_onto(RELIEF_WTE, transform, shape, Resampling.bilinear)
        ground = reproject_onto(
            str(root / "dem_10m.tif"),
            transform,
            shape,
            Resampling.average,
            valid_min=-1e3,
            valid_max=1e4,
        )
        r_dtw = ground - wte
    both = np.isfinite(gate) & np.isfinite(fac_dtw) & np.isfinite(r_dtw)
    blend = np.full(shape, np.nan, "float32")
    blend[both] = gate[both] * fac_dtw[both] + (1.0 - gate[both]) * r_dtw[both]
    out = root / f"wtr_blend_dtw_{res}m_{args.r_prior}.tif"
    write_tif(str(out), blend, transform)
    return f"blend->{out.name} cov {both.mean():.0%}"


def discover_all_fac(res):
    out = []
    for d in sorted(HUC8_ROOT.iterdir()):
        if not d.is_dir() or d.name == "dem_tiles":
            continue
        if not (d / f"wtr_L_edt_{int(res)}m.tif").exists():
            continue
        if (d / f"wtr_{int(res)}m.tif").exists():
            continue
        out.append(f"huc8/{d.name}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basins", nargs="*", default=STUDY_BASINS)
    ap.add_argument(
        "--all-fac",
        action="store_true",
        help="every basin with an L raster lacking WTR (resume-safe)",
    )
    ap.add_argument("--res", type=float, default=30.0)
    ap.add_argument(
        "--m", type=float, default=8.0, help="WTR constant: 8 (1-D) | 16 (radial)"
    )
    ap.add_argument(
        "--l-scale",
        type=float,
        default=1.0,
        help="global L multiplier (calibration sets this)",
    )
    ap.add_argument("--l-estimator", choices=["edt", "dd"], default="edt")
    ap.add_argument(
        "--beta", type=float, default=1.0, help="gate steepness in log10 WTR"
    )
    ap.add_argument(
        "--logwtr0",
        type=float,
        default=0.0,
        help="gate midpoint in log10 WTR (0 -> WTR=1)",
    )
    ap.add_argument("--d-floor-m", type=float, default=1.0)
    ap.add_argument("--r-source", choices=["etrm", "reitz"], default="reitz")
    ap.add_argument(
        "--r-window-km",
        type=float,
        default=10.0,
        help="areal-mean recharge window (mound-feeding footprint); 0 = point sample",
    )
    ap.add_argument(
        "--logwtr-floor",
        type=float,
        default=-9.0,
        help="log10 WTR floor for genuine-zero-recharge / extreme cells (g->0)",
    )
    ap.add_argument("--h-source", choices=["basinfill", "avg"], default="basinfill")
    ap.add_argument(
        "--blend", action="store_true", help="also write g*FAC + (1-g)*R DTW raster"
    )
    ap.add_argument("--r-prior", choices=["str_top2", "relief_idw"], default="str_top2")
    args = ap.parse_args()

    basins = discover_all_fac(args.res) if args.all_fac else args.basins
    print(
        f"build_wtr: {len(basins)} basin(s) @ {int(args.res)}m  R={args.r_source} H={args.h_source} "
        f"L={args.l_estimator} m={args.m} beta={args.beta} logwtr0={args.logwtr0}"
    )
    ok = failed = 0
    failed_ids = []
    for rel in basins:
        try:
            build_basin(rel, args)
            ok += 1
        except Exception as e:
            print(f"  FAILED {rel}: {type(e).__name__}: {e}")
            failed += 1
            failed_ids.append(rel)
    print(f"build_wtr: ok={ok} failed={failed} of {len(basins)}")
    if failed_ids:
        print(f"  failed basins -> rerun to retry: {failed_ids}")


if __name__ == "__main__":
    main()
