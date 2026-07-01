#!/usr/bin/env python
"""Per-HUC8 L (characteristic drainage half-spacing) for the Water Table Ratio.

L is the ONE WTR term whose magnitude is not on disk and whose break must be
*calibrated* rather than asserted -- and it enters WTR squared, so it dominates
the sensitivity. Per the WTR plan we build BOTH candidate estimators and let
blocked-CV calibration pick:

  Estimator A -- per-cell EDT distance-to-stream (`L_edt`, primary candidate).
    Euclidean distance from each cell to the nearest drainage cell. Computed on
    the HALOED stream raster (the FAC window already carries a buffer) BEFORE
    clipping to the basin, so basin-interior cells see drainage just outside the
    cut; a companion `edge_flag` marks cells whose nearest stream may lie beyond
    the haloed window (no silent NaN patch).

  Estimator B -- drainage-density inversion (`L_dd`, cross-check). Literature
    canonical L = 1/(2*Dd). A scalar `Dd = sum(length_m)/basin_area` and a
    windowed spatially-varying `Dd(x)` from the channel-cell fraction in a moving
    window (reuses the build_misc_covariates drainage-density idea) -> L_dd(x).

The stream-network knob (`--stream-source`) is the real L lever -- which lines
count as drainage sets L's magnitude. It uses the pre-built HALOED FAC rasters:
  streams10m   : the full rasterized regional network (streams_10m.tif)   [default]
  strahler_min : stream_order.tif >= --strahler-min
  strahler_topk: top-k Strahler order VALUES (order >= max_order - k + 1)
  fac_extract  : flow_accumulation.tif >= --fac-threshold

Outputs per basin (default grid 30 m, EPSG:5070, aligned to the res lattice):
  wtr_L_edt_{res}m.tif        L_edt after --l-def transform, capped at --l-cap-mult * scalar
  wtr_dist_stream_{res}m.tif  raw EDT distance-to-stream (uncapped, diagnostic)
  wtr_L_edge_flag_{res}m.tif  uint8 1 where the haloed-window edge may truncate the nearest stream
  wtr_L_dd_{res}m.tif         windowed drainage-density-inversion L (capped)
  wtr_L_scalar.json           scalar Dd / L = 1/(2*Dd), basin area, network length
  wtr_L_compare.json          p10/p50/p90 of L_edt, L_dd, scalar + ratios + log-correlation

Non-default --stream-source / --l-def append a suffix so calibration sweeps do
not overwrite the default product.

Run:
    uv run python utils/build_wtr_L.py --res 30                       # five huc8 pilots
    uv run python utils/build_wtr_L.py --basins huc8/mt_ruby --res 30 --l-def local_max
    uv run python utils/build_wtr_L.py --all-fac --res 30             # build-out, resume-safe
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import scipy.ndimage as ndi
from rasterio.features import rasterize
from rasterio.transform import from_origin
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


def out_suffix(stream_source, strahler_min, strahler_topk, fac_threshold, l_def):
    """Filename suffix so non-default sweeps do not clobber the default product."""
    parts = []
    if stream_source == "strahler_min":
        parts.append(f"strmin{strahler_min}")
    elif stream_source == "strahler_topk":
        parts.append(f"strtop{strahler_topk}")
    elif stream_source == "fac_extract":
        parts.append(f"fac{int(fac_threshold)}")
    if l_def != "direct":
        parts.append(l_def)
    return ("_" + "_".join(parts)) if parts else ""


def discover_all_fac(suffix, res):
    """FAC basins under huc8/ lacking the L_edt output (resume-safe)."""
    out = []
    for d in sorted(HUC8_ROOT.iterdir()):
        if not d.is_dir() or d.name == "dem_tiles":
            continue
        fac = d / "rem" / f"{d.name}_scalable" / "fac_head_depth_rem_10m.tif"
        if not fac.exists() or (d / f"wtr_L_edt_{int(res)}m{suffix}.tif").exists():
            continue
        out.append(f"huc8/{d.name}")
    return out


def aligned_grid(bounds, res):
    """Floor/ceil bounds onto the global res lattice (matches build_str7_idw_raster)."""
    minx, miny, maxx, maxy = bounds
    minx, miny = np.floor(minx / res) * res, np.floor(miny / res) * res
    maxx, maxy = np.ceil(maxx / res) * res, np.ceil(maxy / res) * res
    width, height = int(round((maxx - minx) / res)), int(round((maxy - miny) / res))
    return from_origin(minx, maxy, res, res), width, height


def read_stream_mask(root, stream_source, strahler_min, strahler_topk, fac_threshold):
    """Boolean drainage mask on its native (haloed) 10 m grid + that grid's transform."""
    if stream_source == "streams10m":
        path = root / "streams_10m.tif"
        with rasterio.open(path) as s:
            a, nod, tr, crs, bnds = s.read(1), s.nodata, s.transform, s.crs, s.bounds
        mask = np.isfinite(a) & (a > 0)
        if nod is not None:
            mask &= a != nod
    elif stream_source in ("strahler_min", "strahler_topk"):
        path = root / "stream_order.tif"
        with rasterio.open(path) as s:
            a, nod, tr, crs, bnds = s.read(1), s.nodata, s.transform, s.crs, s.bounds
        valid = np.isfinite(a) & (a > 0)
        if nod is not None:
            valid &= a != nod
        if stream_source == "strahler_min":
            mask = valid & (a >= strahler_min)
        else:
            mx = int(a[valid].max())
            mask = valid & (a >= mx - strahler_topk + 1)
    elif stream_source == "fac_extract":
        path = root / "flow_accumulation.tif"
        with rasterio.open(path) as s:
            a, nod, tr, crs, bnds = s.read(1), s.nodata, s.transform, s.crs, s.bounds
        valid = np.isfinite(a)
        if nod is not None:
            valid &= a != nod
        mask = valid & (a >= fac_threshold)
    else:
        raise ValueError(f"unknown --stream-source {stream_source}")
    if mask.sum() == 0:
        raise ValueError(
            f"empty drainage mask from {path.name} (source={stream_source})"
        )
    return mask, tr, crs, bnds, str(path)


def reproject_to(
    src, src_tr, src_crs, dst_tr, dst_shape, resampling, src_nodata, dst_nodata
):
    dst = np.full(dst_shape, dst_nodata, "float32")
    reproject(
        source=src.astype("float32"),
        destination=dst,
        src_transform=src_tr,
        src_crs=src_crs,
        dst_transform=dst_tr,
        dst_crs="EPSG:5070",
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return dst


def write_tif(path, arr, transform, dtype="float32", nodata=NODATA):
    if dtype == "float32":
        arr = np.where(np.isfinite(arr), arr, nodata).astype("float32")
    else:
        arr = arr.astype(dtype)
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
        predictor=3 if dtype == "float32" else 2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)


def pct(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return dict(
        p10=float(np.percentile(a, 10)),
        p50=float(np.percentile(a, 50)),
        p90=float(np.percentile(a, 90)),
        n=int(a.size),
    )


def build_basin(rel, args):
    root = H / rel
    res = args.res
    suffix = out_suffix(
        args.stream_source,
        args.strahler_min,
        args.strahler_topk,
        args.fac_threshold,
        args.l_def,
    )
    boundary = gpd.read_file(root / "basin_boundary.fgb").to_crs(5070)

    # --- haloed EDT grid: mask source bounds aligned to res -> EDT here, crop later ---
    mask10, src_tr, src_crs, src_bnds, src_path = read_stream_mask(
        root,
        args.stream_source,
        args.strahler_min,
        args.strahler_topk,
        args.fac_threshold,
    )
    htr, hW, hH = aligned_grid(
        (src_bnds.left, src_bnds.bottom, src_bnds.right, src_bnds.top), res
    )
    hmask = (
        reproject_to(
            mask10,
            src_tr,
            src_crs,
            htr,
            (hH, hW),
            Resampling.max,
            src_nodata=None,
            dst_nodata=0.0,
        )
        > 0.5
    )

    dist_h = ndi.distance_transform_edt(~hmask, sampling=res).astype(
        "float32"
    )  # meters
    if args.l_def == "direct":
        L_h = dist_h
    elif args.l_def == "local_max":
        win = max(1, int(round(args.l_window_m / res)))
        L_h = ndi.maximum_filter(dist_h, size=win).astype("float32")
    else:
        raise ValueError(f"unknown --l-def {args.l_def}")

    # edge flag: nearest stream may be beyond the haloed window where dist >= dist-to-edge
    ii = np.arange(hH)[:, None].astype("float32") * res
    jj = np.arange(hW)[None, :].astype("float32") * res
    d_edge = np.minimum(
        np.minimum(ii, (hH - 1) * res - ii), np.minimum(jj, (hW - 1) * res - jj)
    )
    edge_h = (dist_h >= d_edge).astype("float32")

    # --- canonical per-basin output grid (basin bounds aligned to res) ---
    tr, W, height = aligned_grid(tuple(boundary.total_bounds), res)
    inside = (
        rasterize(
            [(g, 1) for g in boundary.geometry],
            out_shape=(height, W),
            transform=tr,
            fill=0,
            dtype="uint8",
        )
        > 0
    )

    L_edt = reproject_to(
        L_h, htr, src_crs, tr, (height, W), Resampling.bilinear, np.nan, np.nan
    )
    dist = reproject_to(
        dist_h, htr, src_crs, tr, (height, W), Resampling.bilinear, np.nan, np.nan
    )
    edge = (
        reproject_to(edge_h, htr, src_crs, tr, (height, W), Resampling.max, 0.0, 0.0)
        > 0.5
    )
    chan = (
        reproject_to(
            hmask.astype("float32"),
            htr,
            src_crs,
            tr,
            (height, W),
            Resampling.max,
            0.0,
            0.0,
        )
        > 0.5
    )

    # --- scalar drainage density / L = 1/(2*Dd) ---
    st = gpd.read_file(root / "streams_regional.fgb").to_crs(5070)
    length_m = float(st["length_m"].sum())
    area_m2 = float(boundary.geometry.area.sum())
    dd_scalar = length_m / area_m2  # 1/m
    L_scalar = 1.0 / (2.0 * dd_scalar)
    cap = args.l_cap_mult * L_scalar if args.l_cap_mult > 0 else None

    # --- windowed L_dd = res / (2 * channel_fraction(x)) ---
    win = max(1, int(round(args.l_window_m / res)))
    chan_f = ndi.uniform_filter(
        chan.astype("float32"), size=win, mode="constant", cval=0.0
    )
    valid_f = ndi.uniform_filter(
        inside.astype("float32"), size=win, mode="constant", cval=0.0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(
            valid_f > 1e-6, chan_f / valid_f, np.nan
        )  # channel-cell fraction
        L_dd = np.where(frac > 0, res / (2.0 * frac), np.nan).astype("float32")
    # frac==0 (no drainage in window) -> L diverges; bound at the literature scalar cap, count it
    n_diverge = int(np.sum(inside & np.isfinite(frac) & (frac <= 0)))
    if cap is not None:
        L_dd = np.where(np.isnan(L_dd) & inside & (frac == 0), cap, L_dd)
        L_dd = np.minimum(L_dd, cap)
        L_edt_capped = np.minimum(L_edt, cap)
    else:
        L_edt_capped = L_edt
    n_capped_edt = int(np.sum(inside & (L_edt > (cap if cap else np.inf))))

    # --- clip to basin ---
    L_edt_capped = np.where(inside, L_edt_capped, np.nan)
    dist = np.where(inside, dist, np.nan)
    L_dd = np.where(inside, L_dd, np.nan)
    edge_u8 = np.where(inside, edge.astype("uint8"), 255).astype("uint8")

    write_tif(str(root / f"wtr_L_edt_{int(res)}m{suffix}.tif"), L_edt_capped, tr)
    write_tif(str(root / f"wtr_dist_stream_{int(res)}m{suffix}.tif"), dist, tr)
    write_tif(str(root / f"wtr_L_dd_{int(res)}m{suffix}.tif"), L_dd, tr)
    write_tif(
        str(root / f"wtr_L_edge_flag_{int(res)}m{suffix}.tif"),
        edge_u8,
        tr,
        dtype="uint8",
        nodata=255,
    )

    scalar_json = dict(
        basin=rel,
        res_m=res,
        stream_source=args.stream_source,
        network_length_m=length_m,
        basin_area_m2=area_m2,
        Dd_per_m=dd_scalar,
        L_scalar_m=L_scalar,
        l_cap_mult=args.l_cap_mult,
        L_cap_m=cap,
        n_Ldd_window_no_drainage=n_diverge,
        n_Ledt_capped=n_capped_edt,
        edge_flag_fraction=float(np.mean(edge[inside])) if inside.any() else None,
    )
    (root / f"wtr_L_scalar{suffix}.json").write_text(json.dumps(scalar_json, indent=2))

    e = L_edt_capped[inside]
    dd = L_dd[inside]
    both = np.isfinite(e) & np.isfinite(dd) & (e > 0) & (dd > 0)
    logcorr = None
    if both.sum() > 100:
        le, ld = np.log10(e[both]), np.log10(dd[both])
        logcorr = float(np.corrcoef(le, ld)[0, 1])
    compare = dict(
        basin=rel,
        L_edt=pct(e),
        L_dd=pct(dd),
        L_scalar_m=L_scalar,
        ratio_medLedt_scalar=(float(np.nanmedian(e)) / L_scalar)
        if np.isfinite(np.nanmedian(e))
        else None,
        ratio_medLdd_scalar=(float(np.nanmedian(dd)) / L_scalar)
        if np.isfinite(np.nanmedian(dd))
        else None,
        log10_corr_Ledt_Ldd=logcorr,
    )
    (root / f"wtr_L_compare{suffix}.json").write_text(json.dumps(compare, indent=2))

    edt50 = compare["L_edt"]["p50"] if compare["L_edt"] else float("nan")
    dd50 = compare["L_dd"]["p50"] if compare["L_dd"] else float("nan")
    cap_s = f"{cap:.0f}m" if cap else "off"
    print(
        f"  {rel}: src={Path(src_path).name} grid {W}x{height}@{int(res)}m  "
        f"L_scalar={L_scalar:.0f}m  L_edt p50={edt50:.0f}  L_dd p50={dd50:.0f}  "
        f"logcorr={logcorr}  edge={scalar_json['edge_flag_fraction']:.1%}  cap={cap_s}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--basins",
        nargs="*",
        default=STUDY_BASINS,
        help="rel-paths under /data/ssd2/handily",
    )
    ap.add_argument(
        "--all-fac",
        action="store_true",
        help="every FAC basin lacking the L output (resume-safe)",
    )
    ap.add_argument(
        "--res", type=float, default=30.0, help="output grid resolution (m)"
    )
    ap.add_argument(
        "--stream-source",
        choices=["streams10m", "strahler_min", "strahler_topk", "fac_extract"],
        default="streams10m",
        help="which drainage lines define L (the real L lever)",
    )
    ap.add_argument("--strahler-min", type=int, default=2)
    ap.add_argument("--strahler-topk", type=int, default=2)
    ap.add_argument("--fac-threshold", type=float, default=5000.0)
    ap.add_argument(
        "--l-def",
        choices=["direct", "local_max"],
        default="direct",
        help="direct=per-cell EDT; local_max=neighborhood max over --l-window-m",
    )
    ap.add_argument(
        "--l-window-m",
        type=float,
        default=2000.0,
        help="window for local_max and windowed L_dd",
    )
    ap.add_argument(
        "--l-cap-mult",
        type=float,
        default=3.0,
        help="cap L at this * scalar half-spacing (0=off)",
    )
    args = ap.parse_args()

    suffix = out_suffix(
        args.stream_source,
        args.strahler_min,
        args.strahler_topk,
        args.fac_threshold,
        args.l_def,
    )
    basins = discover_all_fac(suffix, args.res) if args.all_fac else args.basins
    print(
        f"build_wtr_L: {len(basins)} basin(s) @ {args.res}m  source={args.stream_source} l_def={args.l_def}{suffix}"
    )
    ok = failed = 0
    failed_ids = []
    for rel in basins:
        try:
            build_basin(rel, args)
            ok += 1
        except Exception as e:  # isolate per-basin failures over a large batch
            print(f"  FAILED {rel}: {type(e).__name__}: {e}")
            failed += 1
            failed_ids.append(rel)
    print(f"build_wtr_L: ok={ok} failed={failed} of {len(basins)}")
    if failed_ids:
        print(f"  failed basins -> rerun to retry: {failed_ids}")


if __name__ == "__main__":
    main()
