"""Smooth GNN DTW/WTE along straight FAC channels (playa / fan striping fix).

WBT D8 routing on filled flats resolves ties into long, dead-straight, parallel
"channels". The FAC-REM sags along each one, the lattice GNN sees that sag in its
``fac_rem_dtw_m`` feature, and the 10 m render adds the fine FAC term on top, so
playa bottoms and fan toes come out as 500 m-period stripes (Carson Sink, Gabbs
Valley, Dixie). Nothing in the field data supports the stripes; they are a routing
artefact.

This is a post-pass on the rendered rasters, not a change to the model or recipe:

1. Find straight runs in ``streams_regional_scalable_v2.fgb``: station every
   ``step_m`` along each reach, flag every ``window_m`` window whose chord/arc ratio
   is >= ``chord_ratio``, merge flagged stations into runs, keep runs >=
   ``min_run_m``. Attribute each run with its ground slope (fit along the run on
   ``gnn_dtw_100m + gnn_wte_100m``, i.e. the lattice surface) and smooth only runs
   flatter than ``max_slope_pct``: fix_flats invents the straight lines on flats, so
   those carry the stripes, while straight fan channels on 1-5 % slopes are real
   routing and smoothing them hurts (Dixie wells near >2 % runs lost ~7 m MAD).
2. Rasterise the runs buffered by ``buffer_m`` into a weight in [0, 1], feathered
   to 0 over ``feather_m`` outside the buffer (nested buffers, ``feather_steps`` of
   them, so no distance transform is needed on the 10 m grid).
3. Inside that weight, and only on flat ground (slope of the 300 m-smoothed lattice
   surface <= ``max_cell_slope_pct``; the 700 m corridor otherwise reaches onto canyon
   walls and mountain fronts, which both get smoothed and feed 50-150 m DTW into the
   playa: Goose and Eureka-Saline tails of +35-45 m; the mask is evaluated at 100 m
   and again at the layer's own resolution, so 10 m wash relief that the 10 m render
   follows is left alone), replace WTE with a NaN-aware
   Gaussian (``sigma_m``) of the flat-ground WTE and move DTW by the same amount
   (DTW = DEM - WTE, so no DEM is needed). Shifts are capped at ``max_shift_m``.
   A plain Gaussian biases a curved surface by sigma^2/2 * laplacian (a playa
   trough with WTE rising onto the fans reads shallower after smoothing), so by
   default the smoother is the curvature-corrected ``2 G(sigma) - G(sigma sqrt 2)``,
   which cancels that term and still kills the ~500 m stripe period.
   Cells at exactly 0 m DTW (water-zeroed) are left alone; DTW is re-clamped >= 0.

Applied to both the 100 m lattice outputs and the 10 m render, side by side
(``*<suffix>.tif``, default ``_sm``) by default, or in place with ``--inplace`` (the originals are kept
as ``*_presmooth.tif`` the first time). A ``straight_smooth_run.json`` records
the parameters and the fraction of valid cells touched.

Example (pilot, side by side):

    python utils/postprocess_straight_fac.py --model-name \
        gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold_parity \
        --basins 16060002 16050203 --huc8-root /data/ssd2/handily/huc8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio import features
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString

log = logging.getLogger("postprocess_straight_fac")

HUC8_ROOT = Path(os.environ.get("HANDILY_HUC8_ROOT", "/data/ssd2/handily/huc8"))
STREAMS_NAME = "streams_regional_scalable_v2.fgb"
RUNS_NAME = "straight_runs_v2.fgb"
LAYERS = (
    ("gnn_wte_100m.tif", "gnn_dtw_100m.tif"),
    ("gnn_wte_10m.tif", "gnn_dtw_10m.tif"),
)


def straight_runs(
    streams: gpd.GeoDataFrame,
    window_m: float,
    step_m: float,
    chord_ratio: float,
    min_run_m: float,
) -> gpd.GeoDataFrame:
    """Sub-reach runs where every ``window_m`` window is straighter than ``chord_ratio``."""
    n = int(round(window_m / step_m))
    rows = []
    for sid, geom in zip(streams["stream_id"], streams.geometry):
        if geom.geom_type != "LineString" or geom.length < window_m:
            continue
        s = np.arange(0.0, geom.length + 1e-6, step_m)
        pts = shapely.get_coordinates(shapely.line_interpolate_point(geom, s))
        if len(pts) <= n:
            continue
        chord = np.hypot(*(pts[n:] - pts[:-n]).T)
        hit = chord / window_m >= chord_ratio  # window k covers stations k..k+n
        flags = np.zeros(len(pts), dtype=bool)
        for k in np.flatnonzero(hit):
            flags[k : k + n + 1] = True
        edges = np.diff(np.r_[0, flags.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            run_len = (b - a - 1) * step_m
            if b - a >= 2 and run_len >= min_run_m:
                rows.append(
                    {
                        "stream_id": sid,
                        "len_m": float(run_len),
                        "geometry": LineString(pts[a:b]),
                    }
                )
    if not rows:  # steep basins can have no 1 km straight window at all
        return gpd.GeoDataFrame(
            {"stream_id": [], "len_m": []},
            geometry=gpd.GeoSeries([], crs=streams.crs),
            crs=streams.crs,
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=streams.crs)


def run_slopes(
    runs: gpd.GeoDataFrame, dtw_path: Path, wte_path: Path, step_m: float = 100.0
) -> np.ndarray:
    """|slope| in percent of the lattice ground surface (DTW + WTE) fitted along each run."""
    out = np.full(len(runs), np.nan)
    with rasterio.open(dtw_path) as d, rasterio.open(wte_path) as w:
        for i, g in enumerate(runs.geometry):
            s = np.arange(0.0, g.length + 1e-6, step_m)
            xy = shapely.get_coordinates(shapely.line_interpolate_point(g, s))
            dtw = np.array([v[0] for v in d.sample(xy)], dtype="float64")
            wte = np.array([v[0] for v in w.sample(xy)], dtype="float64")
            if d.nodata is not None:
                dtw[dtw == d.nodata] = np.nan
            if w.nodata is not None:
                wte[wte == w.nodata] = np.nan
            z = dtw + wte
            ok = np.isfinite(z)
            if ok.sum() >= 3:
                out[i] = abs(np.polyfit(s[ok], z[ok], 1)[0]) * 100.0
    return out


def run_weight(
    runs: gpd.GeoDataFrame,
    transform,
    shape: tuple[int, int],
    buffer_m: float,
    feather_m: float,
    feather_steps: int = 6,
) -> np.ndarray:
    """1 inside the buffered runs, linear feather to 0 over ``feather_m``, 0 beyond.

    The feather is the mean of ``feather_steps`` nested buffer masks (one uint8 plane at
    a time instead of a float64 distance transform), blurred by one ring width so the
    ramp is continuous rather than a staircase.
    """
    acc = np.zeros(shape, dtype="float32")
    if len(runs) == 0:
        return acc
    geoms = list(runs.geometry)
    for k in range(feather_steps):
        radius = buffer_m + feather_m * k / feather_steps
        acc += features.rasterize(
            ((g.buffer(radius), 1) for g in geoms),
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        )
    acc /= feather_steps
    # the nested buffers give a staircase of 1/feather_steps per ring; where the delta is
    # several metres that reads as 1-2 m contour steps every ring (Gabbs playa margin),
    # so blur the weight by one ring width to make the ramp continuous
    ring_px = feather_m / feather_steps / abs(transform.a)
    if ring_px > 0.3:
        acc = gaussian_filter(acc, ring_px, mode="nearest")
        np.clip(acc, 0.0, 1.0, out=acc)
    return acc


def flat_ground(
    dtw: np.ndarray,
    wte: np.ndarray,
    res_m: float,
    max_slope_pct: float,
    smooth_m: float = 300.0,
) -> np.ndarray:
    """Cells whose terrain-scale ground slope (DTW + WTE, Gaussian ``smooth_m``) is <= the gate."""
    z = nan_gaussian(dtw + wte, smooth_m / res_m)
    gy, gx = np.gradient(z, res_m)
    slope = np.hypot(gx, gy) * 100.0
    return np.isfinite(slope) & (slope <= max_slope_pct)


def valid_n(arr: np.ndarray) -> int:
    return int(np.isfinite(arr).sum())


def upsample_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour blow-up of a coarse mask onto a finer grid of the same extent."""
    fy = int(round(shape[0] / mask.shape[0]))
    fx = int(round(shape[1] / mask.shape[1]))
    up = np.repeat(np.repeat(mask, fy, axis=0), fx, axis=1)
    out = np.zeros(shape, dtype=bool)
    h, w = min(shape[0], up.shape[0]), min(shape[1], up.shape[1])
    out[:h, :w] = up[:h, :w]
    return out


def nan_gaussian(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    """Normalised-convolution Gaussian: NaNs carry no weight and stay NaN."""
    valid = np.isfinite(arr)
    vals = gaussian_filter(
        np.where(valid, arr, 0.0).astype("float32"), sigma_px, mode="nearest"
    )
    wts = gaussian_filter(valid.astype("float32"), sigma_px, mode="nearest")
    ok = valid & (wts > 1e-3)
    vals[ok] /= wts[ok]
    vals[~ok] = np.nan
    return vals


def smooth_pair(
    wte: np.ndarray,
    dtw: np.ndarray,
    weight: np.ndarray,
    sigma_px: float,
    flat: np.ndarray,
    max_shift: float,
    curvature_correct: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Blend smoothed flat-ground WTE in by ``weight``; shift DTW the same; clamp >= 0."""
    touch = (weight > 0) & flat & np.isfinite(wte) & np.isfinite(dtw) & (dtw != 0.0)
    wte_in = np.where(flat, wte, np.nan).astype("float32")
    wte_s = nan_gaussian(wte_in, sigma_px)
    if curvature_correct:
        wte_s = 2.0 * wte_s - nan_gaussian(wte_in, sigma_px * np.sqrt(2.0))
    delta = np.where(touch & np.isfinite(wte_s), weight * (wte_s - wte), 0.0).astype(
        "float32"
    )
    delta = np.clip(delta, -max_shift, max_shift)
    wte_out = wte + delta
    dtw_out = dtw - delta
    neg = touch & (dtw_out < 0)
    wte_out[neg] += dtw_out[neg]  # WTE back to the surface where the shift overshot
    dtw_out[neg] = 0.0
    return wte_out, dtw_out, int(touch.sum())


def _read(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        prof = src.profile
        tags = src.tags()
        predictor = src.tags(ns="IMAGE_STRUCTURE").get("PREDICTOR")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    if predictor is not None:
        prof["predictor"] = int(predictor)
    return arr, {"profile": prof, "tags": tags}


def _write(path: Path, arr: np.ndarray, meta: dict) -> None:
    prof = dict(meta["profile"])
    nodata = prof.get("nodata")
    out = arr.astype("float32")
    if nodata is not None:
        out = np.where(np.isfinite(out), out, nodata).astype("float32")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)
        dst.update_tags(
            **{
                k: v
                for k, v in meta["tags"].items()
                if not k.startswith("AREA_OR_POINT")
            }
        )


def process_basin(basin: str, model_name: str, args: argparse.Namespace) -> dict:
    bdir = args.huc8_root / basin
    gdir = bdir / "gnn" / model_name
    streams_path = bdir / STREAMS_NAME
    if not streams_path.exists():
        raise SystemExit(
            f"{basin}: no {STREAMS_NAME} (scalable_v2 recipe not built here)"
        )
    streams = gpd.read_file(streams_path)
    runs = straight_runs(
        streams, args.window_m, args.step_m, args.chord_ratio, args.min_run_m
    )
    runs["slope_pct"] = run_slopes(runs, gdir / LAYERS[0][1], gdir / LAYERS[0][0])
    runs["smooth"] = runs["slope_pct"] <= args.max_slope_pct
    if len(runs):
        runs.to_file(bdir / RUNS_NAME, driver="FlatGeobuf")
    used = runs[runs["smooth"]]
    rec = {
        "basin": basin,
        "model": model_name,
        "n_reaches": int(len(streams)),
        "network_km": float(streams.length.sum() / 1000),
        "n_runs": int(len(runs)),
        "runs_km": float(runs["len_m"].sum() / 1000) if len(runs) else 0.0,
        "n_runs_flat": int(len(used)),
        "runs_flat_km": float(used["len_m"].sum() / 1000) if len(used) else 0.0,
        "params": {
            k: getattr(args, k)
            for k in (
                "window_m",
                "step_m",
                "chord_ratio",
                "min_run_m",
                "buffer_m",
                "feather_m",
                "feather_steps",
                "sigma_m",
                "max_slope_pct",
                "max_cell_slope_pct",
                "max_shift_m",
                "plain_gaussian",
            )
        },
        "layers": {},
    }
    log.info(
        "%s: %d straight runs, %.0f of %.0f network km; %d flat (<= %.2f%%), %.0f km",
        basin,
        rec["n_runs"],
        rec["runs_km"],
        rec["network_km"],
        rec["n_runs_flat"],
        args.max_slope_pct,
        rec["runs_flat_km"],
    )
    flat100 = None
    if len(used):
        dtw100, m100 = _read(gdir / LAYERS[0][1])
        wte100, _ = _read(gdir / LAYERS[0][0])
        flat100 = flat_ground(
            dtw100, wte100, abs(m100["profile"]["transform"].a), args.max_cell_slope_pct
        )
        rec["frac_flat_100m"] = float(flat100.sum() / max(np.isfinite(dtw100).sum(), 1))
        del dtw100, wte100
    for wte_name, dtw_name in LAYERS if len(used) else ():
        wte_path, dtw_path = gdir / wte_name, gdir / dtw_name
        if not wte_path.exists() or not dtw_path.exists():
            log.info("%s: skipping %s (missing)", basin, dtw_name)
            continue
        src_wte, src_dtw = wte_path, dtw_path
        if args.inplace:
            pre_wte = wte_path.with_name(wte_path.stem + "_presmooth.tif")
            pre_dtw = dtw_path.with_name(dtw_path.stem + "_presmooth.tif")
            if pre_wte.exists() and pre_dtw.exists():
                src_wte, src_dtw = (
                    pre_wte,
                    pre_dtw,
                )  # idempotent re-runs start from the raw render
        wte, meta = _read(src_wte)
        dtw, meta_d = _read(src_dtw)
        res = abs(meta["profile"]["transform"].a)
        weight = run_weight(
            used,
            meta["profile"]["transform"],
            wte.shape,
            args.buffer_m,
            args.feather_m,
            args.feather_steps,
        )
        # flat at the 100 m terrain scale AND at the layer's own resolution: a wash
        # floor that is flat at 100 m can carry 10 m relief that the 10 m render
        # follows (15010005: 10 m WTE 30 m below the 100 m WTE, shifts at the cap)
        flat = flat_ground(dtw, wte, res, args.max_cell_slope_pct)
        if wte.shape != flat100.shape:
            flat &= upsample_mask(flat100, wte.shape)
        else:
            flat &= flat100
        rec[f"frac_flat_{int(res)}m"] = float(flat.sum() / max(valid_n(dtw), 1))
        wte_out, dtw_out, n_touch = smooth_pair(
            wte,
            dtw,
            weight,
            args.sigma_m / res,
            flat,
            args.max_shift_m,
            not args.plain_gaussian,
        )
        valid = int(np.isfinite(dtw).sum())
        if args.inplace:
            if not pre_wte.exists():
                os.replace(wte_path, pre_wte)
                os.replace(dtw_path, pre_dtw)
            out_wte, out_dtw = wte_path, dtw_path
        else:
            out_wte = wte_path.with_name(wte_path.stem + args.suffix + ".tif")
            out_dtw = dtw_path.with_name(dtw_path.stem + args.suffix + ".tif")
        _write(out_wte, wte_out, meta)
        _write(out_dtw, dtw_out, meta_d)
        moved = dtw_out - dtw
        m = np.isfinite(moved) & (moved != 0.0)
        rec["layers"][dtw_name] = {
            "res_m": res,
            "n_valid": valid,
            "n_touched": n_touch,
            "frac_touched": n_touch / valid if valid else 0.0,
            "dtw_shift_p05_p50_p95_m": [
                float(v) for v in np.percentile(moved[m], [5, 50, 95])
            ]
            if m.any()
            else None,
            "frac_touched_at_cap": float(
                np.mean(np.abs(moved[m]) >= args.max_shift_m - 1e-3)
            )
            if m.any()
            else 0.0,
            "out": str(out_dtw),
        }
        log.info(
            "%s: %s %.1f%% of valid cells touched, shift p05/p50/p95 %s",
            basin,
            dtw_name,
            100 * rec["layers"][dtw_name]["frac_touched"],
            rec["layers"][dtw_name]["dtw_shift_p05_p50_p95_m"],
        )
    if not len(used):
        log.info("%s: no flat straight runs; rasters left untouched", basin)
    rec_name = (
        "straight_smooth_run.json"
        if args.inplace
        else f"straight_smooth_run{args.suffix}.json"
    )
    (gdir / rec_name).write_text(json.dumps(rec, indent=2))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--basins", nargs="+", required=True)
    ap.add_argument("--huc8-root", type=Path, default=HUC8_ROOT)
    ap.add_argument(
        "--window-m", type=float, default=1000.0, help="straightness test window"
    )
    ap.add_argument(
        "--step-m", type=float, default=50.0, help="station spacing along reaches"
    )
    ap.add_argument(
        "--chord-ratio", type=float, default=0.995, help="chord/arc >= this is straight"
    )
    ap.add_argument(
        "--min-run-m", type=float, default=1000.0, help="keep runs at least this long"
    )
    ap.add_argument(
        "--max-slope-pct",
        type=float,
        default=1.0,
        help="smooth only runs flatter than this",
    )
    ap.add_argument(
        "--max-cell-slope-pct",
        type=float,
        default=2.0,
        help="only cells on ground flatter than this are smoothed or feed the smoother",
    )
    ap.add_argument(
        "--max-shift-m", type=float, default=10.0, help="cap on |DTW shift| per cell"
    )
    ap.add_argument(
        "--buffer-m", type=float, default=400.0, help="full-weight corridor half-width"
    )
    ap.add_argument(
        "--feather-m", type=float, default=300.0, help="weight ramps to 0 over this"
    )
    ap.add_argument(
        "--feather-steps",
        type=int,
        default=6,
        help="nested buffers in the feather ramp",
    )
    ap.add_argument(
        "--sigma-m",
        type=float,
        default=600.0,
        help="Gaussian sigma for the smoothed WTE",
    )
    ap.add_argument(
        "--plain-gaussian",
        action="store_true",
        help="plain Gaussian instead of the curvature-corrected 2G(s) - G(s sqrt 2)",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="overwrite the renders (keeps *_presmooth.tif)",
    )
    ap.add_argument(
        "--suffix",
        default="_sm",
        help="side-by-side output suffix (ignored with --inplace)",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    for b in args.basins:
        process_basin(b, args.model_name, args)


if __name__ == "__main__":
    main()
