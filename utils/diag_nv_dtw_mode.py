"""E0 diagnostic: the 3 m mode in the Nevada ``r1e`` depth-to-water render.

``notes/SHALLOW_LEVER_EXPERIMENTS.md`` section 4, experiment E0. The 72-basin
Nevada render of the model of record ``gnn_conus_monitoring_water_src_r1e``
carries a large mode at 3.00 m depth to water (202,098 cells at exactly 3.000 m,
9.47 % of the footprint in 3.00-3.25 m,
``notes/NV_MA_HANDILY_SHAPE_COMPARISON.md``). The model is a four-expert gate

    pred_wte = w_fac * fac_wte + w_deep * deep_wte
             + w_mirror * (z_surf - 3.0) + w_head * head_wte     (metres)

and ``DTW = z_surf - pred_wte`` (metres). The mirror expert is the land surface
minus the constant ``mirror_depth_m = 3.0 m``, so its own DTW is exactly 3.0 m
everywhere. Hypothesis under test: at the mode cells the gate selects the mirror
nearly outright (``w_mirror`` near 1), so the prediction collapses onto the
constant rather than the free head correcting it.

Aggregation caveat that the mixture check must carry (``infer_conus_gnn.run_folds``
docstring): the shipped WTE, head WTE, sigma and fold spread are fold MEDIANS
over the eight fold checkpoints while the gate weights are a renormalised fold
MEAN, so the per-fold mixture identity (checked to 1e-3 m at inference) is not
expected to close exactly on the fold-aggregated rasters. The closure error is
measured here at cells where the FAC expert is absent, and it prices the
"implied FAC expert" inversion of item 3.

Footprint: each basin render is a rectangle and the rectangles overlap, so each
basin is clipped to its WBD HUC8 polygon before it is pasted into one 100 m
EPSG:5070 mosaic, exactly as ``utils/compare_nv_ma_handily_shape.py`` does; the
script asserts that no valid cell is claimed by two basins after the clip.

Outputs (CSV + PNG + ``run_summary.json``) land in
``/data/ssd2/handily/nv/regional/shallow_levers/e0_mode_diag``.

Run::

    uv run --directory /home/dgketchum/code/handily python \
        /home/dgketchum/code/handily/utils/diag_nv_dtw_mode.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "figures" / "nwi"))

from nwi_common import HUC8_POLYS, rendered_basins  # noqa: E402

RENDER_ARM = "gnn_conus_monitoring_water_src_r1e"
HUC8_ROOT = Path("/data/ssd2/handily/huc8")
DEM_100M = Path("/data/ssd2/handily/conus/covariates/elev48i0100a.tif")
MANIFEST = Path(
    "/data/ssd2/handily/conus/wte_gnn/gnn_conus_monitoring_water_src_r1e/"
    "models/inference_manifest.json"
)
OUT_DIR = Path("/data/ssd2/handily/nv/regional/shallow_levers/e0_mode_diag")

CELL = 100.0  # m
NODATA = -9999.0
CELL_KM2 = (CELL / 1000.0) ** 2  # 0.01 km2 per 100 m cell

#: The mode window (m depth to water) the experiment plan names.
MODE_LO, MODE_HI = 2.9, 3.3
#: Histogram support (m) and bin width (m).
HIST_LO, HIST_HI, HIST_W = -1.0, 6.0, 0.05

#: Single-band layers pulled from each basin render, keyed by output name.
SINGLE_LAYERS = {
    "dtw": "gnn_dtw_100m.tif",
    "wte": "gnn_wte_100m.tif",
    "sigma": "gnn_sigma_100m.tif",
    "fold_spread": "gnn_fold_spread_100m.tif",
    "r_wte": "gnn_r_wte_100m.tif",
    "head_wte": "gnn_head_wte_100m.tif",
    "deep_wte": "gnn_deep_wte_100m.tif",
}
GATE_FILE = "gnn_gate_w_100m.tif"
GATE_BANDS = ("w_fac", "w_deep", "w_mirror", "w_head")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def quant(v: np.ndarray, name: str, unit: str = "m") -> dict:
    """p10 / median / p90 / mean of a 1-D sample, all in ``unit``."""
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError(f"{name}: no finite values; investigate before summarising")
    return {
        "quantity": name,
        "unit": unit,
        "n": int(v.size),
        "p10": float(np.percentile(v, 10)),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "mean": float(v.mean()),
    }


# ---------------------------------------------------------------------------
# Mosaic
# ---------------------------------------------------------------------------


def build_mosaic(basins: list[str]) -> dict:
    """Clip every basin render to its HUC8 polygon and paste into one grid.

    Same construction as ``compare_nv_ma_handily_shape.build_mosaic``, extended
    to the four-band gate raster and the three extra expert-surface layers.
    """
    paths = {b: HUC8_ROOT / b / "gnn" / RENDER_ARM for b in basins}
    bounds = []
    for b in basins:
        with rasterio.open(paths[b] / SINGLE_LAYERS["dtw"]) as ds:
            if ds.transform.a != CELL or ds.transform.e != -CELL:
                raise ValueError(f"{b}: not a 100 m north-up grid")
            if ds.crs.to_epsg() != 5070:
                raise ValueError(f"{b}: CRS is {ds.crs}, expected EPSG:5070")
            bounds.append(ds.bounds)
    left = min(v.left for v in bounds)
    right = max(v.right for v in bounds)
    bottom = min(v.bottom for v in bounds)
    top = max(v.top for v in bounds)
    if (left % CELL) or (top % CELL):
        raise ValueError("mosaic origin is not on the 100 m grid")
    nrow = int(round((top - bottom) / CELL))
    ncol = int(round((right - left) / CELL))
    transform = from_origin(left, top, CELL, CELL)
    log(f"mosaic grid {nrow} x {ncol} cells, origin ({left:.0f}, {top:.0f})")

    hucs = gpd.read_parquet(HUC8_POLYS, columns=["huc8", "geometry"])
    hucs["huc8"] = hucs["huc8"].astype(str).str.zfill(8)
    hucs = hucs[hucs["huc8"].isin(basins)].set_index("huc8")
    if len(hucs) != len(basins):
        raise ValueError(f"{len(hucs)} HUC8 polygons for {len(basins)} basins")
    if hucs.crs.to_epsg() != 5070:
        raise ValueError("HUC8 polygons are not EPSG:5070")

    names = list(SINGLE_LAYERS) + list(GATE_BANDS)
    out = {k: np.full((nrow, ncol), np.nan, np.float32) for k in names}
    basin_idx = np.full((nrow, ncol), -1, np.int16)
    n_preclip = 0
    n_dropped = 0
    n_overlap = 0

    for k, b in enumerate(basins):
        with rasterio.open(paths[b] / SINGLE_LAYERS["dtw"]) as ds:
            bb, btr = ds.bounds, ds.transform
            h, w = ds.height, ds.width
            dtw = ds.read(1).astype(np.float32)
        r0 = int(round((top - bb.top) / CELL))
        c0 = int(round((bb.left - left) / CELL))
        dtw[dtw == NODATA] = np.nan
        pre = np.isfinite(dtw)
        n_preclip += int(pre.sum())

        inside = (
            rasterize(
                [(hucs.loc[b, "geometry"], 1)],
                out_shape=(h, w),
                transform=btr,
                fill=0,
                dtype="uint8",
            )
            == 1
        )
        keep = pre & inside
        n_dropped += int((pre & ~inside).sum())
        sl = (slice(r0, r0 + h), slice(c0, c0 + w))
        n_overlap += int((keep & (basin_idx[sl] >= 0)).sum())
        basin_idx[sl] = np.where(keep, k, basin_idx[sl])

        for name, fn in SINGLE_LAYERS.items():
            if name == "dtw":
                arr = dtw
            else:
                with rasterio.open(paths[b] / fn) as ds:
                    arr = ds.read(1).astype(np.float32)
                arr[arr == NODATA] = np.nan
            out[name][sl] = np.where(keep, arr, out[name][sl])
        with rasterio.open(paths[b] / GATE_FILE) as ds:
            if tuple(ds.descriptions) != GATE_BANDS:
                raise ValueError(f"{b}: gate bands are {ds.descriptions}")
            gate = ds.read().astype(np.float32)
        gate[gate == NODATA] = np.nan
        for i, name in enumerate(GATE_BANDS):
            out[name][sl] = np.where(keep, gate[i], out[name][sl])
        if (k + 1) % 12 == 0:
            log(f"  mosaicked {k + 1}/{len(basins)} basins")

    if n_overlap:
        raise ValueError(f"{n_overlap} overlapping valid cells after HUC8 clipping")
    valid = basin_idx >= 0
    meta = {
        "nrow": nrow,
        "ncol": ncol,
        "left_m": left,
        "top_m": top,
        "n_preclip_valid_cells": n_preclip,
        "n_cells_dropped_by_huc8_clip": n_dropped,
        "n_valid_cells": int(valid.sum()),
        "valid_area_km2": float(valid.sum()) * CELL_KM2,
        "overlap_after_clip": n_overlap,
    }
    return {
        "arrays": out,
        "transform": transform,
        "basin_idx": basin_idx,
        "valid": valid,
        "meta": meta,
    }


def dem_on_grid(transform, nrow: int, ncol: int) -> np.ndarray:
    """The 100 m DEM (m) read on the mosaic grid; requires exact alignment."""
    with rasterio.open(DEM_100M) as ds:
        dt = ds.transform
        if (dt.a, dt.e) != (CELL, -CELL) or ds.crs.to_epsg() != 5070:
            raise ValueError(f"{DEM_100M}: not a 100 m EPSG:5070 north-up grid")
        col0 = (transform.c - dt.c) / CELL
        row0 = (dt.f - transform.f) / CELL
        if col0 != int(col0) or row0 != int(row0):
            raise ValueError("mosaic grid is not cell-aligned with the DEM")
        arr = ds.read(
            1, window=Window(int(col0), int(row0), ncol, nrow), boundless=True
        ).astype(np.float32)
        nd = ds.nodata
    if nd is not None:
        arr[arr == nd] = np.nan
    return arr


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def histogram(dtw: np.ndarray, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Item 1: 0.05 m histogram of rendered DTW from -1 to 6 m."""
    n_valid = int(dtw.size)
    edges = np.round(
        np.arange(HIST_LO, HIST_HI + 0.5 * HIST_W, HIST_W).astype("float64"), 10
    )
    counts, _ = np.histogram(dtw, bins=edges)
    tab = pd.DataFrame(
        {
            "bin_lo_m": edges[:-1],
            "bin_hi_m": edges[1:],
            "n_cells": counts,
            "frac_of_valid_footprint": counts / n_valid,
            "area_km2": counts * CELL_KM2,
        }
    )
    tab.to_csv(out_dir / "e0_dtw_histogram_0p05m.csv", index=False)

    top = tab.sort_values("n_cells", ascending=False).head(5)
    f_narrow = float(np.mean((dtw >= 2.95) & (dtw < 3.05)))
    f_wide = float(np.mean((dtw >= 3.00) & (dtw < 3.25)))
    exact = int(np.sum(dtw == np.float32(3.0)))
    near = int(np.sum(np.abs(dtw - 3.0) <= 1e-3))
    summary = {
        "n_valid_cells": n_valid,
        "valid_area_km2": n_valid * CELL_KM2,
        "n_cells_below_hist_lo": int(np.sum(dtw < HIST_LO)),
        "n_cells_above_hist_hi": int(np.sum(dtw >= HIST_HI)),
        "top5_bins": top.to_dict("records"),
        "frac_2p95_to_3p05_m": f_narrow,
        "area_2p95_to_3p05_km2": f_narrow * n_valid * CELL_KM2,
        "frac_3p00_to_3p25_m": f_wide,
        "area_3p00_to_3p25_km2": f_wide * n_valid * CELL_KM2,
        "n_cells_exactly_3p0_float32": exact,
        "n_cells_within_1e-3_m_of_3p0": near,
        "frac_exactly_3p0_float32": exact / n_valid,
        "frac_within_1e-3_m_of_3p0": near / n_valid,
        "median_dtw_m": float(np.median(dtw)),
    }
    return tab, summary


def hist_figure(tab: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mid = 0.5 * (tab["bin_lo_m"] + tab["bin_hi_m"])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(mid, 100 * tab["frac_of_valid_footprint"], width=HIST_W, color="#666666")
    ax.axvline(3.0, color="#D55E00", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(3.05, ax.get_ylim()[1] * 0.95, "mirror expert, 3.0 m", color="#D55E00")
    ax.set_xlim(HIST_LO, HIST_HI)
    ax.set_xlabel("Rendered depth to water (m)")
    ax.set_ylabel("Share of valid footprint per 0.05 m bin (%)")
    ax.set_title("handily r1e, 72-basin Nevada render")
    fig.tight_layout()
    path = out_dir / "e0_dtw_histogram_0p05m.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def contrast(arrays: dict, mode: np.ndarray, out_dir: Path) -> pd.DataFrame:
    """Item 2: per-quantity distribution at mode cells vs the rest."""
    rows = []
    for label, sel in (("mode 2.9-3.3 m", mode), ("other valid", ~mode)):
        for name, unit in (
            ("w_fac", "dimensionless"),
            ("w_deep", "dimensionless"),
            ("w_mirror", "dimensionless"),
            ("w_head", "dimensionless"),
            ("sigma", "m"),
            ("fold_spread", "m"),
            ("head_minus_wte", "m"),
            ("r_minus_wte", "m"),
            ("head_dtw", "m"),
            ("deep_dtw", "m"),
            ("r_dtw", "m"),
        ):
            rows.append({"group": label, **quant(arrays[name][sel], name, unit)})
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "e0_gate_distributions.csv", index=False)
    return tab


def mirror_shares(arrays: dict, mode: np.ndarray) -> dict:
    wm = arrays["w_mirror"]
    out = {}
    for label, sel in (("mode", mode), ("all_valid", np.ones(wm.size, bool))):
        for thr in (0.5, 0.9, 0.95, 0.99):
            out[f"frac_{label}_w_mirror_ge_{thr}"] = float(np.mean(wm[sel] >= thr))
    out["n_mode_cells"] = int(mode.sum())
    out["n_valid_cells"] = int(wm.size)
    out["frac_footprint_in_mode"] = float(mode.mean())
    return out


def implied_fac(arrays: dict, mode: np.ndarray, out_dir: Path) -> dict:
    """Item 3: invert the mixture for the unwritten FAC expert surface.

    ``fac_wte = (wte - w_deep*deep - w_mirror*mirror - w_head*head) / w_fac``
    (m), only where ``w_fac > 0.05`` so the division is conditioned. Reported as
    the implied FAC DTW, ``z_surf - fac_wte`` (m). The closure error measured on
    cells where the FAC expert carries essentially no weight prices the
    fold-median / fold-mean aggregation mismatch described in the module
    docstring; the inversion inherits that error divided by ``w_fac``.
    """
    z = arrays["z_surf"]
    mirror = z - 3.0
    mix_no_fac = (
        arrays["w_deep"] * arrays["deep_wte"]
        + arrays["w_mirror"] * mirror
        + arrays["w_head"] * arrays["head_wte"]
    )
    closure = arrays["wte"] - mix_no_fac  # = w_fac * fac_wte if the identity held
    low = arrays["w_fac"] <= 0.001
    rows = []
    for label, sel in (
        ("w_fac <= 0.001 (FAC effectively absent)", low),
        ("w_fac <= 0.001 and mode", low & mode),
    ):
        if sel.sum():
            rows.append({"group": label, **quant(closure[sel], "closure_error", "m")})

    ok = arrays["w_fac"] > 0.05
    fac_wte = np.where(ok, closure / np.where(ok, arrays["w_fac"], 1.0), np.nan)
    fac_dtw = z - fac_wte
    for label, sel in (
        ("mode 2.9-3.3 m, w_fac > 0.05", ok & mode),
        ("other valid, w_fac > 0.05", ok & ~mode),
    ):
        if sel.sum():
            rows.append({"group": label, **quant(fac_dtw[sel], "implied_fac_dtw", "m")})
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "e0_implied_fac_expert.csv", index=False)
    return {
        "table": tab.to_dict("records"),
        "n_w_fac_gt_0p05": int(ok.sum()),
        "frac_valid_w_fac_gt_0p05": float(ok.mean()),
        "n_w_fac_le_0p001": int(low.sum()),
        "frac_mode_w_fac_gt_0p05": float(np.mean(ok[mode])),
        "median_abs_closure_error_m": float(np.median(np.abs(closure[low])))
        if low.any()
        else float("nan"),
        "p99_abs_closure_error_m": float(np.percentile(np.abs(closure[low]), 99))
        if low.any()
        else float("nan"),
    }


def spatial(
    grid: dict,
    basins: list[str],
    dtw2d: np.ndarray,
    z2d: np.ndarray,
    out_dir: Path,
) -> dict:
    """Item 4: per-basin mode share, patchiness, and the slope contrast."""
    valid = grid["valid"]
    mode2d = valid & (dtw2d >= MODE_LO) & (dtw2d < MODE_HI)
    bidx = grid["basin_idx"]
    rows = []
    for k, b in enumerate(basins):
        m = bidx == k
        n = int(m.sum())
        rows.append(
            {
                "basin": b,
                "n_valid_cells": n,
                "area_km2": n * CELL_KM2,
                "n_mode_cells": int((mode2d & m).sum()),
                "frac_mode": float((mode2d & m).sum() / n) if n else np.nan,
                "median_dtw_m": float(np.median(dtw2d[m])) if n else np.nan,
            }
        )
    tab = pd.DataFrame(rows).sort_values("frac_mode", ascending=False)
    tab.to_csv(out_dir / "e0_per_basin_mode_share.csv", index=False)

    # Patchiness: a mode cell whose rook (100 m) neighbour is also a mode cell.
    pad = np.zeros((mode2d.shape[0] + 2, mode2d.shape[1] + 2), bool)
    pad[1:-1, 1:-1] = mode2d
    rook = (
        pad[:-2, 1:-1].astype(np.uint8) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    )
    queen = rook + pad[:-2, :-2] + pad[:-2, 2:] + pad[2:, :-2] + pad[2:, 2:]
    n_mode = int(mode2d.sum())
    patch = {
        "n_mode_cells": n_mode,
        "frac_mode_with_rook_mode_neighbour": float((rook[mode2d] > 0).mean()),
        "frac_mode_with_queen_mode_neighbour": float((queen[mode2d] > 0).mean()),
        "frac_mode_fully_surrounded_rook": float((rook[mode2d] == 4).mean()),
    }

    # Slope of the 100 m land surface, degrees. np.gradient on the DEM read on
    # the mosaic grid: valid cells at the footprint edge have finite neighbours
    # because the DEM is CONUS-wide, so no NaN handling is needed inside the
    # footprint; cells whose DEM neighbour is DEM nodata are excluded explicitly.
    gy, gx = np.gradient(z2d.astype(np.float64), CELL, CELL)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    fin = np.isfinite(slope_deg)
    n_no_slope_in_valid = int((valid & ~fin).sum())
    slope_tab = []
    for label, sel in (
        ("mode 2.9-3.3 m", mode2d & fin),
        ("all valid", valid & fin),
        ("other valid", valid & ~mode2d & fin),
    ):
        v = slope_deg[sel]
        slope_tab.append(
            {
                "group": label,
                "n_cells": int(v.size),
                "frac_slope_lt_1_deg": float(np.mean(v < 1.0)),
                "frac_slope_lt_0p5_deg": float(np.mean(v < 0.5)),
                "median_slope_deg": float(np.median(v)),
                "p90_slope_deg": float(np.percentile(v, 90)),
            }
        )
    pd.DataFrame(slope_tab).to_csv(out_dir / "e0_slope_contrast.csv", index=False)

    return {
        "per_basin_top10": tab.head(10).to_dict("records"),
        "per_basin_bottom5": tab.tail(5).to_dict("records"),
        "patchiness": patch,
        "slope": slope_tab,
        "n_valid_cells_without_slope": n_no_slope_in_valid,
    }


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    man = json.loads(MANIFEST.read_text())
    # The manifest names the experts; the renderer writes the bands as "w_<name>".
    if tuple(f"w_{e}" for e in man["gate_experts"]) != GATE_BANDS:
        raise ValueError(f"gate experts are {man['gate_experts']}")
    mirror_depth = float(man["flags"]["mirror_depth_m"])
    if mirror_depth != 3.0:
        raise ValueError(f"mirror depth is {mirror_depth} m, this script assumes 3.0 m")
    log(f"model gate experts {GATE_BANDS}, mirror depth {mirror_depth:.1f} m")

    basins = rendered_basins()
    log(f"rendered basins: {len(basins)}")
    grid = build_mosaic(basins)
    log(f"  clip bookkeeping: {grid['meta']}")

    a = grid["arrays"]
    valid = grid["valid"]
    tr = grid["transform"]
    z2d = dem_on_grid(tr, grid["meta"]["nrow"], grid["meta"]["ncol"])

    # The renderer writes dtw = z_surf - wte, so DTW + WTE must reproduce the DEM.
    zs = a["dtw"] + a["wte"]
    d = np.abs(zs[valid] - z2d[valid])
    n_dem_nodata = int((~np.isfinite(z2d[valid])).sum())
    dem_check = {
        "n_valid_cells": int(valid.sum()),
        "n_valid_cells_dem_nodata": n_dem_nodata,
        "median_abs_diff_m": float(np.nanmedian(d)),
        "p99_abs_diff_m": float(np.nanpercentile(d, 99)),
        "max_abs_diff_m": float(np.nanmax(d)),
        "frac_gt_0p5_m": float(np.nanmean(d > 0.5)),
    }
    log(f"  z_surf identity (DTW + WTE vs {DEM_100M.name}): {dem_check}")

    gate_sum = a["w_fac"] + a["w_deep"] + a["w_mirror"] + a["w_head"]
    gate_check = {
        "median_abs_sum_minus_1": float(np.median(np.abs(gate_sum[valid] - 1.0))),
        "max_abs_sum_minus_1": float(np.max(np.abs(gate_sum[valid] - 1.0))),
    }
    log(f"  gate weight sum: {gate_check}")

    # Flatten to the valid footprint; z_surf taken from the render's own
    # reconstruction so items 2-3 are internally consistent even if the DEM
    # differs (the check above reports whether it does).
    flat = {k: v[valid] for k, v in a.items()}
    flat["z_surf"] = zs[valid]
    flat["head_minus_wte"] = flat["head_wte"] - flat["wte"]
    flat["r_minus_wte"] = flat["r_wte"] - flat["wte"]
    flat["head_dtw"] = flat["z_surf"] - flat["head_wte"]
    flat["deep_dtw"] = flat["z_surf"] - flat["deep_wte"]
    flat["r_dtw"] = flat["z_surf"] - flat["r_wte"]
    dtw = flat["dtw"]
    mode = (dtw >= MODE_LO) & (dtw < MODE_HI)
    log(f"  mode cells [{MODE_LO}, {MODE_HI}) m: {int(mode.sum())}")

    tab, hist_summary = histogram(dtw, out_dir)
    png = hist_figure(tab, out_dir)
    log(f"  wrote {png}")
    gate_tab = contrast(flat, mode, out_dir)
    shares = mirror_shares(flat, mode)
    log(f"  mirror shares: {shares}")
    fac = implied_fac(flat, mode, out_dir)
    sp = spatial(grid, basins, a["dtw"], z2d, out_dir)

    summary = {
        "render_arm": RENDER_ARM,
        "n_basins": len(basins),
        "mirror_depth_m": mirror_depth,
        "mosaic": grid["meta"],
        "dem_identity_check": dem_check,
        "gate_weight_sum_check": gate_check,
        "item1_histogram": hist_summary,
        "item2_distributions": gate_tab.to_dict("records"),
        "item2_mirror_shares": shares,
        "item3_implied_fac": fac,
        "item4_spatial": sp,
        "notes": [
            "Footprint: each basin clipped to its WBD HUC8 polygon; zero "
            "overlapping valid cells after the clip, so no cell is double counted.",
            "Rendered WTE/head WTE/sigma/fold spread are fold medians while the "
            "gate weights are a renormalised fold mean, so the per-fold mixture "
            "identity does not close exactly on these rasters; the closure error "
            "is reported in item3_implied_fac.",
            "The FAC expert WTE surface is not written by the renderer; item 3 "
            "recovers it only by inverting the mixture.",
        ],
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    log(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
