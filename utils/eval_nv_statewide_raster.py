"""Statewide 100 m raster evaluation of one or more handily render arms over the
72-basin Nevada HUC8 footprint (``notes/nv_validation_plan.md`` section 3,
protocol and metric definitions from ``notes/NV_STATEWIDE_RASTER_EVAL.md``).

Arm-parameterised: ``--arms name=<render dir name>`` names the render
directories under ``<huc8-root>/<basin>/gnn/``; ``--primary-arm`` is the arm
under evaluation and ``--reference-arm`` the previously shipped map it is
compared against. Ma is re-sampled at the same coordinates with the repo's own
``sample_benchmark_rasters.sample_ma_tiles``.

What it writes to ``--out-dir`` (every intermediate is kept):

* ``sites_sampled.parquet`` -- every layer of every arm plus Ma at all target
  sites, with the basin that produced the cell and the stratification columns.
* ``verify_per_basin.csv`` -- structural, provenance, negative-DTW, leak-gate
  and raster-sanity checks, one row per basin.
* ``verify_provenance.csv`` / ``verify_crossmachine.csv`` -- provenance value
  uniformity and the cross-machine pilot diff.
* ``panel_core.csv`` / ``panel_depth.csv`` / ``panel_dist.csv`` /
  ``panel_nwis.csv`` / ``panel_shallow.csv`` -- section 3.1 panels for every
  scored set.
* ``delta_bootstrap.csv`` -- paired block-bootstrap deltas (Ma - primary and
  reference - primary) on MAD and RMSE, overall and by band, under both
  blockings.
* ``rendering_loss.csv``, ``sigma_calibration.csv``, ``sigma_deciles.csv``,
  ``negatives.csv``, ``per_basin_mad.csv``.
* ``calling_curve.csv`` / ``calling_summary.csv`` / ``calling_cells.csv`` --
  the shallow calling layers on the rendered rasters and the fraction of map
  cells each cut calls statewide.
* ``report.md`` -- the same tables rendered as markdown for the note.

Definitions (repeated here because every number carries units):

* residual ``r = pred - obs`` (m), positive = predicted too deep.
* MAD = ``median(|r|)`` (m); median resid = ``median(r)`` (m); bias =
  ``mean(r)`` (m); RMSE = ``sqrt(mean(r^2))`` (m); p95 = ``percentile(|r|, 95)``
  (m).
* delta = ``stat(other) - stat(primary)`` (m), positive = the primary arm is
  better. CI95 from a paired block bootstrap (2000 resamples, blocks drawn with
  replacement, the same resampled site set used for both predictors); the CI
  bounds the difference in that statistic between the two predictors on this
  site set, nothing else.
* precision / recall / F1 / PR-AUC / coverage / Brier skill are dimensionless.
  Brier skill = ``1 - Brier / Brier_climatology``, positive = better than the
  stratum base rate.

Run::

    uv run --directory /home/dgketchum/code/handily python \
        /home/dgketchum/code/handily/utils/eval_nv_statewide_raster.py \
        --arms r1e=gnn_conus_monitoring_water_src_r1e \
               m20_ord=gnn_conus_monitoring_water_src_r1e_m20_ord \
        --primary-arm m20_ord --reference-arm r1e \
        --point-preds m20_ord=/data/ssd2/handily/nv/regional/wells/ndwr_admit_m20_ord_ramp_pt_preds.parquet \
        --point-preds r1e=/data/ssd2/handily/nv/regional/wells/ndwr_admit_ramp_pt_preds.parquet \
        --out-dir /data/ssd2/handily/nv/regional/statewide_eval_m20_ord
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "utils"))

from compare_nv_ma_handily_shape import (  # noqa: E402
    DEPTH_BANDS,
    DEPTH_LABELS,
    SHALLOW_THRESHOLDS,
    panel_stats,
)
from sample_benchmark_rasters import sample_ma_tiles  # noqa: E402
from score_nv_shallow_panel import (  # noqa: E402
    BlockBoot,
    dist_to_irrigation_m,
    f1,
    laplace_cdf,
    log,
    pr_at,
)

HUC8_ROOT = "/data/ssd2/handily/huc8"
TARGETS = "/data/ssd2/handily/nv/regional/wells/ndwr_admission_targets.parquet"
SOURCES = "/data/ssd2/handily/nv/regional/wells/ndwr_admitted_sources_frozen.parquet"
MA_DIR = "/nas/gwx/wtd_states"

#: Layers every arm carries (single band).
BASE_LAYERS = {
    "dtw": "gnn_dtw_100m.tif",
    "sigma": "gnn_sigma_100m.tif",
    "fold_spread": "gnn_fold_spread_100m.tif",
}
ORDINAL_TIF = "gnn_p_dtw_lt_100m.tif"
EXPECTED_TIFS_BASE = (
    "gnn_dtw_100m.tif",
    "gnn_wte_100m.tif",
    "gnn_sigma_100m.tif",
    "gnn_gate_w_100m.tif",
    "gnn_head_wte_100m.tif",
    "gnn_deep_wte_100m.tif",
    "gnn_r_wte_100m.tif",
    "gnn_fold_spread_100m.tif",
)

#: Distance to nearest admitted source (km) and to nearest NWIS well (km).
SRC_EDGES = (0.0, 0.3, 1.0, 5.0, 20.0, np.inf)
SRC_LABELS = ("0-0.3 km", "0.3-1 km", "1-5 km", "5-20 km", "20+ km")
NWIS_EDGES = (0.0, 1.0, 5.0, 20.0, np.inf)
NWIS_LABELS = ("0-1 km", "1-5 km", "5-20 km", "20+ km")

#: Calling layer.
CALL_THRESHOLDS = (2.0, 3.0, 5.0)
CUTS = np.round(np.arange(0.05, 0.96, 0.05), 2)
GATE_U_PRECISION, GATE_U_RECALL = 0.32, 0.55
GATE_G_PRECISION, GATE_G_RECALL = 0.71, 0.50
#: Mirror plateau of the m20 family, on the rendered DTW surface (m).
PLATEAU = (2.0, 3.0)

PROVENANCE_FIELDS = (
    "model",
    "model_dir",
    "r_source",
    "r_exclude_km",
    "deep_exclude_km",
    "query_writeback",
    "extra_sources_path",
    "wells",
    "bundle",
    "surface",
)


# ---------------------------------------------------------------------------
# small metric helpers
# ---------------------------------------------------------------------------
def mad(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.median(np.abs(pred - obs)))


def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the precision-recall curve by the step rule (sklearn's AP)."""
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, ys.size + 1)
    return float((prec * ys).sum() / ys.sum())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def brier_skill(y: np.ndarray, p: np.ndarray) -> float:
    b0 = float(np.mean((y.mean() - y) ** 2))
    return 1.0 - brier(y, p) / b0 if b0 > 0 else np.nan


def band_masks(v: np.ndarray, edges, labels) -> dict[str, np.ndarray]:
    return {
        lab: (v >= lo) & (v < hi) for lab, lo, hi in zip(labels, edges[:-1], edges[1:])
    }


def md_table(df: pd.DataFrame) -> str:
    """Markdown table from a frame, values already formatted as strings."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# raster pass: verification, site sampling, statewide called-cell counts
# ---------------------------------------------------------------------------
def basin_dirs(root: Path, arm_dir: str) -> dict[str, Path]:
    found = {}
    for d in sorted(root.glob(f"*/gnn/{arm_dir}")):
        found[d.parents[1].name] = d
    return found


def read_layers(d: Path, with_ordinal: bool) -> tuple[dict, dict]:
    """Read an arm's layers for one basin; returns (arrays, grid profile)."""
    arrays, prof = {}, {}
    for key, name in BASE_LAYERS.items():
        with rasterio.open(d / name) as ds:
            a = ds.read(1).astype("float64")
            nd = ds.nodata
            if nd is not None:
                a[a == nd] = np.nan
            arrays[key] = a
            if not prof:
                prof = {
                    "crs": str(ds.crs),
                    "transform": tuple(ds.transform)[:6],
                    "shape": (ds.height, ds.width),
                    "nodata": nd,
                    "dtype": ds.dtypes[0],
                }
    if with_ordinal:
        with rasterio.open(d / ORDINAL_TIF) as ds:
            nd = ds.nodata
            for i, desc in enumerate(ds.descriptions, start=1):
                a = ds.read(i).astype("float64")
                if nd is not None:
                    a[a == nd] = np.nan
                arrays[f"ord_{desc}"] = a
            prof["ordinal_bands"] = list(ds.descriptions)
    return arrays, prof


def call_probability(arrays: dict, variable: str, t: float) -> np.ndarray | None:
    """P(DTW < t) surface for one calling variable, or None if unavailable."""
    if variable == "laplace":
        return laplace_cdf(t - arrays["dtw"], arrays["sigma"])
    key = f"ord_p_dtw_lt_{t:g}m"
    return arrays.get(key)


def scan_basins(
    root: Path,
    arms: dict[str, str],
    primary: str,
    reference: str,
    sites: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One pass over all basins: verification rows, sampled site values, and
    statewide counts of map cells called at each probability cut."""
    dirs = {a: basin_dirs(root, d) for a, d in arms.items()}
    basins = sorted(dirs[primary])
    log(
        f"basins with {primary}: {len(basins)}; with {reference}: {len(dirs[reference])}"
    )

    x = sites["x5070"].to_numpy("float64")
    y = sites["y5070"].to_numpy("float64")
    n = len(sites)
    hits = np.zeros(n, dtype="int32")
    basin_of = np.full(n, "", dtype=object)
    samp = {
        f"{a}__{k}": np.full(n, np.nan)
        for a in arms
        for k in ("dtw", "sigma", "fold_spread")
    }
    for t in (2.0, 5.0, 10.0):
        samp[f"{primary}__p_dtw_lt_{t:g}m"] = np.full(n, np.nan)

    verify_rows, cell_rows = [], []
    for basin in basins:
        arrs, profs = {}, {}
        for a in arms:
            d = dirs[a].get(basin)
            if d is None:
                raise SystemExit(f"{basin}: arm {a} missing")
            arrs[a], profs[a] = read_layers(d, with_ordinal=(a == primary))
        pdir = dirs[primary][basin]
        man = json.loads((pdir / "infer_run.json").read_text())

        tif_names = {p.name for p in pdir.glob("*.tif")}
        expect = set(EXPECTED_TIFS_BASE) | {ORDINAL_TIF}
        with rasterio.open(pdir / "gnn_gate_w_100m.tif") as ds:
            gate_bands = list(ds.descriptions)
        grid_same = all(
            profs[primary][k] == profs[reference][k]
            for k in ("crs", "transform", "shape", "nodata")
        )

        dtw = arrs[primary]["dtw"]
        sig = arrs[primary]["sigma"]
        valid = np.isfinite(dtw)
        v = dtw[valid]
        ref_dtw = arrs[reference]["dtw"]
        ref_valid = np.isfinite(ref_dtw)
        row = {
            "basin": basin,
            "n_tifs": len(tif_names),
            "tifs_complete": tif_names == expect,
            "manifest": (pdir / "infer_run.json").exists(),
            "gate_bands": "|".join(gate_bands),
            "ordinal_bands": "|".join(profs[primary].get("ordinal_bands", [])),
            "grid_matches_reference": grid_same,
            "crs": profs[primary]["crs"],
            "nodata": profs[primary]["nodata"],
            "n_cells_window": int(dtw.size),
            "valid_cells": int(valid.sum()),
            "valid_frac": float(valid.mean()),
            "dtw_p10_m": float(np.percentile(v, 10)),
            "dtw_p50_m": float(np.percentile(v, 50)),
            "dtw_p90_m": float(np.percentile(v, 90)),
            "dtw_min_m": float(v.min()),
            "dtw_max_m": float(v.max()),
            "sigma_p50_m": float(np.nanmedian(sig[valid])),
            "frac_lt_2m": float((v < 2).mean()),
            "frac_lt_3m": float((v < 3).mean()),
            "frac_lt_5m": float((v < 5).mean()),
            "frac_ge_30m": float((v >= 30).mean()),
            "neg_cells": int((v < 0).sum()),
            "pct_neg_dtw": float((v < 0).mean() * 100.0),
            "manifest_pct_neg_dtw": float(man.get("pct_negative_dtw", np.nan)) * 100.0,
            "ref_pct_neg_dtw": float((ref_dtw[ref_valid] < 0).mean() * 100.0),
            "ref_dtw_min_m": float(ref_dtw[ref_valid].min()),
            "leak_gate_status": (man.get("leak_gate") or {}).get("status", "-"),
            "leak_gate_n_wells": (man.get("leak_gate") or {}).get("n_wells", 0),
            "water_flatten_mode": (man.get("water_flatten") or {}).get("mode", "-"),
            "source_edges_k": (man.get("source_edges") or {}).get("k", None),
            "n_sources_extra": (man.get("source_edges") or {}).get(
                "n_sources_extra", None
            ),
        }
        for t in (2.0, 5.0, 10.0):
            key = f"ord_p_dtw_lt_{t:g}m"
            row[f"ord_p_lt_{t:g}m_median"] = (
                float(np.nanmedian(arrs[primary][key][valid]))
                if key in arrs[primary]
                else np.nan
            )
        for fld in PROVENANCE_FIELDS:
            row[f"prov_{fld}"] = man.get(fld)
        verify_rows.append(row)

        # statewide called-cell counts (basin valid footprints are disjoint)
        plateau = valid & (dtw >= PLATEAU[0]) & (dtw < PLATEAU[1])
        for a in arms:
            for var in ("laplace", "ordinal"):
                if var == "ordinal" and a != primary:
                    continue
                for t in CALL_THRESHOLDS:
                    p = call_probability(arrs[a], var, t)
                    if p is None:
                        continue
                    own = np.isfinite(arrs[a]["dtw"])
                    own_plateau = (
                        own
                        & (arrs[a]["dtw"] >= PLATEAU[0])
                        & (arrs[a]["dtw"] < PLATEAU[1])
                    )
                    for c in CUTS:
                        called = valid & np.isfinite(p) & (p >= c)
                        cell_rows.append(
                            {
                                "basin": basin,
                                "arm": a,
                                "variable": var,
                                "threshold_m": t,
                                "cut": float(c),
                                "valid_cells": int(valid.sum()),
                                "called_cells": int(called.sum()),
                                "called_on_primary_plateau": int(
                                    (called & plateau).sum()
                                ),
                                "called_on_own_plateau": int(
                                    (called & own_plateau).sum()
                                ),
                            }
                        )

        # site sampling on this basin's grid
        tr = profs[primary]["transform"]
        h, w = profs[primary]["shape"]
        col = np.floor((x - tr[2]) / tr[0]).astype("int64")
        r = np.floor((y - tr[5]) / tr[4]).astype("int64")
        inb = (col >= 0) & (col < w) & (r >= 0) & (r < h)
        if inb.any():
            idx = np.where(inb)[0]
            ok = np.isfinite(dtw[r[idx], col[idx]])
            idx = idx[ok]
            hits[idx] += 1
            basin_of[idx] = basin
            for a in arms:
                for k in ("dtw", "sigma", "fold_spread"):
                    samp[f"{a}__{k}"][idx] = arrs[a][k][r[idx], col[idx]]
            for t in (2.0, 5.0, 10.0):
                key = f"ord_p_dtw_lt_{t:g}m"
                if key in arrs[primary]:
                    samp[f"{primary}__p_dtw_lt_{t:g}m"][idx] = arrs[primary][key][
                        r[idx], col[idx]
                    ]
        log(
            f"  {basin}: valid {int(valid.sum())} cells, sites so far {int((hits > 0).sum())}"
        )

    dist = pd.Series(hits).value_counts().sort_index()
    log(f"n_basin_hits distribution: {dist.to_dict()}")
    if (hits != 1).any():
        raise SystemExit(
            f"{int((hits != 1).sum())} sites do not hit exactly one basin raster "
            f"(distribution {dist.to_dict()}) - the sampling join is ambiguous"
        )

    out = sites.copy()
    out["basin_hit"] = basin_of.astype(str)
    out["n_basin_hits"] = hits
    for k, vv in samp.items():
        out[k] = vv
    return pd.DataFrame(verify_rows), out, pd.DataFrame(cell_rows)


def cross_machine_check(a_dir: Path, b_dir: Path) -> pd.DataFrame:
    """Cell-level |difference| between two renders of the same basin."""
    rows = []
    layers = [(k, v, 1) for k, v in BASE_LAYERS.items()]
    with rasterio.open(a_dir / ORDINAL_TIF) as ds:
        ord_bands = list(ds.descriptions)
    layers += [(b, ORDINAL_TIF, i) for i, b in enumerate(ord_bands, start=1)]
    for name, tif, band in layers:
        with rasterio.open(a_dir / tif) as da, rasterio.open(b_dir / tif) as db:
            aa = da.read(band).astype("float64")
            bb = db.read(band).astype("float64")
            nda, ndb = da.nodata, db.nodata
        if nda is not None:
            aa[aa == nda] = np.nan
        if ndb is not None:
            bb[bb == ndb] = np.nan
        m = np.isfinite(aa) & np.isfinite(bb)
        d = np.abs(aa[m] - bb[m])
        rows.append(
            {
                "layer": name,
                "n_common_cells": int(m.sum()),
                "median_abs_diff": float(np.median(d)),
                "p90_abs_diff": float(np.percentile(d, 90)),
                "p99_abs_diff": float(np.percentile(d, 99)),
                "max_abs_diff": float(d.max()),
                "frac_gt_0.01": float((d > 0.01).mean()),
                "frac_gt_1": float((d > 1.0).mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def scored_sets(s: pd.DataFrame) -> dict[str, np.ndarray]:
    elig = s["is_admission_eligible"].to_numpy(bool)
    pump = s["is_pumping"].to_numpy(bool)
    pre80 = (~pump) & (s["por_end_yr"].to_numpy("float64") < 1980)
    return {
        "A_primary_eligible": elig,
        "B_secondary_pumping": pump,
        "C_secondary_pre1980": pre80,
        "D_context_all_targets": np.ones(len(s), bool),
    }


def core_rows(obs, preds: dict[str, np.ndarray], mask, set_name, stratum, stratum_kind):
    rows = []
    for name, p in preds.items():
        m = mask & np.isfinite(p) & np.isfinite(obs)
        st = panel_stats(obs[m], p[m])
        st.update(
            {
                "set": set_name,
                "stratum_kind": stratum_kind,
                "stratum": stratum,
                "predictor": name,
            }
        )
        rows.append(st)
    return rows


def delta_table(
    boot_map: dict[str, tuple[BlockBoot, np.ndarray]],
    obs,
    primary_pred,
    others: dict[str, np.ndarray],
    strata: dict[str, np.ndarray],
    set_name: str,
    stratum_kind: str,
) -> list[dict]:
    rows = []
    for other_name, other in others.items():
        for lab, m in strata.items():
            if m.sum() < 30:
                rows.append(
                    {
                        "set": set_name,
                        "comparison": f"{other_name} - primary",
                        "stratum_kind": stratum_kind,
                        "stratum": lab,
                        "n": int(m.sum()),
                        "note": "n<30",
                    }
                )
                continue
            row = {
                "set": set_name,
                "comparison": f"{other_name} - primary",
                "stratum_kind": stratum_kind,
                "stratum": lab,
                "n": int(m.sum()),
                "mad_delta_m": mad(obs[m], other[m]) - mad(obs[m], primary_pred[m]),
                "rmse_delta_m": rmse(obs[m], other[m]) - rmse(obs[m], primary_pred[m]),
            }
            for bname, (boot, blocks) in boot_map.items():
                lo, hi = boot.ci(
                    lambda i: mad(obs[i], other[i]) - mad(obs[i], primary_pred[i]), m
                )
                row[f"mad_lo_{bname}"], row[f"mad_hi_{bname}"] = lo, hi
                lo, hi = boot.ci(
                    lambda i: rmse(obs[i], other[i]) - rmse(obs[i], primary_pred[i]), m
                )
                row[f"rmse_lo_{bname}"], row[f"rmse_hi_{bname}"] = lo, hi
                row[f"n_blocks_{bname}"] = int(len(np.unique(blocks[m])))
            rows.append(row)
    return rows


def shallow_rows(obs, preds, mask, set_name):
    rows = []
    for thr in SHALLOW_THRESHOLDS:
        for name, p in preds.items():
            m = mask & np.isfinite(p) & np.isfinite(obs)
            prec, rec = pr_at(obs[m], p[m] < thr, thr)
            rows.append(
                {
                    "set": set_name,
                    "threshold_m": thr,
                    "predictor": name,
                    "n": int(m.sum()),
                    "n_obs_shallow": int((obs[m] < thr).sum()),
                    "n_pred_shallow": int((p[m] < thr).sum()),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1(prec, rec),
                }
            )
    return rows


def sigma_panel(obs, pred, sigma, set_name, arm) -> tuple[dict, list[dict]]:
    r = np.abs(pred - obs)
    ok = np.isfinite(r) & np.isfinite(sigma) & (sigma > 0)
    r, s = r[ok], sigma[ok]
    ratio = r / s
    summary = {
        "set": set_name,
        "arm": arm,
        "n": int(ok.sum()),
        "cov_1sigma": float((ratio <= 1).mean()),
        "cov_2sigma": float((ratio <= 2).mean()),
        "gaussian_target_1s": 0.6827,
        "gaussian_target_2s": 0.9545,
        "laplace_target_1b": 0.6321,
        "laplace_target_2b": 0.8647,
        "k_for_68.27pct": float(np.percentile(ratio, 68.27)),
        "k_for_95.45pct": float(np.percentile(ratio, 95.45)),
        "spearman_sigma_absresid": float(spearmanr(s, r).statistic),
        "median_sigma_m": float(np.median(s)),
        "median_abs_resid_m": float(np.median(r)),
    }
    q = np.quantile(s, np.linspace(0, 1, 11))
    dec = []
    for i in range(10):
        lo, hi = q[i], q[i + 1]
        m = (s >= lo) & (s <= hi) if i == 9 else (s >= lo) & (s < hi)
        if m.sum() == 0:
            continue
        dec.append(
            {
                "set": set_name,
                "arm": arm,
                "decile": f"D{i + 1}",
                "sigma_lo_m": float(lo),
                "sigma_hi_m": float(hi),
                "n": int(m.sum()),
                "median_sigma_m": float(np.median(s[m])),
                "median_abs_resid_m": float(np.median(r[m])),
                "mad_over_sigma": float(np.median(r[m]) / np.median(s[m])),
                "cov_1sigma": float((ratio[m] <= 1).mean()),
                "cov_2sigma": float((ratio[m] <= 2).mean()),
            }
        )
    return summary, dec


# ---------------------------------------------------------------------------
# calling layers
# ---------------------------------------------------------------------------
def calling_layers(
    s: pd.DataFrame,
    primary: str,
    reference: str,
    mask_primary: np.ndarray,
    boot: BlockBoot,
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = s["obs_dtw_m"].to_numpy("float64")
    irr = s["dist_irr_m"].to_numpy("float64") <= 500.0
    strata = {
        "primary statewide": mask_primary,
        "primary irr500 obs<10 m": mask_primary & irr & (obs < 10.0),
    }
    probs: dict[tuple[str, str, float], np.ndarray] = {}
    for arm in (primary, reference):
        pred = s[f"{arm}__dtw"].to_numpy("float64")
        sig = s[f"{arm}__sigma"].to_numpy("float64")
        for t in CALL_THRESHOLDS:
            probs[(arm, "laplace", t)] = laplace_cdf(t - pred, sig)
    for t in (2.0, 5.0):
        col = f"{primary}__p_dtw_lt_{t:g}m"
        if col in s.columns:
            probs[(primary, "ordinal", t)] = s[col].to_numpy("float64")

    cell_tot = (
        cells.groupby(["arm", "variable", "threshold_m", "cut"])[
            [
                "valid_cells",
                "called_cells",
                "called_on_primary_plateau",
                "called_on_own_plateau",
            ]
        ]
        .sum()
        .reset_index()
    )
    cell_tot["frac_cells_called"] = cell_tot["called_cells"] / cell_tot["valid_cells"]
    cell_tot["frac_called_on_primary_plateau"] = np.where(
        cell_tot["called_cells"] > 0,
        cell_tot["called_on_primary_plateau"] / cell_tot["called_cells"],
        np.nan,
    )
    cell_tot["frac_called_on_own_plateau"] = np.where(
        cell_tot["called_cells"] > 0,
        cell_tot["called_on_own_plateau"] / cell_tot["called_cells"],
        np.nan,
    )
    cl = cell_tot.set_index(["arm", "variable", "threshold_m", "cut"])

    curve, summary = [], []
    for (arm, var, t), p in probs.items():
        y = (obs < t).astype("float64")
        for lab, m in strata.items():
            rows = []
            for c in CUTS:
                called = p >= c
                prec, rec = pr_at(obs[m], called[m], t)
                plo, phi = boot.ci(lambda i: pr_at(obs[i], p[i] >= c, t)[0], m)
                rlo, rhi = boot.ci(lambda i: pr_at(obs[i], p[i] >= c, t)[1], m)
                key = (arm, var, t, float(c))
                rows.append(
                    {
                        "arm": arm,
                        "variable": var,
                        "threshold_m": t,
                        "stratum": lab,
                        "cut": float(c),
                        "n_sites": int(m.sum()),
                        "n_obs_shallow": int(y[m].sum()),
                        "n_called": int(called[m].sum()),
                        "precision": prec,
                        "precision_lo": plo,
                        "precision_hi": phi,
                        "recall": rec,
                        "recall_lo": rlo,
                        "recall_hi": rhi,
                        "f1": f1(prec, rec),
                        "frac_map_cells_called": float(cl.loc[key, "frac_cells_called"])
                        if key in cl.index
                        else np.nan,
                        "frac_called_on_plateau": float(
                            cl.loc[key, "frac_called_on_own_plateau"]
                        )
                        if key in cl.index
                        else np.nan,
                    }
                )
            curve += rows
            df = pd.DataFrame(rows)
            ok_r = df[df["recall"] >= GATE_U_RECALL]
            best = ok_r.loc[ok_r["precision"].idxmax()] if len(ok_r) else None
            ok_g = df[
                (df["recall"] >= GATE_G_RECALL) & (df["precision"] >= GATE_G_PRECISION)
            ]
            ap_lo, ap_hi = boot.ci(lambda i: average_precision(y[i], p[i]), m)
            summary.append(
                {
                    "arm": arm,
                    "variable": var,
                    "threshold_m": t,
                    "stratum": lab,
                    "n_sites": int(m.sum()),
                    "n_obs_shallow": int(y[m].sum()),
                    "base_rate": float(y[m].mean()),
                    "pr_auc": average_precision(y[m], p[m]),
                    "pr_auc_lo": ap_lo,
                    "pr_auc_hi": ap_hi,
                    "brier": brier(y[m], p[m]),
                    "brier_skill": brier_skill(y[m], p[m]),
                    "best_precision_at_recall_ge_0.55": None
                    if best is None
                    else float(best["precision"]),
                    "best_prec_lo": None
                    if best is None
                    else float(best["precision_lo"]),
                    "best_prec_hi": None
                    if best is None
                    else float(best["precision_hi"]),
                    "cut": None if best is None else float(best["cut"]),
                    "recall_at_cut": None if best is None else float(best["recall"]),
                    "frac_map_cells_called_at_cut": None
                    if best is None
                    else float(best["frac_map_cells_called"]),
                    "frac_called_on_plateau_at_cut": None
                    if best is None
                    else float(best["frac_called_on_plateau"]),
                    "gate_U_precision_ge_0.32_at_recall_ge_0.55": bool(
                        best is not None and best["precision"] >= GATE_U_PRECISION
                    ),
                    "gate_G_precision_ge_0.71_at_recall_ge_0.50": bool(len(ok_g) > 0),
                    "gate_G_best_cut": None
                    if not len(ok_g)
                    else float(ok_g.loc[ok_g["recall"].idxmax(), "cut"]),
                    "gate_G_recall": None
                    if not len(ok_g)
                    else float(ok_g["recall"].max()),
                }
            )
            log(f"  calling {arm} {var} <{t:g} m [{lab}] done")
    return pd.DataFrame(curve), pd.DataFrame(summary), cell_tot


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True, help="name=render_dir_name")
    ap.add_argument("--primary-arm", required=True)
    ap.add_argument("--reference-arm", required=True)
    ap.add_argument("--point-preds", nargs="*", default=[], help="name=parquet")
    ap.add_argument("--huc8-root", default=HUC8_ROOT)
    ap.add_argument("--targets", default=TARGETS)
    ap.add_argument("--sources", default=SOURCES)
    ap.add_argument("--ma-dir", default=MA_DIR)
    ap.add_argument("--pilot-basin", default="16040103")
    ap.add_argument(
        "--pilot-dir", default=None, help="second render of the pilot basin"
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arms = dict(a.split("=", 1) for a in args.arms)
    points = dict(p.split("=", 1) for p in args.point_preds)
    primary, reference = args.primary_arm, args.reference_arm
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sites = pd.read_parquet(args.targets)
    src = pd.read_parquet(args.sources)
    log(f"targets {len(sites)}, admitted sources {len(src)}")

    tree = cKDTree(src[["x5070", "y5070"]].to_numpy("float64"))
    d_src, _ = tree.query(sites[["x5070", "y5070"]].to_numpy("float64"), k=1)
    sites["dist_src_km"] = d_src / 1000.0
    n_coincident = int((d_src < 1.0).sum())
    log(
        f"disjointness: {n_coincident} target sites within 1 m of an admitted "
        f"source coordinate; {int((d_src <= 1.0).sum())} at exactly 1.000 m "
        f"(construction offset, distinct sites); "
        f"{int((d_src <= 100.0).sum())} within one 100 m cell of a source"
    )
    if n_coincident:
        raise SystemExit("scored targets share coordinates with the admitted pool")

    sites["dist_irr_m"] = dist_to_irrigation_m(sites)
    ma_res = sample_ma_tiles(
        sites["x5070"].to_numpy("float64"),
        sites["y5070"].to_numpy("float64"),
        Path(args.ma_dir),
    )
    sites["ma_resampled_m"] = ma_res
    stored = sites["ma_dtw_m"].to_numpy("float64")
    both = np.isfinite(ma_res) & np.isfinite(stored)
    log(
        f"Ma provenance: {int(both.sum())} sites with both; identical "
        f"{float((np.abs(ma_res[both] - stored[both]) < 1e-6).mean()):.4f}, "
        f">0.01 m {int((np.abs(ma_res[both] - stored[both]) > 0.01).sum())}, "
        f">1 m {int((np.abs(ma_res[both] - stored[both]) > 1.0).sum())}"
    )

    verify, s, cells = scan_basins(
        Path(args.huc8_root), arms, primary, reference, sites
    )
    verify.to_csv(out / "verify_per_basin.csv", index=False)
    cells.to_csv(out / "calling_cells_per_basin.csv", index=False)

    for name, path in points.items():
        p = pd.read_parquet(path)
        keep = ["x5070", "y5070", "pred_dtw_m", "sigma_m"] + [
            c for c in p.columns if c.startswith("p_dtw_lt_")
        ]
        p = p[keep].drop_duplicates(["x5070", "y5070"])
        p = p.rename(
            columns={c: f"{name}__pt_{c}" for c in keep if c not in ("x5070", "y5070")}
        )
        before = len(s)
        s = s.merge(p, on=["x5070", "y5070"], how="left")
        if len(s) != before:
            raise SystemExit(f"{name}: point-path coordinate join is not 1:1")
    s.to_parquet(out / "sites_sampled.parquet", index=False)
    log(f"wrote {out / 'sites_sampled.parquet'} ({len(s)} rows)")

    prov = pd.DataFrame(
        {
            "field": [f for f in PROVENANCE_FIELDS]
            + ["water_flatten_mode", "source_edges_k", "n_sources_extra"],
            "unique_values": [
                "; ".join(sorted({str(v) for v in verify[f"prov_{f}"]}))
                for f in PROVENANCE_FIELDS
            ]
            + [
                "; ".join(sorted({str(v) for v in verify[c]}))
                for c in ("water_flatten_mode", "source_edges_k", "n_sources_extra")
            ],
        }
    )
    prov.to_csv(out / "verify_provenance.csv", index=False)

    cm = pd.DataFrame()
    if args.pilot_dir:
        cm = cross_machine_check(
            Path(args.huc8_root) / args.pilot_basin / "gnn" / arms[primary],
            Path(args.pilot_dir),
        )
        cm.to_csv(out / "verify_crossmachine.csv", index=False)

    # ---------------- panels ----------------
    obs = s["obs_dtw_m"].to_numpy("float64")
    preds = {
        f"{primary} raster": s[f"{primary}__dtw"].to_numpy("float64"),
        f"{reference} raster": s[f"{reference}__dtw"].to_numpy("float64"),
        "Ma raster (resampled here)": s["ma_resampled_m"].to_numpy("float64"),
        "Ma (stored column)": s["ma_dtw_m"].to_numpy("float64"),
    }
    finite_core = np.isfinite(obs)
    for name in (
        f"{primary} raster",
        f"{reference} raster",
        "Ma raster (resampled here)",
    ):
        finite_core &= np.isfinite(preds[name])

    blocks_w25r = s["w25r_basin"].fillna("__null__").astype(str).to_numpy()
    blocks_huc8 = s["basin_hit"].astype(str).to_numpy()
    log(
        f"w25r_basin null on {int(s['w25r_basin'].isna().sum())} of {len(s)} targets "
        f"({100 * s['w25r_basin'].isna().mean():.2f} %)"
    )
    boot_w25r = BlockBoot(blocks_w25r, args.n_boot, args.seed)
    boot_huc8 = BlockBoot(blocks_huc8, args.n_boot, args.seed)
    boot_map = {"w25r": (boot_w25r, blocks_w25r), "huc8": (boot_huc8, blocks_huc8)}

    sets = scored_sets(s)
    depth = band_masks(obs, [b[0] for b in DEPTH_BANDS] + [np.inf], DEPTH_LABELS)
    dist_src = band_masks(s["dist_src_km"].to_numpy("float64"), SRC_EDGES, SRC_LABELS)
    dist_nwis = band_masks(
        s["dist_nwis_km"].to_numpy("float64"), NWIS_EDGES, NWIS_LABELS
    )

    core, dep, dst, nws, shal, deltas = [], [], [], [], [], []
    for set_name, sm in sets.items():
        m = sm & finite_core
        log(f"panel {set_name}: n {int(m.sum())}")
        core += core_rows(obs, preds, m, set_name, "all", "overall")
        for lab, bm in depth.items():
            dep += core_rows(obs, preds, m & bm, set_name, lab, "depth")
        for lab, bm in dist_src.items():
            dst += core_rows(obs, preds, m & bm, set_name, lab, "dist_src")
        for lab, bm in dist_nwis.items():
            nws += core_rows(obs, preds, m & bm, set_name, lab, "dist_nwis")
        shal += shallow_rows(obs, preds, m, set_name)

        others = {
            "Ma": preds["Ma raster (resampled here)"],
            reference: preds[f"{reference} raster"],
        }
        pp = preds[f"{primary} raster"]
        deltas += delta_table(
            boot_map, obs, pp, others, {"all": m}, set_name, "overall"
        )
        deltas += delta_table(
            boot_map,
            obs,
            pp,
            others,
            {k: m & v for k, v in depth.items()},
            set_name,
            "depth",
        )
        deltas += delta_table(
            boot_map,
            obs,
            pp,
            others,
            {k: m & v for k, v in dist_src.items()},
            set_name,
            "dist_src",
        )
        if set_name == "A_primary_eligible":
            deltas += delta_table(
                boot_map,
                obs,
                pp,
                others,
                {k: m & v for k, v in dist_nwis.items()},
                set_name,
                "dist_nwis",
            )

    pd.DataFrame(core).to_csv(out / "panel_core.csv", index=False)
    pd.DataFrame(dep).to_csv(out / "panel_depth.csv", index=False)
    pd.DataFrame(dst).to_csv(out / "panel_dist.csv", index=False)
    pd.DataFrame(nws).to_csv(out / "panel_nwis.csv", index=False)
    pd.DataFrame(shal).to_csv(out / "panel_shallow.csv", index=False)
    pd.DataFrame(deltas).to_csv(out / "delta_bootstrap.csv", index=False)

    # ---------------- rendering loss ----------------
    rl = []
    for arm in (primary, reference):
        col = f"{arm}__pt_pred_dtw_m"
        if col not in s.columns:
            continue
        mapv = s[f"{arm}__dtw"].to_numpy("float64")
        ptv = s[col].to_numpy("float64")
        for set_name in ("A_primary_eligible", "D_context_all_targets"):
            m = sets[set_name] & np.isfinite(mapv) & np.isfinite(ptv)
            d = np.abs(mapv[m] - ptv[m])
            rl.append(
                {
                    "arm": arm,
                    "set": set_name,
                    "band": "all",
                    "n": int(m.sum()),
                    "median_abs_diff_m": float(np.median(d)),
                    "p90_abs_diff_m": float(np.percentile(d, 90)),
                    "p99_abs_diff_m": float(np.percentile(d, 99)),
                    "max_abs_diff_m": float(d.max()),
                    "mean_abs_diff_m": float(d.mean()),
                    "map_MAD_m": mad(obs[m], mapv[m]),
                    "point_MAD_m": mad(obs[m], ptv[m]),
                }
            )
            for lab, bm in depth.items():
                mm = m & bm
                if mm.sum() == 0:
                    continue
                d = np.abs(mapv[mm] - ptv[mm])
                rl.append(
                    {
                        "arm": arm,
                        "set": set_name,
                        "band": lab,
                        "n": int(mm.sum()),
                        "median_abs_diff_m": float(np.median(d)),
                        "p90_abs_diff_m": float(np.percentile(d, 90)),
                        "p99_abs_diff_m": float(np.percentile(d, 99)),
                        "max_abs_diff_m": float(d.max()),
                        "mean_abs_diff_m": float(d.mean()),
                        "map_MAD_m": mad(obs[mm], mapv[mm]),
                        "point_MAD_m": mad(obs[mm], ptv[mm]),
                    }
                )
    pd.DataFrame(rl).to_csv(out / "rendering_loss.csv", index=False)

    # ---------------- sigma ----------------
    sig_sum, sig_dec = [], []
    for arm in (primary, reference):
        for set_name in ("A_primary_eligible", "D_context_all_targets"):
            m = sets[set_name] & finite_core
            su, de = sigma_panel(
                obs[m],
                s[f"{arm}__dtw"].to_numpy("float64")[m],
                s[f"{arm}__sigma"].to_numpy("float64")[m],
                set_name,
                arm,
            )
            sig_sum.append(su)
            sig_dec += de
    pd.DataFrame(sig_sum).to_csv(out / "sigma_calibration.csv", index=False)
    pd.DataFrame(sig_dec).to_csv(out / "sigma_deciles.csv", index=False)

    # ---------------- negatives at scored sites ----------------
    neg = []
    for arm in (primary, reference):
        v = s[f"{arm}__dtw"].to_numpy("float64")
        for set_name, sm in sets.items():
            m = sm & np.isfinite(v)
            nmask = m & (v < 0)
            neg.append(
                {
                    "arm": arm,
                    "set": set_name,
                    "n": int(m.sum()),
                    "n_negative": int(nmask.sum()),
                    "frac_negative": float(nmask.sum() / max(m.sum(), 1)),
                    "min_dtw_m": float(v[m].min()),
                    "median_obs_at_negative_m": float(np.median(obs[nmask]))
                    if nmask.any()
                    else np.nan,
                }
            )
    pd.DataFrame(neg).to_csv(out / "negatives.csv", index=False)

    # ---------------- per-basin MAD ----------------
    pm = sets["A_primary_eligible"] & finite_core
    pb_rows = []
    for basin in sorted(set(s["basin_hit"][pm])):
        bm = pm & (s["basin_hit"].to_numpy() == basin)
        pb_rows.append(
            {
                "basin": basin,
                "n": int(bm.sum()),
                f"{primary}_MAD_m": mad(obs[bm], preds[f"{primary} raster"][bm]),
                f"{reference}_MAD_m": mad(obs[bm], preds[f"{reference} raster"][bm]),
                "ma_MAD_m": mad(obs[bm], preds["Ma raster (resampled here)"][bm]),
            }
        )
    pb = pd.DataFrame(pb_rows)
    pb.to_csv(out / "per_basin_mad.csv", index=False)

    # ---------------- calling layers ----------------
    curve, csum, ctot = calling_layers(s, primary, reference, pm, boot_huc8, cells)
    curve.to_csv(out / "calling_curve.csv", index=False)
    csum.to_csv(out / "calling_summary.csv", index=False)
    ctot.to_csv(out / "calling_cells.csv", index=False)

    # ---------------- report ----------------
    with open(out / "report.md", "w") as fh:
        fh.write(f"# NV statewide raster eval: {primary} vs {reference} vs Ma\n\n")
        fh.write(f"sites {len(s)}; basins {len(verify)}; n_boot {args.n_boot}\n\n")
        fh.write("## verification (per basin)\n\n")
        fh.write(verify.to_csv(index=False))
        fh.write("\n## provenance\n\n")
        fh.write(md_table(prov))
        if len(cm):
            fh.write("\n## cross-machine pilot diff\n\n")
            fh.write(md_table(cm.round(6).astype(str)))
        for name, df in (
            ("core panels", pd.DataFrame(core)),
            ("depth bands", pd.DataFrame(dep)),
            ("distance to admitted source", pd.DataFrame(dst)),
            ("distance to NWIS", pd.DataFrame(nws)),
            ("shallow class", pd.DataFrame(shal)),
            ("deltas", pd.DataFrame(deltas)),
            ("rendering loss", pd.DataFrame(rl)),
            ("sigma", pd.DataFrame(sig_sum)),
            ("sigma deciles", pd.DataFrame(sig_dec)),
            ("negatives", pd.DataFrame(neg)),
            ("per-basin MAD", pb),
            ("calling summary", csum),
            ("calling cells", ctot),
        ):
            fh.write(f"\n## {name}\n\n")
            fh.write(df.round(4).to_csv(index=False))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
