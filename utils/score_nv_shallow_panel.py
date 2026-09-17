"""Nevada primary-panel scorer for the shallow-lever arms (tier 2 of
``notes/SHALLOW_LEVER_EXPERIMENTS.md``).

Scores one or more ``predict_gnn_at_points.py`` outputs at the NDWR primary
held-out sites (``is_admission_eligible`` rows of the admission-targets file)
against a reference arm and against Ma, on the common footprint, with the full
metric panel, the doctrine depth bands, distance-to-source bins, an irrigated
stratum, shallow precision / recall at 2 / 5 / 10 m, a paired basin-block
bootstrap, and the calling-surface sweep (sigma, fold spread, Laplace CDF, and
ordinal probabilities where an arm carries them).

Definitions (every number in metres unless stated):

* residual = predicted - observed depth to water; negative = predicted shallower
  than the well.
* MAD = median |residual|; bias = mean residual; RMSE = root mean square
  residual; p95 = 95th percentile of |residual|.
* shallow precision at t = fraction of sites called shallower than t that are
  observed shallower than t; recall = fraction of observed-shallow sites called
  shallow (dimensionless).
* delta rows: ``delta = other - arm`` in MAD / RMSE (m, positive = arm better)
  and ``delta = arm - other`` in precision / recall (dimensionless, positive =
  arm better). The CI95 bounds that difference on this site set under a paired
  block bootstrap, blocks = the WBD HUC8 polygon containing the site, drawn with
  replacement, the same resampled site set applied to both predictors.
* irrigated stratum: distance from the site to the nearest 2017 irrigated
  100 m cell (``irrigation_2017.tif``); ``irr0`` = on an irrigated cell,
  ``irr500`` = within 500 m.
* calling surface: call = predicted DTW < t AND gate variable below its cutoff
  (sigma / fold spread; cutoffs are quantiles over the sites predicted < t), or
  P(DTW < t) >= p_cut for the Laplace-CDF and ordinal probability variables.
  coverage = fraction of sites predicted < t that are called.

Run::

    uv run --directory /home/dgketchum/code/handily python \
        /home/dgketchum/code/handily/utils/score_nv_shallow_panel.py \
        --arm r1e=/data/ssd2/handily/nv/regional/wells/ndwr_admit_ramp_pt_preds.parquet \
        --arm m20=/data/ssd2/handily/nv/regional/wells/shallow/ndwr_pt_preds_m20.parquet \
        --ref r1e --out-dir /data/ssd2/handily/nv/regional/shallow_levers/tier2
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
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "utils"))

from compare_nv_ma_handily_shape import (  # noqa: E402
    DEPTH_BANDS,
    DEPTH_LABELS,
    DIST_EDGES,
    DIST_LABELS,
    SHALLOW_THRESHOLDS,
    panel_stats,
)

HUC8_POLYS = "/nas/hydrography/HUC_Boundaries/wbd_national/wbdhu8_5070.parquet"
TARGETS = "/data/ssd2/handily/nv/regional/wells/ndwr_admission_targets.parquet"
SOURCES = "/data/ssd2/handily/nv/regional/wells/ndwr_admitted_sources_frozen.parquet"
WELLS_FGB = "/data/ssd2/handily/nv/regional/wells/nv_wells_attributed.fgb"
IRRIGATION = "/nas/handily/covariates/anthropogenic/irrigation_2017.tif"
WATER_TABLE_CLASSES = ("unconfined", "unconfined_marginal")
CALL_THRESHOLDS = (2.0, 5.0)
GATE_QUANTILES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)
PROB_CUTS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
#: attractor windows watched on the sites observed under 2 m: the r1e 3 m mode
#: and the 2 m mode a --mirror-depth-m 2 arm would produce.
MODE_WINDOWS = {"2.9-3.3 m": (2.9, 3.3), "1.9-2.3 m": (1.9, 2.3)}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_targets(path: str) -> pd.DataFrame:
    t = pd.read_parquet(path)
    t = t[t["is_admission_eligible"].to_numpy(bool)].reset_index(drop=True)
    for c in ("obs_dtw_m", "ma_dtw_m"):
        if c not in t.columns:
            raise SystemExit(f"targets file lacks {c}")
    return t


def huc8_blocks(t: pd.DataFrame) -> np.ndarray:
    """Bootstrap block = the WBD HUC8 polygon containing the site (the render
    basin). ``w25r_basin`` in the targets file is the controlling-reach basin
    and is null for 45% of primary sites, so it is not used. A site outside
    every polygon (rare, coastline/state-edge artefacts) takes the nearest."""
    polys = gpd.read_parquet(HUC8_POLYS)[["huc8", "geometry"]]
    pts = gpd.GeoDataFrame(
        {"_i": np.arange(len(t))},
        geometry=gpd.points_from_xy(t["x5070"], t["y5070"]),
        crs="EPSG:5070",
    )
    j = gpd.sjoin(pts, polys, how="left", predicate="within").drop_duplicates("_i")
    blk = j.sort_values("_i")["huc8"].to_numpy(object)
    miss = pd.isna(blk)
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss], polys, how="left").drop_duplicates("_i")
        blk[miss] = near.sort_values("_i")["huc8"].to_numpy(object)
        log(f"  {int(miss.sum())} sites outside every HUC8 polygon -> nearest polygon")
    return blk.astype(str)


def join_arm(t: pd.DataFrame, path: str, name: str) -> pd.DataFrame:
    """Join an arm's point predictions onto the targets by exact coordinate."""
    p = pd.read_parquet(path)
    keep = ["x5070", "y5070", "pred_dtw_m", "fold_spread_m"] + [
        c
        for c in p.columns
        if c == "sigma_m" or c.startswith("p_dtw_lt_") or c.startswith("gate_w_")
    ]
    p = p[keep].drop_duplicates(["x5070", "y5070"])
    m = t[["x5070", "y5070"]].merge(p, on=["x5070", "y5070"], how="left")
    if len(m) != len(t):
        raise SystemExit(f"{name}: coordinate join is not 1:1")
    miss = int(m["pred_dtw_m"].isna().sum())
    if miss:
        raise SystemExit(f"{name}: {miss} primary sites have no prediction")
    return m.drop(columns=["x5070", "y5070"]).add_prefix(f"{name}__")


def confinement_class(t: pd.DataFrame) -> np.ndarray:
    g = gpd.read_file(WELLS_FGB, columns=["confinement_class"])
    tree = cKDTree(np.c_[g.geometry.x.to_numpy(), g.geometry.y.to_numpy()])
    d, i = tree.query(t[["x5070", "y5070"]].to_numpy("float64"), k=1)
    cls = g["confinement_class"].to_numpy(object)[i]
    far = d > 1.0
    if far.any():
        log(
            f"  {int(far.sum())} sites have no attributed well within 1 m -> class 'unmatched'"
        )
        cls[far] = "unmatched"
    return cls


def dist_to_irrigation_m(t: pd.DataFrame) -> np.ndarray:
    x = t["x5070"].to_numpy("float64")
    y = t["y5070"].to_numpy("float64")
    with rasterio.open(IRRIGATION) as ds:
        pad = 5000.0
        w = (
            from_bounds(
                x.min() - pad, y.min() - pad, x.max() + pad, y.max() + pad, ds.transform
            )
            .round_offsets()
            .round_lengths()
        )
        arr = ds.read(1, window=w)
        tr = ds.window_transform(w)
        res = float(ds.res[0])
    irr = arr == 1
    dist = distance_transform_edt(~irr) * res
    col = np.floor((x - tr.c) / tr.a).astype(int)
    row = np.floor((y - tr.f) / tr.e).astype(int)
    if (
        (row < 0).any()
        or (row >= dist.shape[0]).any()
        or (col < 0).any()
        or (col >= dist.shape[1]).any()
    ):
        raise SystemExit("primary sites fall outside the irrigation window")
    return dist[row, col]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def pr_at(obs: np.ndarray, called: np.ndarray, thr: float) -> tuple[float, float]:
    obs_s = obs < thr
    tp = int((obs_s & called).sum())
    prec = tp / called.sum() if called.sum() else np.nan
    rec = tp / obs_s.sum() if obs_s.sum() else np.nan
    return prec, rec


def f1(p: float, r: float) -> float:
    return (
        2 * p * r / (p + r)
        if np.isfinite(p) and np.isfinite(r) and (p + r) > 0
        else np.nan
    )


def laplace_cdf(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.maximum(b, 1e-6)
    return np.where(x < 0, 0.5 * np.exp(x / b), 1.0 - 0.5 * np.exp(-x / b))


class BlockBoot:
    """Paired block bootstrap over basins; one resample set reused everywhere."""

    def __init__(self, blocks: np.ndarray, n_boot: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        ug = np.unique(blocks)
        gidx = [np.where(blocks == g)[0] for g in ug]
        self.n_blocks = len(ug)
        self.samples = [
            np.concatenate(
                [gidx[j] for j in rng.choice(len(ug), len(ug), replace=True)]
            )
            for _ in range(n_boot)
        ]

    def ci(self, stat, mask: np.ndarray) -> tuple[float, float]:
        """CI95 of ``stat(idx)`` over resamples restricted to ``mask``."""
        vals = []
        for s in self.samples:
            idx = s[mask[s]]
            vals.append(stat(idx) if idx.size else np.nan)
        vals = np.asarray(vals, dtype="float64")
        ok = np.isfinite(vals)
        if ok.sum() < len(vals) * 0.5:
            return np.nan, np.nan
        return float(np.percentile(vals[ok], 2.5)), float(np.percentile(vals[ok], 97.5))


SLOPE_WINDOWS = ((0.0, 5.0), (0.0, 10.0), (2.0, 10.0))


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 30:
        return np.nan
    xc = x - x.mean()
    return float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))


def slope_rows(
    boot: BlockBoot,
    obs: np.ndarray,
    preds: dict[str, np.ndarray],
    ref: str,
    strata: dict[str, np.ndarray],
) -> list[dict]:
    """OLS slope of predicted on observed DTW inside observed-depth windows.

    Slope 1 = calibrated range; below 1 = range compression. CI95 bounds the
    slope (and the paired slope difference vs ``ref``) under the same HUC8
    block bootstrap as the metric panel. Theil-Sen is the robust companion.
    """
    from scipy.stats import theilslopes

    rows = []
    for stratum, base in strata.items():
        for lo, hi in SLOPE_WINDOWS:
            m = base & (obs >= lo) & (obs < hi)
            if m.sum() < 30:
                continue
            for name, p in preds.items():
                row = {
                    "stratum": stratum,
                    "window": f"{lo:g}-{hi:g} m",
                    "predictor": name,
                    "n": int(m.sum()),
                    "ols_slope": _ols_slope(obs[m], p[m]),
                }
                row["ols_slope_lo"], row["ols_slope_hi"] = boot.ci(
                    lambda i, p=p: _ols_slope(obs[i], p[i]), m
                )
                row["theilsen_slope"] = float(theilslopes(p[m], obs[m])[0])
                row["intercept_m"] = float(
                    p[m].mean() - row["ols_slope"] * obs[m].mean()
                )
                if name != ref and ref in preds:
                    r = preds[ref]
                    row["delta_slope_vs_ref"] = row["ols_slope"] - _ols_slope(
                        obs[m], r[m]
                    )
                    row["delta_slope_lo"], row["delta_slope_hi"] = boot.ci(
                        lambda i, p=p, r=r: (
                            _ols_slope(obs[i], p[i]) - _ols_slope(obs[i], r[i])
                        ),
                        m,
                    )
                rows.append(row)
    return rows


def delta_rows(
    boot: BlockBoot,
    obs: np.ndarray,
    arm: np.ndarray,
    other: np.ndarray,
    strata: dict[str, np.ndarray],
    arm_name: str,
    other_name: str,
) -> list[dict]:
    aa, ao = np.abs(arm - obs), np.abs(other - obs)
    rows = []
    for label, m in strata.items():
        if m.sum() < 30:
            continue
        row = {
            "arm": arm_name,
            "vs": other_name,
            "stratum": label,
            "n": int(m.sum()),
            "n_blocks": boot.n_blocks,
        }
        row["delta_mad_m"] = float(np.median(ao[m]) - np.median(aa[m]))
        row["delta_mad_lo_m"], row["delta_mad_hi_m"] = boot.ci(
            lambda i: np.median(ao[i]) - np.median(aa[i]), m
        )
        row["delta_rmse_m"] = float(
            np.sqrt(np.mean(ao[m] ** 2)) - np.sqrt(np.mean(aa[m] ** 2))
        )
        row["delta_rmse_lo_m"], row["delta_rmse_hi_m"] = boot.ci(
            lambda i: np.sqrt(np.mean(ao[i] ** 2)) - np.sqrt(np.mean(aa[i] ** 2)), m
        )
        for thr in CALL_THRESHOLDS:
            ca, co = arm < thr, other < thr
            pa, ra = pr_at(obs[m], ca[m], thr)
            po, ro = pr_at(obs[m], co[m], thr)
            row[f"delta_prec_{thr:g}m"] = pa - po
            row[f"delta_prec_{thr:g}m_lo"], row[f"delta_prec_{thr:g}m_hi"] = boot.ci(
                lambda i, thr=thr, ca=ca, co=co: (
                    pr_at(obs[i], ca[i], thr)[0] - pr_at(obs[i], co[i], thr)[0]
                ),
                m,
            )
            row[f"delta_rec_{thr:g}m"] = ra - ro
            row[f"delta_rec_{thr:g}m_lo"], row[f"delta_rec_{thr:g}m_hi"] = boot.ci(
                lambda i, thr=thr, ca=ca, co=co: (
                    pr_at(obs[i], ca[i], thr)[1] - pr_at(obs[i], co[i], thr)[1]
                ),
                m,
            )
        rows.append(row)
    return rows


def calling_sweep(
    boot: BlockBoot,
    obs: np.ndarray,
    pred: np.ndarray,
    gates: dict[str, np.ndarray],
    probs: dict[str, np.ndarray],
    strata: dict[str, np.ndarray],
    arm_name: str,
) -> list[dict]:
    rows = []
    for thr in CALL_THRESHOLDS:
        shallow = pred < thr
        for label, m in strata.items():
            if (m & shallow).sum() < 10:
                continue
            for gname, g in gates.items():
                cuts = np.quantile(g[m & shallow], GATE_QUANTILES)
                for q, c in zip(GATE_QUANTILES, cuts):
                    called = shallow & (g <= c)
                    rows.append(
                        _sweep_row(
                            boot,
                            obs,
                            called,
                            shallow,
                            m,
                            thr,
                            arm_name,
                            label,
                            gname,
                            f"q{int(q * 100)}",
                            float(c),
                        )
                    )
            for pname, p in probs.items():
                if pname.startswith("p_dtw_lt_") and pname != f"p_dtw_lt_{thr:g}m":
                    continue
                pt = p[f"{thr:g}"] if isinstance(p, dict) else p
                for pc in PROB_CUTS:
                    called = pt >= pc
                    rows.append(
                        _sweep_row(
                            boot,
                            obs,
                            called,
                            shallow,
                            m,
                            thr,
                            arm_name,
                            label,
                            pname,
                            f"p>={pc}",
                            float(pc),
                        )
                    )
    return rows


def _sweep_row(
    boot, obs, called, shallow, m, thr, arm_name, label, gname, cut_label, cut_value
) -> dict:
    prec, rec = pr_at(obs[m], called[m], thr)
    lo, hi = boot.ci(lambda i: pr_at(obs[i], called[i], thr)[0], m)
    return {
        "arm": arm_name,
        "threshold_m": thr,
        "stratum": label,
        "gate": gname,
        "cut": cut_label,
        "cut_value": cut_value,
        "n_sites": int(m.sum()),
        "n_obs_shallow": int((obs[m] < thr).sum()),
        "n_pred_shallow": int(shallow[m].sum()),
        "n_called": int(called[m].sum()),
        "coverage": float(called[m].sum() / shallow[m].sum())
        if shallow[m].sum()
        else np.nan,
        "precision": prec,
        "precision_lo": lo,
        "precision_hi": hi,
        "recall": rec,
        "f1": f1(prec, rec),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm", action="append", required=True, help="NAME=pt_preds.parquet"
    )
    ap.add_argument("--ref", required=True, help="arm NAME used as the reference")
    ap.add_argument("--targets", default=TARGETS)
    ap.add_argument("--sources", default=SOURCES)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t = load_targets(args.targets)
    log(f"primary sites: {len(t)}")
    arms = dict(a.split("=", 1) for a in args.arm)
    if args.ref not in arms:
        raise SystemExit(f"--ref {args.ref} is not an --arm")
    for name, path in arms.items():
        t = pd.concat([t, join_arm(t, path, name)], axis=1)
        log(f"  joined {name}: {path}")

    adm = pd.read_parquet(args.sources, columns=["x5070", "y5070"])
    d_src, _ = cKDTree(adm.to_numpy("float64")).query(
        t[["x5070", "y5070"]].to_numpy("float64")
    )
    t["dist_source_km"] = d_src / 1000.0
    t["confinement_class"] = confinement_class(t)
    t["dist_irr_m"] = dist_to_irrigation_m(t)
    log(
        "  confinement: "
        + json.dumps(t["confinement_class"].value_counts().to_dict())
        + f"; on irrigated cell {int((t['dist_irr_m'] == 0).sum())}, within 500 m {int((t['dist_irr_m'] <= 500).sum())}"
    )

    obs = t["obs_dtw_m"].to_numpy("float64")
    ma = t["ma_dtw_m"].to_numpy("float64")
    preds = {n: t[f"{n}__pred_dtw_m"].to_numpy("float64") for n in arms}
    common = np.isfinite(obs) & np.isfinite(ma)
    for p in preds.values():
        common &= np.isfinite(p)
    log(f"  common footprint (finite obs, Ma, every arm): {int(common.sum())}")
    t = t[common].reset_index(drop=True)
    obs, ma = obs[common], ma[common]
    preds = {n: p[common] for n, p in preds.items()}
    blocks = huc8_blocks(t)
    t["huc8_block"] = blocks
    boot = BlockBoot(blocks, args.n_boot, args.seed)
    log(f"  bootstrap: {args.n_boot} resamples over {boot.n_blocks} HUC8 blocks")

    wt = np.isin(t["confinement_class"].to_numpy(object), WATER_TABLE_CLASSES)
    irr0 = t["dist_irr_m"].to_numpy() == 0
    irr500 = t["dist_irr_m"].to_numpy() <= 500
    d_src = t["dist_source_km"].to_numpy("float64")
    strata: dict[str, np.ndarray] = {"all": np.ones(len(t), bool)}
    for (lo, hi), lab in zip(DEPTH_BANDS, DEPTH_LABELS):
        strata[f"obs {lab}"] = (obs >= lo) & (obs < hi)
    for i, lab in enumerate(DIST_LABELS):
        strata[f"source {lab}"] = (d_src >= DIST_EDGES[i]) & (d_src < DIST_EDGES[i + 1])
    strata["water-table classes"] = wt
    strata["non-water-table classes"] = ~wt
    strata["irr0"] = irr0
    strata["irr500"] = irr500
    strata["irr500 obs<10 m"] = irr500 & (obs < 10)
    strata["irr500 obs<2 m"] = irr500 & (obs < 2)
    call_strata = {
        k: strata[k]
        for k in ("all", "water-table classes", "irr500", "irr500 obs<10 m")
    }

    all_preds = {**preds, "Ma": ma}
    panel, shallow = [], []
    for label, m in strata.items():
        for name, p in all_preds.items():
            row = {"stratum": label, "predictor": name}
            row.update(
                panel_stats(obs[m], p[m]) if m.sum() >= 30 else {"n": int(m.sum())}
            )
            panel.append(row)
        if label in call_strata or label.startswith("source"):
            for thr in SHALLOW_THRESHOLDS:
                for name, p in all_preds.items():
                    pr, rc = pr_at(obs[m], (p < thr)[m], thr)
                    shallow.append(
                        {
                            "stratum": label,
                            "threshold_m": thr,
                            "predictor": name,
                            "n_obs_shallow": int((obs[m] < thr).sum()),
                            "n_pred_shallow": int((p[m] < thr).sum()),
                            "precision": pr,
                            "recall": rc,
                            "f1": f1(pr, rc),
                        }
                    )
    pd.DataFrame(panel).to_csv(out / "panel.csv", index=False)
    pd.DataFrame(shallow).to_csv(out / "shallow_pr.csv", index=False)

    deltas = []
    for name, p in preds.items():
        if name != args.ref:
            deltas += delta_rows(boot, obs, p, preds[args.ref], strata, name, args.ref)
        deltas += delta_rows(boot, obs, p, ma, strata, name, "Ma")
        log(f"  deltas done: {name}")
    pd.DataFrame(deltas).to_csv(out / "deltas.csv", index=False)
    slope_strata = {k: strata[k] for k in ("all", "water-table classes", "irr500")}
    pd.DataFrame(slope_rows(boot, obs, all_preds, args.ref, slope_strata)).to_csv(
        out / "slopes.csv", index=False
    )
    log("  slopes done")

    sweep, modes = [], []
    for name, p in preds.items():
        gates = (
            {"sigma_m": t[f"{name}__sigma_m"].to_numpy("float64")}
            if f"{name}__sigma_m" in t
            else {}
        )
        gates["fold_spread_m"] = t[f"{name}__fold_spread_m"].to_numpy("float64")
        probs: dict = {}
        if f"{name}__sigma_m" in t:
            b = gates["sigma_m"]
            probs["laplace_cdf"] = {
                f"{thr:g}": laplace_cdf(thr - p, b) for thr in CALL_THRESHOLDS
            }
        for thr in CALL_THRESHOLDS:
            c = f"{name}__p_dtw_lt_{thr:g}m"
            if c in t:
                probs[f"p_dtw_lt_{thr:g}m"] = t[c].to_numpy("float64")
        sweep += calling_sweep(boot, obs, p, gates, probs, call_strata, name)
        # Spearman of each gate variable against |residual| on sites predicted < 5 m
        sh = p < 5
        for gname, g in gates.items():
            rho = pd.Series(g[sh]).corr(
                pd.Series(np.abs(p - obs)[sh]), method="spearman"
            )
            modes.append(
                {
                    "arm": name,
                    "stat": f"spearman({gname}, |resid|) pred<5 m",
                    "value": float(rho),
                    "n": int(sh.sum()),
                }
            )
        # attractor check on the sites observed under 2 m
        o2 = obs < 2
        q = np.percentile(p[o2], [10, 50, 90])
        modes.append(
            {
                "arm": name,
                "stat": "pred q10 at obs<2 m",
                "value": float(q[0]),
                "n": int(o2.sum()),
            }
        )
        modes.append(
            {
                "arm": name,
                "stat": "pred q50 at obs<2 m",
                "value": float(q[1]),
                "n": int(o2.sum()),
            }
        )
        modes.append(
            {
                "arm": name,
                "stat": "pred q90 at obs<2 m",
                "value": float(q[2]),
                "n": int(o2.sum()),
            }
        )
        modes.append(
            {
                "arm": name,
                "stat": "frac pred<2 m at obs<2 m",
                "value": float((p[o2] < 2).mean()),
                "n": int(o2.sum()),
            }
        )
        for lab, (lo, hi) in MODE_WINDOWS.items():
            modes.append(
                {
                    "arm": name,
                    "stat": f"frac pred in {lab} at obs<2 m",
                    "value": float(((p[o2] >= lo) & (p[o2] < hi)).mean()),
                    "n": int(o2.sum()),
                }
            )
            modes.append(
                {
                    "arm": name,
                    "stat": f"frac pred in {lab} all sites",
                    "value": float(((p >= lo) & (p < hi)).mean()),
                    "n": int(len(p)),
                }
            )
        log(f"  sweep done: {name}")
    pd.DataFrame(sweep).to_csv(out / "calling_sweep.csv", index=False)
    pd.DataFrame(modes).to_csv(out / "diagnostics.csv", index=False)

    t.to_parquet(out / "sites_scored.parquet", index=False)
    summary = {
        "n_primary": int(len(t)),
        "n_blocks": boot.n_blocks,
        "n_boot": args.n_boot,
        "arms": arms,
        "ref": args.ref,
        "strata_n": {k: int(v.sum()) for k, v in strata.items()},
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
