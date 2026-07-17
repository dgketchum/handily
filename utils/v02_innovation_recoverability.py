"""WP1 ticket 1 — static innovation recoverability harness (residual IDW + kriging).

Implements `notes/plans/HANDILY_V2_FIELD_MODEL_PLAN.md` §14: before any conditional
neural model, answer the prerequisite question — is the remaining static background
error a recoverable spatial field, and over what distance?

Definitions (plan §6.2), all in metres, WTE (head) space:
    innovation_i        = obs_wte_i - S0_oof(x_i)     S0_oof = frozen w25_prod OOF
                                                      gnn_wte_hat_m (leak-free per
                                                      well by fold construction)
    analysis_increment  = A(x | eligible context innovations)
    best_estimate error = innovation - predicted increment   (identical algebra in
                          innovation space; baseline = zero increment)

Context mask contract (plan §6.3, static adaptation):
  - context for a target excludes the target's whole CV fold (matches the OOF
    discipline of the background), the target's physical site (site_id from WP0,
    which also covers nest siblings), and an optional spatial buffer;
  - context innovations use their own OOF background values (never assimilated);
  - variogram/normalization parameters are fit per fold on that fold's context
    pool only; the serialized final model is refit on all wells and is only for
    the analysis product, never for evaluation;
  - empty context must return exactly zero increment and background variance.

Models (plan §6.4 rungs 1-2; rungs 3-4 are learned and NOT run here):
  idw       residual kNN-IDW over context innovations, k x power sweep, plus a
            relief-lifted distance variant (vw from WP0 production convention).
  kriging   simple kriging (mean 0) with per-fold-fit exponential/spherical PSD
            covariance, nugget on the diagonal, predictive variance, calibration
            and sigma-monotonicity checks; per-HUC2 and anisotropy diagnostics.

Outputs (--out-dir, default /data/ssd2/handily/conus/wte_gnn/v02/wp1):
  innovations.parquet         per-well innovation + per-model LOO predictions
  variogram_models.json       per-fold + pooled fits, per-HUC2/anisotropy diags
  recoverability_report.json  full metric panel (skill by distance band, depth
                              band, HUC2, source, aridity/relief, buffer sweep)
  recoverability_summary.md   plain-language answers to the §14 questions

Usage:
    uv run python utils/v02_innovation_recoverability.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v02_metrics import (  # noqa: E402
    DIST_CONTEXT_BANDS_KM,
    paired_improvement,
    sigma_monotonicity,
)

log = logging.getLogger("v02_innovation_recoverability")

WTE_GNN = "/data/ssd2/handily/conus/wte_gnn"
CONTRACT_DIR = f"{WTE_GNN}/v02/contract"
OUT_DIR = f"{WTE_GNN}/v02/wp1"

PAIR_BIN_KM = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0]
DEPTH_BANDS = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
IDW_GRID = [(8, 1.0), (8, 2.0), (16, 1.0), (16, 2.0), (32, 1.0), (32, 2.0)]
BUFFERS_KM = (0.0, 1.0, 5.0)
RELIEF_VW = 100.0  # production relief-lift vertical weight (build_conus_graph_inputs)
N_NEIGHBORS_KRIGE = 32
MAX_NEAR_PAIRS = 4_000_000
N_RANDOM_PAIRS = 4_000_000


def _corr_exp(h, nug, rng_):
    return (1.0 - nug) * np.exp(-h / rng_)


def _corr_sph(h, nug, rng_):
    hr = np.clip(h / rng_, 0.0, 1.0)
    return (1.0 - nug) * (1.0 - 1.5 * hr + 0.5 * hr**3)


CORR_FNS = {"exponential": _corr_exp, "spherical": _corr_sph}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_inputs(args) -> pd.DataFrame:
    contract = Path(args.contract_dir)
    panels = pd.read_parquet(contract / "wells_panels.parquet")
    oof = pd.read_parquet(
        contract / "frozen_baseline" / "gnn_oof_predictions.parquet",
        columns=[
            "canonical_id",
            "is_water_pseudo",
            "obs_wte_m",
            "gnn_wte_hat_m",
            "gnn_dtw_m",
            "obs_dtw_m",
            "gnn_sigma_m",
        ],
    )
    oof = oof[~oof["is_water_pseudo"].astype(bool)]
    panels = panels[~panels["is_water_pseudo"].astype(bool)]
    df = panels.merge(
        oof.drop(columns=["is_water_pseudo"]),
        on="canonical_id",
        how="inner",
        validate="one_to_one",
    )
    n0 = len(df)
    df["innovation_m"] = df["obs_wte_m"] - df["gnn_wte_hat_m"]
    df = df[np.isfinite(df["innovation_m"])]
    # sacrificial geography stays locked out of every WP1 fit and report
    df = df[df["regional_holdout"] != "sacrificial"].reset_index(drop=True)
    log.info(
        "population: %d wells with finite innovation (of %d joined; "
        "sacrificial excluded)",
        len(df),
        n0,
    )
    return df


# --------------------------------------------------------------------------- #
# variogram
# --------------------------------------------------------------------------- #
def collect_pairs(
    xy: np.ndarray, z: np.ndarray, seed: int, near_r_km: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Pair sample for the empirical correlogram: all near-field pairs (dense,
    capped) plus uniform random far-field pairs. Returns (dist_km, cross-product
    of innovations about the mean)."""
    rng = np.random.default_rng(seed)
    n = len(xy)
    tree = cKDTree(xy)
    near = np.array(list(tree.query_pairs(near_r_km * 1000.0)), dtype=np.int64)
    if len(near) > MAX_NEAR_PAIRS:
        near = near[rng.choice(len(near), MAX_NEAR_PAIRS, replace=False)]
    far_i = rng.integers(0, n, N_RANDOM_PAIRS)
    far_j = rng.integers(0, n, N_RANDOM_PAIRS)
    keep = far_i != far_j
    pairs = np.vstack([near, np.column_stack([far_i[keep], far_j[keep]])])
    d_km = np.sqrt(((xy[pairs[:, 0]] - xy[pairs[:, 1]]) ** 2).sum(1)) / 1000.0
    mu = z.mean()
    cross = (z[pairs[:, 0]] - mu) * (z[pairs[:, 1]] - mu)
    return d_km, cross


def binned_correlogram(d_km: np.ndarray, cross: np.ndarray, var: float) -> dict:
    edges = np.asarray(PAIR_BIN_KM)
    idx = np.digitize(d_km, edges) - 1
    mids, rho, counts = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        mids.append(float(0.5 * (edges[b] + edges[b + 1])))
        counts.append(int(m.sum()))
        rho.append(float(cross[m].mean() / var) if m.sum() >= 30 else None)
    return {"bin_mid_km": mids, "rho": rho, "n_pairs": counts}


def fit_correlation_models(cg: dict, var: float) -> dict:
    """Weighted least-squares fit of PSD exponential + spherical correlation
    models to the binned correlogram (pattern follows
    utils/fit_groundwater_anomaly_covariance.py::fit_models)."""
    h = np.asarray(cg["bin_mid_km"], float)
    y = np.asarray(cg["rho"], float)
    w = np.sqrt(np.asarray(cg["n_pairs"], float))
    ok = np.isfinite(y) & (w > 0)
    h, y, w = h[ok], y[ok], w[ok]
    out = {}
    for name, fn in CORR_FNS.items():

        def resid(p, fn=fn):
            return (fn(h, p[0], p[1]) - y) * w

        res = least_squares(
            resid, x0=[0.5, 50.0], bounds=([0.0, 0.5], [1.0, 1000.0]), max_nfev=5000
        )
        nug, r = float(res.x[0]), float(res.x[1])
        pred = fn(h, nug, r)
        out[name] = {
            "nugget_frac": round(nug, 4),
            "range_km": round(r, 2),
            "sill_var_m2": round(var, 3),
            "structured_var_m2": round(var * (1 - nug), 3),
            "mappable_variance_fraction": round(1 - nug, 4),
            "fit_rmse": round(
                float(np.sqrt(np.average((pred - y) ** 2, weights=w**2))), 4
            ),
            "converged": bool(res.success),
        }
    return out


def anisotropy_diagnostic(xy, z, seed: int) -> dict:
    """Direction-binned near-field correlation (4 sectors, <=25 km)."""
    rng = np.random.default_rng(seed)
    tree = cKDTree(xy)
    pairs = np.array(list(tree.query_pairs(25_000.0)), dtype=np.int64)
    if len(pairs) > 2_000_000:
        pairs = pairs[rng.choice(len(pairs), 2_000_000, replace=False)]
    dx = xy[pairs[:, 1], 0] - xy[pairs[:, 0], 0]
    dy = xy[pairs[:, 1], 1] - xy[pairs[:, 0], 1]
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    mu, var = z.mean(), z.var()
    cross = (z[pairs[:, 0]] - mu) * (z[pairs[:, 1]] - mu) / var
    out = {}
    for lo, hi, name in (
        (0, 45, "EW"),
        (45, 90, "NE"),
        (90, 135, "NS"),
        (135, 180, "NW"),
    ):
        m = (ang >= lo) & (ang < hi)
        out[name] = {"rho": round(float(cross[m].mean()), 4), "n_pairs": int(m.sum())}
    return out


# --------------------------------------------------------------------------- #
# predictors under the mask contract
# --------------------------------------------------------------------------- #
def _context_pools(df: pd.DataFrame):
    """Per-fold context pools: for targets in fold f the pool is every well NOT
    in fold f. Site exclusion is applied at query time."""
    folds = np.sort(df["cv_fold"].unique())
    pools = {}
    for f in folds:
        pool_idx = np.where(df["cv_fold"].to_numpy() != f)[0]
        pools[int(f)] = pool_idx
    return pools


def _lifted(xy: np.ndarray, zsurf: np.ndarray, vw: float) -> np.ndarray:
    return np.column_stack([xy, vw * zsurf])


def idw_loo(
    df: pd.DataFrame, k: int, power: float, buffer_km: float, vw: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-fold+site+buffer-out IDW prediction of the innovation at every well.

    Returns (pred, dist_used_km) where dist_used_km is the distance to the
    nearest context well actually used (horizontal km even under relief lift)."""
    xy = df[["x5070", "y5070"]].to_numpy()
    coords = _lifted(xy, df["z_surf_well_m"].to_numpy(float), vw) if vw > 0 else xy
    z = df["innovation_m"].to_numpy(float)
    site = df["site_id"].to_numpy()
    fold = df["cv_fold"].to_numpy()
    pred = np.full(len(df), np.nan)
    dist_used = np.full(len(df), np.nan)
    extra = 16
    for f, pool_idx in _context_pools(df).items():
        tidx = np.where(fold == f)[0]
        if len(pool_idx) == 0:
            pred[tidx] = 0.0  # empty context -> zero increment identity
            continue
        tree = cKDTree(coords[pool_idx])
        kq = min(k + extra, len(pool_idx))
        d, nn = tree.query(coords[tidx], k=kq)
        if kq == 1:
            d, nn = d[:, None], nn[:, None]
        for row, ti in enumerate(tidx):
            cand = pool_idx[nn[row]]
            dd = d[row]
            ok = site[cand] != site[ti]
            if buffer_km > 0:
                hx = np.sqrt(((xy[cand] - xy[ti]) ** 2).sum(1)) / 1000.0
                ok &= hx >= buffer_km
            cand, dd = cand[ok][:k], dd[ok][:k]
            if len(cand) == 0:
                pred[ti] = 0.0  # empty context -> zero increment identity
                continue
            w = 1.0 / np.maximum(dd, 1.0) ** power
            pred[ti] = float(np.sum(w * z[cand]) / np.sum(w))
            dist_used[ti] = np.sqrt(((xy[cand] - xy[ti]) ** 2).sum(1)).min() / 1000.0
    return pred, dist_used


def krige_loo(
    df: pd.DataFrame,
    fold_models: dict,
    buffer_km: float,
    n_neighbors: int = N_NEIGHBORS_KRIGE,
    model_name: str = "exponential",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-fold+site+buffer-out simple kriging (mean 0) of the innovation.

    Covariance per fold f comes from fold_models[f] (fit without fold f).
    Diagonal carries the full variance (nugget = unmappable micro-scale +
    measurement error); the RHS carries structured covariance only. Returns
    (pred, sigma, dist_used_km); empty context -> (0, sqrt(var), nan)."""
    xy = df[["x5070", "y5070"]].to_numpy()
    z = df["innovation_m"].to_numpy(float)
    site = df["site_id"].to_numpy()
    fold = df["cv_fold"].to_numpy()
    corr = CORR_FNS[model_name]
    pred = np.zeros(len(df))
    sigma = np.full(len(df), np.nan)
    dist_used = np.full(len(df), np.nan)
    extra = 16
    for f, pool_idx in _context_pools(df).items():
        m = fold_models[f][model_name]
        nug, rng_km = m["nugget_frac"], m["range_km"]
        var = m["sill_var_m2"]
        tidx = np.where(fold == f)[0]
        if len(pool_idx) == 0:
            sigma[tidx] = float(np.sqrt(var))  # pred stays 0.0
            continue
        tree = cKDTree(xy[pool_idx])
        kq = min(n_neighbors + extra, len(pool_idx))
        d, nn = tree.query(xy[tidx], k=kq)
        if kq == 1:
            d, nn = d[:, None], nn[:, None]
        for row, ti in enumerate(tidx):
            cand = pool_idx[nn[row]]
            dd_km = d[row] / 1000.0
            ok = site[cand] != site[ti]
            if buffer_km > 0:
                ok &= dd_km >= buffer_km
            cand, dd_km = cand[ok][:n_neighbors], dd_km[ok][:n_neighbors]
            sigma[ti] = float(np.sqrt(var))
            if len(cand) == 0:
                continue  # pred stays 0.0 (empty-context identity)
            xn = xy[cand]
            hij = np.sqrt(((xn[:, None, :] - xn[None, :, :]) ** 2).sum(-1)) / 1000.0
            cmat = var * corr(hij, nug, rng_km)
            np.fill_diagonal(cmat, var)
            cmat += np.eye(len(cand)) * 1e-6 * var
            kvec = var * corr(dd_km, nug, rng_km)
            try:
                w = np.linalg.solve(cmat, kvec)
            except np.linalg.LinAlgError:
                continue
            pred[ti] = float(w @ z[cand])
            sigma[ti] = float(np.sqrt(max(var - w @ kvec, 1e-9)))
            dist_used[ti] = float(dd_km.min())
    return pred, sigma, dist_used


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def skill_panel(df: pd.DataFrame, pred_col: str, dist_col: str) -> dict:
    """Improvement of (innovation - increment) over raw innovation, stratified."""
    innov = df["innovation_m"].to_numpy(float)
    resid = innov - df[pred_col].to_numpy(float)
    panel = {"overall": paired_improvement(resid, innov)}

    d = df[dist_col].to_numpy(float)
    bands = {}
    for lo, hi in zip(DIST_CONTEXT_BANDS_KM[:-1], DIST_CONTEXT_BANDS_KM[1:]):
        m = np.isfinite(d) & (d >= lo) & (d < hi)
        label = f"{lo:g}-{hi:g}km" if np.isfinite(hi) else f"{lo:g}+km"
        bands[label] = paired_improvement(resid[m], innov[m])
    panel["by_dist_to_used_context"] = bands

    obs = df["mean_dtw"].to_numpy(float)
    by_depth = {}
    for lo, hi in DEPTH_BANDS:
        m = np.isfinite(obs) & (obs >= lo) & (obs < hi)
        label = f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"
        by_depth[label] = paired_improvement(resid[m], innov[m])
    panel["by_obs_depth_band"] = by_depth

    by_huc2 = {}
    for h2, g in df.groupby("huc2"):
        if len(g) >= 100:
            gi = g["innovation_m"].to_numpy(float)
            gr = gi - g[pred_col].to_numpy(float)
            by_huc2[str(h2)] = paired_improvement(gr, gi, n_boot=500)
    panel["by_huc2"] = by_huc2

    ai = df["aridity_index"].to_numpy(float)
    by_clim = {}
    for label, m in (
        ("arid_ai_lt_0.5", ai < 0.5),
        ("humid_ai_ge_0.5", ai >= 0.5),
    ):
        by_clim[label] = paired_improvement(resid[m], innov[m])
    panel["by_climate"] = by_clim

    tri = df["tri_100m"].to_numpy(float)
    t1, t2 = np.nanquantile(tri, [1 / 3, 2 / 3])
    by_relief = {}
    for label, m in (
        ("relief_low", tri <= t1),
        ("relief_mid", (tri > t1) & (tri <= t2)),
        ("relief_high", tri > t2),
    ):
        by_relief[label] = paired_improvement(resid[m], innov[m])
    panel["by_relief_tri_terciles"] = by_relief
    panel["relief_tercile_edges_tri_m"] = [round(float(t1), 3), round(float(t2), 3)]

    by_source = {}
    for label, m in (
        ("nwis", df["is_nwis"].to_numpy(bool)),
        ("non_nwis", ~df["is_nwis"].to_numpy(bool)),
    ):
        by_source[label] = paired_improvement(resid[m], innov[m])
    panel["by_source"] = by_source
    return panel


def write_summary_md(path: Path, report: dict, models: dict) -> None:
    best = report["best_model"]
    pooled = models["pooled_all_wells"]["exponential"]
    lines = [
        "# WP1 static innovation recoverability — summary",
        "",
        f"Generated {date.today()} by utils/v02_innovation_recoverability.py. "
        "All errors in metres (WTE space). skill = 1 - MAD_model/MAD_background "
        "(dimensionless; positive = analysis beats the frozen background).",
        "",
        "## The §14 question: is the residual background error a recoverable "
        "field, and over what distance?",
        "",
        f"- Pooled innovation variance: {models['pooled_all_wells']['variance_m2']:.1f} m² "
        f"(mean innovation {models['pooled_all_wells']['mean_m']:+.2f} m).",
        f"- Fitted exponential correlogram: nugget fraction "
        f"{pooled['nugget_frac']:.2f}, range {pooled['range_km']:.0f} km → "
        f"mappable variance fraction {pooled['mappable_variance_fraction']:.2f}.",
        f"- Best transparent model under the WP1 mask contract: **{best['name']}** "
        f"(MAD skill {best['overall']['mad_skill']:+.3f}, CI95 "
        f"{best['overall']['mad_skill_ci95']}, n={best['overall']['n']}).",
        "",
        "## Skill by distance to nearest used context well (best model)",
        "",
        "| band | n | MAD background (m) | MAD analysis (m) | skill |",
        "|---|---|---|---|---|",
    ]
    for band, r in report["models"][best["name"]]["by_dist_to_used_context"].items():
        if r.get("n", 0) > 0:
            lines.append(
                f"| {band} | {r['n']} | {r['mad_base_m']:.2f} | "
                f"{r['mad_model_m']:.2f} | {r['mad_skill']:+.3f} |"
            )
    lines += [
        "",
        "## Buffer sweep (support decay)",
        "",
        "| context buffer | overall MAD skill |",
        "|---|---|",
    ]
    for buf, r in report["buffer_sweep"].items():
        lines.append(f"| {buf} | {r['overall']['mad_skill']:+.3f} |")
    lines += ["", "## Calibration (kriging predictive sigma)", ""]
    cal = report.get("kriging_calibration", {})
    for k, v in cal.items():
        lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract-dir", default=CONTRACT_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relief-vw", type=float, default=RELIEF_VW)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_inputs(args)
    xy = df[["x5070", "y5070"]].to_numpy()
    z = df["innovation_m"].to_numpy(float)
    fold = df["cv_fold"].to_numpy()

    # ---- variograms: per-fold (on that fold's context pool) + pooled ----
    models = {"per_fold": {}}
    for f in np.sort(np.unique(fold)):
        m = fold != f
        d_km, cross = collect_pairs(xy[m], z[m], args.seed + int(f))
        var = float(z[m].var())
        cg = binned_correlogram(d_km, cross, var)
        fits = fit_correlation_models(cg, var)
        models["per_fold"][int(f)] = {
            **fits,
            "variance_m2": round(var, 3),
            "correlogram": cg,
        }
    d_km, cross = collect_pairs(xy, z, args.seed)
    var_all = float(z.var())
    cg_all = binned_correlogram(d_km, cross, var_all)
    models["pooled_all_wells"] = {
        **fit_correlation_models(cg_all, var_all),
        "variance_m2": round(var_all, 3),
        "mean_m": round(float(z.mean()), 4),
        "correlogram": cg_all,
        "anisotropy_rho_le25km": anisotropy_diagnostic(xy, z, args.seed),
    }
    per_huc2 = {}
    for h2, g in df.groupby("huc2"):
        if len(g) >= 500:
            gxy = g[["x5070", "y5070"]].to_numpy()
            gz = g["innovation_m"].to_numpy(float)
            dk, cr = collect_pairs(gxy, gz, args.seed)
            v = float(gz.var())
            per_huc2[str(h2)] = {
                **fit_correlation_models(binned_correlogram(dk, cr, v), v),
                "variance_m2": round(v, 3),
                "n_wells": int(len(g)),
            }
    models["per_huc2_diagnostic"] = per_huc2
    fold_models = models["per_fold"]

    # ---- predictors under the mask contract ----
    report = {"models": {}, "buffer_sweep": {}}
    preds = {}

    for k, p in IDW_GRID:
        name = f"idw_k{k}_p{int(p)}"
        pred, dist_used = idw_loo(df, k, p, buffer_km=0.0)
        preds[name] = (pred, dist_used)
        log.info("done %s", name)
    best_idw = min(
        preds,
        key=lambda nm: float(
            np.median(np.abs(z - preds[nm][0])[np.isfinite(preds[nm][0])])
        ),
    )
    k_b, p_b = [g for g in IDW_GRID if f"idw_k{g[0]}_p{int(g[1])}" == best_idw][0]
    pred, dist_used = idw_loo(df, k_b, p_b, buffer_km=0.0, vw=args.relief_vw)
    preds[f"{best_idw}_relief"] = (pred, dist_used)
    log.info("done %s_relief", best_idw)

    for model_name in ("exponential", "spherical"):
        pred, sig, dist_used = krige_loo(df, fold_models, 0.0, model_name=model_name)
        preds[f"krige_{model_name}"] = (pred, dist_used)
        if model_name == "exponential":
            df["krige_sigma_m"] = sig
        log.info("done krige_%s", model_name)

    for name, (pred, dist_used) in preds.items():
        df[f"pred_{name}"] = pred
        df[f"dist_used_{name}"] = dist_used
        report["models"][name] = skill_panel(df, f"pred_{name}", f"dist_used_{name}")

    # winner by overall MAD skill (transparent models only, buffer 0)
    best_name = max(
        report["models"], key=lambda nm: report["models"][nm]["overall"]["mad_skill"]
    )
    report["best_model"] = {"name": best_name, **report["models"][best_name]}

    # ---- buffer sweep on the winner family ----
    for buf in BUFFERS_KM:
        if best_name.startswith("krige"):
            mn = best_name.split("_", 1)[1]
            pred, _, dist_used = krige_loo(df, fold_models, buf, model_name=mn)
        else:
            pred, dist_used = idw_loo(df, k_b, p_b, buffer_km=buf)
        tmp = df.copy()
        tmp["pred_buf"] = pred
        tmp["dist_buf"] = dist_used
        report["buffer_sweep"][f"{buf:g}km"] = skill_panel(tmp, "pred_buf", "dist_buf")
        log.info("buffer sweep %skm done", buf)

    # ---- kriging variance calibration ----
    kp = df["pred_krige_exponential"].to_numpy(float)
    ks = df["krige_sigma_m"].to_numpy(float)
    resid = z - kp
    ok = np.isfinite(resid) & np.isfinite(ks) & (ks > 0)
    zscore = resid[ok] / ks[ok]
    report["kriging_calibration"] = {
        "n": int(ok.sum()),
        "frac_within_1sigma": round(float(np.mean(np.abs(zscore) <= 1)), 4),
        "gaussian_expected_1sigma": 0.6827,
        "frac_within_2sigma": round(float(np.mean(np.abs(zscore) <= 2)), 4),
        "gaussian_expected_2sigma": 0.9545,
        "sigma_monotonicity": sigma_monotonicity(kp[ok], ks[ok], z[ok]),
    }

    # ---- empty-context identity (runtime check, WP1 gate) ----
    solo = df.iloc[:1].copy()
    solo_pred, _ = idw_loo(
        pd.concat([solo], ignore_index=True).assign(cv_fold=0, site_id=0),
        8,
        2.0,
        0.0,
    )
    report["empty_context_identity"] = {
        "increment_m": float(solo_pred[0]),
        "passes": bool(abs(solo_pred[0]) < 1e-9),
    }

    report["population"] = {
        "n_wells": int(len(df)),
        "innovation_variance_m2": round(var_all, 3),
        "innovation_mean_m": round(float(z.mean()), 4),
        "innovation_mad_m": round(float(np.median(np.abs(z))), 4),
        "background_arm": "gnn_conus_monitoring_water_gate_mirror_sigma_w25_prod",
    }

    (out / "variogram_models.json").write_text(json.dumps(models, indent=2))
    (out / "recoverability_report.json").write_text(json.dumps(report, indent=2))
    keep = [
        "canonical_id",
        "x5070",
        "y5070",
        "cv_fold",
        "site_id",
        "huc2",
        "huc4",
        "is_nwis",
        "mean_dtw",
        "obs_wte_m",
        "gnn_wte_hat_m",
        "innovation_m",
        "krige_sigma_m",
        "regional_holdout",
    ] + [c for c in df.columns if c.startswith(("pred_", "dist_used_"))]
    df[[c for c in keep if c in df.columns]].to_parquet(out / "innovations.parquet")
    write_summary_md(out / "recoverability_summary.md", report, models)
    log.info("WP1 recoverability artifacts written to %s", out)


if __name__ == "__main__":
    main()
