"""Dupuit hanging-surface WTE prior -- standalone diagnostic.

Hangs a water-table ELEVATION surface on discharge boundaries (top-2 Strahler
reaches per FAC basin) and lets wells fit the sag/bulge SHAPE between them.
For an unconfined aquifer between discharge boundaries, Dupuit gives h^2
interpolated between boundary heads plus a quadratic term in distance whose
coefficient is N/K -- the one identifiable regime parameter. We never compute
N or K from maps (the Zell & Sanford NO-GO: map-derived N/K is wrong in exactly
the deep mountain-west basins); wells measure the shape directly.

Variants, all sharing the same boundary interpolation h_interp (kNN-IDW of
boundary reach elevations) and distance-to-boundary d:

  v0  gamma = delta = 0: the pure hanging surface (str_top2-style control).
  v1  h-space:  wte = h_interp + gamma*d + delta*d^2   (shape fitted per HUC8)
  v2  h^2-space with effective base zb: (wte-zb)^2 = (h_interp-zb)^2
      + gamma*d + delta*d^2 ; zb = (train-fold floor of heads) - b0 with b0
      chosen per HUC8 from a small grid on the train folds.

Shape parameters are fitted per HUC8 (fallback HUC4-pooled, then gamma=delta=0)
with trimmed OLS, LEAVE-FOLD-OUT cross-fit on the bundle's cv_fold -- the same
honesty contract as the relief-IDW R. Judged as a PRIOR: it does not need to
fit everywhere, it needs to beat the incumbent priors somewhere structural
(deep bands / western HUC2s) without being garbage elsewhere.

    uv run python utils/build_dupuit_wte.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

WTE_DIR = "/data/ssd2/handily/conus/wte_gnn"
BUNDLE = f"{WTE_DIR}/graph_conus_monitoring_dd"
GEOM = f"{WTE_DIR}/fac_flowline_geom_conus.parquet"
OUT_DIR = f"{WTE_DIR}/dupuit_prior"

DEPTH_BANDS = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
B0_GRID = (25.0, 100.0, 400.0)  # effective-base offsets (m) tried per HUC8 (v2)
MIN_FIT_WELLS = 20


def build_boundaries(reach_nodes: str, geom: str, top_orders: int) -> pd.DataFrame:
    rn = pd.read_parquet(
        reach_nodes, columns=["comid", "basin", "streamorde", "reach_elev_m"]
    )
    g = pd.read_parquet(geom, columns=["comid", "cx", "cy"])
    if rn.comid.duplicated().any():
        raise SystemExit("reach_nodes comid not unique -- cannot join geometry")
    rn = rn.merge(g, on="comid", how="left", validate="1:1")
    basin_max = rn.groupby("basin")["streamorde"].transform("max")
    b = rn[rn.streamorde >= basin_max - (top_orders - 1)]
    b = b[np.isfinite(b.cx) & np.isfinite(b.cy) & np.isfinite(b.reach_elev_m)]
    log.info(
        "boundary set: %d of %d reaches (top-%d orders/basin, %d basins)",
        len(b),
        len(rn),
        top_orders,
        b.basin.nunique(),
    )
    return b.reset_index(drop=True)


def hang_interp(
    bxy: np.ndarray, bhead: np.ndarray, qxy: np.ndarray, k: int, power: float
) -> tuple[np.ndarray, np.ndarray]:
    """kNN-IDW of boundary heads at query points + distance to nearest boundary."""
    tree = cKDTree(bxy)
    kk = min(k, len(bxy))
    dist, idx = tree.query(qxy, k=kk)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / np.maximum(dist, 1.0) ** power
    h = (w * bhead[idx]).sum(1) / w.sum(1)
    return h, dist[:, 0]


def trimmed_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """2-pass trimmed OLS for y ~ [d, d^2] (no intercept: shape is 0 at d=0)."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    r = y - x @ beta
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r) <= 3.0 * max(s, 1e-6)
    if keep.sum() >= max(MIN_FIT_WELLS // 2, 5) and keep.sum() < len(y):
        beta, *_ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
    return beta


def _fit_groups(train: pd.DataFrame, col_y: str) -> dict[str, np.ndarray]:
    """Per-HUC8 trimmed-OLS shape params with HUC4 fallback; key -> beta[2]."""
    params: dict[str, np.ndarray] = {}
    for level in ("huc8", "huc4"):
        for key, grp in train.groupby(level):
            if len(grp) < MIN_FIT_WELLS or f"{level}:{key}" in params:
                continue
            x = np.c_[grp["d_m"].to_numpy(), grp["d_m"].to_numpy() ** 2]
            params[f"{level}:{key}"] = trimmed_ols(x, grp[col_y].to_numpy())
    return params


def _lookup(params: dict[str, np.ndarray], huc8: str, huc4: str) -> np.ndarray:
    for key in (f"huc8:{huc8}", f"huc4:{huc4}"):
        if key in params:
            return params[key]
    return np.zeros(2)


def crossfit_v1(wells: pd.DataFrame) -> np.ndarray:
    out = np.full(len(wells), np.nan)
    for f in sorted(wells.cv_fold.unique()):
        tr = wells[wells.cv_fold != f].copy()
        tr["y"] = tr.wte_obs_m - tr.h_interp
        params = _fit_groups(tr, "y")
        te = wells.cv_fold == f
        sub = wells[te]
        beta = np.stack([_lookup(params, h8, h4) for h8, h4 in zip(sub.huc8, sub.huc4)])
        d = sub.d_m.to_numpy()
        out[te.to_numpy()] = (
            sub.h_interp.to_numpy() + beta[:, 0] * d + beta[:, 1] * d**2
        )
    return out


def crossfit_v2(wells: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """h^2-space with per-HUC8 effective base zb = train head floor - b0 (grid)."""
    out = np.full(len(wells), np.nan)
    b0_pick: dict[str, float] = {}
    for f in sorted(wells.cv_fold.unique()):
        tr = wells[wells.cv_fold != f]
        te_mask = (wells.cv_fold == f).to_numpy()
        sub = wells[te_mask]
        pred_te = np.full(len(sub), np.nan)
        # zb floor per huc8 from TRAIN wells only (min of obs & interp heads)
        floor8 = (
            pd.concat(
                [tr.groupby("huc8").wte_obs_m.min(), tr.groupby("huc8").h_interp.min()],
                axis=1,
            )
            .min(axis=1)
            .to_dict()
        )
        floor4 = (
            pd.concat(
                [tr.groupby("huc4").wte_obs_m.min(), tr.groupby("huc4").h_interp.min()],
                axis=1,
            )
            .min(axis=1)
            .to_dict()
        )
        best_rmse: dict[str, float] = {}
        best: dict[str, tuple[float, dict]] = {}
        for b0 in B0_GRID:
            t = tr.copy()
            zb8 = t.huc8.map(floor8).fillna(t.huc4.map(floor4)) - b0
            t["zb"] = zb8
            t["y"] = (t.wte_obs_m - t.zb) ** 2 - (t.h_interp - t.zb) ** 2
            params = _fit_groups(t, "y")
            # train-side RMSE in DTW space per huc8 to pick b0
            beta = np.stack([_lookup(params, h8, h4) for h8, h4 in zip(t.huc8, t.huc4)])
            d = t.d_m.to_numpy()
            s = (
                (t.h_interp.to_numpy() - t.zb.to_numpy()) ** 2
                + beta[:, 0] * d
                + beta[:, 1] * d**2
            )
            wte_hat = t.zb.to_numpy() + np.sqrt(np.maximum(s, 0.0))
            err = (t.z_surf_well_m.to_numpy() - wte_hat) - t.mean_dtw.to_numpy()
            t2 = t.assign(err2=err**2)
            for h8, grp in t2.groupby("huc8"):
                rmse = float(np.sqrt(grp.err2.mean()))
                if h8 not in best_rmse or rmse < best_rmse[h8]:
                    best_rmse[h8] = rmse
                    best[h8] = (b0, params)
        # predict test wells with their huc8's chosen (b0, params)
        for i, (h8, h4, hi, d, _zs) in enumerate(
            zip(sub.huc8, sub.huc4, sub.h_interp, sub.d_m, sub.z_surf_well_m)
        ):
            if h8 in best:
                b0, params = best[h8]
            elif best_rmse:
                b0, params = best[min(best_rmse, key=best_rmse.get)]
            else:
                b0, params = B0_GRID[1], {}
            zb = floor8.get(h8, floor4.get(h4, hi)) - b0
            beta = _lookup(params, h8, h4)
            s = (hi - zb) ** 2 + beta[0] * d + beta[1] * d**2
            pred_te[i] = zb + np.sqrt(max(s, 0.0))
            b0_pick[h8] = b0
        out[te_mask] = pred_te
    return out, {"b0_by_huc8_last_fold": b0_pick}


def band_mad(obs: np.ndarray, err: np.ndarray) -> dict:
    o = {}
    for lo, hi in DEPTH_BANDS:
        m = (obs >= lo) & (obs < hi)
        tag = f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"
        o[tag] = round(float(np.median(np.abs(err[m]))), 2) if m.sum() else None
    return o


def panel(wells: pd.DataFrame, preds: dict[str, np.ndarray]) -> dict:
    obs = wells.mean_dtw.to_numpy()
    out = {}
    for name, dtw_hat in preds.items():
        err = dtw_hat - obs
        by_huc2 = {}
        for h2 in sorted(wells.huc2.unique()):
            m = (wells.huc2 == h2).to_numpy()
            if m.sum() < 100:
                continue
            deep = m & (obs >= 30)
            by_huc2[h2] = {
                "mad": round(float(np.nanmedian(np.abs(err[m]))), 2),
                "deep_mad": round(float(np.nanmedian(np.abs(err[deep]))), 2)
                if deep.sum() >= 30
                else None,
                "deep_medR": round(float(np.nanmedian(err[deep])), 2)
                if deep.sum() >= 30
                else None,
            }
        out[name] = {
            "n": int(np.isfinite(err).sum()),
            "mad": round(float(np.nanmedian(np.abs(err))), 2),
            "medR": round(float(np.nanmedian(err)), 2),
            "bias": round(float(np.nanmean(err)), 2),
            "rmse": round(float(np.sqrt(np.nanmean(err**2))), 2),
            "by_depth": band_mad(obs, err),
            "by_huc2": by_huc2,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--geom", default=GEOM)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--top-orders", type=int, default=2)
    ap.add_argument("--idw-k", type=int, default=8)
    ap.add_argument("--idw-power", type=float, default=2.0)
    args = ap.parse_args()

    wells = pd.read_parquet(
        f"{args.bundle}/query_nodes.parquet",
        columns=[
            "canonical_id",
            "source",
            "is_nwis",
            "x5070",
            "y5070",
            "huc8",
            "huc4",
            "huc2",
            "cv_fold",
            "mean_dtw",
            "wte_obs_m",
            "z_surf_well_m",
            "regional_idw_dtw_oof_m",
            "deep_regional_wte_idw_oof_m",
            "fac_rem_dtw_m",
        ],
    )
    bnd = build_boundaries(
        f"{args.bundle}/reach_nodes.parquet", args.geom, args.top_orders
    )
    h_interp, d = hang_interp(
        bnd[["cx", "cy"]].to_numpy("float64"),
        bnd.reach_elev_m.to_numpy("float64"),
        wells[["x5070", "y5070"]].to_numpy("float64"),
        args.idw_k,
        args.idw_power,
    )
    wells["h_interp"] = h_interp
    wells["d_m"] = d
    log.info(
        "hang interp: d_m median %.0f p90 %.0f; (z_surf - h_interp) median %.1f m",
        np.median(d),
        np.percentile(d, 90),
        float(np.median(wells.z_surf_well_m - h_interp)),
    )

    z = wells.z_surf_well_m.to_numpy()
    v1 = crossfit_v1(wells)
    v2, v2_meta = crossfit_v2(wells)
    preds = {
        "dupuit_v0_hang": np.maximum(z - h_interp, 0.0),
        "dupuit_v1_h": np.maximum(z - v1, 0.0),
        "dupuit_v2_h2": np.maximum(z - v2, 0.0),
        "relief_idw_R": np.maximum(wells.regional_idw_dtw_oof_m.to_numpy(), 0.0),
        "deep_idw": np.maximum(z - wells.deep_regional_wte_idw_oof_m.to_numpy(), 0.0),
        "fac_rem": np.maximum(wells.fac_rem_dtw_m.to_numpy(), 0.0),
    }
    result = panel(wells, preds)

    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    per_well = wells[
        [
            "canonical_id",
            "source",
            "is_nwis",
            "huc8",
            "huc4",
            "huc2",
            "cv_fold",
            "mean_dtw",
            "h_interp",
            "d_m",
        ]
    ].copy()
    for name, dtw_hat in preds.items():
        if name.startswith("dupuit"):
            per_well[name] = dtw_hat
    per_well.to_parquet(outd / "dupuit_prior_wells.parquet", index=False)
    meta = {
        "bundle": args.bundle,
        "top_orders": args.top_orders,
        "idw_k": args.idw_k,
        "idw_power": args.idw_power,
        "b0_grid": list(B0_GRID),
        "min_fit_wells": MIN_FIT_WELLS,
        "crossfit": "leave-fold-out on bundle cv_fold (HUC12-blocked); shape params "
        "per HUC8, HUC4 fallback, gamma=delta=0 when sparse",
        "panel": result,
        **v2_meta,
    }
    (outd / "dupuit_prior_panel.json").write_text(json.dumps(meta, indent=1))
    hdr = f"{'predictor':<16}{'MAD':>7}{'medR':>7}{'bias':>7}{'RMSE':>8}   0-2/2-5/5-10/10-30/30+"
    log.info(hdr)
    for name, row in result.items():
        bands = "/".join(str(v) for v in row["by_depth"].values())
        log.info(
            "%-16s%7.2f%7.2f%7.2f%8.2f   %s",
            name,
            row["mad"],
            row["medR"],
            row["bias"],
            row["rmse"],
            bands,
        )
    log.info("wrote %s", outd / "dupuit_prior_panel.json")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
