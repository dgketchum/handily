"""Step 5 of the CONUS WTE/DTW GNN: score GNN vs benchmarks with the full panel.

Consumes the OOF predictions from ``train_conus_gnn.py`` and reports the metric
panel CLAUDE.md mandates (never MAD alone): central error, bias AND median
residual side by side, RMSE, depth-banded and region-banded breakdowns, and
shallow-class precision/recall -- all on a common footprint with per-predictor
coverage.

Predictors compared:
  * gnn       -- regional IDW prior + GNN residual (the model under test)
  * regional  -- leave-one-HUC4-out IDW(kNN) of neighbor well DTW (the floor the
                 GNN must beat to justify the graph)
  * janssen   -- CONUS-wide modeled WTD benchmark (the bar to beat; NWIS-trained,
                 so NWIS wells are split out of the headline)
  * hand_cal  -- leak-free per-fold isotonic HAND->DTW calibration (does the graph
                 add anything over calibrating raw HAND directly?)
  * ma        -- OPTIONAL per-state Ma WTD rasters (MT/NM/TX only; the canonical
                 benchmark where it exists -- the arid regime HAND priors fail in)

Headline = non-NWIS wells; a separate NWIS panel exposes the benchmark's
leakage-inflated skill. Also writes conus_residuals.fgb for QGIS.

    uv run python utils/score_conus_gnn.py \\
        --gnn-dir /data/ssd2/handily/conus/wte_gnn/gnn
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("score_conus_gnn")

DEPTH_BANDS = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
SHALLOW_THRESHOLDS = [2.0, 5.0, 10.0]
DEFAULT_MA = [
    "Ma_MT=/nas/gwx/wtd_states/wtd_montana.tif",
    "Ma_NM=/nas/gwx/wtd_states/wtd_new_mexico.tif",
    "Ma_TX=/nas/gwx/wtd_states/wtd_texas.tif",
]


def core_metrics(pred: np.ndarray, obs: np.ndarray) -> dict:
    """Central error + the bias/median split + RMSE on finite-pair rows."""
    m = np.isfinite(pred) & np.isfinite(obs)
    if m.sum() == 0:
        return {"n": 0}
    r = pred[m] - obs[m]
    return {
        "n": int(m.sum()),
        "mad_m": float(np.median(np.abs(r))),
        "bias_mean_m": float(np.mean(r)),
        "median_resid_m": float(np.median(r)),
        "rmse_m": float(np.sqrt(np.mean(r**2))),
    }


def depth_banded(pred: np.ndarray, obs: np.ndarray) -> dict:
    """MAD per observed-depth band -- exposes the shallow-good / deep-saturated split."""
    out = {}
    for lo, hi in DEPTH_BANDS:
        sel = np.isfinite(obs) & np.isfinite(pred) & (obs >= lo) & (obs < hi)
        label = f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"
        if sel.sum() == 0:
            out[label] = {"n": 0}
            continue
        r = pred[sel] - obs[sel]
        out[label] = {
            "n": int(sel.sum()),
            "mad_m": float(np.median(np.abs(r))),
            "median_resid_m": float(np.median(r)),
        }
    return out


def shallow_skill(pred: np.ndarray, obs: np.ndarray) -> dict:
    """Precision/recall for the 'shallow water table' call at each threshold."""
    out = {}
    m = np.isfinite(pred) & np.isfinite(obs)
    p, o = pred[m], obs[m]
    for thr in SHALLOW_THRESHOLDS:
        pred_s, obs_s = p < thr, o < thr
        tp = int((pred_s & obs_s).sum())
        fp = int((pred_s & ~obs_s).sum())
        fn = int((~pred_s & obs_s).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        out[f"<{thr:g}m"] = {
            "precision": prec,
            "recall": rec,
            "n_obs_shallow": int(obs_s.sum()),
        }
    return out


def region_banded(df: pd.DataFrame, predcol: str, obscol: str) -> dict:
    """MAD per HUC2 region for the named predictor (spatial hotspot localization)."""
    out = {}
    for h2, g in df.groupby("huc2"):
        r = g[predcol].to_numpy() - g[obscol].to_numpy()
        m = np.isfinite(r)
        if m.sum() < 25:
            continue
        out[str(h2)] = {"n": int(m.sum()), "mad_m": float(np.median(np.abs(r[m])))}
    return out


def fit_hand_cal_oof(df: pd.DataFrame) -> np.ndarray:
    """Leak-free per-fold isotonic HAND->DTW calibration (the floor baseline).

    For each CV fold, fit a monotonic map on the *other* folds' (hand, dtw) pairs
    and predict the held-out fold -- so a well's own basin never calibrates its own
    prediction. Wells with no HAND stay NaN.
    """
    from sklearn.isotonic import IsotonicRegression

    hand = df["hand_m"].to_numpy("float64")
    obs = df["obs_dtw_m"].to_numpy("float64")
    fold = df["cv_fold"].to_numpy()
    pred = np.full(len(df), np.nan)
    for f in np.unique(fold):
        te = fold == f
        tr = (~te) & np.isfinite(hand) & np.isfinite(obs)
        ap = te & np.isfinite(hand)
        if tr.sum() < 50 or ap.sum() == 0:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(hand[tr], obs[tr])
        pred[ap] = iso.predict(hand[ap])
    return pred


def sample_ma(df: pd.DataFrame, specs: list[str]) -> np.ndarray:
    """Sample per-state Ma WTD rasters at well 5070 coords; first finite hit wins.

    Each raster covers one state; a well takes Ma from whichever raster has valid
    data there. Reprojects well coords to each raster's CRS. Returns NaN where no
    raster covers the well (most of CONUS -- Ma is a per-state panel, not headline).
    """
    import rasterio
    from pyproj import Transformer

    out = np.full(len(df), np.nan)
    x5070 = df["x5070"].to_numpy("float64")
    y5070 = df["y5070"].to_numpy("float64")
    for spec in specs:
        label, _, path = spec.partition("=")
        if not Path(path).exists():
            log.warning("Ma raster missing, skipping: %s", path)
            continue
        try:
            with rasterio.open(path) as ds:
                tr = Transformer.from_crs(5070, ds.crs, always_xy=True)
                rx, ry = tr.transform(x5070, y5070)
                left, bottom, right, top = ds.bounds
                cand = (rx >= left) & (rx <= right) & (ry >= bottom) & (ry <= top)
                cand &= ~np.isfinite(out)  # only fill wells not yet covered
                if cand.sum() == 0:
                    continue
                pts = list(zip(rx[cand], ry[cand]))
                vals = np.array([v[0] for v in ds.sample(pts)], dtype="float64")
                nd = ds.nodata
                if nd is not None:
                    vals[vals == nd] = np.nan
                vals[(vals < -1.0) | (vals > 1e4)] = np.nan
                idx = np.where(cand)[0]
                out[idx] = vals
        except Exception as e:  # noqa: BLE001 - raster availability is best-effort
            log.warning("Ma sample failed for %s: %r", path, e)
    log.info("Ma coverage: %d/%d wells", int(np.isfinite(out).sum()), len(out))
    return out


def full_panel(
    df: pd.DataFrame, predcols: list[str], obscol: str, common: list[str]
) -> dict:
    """Per-predictor core metrics (own + common footprint) + banded breakdowns.

    ``common`` predictors define the apples-to-apples footprint where every one of
    them is finite; central metrics are reported both on each predictor's own
    coverage and restricted to that common footprint.
    """
    obs = df[obscol].to_numpy("float64")
    cf = np.isfinite(obs)
    for c in common:
        cf &= np.isfinite(df[c].to_numpy("float64"))
    panel = {
        "n_wells": int(len(df)),
        "common_footprint_n": int(cf.sum()),
        "predictors": {},
    }
    for c in predcols:
        pred = df[c].to_numpy("float64")
        panel["predictors"][c] = {
            "coverage": float(np.isfinite(pred).mean()),
            "own_footprint": core_metrics(pred, obs),
            "common_footprint": core_metrics(pred[cf], obs[cf]),
            "by_depth_band": depth_banded(pred, obs),
            "shallow_skill": shallow_skill(pred, obs),
            "by_huc2": region_banded(df, c, obscol),
        }
    return panel


def log_panel(title: str, panel: dict, predcols: list[str]) -> None:
    log.info("")
    log.info(
        "=== %s (n=%d, common footprint n=%d) ===",
        title,
        panel["n_wells"],
        panel["common_footprint_n"],
    )
    hdr = f"{'predictor':<12} {'cov':>5} {'n(cf)':>7} {'MAD':>7} {'bias':>8} {'medR':>8} {'RMSE':>8}"
    log.info(hdr)
    for c in predcols:
        p = panel["predictors"][c]
        cf = p["common_footprint"]
        if cf.get("n", 0) == 0:
            log.info(
                "%-12s %5.0f%% %7s  (no common-footprint coverage)",
                c,
                100 * p["coverage"],
                "-",
            )
            continue
        log.info(
            "%-12s %5.0f%% %7d %7.2f %8.2f %8.2f %8.2f",
            c,
            100 * p["coverage"],
            cf["n"],
            cf["mad_m"],
            cf["bias_mean_m"],
            cf["median_resid_m"],
            cf["rmse_m"],
        )
    # depth bands for the model vs the controls/bar so the structure is visible
    for c in [
        x for x in ("gnn", "fusion", "regional_deep", "janssen", "ma") if x in predcols
    ]:
        bands = panel["predictors"][c]["by_depth_band"]
        cells = " ".join(
            f"{b}:{v['mad_m']:.1f}({v['n']})" for b, v in bands.items() if v.get("n")
        )
        log.info("  %s MAD by obs-depth: %s", c, cells)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gnn-dir", default="/data/ssd2/handily/conus/wte_gnn/gnn")
    ap.add_argument("--out-dir", default=None, help="defaults to --gnn-dir")
    ap.add_argument(
        "--tabular-dir",
        default=None,
        help="dir with tabular_oof_predictions.parquet (the fusion control); "
        "adds a 'fusion' predictor on the SAME common footprint",
    )
    ap.add_argument(
        "--ma",
        action="append",
        default=None,
        help="LABEL=path per-state Ma raster (repeatable); default MT/NM/TX",
    )
    ap.add_argument("--no-ma", action="store_true", help="skip the Ma per-state panel")
    args = ap.parse_args()
    gdir = Path(args.gnn_dir)
    out_dir = Path(args.out_dir) if args.out_dir else gdir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(gdir / "gnn_oof_predictions.parquet")
    df["huc2"] = df["huc2"].astype(str).str.zfill(2)
    log.info(
        "loaded %d OOF predictions (%d non-NWIS, %d NWIS)",
        len(df),
        int((~df["is_nwis"]).sum()),
        int(df["is_nwis"].sum()),
    )

    # standardize predictor columns
    df["gnn"] = df["gnn_dtw_m"]
    df["regional"] = df["regional_idw_dtw_oof_m"]
    df["janssen"] = df["janssen_dtw_m"]
    df["hand_cal"] = fit_hand_cal_oof(df)
    predcols = ["gnn", "regional", "janssen", "hand_cal"]
    # `common` stays the 4-core set so the headline footprint matches the prior
    # run; predictors below are finite everywhere `regional` is, so they are
    # scored on this same common footprint without shrinking it.
    common = ["gnn", "regional", "janssen", "hand_cal"]

    # Deep regional datum standalone skill (carried by the trainer when built).
    if "regional_deep_idw_dtw_oof_m" in df.columns:
        df["regional_deep"] = df["regional_deep_idw_dtw_oof_m"]
        predcols.append("regional_deep")

    # Tabular-fusion control: same features/folds, no message passing.
    if args.tabular_dir:
        tab = pd.read_parquet(
            Path(args.tabular_dir) / "tabular_oof_predictions.parquet"
        )
        df = df.merge(tab[["canonical_id", "tab_dtw_m"]], on="canonical_id", how="left")
        df["fusion"] = df["tab_dtw_m"]
        n_missing = int(df["fusion"].isna().sum())
        if n_missing:
            log.warning(
                "fusion: %d wells unmatched in tabular OOF (left-join NaN)", n_missing
            )
        predcols.append("fusion")

    ma_specs = [] if args.no_ma else (args.ma if args.ma else DEFAULT_MA)
    if ma_specs:
        df["ma"] = sample_ma(df, ma_specs)
        predcols.append("ma")

    non_nwis = df[~df["is_nwis"]].reset_index(drop=True)
    nwis = df[df["is_nwis"]].reset_index(drop=True)

    headline = full_panel(non_nwis, predcols, "obs_dtw_m", common)
    log_panel("HEADLINE -- non-NWIS wells", headline, predcols)
    nwis_panel = full_panel(nwis, predcols, "obs_dtw_m", common) if len(nwis) else None
    if nwis_panel:
        log_panel(
            "NWIS wells (benchmark leakage-prone; diagnostic only)",
            nwis_panel,
            predcols,
        )

    # Ma per-state sub-panel: only wells Ma actually covers, all predictors compared
    ma_panel = None
    if "ma" in predcols:
        ma_cov = non_nwis[np.isfinite(non_nwis["ma"].to_numpy("float64"))].reset_index(
            drop=True
        )
        if len(ma_cov) >= 50:
            ma_panel = full_panel(
                ma_cov, predcols, "obs_dtw_m", ["gnn", "regional", "janssen", "ma"]
            )
            log_panel("Ma-covered states (MT/NM/TX), non-NWIS", ma_panel, predcols)
        else:
            log.info("Ma coverage < 50 non-NWIS wells; skipping Ma sub-panel")

    summary = {
        "gnn_dir": str(gdir),
        "predictors": predcols,
        "common_footprint_predictors": common,
        "headline_non_nwis": headline,
        "nwis_panel": nwis_panel,
        "ma_covered_panel": ma_panel,
        "ma_specs": ma_specs,
        "metric_definitions": {
            "residual": "pred_dtw - obs_dtw (positive = predicted too deep)",
            "mad_m": "median(|residual|)",
            "bias_mean_m": "mean(residual)",
            "median_resid_m": "median(residual)",
            "rmse_m": "sqrt(mean(residual^2))",
            "common_footprint": "rows where obs + all common predictors finite",
        },
        "bar_to_beat": "beat janssen (CONUS bar) AND beat hand_cal (graph must earn its place)",
    }
    (out_dir / "conus_score_panel.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    # residuals fgb for QGIS
    keep = (
        [
            "canonical_id",
            "source",
            "is_nwis",
            "huc2",
            "cv_fold",
            "obs_dtw_m",
            "gnn",
            "regional",
            "janssen",
            "hand_cal",
            "hand_m",
        ]
        + (["regional_deep"] if "regional_deep" in predcols else [])
        + (["fusion"] if "fusion" in predcols else [])
        + (["ma"] if "ma" in predcols else [])
    )
    res = df[keep].copy()
    for c in predcols:
        res[f"resid_{c}"] = res[c] - res["obs_dtw_m"]
    gres = gpd.GeoDataFrame(
        res, geometry=gpd.points_from_xy(df["x5070"], df["y5070"]), crs=5070
    )
    gres.to_file(out_dir / "conus_residuals.fgb", driver="FlatGeobuf")
    log.info("wrote conus_score_panel.json + conus_residuals.fgb -> %s", out_dir)


if __name__ == "__main__":
    main()
