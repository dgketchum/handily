"""Step 5 of the CONUS WTE/DTW GNN: score GNN vs benchmarks with the full panel.

Consumes the OOF predictions from ``train_conus_gnn.py`` and reports the metric
panel CLAUDE.md mandates (never MAD alone): central error, bias AND median
residual side by side, RMSE, depth-banded and region-banded breakdowns, and
shallow-class precision/recall -- all on a common footprint with per-predictor
coverage.

Headline metrics are always DTW (``gnn_dtw_m``/``tab_dtw_m``), so both target modes
score on the identical panel. In dtw_residual mode the GNN DTW is the regional IDW
prior + GNN residual; in WTE (head-target) mode it is reconstructed as
``z_surf_well_m - gnn_wte_hat_m``. WTE runs additionally get an identity QA check
(``abs(wte_err) == abs(dtw_err)``) and a non-headline head-space core-metrics block.

Predictors compared:
  * gnn       -- GNN DTW under test (regional prior + residual, or z_surf - wte_hat)
  * regional  -- leave-one-HUC4-out IDW(kNN) of neighbor well DTW (the floor the
                 GNN must beat to justify the graph)
  * janssen   -- CONUS-wide modeled WTD benchmark (the bar to beat; NWIS-trained,
                 so NWIS wells are split out of the headline)
  * hand_cal  -- leak-free per-fold isotonic HAND->DTW calibration (does the graph
                 add anything over calibrating raw HAND directly?)
  * ma        -- per-state Ma WTD rasters auto-discovered from /nas/gwx/wtd_states
                 (the canonical benchmark -- the arid regime HAND priors fail in);
                 a regional coverage hole over the modeled footprint is a hard error
                 (see assert_ma_covers_footprint), not a silent 0%

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

# Head-space target modes: DTW is reconstructed as z_surf - wte_hat, so they get
# the WTE identity QA check + the head-space core-metrics block. wte_residual
# predicts a head residual over R but reconstructs the same way (wte_hat = R + resid).
HEAD_SPACE_MODES = ("wte", "wte_residual")

DEPTH_BANDS = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
SHALLOW_THRESHOLDS = [2.0, 5.0, 10.0]
MA_DIR = Path("/nas/gwx/wtd_states")


def discover_ma_specs(ma_dir: Path = MA_DIR) -> list[str]:
    """All per-state Ma WTD rasters present on disk, as LABEL=path specs.

    Globs ``wtd_<state>.tif`` so any synced state is scored automatically; a
    footprint over an un-synced state then trips assert_ma_covers_footprint rather
    than silently reading 0% Ma there (the bug that hid NV before wtd_nevada.tif was
    wired). All CONUS states are available to sync to ``ma_dir``.
    """
    return [f"Ma_{p.stem[len('wtd_') :]}={p}" for p in sorted(ma_dir.glob("wtd_*.tif"))]


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
    """Per observed-depth-band error STRUCTURE -- not MAD alone.

    The 30+ m regime is a high-variance catastrophic tail that MAD suppresses; the
    handoff acceptance turns on it, so each band reports RMSE, the p90/p95 absolute-
    error percentiles, and the catastrophic-miss fractions (>10 m, >25 m) alongside
    the central MAD/median/bias. Exposes the shallow-good / deep-saturated split AND
    the deep tail's spread in the same panel.
    """
    out = {}
    for lo, hi in DEPTH_BANDS:
        sel = np.isfinite(obs) & np.isfinite(pred) & (obs >= lo) & (obs < hi)
        label = f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"
        if sel.sum() == 0:
            out[label] = {"n": 0}
            continue
        r = pred[sel] - obs[sel]
        a = np.abs(r)
        out[label] = {
            "n": int(sel.sum()),
            "mad_m": float(np.median(a)),
            "median_resid_m": float(np.median(r)),
            "bias_mean_m": float(np.mean(r)),
            "rmse_m": float(np.sqrt(np.mean(r**2))),
            "p90_abs_err_m": float(np.percentile(a, 90)),
            "p95_abs_err_m": float(np.percentile(a, 95)),
            "frac_abs_err_gt_10m": float(np.mean(a > 10.0)),
            "frac_abs_err_gt_25m": float(np.mean(a > 25.0)),
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


def gate_diagnostics(df: pd.DataFrame, gate_col: str = "aquifer_gate") -> dict:
    """Aquifer-router gate behaviour vs depth/region/error -- the fail-flat monitor.

    A learned aquifer router that stays FLAT by depth band is the same dead-gate
    failure mode as the old FAC gate: it never localizes the deep-regional regime it
    was added for. Reports mean/median gate by observed-DTW band, by predicted-DTW
    band, by HUC2, and the gate-vs-absolute-error correlation. NaN gates (no aquifer
    edge / route off) are dropped per band. Returns ``{}`` if the column is absent.
    """
    if gate_col not in df.columns:
        return {}
    g = df[gate_col].to_numpy("float64")
    obs = df["obs_dtw_m"].to_numpy("float64")
    pred = df["gnn_dtw_m"].to_numpy("float64")
    fin = np.isfinite(g)

    def _by_band(band_vals: np.ndarray) -> dict:
        out = {}
        for lo, hi in DEPTH_BANDS:
            sel = fin & np.isfinite(band_vals) & (band_vals >= lo) & (band_vals < hi)
            label = f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"
            if sel.sum() == 0:
                out[label] = {"n": 0}
                continue
            out[label] = {
                "n": int(sel.sum()),
                "mean_gate": float(np.mean(g[sel])),
                "median_gate": float(np.median(g[sel])),
            }
        return out

    by_huc2 = {}
    if "huc2" in df.columns:
        for h2, sub in df.groupby("huc2"):
            gg = sub[gate_col].to_numpy("float64")
            m = np.isfinite(gg)
            if m.sum() < 25:
                continue
            by_huc2[str(h2)] = {"n": int(m.sum()), "mean_gate": float(np.mean(gg[m]))}

    abs_err = np.abs(pred - obs)
    cm = fin & np.isfinite(abs_err)
    corr = (
        float(np.corrcoef(g[cm], abs_err[cm])[0, 1])
        if cm.sum() >= 25 and np.std(g[cm]) > 0
        else float("nan")
    )
    return {
        "n_finite_gate": int(fin.sum()),
        "mean_gate": float(np.mean(g[fin])) if fin.any() else float("nan"),
        "by_obs_depth": _by_band(obs),
        "by_pred_depth": _by_band(pred),
        "by_huc2": by_huc2,
        "gate_vs_abs_err_corr": corr,
    }


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


def assert_ma_covers_footprint(
    df: pd.DataFrame, min_wells: int = 50, min_cov: float = 0.5
) -> None:
    """Fail loudly when the Ma benchmark has a regional hole over the modeled area.

    Ma is a per-state raster panel, so a missing state raster shows up as an entire
    HUC2 with ~0% Ma coverage -- exactly how NV read 0% before wtd_nevada.tif was
    wired. Scoring against a benchmark that is absent over part of the footprint
    understates its error and quietly drops those wells from the comparison, so a
    regional gap is a hard error, not a silent 0%. Scattered within-state nodata
    (lakes) stays well above ``min_cov``; only well-populated HUC2s (>= ``min_wells``)
    are gated so a handful of edge wells in a corner of the footprint can't trip it.
    """
    ma = np.isfinite(df["ma"].to_numpy("float64"))
    log.info(
        "Ma footprint coverage: %d/%d (%.1f%%)", int(ma.sum()), len(ma), 100 * ma.mean()
    )
    gaps = []
    for h2, g in df.groupby("huc2"):
        n = len(g)
        if n < min_wells:
            continue
        cov = float(np.isfinite(g["ma"].to_numpy("float64")).mean())
        log.info("  HUC2 %s Ma coverage: %.1f%% (n=%d)", h2, 100 * cov, n)
        if cov < min_cov:
            gaps.append((h2, n, cov))
    if gaps:
        detail = ", ".join(
            f"HUC2 {h2}: {100 * cov:.0f}% (n={n})" for h2, n, cov in gaps
        )
        raise SystemExit(
            f"Ma benchmark coverage gap over the modeled footprint -- {detail}. "
            "The Ma comparison would silently omit these regions. Sync the missing "
            f"per-state WTD raster(s) to {MA_DIR} (all CONUS states are available on "
            "the other machine), or pass --no-ma to score without the Ma benchmark."
        )


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
    # Deep-tail RMSE/p95 for the model and the bar: acceptance turns on the 30+ band
    # spread (MAD hides the catastrophic tail), so log it explicitly side by side.
    for c in [x for x in ("gnn", "janssen", "ma") if x in predcols]:
        bands = panel["predictors"][c]["by_depth_band"]
        cells = " ".join(
            f"{b}:RMSE{v['rmse_m']:.1f}/p95 {v['p95_abs_err_m']:.1f}(n{v['n']})"
            for b, v in bands.items()
            if v.get("n")
        )
        log.info("  %s deep-tail RMSE/p95 by obs-depth: %s", c, cells)


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
        help="LABEL=path per-state Ma raster (repeatable); "
        "default: all wtd_<state>.tif under /nas/gwx/wtd_states",
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

    # Target mode (from the trainer run.json) decides only WTE-specific QA; the
    # headline panel stays DTW-based in both modes via gnn_dtw_m/tab_dtw_m.
    target_mode = None
    run_json = gdir / "gnn_run.json"
    if run_json.exists():
        target_mode = json.loads(run_json.read_text()).get("target_mode")
    wte_identity = None
    if target_mode in HEAD_SPACE_MODES:
        missing = [
            c
            for c in ("z_surf_well_m", "obs_wte_m", "gnn_wte_hat_m", "gnn_dtw_m")
            if c not in df.columns
        ]
        if missing:
            raise SystemExit(f"WTE-mode predictions missing columns: {missing}")
        # The loss identity: z_surf is exact and shared, so |wte_hat - wte_obs|
        # must equal |dtw_hat - dtw_obs| at every well. A non-trivial max delta
        # means the DTW reconstruction (z_surf - wte_hat) is broken.
        wte_err = np.abs(
            df["gnn_wte_hat_m"].to_numpy("float64")
            - df["obs_wte_m"].to_numpy("float64")
        )
        dtw_err = np.abs(
            df["gnn_dtw_m"].to_numpy("float64") - df["obs_dtw_m"].to_numpy("float64")
        )
        max_delta = float(np.nanmax(np.abs(wte_err - dtw_err)))
        log.info(
            "WTE identity check: max |abs(wte_err) - abs(dtw_err)| = %.3e m", max_delta
        )
        wte_identity = {"max_abs_error_delta_m": max_delta}

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

    # FAC-REM standalone: the shallow terrain DTW prior ON ITS OWN, not just embedded
    # as a model feature. CLAUDE.md mandates comparing FAC directly to the benchmarks
    # on the same footprint -- otherwise a model's shallow skill is attributed to FAC
    # without ever showing FAC alone. Reconstruct depth = z_surf - fac_rem_wte (the
    # head-space modes carry both). Kept out of `common` so partial FAC coverage never
    # shrinks the shared footprint; here (FAC-footprint run) it is finite everywhere.
    if {"fac_rem_wte_m", "z_surf_well_m"}.issubset(df.columns):
        df["fac_rem"] = df["z_surf_well_m"] - df["fac_rem_wte_m"]
        predcols.append("fac_rem")

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

    ma_specs = [] if args.no_ma else (args.ma if args.ma else discover_ma_specs())
    if ma_specs:
        df["ma"] = sample_ma(df, ma_specs)
        assert_ma_covers_footprint(df)
        predcols.append("ma")

    # Aquifer-router gate diagnostics (present only when the trainer ran a learned
    # aquifer route); computed on non-NWIS wells, the headline population.
    gate_diag = None

    non_nwis = df[~df["is_nwis"]].reset_index(drop=True)
    nwis = df[df["is_nwis"]].reset_index(drop=True)

    headline = full_panel(non_nwis, predcols, "obs_dtw_m", common)
    log_panel("HEADLINE -- non-NWIS wells", headline, predcols)

    if "aquifer_gate" in df.columns:
        gate_diag = gate_diagnostics(non_nwis, "aquifer_gate")
        bands = gate_diag.get("by_obs_depth", {})
        cells = " ".join(
            f"{b}:{v['mean_gate']:.3f}(n{v['n']})"
            for b, v in bands.items()
            if v.get("n")
        )
        log.info("aquifer gate mean OOF by obs-depth: %s", cells)
        log.info(
            "aquifer gate vs |err| corr=%.3f (flat-by-depth gate == dead router)",
            gate_diag.get("gate_vs_abs_err_corr", float("nan")),
        )
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
            log_panel("Ma-covered footprint, non-NWIS", ma_panel, predcols)
        else:
            log.info("Ma coverage < 50 non-NWIS wells; skipping Ma sub-panel")

    # NV closed-basin sub-panel (HUC2=16): the regime where "downstream" never reaches a
    # high-order datum (Phase-6 ho_any hypothesis). huc2 is zero-filled to 2 digits above,
    # so the literal is "16". Same >=50-well guard + all-predictor compare as the Ma panel.
    nv_panel = None
    if "huc2" in non_nwis.columns:
        nv = non_nwis[non_nwis["huc2"] == "16"].reset_index(drop=True)
        if len(nv) >= 50:
            nv_panel = full_panel(nv, predcols, "obs_dtw_m", common)
            log_panel("NV closed basins (HUC2=16), non-NWIS", nv_panel, predcols)
        else:
            log.info("NV (HUC2=16) < 50 non-NWIS wells; skipping NV sub-panel")

    # Non-headline head-space diagnostics (WTE mode only): same metrics, but in
    # WTE units against obs_wte_m. By the identity above the GNN MAD equals its DTW
    # MAD; the value is comparing the head priors to each other in their own space.
    wte_core = None
    if target_mode in HEAD_SPACE_MODES:
        obs_wte = non_nwis["obs_wte_m"].to_numpy("float64")
        wte_core = {
            c: core_metrics(non_nwis[c].to_numpy("float64"), obs_wte)
            for c in (
                "gnn_wte_hat_m",
                "regional_wte_idw_oof_m",
                "deep_regional_wte_idw_oof_m",
                "hand_wte_m",
                "fac_rem_wte_m",
            )
            if c in non_nwis.columns
        }

    summary = {
        "gnn_dir": str(gdir),
        "target_mode": target_mode,
        "predictors": predcols,
        "common_footprint_predictors": common,
        "headline_non_nwis": headline,
        "nwis_panel": nwis_panel,
        "ma_covered_panel": ma_panel,
        "nv_closed_basin_panel": nv_panel,
        "wte_identity_check": wte_identity,
        "wte_core_metrics": wte_core,
        "diagnostics": {
            "aquifer_gate_by_depth": gate_diag.get("by_obs_depth")
            if gate_diag
            else None,
            "aquifer_gate_by_pred_depth": gate_diag.get("by_pred_depth")
            if gate_diag
            else None,
            "aquifer_gate_by_huc2": gate_diag.get("by_huc2") if gate_diag else None,
            "aquifer_gate_vs_abs_err_corr": gate_diag.get("gate_vs_abs_err_corr")
            if gate_diag
            else None,
        },
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
        + (["fac_rem"] if "fac_rem" in predcols else [])
        + (["ma"] if "ma" in predcols else [])
        + (["aquifer_gate"] if "aquifer_gate" in df.columns else [])
    )
    res = df[keep].copy()
    for c in predcols:
        res[f"resid_{c}"] = res[c] - res["obs_dtw_m"]
    # Head-space WTE ELEVATION surface (head-space modes only): the GNN's native
    # output is the predicted water-table elevation, dtw = z_surf - wte_m. Carry it
    # (+ observed WTE + land surface) so the predicted water-table surface can be
    # rendered/compared directly in QGIS, not only as a DTW depth.
    if "gnn_wte_hat_m" in df.columns:
        res["wte_m"] = df["gnn_wte_hat_m"].to_numpy("float64")
        if "obs_wte_m" in df.columns:
            res["obs_wte_m"] = df["obs_wte_m"].to_numpy("float64")
        if "z_surf_well_m" in df.columns:
            res["z_surf_m"] = df["z_surf_well_m"].to_numpy("float64")
        # Head residual over the regional base R (= what the GNN targets/predicts):
        # wte_resid_obs = obs_wte - R is the residual field the model must learn; the
        # predicted counterpart wte_resid_hat = wte_hat - R is what it produced. Carry R
        # too so both residuals are interpretable against the base surface in QGIS.
        if "regional_wte_idw_oof_m" in df.columns:
            r_wte = df["regional_wte_idw_oof_m"].to_numpy("float64")
            res["regional_wte_m"] = r_wte
            res["wte_resid_obs_m"] = df["obs_wte_m"].to_numpy("float64") - r_wte
            res["wte_resid_hat_m"] = df["gnn_wte_hat_m"].to_numpy("float64") - r_wte
    gres = gpd.GeoDataFrame(
        res, geometry=gpd.points_from_xy(df["x5070"], df["y5070"]), crs=5070
    )
    gres.to_file(out_dir / "conus_residuals.fgb", driver="FlatGeobuf")
    log.info("wrote conus_score_panel.json + conus_residuals.fgb -> %s", out_dir)


if __name__ == "__main__":
    main()
