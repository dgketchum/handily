"""Reusable OOF-GBM ceiling gate for WTE-residual GNN topology features.

Formalizes the "does a candidate feature family earn a GNN run?" ceiling test we
have re-created three times in scratchpad scripts (HAF, geology stack, terrain
multiscale). Given a ``wte_residual`` graph bundle it trains a leak-free out-of-fold
``HistGradientBoostingRegressor`` on the residual target from (a) the baseline query
features in the manifest and (b) baseline + ``--candidate-cols``, reconstructs DTW
EXACTLY as ``train_conus_gnn`` does (``dtw_hat = dtw_base - resid_hat`` where
``dtw_base = z_surf - R``), and reports the full metric panel + per-band deltas +
candidate permutation importances + finite-fraction. It PASSES iff the candidate
clears one of the plan's three thresholds.

This is a CEILING test on the shared tabular manifold: a candidate that a GBM cannot
exploit still MIGHT help the GNN (message passing sees graph structure the GBM does
not), so a fail is "skip the expensive GNN run for now", not "the idea is dead". A
tabular WIN is the green light to spend a GNN run. Calibration from prior runs:
HAF = +0.067 R^2 (real signal, GNN confirmed it); geology stack <= +0.014 (dead).

    uv run python utils/gbm_gate_wte_residual.py \\
        --graph-dir /data/ssd2/handily/conus/wte_gnn/graph_relief_idw_facrem \\
        --candidate-cols ds_datum_drop_m,log1p_ds_datum_dist_m,ds_datum_order_rel \\
        [--extra-parquet <path keyed by canonical_id>] \\
        --out-json <path>

Leakage/NaN discipline: OOF folds mirror the trainer's CV; the GBM handles NaN
natively (learned split), so candidate NaNs are NEVER silently imputed -- the
finite-fraction per candidate column is reported so a mostly-empty column is caught.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_conus_gnn import DEPTH_BANDS, core_metrics, depth_banded  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gbm_gate_wte_residual")

# Gate thresholds (plan 0a). ΔR^2 measured on the residual target; band MAD in metres.
GATE_R2_DELTA = 0.02
GATE_BAND_MAD_DELTA = 0.15
DEEP_THRESHOLD_M = 30.0


def _hgb(max_iter: int = 500, learning_rate: float = 0.05, min_samples_leaf: int = 40):
    """The exact ceiling-test control config (plan 0a); squared-error to match R^2."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=max_iter,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
    )


def _r2(hat: np.ndarray, obs: np.ndarray) -> float:
    """OOF coefficient of determination on finite-pair rows (NaN if too few)."""
    m = np.isfinite(hat) & np.isfinite(obs)
    if m.sum() < 2:
        return float("nan")
    o = obs[m]
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    ss_res = float(np.sum((o - hat[m]) ** 2))
    return 1.0 - ss_res / ss_tot


def _oof_predict(df, feat_cols, target_col, fold_col, make_model):
    """Leak-free OOF residual predictions + the per-fold fitted models.

    Fold ``f`` is predicted ONLY from other folds' rows with a finite target; rows
    whose target is non-finite are excluded from training and left NaN in the OOF
    vector (never imputed -- an unscored well stays unscored). ``X`` is passed raw so
    the GBM's native NaN handling learns a missing-branch instead of us filling it.
    """
    X = df[feat_cols].to_numpy("float64")
    y = df[target_col].to_numpy("float64")
    fold = df[fold_col].to_numpy()
    oof = np.full(len(df), np.nan)
    models: dict = {}
    for f in np.unique(fold):
        tr = (fold != f) & np.isfinite(y)
        te = fold == f
        if tr.sum() < 2 or te.sum() < 1:
            log.warning(
                "fold %s: too few rows (train=%d test=%d), skipped",
                f,
                int(tr.sum()),
                int(te.sum()),
            )
            continue
        m = make_model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
        models[f] = m
    return oof, models


def _predict_with_fold_models(X, fold, models):
    """Predict each fold's rows with that fold's OOF model (for permutation reuse)."""
    out = np.full(len(X), np.nan)
    for f, m in models.items():
        te = fold == f
        if te.any():
            out[te] = m.predict(X[te])
    return out


def _perm_importance(
    df, feat_cols, cand_cols, models, target_col, fold_col, seed, repeats
):
    """Marginal OOF permutation importance (ΔR^2 when a column is shuffled).

    Reuses the fitted fold models: each candidate column is permuted across all rows
    and re-predicted through the SAME OOF split, so the drop reflects the fitted
    model's reliance on that column, not a refit. Positive => the column carries
    signal the model used.
    """
    rng = np.random.RandomState(seed)
    X = df[feat_cols].to_numpy("float64")
    y = df[target_col].to_numpy("float64")
    fold = df[fold_col].to_numpy()
    base_r2 = _r2(_predict_with_fold_models(X, fold, models), y)
    idx = {c: feat_cols.index(c) for c in cand_cols}
    out = {}
    for c, j in idx.items():
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(len(X)), j]
            drops.append(base_r2 - _r2(_predict_with_fold_models(Xp, fold, models), y))
        out[c] = float(np.mean(drops))
    return out


def run_gate(
    df: pd.DataFrame,
    baseline_cols,
    candidate_cols,
    *,
    target_col: str,
    dtw_base_col: str,
    obs_dtw_col: str,
    fold_col: str,
    deep_threshold_m: float = DEEP_THRESHOLD_M,
    make_model=None,
    perm_repeats: int = 5,
    seed: int = 0,
) -> dict:
    """OOF-GBM ceiling gate: baseline features vs baseline + candidate columns.

    Returns a JSON-serialisable dict with the residual-target R^2 (overall + obs-30+m
    subset), the reconstructed-DTW panel + per-band MAD deltas for both models,
    candidate permutation importances, per-candidate finite-fraction, and the gate
    verdict (which of the three criteria fired). No file I/O -- unit-testable.
    """
    make_model = make_model or _hgb
    baseline_cols = list(dict.fromkeys(baseline_cols))
    cand_feats = list(dict.fromkeys([*baseline_cols, *candidate_cols]))

    resid_obs = df[target_col].to_numpy("float64")
    dtw_base = df[dtw_base_col].to_numpy("float64")
    obs_dtw = df[obs_dtw_col].to_numpy("float64")

    base_hat, _ = _oof_predict(df, baseline_cols, target_col, fold_col, make_model)
    cand_hat, cand_models = _oof_predict(
        df, cand_feats, target_col, fold_col, make_model
    )

    # Ceiling metric: R^2 on the residual target itself (overall + the deep tail).
    r2_base = _r2(base_hat, resid_obs)
    r2_cand = _r2(cand_hat, resid_obs)
    deep = np.isfinite(obs_dtw) & (obs_dtw >= deep_threshold_m)
    r2_base_deep = _r2(base_hat[deep], resid_obs[deep])
    r2_cand_deep = _r2(cand_hat[deep], resid_obs[deep])

    # Reconstruct DTW exactly as the trainer: dtw_hat = (z_surf - R) - resid_hat.
    dtw_base_hat = dtw_base - base_hat
    dtw_cand_hat = dtw_base - cand_hat
    panel_base = {
        "core": core_metrics(dtw_base_hat, obs_dtw),
        "bands": depth_banded(dtw_base_hat, obs_dtw),
    }
    panel_cand = {
        "core": core_metrics(dtw_cand_hat, obs_dtw),
        "bands": depth_banded(dtw_cand_hat, obs_dtw),
    }

    # Per-band MAD delta (positive == candidate improves; bands empty in either -> None).
    band_delta = {}
    for lo, hi in DEPTH_BANDS:
        label = f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"
        b, c = panel_base["bands"].get(label, {}), panel_cand["bands"].get(label, {})
        if b.get("n", 0) and c.get("n", 0):
            band_delta[label] = float(b["mad_m"] - c["mad_m"])
        else:
            band_delta[label] = None

    perm = _perm_importance(
        df,
        cand_feats,
        list(candidate_cols),
        cand_models,
        target_col,
        fold_col,
        seed,
        perm_repeats,
    )
    finite = {
        c: float(np.isfinite(df[c].to_numpy("float64")).mean()) for c in candidate_cols
    }

    d_r2 = r2_cand - r2_base
    d_r2_deep = r2_cand_deep - r2_base_deep
    gains = [v for v in band_delta.values() if v is not None]
    best_band_gain = max(gains) if gains else float("nan")
    worst_band_gain = min(gains) if gains else float("nan")
    # Band criterion: some band improves >= 0.15 m AND no band regresses by >= 0.15 m.
    band_ok = (
        bool(gains)
        and best_band_gain >= GATE_BAND_MAD_DELTA
        and worst_band_gain > -GATE_BAND_MAD_DELTA
    )
    overall_ok = np.isfinite(d_r2) and d_r2 >= GATE_R2_DELTA
    deep_ok = np.isfinite(d_r2_deep) and d_r2_deep >= GATE_R2_DELTA

    reasons = []
    if overall_ok:
        reasons.append(f"overall ΔR²={d_r2:+.3f} >= {GATE_R2_DELTA}")
    if deep_ok:
        reasons.append(f"30+m ΔR²={d_r2_deep:+.3f} >= {GATE_R2_DELTA}")
    if band_ok:
        reasons.append(
            f"band MAD gain={best_band_gain:+.2f}m (worst {worst_band_gain:+.2f}m)"
        )

    return {
        "baseline_cols": baseline_cols,
        "candidate_cols": list(candidate_cols),
        "n_rows": int(len(df)),
        "n_resid_finite": int(np.isfinite(resid_obs).sum()),
        "n_deep": int(deep.sum()),
        "r2_resid_baseline": r2_base,
        "r2_resid_candidate": r2_cand,
        "delta_r2": float(d_r2) if np.isfinite(d_r2) else None,
        "r2_resid_deep_baseline": r2_base_deep,
        "r2_resid_deep_candidate": r2_cand_deep,
        "delta_r2_deep": float(d_r2_deep) if np.isfinite(d_r2_deep) else None,
        "panel_baseline": panel_base,
        "panel_candidate": panel_cand,
        "band_mad_delta_m": band_delta,
        "permutation_importance": perm,
        "finite_fraction": finite,
        "gate": {
            "overall_ok": bool(overall_ok),
            "deep_ok": bool(deep_ok),
            "band_ok": bool(band_ok),
            "passes": bool(overall_ok or deep_ok or band_ok),
            "reasons": reasons,
        },
    }


def _log_verdict(res: dict) -> None:
    g = res["gate"]
    log.info(
        "resid R²: baseline %.3f -> candidate %.3f (Δ %+.3f) | 30+m %.3f -> %.3f (Δ %+.3f, n=%d)",
        res["r2_resid_baseline"],
        res["r2_resid_candidate"],
        res["delta_r2"] if res["delta_r2"] is not None else float("nan"),
        res["r2_resid_deep_baseline"],
        res["r2_resid_deep_candidate"],
        res["delta_r2_deep"] if res["delta_r2_deep"] is not None else float("nan"),
        res["n_deep"],
    )
    cb, cc = res["panel_baseline"]["core"], res["panel_candidate"]["core"]
    log.info(
        "DTW panel: MAD %.2f->%.2f  bias %.2f->%.2f  medR %.2f->%.2f  RMSE %.2f->%.2f",
        cb.get("mad_m", float("nan")),
        cc.get("mad_m", float("nan")),
        cb.get("bias_mean_m", float("nan")),
        cc.get("bias_mean_m", float("nan")),
        cb.get("median_resid_m", float("nan")),
        cc.get("median_resid_m", float("nan")),
        cb.get("rmse_m", float("nan")),
        cc.get("rmse_m", float("nan")),
    )
    for band, d in res["band_mad_delta_m"].items():
        log.info(
            "  band %-8s MAD delta %s", band, f"{d:+.3f} m" if d is not None else "n/a"
        )
    for c, imp in res["permutation_importance"].items():
        log.info(
            "  perm-importance %-28s ΔR²=%+.4f  finite=%.3f",
            c,
            imp,
            res["finite_fraction"][c],
        )
    log.info(
        "GATE %s%s",
        "PASS" if g["passes"] else "FAIL",
        (" -- " + "; ".join(g["reasons"])) if g["reasons"] else "",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--graph-dir", required=True, help="wte_residual graph bundle dir")
    ap.add_argument(
        "--candidate-cols",
        required=True,
        help="comma-separated candidate feature columns",
    )
    ap.add_argument(
        "--extra-parquet",
        default=None,
        help="optional parquet keyed by canonical_id with candidate cols",
    )
    ap.add_argument(
        "--out-json", default=None, help="write the full gate result JSON here"
    )
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-samples-leaf", type=int, default=40)
    ap.add_argument("--perm-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gdir = Path(args.graph_dir)
    manifest = json.loads((gdir / "graph_manifest.json").read_text())
    if manifest.get("target_mode") != "wte_residual":
        raise SystemExit(
            f"gbm_gate is wte_residual-only; bundle target_mode={manifest.get('target_mode')!r} "
            "(the ceiling test is defined on obs_wte - R residual space)"
        )
    df = pd.read_parquet(gdir / "query_nodes.parquet")

    candidate_cols = [c.strip() for c in args.candidate_cols.split(",") if c.strip()]
    if args.extra_parquet:
        extra = pd.read_parquet(args.extra_parquet)
        join_cols = ["canonical_id", *[c for c in candidate_cols if c in extra.columns]]
        df = df.merge(
            extra[join_cols].drop_duplicates("canonical_id"),
            on="canonical_id",
            how="left",
        )
    missing = [c for c in candidate_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"candidate cols not in query_nodes or --extra-parquet: {missing}"
        )

    # Strip candidates from the baseline. When the gate is pointed at a bundle built WITH
    # the candidate flag (e.g. --graph-dir graph_relief_idw_dsdatum per plan §1.2), the
    # candidate cols are ALREADY in manifest query_feature_cols; leaving them in baseline
    # would make baseline == candidate and force ΔR²=0 -- a silent false-negative. The
    # gate always measures candidate-OVER-clean-baseline.
    baseline_cols = [
        c for c in manifest["query_feature_cols"] if c not in candidate_cols
    ]
    n_stripped = len(manifest["query_feature_cols"]) - len(baseline_cols)
    if n_stripped:
        log.info(
            "stripped %d candidate col(s) from the manifest baseline (bundle built WITH "
            "the candidate flag) -> measuring candidate over the clean baseline",
            n_stripped,
        )
    res = run_gate(
        df,
        baseline_cols,
        candidate_cols,
        target_col=manifest["target_col"],
        dtw_base_col=manifest["dtw_base_col"],
        obs_dtw_col=manifest["obs_dtw_col"],
        fold_col=manifest["cv_fold_col"],
        make_model=lambda: _hgb(
            args.max_iter, args.learning_rate, args.min_samples_leaf
        ),
        perm_repeats=args.perm_repeats,
        seed=args.seed,
    )
    _log_verdict(res)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=2))
        log.info("wrote gate result -> %s", args.out_json)


if __name__ == "__main__":
    main()
