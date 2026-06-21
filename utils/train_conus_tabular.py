"""Tabular-fusion control for the CONUS WTE/DTW GNN: does message passing earn
its place over plain feature fusion?

Trains a per-HUC4-fold gradient-boosted (or linear) regressor on the SAME query
features, residual target, regional base, and CV folds as ``train_conus_gnn.py``,
reconstructing ``dtw = regional_base + residual_hat`` exactly as the GNN does. The
OOF predictions are written in the scorer's schema so ``score_conus_gnn.py
--tabular-dir`` scores this control on the IDENTICAL metric panel and common
footprint. If the GNN does not beat this on the full panel, the graph is not
justified at these features -- the headline 5.21 would just be "fuse HAND +
regional", which a GBM gets without any message passing.

Feature handling is byte-for-byte the GNN's: ``fit_stats``/``apply_stats`` from
``train_wte_gnn`` z-score on TRAIN rows only and append NaN-indicator columns
(median-imputed + flagged, never silently filled). The estimator is the audited
project default ``HistGradientBoostingRegressor`` (mirrors
``build_wte_regional_prior._hgb``); ``--model linear`` swaps in a ridge floor.

    uv run python utils/train_conus_tabular.py \\
        --graph-dir /data/ssd2/handily/conus/wte_gnn/graph \\
        --out-dir   /data/ssd2/handily/conus/wte_gnn/tabular
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
from train_wte_gnn import apply_stats, fit_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_conus_tabular")


def _hgb():
    """Audited project default (build_wte_regional_prior._hgb), shallow + robust."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    kw = dict(
        max_iter=300,
        learning_rate=0.04,
        max_depth=2,
        min_samples_leaf=20,
        l2_regularization=2.0,
    )
    try:
        return HistGradientBoostingRegressor(loss="absolute_error", **kw)
    except (TypeError, ValueError):  # older sklearn without absolute_error
        return HistGradientBoostingRegressor(loss="squared_error", **kw)


def _linear():
    from sklearn.linear_model import Ridge

    return Ridge(alpha=1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-dir", default="/data/ssd2/handily/conus/wte_gnn/graph")
    ap.add_argument("--out-dir", default="/data/ssd2/handily/conus/wte_gnn/tabular")
    ap.add_argument("--model", choices=["hgb", "linear"], default="hgb")
    ap.add_argument(
        "--importance-sample",
        type=int,
        default=20000,
        help="capped per-fold test rows for permutation importance (0 = skip)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gdir = Path(args.graph_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    man = json.loads((gdir / "graph_manifest.json").read_text())
    query_cols = man["query_feature_cols"]
    target_col = man["target_col"]
    fold_col = man["cv_fold_col"]
    regional_prior_col = man["regional_prior_col"]
    obs_dtw_col = man["obs_dtw_col"]
    log.info(
        "model=%s  features=%s  base=%s  target=%s",
        args.model,
        query_cols,
        regional_prior_col,
        target_col,
    )

    qn = (
        pd.read_parquet(gdir / "query_nodes.parquet")
        .sort_values("query_node_idx")
        .reset_index(drop=True)
    )
    regional = qn[regional_prior_col].to_numpy("float64")
    obs = qn[obs_dtw_col].to_numpy("float64")
    target = qn[target_col].to_numpy("float64")
    fold = qn[fold_col].to_numpy()
    for nm, arr in (("regional", regional), ("obs", obs), ("target", target)):
        if not np.isfinite(arr).all():
            raise SystemExit(
                f"{int((~np.isfinite(arr)).sum())} non-finite {nm} in query nodes"
            )

    # Feature names match apply_stats' column order: z-scored query cols, then a
    # NaN-indicator per ever-NaN col (whole-df, so stable across folds).
    nan_cols = [c for c in query_cols if qn[c].isna().any()]
    feat_names = list(query_cols) + [f"{c}__isnan" for c in nan_cols]

    resid_oof = np.full(len(qn), np.nan)
    imp_acc = np.zeros(len(feat_names)) if args.importance_sample else None
    imp_n = 0
    for f in sorted(np.unique(fold)):
        te = fold == f
        tr = ~te
        stats = fit_stats(qn, query_cols, tr)
        x = apply_stats(qn, stats)
        model = _hgb() if args.model == "hgb" else _linear()
        model.fit(x[tr], target[tr])
        resid_oof[te] = model.predict(x[te])
        fold_mad = float(np.nanmedian(np.abs((regional[te] + resid_oof[te]) - obs[te])))
        log.info(
            "fold %d: train=%d test=%d | test DTW-MAD=%.3f",
            f,
            tr.sum(),
            te.sum(),
            fold_mad,
        )
        if args.importance_sample:
            from sklearn.inspection import permutation_importance

            te_idx = np.where(te)[0]
            if len(te_idx) > args.importance_sample:
                te_idx = rng.choice(te_idx, args.importance_sample, replace=False)
            r = permutation_importance(
                model, x[te_idx], target[te_idx], n_repeats=3, random_state=args.seed
            )
            imp_acc += r.importances_mean
            imp_n += 1

    if not np.isfinite(resid_oof).all():
        raise SystemExit(
            f"{int((~np.isfinite(resid_oof)).sum())} queries got no OOF prediction"
        )

    tab_dtw = regional + resid_oof
    out = pd.DataFrame(
        {
            "canonical_id": qn["canonical_id"].to_numpy(),
            "cv_fold": fold,
            "obs_dtw_m": obs,
            "regional_base_m": regional,
            "tab_residual_hat_m": resid_oof,
            "tab_dtw_m": tab_dtw,
        }
    )
    out.to_parquet(out_dir / "tabular_oof_predictions.parquet")

    importances = dict(zip(feat_names, (imp_acc / imp_n).tolist())) if imp_n else None
    run = {
        "graph_dir": str(gdir),
        "model": args.model,
        "feature_cols": query_cols,
        "feature_names": feat_names,
        "regional_prior_col": regional_prior_col,
        "target_col": target_col,
        "final_dtw_definition": man["final_dtw_definition"],
        "oof_dtw_mad_m": float(np.nanmedian(np.abs(tab_dtw - obs))),
        "permutation_importance_mean": importances,
    }
    (out_dir / "tabular_run.json").write_text(json.dumps(run, indent=2, default=str))
    log.info(
        "OOF DTW-MAD=%.3f m | wrote tabular_oof_predictions.parquet + tabular_run.json -> %s",
        run["oof_dtw_mad_m"],
        out_dir,
    )


if __name__ == "__main__":
    main()
