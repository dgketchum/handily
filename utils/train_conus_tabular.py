"""Tabular-fusion control for the CONUS WTE/DTW GNN: does message passing earn
its place over plain feature fusion?

Trains a per-HUC4-fold gradient-boosted (or linear) regressor on the SAME query
features, native target, base, and CV folds as ``train_conus_gnn.py``, reconstructing
DTW exactly as the GNN does -- ``regional_base + residual_hat`` in dtw_residual mode,
or ``z_surf - wte_hat`` in WTE (head-target) mode. The OOF predictions are written in
the scorer's schema so ``score_conus_gnn.py --tabular-dir`` scores this control on the
IDENTICAL metric panel and common footprint. If the GNN does not beat this on the full
panel, the graph is not justified at these features -- the headline 5.21 would just be
"fuse HAND + regional", which a GBM gets without any message passing.

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
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conus_graph_inputs import (  # noqa: E402
    ANCHOR_CLASSES,
    DEM,
    TARGET_DTW_RESIDUAL,
    TARGET_WTE,
    sample_coarse,
)
from train_wte_gnn import apply_stats, fit_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_conus_tabular")

# Neighborhood-density radii for the anchor ablation. The plan calls for a
# same-HUC12 count, but neither wells nor anchors carry a HUC12 id (an explicit
# WBD join is out of scope here); the ~5 km radius is the HUC12-scale substitute
# and the 1/20 km radii bracket it. Logged, not silently dropped.
ANCHOR_RADII_M = (1000.0, 5000.0, 20000.0)


def build_anchor_tabular_features(
    qn: pd.DataFrame, gdir: Path, dem_path: str, with_head: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """Strong per-anchor-class geometry features -- the kill-test control.

    The GNN's anchor gain is only "earned" if message passing beats a GBM that is
    *handed* the anchor geometry as plain columns. So, per class (spring/open_water/
    wetland) and per well, build: nearest-anchor distance, nearest-anchor relative
    elevation (well land-surface minus that anchor's DEM head -- the gate against
    pulling an upland well down to a valley spring), nearest-anchor head_uncertainty,
    and within-radius counts (``ANCHOR_RADII_M``). Plus the on-network signal the
    graph would otherwise monopolize: distance to the nearest anchor attached to the
    well's controlling (rank-0 lateral) reach.

    In WTE (head-target) mode ``with_head`` adds the nearest-anchor *absolute head*
    (``anch_{c}_nn_head_m``), the head-space boundary value the GNN injects on its
    anchor value channel -- so the GBM sees the same head-magnitude signal. The
    surf-minus-head scalar is already ``anch_{c}_nn_rel_elev_m`` (kept canonical, not
    duplicated). Controlling-reach anchor head is deferred (the ctrl block resolves a
    min-distance, not a per-anchor head); logged as such.

    Returns the feature frame (aligned to ``qn`` row order) and the feature-name list.
    Absent values stay NaN (a class with no anchors yields no columns; an off-DEM
    well yields NaN rel-elev) -- the trainer median-imputes + flags, never fills.
    """
    apath = gdir / "anchor_nodes.parquet"
    if not apath.exists():
        return pd.DataFrame(index=qn.index), []
    anodes = pd.read_parquet(apath)
    qxy = qn[["x5070", "y5070"]].to_numpy("float64")
    well_surf = sample_coarse(dem_path, qxy[:, 0], qxy[:, 1])
    log.info(
        "anchor ablation: %d anchors, well land-surface finite frac=%.4f",
        len(anodes),
        float(np.isfinite(well_surf).mean()),
    )

    out: dict[str, np.ndarray] = {}
    for c in ANCHOR_CLASSES:
        sub = anodes[anodes["anchor_class"] == c]
        if len(sub) == 0:
            continue
        axy = sub[["x5070", "y5070"]].to_numpy("float64")
        head = sub["head_m"].to_numpy("float64")
        unc = sub["head_uncertainty_m"].to_numpy("float64")
        tree = cKDTree(axy)
        d, idx = tree.query(qxy, k=1)
        out[f"anch_{c}_nn_dist_m"] = d
        out[f"anch_{c}_nn_rel_elev_m"] = well_surf - head[idx]
        out[f"anch_{c}_nn_head_unc_m"] = unc[idx]
        if with_head:
            out[f"anch_{c}_nn_head_m"] = head[idx]
        for r in ANCHOR_RADII_M:
            out[f"anch_{c}_cnt_{int(r // 1000)}km"] = tree.query_ball_point(
                qxy, r, return_length=True
            ).astype("float64")
        log.info("  class %s: %d anchors", c, len(sub))

    # On-network: nearest anchor on the well's controlling reach. The lateral edge
    # with rank 0 (is_controlling) gives each well its controlling reach; anchor->
    # reach edges give that reach's anchors; the feature is the min well->anchor
    # euclidean distance over them (NaN if the controlling reach has no anchor).
    lat = pd.read_parquet(gdir / "lateral_edges.parquet")
    ctrl = lat.loc[lat["is_controlling"] == 1, ["query_node_idx", "reach_node_idx"]]
    ar = pd.read_parquet(gdir / "anchor_to_reach_edges.parquet")[
        ["anchor_node_idx", "reach_node_idx"]
    ]
    axy_all = anodes.set_index("anchor_node_idx")[["x5070", "y5070"]]
    pairs = ctrl.merge(ar, on="reach_node_idx", how="inner")
    if len(pairs):
        wxy = qn.loc[pairs["query_node_idx"].to_numpy(), ["x5070", "y5070"]].to_numpy()
        anc = axy_all.reindex(pairs["anchor_node_idx"].to_numpy()).to_numpy()
        pairs["d"] = np.hypot(wxy[:, 0] - anc[:, 0], wxy[:, 1] - anc[:, 1])
        ctrl_dist = pairs.groupby("query_node_idx")["d"].min()
        out["anch_ctrl_reach_nn_dist_m"] = ctrl_dist.reindex(
            np.arange(len(qn))
        ).to_numpy()
        log.info(
            "  controlling-reach anchor: %d/%d wells have an on-network anchor",
            int(np.isfinite(out["anch_ctrl_reach_nn_dist_m"]).sum()),
            len(qn),
        )

    feats = pd.DataFrame(out, index=qn.index)
    return feats, list(feats.columns)


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


def _native_to_dtw(native: np.ndarray, base: np.ndarray, mode: str) -> np.ndarray:
    """Reconstruct DTW from the native prediction, mirroring train_conus_gnn.

    ``dtw_residual``: native is a residual over the regional DTW prior -> base + native.
    ``wte``: native is absolute water-table elevation -> z_surf - native (base = z_surf).
    """
    return base - native if mode == TARGET_WTE else base + native


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
    ap.add_argument(
        "--no-anchor-ablation",
        action="store_true",
        help="skip the per-class anchor geometry features even if the bundle has anchors",
    )
    ap.add_argument("--dem", default=DEM, help="land-surface DEM for anchor rel-elev")
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
    target_mode = man.get("target_mode", TARGET_DTW_RESIDUAL)
    regional_prior_col = man.get("regional_prior_col")  # None in WTE mode
    surface_col = man.get("surface_elev_col")
    obs_wte_col = man.get("obs_wte_col")
    obs_dtw_col = man["obs_dtw_col"]
    if target_mode == TARGET_WTE:
        base_col = surface_col
        if base_col is None:
            raise SystemExit("WTE bundle missing surface_elev_col in manifest")
    else:
        base_col = regional_prior_col
        if base_col is None:
            raise SystemExit(
                "dtw_residual bundle missing regional_prior_col in manifest"
            )
    log.info(
        "model=%s  mode=%s  features=%s  base=%s  target=%s",
        args.model,
        target_mode,
        query_cols,
        base_col,
        target_col,
    )

    qn = (
        pd.read_parquet(gdir / "query_nodes.parquet")
        .sort_values("query_node_idx")
        .reset_index(drop=True)
    )

    # Anchor-geometry ablation (the kill test): hand the GBM the same anchor signal
    # the GNN message-passes, as plain per-class columns. Auto-enabled when the
    # bundle carries anchors; --no-anchor-ablation runs the widened control alone.
    query_cols = list(query_cols)
    anchor_cols: list[str] = []
    if not args.no_anchor_ablation:
        anchor_feats, anchor_cols = build_anchor_tabular_features(
            qn, gdir, args.dem, with_head=(target_mode == TARGET_WTE)
        )
        if anchor_cols:
            qn = pd.concat([qn, anchor_feats], axis=1)
            query_cols += anchor_cols
            log.info(
                "anchor ablation: +%d feature cols -> %s", len(anchor_cols), anchor_cols
            )
        else:
            log.info("no anchors in bundle: running widened control only")

    base = qn[base_col].to_numpy("float64")
    obs = qn[obs_dtw_col].to_numpy("float64")
    target = qn[target_col].to_numpy("float64")
    fold = qn[fold_col].to_numpy()
    obs_wte = qn[obs_wte_col].to_numpy("float64") if target_mode == TARGET_WTE else None
    checks = [("base", base), ("obs", obs), ("target", target)]
    if obs_wte is not None:
        checks.append(("obs_wte", obs_wte))
    for nm, arr in checks:
        if not np.isfinite(arr).all():
            raise SystemExit(
                f"{int((~np.isfinite(arr)).sum())} non-finite {nm} in query nodes"
            )

    # Feature names match apply_stats' column order: z-scored query cols, then a
    # NaN-indicator per ever-NaN col (whole-df, so stable across folds).
    nan_cols = [c for c in query_cols if qn[c].isna().any()]
    feat_names = list(query_cols) + [f"{c}__isnan" for c in nan_cols]

    native_oof = np.full(len(qn), np.nan)
    imp_acc = np.zeros(len(feat_names)) if args.importance_sample else None
    imp_n = 0
    for f in sorted(np.unique(fold)):
        te = fold == f
        tr = ~te
        stats = fit_stats(qn, query_cols, tr)
        x = apply_stats(qn, stats)
        model = _hgb() if args.model == "hgb" else _linear()
        model.fit(x[tr], target[tr])
        native_oof[te] = model.predict(x[te])
        fold_dtw = _native_to_dtw(native_oof[te], base[te], target_mode)
        fold_mad = float(np.nanmedian(np.abs(fold_dtw - obs[te])))
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

    if not np.isfinite(native_oof).all():
        raise SystemExit(
            f"{int((~np.isfinite(native_oof)).sum())} queries got no OOF prediction"
        )

    tab_dtw = _native_to_dtw(native_oof, base, target_mode)
    out_cols = {
        "canonical_id": qn["canonical_id"].to_numpy(),
        "cv_fold": fold,
        "obs_dtw_m": obs,
        "tab_dtw_m": tab_dtw,
    }
    if target_mode == TARGET_WTE:
        out_cols["z_surf_well_m"] = base
        out_cols["obs_wte_m"] = obs_wte
        out_cols["tab_wte_hat_m"] = native_oof
    else:
        out_cols["regional_base_m"] = base
        out_cols["tab_residual_hat_m"] = native_oof
    out = pd.DataFrame(out_cols)
    out.to_parquet(out_dir / "tabular_oof_predictions.parquet")

    importances = dict(zip(feat_names, (imp_acc / imp_n).tolist())) if imp_n else None
    run = {
        "graph_dir": str(gdir),
        "model": args.model,
        "target_mode": target_mode,
        "feature_cols": query_cols,
        "feature_names": feat_names,
        "base_col": base_col,
        "regional_prior_col": regional_prior_col,
        "surface_elev_col": surface_col,
        "target_col": target_col,
        "importance_target": target_col,
        "native_prediction_col": (
            "tab_wte_hat_m" if target_mode == TARGET_WTE else "tab_residual_hat_m"
        ),
        "dtw_reconstruction": (
            "z_surf_well_m - tab_wte_hat_m"
            if target_mode == TARGET_WTE
            else "regional_base_m + tab_residual_hat_m"
        ),
        "final_dtw_definition": man["final_dtw_definition"],
        "oof_dtw_mad_m": float(np.nanmedian(np.abs(tab_dtw - obs))),
        "anchor_ablation": {
            "enabled": bool(anchor_cols),
            "anchor_feature_cols": anchor_cols,
            "radii_m": list(ANCHOR_RADII_M),
            "huc12_count_substitute": "within-radius counts (~5km ~= HUC12 scale)",
        },
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
