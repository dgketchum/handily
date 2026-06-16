"""Phase 3 of the REM-regional WTE hybrid: point-level signal diagnosis.

Before any expensive raster prediction, prove at the well points that:

  1. A *transferable* regional model (gridMET climate + topographic position only,
     no DEM, no coordinates, no pre-built water-table product) cuts FAC's deep
     arid-basin bias, and
  2. blending it with FAC under a physically-transparent gate recovers the best of
     both -- FAC keeps the riparian/shallow win, the regional model owns the deep
     table.

Everything is scored out-of-fold under spatial blocked CV (``block_20km``) so the
numbers reflect transfer, not coordinate memorization. Ma is scored as a benchmark
only -- never a training label (see CLAUDE.md). Springs are scored as shallow
discharge constraints, never as regional-aquifer training labels.

The regional model predicts DTW (depth) directly rather than WTE. DTW is the
climate-driven quantity (aridity + height above regional base level set how far
below ground the table sits); predicting it avoids letting absolute DEM elevation
dominate, which is what makes the fit transfer toward statewide / CONUS scale.
WTE = DEM - DTW is recovered after the fact for the blend.

Gate: p_fac = exp(-max(elev_above_str7, 0) / h0). The hybrid is scored under
nested blocked CV: the held-out regional component reuses the leak-free global
OOF regional prediction, while the gate length scale h0 is fit per outer fold on
an INNER blocked-CV regional prediction of the training rows only -- so h0 never
sees the held-out fold's labels. Near the Rio Grande mainstem (regional base
level) p_fac -> 1 and FAC dominates; high on the basin-fill uplands p_fac -> 0 and
the regional model dominates.

Springs are scored as pure holdout shallow-discharge constraints: the regional
model is trained on water-table WELLS ONLY (never on springs), per CLAUDE.md.

Output (under ``<out_dir>/diagnostics``):
  wte_diagnostic_scores.csv        product x group MAD/bias/RMSE/recall/precision
  wte_point_cv_predictions.parquet per-point out-of-fold predictions
  feature_correlations.csv         Spearman rho vs observed DTW and |FAC error|
  diagnostics_run.json             config, feature lists, counts, runtime
"""

import argparse
import json
import tomllib
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

# Transferable regional features: gridMET climate + topographic position only.
# Deliberately EXCLUDES dem_m, x/y coordinates, and every pre-built water-table
# product (Ma, Zell-Sanford, Fan) -- the point is a physically-driven model that
# carries beyond this basin, not a copy of an existing DTW raster.
CLIMATE = ["aridity_index", "precip_mm_yr", "pet_mm_yr", "vpd_kpa", "tmax_c"]
TOPO = [
    "elev_above_str1_m",
    "elev_above_str5_m",
    "elev_above_str7_m",
    "log1p_dist_str1_m",
    "log1p_dist_str5_m",
    "log1p_dist_str7_m",
    "slope_m_m",
    "local_relief_300m_m",
    "log1p_drainage_km2",
]
# Physical-flux / aquifer-property products (recharge, ET, baseflow index). These
# are drivers, not pre-built water-table tables, so they are an honest ablation.
PHYSICAL = [
    "reitz_effective_recharge",
    "reitz_total_recharge",
    "reitz_et",
    "wolock_bfi",
]

FEATURE_SETS = {
    "climate_topo": CLIMATE + TOPO,
    "climate_topo_physical": CLIMATE + TOPO + PHYSICAL,
}
GATE_VAR = "elev_above_str7_m"
GATE_SCALES_M = [20.0, 40.0, 60.0, 80.0, 120.0, 160.0, 200.0]
DEPTH_BINS = [-np.inf, 2, 5, 10, 30, 60, 100, np.inf]
DEPTH_LABELS = ["<2", "2-5", "5-10", "10-30", "30-60", "60-100", ">100"]
SHALLOW_M = 5.0


def _gate(elev_above, h0):
    """p_fac: FAC authority decays with height above regional base level."""
    return np.exp(-np.clip(elev_above, 0, None) / h0)


def _regional_model():
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=15,
        l2_regularization=1.0,
    )


def _blocked_cv_regional(d, feats, target, group_col):
    """Out-of-fold regional DTW prediction under spatial blocked CV."""
    X = d[feats].to_numpy("float64")
    y = d[target].to_numpy("float64")
    w = d["weight"].to_numpy("float64")
    grp = d[group_col].to_numpy()
    pred = np.full(len(d), np.nan)
    n_splits = min(5, len(np.unique(grp)))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, grp):
        mdl = _regional_model()
        mdl.fit(X[tr], y[tr], sample_weight=w[tr])
        pred[te] = mdl.predict(X[te])
    return np.clip(pred, 0, None)


def _hybrid_oof(wt, feats, obs, reg_global, group_col):
    """Leak-free FAC-authority blend scored out-of-fold (nested CV).

    The held-out regional component is reused from the global blocked-CV regional
    prediction ``reg_global`` -- for each fold that is a fit on all *other* folds,
    so it is leak-free for the test rows (GroupKFold is deterministic in the
    groups, so the partition matches). The gate length scale h0 is the only thing
    fit from the training rows, and it needs their regional prediction; that is
    recomputed by INNER blocked CV inside each outer-training set so h0 never sees
    the held-out fold's labels (the prior nested-CV leak).
    """
    fac = wt["fac_dtw_m"].to_numpy("float64")
    elev = wt[GATE_VAR].to_numpy("float64")
    w = wt["weight"].to_numpy("float64")
    grp = wt[group_col].to_numpy()
    hyb = np.full(len(wt), np.nan)
    p_fac = np.full(len(wt), np.nan)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(grp))))
    for tr, te in gkf.split(elev.reshape(-1, 1), obs, grp):
        reg_tr = _blocked_cv_regional(
            wt.iloc[tr].assign(_t=obs[tr]), feats, "_t", group_col
        )
        best_h, best_mad = GATE_SCALES_M[0], np.inf
        for h0 in GATE_SCALES_M:
            p = _gate(elev[tr], h0)
            blend = (1 - p) * reg_tr + p * fac[tr]
            mad = np.average(np.abs(blend - obs[tr]), weights=w[tr])
            if mad < best_mad:
                best_mad, best_h = mad, h0
        p_te = _gate(elev[te], best_h)
        hyb[te] = np.clip((1 - p_te) * reg_global[te] + p_te * fac[te], 0, None)
        p_fac[te] = p_te
    return hyb, p_fac


def _metrics(pred, obs, w):
    err = pred - obs
    m = {
        "n": int(len(obs)),
        "mad": float(np.median(np.abs(err))),
        "wmad": float(np.average(np.abs(err), weights=w)),
        "bias": float(np.median(err)),
        "rmse": float(np.sqrt(np.average(err**2, weights=w))),
    }
    obs_sh, pred_sh = obs < SHALLOW_M, pred < SHALLOW_M
    m["shallow_recall"] = float((pred_sh & obs_sh).sum() / max(obs_sh.sum(), 1))
    m["shallow_precision"] = float((pred_sh & obs_sh).sum() / max(pred_sh.sum(), 1))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    t0 = perf_counter()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    table = Path(cfg["covariate_table"]["out"])
    out_dir = Path(cfg["paths"]["out_dir"]).parent / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    group_col = cfg.get("regional_model", {}).get("spatial_cv_block", "block_20km")

    g = gpd.read_parquet(table)

    # In-grid water-table wells: need a finite FAC prediction, observed DTW, and the
    # gate / topo features (NaN here means the well is outside the basin DEM grid,
    # where neither FAC nor the topo-position model is defined -- reported, not
    # silently dropped).
    need = ["fac_dtw_m", "ma_dtw_m", "target_dtw_m", GATE_VAR] + CLIMATE + TOPO
    wells = g[(g.point_type == "well") & g.is_water_table_label].copy()
    in_grid = wells[need].notna().all(axis=1)
    n_oob = int((~in_grid).sum())
    wt = wells[in_grid].copy()
    wt["weight"] = wt["weight"].fillna(0.0).clip(lower=1e-3)

    # --- feature correlations (signal sanity) ---------------------------------
    corr_rows = []
    for c in CLIMATE + TOPO + PHYSICAL + ["ma_dtw_m"]:
        if c not in wt:
            continue
        m = wt[c].notna()
        if m.sum() < 10:
            continue
        r_obs, _ = spearmanr(wt.loc[m, c], wt.loc[m, "target_dtw_m"])
        r_err, _ = spearmanr(wt.loc[m, c], wt.loc[m, "fac_residual_dtw_m"].abs())
        corr_rows.append(
            {
                "feature": c,
                "rho_vs_obs_dtw": round(r_obs, 3),
                "rho_vs_abs_fac_err": round(r_err, 3),
                "n": int(m.sum()),
            }
        )
    pd.DataFrame(corr_rows).to_csv(out_dir / "feature_correlations.csv", index=False)

    # --- regional models + hybrid (out-of-fold) -------------------------------
    obs = wt["target_dtw_m"].to_numpy("float64")
    fac = wt["fac_dtw_m"].to_numpy("float64")
    ma = wt["ma_dtw_m"].to_numpy("float64")
    w = wt["weight"].to_numpy("float64")

    products = {"FAC": fac, "Ma": ma}
    feature_set_used = {}
    for set_name, feats in FEATURE_SETS.items():
        usable = [c for c in feats if wt[c].notna().mean() > 0.9]
        feature_set_used[set_name] = usable
        reg = _blocked_cv_regional(wt.assign(_t=obs), usable, "_t", group_col)
        wt[f"regional_dtw__{set_name}"] = reg
        products[f"Regional[{set_name}]"] = reg
        hyb, p_fac = _hybrid_oof(wt, usable, obs, reg, group_col)
        wt[f"p_fac__{set_name}"] = p_fac
        wt[f"hybrid_dtw__{set_name}"] = hyb
        products[f"Hybrid[{set_name}]"] = hyb

    # --- scorecard: overall, depth bins, riparian-vs-upland -------------------
    wt["depth_bin"] = pd.cut(obs, DEPTH_BINS, labels=DEPTH_LABELS)
    deep = obs >= 30
    rows = []
    for name, pred in products.items():
        rows.append(
            {"product": name, "group": "all_water_table", **_metrics(pred, obs, w)}
        )
        rows.append(
            {
                "product": name,
                "group": "deep>=30m",
                **_metrics(pred[deep], obs[deep], w[deep]),
            }
        )
        rows.append(
            {
                "product": name,
                "group": "shallow<5m",
                **_metrics(
                    pred[obs < SHALLOW_M], obs[obs < SHALLOW_M], w[obs < SHALLOW_M]
                ),
            }
        )
        for lab in DEPTH_LABELS:
            mask = (wt["depth_bin"] == lab).to_numpy()
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "product": name,
                    "group": f"depth {lab}m",
                    **_metrics(pred[mask], obs[mask], w[mask]),
                }
            )
    score = pd.DataFrame(rows)
    score.to_csv(out_dir / "wte_diagnostic_scores.csv", index=False)

    # --- springs: shallow-discharge capture (scoring only) --------------------
    spring_rows = []
    sp = g[(g.point_type == "spring")].copy()
    for set_name in FEATURE_SETS:
        if sp.empty:
            break
        # Springs are shallow-discharge constraints, NOT regional-aquifer labels
        # (CLAUDE.md): the regional model is trained on water-table WELLS ONLY and
        # springs are scored as pure holdout points. capture = predicted shallow.
        feats = feature_set_used[set_name]
        sp_in = sp[sp[feats + [GATE_VAR, "fac_dtw_m"]].notna().all(axis=1)].copy()
        if sp_in.empty:
            continue
        sp_mdl = _regional_model()
        sp_mdl.fit(wt[feats].to_numpy("float64"), obs, sample_weight=w)
        sp_reg = np.clip(sp_mdl.predict(sp_in[feats].to_numpy("float64")), 0, None)
        sp_p = _gate(sp_in[GATE_VAR].to_numpy("float64"), 60.0)
        sp_hyb = np.clip(
            (1 - sp_p) * sp_reg + sp_p * sp_in["fac_dtw_m"].to_numpy("float64"), 0, None
        )
        for prod, vals in [
            ("FAC", sp_in["fac_dtw_m"].to_numpy()),
            (f"Regional[{set_name}]", sp_reg),
            (f"Hybrid[{set_name}]", sp_hyb),
        ]:
            spring_rows.append(
                {
                    "product": prod,
                    "n": int(len(sp_in)),
                    "capture<2m": float((vals < 2).mean()),
                    "capture<5m": float((vals < 5).mean()),
                    "median_pred_dtw_m": float(np.median(vals)),
                }
            )
    if spring_rows:
        pd.DataFrame(spring_rows).to_csv(
            out_dir / "spring_diagnostic_scores.csv", index=False
        )

    # --- persist per-point predictions ----------------------------------------
    keep = [
        "canonical_id",
        "tier",
        "confinement_class",
        "target_dtw_m",
        "fac_dtw_m",
        "ma_dtw_m",
        GATE_VAR,
        group_col,
        "depth_bin",
        "x_5070",
        "y_5070",
        "geometry",
    ]
    keep += [
        c
        for c in wt.columns
        if c.startswith(("regional_dtw__", "p_fac__", "hybrid_dtw__"))
    ]
    wt[[c for c in keep if c in wt.columns]].to_parquet(
        out_dir / "wte_point_cv_predictions.parquet"
    )

    run = {
        "config": str(args.config),
        "table": str(table),
        "group_col": group_col,
        "n_water_table_in_grid": int(len(wt)),
        "n_water_table_out_of_grid": n_oob,
        "n_deep_ge30": int(deep.sum()),
        "n_springs_scored": int(len(sp)),
        "feature_sets": feature_set_used,
        "gate_var": GATE_VAR,
        "gate_scales_m": GATE_SCALES_M,
        "runtime_s": round(perf_counter() - t0, 1),
    }
    with open(out_dir / "diagnostics_run.json", "w") as f:
        json.dump(run, f, indent=2)

    # --- console summary -------------------------------------------------------
    print(
        f"water-table wells in-grid: {len(wt)}  (out-of-grid dropped: {n_oob})  "
        f"deep>=30m: {int(deep.sum())}"
    )
    show = score[score.group.isin(["all_water_table", "deep>=30m", "shallow<5m"])]
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(
            show[
                [
                    "product",
                    "group",
                    "n",
                    "mad",
                    "bias",
                    "rmse",
                    "shallow_recall",
                    "shallow_precision",
                ]
            ].to_string(index=False)
        )
    print(f"\nwrote diagnostics -> {out_dir}")


if __name__ == "__main__":
    main()
