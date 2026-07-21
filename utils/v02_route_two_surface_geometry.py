"""E2 — re-attempt the WP2 two-surface routing with aquifer-geometry features,
using a transparent CPU GBT router over the EXISTING trained components (no
component retraining). See notes/E2_AQUIFER_GEOMETRY.md task (b).

WP2's in-GNN membership head pi routed at chance (OOF AUC 0.48-0.55) for lack of
data that discriminates deep-vs-shallow regime. E2 tests whether the new
aquifer-geometry layers (utils/v02_build_geometry_layers.py) make the routing
learnable. A HistGradientBoosting classifier predicts, per well, which frozen
component is the ORACLE-best (phreatic vs regional), cross-fit on HUC4-blocked
folds (the frozen cv_fold is NOT HUC4-nested, so a fresh grouping is required),
sacrificial HUC4s excluded. Two feature sets isolate the geometry contribution:

  * A (control)  = candidates WP2 had at inference (terrain/climate/priors +
                   component outputs). NO geometry.
  * B (treatment)= A + geometry (gravity depth-to-basement, basin-fill thickness,
                   High-Plains base-of-aquifer depth, gravity anomaly, coverage
                   flags, and a coalesced aquifer-fill-depth composite).

The routed blend (soft: p_route*phreatic + (1-p_route)*regional; hard: argmax) is
scored against the frozen w25_prod baseline, the single components, the WP2
mixture / pi-hard, and the oracle ceiling, on the same footprint. Full metric
panel (MAD/bias/median-resid/RMSE, depth-banded, n, metres) with paired-bootstrap
CI95 on the MAD skill. Gate (acceptance_gates.json wp2_two_surface): >=15 % 30+ m
MAD reduction on a separated deep panel (mech_deep_far_arid binding; 10-30 m
reported).

Usage:
    uv run python utils/v02_route_two_surface_geometry.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_conus_gnn import core_metrics, depth_banded, shallow_skill  # noqa: E402
from v02_metrics import paired_improvement  # noqa: E402
from v02_score_two_surface import (  # noqa: E402
    component_dtw,
    hard_assigned_dtw,
    oracle_best_dtw,
)

log = logging.getLogger("v02_route_two_surface_geometry")

WTE = "/data/ssd2/handily/conus/wte_gnn"
RUN_DIR = f"{WTE}/v02/wp2/gnn_w25_two_surface"
BASELINE = f"{WTE}/v02/contract/frozen_baseline/gnn_oof_predictions.parquet"
CONTRACT_DIR = f"{WTE}/v02/contract"
PRIORS = f"{WTE}/v02/wp2/assignment_priors.parquet"
GEOL = "/nas/handily/covariates/geology"
OUT_DIR = f"{WTE}/v02/e2_geometry"

SACRIFICIAL_HUC4 = ("0707", "1019", "1605")
DEPTH_BANDS = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
DEEP_BAND = "30+m"
MID_BAND = "10-30m"
SHALLOW_BANDS = ("0-2m", "2-5m")
N_FOLDS = 8

GEOMETRY_LAYERS = {
    "geom_depth_to_basement_grav_m": "depth_to_basement_grav_gb_m",
    "geom_iso_grav_anom_mgal": "iso_grav_anom_gb_mgal",
    "geom_basin_fill_thickness_m": "basin_fill_thickness_br_m",
    "geom_hp_base_alt_m": "hp_base_of_aquifer_alt_m",
}

# candidate features WP2 had at inference time (terrain/climate/priors + outputs)
CANDIDATE_PANEL_COLS = [
    "aridity_index",
    "slope_deg",
    "tri_100m",
    "dist_to_stream_m",
    "log1p_dist_perm_water_m",
    "tpi_10km",
    "haf_500m",
    "irrigated_2017",
]


# --------------------------------------------------------------------------- #
# pure logic (unit-testable)
# --------------------------------------------------------------------------- #
def oracle_phreatic_label(phreatic_dtw, regional_dtw, obs) -> np.ndarray:
    """1 where the phreatic component is the (weakly) better fit, else 0.

    The router's supervised target. Uses obs, so it is only ever evaluated
    out-of-fold (HUC4-blocked cross-fit) to stay honest."""
    p = np.abs(np.asarray(phreatic_dtw, float) - np.asarray(obs, float))
    r = np.abs(np.asarray(regional_dtw, float) - np.asarray(obs, float))
    return (p <= r).astype(int)


def assign_group_folds(groups: np.ndarray, k: int) -> np.ndarray:
    """Assign each row to one of ``k`` folds so every group is wholly in one fold.

    Greedy largest-group-to-emptiest-fold balancing (deterministic; groups sorted
    by descending size then name). Guarantees no group spans folds — the HUC4
    blocking the mission requires. Returns an int fold index per row."""
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    order = sorted(range(len(uniq)), key=lambda i: (-counts[i], str(uniq[i])))
    load = np.zeros(k, dtype=np.int64)
    g2f = {}
    for i in order:
        f = int(np.argmin(load))
        g2f[uniq[i]] = f
        load[f] += counts[i]
    return np.array([g2f[g] for g in groups], dtype=int)


def oracle_fraction_recovered(mad_base, mad_routed, mad_oracle) -> float | None:
    """Fraction of the oracle MAD headroom the router recovered (dimensionless).

    (mad_base - mad_routed) / (mad_base - mad_oracle); 1.0 = matches oracle,
    0.0 = no better than baseline, negative = worse than baseline. None when the
    oracle offers no headroom over baseline."""
    denom = mad_base - mad_oracle
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        return None
    return float((mad_base - mad_routed) / denom)


def routed_soft(p_route, phreatic_dtw, regional_dtw) -> np.ndarray:
    """Soft-blended routed DTW (m): p_route*phreatic + (1-p_route)*regional."""
    p = np.asarray(p_route, float)
    return p * np.asarray(phreatic_dtw, float) + (1.0 - p) * np.asarray(
        regional_dtw, float
    )


# --------------------------------------------------------------------------- #
# data + features
# --------------------------------------------------------------------------- #
def sample_raster(path: str, xy: list[tuple[float, float]]) -> np.ndarray:
    with rasterio.open(path) as ds:
        vals = np.array([s[0] for s in ds.sample(xy)], dtype=float)
        nd = ds.nodata
    if nd is not None:
        vals[vals == nd] = np.nan
    return vals


def load_frame(args) -> pd.DataFrame:
    """Scored real-well frame with components, candidates, baseline, geometry."""
    oof = pd.read_parquet(args.run_dir_oof)
    base = pd.read_parquet(
        args.baseline, columns=["canonical_id", "is_water_pseudo", "gnn_dtw_m"]
    ).rename(columns={"gnn_dtw_m": "base_dtw_m"})
    pcols = [
        "canonical_id",
        "is_water_pseudo",
        "huc4",
        "regional_holdout",
        "mech_deep_far_arid",
        "mech_shallow_irrigated",
        "panel_buffered_2p5km",
    ] + CANDIDATE_PANEL_COLS
    panels = pd.read_parquet(args.panels, columns=pcols)

    real = oof[~oof["is_water_pseudo"].astype(bool)].copy()
    base_r = base[~base["is_water_pseudo"].astype(bool)].drop(
        columns=["is_water_pseudo"]
    )
    pan_r = panels[~panels["is_water_pseudo"].astype(bool)].drop(
        columns=["is_water_pseudo"]
    )
    df = real.merge(base_r, on="canonical_id", how="left", validate="one_to_one")
    df = df.merge(pan_r, on="canonical_id", how="left", validate="one_to_one")

    fin = np.isfinite(df["obs_dtw_m"].to_numpy(float))
    sac = df["huc4"].isin(SACRIFICIAL_HUC4) | (df["regional_holdout"] == "sacrificial")
    df = df[fin & ~sac.to_numpy()].reset_index(drop=True)

    # component DTWs (identical construction to v02_score_two_surface)
    z = df["z_surf_well_m"].to_numpy(float)
    r_wte = df["regional_wte_idw_oof_m"].to_numpy(float)
    obs = df["obs_dtw_m"].to_numpy(float)
    pi = df["ts_pi_phreatic"].to_numpy(float)
    df["phreatic_dtw_m"] = component_dtw(z, r_wte, df["ts_native_p_m"].to_numpy(float))
    df["regional_dtw_m"] = component_dtw(z, r_wte, df["ts_native_r_m"].to_numpy(float))
    df["oracle_dtw_m"] = oracle_best_dtw(
        df["phreatic_dtw_m"].to_numpy(float), df["regional_dtw_m"].to_numpy(float), obs
    )
    df["wp2_hard_dtw_m"] = hard_assigned_dtw(
        pi, df["phreatic_dtw_m"].to_numpy(float), df["regional_dtw_m"].to_numpy(float)
    )
    df["y_oracle_phreatic"] = oracle_phreatic_label(
        df["phreatic_dtw_m"].to_numpy(float), df["regional_dtw_m"].to_numpy(float), obs
    )
    df["comp_disagree_m"] = df["phreatic_dtw_m"] - df["regional_dtw_m"]
    df["abs_comp_disagree_m"] = df["comp_disagree_m"].abs()
    df["fac_rem_dtw_m"] = z - df["fac_rem_wte_m"].to_numpy(float)

    # geometry samples
    xy = list(zip(df["x5070"].to_numpy(float), df["y5070"].to_numpy(float)))
    for feat, fname in GEOMETRY_LAYERS.items():
        df[feat] = sample_raster(f"{args.geol_dir}/{fname}.tif", xy)
    df["geom_hp_base_depth_m"] = z - df["geom_hp_base_alt_m"].to_numpy(float)
    # coalesced "depth of the aquifer/fill system" composite (max deep-well coverage)
    df["geom_aquifer_fill_depth_m"] = (
        df["geom_hp_base_depth_m"]
        .where(df["geom_hp_base_depth_m"].notna(), df["geom_basin_fill_thickness_m"])
        .where(
            df["geom_hp_base_depth_m"].notna()
            | df["geom_basin_fill_thickness_m"].notna(),
            df["geom_depth_to_basement_grav_m"],
        )
    )
    df["geom_has_hp"] = df["geom_hp_base_alt_m"].notna().astype(float)
    df["geom_has_br"] = df["geom_basin_fill_thickness_m"].notna().astype(float)
    df["geom_has_gb"] = df["geom_depth_to_basement_grav_m"].notna().astype(float)
    return df


def feature_columns() -> tuple[list[str], list[str]]:
    """Return (control_A_cols, geometry_extra_cols). B = A + geometry."""
    a_cols = CANDIDATE_PANEL_COLS + [
        "regional_idw_dtw_oof_m",
        "regional_deep_idw_dtw_oof_m",
        "hand_m",
        "fac_rem_dtw_m",
        "ts_native_p_m",
        "ts_native_r_m",
        "ts_sigma_p_m",
        "ts_sigma_r_m",
        "ts_pi_phreatic",
        "phreatic_dtw_m",
        "regional_dtw_m",
        "comp_disagree_m",
        "abs_comp_disagree_m",
    ]
    geom_cols = [
        "geom_depth_to_basement_grav_m",
        "geom_iso_grav_anom_mgal",
        "geom_basin_fill_thickness_m",
        "geom_hp_base_depth_m",
        "geom_hp_base_alt_m",
        "geom_aquifer_fill_depth_m",
        "geom_has_hp",
        "geom_has_br",
        "geom_has_gb",
    ]
    return a_cols, geom_cols


def crossfit_router(
    df: pd.DataFrame, feat_cols: list[str], folds: np.ndarray, seed: int = 0
) -> np.ndarray:
    """HUC4-blocked cross-fit HistGBT classifier -> OOF P(phreatic-best)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = df[feat_cols].to_numpy(float)
    y = df["y_oracle_phreatic"].to_numpy(int)
    p = np.full(len(df), np.nan)
    for f in range(N_FOLDS):
        tr = folds != f
        te = folds == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
        # guard against a fold whose training set is single-class
        if len(np.unique(y[tr])) < 2:
            p[te] = float(y[tr].mean())
            continue
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def arm_panel(pred: np.ndarray, obs: np.ndarray) -> dict:
    return {
        "overall": core_metrics(pred, obs),
        "by_depth_band": depth_banded(pred, obs),
        "shallow_skill": shallow_skill(pred, obs),
    }


def band_reductions(pred: np.ndarray, base: np.ndarray, obs: np.ndarray) -> dict:
    out = {}
    for lo, hi in DEPTH_BANDS:
        label = f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"
        sel = np.isfinite(obs) & (obs >= lo) & (obs < hi)
        out[label] = paired_improvement((pred - obs)[sel], (base - obs)[sel])
    return out


def _round(obj, nd=4):
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), nd)
    return obj


def router_accuracy(p_route: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    m = mask & np.isfinite(p_route)
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    pred = (p_route[m] >= 0.5).astype(int)
    yy = y[m]
    out = {
        "n": n,
        "accuracy": float((pred == yy).mean()),
        "phreatic_base_rate": float(yy.mean()),
        "predicted_phreatic_rate": float(pred.mean()),
    }
    out["auc"] = float(roc_auc_score(yy, p_route[m])) if 0 < yy.sum() < n else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=RUN_DIR)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--contract-dir", default=CONTRACT_DIR)
    ap.add_argument("--geol-dir", default=GEOL)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.run_dir_oof = f"{args.run_dir}/gnn_oof_predictions.parquet"
    args.panels = f"{args.contract_dir}/wells_panels.parquet"
    gate = json.loads(Path(f"{args.contract_dir}/acceptance_gates.json").read_text())[
        "wp2_two_surface"
    ]
    min_red = float(gate["min_deep_band_error_reduction"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_frame(args)
    obs = df["obs_dtw_m"].to_numpy(float)
    folds = assign_group_folds(df["huc4"].to_numpy(), N_FOLDS)
    log.info("scored wells %d | folds %s", len(df), np.bincount(folds).tolist())

    a_cols, geom_cols = feature_columns()
    b_cols = a_cols + geom_cols
    log.info("router A features %d, B features %d", len(a_cols), len(b_cols))

    p_a = crossfit_router(df, a_cols, folds, args.seed)
    p_b = crossfit_router(df, b_cols, folds, args.seed)
    df["p_route_A"] = p_a
    df["p_route_B"] = p_b

    pdtw = df["phreatic_dtw_m"].to_numpy(float)
    rdtw = df["regional_dtw_m"].to_numpy(float)
    preds = {
        "routed_B_soft": routed_soft(p_b, pdtw, rdtw),
        "routed_B_hard": hard_assigned_dtw(p_b, pdtw, rdtw),
        "routed_A_soft": routed_soft(p_a, pdtw, rdtw),
        "routed_A_hard": hard_assigned_dtw(p_a, pdtw, rdtw),
        "frozen_baseline": df["base_dtw_m"].to_numpy(float),
        "phreatic_only": pdtw,
        "regional_only": rdtw,
        "wp2_mixture": df["gnn_dtw_m"].to_numpy(float),
        "wp2_pi_hard": df["wp2_hard_dtw_m"].to_numpy(float),
        "oracle_best": df["oracle_dtw_m"].to_numpy(float),
    }

    panel_masks = {
        "all": np.ones(len(df), bool),
        "mech_deep_far_arid": df["mech_deep_far_arid"].to_numpy(bool),
        "panel_buffered_2p5km": df["panel_buffered_2p5km"].to_numpy(bool),
    }

    report = {
        "experiment": "E2 aquifer-geometry two-surface router",
        "run_dir": args.run_dir,
        "geol_dir": args.geol_dir,
        "n_scored_wells": int(len(df)),
        "min_deep_band_error_reduction": min_red,
        "definitions": {
            "target": "obs_dtw_m = observed depth to unconfined water table (m).",
            "oracle_phreatic_label": "1 if |phreatic-obs|<=|regional-obs| (router "
            "target; used only OOF).",
            "p_route": "OOF P(phreatic-best) from the HUC4-blocked HistGBT router.",
            "routed_soft": "p_route*phreatic + (1-p_route)*regional (m).",
            "routed_hard": "p_route>=0.5 -> phreatic else regional (m).",
            "mad_skill": "1 - MAD_model/MAD_base (dimensionless; + = better than "
            "frozen w25_prod baseline); ci95 = percentile paired-well bootstrap.",
            "oracle_fraction_recovered": "(MAD_base-MAD_routed)/(MAD_base-MAD_oracle) "
            "in-band (dimensionless; 1=oracle, 0=baseline).",
            "geometry_layers": GEOMETRY_LAYERS,
            "sacrificial_excluded": list(SACRIFICIAL_HUC4),
            "router_folds": f"{N_FOLDS} HUC4-blocked (each HUC4 wholly in one fold)",
        },
        "geometry_coverage": {
            "any_geometry_all": float(
                df[["geom_has_hp", "geom_has_br", "geom_has_gb"]].max(axis=1).mean()
            ),
            "any_geometry_deep": float(
                df.loc[obs >= 30, ["geom_has_hp", "geom_has_br", "geom_has_gb"]]
                .max(axis=1)
                .mean()
            ),
            "hp_all": float(df["geom_has_hp"].mean()),
            "br_all": float(df["geom_has_br"].mean()),
            "gb_all": float(df["geom_has_gb"].mean()),
        },
        "router_skill": {},
        "panels": {},
    }

    # router discrimination vs oracle label (overall + deep)
    y = df["y_oracle_phreatic"].to_numpy(int)
    deep_mask = obs >= 30
    for name, p in (("A_control", p_a), ("B_geometry", p_b)):
        report["router_skill"][name] = {
            "overall": router_accuracy(p, y, np.ones(len(df), bool)),
            "deep_30plus": router_accuracy(p, y, deep_mask),
            "mid_10_30": router_accuracy(p, y, (obs >= 10) & (obs < 30)),
        }

    # per-panel arm metrics + routed-B/A vs baseline band reductions
    base = preds["frozen_baseline"]
    for pname, mask in panel_masks.items():
        sub_obs = obs[mask]
        node = {"n": int(mask.sum()), "arms": {}}
        for aname, pred in preds.items():
            node["arms"][aname] = arm_panel(pred[mask], sub_obs)
        node["routed_B_soft_vs_baseline"] = band_reductions(
            preds["routed_B_soft"][mask], base[mask], sub_obs
        )
        node["routed_B_hard_vs_baseline"] = band_reductions(
            preds["routed_B_hard"][mask], base[mask], sub_obs
        )
        node["routed_A_soft_vs_baseline"] = band_reductions(
            preds["routed_A_soft"][mask], base[mask], sub_obs
        )
        report["panels"][pname] = node

    # ---- gate verdict (binding: mech_deep_far_arid, 30+m) ----
    deep_panel = report["panels"]["mech_deep_far_arid"]
    mad_oracle_deep = deep_panel["arms"]["oracle_best"]["by_depth_band"][DEEP_BAND][
        "mad_m"
    ]
    mad_base_deep = deep_panel["arms"]["frozen_baseline"]["by_depth_band"][DEEP_BAND][
        "mad_m"
    ]

    def _verdict(blend_key: str) -> dict:
        reds = deep_panel[f"{blend_key}_vs_baseline"]
        deep = reds[DEEP_BAND]
        mid = reds[MID_BAND]
        mad_routed = deep.get("mad_model_m")
        return {
            "deep_30plus": {
                "n": deep.get("n"),
                "mad_routed_m": mad_routed,
                "mad_base_m": deep.get("mad_base_m"),
                "mad_skill": deep.get("mad_skill"),
                "mad_skill_ci95": deep.get("mad_skill_ci95"),
                "rmse_skill": deep.get("rmse_skill"),
                "oracle_fraction_recovered": oracle_fraction_recovered(
                    mad_base_deep, mad_routed, mad_oracle_deep
                )
                if mad_routed is not None
                else None,
                "pass": bool(
                    deep.get("mad_skill") is not None
                    and deep.get("mad_skill") >= min_red
                ),
            },
            "mid_10_30_diagnostic": {
                "n": mid.get("n"),
                "mad_skill": mid.get("mad_skill"),
                "mad_skill_ci95": mid.get("mad_skill_ci95"),
            },
        }

    # shallow non-regression for the routed blend vs baseline (context)
    all_panel = report["panels"]["all"]
    shallow = {}
    for lbl in SHALLOW_BANDS:
        rb = all_panel["routed_B_soft_vs_baseline"][lbl]
        shallow[lbl] = {
            "mad_routed_m": rb.get("mad_model_m"),
            "mad_base_m": rb.get("mad_base_m"),
            "mad_skill": rb.get("mad_skill"),
        }

    report["gate_verdict"] = {
        "binding_panel": "mech_deep_far_arid",
        "min_deep_band_error_reduction": min_red,
        "oracle_ceiling": {
            "mad_base_deep_m": mad_base_deep,
            "mad_oracle_deep_m": mad_oracle_deep,
            "oracle_deep_mad_skill": _round(
                1.0 - mad_oracle_deep / max(mad_base_deep, 1e-9)
            ),
        },
        "routed_B_soft": _verdict("routed_B_soft"),
        "routed_B_hard": _verdict("routed_B_hard"),
        "routed_A_soft_control": _verdict("routed_A_soft"),
        "shallow_non_regression_routed_B_soft": shallow,
    }
    vb_soft = report["gate_verdict"]["routed_B_soft"]["deep_30plus"]["pass"]
    vb_hard = report["gate_verdict"]["routed_B_hard"]["deep_30plus"]["pass"]
    report["gate_verdict"]["overall_pass"] = bool(vb_soft or vb_hard)

    report = _round(report)
    (out / "e2_router_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    write_summary(report, out / "e2_router_summary.md")
    log.info(
        "GATE deep 30+m: B_soft skill=%s pass=%s | B_hard skill=%s pass=%s | "
        "oracle_ceiling=%s | OVERALL %s",
        report["gate_verdict"]["routed_B_soft"]["deep_30plus"]["mad_skill"],
        vb_soft,
        report["gate_verdict"]["routed_B_hard"]["deep_30plus"]["mad_skill"],
        vb_hard,
        report["gate_verdict"]["oracle_ceiling"]["oracle_deep_mad_skill"],
        report["gate_verdict"]["overall_pass"],
    )


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def write_summary(report: dict, path: Path) -> None:
    gv = report["gate_verdict"]
    cov = report["geometry_coverage"]
    lines = [
        "# E2 aquifer-geometry two-surface router — score summary",
        "",
        f"Scored {report['n_scored_wells']} unconfined real wells (sacrificial HUC4s "
        f"{', '.join(SACRIFICIAL_HUC4)} excluded), HUC4-blocked {N_FOLDS}-fold router "
        "cross-fit. All errors in metres; skills/AUC dimensionless.",
        "",
        f"Geometry coverage: any-geometry {cov['any_geometry_all']:.1%} of all wells, "
        f"{cov['any_geometry_deep']:.1%} of 30+ m wells "
        f"(HP {cov['hp_all']:.1%}, B&R {cov['br_all']:.1%}, GB-grav {cov['gb_all']:.1%}).",
        "",
        "## Router discrimination vs oracle assignment (OOF)",
        "",
        "| feature set | scope | n | accuracy | AUC | phreatic base-rate |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("A_control", "B_geometry"):
        for scope in ("overall", "deep_30plus", "mid_10_30"):
            r = report["router_skill"][name][scope]
            lines.append(
                f"| {name} | {scope} | {r.get('n')} | {_fmt(r.get('accuracy'))} | "
                f"{_fmt(r.get('auc'))} | {_fmt(r.get('phreatic_base_rate'))} |"
            )
    oc = gv["oracle_ceiling"]
    lines += [
        "",
        "## Gate — deep-band (30+ m) MAD reduction on mech_deep_far_arid",
        "",
        f"Oracle ceiling: baseline MAD {_fmt(oc['mad_base_deep_m'])} m -> oracle "
        f"{_fmt(oc['mad_oracle_deep_m'])} m (skill {_fmt(oc['oracle_deep_mad_skill'])}).",
        f"Gate threshold: MAD skill >= {gv['min_deep_band_error_reduction']:.2f}.",
        "",
        "| routed blend | n | MAD routed | MAD base | MAD skill | CI95 | RMSE skill | oracle frac | PASS |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in ("routed_B_soft", "routed_B_hard", "routed_A_soft_control"):
        d = gv[key]["deep_30plus"]
        ci = d.get("mad_skill_ci95")
        cistr = f"[{_fmt(ci[0])},{_fmt(ci[1])}]" if isinstance(ci, list) else "-"
        lines.append(
            f"| {key} | {d.get('n')} | {_fmt(d.get('mad_routed_m'))} | "
            f"{_fmt(d.get('mad_base_m'))} | {_fmt(d.get('mad_skill'))} | {cistr} | "
            f"{_fmt(d.get('rmse_skill'))} | {_fmt(d.get('oracle_fraction_recovered'))} | "
            f"{_fmt(d.get('pass'))} |"
        )
    lines += [
        "",
        "10-30 m diagnostic (routed_B_soft): MAD skill "
        f"{_fmt(gv['routed_B_soft']['mid_10_30_diagnostic']['mad_skill'])} "
        f"CI95 {gv['routed_B_soft']['mid_10_30_diagnostic'].get('mad_skill_ci95')}.",
        "",
        f"### GATE OVERALL (routed_B soft OR hard >= "
        f"{gv['min_deep_band_error_reduction']:.2f}): "
        f"{'PASS' if gv['overall_pass'] else 'FAIL'}",
        "",
        "## Deep panel (mech_deep_far_arid) MAD by observed-depth band — all arms",
        "",
        "| arm | overall | 0-2m | 2-5m | 5-10m | 10-30m | 30+m |",
        "|---|---|---|---|---|---|---|",
    ]
    arms = report["panels"]["mech_deep_far_arid"]["arms"]
    for aname, node in arms.items():
        b = node["by_depth_band"]

        def bm(lbl):
            return _fmt(b.get(lbl, {}).get("mad_m"))

        lines.append(
            f"| {aname} | {_fmt(node['overall'].get('mad_m'))} | {bm('0-2m')} | "
            f"{bm('2-5m')} | {bm('5-10m')} | {bm('10-30m')} | {bm('30+m')} |"
        )
    sh = gv["shallow_non_regression_routed_B_soft"]
    lines += [
        "",
        "## Shallow non-regression (routed_B_soft vs baseline, all wells)",
        "",
        "| band | MAD routed | MAD base | MAD skill |",
        "|---|---|---|---|",
    ]
    for lbl, d in sh.items():
        lines.append(
            f"| {lbl} | {_fmt(d['mad_routed_m'])} | {_fmt(d['mad_base_m'])} | "
            f"{_fmt(d['mad_skill'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
