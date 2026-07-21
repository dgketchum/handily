"""E2b — post-hoc aquifer-geometry BOUNDARY CONSTRAINT (depth clamp).

E2 (notes/E2_AQUIFER_GEOMETRY.md) showed the gravity / base-of-aquifer geometry
carries only a tiny routing signal; its top recommendation is to use geometry as a
physical BOUND on predicted depth rather than a router input. E2b tests that
post-hoc on EXISTING out-of-fold predictions — no retraining, no trainer edits, CPU
only.

Orientation (see notes/E2B_BOUNDARY_CLAMP.md). An unconfined water table sits within
the saturated aquifer, at/above the aquifer base:  WTE >= base_alt. With
DTW = z_surf - WTE this is an UPPER bound (cap) on depth:

    DTW <= z_surf - base_alt            (High Plains base-of-aquifer altitude)
    DTW <= basin_fill_thickness         (B&R basin fill)
    DTW <= depth_to_basement            (Great Basin gravity)

Composite = coalesce over layers with a valid positive depth (>0); nan where no
layer covers -> constraint NOT APPLICABLE (prediction unchanged, never imputed).
iso_grav_anom (mGal) is not a length and defines no bound -> excluded.

    cap  = D_geom + margin_m                 (margin_m >= 0 loosens; hedges noise)
    hard: DTW' = min(DTW, cap)
    soft: DTW' = cap + lam*(DTW - cap) if DTW > cap else DTW   (lam in [0,1))

Binds only when DTW > cap, so DTW' <= DTW: the clamp can only REDUCE depth. Scored on
the frozen v0.2 contract (unconfined wells, HUC4-blocked OOF folds, sacrificial
0707/1019/1605 excluded), full metric panel with paired-bootstrap CI95, clamped vs
unclamped, on both the geometry-covered footprint and all wells.

Usage:
    uv run python utils/v02_clamp_geometry.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_conus_gnn import core_metrics, depth_banded, shallow_skill  # noqa: E402

log = logging.getLogger("v02_clamp_geometry")

WTE = "/data/ssd2/handily/conus/wte_gnn"
BASELINE = f"{WTE}/v02/contract/frozen_baseline/gnn_oof_predictions.parquet"
E1_LEADER = f"{WTE}/gnn_conus_monitoring_water_e1_writeback/gnn_oof_predictions.parquet"
CONTRACT_DIR = f"{WTE}/v02/contract"
GEOL = "/nas/handily/covariates/geology"
OUT_DIR = f"{WTE}/v02/e2b_boundary"

SACRIFICIAL_HUC4 = ("0707", "1019", "1605")
DEPTH_BANDS = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
BAND_LABELS = ["0-2m", "2-5m", "5-10m", "10-30m", "30+m"]
SHALLOW_BANDS = ("0-2m", "2-5m")
MARGINS = (0, 10, 25, 50, 100, 200)
LAMBDAS = (0.25, 0.5, 0.75)


# --------------------------------------------------------------------------- #
# pure logic (unit-tested)
# --------------------------------------------------------------------------- #
def dtw_cap_from_base_alt(z_surf, base_alt) -> np.ndarray:
    """Depth-to-base-of-aquifer = z_surf - base_alt (m). Upper bound on DTW."""
    return np.asarray(z_surf, float) - np.asarray(base_alt, float)


def valid_depth(x) -> np.ndarray:
    """Keep only finite, strictly-positive depths; else nan (not-applicable)."""
    x = np.asarray(x, float)
    return np.where(np.isfinite(x) & (x > 0.0), x, np.nan)


def composite_bound(b_hp, b_fill, b_grav) -> np.ndarray:
    """Coalesce valid-positive depth bounds, priority HP -> fill -> gravity.

    Coverage is near-disjoint, so coalesce ~= min in practice; priority matches E2's
    composite. nan where no layer covers (constraint not applicable)."""
    b_hp, b_fill, b_grav = valid_depth(b_hp), valid_depth(b_fill), valid_depth(b_grav)
    out = np.where(
        np.isfinite(b_hp), b_hp, np.where(np.isfinite(b_fill), b_fill, b_grav)
    )
    return out


def hard_clamp(pred, cap) -> np.ndarray:
    """min(pred, cap) where cap is finite; unchanged where cap is nan."""
    pred = np.asarray(pred, float)
    cap = np.asarray(cap, float)
    binds = np.isfinite(cap) & (pred > cap)
    return np.where(binds, cap, pred)


def soft_clamp(pred, cap, lam) -> np.ndarray:
    """Shrink only the exceedance: cap + lam*(pred-cap) where pred>cap (cap finite).

    lam=0 reproduces the hard clamp; lam=1 is a no-op."""
    pred = np.asarray(pred, float)
    cap = np.asarray(cap, float)
    binds = np.isfinite(cap) & (pred > cap)
    shrunk = cap + float(lam) * (pred - cap)
    return np.where(binds, shrunk, pred)


def bind_mask(pred, cap) -> np.ndarray:
    """True where the cap would bind (finite cap and pred strictly exceeds it)."""
    pred = np.asarray(pred, float)
    cap = np.asarray(cap, float)
    return np.isfinite(cap) & (pred > cap)


def paired_skill_ci(err_model, err_base, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired-well MAD and RMSE skill (1 - model/base) with percentile CI95.

    Extends v02_metrics.paired_improvement with an RMSE-skill bootstrap CI so the
    gate ('CI-positive MAD or RMSE') is evaluable for both. Positive skill = clamped
    better than unclamped. Dimensionless; errors in metres."""
    e_m = np.abs(np.asarray(err_model, float))
    e_b = np.abs(np.asarray(err_base, float))
    ok = np.isfinite(e_m) & np.isfinite(e_b)
    e_m, e_b = e_m[ok], e_b[ok]
    n = len(e_m)
    if n == 0:
        return {"n": 0}
    mad_m, mad_b = float(np.median(e_m)), float(np.median(e_b))
    rmse_m = float(np.sqrt(np.mean(e_m**2)))
    rmse_b = float(np.sqrt(np.mean(e_b**2)))
    rng = np.random.default_rng(seed)
    mad_sk = np.empty(n_boot)
    rmse_sk = np.empty(n_boot)
    chunk = max(1, min(n_boot, int(2e7 // max(n, 1))))
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - s), n))
        bm, bb = e_m[idx], e_b[idx]
        mad_sk[s : s + len(idx)] = 1.0 - np.median(bm, axis=1) / np.maximum(
            np.median(bb, axis=1), 1e-9
        )
        rmse_sk[s : s + len(idx)] = 1.0 - np.sqrt(np.mean(bm**2, axis=1)) / np.maximum(
            np.sqrt(np.mean(bb**2, axis=1)), 1e-9
        )
    return {
        "n": n,
        "mad_model_m": round(mad_m, 4),
        "mad_base_m": round(mad_b, 4),
        "mad_skill": round(1.0 - mad_m / max(mad_b, 1e-9), 4),
        "mad_skill_ci95": [
            round(float(q), 4) for q in np.percentile(mad_sk, [2.5, 97.5])
        ],
        "rmse_model_m": round(rmse_m, 4),
        "rmse_base_m": round(rmse_b, 4),
        "rmse_skill": round(1.0 - rmse_m / max(rmse_b, 1e-9), 4),
        "rmse_skill_ci95": [
            round(float(q), 4) for q in np.percentile(rmse_sk, [2.5, 97.5])
        ],
    }


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def sample_raster(path: str, xy: list[tuple[float, float]]) -> np.ndarray:
    with rasterio.open(path) as ds:
        vals = np.array([s[0] for s in ds.sample(xy)], dtype=float)
        nd = ds.nodata
    if nd is not None:
        vals[vals == nd] = np.nan
    return vals


def load_frame(args) -> pd.DataFrame:
    """Scored-well frame: obs, z_surf, xy, panel masks, geometry bounds + all
    prediction sets, filtered to the frozen contract (unconfined, non-sacrificial)."""
    base = pd.read_parquet(args.baseline)
    real = base[~base["is_water_pseudo"].astype(bool)].copy()
    pan = pd.read_parquet(
        args.panels,
        columns=[
            "canonical_id",
            "is_water_pseudo",
            "huc4",
            "confinement_class",
            "regional_holdout",
            "mech_deep_far_arid",
            "panel_buffered_2p5km",
        ],
    )
    pan = pan[~pan["is_water_pseudo"].astype(bool)].drop(columns=["is_water_pseudo"])
    df = real.merge(pan, on="canonical_id", how="left", validate="one_to_one")

    fin = np.isfinite(df["obs_dtw_m"].to_numpy(float))
    sac = df["huc4"].isin(SACRIFICIAL_HUC4) | (df["regional_holdout"] == "sacrificial")
    conf = df["confinement_class"].isin(["unconfined", "unconfined_marginal"])
    df = df[fin & ~sac.to_numpy() & conf.to_numpy()].reset_index(drop=True)

    df = df.rename(columns={"gnn_dtw_m": "pred_frozen_baseline"})

    # E1 leader predictions (aligned by canonical_id)
    e1 = pd.read_parquet(args.e1_leader, columns=["canonical_id", "gnn_dtw_m"]).rename(
        columns={"gnn_dtw_m": "pred_e1_leader"}
    )
    df = df.merge(e1, on="canonical_id", how="left", validate="one_to_one")

    # geometry sampling + per-layer bounds
    z = df["z_surf_well_m"].to_numpy(float)
    xy = list(zip(df["x5070"].to_numpy(float), df["y5070"].to_numpy(float)))
    hp_alt = sample_raster(f"{args.geol_dir}/hp_base_of_aquifer_alt_m.tif", xy)
    fill = sample_raster(f"{args.geol_dir}/basin_fill_thickness_br_m.tif", xy)
    grav = sample_raster(f"{args.geol_dir}/depth_to_basement_grav_gb_m.tif", xy)
    df["bound_hp_m"] = valid_depth(dtw_cap_from_base_alt(z, hp_alt))
    df["bound_fill_m"] = valid_depth(fill)
    df["bound_grav_m"] = valid_depth(grav)
    df["bound_composite_m"] = composite_bound(
        df["bound_hp_m"], df["bound_fill_m"], df["bound_grav_m"]
    )
    return df


def add_routed_b_soft(df: pd.DataFrame, args) -> bool:
    """Reconstruct E2's geometry-routed soft blend per well (frozen WP2 components +
    HUC4-blocked geometry router), aligned by canonical_id. Returns success flag."""
    try:
        import v02_route_two_surface_geometry as e2

        rargs = SimpleNamespace(
            run_dir_oof=f"{WTE}/v02/wp2/gnn_w25_two_surface/gnn_oof_predictions.parquet",
            baseline=args.baseline,
            panels=args.panels,
            geol_dir=args.geol_dir,
        )
        rdf = e2.load_frame(rargs)
        folds = e2.assign_group_folds(rdf["huc4"].to_numpy(), e2.N_FOLDS)
        _, geom_cols = e2.feature_columns()
        a_cols, _ = e2.feature_columns()
        b_cols = a_cols + geom_cols
        p_b = e2.crossfit_router(rdf, b_cols, folds, seed=args.seed)
        routed = e2.routed_soft(
            p_b,
            rdf["phreatic_dtw_m"].to_numpy(float),
            rdf["regional_dtw_m"].to_numpy(float),
        )
        rmap = pd.DataFrame(
            {"canonical_id": rdf["canonical_id"].to_numpy(), "r": routed}
        )
        df["pred_routed_B_soft"] = df["canonical_id"].map(
            dict(zip(rmap["canonical_id"], rmap["r"]))
        )
        return True
    except Exception as e:  # noqa: BLE001 — optional set; log and skip
        log.warning("routed_B_soft reconstruction skipped: %s", e)
        return False


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
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


def band_ci_panel(clamped, unclamped, obs) -> dict:
    """Depth-banded paired MAD/RMSE skill (clamped vs unclamped) with CI95."""
    out = {}
    for (lo, hi), lbl in zip(DEPTH_BANDS, BAND_LABELS):
        sel = np.isfinite(obs) & (obs >= lo) & (obs < hi)
        out[lbl] = paired_skill_ci((clamped - obs)[sel], (unclamped - obs)[sel])
    out["overall"] = paired_skill_ci(clamped - obs, unclamped - obs)
    return out


def arm_metrics(pred, obs) -> dict:
    return {
        "overall": core_metrics(pred, obs),
        "by_depth_band": depth_banded(pred, obs),
        "shallow_skill": shallow_skill(pred, obs),
    }


def bind_stats(pred, obs, cap) -> dict:
    """Per-depth-band bind rate + per-bind residual-improvement diagnostics."""
    out = {}
    for (lo, hi), lbl in zip(DEPTH_BANDS, BAND_LABELS):
        sel = np.isfinite(cap) & np.isfinite(obs) & (obs >= lo) & (obs < hi)
        n = int(sel.sum())
        b = sel & (pred > cap)
        nb = int(b.sum())
        node = {"n_covered": n, "n_bind": nb, "bind_rate": (nb / n) if n else None}
        if nb:
            new_r = np.minimum(pred[b], cap[b]) - obs[b]
            old_r = pred[b] - obs[b]
            d = np.abs(new_r) - np.abs(old_r)
            node["mean_abs_resid_delta_m"] = float(d.mean())
            node["frac_bind_improved"] = float((d < 0).mean())
            node["frac_cap_below_obs"] = float((cap[b] < obs[b]).mean())
        out[lbl] = node
    return out


def sweep_deep(pred, obs, comp, footprint) -> dict:
    """Deep-band (30+ m) MAD/RMSE skill + bind rate over margin & lambda sweep."""
    cov = np.isfinite(comp) & footprint
    d = cov & (obs >= 30)
    base = (pred - obs)[d]
    n = int(d.sum())
    out = {"n_deep_covered": n, "hard_margin": {}, "soft_lambda": {}}
    if n == 0:
        return out
    for mg in MARGINS:
        cap = comp + mg
        cl = hard_clamp(pred, cap)
        r = (cl - obs)[d]
        nb = int((d & (pred > cap)).sum())
        out["hard_margin"][f"m{mg}"] = {
            "bind_rate": nb / n,
            **{
                k: v
                for k, v in paired_skill_ci(r, base).items()
                if k in ("mad_skill", "mad_skill_ci95", "rmse_skill", "rmse_skill_ci95")
            },
        }
    for lam in LAMBDAS:
        cl = soft_clamp(pred, comp, lam)
        r = (cl - obs)[d]
        out["soft_lambda"][f"l{lam}"] = {
            k: v
            for k, v in paired_skill_ci(r, base).items()
            if k in ("mad_skill", "mad_skill_ci95", "rmse_skill", "rmse_skill_ci95")
        }
    return out


def gate_from_sweep(sweep_cov: dict, shallow_ci: dict) -> dict:
    """PASS iff any variant gives CI-positive 30+ m MAD OR RMSE (lower CI > 0) with no
    CI-positive shallow degradation (upper CI of shallow skill >= 0)."""

    def ci_pos(node, key):
        ci = node.get(f"{key}_ci95")
        return bool(ci and ci[0] > 0.0)

    deep_hit = None
    for tag, node in {
        **{f"hard_{k}": v for k, v in sweep_cov.get("hard_margin", {}).items()},
        **{f"soft_{k}": v for k, v in sweep_cov.get("soft_lambda", {}).items()},
    }.items():
        if ci_pos(node, "mad_skill") or ci_pos(node, "rmse_skill"):
            deep_hit = tag
            break

    # shallow regression: any shallow band whose skill CI upper < 0 (clamped worse)
    shallow_reg = []
    for lbl in SHALLOW_BANDS:
        node = shallow_ci.get(lbl, {})
        for key in ("mad_skill", "rmse_skill"):
            ci = node.get(f"{key}_ci95")
            if ci and ci[1] < 0.0:
                shallow_reg.append(f"{lbl}:{key}")
    return {
        "deep_ci_positive_variant": deep_hit,
        "shallow_regression_bands": shallow_reg,
        "pass": bool(deep_hit is not None and not shallow_reg),
    }


def score_prediction_set(df: pd.DataFrame, pred_col: str, seed: int) -> dict:
    """Full clamped-vs-unclamped evaluation for one prediction set."""
    pred = df[pred_col].to_numpy(float)
    obs = df["obs_dtw_m"].to_numpy(float)
    comp = df["bound_composite_m"].to_numpy(float)
    cov = np.isfinite(comp)
    deep_far = df["mech_deep_far_arid"].to_numpy(bool)
    fin = np.isfinite(pred) & np.isfinite(obs)

    node = {
        "n_all_finite": int(fin.sum()),
        "n_covered": int((cov & fin).sum()),
        "coverage_frac_all": float(cov[fin].mean()) if fin.any() else None,
        "coverage_frac_deep": float(cov[fin & (obs >= 30)].mean())
        if (fin & (obs >= 30)).any()
        else None,
        "unclamped_panel": {},
        "primary_clamp": {},
        "sweep": {},
        "bind_stats": {},
    }

    # primary clamp = hard, margin 0 (pure physical cap)
    comp0 = comp
    clamped = hard_clamp(pred, comp0)

    for fpname, fp in (("covered", cov), ("all", np.ones(len(df), bool))):
        m = fp & fin
        node["unclamped_panel"][fpname] = arm_metrics(pred[m], obs[m])
        node["primary_clamp"][fpname] = {
            "clamped_panel": arm_metrics(clamped[m], obs[m]),
            "band_skill_ci": band_ci_panel(clamped[m], pred[m], obs[m]),
        }
        node["sweep"][fpname] = sweep_deep(
            pred[m], obs[m], comp[m], np.ones(int(m.sum()), bool)
        )
        node["bind_stats"][fpname] = bind_stats(pred[m], obs[m], comp0[m])

    # deep-far-arid separated panel (E2's binding panel) — covered only
    m = deep_far & cov & fin
    if m.sum():
        node["mech_deep_far_arid_covered"] = {
            "n": int(m.sum()),
            "sweep": sweep_deep(pred[m], obs[m], comp[m], np.ones(int(m.sum()), bool)),
        }

    # gate on the covered footprint
    shallow_ci = {
        lbl: node["primary_clamp"]["covered"]["band_skill_ci"][lbl]
        for lbl in SHALLOW_BANDS
    }
    node["gate"] = gate_from_sweep(node["sweep"]["covered"], shallow_ci)
    return node


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--e1-leader", default=E1_LEADER)
    ap.add_argument("--contract-dir", default=CONTRACT_DIR)
    ap.add_argument("--geol-dir", default=GEOL)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-routed", action="store_true", help="skip routed_B_soft set")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.panels = f"{args.contract_dir}/wells_panels.parquet"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_frame(args)
    log.info("scored wells %d", len(df))

    pred_sets = ["frozen_baseline", "e1_leader"]
    if not args.no_routed and add_routed_b_soft(df, args):
        pred_sets.append("routed_B_soft")
    log.info("prediction sets: %s", pred_sets)

    obs = df["obs_dtw_m"].to_numpy(float)
    report = {
        "experiment": "E2b aquifer-geometry boundary constraint (post-hoc depth clamp)",
        "n_scored_wells": int(len(df)),
        "definitions": {
            "orientation": "unconfined WTE >= base_alt => DTW <= z_surf-base_alt "
            "(upper bound/cap on depth). clamp reduces over-deep predictions only.",
            "cap": "D_geom + margin_m; D_geom = coalesce(z-hp_base_alt, fill_thick, "
            "basement_depth) over valid-positive layers; nan => not applicable.",
            "mad_skill": "1 - MAD_clamped/MAD_unclamped (dimensionless; + = clamp "
            "better); ci95 = percentile paired-well bootstrap.",
            "rmse_skill": "1 - RMSE_clamped/RMSE_unclamped (dimensionless).",
            "gate": "CI-positive 30+ m MAD or RMSE improvement (lower CI>0) on covered "
            "footprint with zero shallow regression (0-2/2-5 m skill upper CI>=0).",
            "sacrificial_excluded": list(SACRIFICIAL_HUC4),
            "iso_grav_anom_excluded": "mGal, not a length -> no physical depth bound.",
        },
        "geometry_coverage": {
            "hp_all": float(df["bound_hp_m"].notna().mean()),
            "fill_all": float(df["bound_fill_m"].notna().mean()),
            "grav_all": float(df["bound_grav_m"].notna().mean()),
            "composite_all": float(df["bound_composite_m"].notna().mean()),
            "composite_deep": float(
                df.loc[obs >= 30, "bound_composite_m"].notna().mean()
            ),
        },
        # bind rate PER geometry layer (margin 0, all-well finite footprint)
        "bind_rate_per_layer": {},
        "prediction_sets": {},
    }

    # per-layer bind rate on the frozen baseline prediction (representative)
    p0 = df["pred_frozen_baseline"].to_numpy(float)
    for layer in ("bound_hp_m", "bound_fill_m", "bound_grav_m", "bound_composite_m"):
        cap = df[layer].to_numpy(float)
        cov = np.isfinite(cap) & np.isfinite(p0)
        b = cov & (p0 > cap)
        report["bind_rate_per_layer"][layer] = {
            "n_covered": int(cov.sum()),
            "n_bind": int(b.sum()),
            "bind_rate": float(b.sum() / cov.sum()) if cov.sum() else None,
            "deep_bind_rate": float(
                (b & (obs >= 30)).sum() / max((cov & (obs >= 30)).sum(), 1)
            ),
        }

    for name in pred_sets:
        report["prediction_sets"][name] = score_prediction_set(
            df, f"pred_{name}", args.seed
        )

    # E2 reproduction check: routed_B_soft deep RMSE tail-trim vs E2's +0.13
    e2_repro = None
    if "routed_B_soft" in pred_sets:
        sw = (
            report["prediction_sets"]["routed_B_soft"]
            .get("mech_deep_far_arid_covered", {})
            .get("sweep", {})
        )
        best_rmse = None
        for grp in ("hard_margin", "soft_lambda"):
            for tag, node in sw.get(grp, {}).items():
                rs = node.get("rmse_skill")
                if rs is not None and (best_rmse is None or rs > best_rmse[1]):
                    best_rmse = (f"{grp}:{tag}", rs)
        e2_repro = {
            "e2_reference_rmse_skill": 0.13,
            "best_clamp_deep_rmse_skill_routed_B_soft": best_rmse,
            "note": "E2's +0.13 was routed_B_soft vs frozen baseline (a different "
            "denominator). Here rmse_skill is clamped vs UNCLAMPED routed_B_soft.",
        }
    report["e2_rmse_reproduction"] = e2_repro

    report = _round(report)
    (out / "e2b_clamp_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    write_summary(report, out / "e2b_clamp_summary.md")
    for name in pred_sets:
        g = report["prediction_sets"][name]["gate"]
        log.info(
            "GATE %s: pass=%s deep_hit=%s shallow_reg=%s",
            name,
            g["pass"],
            g["deep_ci_positive_variant"],
            g["shallow_regression_bands"],
        )


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def _ci(ci):
    return f"[{_fmt(ci[0])},{_fmt(ci[1])}]" if isinstance(ci, list) else "-"


def write_summary(report: dict, path: Path) -> None:
    cov = report["geometry_coverage"]
    lines = [
        "# E2b aquifer-geometry boundary clamp — score summary",
        "",
        f"Scored {report['n_scored_wells']} unconfined wells (sacrificial HUC4s "
        f"{', '.join(SACRIFICIAL_HUC4)} excluded). Orientation: DTW cap = "
        "z_surf-base_alt / fill_thickness / basement_depth. Errors in metres; "
        "skills dimensionless (1 - clamped/unclamped).",
        "",
        f"Geometry coverage (composite): {cov['composite_all']:.1%} of all wells, "
        f"{cov['composite_deep']:.1%} of 30+ m wells "
        f"(HP {cov['hp_all']:.1%}, B&R fill {cov['fill_all']:.1%}, GB grav "
        f"{cov['grav_all']:.1%}).",
        "",
        "## Bind rate per geometry layer (frozen baseline pred, margin 0)",
        "",
        "| layer | n covered | n bind | bind rate | deep(30+) bind rate |",
        "|---|---|---|---|---|",
    ]
    for layer, d in report["bind_rate_per_layer"].items():
        lines.append(
            f"| {layer} | {d['n_covered']} | {d['n_bind']} | "
            f"{_fmt(d['bind_rate'])} | {_fmt(d['deep_bind_rate'])} |"
        )

    for name, node in report["prediction_sets"].items():
        g = node["gate"]
        lines += [
            "",
            f"## {name}",
            "",
            f"Coverage: {node['n_covered']}/{node['n_all_finite']} wells "
            f"({_fmt(node['coverage_frac_all'])} all, {_fmt(node['coverage_frac_deep'])} "
            "deep). **GATE: "
            f"{'PASS' if g['pass'] else 'FAIL'}** (deep CI-positive variant="
            f"{g['deep_ci_positive_variant']}, shallow regression="
            f"{g['shallow_regression_bands'] or 'none'}).",
            "",
            "### Deep 30+ m margin sweep (covered footprint; clamped vs unclamped)",
            "",
            "| variant | bind rate | MAD skill | MAD CI95 | RMSE skill | RMSE CI95 |",
            "|---|---|---|---|---|---|",
        ]
        sw = node["sweep"]["covered"]
        for mg, d in sw.get("hard_margin", {}).items():
            lines.append(
                f"| hard {mg} | {_fmt(d.get('bind_rate'))} | {_fmt(d.get('mad_skill'))} | "
                f"{_ci(d.get('mad_skill_ci95'))} | {_fmt(d.get('rmse_skill'))} | "
                f"{_ci(d.get('rmse_skill_ci95'))} |"
            )
        for lam, d in sw.get("soft_lambda", {}).items():
            lines.append(
                f"| soft {lam} | - | {_fmt(d.get('mad_skill'))} | "
                f"{_ci(d.get('mad_skill_ci95'))} | {_fmt(d.get('rmse_skill'))} | "
                f"{_ci(d.get('rmse_skill_ci95'))} |"
            )
        # primary clamp depth-banded panel (covered)
        lines += [
            "",
            "### Primary clamp (hard, margin 0) depth-banded — covered footprint",
            "",
            "| band | n | unclamped MAD | clamped MAD | MAD skill | MAD CI95 | "
            "unclamped RMSE | clamped RMSE | RMSE skill |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        un = node["unclamped_panel"]["covered"]["by_depth_band"]
        cl = node["primary_clamp"]["covered"]["clamped_panel"]["by_depth_band"]
        bs = node["primary_clamp"]["covered"]["band_skill_ci"]
        for lbl in BAND_LABELS:
            u, c, s = un.get(lbl, {}), cl.get(lbl, {}), bs.get(lbl, {})
            lines.append(
                f"| {lbl} | {u.get('n', 0)} | {_fmt(u.get('mad_m'))} | "
                f"{_fmt(c.get('mad_m'))} | {_fmt(s.get('mad_skill'))} | "
                f"{_ci(s.get('mad_skill_ci95'))} | {_fmt(u.get('rmse_m'))} | "
                f"{_fmt(c.get('rmse_m'))} | {_fmt(s.get('rmse_skill'))} |"
            )
        # bind stats covered
        lines += [
            "",
            "### Bind diagnostics (primary clamp, covered) — did binds improve?",
            "",
            "| band | n covered | n bind | bind rate | mean abs-resid delta (m) | "
            "frac improved | frac cap<obs |",
            "|---|---|---|---|---|---|---|",
        ]
        for lbl in BAND_LABELS:
            d = node["bind_stats"]["covered"].get(lbl, {})
            lines.append(
                f"| {lbl} | {d.get('n_covered', 0)} | {d.get('n_bind', 0)} | "
                f"{_fmt(d.get('bind_rate'))} | {_fmt(d.get('mean_abs_resid_delta_m'))} | "
                f"{_fmt(d.get('frac_bind_improved'))} | {_fmt(d.get('frac_cap_below_obs'))} |"
            )

    if report.get("e2_rmse_reproduction"):
        e = report["e2_rmse_reproduction"]
        lines += [
            "",
            "## E2 +0.13 deep-RMSE reproduction check (routed_B_soft)",
            "",
            f"E2 reference deep RMSE skill (routed vs baseline): "
            f"{e['e2_reference_rmse_skill']}. Best clamp deep RMSE skill (clamped vs "
            f"UNCLAMPED routed_B_soft): {e['best_clamp_deep_rmse_skill_routed_B_soft']}.",
            "",
            e["note"],
        ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
