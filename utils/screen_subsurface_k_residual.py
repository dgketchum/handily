"""Rung-0 residual screen: do CONUS2 lnK / Zell-Sanford transmissivity explain deep-band
residual variance the existing geology bundle does not?

Stage 1 (``--build``) assembles the screen frame: the FROZEN w25_prod baseline OOF
residuals on the v0.2 scoring population (real wells, finite obs, sacrificial HUC4s
dropped, UNCONFINED only) joined to the candidate + control geology covariates sampled
at the well coordinates.

Stage 2 (``--analyze``) runs, per observed-depth band, (a) partial Spearman/Pearson
correlation of |residual| and residual against each candidate controlling for the
existing geology controls, and (b) incremental gradient-boosting R^2 (residual as
target) with vs without the candidates, under HUC4-blocked CV so the increment is not
a spatial-memorisation artefact.

Definitions (all reported with units):
- residual_m = gnn_dtw_m - obs_dtw_m (m; positive = predicted too deep).
- partial r = correlation of the two residualised series after linear removal of the
  control block (dimensionless, -1..1).
- incremental R^2 = R^2(controls + candidates) - R^2(controls), out-of-fold under
  HUC4-blocked CV (dimensionless; positive = candidates add explanatory power).

Usage:
    uv run python utils/screen_subsurface_k_residual.py --build
    uv run python utils/screen_subsurface_k_residual.py --analyze
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
from build_stacker_features import sample_coarse  # noqa: E402

log = logging.getLogger("screen_subsurface_k")

WTE_GNN = "/data/ssd2/handily/conus/wte_gnn"
BASELINE_OOF = (
    f"{WTE_GNN}/gnn_conus_monitoring_water_gate_mirror_sigma_w25_prod/"
    "gnn_oof_predictions.parquet"
)
QUERY_NODES = f"{WTE_GNN}/graph_conus_monitoring_water/query_nodes.parquet"
PANELS = f"{WTE_GNN}/v02/contract/wells_panels.parquet"
OUT_DIR = Path(f"{WTE_GNN}/subsurface_k_screen")

SACRIFICIAL_HUC4 = ("0707", "1019", "1605")
UNCONFINED = ("unconfined", "unconfined_marginal")
DEPTH_BANDS = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 30.0), (30.0, np.inf)]

GEO = "/nas/handily/covariates/geology"
SUBK = "/nas/handily/covariates/subsurface_k"
ZS_TRANS = (
    "/nas/gwx/studies/analysis_ready/zell_sanford_2020_dtw/"
    "zell_sanford_2020_trans_5070.tif"
)

# Control block: exactly the five geology channels the model ALREADY sees. perm +
# sediment thickness are live query features; all five are in the MAE v2 payload
# manifest (configs/mae/manifest_v2.json).
CONTROL_SPECS = {
    "ctl_perm_logk_m2": (f"{GEO}/permeability_logk_x100.tif", 1.0 / 100.0),
    "ctl_porosity_frac": (f"{GEO}/porosity_x100.tif", 1.0 / 100.0),
    "ctl_sed_thickness_avg_m": (f"{GEO}/sediment_thickness_avg_m.tif", 1.0),
    "ctl_sed_thickness_basinfill_m": (
        f"{GEO}/sediment_thickness_basinfill_m.tif",
        1.0,
    ),
    "ctl_depth_to_bedrock_abs_m": (f"{GEO}/depth_to_bedrock_abs_cm.tif", 1.0 / 100.0),
}

# Candidate block. lnK entries are added only if the rasters exist (rung-A dependent).
CANDIDATE_SPECS = {
    "cnd_zs_log_trans": (ZS_TRANS, "log10"),
}

# "Geology ceiling" block: every OTHER geology raster in the covariate bank that the
# production bundle does NOT see. These are the aquifer-geometry / lithology fields that
# a CONUS2-style hydrostratigraphy is assembled from, so their joint incremental R^2
# bounds what ANY new static-geology covariate (lnK included) could contribute to the
# frozen model's residual. Continuous first, then categorical (passed to HistGBT as
# declared categoricals, not as ordered codes).
CEILING_CONT_SPECS = {
    "ceil_basin_fill_thickness_br_m": (f"{GEO}/basin_fill_thickness_br_m.tif", 1.0),
    "ceil_depth_to_basement_grav_m": (f"{GEO}/depth_to_basement_grav_gb_m.tif", 1.0),
    "ceil_hp_base_of_aquifer_alt_m": (f"{GEO}/hp_base_of_aquifer_alt_m.tif", 1.0),
    "ceil_iso_grav_anom_mgal": (f"{GEO}/iso_grav_anom_gb_mgal.tif", 1.0),
}
CEILING_CAT_SPECS = {
    "ceilcat_aquifer_code": (f"{GEO}/aquifer_code.tif", 1.0),
    "ceilcat_aquifer_rocktype": (f"{GEO}/aquifer_rocktype.tif", 1.0),
    "ceilcat_lithology_glim": (f"{GEO}/lithology_glim.tif", 1.0),
    "ceilcat_karst_type": (f"{GEO}/karst_type.tif", 1.0),
}


def _band_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"


def discover_lnk_candidates() -> dict:
    """Any *_5070.tif under the subsurface_k bank enters as a candidate (linear)."""
    out = {}
    d = Path(SUBK)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*_5070.tif")):
        out[f"cnd_{p.stem.replace('_5070', '')}"] = (str(p), "linear")
    return out


def build(args) -> pd.DataFrame:
    oof = pd.read_parquet(BASELINE_OOF)
    panels = pd.read_parquet(
        PANELS, columns=["canonical_id", "is_water_pseudo", "huc4", "confinement_class"]
    )
    panels = panels[~panels["is_water_pseudo"].astype(bool)].drop(
        columns=["is_water_pseudo"]
    )
    real = oof[~oof["is_water_pseudo"].astype(bool)].copy()
    df = real.merge(panels, on="canonical_id", how="left", validate="one_to_one")

    fin = np.isfinite(df["obs_dtw_m"].to_numpy(float))
    sac = df["huc4"].isin(SACRIFICIAL_HUC4).to_numpy()
    unconf = df["confinement_class"].isin(UNCONFINED).to_numpy()
    diag = {
        "n_oof_rows": int(len(oof)),
        "n_real": int(len(real)),
        "n_finite_obs": int(fin.sum()),
        "n_sacrificial_dropped": int((fin & sac).sum()),
        "n_confined_dropped": int((fin & ~sac & ~unconf).sum()),
    }
    df = df[fin & ~sac & unconf].reset_index(drop=True)
    diag["n_scored_unconfined"] = int(len(df))
    log.info("population: %s", json.dumps(diag))

    df["residual_m"] = df["gnn_dtw_m"].to_numpy(float) - df["obs_dtw_m"].to_numpy(float)
    df["abs_residual_m"] = np.abs(df["residual_m"].to_numpy(float))

    qn = pd.read_parquet(
        QUERY_NODES,
        columns=["canonical_id", "perm_logk_m2", "sediment_thickness_m", "well_depth"],
    )
    qn = qn.drop_duplicates("canonical_id")
    df = df.merge(qn, on="canonical_id", how="left", validate="one_to_one")

    x = df["x5070"].to_numpy(float)
    y = df["y5070"].to_numpy(float)

    specs = dict(CONTROL_SPECS)
    specs.update(CEILING_CONT_SPECS)
    specs.update(CEILING_CAT_SPECS)
    for name, (path, scale) in specs.items():
        v = sample_coarse(path, x, y) * scale
        df[name] = v
        log.info(
            "%s: nan %.3f%% min %.4g max %.4g",
            name,
            100.0 * np.isnan(v).mean(),
            np.nanmin(v),
            np.nanmax(v),
        )

    cands = dict(CANDIDATE_SPECS)
    cands.update(discover_lnk_candidates())
    for name, (path, mode) in cands.items():
        if not Path(path).exists():
            log.warning("candidate raster missing, skipped: %s", path)
            continue
        raw = sample_coarse(path, x, y)
        if mode == "log10":
            pos = raw > 0
            v = np.where(pos, np.log10(np.where(pos, raw, 1.0)), np.nan)
        else:
            v = raw
        df[name] = v
        log.info(
            "%s: nan %.3f%% min %.4g max %.4g",
            name,
            100.0 * np.isnan(v).mean(),
            np.nanmin(v),
            np.nanmax(v),
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "screen_wells.parquet"
    df.to_parquet(out, index=False)
    (OUT_DIR / "screen_build.json").write_text(
        json.dumps(
            {
                "population": diag,
                "baseline_oof": BASELINE_OOF,
                "controls": {k: v[0] for k, v in specs.items()},
                "candidates": {
                    k: v[0] for k, v in cands.items() if Path(v[0]).exists()
                },
            },
            indent=2,
        )
    )
    log.info("wrote %s (%d rows, %d cols)", out, len(df), df.shape[1])
    return df


def _resid_out(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residual of y after OLS on X (intercept added)."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def partial_corr(
    y: np.ndarray, c: np.ndarray, X: np.ndarray
) -> tuple[float, float, int]:
    """Partial Pearson + Spearman of y vs c controlling for X. Rows with any NaN in
    y/c/X are dropped (reported as n)."""
    from scipy.stats import rankdata

    ok = np.isfinite(y) & np.isfinite(c) & np.isfinite(X).all(1)
    n = int(ok.sum())
    if n < 100:
        return float("nan"), float("nan"), n
    yv, cv, Xv = y[ok], c[ok], X[ok]
    ry = _resid_out(yv, Xv)
    rc = _resid_out(cv, Xv)
    pear = float(np.corrcoef(ry, rc)[0, 1])
    # Spearman partial: rank-transform first, then residualise on ranked controls.
    Xr = np.column_stack([rankdata(Xv[:, j]) for j in range(Xv.shape[1])])
    rys = _resid_out(rankdata(yv), Xr)
    rcs = _resid_out(rankdata(cv), Xr)
    spear = float(np.corrcoef(rys, rcs)[0, 1])
    return pear, spear, n


def blocked_gbm_r2(
    df: pd.DataFrame, target: str, cols: list[str], seed: int = 0
) -> tuple[float, int]:
    """Out-of-fold R^2 of a HistGradientBoosting fit of `target` on `cols` under
    HUC4-blocked 5-fold CV. NaN features are handled natively by the learner (it
    routes missing values); rows with a NaN TARGET are dropped."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    sub = df[np.isfinite(df[target].to_numpy(float))]
    X = sub[cols].to_numpy(float)
    y = sub[target].to_numpy(float)
    g = sub["huc4"].astype(str).to_numpy()
    # ceilcat_* are class codes, not magnitudes: declare them categorical so the
    # learner partitions classes instead of ordering them.
    cat_mask = np.array([c.startswith("ceilcat_") for c in cols], bool)
    cat_arg = cat_mask if cat_mask.any() else None
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups=g):
        m = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=seed,
            categorical_features=cat_arg,
        )
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot, len(y)


def analyze(args) -> dict:
    df = pd.read_parquet(OUT_DIR / "screen_wells.parquet")
    ctl = [c for c in df.columns if c.startswith("ctl_")]
    cnd = [c for c in df.columns if c.startswith("cnd_")]
    if args.candidate_grep:
        want = tuple(s.strip() for s in args.candidate_grep.split(",") if s.strip())
        cnd = [c for c in cnd if any(w in c for w in want)]
        log.info("candidate filter %s -> %d candidates", want, len(cnd))
    if not cnd:
        raise SystemExit("no cnd_* candidate columns in the screen frame")
    ceil = [c for c in df.columns if c.startswith(("ceil_", "ceilcat_"))]

    # COVERAGE CONFOUND. The reconstructed CONUS2 lnK rasters are missing over whole
    # regions (the source indicator grid has no eastern seaboard / coastal plain /
    # California). A tree learner handles NaN by ROUTING it, so the candidate block
    # smuggles in a coarse east/west regional indicator that a HUC4-blocked split
    # happily transfers -- an apparent incremental R^2 could be pure coverage, not
    # conductivity. Two guards: a missingness-ONLY candidate on the full footprint
    # (if it reproduces the gain, the gain is coverage), and --covered-only, which
    # restricts every well to the lnK footprint so missingness cannot act at all.
    lnk_cols = [c for c in cnd if "lnk" in c]
    if lnk_cols and not args.covered_only:
        df["cndmiss_lnk_absent"] = df[lnk_cols[0]].isna().astype(float)
    covered_note = None
    if args.covered_only and lnk_cols:
        keep = df[lnk_cols].notna().all(axis=1).to_numpy()
        covered_note = {
            "n_before": int(len(df)),
            "n_after": int(keep.sum()),
            "dropped_frac": round(float(1.0 - keep.mean()), 4),
        }
        df = df[keep].reset_index(drop=True)
        log.info("covered-only footprint: %s", json.dumps(covered_note))
    miss_col = [c for c in df.columns if c.startswith("cndmiss_")]

    log.info("controls: %s", ctl)
    log.info("candidates: %s", cnd)
    log.info("ceiling block: %s", ceil)

    obs = df["obs_dtw_m"].to_numpy(float)
    report = {
        "definitions": {
            "residual_m": "gnn_dtw_m - obs_dtw_m (m; + = predicted too deep)",
            "partial_r": "Pearson/Spearman correlation after linear removal of the "
            "control block from both series (dimensionless, -1..1)",
            "incremental_r2": "out-of-fold R^2(controls+candidates) - R^2(controls), "
            "HUC4-blocked 5-fold GroupKFold (dimensionless; + = candidates help)",
            "geology_ceiling_r2": "out-of-fold R^2(controls + EVERY other geology "
            "raster in the covariate bank) - R^2(controls). Bounds what ANY new "
            "static-geology covariate could add to the frozen residual, so a "
            "candidate's incremental R^2 is read against this ceiling, not against 0.",
            "coverage_confound_r2": "out-of-fold R^2(controls + a single 0/1 "
            "'lnK raster is absent here' flag) - R^2(controls). The null model for "
            "the lnK increment: any candidate gain not exceeding this is regional "
            "coverage structure, not conductivity.",
            "covered_only": "when true, every well outside the lnK footprint is "
            "dropped BEFORE any fit, so missingness cannot carry information.",
            "controls": ctl,
            "candidates": cnd,
            "ceiling_block": ceil,
            "missingness_flag": miss_col,
        },
        "covered_only": bool(args.covered_only),
        "covered_only_population": covered_note,
        "bands": {},
    }

    for lo, hi in DEPTH_BANDS:
        lbl = _band_label(lo, hi)
        sel = np.isfinite(obs) & (obs >= lo) & (obs < hi)
        sub = df[sel].reset_index(drop=True)
        Xc = sub[ctl].to_numpy(float)
        band = {"n": int(sel.sum()), "partial_corr": {}}
        for tgt in ("residual_m", "abs_residual_m"):
            band["partial_corr"][tgt] = {}
            yv = sub[tgt].to_numpy(float)
            for c in cnd:
                p, s, n = partial_corr(yv, sub[c].to_numpy(float), Xc)
                band["partial_corr"][tgt][c] = {
                    "partial_pearson": round(p, 4) if np.isfinite(p) else None,
                    "partial_spearman": round(s, 4) if np.isfinite(s) else None,
                    "n": n,
                }
        if sub["huc4"].nunique() >= 5 and len(sub) >= 500:
            r2_c, n_c = blocked_gbm_r2(sub, "residual_m", ctl)
            r2_cc, _ = blocked_gbm_r2(sub, "residual_m", ctl + cnd)
            band["gbm_residual_m"] = {
                "r2_controls": round(r2_c, 4),
                "r2_controls_plus_candidates": round(r2_cc, 4),
                "incremental_r2": round(r2_cc - r2_c, 4),
                "n": n_c,
            }
            # per-candidate increment, so a null result cannot hide inside a block
            for c in cnd:
                r2_1, _ = blocked_gbm_r2(sub, "residual_m", ctl + [c])
                band["gbm_residual_m"][f"incremental_r2__{c}"] = round(r2_1 - r2_c, 4)
            if ceil:
                r2_ceil, _ = blocked_gbm_r2(sub, "residual_m", ctl + ceil)
                band["gbm_residual_m"]["geology_ceiling_r2"] = round(r2_ceil - r2_c, 4)
            if miss_col:
                r2_m, _ = blocked_gbm_r2(sub, "residual_m", ctl + miss_col)
                band["gbm_residual_m"]["coverage_confound_r2"] = round(r2_m - r2_c, 4)
        report["bands"][lbl] = band
        log.info("%s: %s", lbl, json.dumps(band.get("gbm_residual_m", {})))

    # all-band pooled
    r2_c, n_c = blocked_gbm_r2(df, "residual_m", ctl)
    r2_cc, _ = blocked_gbm_r2(df, "residual_m", ctl + cnd)
    report["pooled_gbm_residual_m"] = {
        "r2_controls": round(r2_c, 4),
        "r2_controls_plus_candidates": round(r2_cc, 4),
        "incremental_r2": round(r2_cc - r2_c, 4),
        "n": n_c,
    }
    for c in cnd:
        r2_1, _ = blocked_gbm_r2(df, "residual_m", ctl + [c])
        report["pooled_gbm_residual_m"][f"incremental_r2__{c}"] = round(r2_1 - r2_c, 4)
    if ceil:
        r2_ceil, _ = blocked_gbm_r2(df, "residual_m", ctl + ceil)
        report["pooled_gbm_residual_m"]["geology_ceiling_r2"] = round(r2_ceil - r2_c, 4)
    if miss_col:
        r2_m, _ = blocked_gbm_r2(df, "residual_m", ctl + miss_col)
        report["pooled_gbm_residual_m"]["coverage_confound_r2"] = round(r2_m - r2_c, 4)
    out = OUT_DIR / (
        args.report_name
        or (
            "screen_report_covered_only.json"
            if args.covered_only
            else "screen_report.json"
        )
    )
    out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", out)
    return report


def sidecar(args) -> None:
    """Emit a --extra-query-features sidecar: query_node_idx + the named candidate
    columns sampled at EVERY query node of the bundle (wells + pseudo-rows alike --
    the trainer's forward runs over all of them)."""
    qn = pd.read_parquet(QUERY_NODES, columns=["query_node_idx", "x5070", "y5070"])
    x = qn["x5070"].to_numpy(float)
    y = qn["y5070"].to_numpy(float)
    specs = dict(CANDIDATE_SPECS)
    specs.update(discover_lnk_candidates())
    want = args.sidecar_cols.split(",") if args.sidecar_cols else list(specs)
    out = pd.DataFrame({"query_node_idx": qn["query_node_idx"].to_numpy()})
    for name in want:
        if name not in specs:
            raise SystemExit(f"unknown candidate column {name}; have {sorted(specs)}")
        path, mode = specs[name]
        raw = sample_coarse(path, x, y)
        if mode == "log10":
            pos = raw > 0
            v = np.where(pos, np.log10(np.where(pos, raw, 1.0)), np.nan)
        else:
            v = raw
        out[name] = v
        log.info(
            "%s: nan %.3f%% min %.4g max %.4g",
            name,
            100.0 * np.isnan(v).mean(),
            np.nanmin(v),
            np.nanmax(v),
        )
    if args.sidecar_coverage_only:
        # Control arm for the coverage confound: keep ONLY the 0/1 "this raster has
        # no value here" flag and drop the values themselves. An arm on this sidecar
        # measures what the lnK arm gets from regional coverage structure alone, so
        # lnK_arm - coverage_arm isolates the conductivity content.
        flag = out[want[0]].isna().astype("float64")
        out = pd.DataFrame(
            {
                "query_node_idx": out["query_node_idx"].to_numpy(),
                "lnk_absent_flag": flag.to_numpy(),
            }
        )
        log.info("coverage-only sidecar: flag mean %.4f", float(flag.mean()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / args.sidecar_name
    out.to_parquet(p, index=False)
    log.info("wrote %s (%d rows, cols %s)", p, len(out), list(out.columns))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument(
        "--sidecar",
        action="store_true",
        help="emit a --extra-query-features sidecar over all bundle query nodes",
    )
    ap.add_argument("--sidecar-cols", default=None, help="comma-separated cnd_* names")
    ap.add_argument("--sidecar-name", default="sidecar_query_features.parquet")
    ap.add_argument(
        "--sidecar-coverage-only",
        action="store_true",
        help="emit ONLY the 0/1 missingness flag of the first --sidecar-cols column "
        "(control arm isolating the coverage confound from conductivity)",
    )
    ap.add_argument(
        "--covered-only",
        action="store_true",
        help="restrict --analyze to wells inside the lnK footprint, so the raster's "
        "regionally-structured missingness cannot masquerade as signal",
    )
    ap.add_argument(
        "--candidate-grep",
        default=None,
        help="comma-separated substrings; keep only cnd_* columns containing one of "
        "them. Lets the open-route reconstruction and the credentialed full-domain "
        "field be screened as separate candidate blocks in one frame, so their "
        "incremental R2 stays comparable to the earlier partial-domain table.",
    )
    ap.add_argument(
        "--report-name",
        default=None,
        help="override the screen_report*.json filename (keeps prior reports intact)",
    )
    args = ap.parse_args()
    if not (args.build or args.analyze or args.sidecar):
        raise SystemExit("pass --build and/or --analyze and/or --sidecar")
    if args.build:
        build(args)
    if args.analyze:
        analyze(args)
    if args.sidecar:
        sidecar(args)


if __name__ == "__main__":
    main()
