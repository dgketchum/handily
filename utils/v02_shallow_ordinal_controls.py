"""Control baselines for the Handily v0.2 ordinal shallow-water probability head.

Work Package 5 (v2 plan §10). Establishes the bar the future ordinal shallow head
must beat for the monotone class targets

    P(WTD < 2 m) <= P(WTD < 5 m) <= P(WTD < 10 m).

Every control produces an out-of-fold (OOF) score/probability per class threshold
t in {2, 5, 10} m, where the observed class is ``y_t = mean_dtw < t`` (mean_dtw is
depth to the unconfined water table, m). Controls (v2 plan §10.2):

1. prevalence          -- per-fold constant equal to the training-folds prevalence.
2. logistic            -- LogisticRegression on the 30 frozen query features
                          (median-impute + standardize, fit on training folds only).
3. gbt                 -- HistGradientBoostingClassifier on the same features
                          (native NaN handling; no imputation).
4. fac_threshold       -- deterministic call ``fac_rem_dtw_m < t`` (0/1 where finite);
                          threshold-free PR-AUC uses score = -fac_rem_dtw_m.
5. gnn_point           -- deterministic call ``gnn_dtw_m < t``; PR-AUC uses -gnn_dtw_m.
6. gnn_laplace_cdf     -- P(DTW < t) as the Laplace CDF at t with loc=gnn_dtw_m,
                          scale=gnn_sigma_m. Honest probabilistic readout of the
                          current production model -- the strongest control.
7. gnn_sigma_selective -- ``gnn_dtw_m < t`` calls only where gnn_sigma_m <= population
                          median sigma; abstains elsewhere (coverage/abstention).

Discipline (v2 plan §5.3, §10.3, and CLAUDE.md):
- real wells only (is_water_pseudo == False), finite mean_dtw; sacrificial
  regional-holdout rows are dropped from every fit and report;
- never uses Ma (mean_dtw is the label, not Ma) or Janssen columns;
- headline metrics are threshold-free (PR-AUC, Brier skill); the p>=0.5 operating
  point is reported as descriptive only -- no threshold is selected on any geography.

Outputs to ``.../v02/wp5/``:
- shallow_controls_report.json   -- full stratified metric panel + definitions
- shallow_controls_oof.parquet   -- per-well OOF probabilities/scores per control
- shallow_controls_summary.md    -- plain-language PR-AUC / Brier-skill table + the bar
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v02_metrics import coverage_abstention, prob_class_panel  # noqa: E402

log = logging.getLogger("v02_shallow_ordinal_controls")

WTE_GNN = "/data/ssd2/handily/conus/wte_gnn"
CONTRACT_DIR = f"{WTE_GNN}/v02/contract"
OUT_DIR = f"{WTE_GNN}/v02/wp5"

THRESHOLDS_M = (2.0, 5.0, 10.0)
HUC2_MIN_WELLS = 500
DIST_TRAIN_BANDS_KM = [(0.0, 2.0), (2.0, 10.0), (10.0, np.inf)]

# probabilistic controls carry a calibrated P(class); score/selective controls do not
PROB_CONTROLS = ("prevalence", "logistic", "gbt", "gnn_laplace_cdf")


# --------------------------------------------------------------------------- #
# leaf statistical helpers (unit-tested)
# --------------------------------------------------------------------------- #
def laplace_cdf(t, loc, scale) -> np.ndarray:
    """Laplace CDF P(X < t) for X ~ Laplace(loc, scale>0), elementwise.

    F(t) = 0.5*exp((t-loc)/b)      for t <  loc
         = 1 - 0.5*exp(-(t-loc)/b) for t >= loc
    Dimensionless probability in [0, 1]; t/loc/scale share the target units (m).
    """
    z = (np.asarray(t, float) - np.asarray(loc, float)) / np.asarray(scale, float)
    return np.where(z < 0.0, 0.5 * np.exp(z), 1.0 - 0.5 * np.exp(-z))


def ordinal_monotonicity(p2, p5, p10, tol: float = 1e-9) -> dict:
    """Fraction of wells where p2 <= p5 <= p10 holds (within tol).

    frac_monotone is dimensionless in [0, 1]; a value < 1 flags non-nested class
    probabilities (the monotone-ordinal contract of v2 plan §10.1 is violated).
    """
    p2 = np.asarray(p2, float)
    p5 = np.asarray(p5, float)
    p10 = np.asarray(p10, float)
    ok = np.isfinite(p2) & np.isfinite(p5) & np.isfinite(p10)
    mono = (p2[ok] <= p5[ok] + tol) & (p5[ok] <= p10[ok] + tol)
    n = int(ok.sum())
    return {
        "n": n,
        "frac_monotone": round(float(mono.mean()), 6) if n else None,
        "n_violations": int((~mono).sum()),
    }


def prevalence_oof(y, folds) -> np.ndarray:
    """Per-fold constant prediction = prevalence of ``y`` over the OTHER folds.

    Honest OOF: a well's forecast never sees its own fold. Returns P in [0, 1].
    """
    y = np.asarray(y, float)
    folds = np.asarray(folds)
    out = np.full(len(y), np.nan)
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        out[te] = float(y[tr].mean()) if tr.any() else float(y.mean())
    return out


def _oof_predict(pipe_factory, X, y, folds) -> np.ndarray:
    """Generic OOF loop: fit ``pipe_factory()`` on train folds, predict on held fold."""
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    folds = np.asarray(folds)
    out = np.full(len(y), np.nan)
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        ytr = y[tr]
        if np.unique(ytr).size < 2:
            # degenerate training fold (one class only): fall back to prevalence.
            log.warning("fold %s single-class training set; prevalence fallback", f)
            out[te] = float(ytr.mean()) if len(ytr) else float(y.mean())
            continue
        model = pipe_factory()
        model.fit(X[tr], ytr)
        out[te] = model.predict_proba(X[te])[:, 1]
    return out


def logistic_oof(X, y, folds) -> np.ndarray:
    """OOF LogisticRegression probs; median-impute + standardize fit on train only."""
    return _oof_predict(
        lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(solver="lbfgs", max_iter=2000)),
            ]
        ),
        X,
        y,
        folds,
    )


def gbt_oof(X, y, folds) -> np.ndarray:
    """OOF HistGradientBoostingClassifier probs (default params; native NaN)."""
    return _oof_predict(
        lambda: HistGradientBoostingClassifier(random_state=0),
        X,
        y,
        folds,
    )


def reliability_table(p, y, n_bins: int = 10) -> list:
    """10-bin reliability table: mean predicted p vs observed class frequency.

    Each bin reports n (count), mean_pred (dimensionless), and obs_freq
    (dimensionless). A calibrated forecast has mean_pred ~ obs_freq per bin.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        out.append(
            {
                "bin": f"[{lo:.1f},{hi:.1f}{']' if i == n_bins - 1 else ')'}",
                "n": n,
                "mean_pred": round(float(p[m].mean()), 4) if n else None,
                "obs_freq": round(float(y[m].mean()), 4) if n else None,
            }
        )
    return out


def pr_at_op(p, y, op: float = 0.5) -> dict:
    """Precision/recall/confusion at a fixed operating point p >= op (descriptive).

    precision = TP/(TP+FP), recall = TP/(TP+FN) -- both dimensionless in [0, 1].
    """
    p = np.asarray(p, float)
    y = np.asarray(y, bool)
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    pred = p >= op
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    return {
        "n": int(len(p)),
        "op": op,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
    }


def score_class_panel(score, y) -> dict:
    """Threshold-free PR-AUC panel for a real-valued score (higher = more shallow).

    n / base_rate dimensionless-ish (count, fraction); pr_auc = average precision
    (dimensionless in [0, 1]; the prevalence-only baseline PR-AUC equals base_rate).
    """
    score = np.asarray(score, float)
    y = np.asarray(y, bool)
    ok = np.isfinite(score)
    score, y = score[ok], y[ok]
    n = len(score)
    if n == 0 or y.sum() in (0, n):
        return {"n": int(n), "degenerate": True}
    return {
        "n": int(n),
        "base_rate": round(float(y.mean()), 4),
        "pr_auc": round(float(average_precision_score(y, score)), 4),
    }


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_population(contract_dir: str, feature_cols: list) -> pd.DataFrame:
    """Load + join the frozen contract and restrict to the WP5 scoring population.

    Population = real wells (is_water_pseudo == False) with finite mean_dtw, minus
    sacrificial regional-holdout rows (hard project rule).
    """
    contract = Path(contract_dir)
    fb = contract / "frozen_baseline"

    q_cols = [
        "canonical_id",
        "mean_dtw",
        "cv_fold",
        "block_40km",
        "is_water_pseudo",
        "fac_rem_dtw_m",
        "huc2",
        *feature_cols,
    ]
    q = pd.read_parquet(fb / "query_nodes.parquet", columns=list(dict.fromkeys(q_cols)))
    oof = pd.read_parquet(
        fb / "gnn_oof_predictions.parquet",
        columns=["canonical_id", "gnn_dtw_m", "gnn_sigma_m", "is_water_pseudo"],
    )
    panels = pd.read_parquet(
        contract / "wells_panels.parquet",
        columns=[
            "canonical_id",
            "regional_holdout",
            "mech_shallow_irrigated",
            "mech_deep_far_arid",
            "dist_train_km",
        ],
    )

    n_all = len(q)
    q = q[~q["is_water_pseudo"].astype(bool)].copy()
    oof = oof[~oof["is_water_pseudo"].astype(bool)].drop(columns=["is_water_pseudo"])
    df = q.drop(columns=["is_water_pseudo"]).merge(
        oof, on="canonical_id", how="inner", validate="one_to_one"
    )
    df = df.merge(panels, on="canonical_id", how="inner", validate="one_to_one")

    n_real = len(df)
    df = df[np.isfinite(df["mean_dtw"])]
    n_finite = len(df)
    df = df[df["regional_holdout"] != "sacrificial"].reset_index(drop=True)
    log.info(
        "population: %d scoring wells (of %d rows; %d real, %d finite mean_dtw; "
        "sacrificial excluded)",
        len(df),
        n_all,
        n_real,
        n_finite,
    )
    return df


# --------------------------------------------------------------------------- #
# per-stratum metric panels
# --------------------------------------------------------------------------- #
def _prob_panel(p, y, mask) -> dict:
    return {
        "prob_panel": prob_class_panel(p[mask], y[mask]),
        "op_0p5": pr_at_op(p[mask], y[mask], 0.5),
    }


def _score_panel(score, call, y, mask, finite) -> dict:
    sel = mask & finite
    return {
        "coverage": coverage_abstention(int(mask.sum()), int(sel.sum())),
        "score_panel": score_class_panel(score[sel], y[sel]),
        "op_hardcall": pr_at_op(call[sel].astype(float), y[sel], 0.5),
    }


def _selective_panel(call, y, mask, covered) -> dict:
    sel = mask & covered
    return {
        "coverage": coverage_abstention(int(mask.sum()), int(sel.sum())),
        "op_hardcall_covered": pr_at_op(call[sel].astype(float), y[sel], 0.5),
    }


def stratum_panel(mask: np.ndarray, arr: dict) -> dict:
    """Full per-threshold control panel restricted to ``mask`` rows."""
    out = {"n": int(mask.sum())}
    for name in PROB_CONTROLS:
        out[name] = {
            f"t{int(t)}": _prob_panel(arr[f"{name}_p"][j], arr["y"][j], mask)
            for j, t in enumerate(THRESHOLDS_M)
        }
    out["fac_threshold"] = {
        f"t{int(t)}": _score_panel(
            arr["neg_fac"], arr["fac_call"][j], arr["y"][j], mask, arr["fac_finite"]
        )
        for j, t in enumerate(THRESHOLDS_M)
    }
    out["gnn_point"] = {
        f"t{int(t)}": _score_panel(
            arr["neg_gnn"], arr["gnn_call"][j], arr["y"][j], mask, arr["gnn_finite"]
        )
        for j, t in enumerate(THRESHOLDS_M)
    }
    out["gnn_sigma_selective"] = {
        f"t{int(t)}": _selective_panel(
            arr["gnn_call"][j], arr["y"][j], mask, arr["sigma_covered"]
        )
        for j, t in enumerate(THRESHOLDS_M)
    }
    return out


# --------------------------------------------------------------------------- #
# report assembly
# --------------------------------------------------------------------------- #
def build_strata(df: pd.DataFrame) -> dict:
    """Named boolean masks over the population (v2 plan §5.1 mechanism/region slices)."""
    n = len(df)
    strata = {"all": np.ones(n, bool)}
    strata["mech_shallow_irrigated"] = df["mech_shallow_irrigated"].to_numpy(bool)
    strata["mech_deep_far_arid"] = df["mech_deep_far_arid"].to_numpy(bool)

    huc2 = df["huc2"].astype(str).to_numpy()
    counts = pd.Series(huc2).value_counts()
    for h2 in sorted(counts[counts >= HUC2_MIN_WELLS].index):
        strata[f"huc2_{h2}"] = huc2 == h2

    d = df["dist_train_km"].to_numpy(float)
    for lo, hi in DIST_TRAIN_BANDS_KM:
        label = (
            f"dist_train_{lo:g}_{hi:g}km"
            if np.isfinite(hi)
            else f"dist_train_{lo:g}plus_km"
        )
        strata[label] = np.isfinite(d) & (d >= lo) & (d < hi)
    return strata


def definitions_block() -> dict:
    return {
        "target": "mean_dtw = observed depth to the unconfined water table (m).",
        "class_y_t": "y_t = (mean_dtw < t); binary shallow-water class at threshold t m.",
        "thresholds_m": list(THRESHOLDS_M),
        "population": (
            "real wells (is_water_pseudo == False) with finite mean_dtw, minus "
            "sacrificial regional-holdout rows (dropped from all fits and reports)."
        ),
        "oof": (
            "out-of-fold over cv_fold (8 HUC12-blocked folds); every model is fit on "
            "the 7 training folds and predicts the held fold. prevalence uses the "
            "training-folds class rate."
        ),
        "brier": "mean((p - y)^2); lower better; dimensionless (probability^2 space).",
        "brier_skill": (
            "1 - brier / brier_prevalence; dimensionless; positive = beats the "
            "prevalence-only forecast (brier_prevalence = base*(1-base))."
        ),
        "pr_auc": (
            "average precision (area under precision-recall curve); dimensionless in "
            "[0,1]; threshold-free; the no-skill baseline PR-AUC equals base_rate. For "
            "fac_threshold / gnn_point it is computed on a continuous score "
            "(-fac_rem_dtw_m / -gnn_dtw_m); higher score = more likely shallow."
        ),
        "base_rate": "observed prevalence of y_t on the stratum (dimensionless fraction).",
        "op_0p5": (
            "descriptive operating point at p >= 0.5: precision = TP/(TP+FP), "
            "recall = TP/(TP+FN) (dimensionless). NOT a selected threshold."
        ),
        "op_hardcall": "same precision/recall for a deterministic 0/1 call (fac/gnn < t).",
        "coverage_frac": (
            "n_scored / n_population on the stratum (dimensionless); abstention_frac = "
            "1 - coverage_frac. fac_threshold abstains where fac_rem_dtw_m is NaN; "
            "gnn_sigma_selective abstains where gnn_sigma_m > population-median sigma."
        ),
        "reliability_table": (
            "10 equal-width predicted-probability bins; mean_pred vs obs_freq per bin "
            "(both dimensionless); calibration diagnostic for probabilistic controls."
        ),
        "ordinal_monotonicity": (
            "fraction of wells with p2 <= p5 <= p10 within 1e-9 (dimensionless); the "
            "monotone-ordinal contract of v2 plan §10.1."
        ),
        "sigma_median_m": "population median of gnn_sigma_m (Laplace scale, m).",
        "rules": (
            "no threshold selected on any geography (headline = threshold-free PR-AUC / "
            "Brier skill); no Ma or Janssen columns used; sacrificial rows dropped."
        ),
    }


def compute_controls(df: pd.DataFrame, feature_cols: list) -> dict:
    """Compute every control's OOF array over the population, keyed for reuse."""
    y = df["mean_dtw"].to_numpy(float)
    folds = df["cv_fold"].to_numpy()
    X = df[feature_cols].to_numpy(float)
    fac = df["fac_rem_dtw_m"].to_numpy(float)
    gnn = df["gnn_dtw_m"].to_numpy(float)
    sigma = df["gnn_sigma_m"].to_numpy(float)
    sigma_med = float(np.median(sigma[np.isfinite(sigma)]))

    arr = {
        "y": [y < t for t in THRESHOLDS_M],
        "neg_fac": -fac,
        "neg_gnn": -gnn,
        "fac_finite": np.isfinite(fac),
        "gnn_finite": np.isfinite(gnn),
        "fac_call": [np.where(np.isfinite(fac), fac < t, np.nan) for t in THRESHOLDS_M],
        "gnn_call": [np.where(np.isfinite(gnn), gnn < t, np.nan) for t in THRESHOLDS_M],
        "sigma_covered": np.isfinite(sigma) & (sigma <= sigma_med),
        "sigma_median_m": sigma_med,
    }

    for j, t in enumerate(THRESHOLDS_M):
        yt = arr["y"][j].astype(int)
        log.info("fitting controls for threshold <%g m (prevalence %.4f)", t, yt.mean())
        arr[f"prevalence_p_{j}"] = prevalence_oof(yt, folds)
        arr[f"logistic_p_{j}"] = logistic_oof(X, yt, folds)
        arr[f"gbt_p_{j}"] = gbt_oof(X, yt, folds)
        arr[f"gnn_laplace_cdf_p_{j}"] = laplace_cdf(t, gnn, sigma)

    # index probabilistic controls as name_p -> list over thresholds
    for name in PROB_CONTROLS:
        arr[f"{name}_p"] = [arr[f"{name}_p_{j}"] for j in range(len(THRESHOLDS_M))]
    return arr


def oof_frame(df: pd.DataFrame, arr: dict) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "canonical_id": df["canonical_id"].to_numpy(),
            "cv_fold": df["cv_fold"].to_numpy(),
            "mean_dtw": df["mean_dtw"].to_numpy(float),
            "gnn_sigma_m": df["gnn_sigma_m"].to_numpy(float),
            "sigma_covered": arr["sigma_covered"],
            "score_neg_fac_rem": arr["neg_fac"],
            "score_neg_gnn": arr["neg_gnn"],
        }
    )
    for j, t in enumerate(THRESHOLDS_M):
        ti = int(t)
        out[f"y_lt{ti}"] = arr["y"][j]
        for name in PROB_CONTROLS:
            out[f"p_{name}_t{ti}"] = arr[f"{name}_p"][j]
        out[f"fac_call_t{ti}"] = arr["fac_call"][j]
        out[f"gnn_point_call_t{ti}"] = arr["gnn_call"][j]
        out[f"gnn_sigma_selective_call_t{ti}"] = np.where(
            arr["sigma_covered"], arr["gnn_call"][j], np.nan
        )
    return out


# --------------------------------------------------------------------------- #
# markdown summary
# --------------------------------------------------------------------------- #
def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x is not None else "-"


def write_summary(report: dict, path: Path) -> None:
    strata = report["strata"]
    allw = strata["all"]
    lines = [
        "# WP5 shallow-water ordinal controls -- summary",
        "",
        f"Population: {report['population']['n_scoring_wells']} real wells "
        f"(sacrificial dropped; finite mean_dtw). OOF over "
        f"{report['population']['n_folds']} HUC12-blocked folds. "
        f"Population-median gnn_sigma = {report['sigma_median_m']:.3f} m.",
        "",
        "All numbers are on the **all-wells** stratum. PR-AUC and Brier skill are "
        "threshold-free / calibration headline metrics (dimensionless). The p>=0.5 "
        "operating point in the JSON is descriptive only -- no threshold is selected "
        "on any geography (v2 plan §10.3).",
        "",
    ]
    control_order = [
        "prevalence",
        "logistic",
        "gbt",
        "fac_threshold",
        "gnn_point",
        "gnn_laplace_cdf",
        "gnn_sigma_selective",
    ]
    bars = {}
    for t in THRESHOLDS_M:
        ti = int(t)
        key = f"t{ti}"
        base = allw["prevalence"][key]["prob_panel"]["base_rate"]
        lines += [
            f"## P(WTD < {ti} m)  (base rate = {base:.4f})",
            "",
            "| control | PR-AUC | Brier skill | note |",
            "|---|---|---|---|",
        ]
        best_name, best_pr = None, -np.inf
        for name in control_order:
            node = allw[name][key]
            note = ""
            if name in PROB_CONTROLS:
                pa = node["prob_panel"]
                pr = pa.get("pr_auc")
                bs = pa.get("brier_skill")
                note = "probabilistic"
            elif name in ("fac_threshold", "gnn_point"):
                pr = node["score_panel"].get("pr_auc")
                bs = None
                note = f"score-based; coverage {node['coverage']['coverage_frac']:.3f}"
            else:  # gnn_sigma_selective
                pr = None
                bs = None
                op = node["op_hardcall_covered"]
                note = (
                    f"selective; coverage {node['coverage']['coverage_frac']:.3f}; "
                    f"prec {_fmt(op['precision'])} rec {_fmt(op['recall'])}"
                )
            lines.append(f"| {name} | {_fmt(pr)} | {_fmt(bs)} | {note} |")
            if isinstance(pr, (int, float)) and pr is not None and pr > best_pr:
                best_name, best_pr = name, pr
        bars[ti] = (best_name, best_pr)
        lc = allw["gnn_laplace_cdf"][key]["prob_panel"]
        lines += [
            "",
            f"gnn_laplace_cdf (honest production readout): PR-AUC "
            f"{_fmt(lc.get('pr_auc'))}, Brier skill {_fmt(lc.get('brier_skill'))}, "
            f"Brier {_fmt(lc.get('brier'), 5)}.",
            "",
        ]

    lines += ["## The bar (v2 plan §10.3)", ""]
    for t in THRESHOLDS_M:
        ti = int(t)
        name, pr = bars[ti]
        lines.append(
            f"- **P(WTD < {ti} m):** highest control PR-AUC is **{name}** "
            f"(PR-AUC {pr:.4f}). The future GNN ordinal head must beat this "
            f"threshold-free PR-AUC and the gnn_laplace_cdf Brier skill on the "
            f"independent shallow panels to clear the WP5 gate."
        )
    lines += [
        "",
        "gnn_laplace_cdf is the honest probabilistic readout of the current "
        "production model (Laplace CDF of the point+sigma head) and is the "
        "probabilistic control of record: the ordinal head's calibrated Brier "
        "skill must exceed it, not merely a tabular or deterministic baseline. "
        "Coverage/abstention is reported for fac_threshold and gnn_sigma_selective; "
        "the continuous WTD background is retained unchanged if a class head wins.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract-dir", default=CONTRACT_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    manifest = json.loads(
        (
            Path(args.contract_dir) / "frozen_baseline" / "graph_manifest.json"
        ).read_text()
    )
    feature_cols = list(manifest["query_feature_cols"])
    log.info("%d frozen query features", len(feature_cols))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_population(args.contract_dir, feature_cols)
    arr = compute_controls(df, feature_cols)
    strata = build_strata(df)

    report = {
        "work_package": "WP5 shallow-water ordinal controls (v2 plan §10)",
        "definitions": definitions_block(),
        "population": {
            "n_scoring_wells": int(len(df)),
            "n_folds": int(df["cv_fold"].nunique()),
            "prevalence_by_threshold": {
                f"t{int(t)}": round(float(arr["y"][j].mean()), 4)
                for j, t in enumerate(THRESHOLDS_M)
            },
        },
        "sigma_median_m": round(arr["sigma_median_m"], 4),
        "feature_cols": feature_cols,
        "strata": {name: stratum_panel(m, arr) for name, m in strata.items()},
        "reliability": {
            name: {
                f"t{int(t)}": reliability_table(arr[f"{name}_p"][j], arr["y"][j])
                for j, t in enumerate(THRESHOLDS_M)
            }
            for name in PROB_CONTROLS
        },
        "ordinal_monotonicity": {
            name: ordinal_monotonicity(
                arr[f"{name}_p"][0], arr[f"{name}_p"][1], arr[f"{name}_p"][2]
            )
            for name in PROB_CONTROLS
        },
    }
    # hard-call controls nest trivially; record it for completeness
    report["ordinal_monotonicity"]["fac_threshold_call"] = ordinal_monotonicity(
        *[arr["fac_call"][j] for j in range(3)]
    )
    report["ordinal_monotonicity"]["gnn_point_call"] = ordinal_monotonicity(
        *[arr["gnn_call"][j] for j in range(3)]
    )

    report_path = out / "shallow_controls_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float))
    oof_path = out / "shallow_controls_oof.parquet"
    oof_frame(df, arr).to_parquet(oof_path, index=False)
    md_path = out / "shallow_controls_summary.md"
    write_summary(report, md_path)

    log.info("wrote %s", report_path)
    log.info("wrote %s", oof_path)
    log.info("wrote %s", md_path)


if __name__ == "__main__":
    main()
