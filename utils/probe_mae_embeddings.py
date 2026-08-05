"""Frozen-fold probe of MAE neighborhood embeddings for DTW skill.

Answers the falsifiable question (notes/MAE_NEIGHBORHOOD_EMBEDDING.md): does a learned
neighborhood embedding add DTW skill beyond the same rasters point-sampled at the pixel?

For each arm, a HistGradientBoosting probe predicts DTW (mean_dtw, m) at the real
monitoring wells under HUC4-blocked GroupKFold (locked HUC4s 0707/1019/1605 EXCLUDED --
never scored), for three feature sets:
  (a) MAE embedding alone,
  (b) the same MAE-stack channels point-sampled at the center pixel,
  (c) embedding + point covariates.

Emits the full metric panel (MAD, mean bias, median residual, RMSE; depth-banded with
n per cell; meters) per (arm, featset), plus a paired-bootstrap CI on the (c)-(b) MAD
delta -- the decision statistic.

    uv run python utils/probe_mae_embeddings.py \
        --bundle /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water \
        --emb s100=/data/ssd2/handily/conus/mae/embeddings/mae_embeddings_s100.parquet \
        --emb pyr=/data/ssd2/handily/conus/mae/embeddings/mae_embeddings_pyr.parquet \
        --out-dir /data/ssd2/handily/conus/mae/probes/ladder
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
# module import (not from-import) so a --manifest rebind of the channel roster is seen
import build_mae_patches as bmp  # noqa: E402
from build_mae_patches import load_channel_full, snap_to_lattice  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("probe_mae_embeddings")

LOCKED_HUC4 = {"0707", "1019", "1605"}
DEPTH_BANDS = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
# Wide-basin slice: arid West basins wide enough (50-150 km) that only the 4 km / 256 km
# rung can span them. HUC2 13 = Rio Grande (NM rift), 15 = Lower Colorado, 16 = Great
# Basin (NV closed basins). Used to test whether the wide level helps where expected.
WIDE_BASIN_HUC2 = {"13", "15", "16"}
# NV / Great Basin closed-basin slice (HUC2 == 16), the regime where the MAE arm's one
# genuine GNN win landed -- carried here so AEF is judged on the same slice.
NV_HUC2 = {"16"}


def sample_point_covariates(x5070, y5070) -> pd.DataFrame:
    """The MAE-stack channels point-sampled at the well center pixel (baseline b).

    dem_rel is EXCLUDED: its single-pixel value is 0 by construction (center-relative),
    and the underlying absolute elevation is barred as a point feature by the
    no-absolute-elevation policy -- so neighborhood relief is available ONLY via the
    embedding. That asymmetry is the point of the falsifiable test."""
    col, row = snap_to_lattice(x5070, y5070)
    out = {}
    for name, path, tr in bmp.CHANNEL_SPECS:
        if tr == "dem_rel":
            continue
        full, _ = load_channel_full(path, tr)
        out[f"pt_{name}"] = full[row, col].astype("float32")
        del full
    return pd.DataFrame(out)


def oof_predict(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
):
    """Out-of-fold HistGBT predictions under GroupKFold(huc4)."""
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=seed,
        )
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def panel(obs: np.ndarray, pred: np.ndarray) -> dict:
    """MAD / mean bias / median residual / RMSE overall + per depth band (meters)."""
    r = pred - obs

    def cell(mask):
        if mask.sum() == 0:
            return {"n": 0}
        rr = r[mask]
        return {
            "n": int(mask.sum()),
            "MAD_m": float(np.median(np.abs(rr))),
            "bias_m": float(np.mean(rr)),
            "median_resid_m": float(np.median(rr)),
            "RMSE_m": float(np.sqrt(np.mean(rr**2))),
        }

    out = {"overall": cell(np.ones(len(obs), bool))}
    for lo, hi in DEPTH_BANDS:
        out[f"{lo}-{hi}m"] = cell((obs >= lo) & (obs < hi))
    return out


def paired_bootstrap_mad_delta(obs, pred_b, pred_c, groups, n_boot=1000, seed=0):
    """CI on MAD(c)-MAD(b) by resampling HUC4 groups (block bootstrap). Negative = c better."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    ab = np.abs(pred_b - obs)
    ac = np.abs(pred_c - obs)
    deltas = []
    gidx = {g: np.where(groups == g)[0] for g in ug}
    for _ in range(n_boot):
        samp = np.concatenate([gidx[g] for g in rng.choice(ug, len(ug), replace=True)])
        deltas.append(np.median(ac[samp]) - np.median(ab[samp]))
    d = np.array(deltas)
    return {
        "delta_mad_c_minus_b_m": float(np.median(ac) - np.median(ab)),
        "ci95_low_m": float(np.percentile(d, 2.5)),
        "ci95_high_m": float(np.percentile(d, 97.5)),
        "frac_c_better": float(np.mean(d < 0)),
    }


# Cross-arm contrasts: MAD(y) - MAD(x) on the common well set (negative => y better),
# block-bootstrap CI95 over HUC4 groups, for EVERY pair of arms present and both featsets
# (a = emb-only, c = emb+point). Direction: y = the later arm in CONTRAST_ORDER, arms not
# listed there rank after all listed ones in --emb order -- so the historical trio keeps
# its labels (aef_vs_mae, aefmae_vs_aef, aefmae_vs_mae) and a new candidate arm is always
# judged as y against every incumbent x.
CONTRAST_ORDER = ["mae", "aef", "aefmae"]


def build_contrasts(obs, preds_a, preds_c, groups, seed) -> dict:
    """Paired block-bootstrap MAD(y)-MAD(x) for each present arm pair (negative => y better)."""
    arms = list(preds_a)  # insertion order = --emb order

    def rank(a):
        if a in CONTRAST_ORDER:
            return (0, CONTRAST_ORDER.index(a))
        return (1, arms.index(a))

    src = {"a": preds_a, "c": preds_c}
    out = {}
    for i, p in enumerate(arms):
        for q in arms[i + 1 :]:
            xa, ya = sorted((p, q), key=rank)
            for feat, tag in (("a", "embonly"), ("c", "withpoint")):
                if ya in src[feat] and xa in src[feat]:
                    d = paired_bootstrap_mad_delta(
                        obs, src[feat][xa], src[feat][ya], groups, seed=seed
                    )
                    out[f"{ya}_vs_{xa}_{tag}"] = {
                        "delta_mad_y_minus_x_m": d["delta_mad_c_minus_b_m"],
                        "ci95_low_m": d["ci95_low_m"],
                        "ci95_high_m": d["ci95_high_m"],
                        "frac_y_better": d["frac_c_better"],
                    }
    return out


def slice_c_vs_b(mask, obs, pred_c, pred_pt) -> dict:
    """Per-slice emb+point panel and its MAD(c)-MAD(b) delta on the slice rows."""
    c = panel(obs[mask], pred_c[mask]) if mask.any() else {"n": 0}
    d = (
        float(
            np.median(np.abs(pred_c[mask] - obs[mask]))
            - np.median(np.abs(pred_pt[mask] - obs[mask]))
        )
        if mask.any()
        else None
    )
    return {"c_embedding_plus_point": c, "delta_mad_c_minus_b_m": d}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument(
        "--emb", action="append", required=True, help="arm=path.parquet (repeatable)"
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--manifest",
        default=None,
        help="channel-manifest JSON rebinding the roster/paths "
        "(build_mae_patches.apply_manifest). REQUIRED for a fair (c)-(b) gate on an "
        "embedding trained from a manifest roster: baseline b must point-sample the "
        "SAME channels the embedding saw, else the embedding gets credit for merely "
        "carrying covariates absent from b.",
    )
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.manifest:
        bmp.apply_manifest(json.loads(Path(args.manifest).read_text()))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qn = pd.read_parquet(
        Path(args.bundle) / "query_nodes.parquet",
        columns=[
            "query_node_idx",
            "canonical_id",
            "x5070",
            "y5070",
            "huc4",
            "mean_dtw",
            "is_water_pseudo",
            "confinement_class",
            "well_class",
        ],
    )
    w = qn[~qn["is_water_pseudo"].astype(bool)].copy()
    w = w[w["mean_dtw"].notna() & w["huc4"].notna()]
    w = w[~w["huc4"].astype(str).isin(LOCKED_HUC4)]  # never score locked HUC4s
    # unconfined monitoring only (bundle is already screened, assert defensively)
    if "confinement_class" in w:
        w = w[w["confinement_class"].isin(["unconfined", "unconfined_marginal"])]
    w = w.reset_index(drop=True)
    log.info(
        "scoring population: %d real unconfined wells (locked HUC4s excluded)", len(w)
    )

    log.info("sampling point covariates at well pixels")
    pt = sample_point_covariates(
        w["x5070"].to_numpy("float64"), w["y5070"].to_numpy("float64")
    )
    Xpt = pt.to_numpy("float32")
    obs = w["mean_dtw"].to_numpy("float64")
    groups = w["huc4"].astype(str).to_numpy()
    wide = np.array(
        [g[:2] in WIDE_BASIN_HUC2 for g in groups]
    )  # wide-basin arid-West slice
    nv = np.array([g[:2] in NV_HUC2 for g in groups])  # NV/Great Basin closed basins
    log.info(
        "wide-basin slice: %d wells (HUC2 %s); NV slice: %d wells (HUC2 %s)",
        int(wide.sum()),
        sorted(WIDE_BASIN_HUC2),
        int(nv.sum()),
        sorted(NV_HUC2),
    )

    # point-only baseline (shared across arms)
    pred_pt = oof_predict(Xpt, obs, groups, args.n_splits, args.seed)

    report = {
        "population_n": int(len(w)),
        "n_splits": args.n_splits,
        "locked_huc4_excluded": sorted(LOCKED_HUC4),
        "manifest": args.manifest,
        "point_covariates": list(pt.columns),
        "baseline_b_point_only": panel(obs, pred_pt),
        "wide_basin_slice": {
            "huc2": sorted(WIDE_BASIN_HUC2),
            "n": int(wide.sum()),
            "baseline_b_point_only": panel(obs[wide], pred_pt[wide])
            if wide.any()
            else {"n": 0},
        },
        "nv_slice": {
            "huc2": sorted(NV_HUC2),
            "n": int(nv.sum()),
            "baseline_b_point_only": panel(obs[nv], pred_pt[nv])
            if nv.any()
            else {"n": 0},
        },
        "arms": {},
    }
    residuals = w[["query_node_idx", "canonical_id", "huc4", "mean_dtw"]].copy()
    residuals["pred_b_point"] = pred_pt
    preds_a: dict[str, np.ndarray] = {}
    preds_c: dict[str, np.ndarray] = {}

    for spec in args.emb:
        arm, path = spec.split("=", 1)
        e = pd.read_parquet(path)
        ecols = [c for c in e.columns if c.startswith("mae_")]
        m = w.merge(e[["query_node_idx"] + ecols], on="query_node_idx", how="left")
        if m[ecols].isna().any().any():
            n_missing = int(m[ecols].isna().any(axis=1).sum())
            raise SystemExit(
                f"arm {arm}: {n_missing} wells missing embeddings -- extractor incomplete"
            )
        Xe = m[ecols].to_numpy("float32")
        Xc = np.concatenate([Xe, Xpt], axis=1)
        pred_a = oof_predict(Xe, obs, groups, args.n_splits, args.seed)
        pred_c = oof_predict(Xc, obs, groups, args.n_splits, args.seed)
        preds_a[arm] = pred_a
        preds_c[arm] = pred_c
        report["arms"][arm] = {
            "emb_dim": len(ecols),
            "a_embedding_only": panel(obs, pred_a),
            "c_embedding_plus_point": panel(obs, pred_c),
            "decision_c_vs_b": paired_bootstrap_mad_delta(
                obs, pred_pt, pred_c, groups, seed=args.seed
            ),
            "wide_basin": slice_c_vs_b(wide, obs, pred_c, pred_pt),
            "nv": slice_c_vs_b(nv, obs, pred_c, pred_pt),
        }
        residuals[f"pred_a_{arm}"] = pred_a
        residuals[f"pred_c_{arm}"] = pred_c
        log.info(
            "arm=%s: MAD b=%.3f a=%.3f c=%.3f (delta c-b %.3f)",
            arm,
            report["baseline_b_point_only"]["overall"]["MAD_m"],
            report["arms"][arm]["a_embedding_only"]["overall"]["MAD_m"],
            report["arms"][arm]["c_embedding_plus_point"]["overall"]["MAD_m"],
            report["arms"][arm]["decision_c_vs_b"]["delta_mad_c_minus_b_m"],
        )

    report["contrasts"] = build_contrasts(obs, preds_a, preds_c, groups, args.seed)
    for label, c in report["contrasts"].items():
        log.info(
            "contrast %s: delta MAD(y-x) %.3f m CI95 [%.3f, %.3f] frac_y_better %.2f",
            label,
            c["delta_mad_y_minus_x_m"],
            c["ci95_low_m"],
            c["ci95_high_m"],
            c["frac_y_better"],
        )

    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
    residuals.to_parquet(out_dir / "probe_residuals.parquet")
    _write_summary(out_dir / "probe_summary.md", report)
    log.info("probe report -> %s", out_dir / "probe_report.json")


def _write_summary(path: Path, rep: dict) -> None:
    b = rep["baseline_b_point_only"]["overall"]
    lines = [
        "# MAE embedding probe -- DTW skill (meters)",
        "",
        f"Population: {rep['population_n']} real unconfined wells, HUC4-blocked "
        f"{rep['n_splits']}-fold, locked HUC4s {rep['locked_huc4_excluded']} excluded.",
        "",
        "## Overall MAD / bias / median-resid / RMSE (m)",
        "",
        "| arm | featset | MAD | bias | med_resid | RMSE | n |",
        "|---|---|---|---|---|---|---|",
        f"| (baseline) | b: point-only | {b['MAD_m']:.3f} | {b['bias_m']:.3f} | "
        f"{b['median_resid_m']:.3f} | {b['RMSE_m']:.3f} | {b['n']} |",
    ]
    for arm, a in rep["arms"].items():
        for tag, key in [
            ("a: emb-only", "a_embedding_only"),
            ("c: emb+point", "c_embedding_plus_point"),
        ]:
            o = a[key]["overall"]
            lines.append(
                f"| {arm} | {tag} | {o['MAD_m']:.3f} | {o['bias_m']:.3f} | "
                f"{o['median_resid_m']:.3f} | {o['RMSE_m']:.3f} | {o['n']} |"
            )
    lines += [
        "",
        "## Decision statistic: MAD(c) - MAD(b), block-bootstrap 95% CI",
        "",
        "| arm | delta MAD (m) | CI95 | frac c better |",
        "|---|---|---|---|",
    ]
    for arm, a in rep["arms"].items():
        d = a["decision_c_vs_b"]
        lines.append(
            f"| {arm} | {d['delta_mad_c_minus_b_m']:.3f} | "
            f"[{d['ci95_low_m']:.3f}, {d['ci95_high_m']:.3f}] | {d['frac_c_better']:.2f} |"
        )
    lines += ["", "## Depth-banded MAD (m) / n -- c: emb+point vs b: point-only", ""]
    bands = [f"{lo}-{hi}m" for lo, hi in DEPTH_BANDS]
    lines += ["| arm | " + " | ".join(bands) + " |", "|---|" + "---|" * len(bands)]
    bb = rep["baseline_b_point_only"]
    lines.append(
        "| b:point | "
        + " | ".join(
            f"{bb[bd]['MAD_m']:.2f} (n{bb[bd]['n']})" if bb[bd]["n"] else "-"
            for bd in bands
        )
        + " |"
    )
    for arm, a in rep["arms"].items():
        c = a["c_embedding_plus_point"]
        lines.append(
            f"| c:{arm} | "
            + " | ".join(
                f"{c[bd]['MAD_m']:.2f} (n{c[bd]['n']})" if c[bd]["n"] else "-"
                for bd in bands
            )
            + " |"
        )

    def _slice_section(slice_key: str, arm_key: str, title: str) -> None:
        ws = rep.get(slice_key)
        if not (ws and ws.get("n")):
            return
        # the slice stores full panels; the summary shows the overall cell only
        wbp = ws["baseline_b_point_only"]
        wb = wbp.get("overall", wbp)
        lines.extend(
            [
                "",
                f"## {title} (HUC2 {ws['huc2']}, n={ws['n']}) -- MAD (m), c: emb+point",
                "",
                "| arm | featset | MAD | bias | RMSE | delta MAD c-b |",
                "|---|---|---|---|---|---|",
                f"| (baseline) | b: point-only | {wb['MAD_m']:.3f} | {wb['bias_m']:.3f} | "
                f"{wb['RMSE_m']:.3f} | - |",
            ]
        )
        for arm, a in rep["arms"].items():
            wcp = a.get(arm_key, {}).get("c_embedding_plus_point", {})
            wc = wcp.get("overall", wcp)
            wd = a.get(arm_key, {}).get("delta_mad_c_minus_b_m")
            if wc.get("n"):
                lines.append(
                    f"| {arm} | c: emb+point | {wc['MAD_m']:.3f} | {wc['bias_m']:.3f} | "
                    f"{wc['RMSE_m']:.3f} | {wd:+.3f} |"
                )

    _slice_section("wide_basin_slice", "wide_basin", "Wide-basin slice")
    _slice_section("nv_slice", "nv", "NV closed-basin slice")

    contrasts = rep.get("contrasts")
    if contrasts:
        lines += [
            "",
            "## Cross-arm contrasts: MAD(y) - MAD(x), block-bootstrap 95% CI "
            "(negative => y better)",
            "",
            "| contrast (y vs x) | delta MAD (m) | CI95 | frac y better |",
            "|---|---|---|---|",
        ]
        for label, c in contrasts.items():
            lines.append(
                f"| {label} | {c['delta_mad_y_minus_x_m']:.3f} | "
                f"[{c['ci95_low_m']:.3f}, {c['ci95_high_m']:.3f}] | {c['frac_y_better']:.2f} |"
            )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
