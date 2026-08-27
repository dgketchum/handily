"""Warm eval + source-density response curve for --source-obs arms (rung 0+).

Executes layers 5.2 and 5.3 of ``notes/SOURCE_WELL_ASSIMILATION_PLAN.md`` against
a trained source arm: inside each held-out HUC4 fold, a seeded frozen SITE-level
partition turns a fraction p of the fold's wells into visible sources (their obs
enters the forward through the src feature block) and scores the remaining
target sites. Fractions are NESTED under one site permutation, and the scored
target set is FIXED at the max fraction's complement, so every point on the
curve scores the identical wells and the deltas are honestly paired.

Comparators on the same target rows:
- **baseline arm OOF** (--baseline-dir, the no-source twin) -- gates A1/A2;
- **fixed relief-IDW of the SAME visible sources' obs WTE** (bundle idw_k /
  idw_power / r_relief_vw params) -- gate A3: if the learned assimilation cannot
  beat fixed IDW given identical wells, NO-GO.

p=0 replays the trainer's final-forward semantics (sources = out-of-fold pool
only), so it doubles as an OOF round-trip check against the arm's archived OOF
and as gate A2's regression probe.

Discipline (matches the v0.2 scorers): real wells only, UNCONFINED water-table
wells only in the scored set, sacrificial HUC4s dropped, full metric panel
(never MAD alone), paired HUC8-block bootstrap CIs, units on every number.
Site = x/y rounded to 1 m (collocated wells share a site and are never split
across source/target -- the ~4-5 m dispersion floor would leak).

Usage:
    uv run python utils/eval_source_assimilation.py \\
        --arm-dir /data/ssd2/handily/conus/wte_gnn/gnn_conus_monitoring_water_src_r0 \\
        --baseline-dir /data/ssd2/handily/conus/wte_gnn/gnn_conus_monitoring_water_src_base
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conus_graph_inputs import _relief_coords  # noqa: E402
from infer_conus_gnn import (  # noqa: E402
    build_anchors,
    build_model,
    bundle_source_edges,
    ckpt_f_src,
    ckpt_f_srcedge,
    fold_tensors,
    forward_fold,
    load_models,
)
from predict_gnn_at_points import bundle_graph  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_source_assimilation")

SACRIFICIAL_HUC4 = ("0707", "1019", "1605")
UNCONFINED = ("unconfined", "unconfined_marginal")
DEPTH_BANDS = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
DIST_BANDS_KM = [(0, 0.3), (0.3, 1), (1, 5), (5, 20), (20, np.inf)]


def core(err: np.ndarray) -> dict:
    """MAD / median residual / mean bias / RMSE / p95|err|, all metres."""
    e = err[np.isfinite(err)]
    if not len(e):
        return {"n": 0}
    return {
        "n": int(len(e)),
        "mad_m": float(np.median(np.abs(e))),
        "med_resid_m": float(np.median(e)),
        "bias_m": float(np.mean(e)),
        "rmse_m": float(np.sqrt(np.mean(e**2))),
        "p95_abs_m": float(np.percentile(np.abs(e), 95)),
    }


def paired_block_delta(
    err_a: np.ndarray,
    err_b: np.ndarray,
    blocks: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Delta = metric(b) - metric(a) on the same rows (+ = a better), with a
    HUC8-block percentile bootstrap CI95 on the MAD and RMSE deltas."""
    ok = np.isfinite(err_a) & np.isfinite(err_b)
    ea, eb, bl = np.abs(err_a[ok]), np.abs(err_b[ok]), blocks[ok]
    if not len(ea):
        return {"n": 0}
    uniq, inv = np.unique(bl, return_inverse=True)
    rng = np.random.default_rng(seed)
    idx_by_block = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    d_mad = np.empty(n_boot)
    d_rmse = np.empty(n_boot)
    for i in range(n_boot):
        rows = np.concatenate(
            [idx_by_block[j] for j in rng.integers(0, len(uniq), len(uniq))]
        )
        d_mad[i] = np.median(eb[rows]) - np.median(ea[rows])
        d_rmse[i] = np.sqrt(np.mean(eb[rows] ** 2)) - np.sqrt(np.mean(ea[rows] ** 2))
    return {
        "n": int(len(ea)),
        "n_blocks": int(len(uniq)),
        "d_mad_m": float(np.median(np.abs(eb)) - np.median(np.abs(ea))),
        "d_mad_ci95": [float(q) for q in np.percentile(d_mad, [2.5, 97.5])],
        "d_rmse_m": float(np.sqrt(np.mean(eb**2)) - np.sqrt(np.mean(ea**2))),
        "d_rmse_ci95": [float(q) for q in np.percentile(d_rmse, [2.5, 97.5])],
    }


def banded(err: np.ndarray, by: np.ndarray, bands) -> dict:
    out = {}
    for lo, hi in bands:
        m = np.isfinite(by) & (by >= lo) & (by < hi)
        lab = f"{lo}-{hi}" if np.isfinite(hi) else f"{lo}+"
        out[lab] = core(err[m])
    return out


def banded_delta(err_a, err_b, by, bands, blocks, seed=0) -> dict:
    out = {}
    for lo, hi in bands:
        m = np.isfinite(by) & (by >= lo) & (by < hi)
        lab = f"{lo}-{hi}" if np.isfinite(hi) else f"{lo}+"
        out[lab] = paired_block_delta(err_a[m], err_b[m], blocks[m], seed=seed)
    return out


def idw_same_wells(
    src_xy: np.ndarray,
    src_z: np.ndarray,
    src_wte: np.ndarray,
    tgt_xy: np.ndarray,
    tgt_z: np.ndarray,
    vw: float,
    k: int,
    power: float,
) -> np.ndarray:
    """Fixed relief-kNN IDW of the visible sources' obs WTE at the targets --
    the gate-A3 comparator (crossfit R's rule, given the identical well set)."""
    pool = _relief_coords(src_xy, src_z, vw)
    q = _relief_coords(tgt_xy, tgt_z, vw)
    kk = min(k, len(src_xy))
    d, i = cKDTree(pool).query(q, k=kk, workers=-1)
    d = np.atleast_2d(d)
    i = np.atleast_2d(i)
    w = 1.0 / np.maximum(d, 1e-6) ** power
    return (w * src_wte[i]).sum(1) / w.sum(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", required=True, help="--source-obs arm (models/)")
    ap.add_argument(
        "--baseline-dir", required=True, help="no-source twin arm (OOF parquet)"
    )
    ap.add_argument("--fracs", default="0,0.25,0.5,0.75")
    ap.add_argument("--seed", type=int, default=0, help="frozen site-partition seed")
    ap.add_argument(
        "--out-dir", default=None, help="default <arm-dir>/source_warm_eval"
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--oof-tol-m",
        type=float,
        default=1e-3,
        help="p=0 round-trip tolerance vs the arm's archived OOF",
    )
    args = ap.parse_args()
    fracs = sorted(float(f) for f in args.fracs.split(","))
    if fracs[0] != 0.0 or fracs[-1] >= 1.0:
        raise SystemExit("--fracs must start at 0 (gate A2) and stay < 1")

    arm_dir = Path(args.arm_dir)
    out_dir = Path(args.out_dir) if args.out_dir else arm_dir / "source_warm_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = load_models(arm_dir)
    man = models["manifest"]
    f_src = ckpt_f_src(models["folds"][0], man["feature_dims"]["query"])
    f_srcedge = ckpt_f_srcedge(models["folds"][0])
    if f_src is None and f_srcedge is None:
        raise SystemExit(f"{arm_dir} is not a source arm (no src block / src edges)")

    graph, qn, f_mae, writeback = bundle_graph(models, args.device, None)
    gdir = Path(man["graph_dir"])
    bman = json.loads((gdir / "graph_manifest.json").read_text())
    vw = float(bman["r_relief_vw"])
    idw_k, idw_p = int(bman["idw_k"]), float(bman["idw_power"])

    xy = qn[["x5070", "y5070"]].to_numpy("float64")
    z_surf = qn["z_surf_well_m"].to_numpy("float64")
    wte_obs = qn["wte_obs_m"].to_numpy("float64")
    obs_dtw = z_surf - wte_obs
    resid = qn["wte_residual_m"].to_numpy("float64")
    r_wte = qn["regional_wte_idw_oof_m"].to_numpy("float64")
    fold_of = qn["cv_fold"].to_numpy("int64")
    water = qn["is_water_pseudo"].to_numpy(bool)
    eligible = ~water & np.isfinite(resid)  # source pool (trainer's rule)
    scored = (
        eligible
        & np.isfinite(obs_dtw)
        & qn["confinement_class"].isin(UNCONFINED).to_numpy()
        & ~qn["huc4"].astype(str).isin(SACRIFICIAL_HUC4).to_numpy()
    )
    huc8 = qn["huc8"].astype(str).to_numpy()
    log.info(
        "bundle: %d rows, %d eligible sources, %d scored-population wells",
        len(qn),
        int(eligible.sum()),
        int(scored.sum()),
    )

    anchors = build_anchors(
        qn[man["dtw_base_col"]].to_numpy("float64"),
        qn[man["fac_anchor_col"]].to_numpy("float64"),
        qn[man["deep_anchor_col"]].to_numpy("float64"),
        float(man["flags"]["mirror_depth_m"]),
        bool(man["flags"]["mirror_anchor"]),
    )
    dims = dict(man["feature_dims"])
    model = build_model(
        man, dims, args.device, f_mae, writeback, f_src=f_src, f_srcedge=f_srcedge
    )
    src_edges = bundle_source_edges(man, args.device) if f_srcedge else None

    base_oof = pd.read_parquet(
        Path(args.baseline_dir) / "gnn_oof_predictions.parquet"
    ).set_index("canonical_id")
    base_dtw = base_oof["gnn_dtw_m"].reindex(qn["canonical_id"]).to_numpy("float64")
    arm_oof = pd.read_parquet(arm_dir / "gnn_oof_predictions.parquet").set_index(
        "canonical_id"
    )
    arm_oof_dtw = arm_oof["gnn_dtw_m"].reindex(qn["canonical_id"]).to_numpy("float64")

    def fwd(f: int, visible: np.ndarray) -> np.ndarray:
        """Fold f forward with the given visible-source mask -> predicted DTW (m)."""
        ck = models["folds"][f]
        y_c, y_s = float(ck["y_c"]), float(ck["y_s"])
        feat = fold_tensors(ck, qn, anchors, args.device)
        s = torch.as_tensor(visible.astype("float32"), device=args.device)
        val = torch.as_tensor(
            np.nan_to_num((resid - y_c) / y_s, nan=0.0),
            dtype=torch.float32,
            device=args.device,
        )
        if f_src is not None:
            feat["src_x"] = torch.stack([val * s, s], dim=-1)
        if f_srcedge is not None:
            keep = s[src_edges["src"]] > 0
            feat["srcedge_ei"] = src_edges["ei"][:, keep]
            feat["srcedge_ea"] = src_edges["ea"][keep]
            feat["srcedge_val"] = val
        out = forward_fold(model, ck, graph, feat, r_wte)
        return z_surf - out["wte"]

    # frozen nested site partition per fold: one seeded permutation of the fold's
    # scored SITES; sources at frac p = first ceil(p*n) sites; the scored target
    # set is FIXED at the complement of the max fraction (identical wells at
    # every p -> honestly paired curve points).
    site_id = pd.Series(
        [f"{a:.0f}_{b:.0f}" for a, b in np.round(xy, 0)], index=qn.index
    )
    rows_by_p_targets: dict[int, np.ndarray] = {}
    src_sites_by_fold: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(args.seed)
    for f in sorted(models["folds"]):
        held = scored & (fold_of == f)
        sites = np.array(sorted(site_id[held].unique()))
        src_sites_by_fold[f] = rng.permutation(sites)
        n_src_max = int(np.ceil(max(fracs) * len(sites)))
        tgt_sites = set(src_sites_by_fold[f][n_src_max:])
        rows_by_p_targets[f] = held & site_id.isin(tgt_sites).to_numpy()
        log.info(
            "fold %d: %d held-out scored wells, %d sites, %d fixed target wells",
            f,
            int(held.sum()),
            len(sites),
            int(rows_by_p_targets[f].sum()),
        )

    report: dict = {
        "work": "source-obs warm eval + density curve (plan 5.2/5.3, gates A1-A3)",
        "arm_dir": str(arm_dir),
        "baseline_dir": str(args.baseline_dir),
        "seed": args.seed,
        "fracs": fracs,
        "target_set": "fixed complement of the max-fraction source sites (site-level)",
        "idw_comparator": {"vw": vw, "k": idw_k, "power": idw_p},
        "by_frac": {},
    }
    tgt_all = np.zeros(len(qn), bool)
    for f in rows_by_p_targets:
        tgt_all |= rows_by_p_targets[f]

    for p in fracs:
        pred_dtw = np.full(len(qn), np.nan)
        idw_dtw = np.full(len(qn), np.nan)
        dist_km = np.full(len(qn), np.nan)
        for f in sorted(models["folds"]):
            held = scored & (fold_of == f)
            sites = src_sites_by_fold[f]
            n_src = int(np.ceil(p * len(sites)))
            warm_src = held & site_id.isin(set(sites[:n_src])).to_numpy()
            visible = (eligible & (fold_of != f)) | warm_src
            dtw_f = fwd(int(f), visible)
            tgt = rows_by_p_targets[f]
            pred_dtw[tgt] = dtw_f[tgt]
            sxy, txy = xy[visible], xy[tgt]
            idw_wte = idw_same_wells(
                sxy,
                z_surf[visible],
                wte_obs[visible],
                txy,
                z_surf[tgt],
                vw,
                idw_k,
                idw_p,
            )
            idw_dtw[tgt] = z_surf[tgt] - idw_wte
            dist_km[tgt] = cKDTree(sxy).query(txy, workers=-1)[0] / 1000.0
        err = pred_dtw - obs_dtw
        err_base = base_dtw - obs_dtw
        err_idw = idw_dtw - obs_dtw
        err_obs_depth = obs_dtw
        blk = huc8
        m = tgt_all
        entry = {
            "n_targets": int(np.isfinite(pred_dtw[m]).sum()),
            "panel": {
                "source_arm": core(err[m]),
                "baseline_arm": core(err_base[m]),
                "idw_same_wells": core(err_idw[m]),
            },
            "delta_vs_baseline": paired_block_delta(
                err[m], err_base[m], blk[m], seed=args.seed
            ),
            "delta_vs_idw": paired_block_delta(
                err[m], err_idw[m], blk[m], seed=args.seed
            ),
            "by_depth_band_delta_vs_baseline": banded_delta(
                err[m],
                err_base[m],
                err_obs_depth[m],
                DEPTH_BANDS,
                blk[m],
                seed=args.seed,
            ),
            "by_dist_to_source_km": {
                "source_arm": banded(err[m], dist_km[m], DIST_BANDS_KM),
                "delta_vs_baseline": banded_delta(
                    err[m],
                    err_base[m],
                    dist_km[m],
                    DIST_BANDS_KM,
                    blk[m],
                    seed=args.seed,
                ),
                "delta_vs_idw": banded_delta(
                    err[m],
                    err_idw[m],
                    dist_km[m],
                    DIST_BANDS_KM,
                    blk[m],
                    seed=args.seed,
                ),
            },
        }
        if p == 0.0:
            # round-trip: p=0 forward == the trainer's final forward -> archived OOF.
            d = np.nanmax(np.abs(pred_dtw[m] - arm_oof_dtw[m]))
            entry["p0_oof_roundtrip_max_abs_diff_m"] = float(d)
            if not d < args.oof_tol_m:
                raise SystemExit(
                    f"p=0 forward does not reproduce archived OOF (max |diff| "
                    f"{d:.6f} m >= {args.oof_tol_m}); assembly skew -- fix first"
                )
            log.info("p=0 OOF round-trip PASSED (max |diff| %.2e m)", d)
        report["by_frac"][f"{p:g}"] = entry
        log.info(
            "frac %.2f: n=%d src-arm MAD %.3f m (base %.3f, idw %.3f) | dMAD vs "
            "base %+0.3f m %s",
            p,
            entry["n_targets"],
            entry["panel"]["source_arm"].get("mad_m", float("nan")),
            entry["panel"]["baseline_arm"].get("mad_m", float("nan")),
            entry["panel"]["idw_same_wells"].get("mad_m", float("nan")),
            -entry["delta_vs_baseline"].get("d_mad_m", float("nan")),
            entry["delta_vs_baseline"].get("d_mad_ci95"),
        )

    out = out_dir / "source_warm_eval.json"
    out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
