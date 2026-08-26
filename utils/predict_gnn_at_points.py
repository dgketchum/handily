"""Point predictions from saved CONUS GNN fold checkpoints.

Companion to ``infer_conus_gnn.py`` (the raster renderer): forwards a saved
model at an arbitrary set of point coordinates instead of a basin lattice, for
well-level evaluation (e.g. the NV NDWR holdout, NV_HOLDOUT_EVAL_PLAN.md).
Feature construction is deployment-semantics and identical to the renderer's
lattice path (crossfit R/deep interpolation per docs/inference_leakage_prevention.md,
fresh kNN lateral attachment, bundle-frame scalers), extended with the two
pieces the renderer does not build yet:

- **dd/dup query features** — drilled-depth kNN-IDW/p90 (deployment: no
  self-exclusion; ``--dd-self-exclude-m 100`` = training-semantics sensitivity
  variant) and the Dupuit hang features from the bundle's top-2-Strahler
  boundary set.
- **query-writeback + MAE-embedding arms** — the model is constructed with
  ``writeback``/``f_mae`` inferred from the checkpoint state dict (the
  manifest does not record the embedding config). MAE embeddings for the
  points come from ``extract_mae_embeddings.py --coords`` output passed via
  ``--mae-embeddings``; their standardization replays the trainer's global fit
  over the bundle's allq embedding table.

``--oof-check`` replays the bundle's own wells through the checkpoints
(bundle frames verbatim, incl. bundle lateral edges / writeback / mae_x) and
compares against the archived OOF — run it before trusting point output from
any arm the production renderer cannot load.

With writeback ON, reach states depend on the query set: the point set itself
participates in the writeback pass (deployment semantics — a rendered lattice's
cells participate the same way), so the whole point set is forwarded in one
full batch, never chunked.

Usage:
    uv run python utils/predict_gnn_at_points.py --model-dir <dir> --oof-check
    uv run python utils/predict_gnn_at_points.py --model-dir <dir> \
        --points pts.parquet --out preds.parquet [--mae-embeddings emb.parquet]

``--points``: parquet with x5070/y5070 plus any id columns (carried through).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conus_graph_inputs import (  # noqa: E402
    DEM,
    WATER_BLOCKS_PARQUET,
    _relief_coords,
    _sample_gsw_occurrence,
    attach_lateral_attrs,
    build_lateral_edges,
    sample_drilled_depth,
    sample_gridmet,
    sample_relief_etrm,
    sample_terrain_multiscale,
    water_query_features,
)
from build_dupuit_wte import build_boundaries, hang_interp  # noqa: E402
from build_source_edges import EDGE_COLS as SOURCE_EDGE_COLS  # noqa: E402
from build_stacker_features import sample_coarse  # noqa: E402
from fac_rem_registry import sample_fac_rem  # noqa: E402
from infer_conus_gnn import (  # noqa: E402
    assert_mixture_identity,
    build_anchors,
    fold_tensors,
    forward_fold,
    idw_exclude_at_points,
    lateral_tensors,
    load_models,
    prune_for_queries,
    reach_tensors,
    well_pool,
)
from train_wte_gnn import WTEGraphNet, fit_stats  # noqa: E402
from train_wte_gnn import apply_stats as apply_stats_df  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predict_gnn_at_points")


def ckpt_extensions(ck: dict) -> tuple[int | None, bool]:
    """(f_mae, writeback) from a fold checkpoint's state dict.

    The inference manifest records neither the MAE head nor its width (known
    gap), so the weights are the authority: ``mae_enc.0.weight`` is
    (hidden, f_mae); any ``writeback_conv.*`` key means the 6C branch exists.
    """
    sd = ck["state_dict"]
    f_mae = sd["mae_enc.0.weight"].shape[1] if "mae_enc.0.weight" in sd else None
    writeback = any(k.startswith("writeback_conv.") for k in sd)
    return f_mae, writeback


def ckpt_f_src(ck: dict, f_query: int) -> int | None:
    """Source-obs block width from a fold checkpoint (--source-obs arms).

    The src block widens ONLY query_enc's input, so its width is the excess of
    ``query_enc.0.weight`` over the manifest's query feature dim (None = plain arm).
    """
    extra = int(ck["state_dict"]["query_enc.0.weight"].shape[1]) - int(f_query)
    return extra if extra > 0 else None


def ckpt_f_srcedge(ck: dict) -> int | None:
    """Source-edge attr width from a fold checkpoint (--source-edges arms, rung 1).

    ``source_read.score_mlp.0.weight`` (softmax attention) or
    ``source_read.gate_mlp.0.weight`` (--source-edge-gated) is
    (hidden, in_src + in_dst + edge_dim) with in_src = in_dst = hidden
    (``source_enc.0.weight``'s out dim), so the edge-attr width is the excess
    over 2*hidden. None = no source-edge slot.
    """
    sd = ck["state_dict"]
    key = next(
        (
            k
            for k in ("source_read.score_mlp.0.weight", "source_read.gate_mlp.0.weight")
            if k in sd
        ),
        None,
    )
    if key is None:
        return None
    hidden = int(sd["source_enc.0.weight"].shape[0])
    return int(sd[key].shape[1]) - 2 * hidden


def bundle_source_edges(man: dict, device: str) -> dict:
    """Trainer-verbatim source-edge tensors for a --source-edges arm.

    The trainer standardizes the edge attrs by a GLOBAL fit over the full edge
    table (no fold or mask dependence), so refitting from the frozen parquet is
    exact. The path comes from the inference manifest flags.
    """
    path = man["flags"].get("source_edges")
    if not path:
        raise SystemExit(
            "checkpoint has a source_read slot but the inference manifest carries "
            "no source_edges path (flags.source_edges)"
        )
    se = pd.read_parquet(path)
    ea = apply_stats_df(se, fit_stats(se, list(SOURCE_EDGE_COLS), None))
    src = torch.as_tensor(
        se["src_query_node_idx"].to_numpy("int64"), dtype=torch.long, device=device
    )
    dst = torch.as_tensor(
        se["query_node_idx"].to_numpy("int64"), dtype=torch.long, device=device
    )
    return {
        "src": src,
        "ei": torch.stack([src, dst]),
        "ea": torch.as_tensor(ea, dtype=torch.float32, device=device),
    }


def point_source_ctx(
    man: dict,
    bman: dict,
    gdir: Path,
    qxy: np.ndarray,
    z_surf: np.ndarray,
    lat: pd.DataFrame,
    rn_full: pd.DataFrame,
    device: str,
    exclude_m: float = 0.0,
) -> dict:
    """Deployment source context for EXTERNAL points (--source-edges arms).

    The source side of the read carries ONLY the encoded observed residual --
    the model never consumes source covariates through this path -- so the
    bundle wells stay OUT of the query set. We build fresh point<-well kNN
    edges with the build_source_edges recipe (same relief-lifted metric, same
    k, same 4 attrs incl. FAC-basin identity), standardize them with the
    trainer's GLOBAL fit over the frozen bundle edge table (trainer-verbatim,
    like bundle_source_edges), and expose every eligible well to every fold:
    deployment leaves no observation on the table, and external points carry
    no labels into the forward, so the training protocol's same-site/masking
    guards do not apply. ``exclude_m`` > 0 drops edges from wells within that
    radius of the point (an eval-fairness sensitivity knob; 0 = deployment
    semantics -- a map cell at a well reads that well).
    """
    se = pd.read_parquet(man["flags"]["source_edges"])
    stats = fit_stats(se, list(SOURCE_EDGE_COLS), None)
    k = int(se["rank"].max()) + 1
    qn = (
        pd.read_parquet(gdir / "query_nodes.parquet")
        .sort_values("query_node_idx")
        .reset_index(drop=True)
    )
    water = (
        qn["is_water_pseudo"].to_numpy(bool)
        if "is_water_pseudo" in qn.columns
        else np.zeros(len(qn), bool)
    )
    resid_all = qn["wte_residual_m"].to_numpy("float64")
    elig = ~water & np.isfinite(resid_all)
    w = qn[elig].reset_index(drop=True)
    wxy = w[["x5070", "y5070"]].to_numpy("float64")
    wz = np.nan_to_num(w["z_surf_well_m"].to_numpy("float64"), nan=0.0)
    vw = float(bman["r_relief_vw"])

    # controlling-reach FAC basin, raw labels on both sides (equality is what
    # same_basin encodes; build_source_edges factorizes only for compactness)
    basin_of_reach = rn_full.set_index("reach_node_idx")["basin"]
    blat = pd.read_parquet(
        gdir / "lateral_edges.parquet",
        columns=["query_node_idx", "reach_node_idx", "is_controlling"],
    )
    bctrl = (
        blat[blat["is_controlling"].astype(bool)]
        .drop_duplicates("query_node_idx")
        .set_index("query_node_idx")["reach_node_idx"]
    )
    src_basin = w["query_node_idx"].map(bctrl).map(basin_of_reach)
    pctrl = (
        lat[lat["is_controlling"].astype(bool)]
        .drop_duplicates("query_node_idx")
        .set_index("query_node_idx")["reach_node_idx"]
    )
    dest_basin = (
        pd.Series(np.arange(len(qxy), dtype="int64")).map(pctrl).map(basin_of_reach)
    )

    pad = 8 if exclude_m > 0 else 0
    kq = min(k + pad, len(w))
    tree = cKDTree(_relief_coords(wxy, wz, vw))
    rd, loc = tree.query(_relief_coords(qxy, z_surf, vw), k=kq, workers=-1)
    if kq == 1:
        rd, loc = rd[:, None], loc[:, None]
    dest = np.repeat(np.arange(len(qxy), dtype="int64"), kq)
    src = loc.ravel().astype("int64")
    geo_km = np.sqrt(((qxy[dest] - wxy[src]) ** 2).sum(axis=1)) / 1000.0
    ed = pd.DataFrame(
        {
            "dest": dest,
            "src": src,
            "log1p_geo_dist_km": np.log1p(geo_km),
            "log1p_relief_dist_km": np.log1p(rd.ravel() / 1000.0),
            "rel_elev_m": z_surf[dest] - wz[src],
            "same_basin": (
                dest_basin.iloc[dest].notna().to_numpy()
                & (dest_basin.iloc[dest].to_numpy() == src_basin.iloc[src].to_numpy())
            ).astype("float64"),
        }
    )
    if exclude_m > 0:
        ed = ed[geo_km * 1000.0 >= exclude_m]
    ed = ed[ed.groupby("dest").cumcount() < k].reset_index(drop=True)
    deg = ed.groupby("dest").size()
    log.info(
        "point source edges: %d edges to %d/%d points (k=%d, vw=%.0f, "
        "exclude_m=%.0f, %d eligible wells, median rank-0 geo %.2f km, "
        "frac_same_basin %.2f)",
        len(ed),
        int(deg.size),
        len(qxy),
        k,
        vw,
        exclude_m,
        len(w),
        float(np.expm1(ed.groupby("dest")["log1p_geo_dist_km"].min().median())),
        float(ed["same_basin"].mean()),
    )
    ea = apply_stats_df(ed, stats)
    src_t = torch.as_tensor(
        ed["src"].to_numpy("int64"), dtype=torch.long, device=device
    )
    dst_t = torch.as_tensor(
        ed["dest"].to_numpy("int64"), dtype=torch.long, device=device
    )
    return {
        "resid": w["wte_residual_m"].to_numpy("float64"),
        "pool_by_fold": lambda f: np.ones(len(w), dtype=bool),
        "edges": {
            "src": src_t,
            "ei": torch.stack([src_t, dst_t]),
            "ea": torch.as_tensor(ea, dtype=torch.float32, device=device),
        },
    }


def build_model_pt(
    man: dict,
    dims: dict,
    device: str,
    f_mae: int | None,
    writeback: bool,
    f_src: int | None = None,
    f_srcedge: int | None = None,
) -> WTEGraphNet:
    """infer_conus_gnn.build_model extended with the writeback/MAE/src ctor args."""
    flags = man["flags"]
    return WTEGraphNet(
        dims["reach"],
        dims["query"],
        dims["channel_edge"],
        dims["lateral_edge"],
        int(man["effective_hidden"]),
        int(man["channel_layers"]),
        float(man["dropout"]),
        sigma=bool(flags["sigma_head"]),
        prior_gate=True,
        mirror_anchor=bool(flags["mirror_anchor"]),
        directional_edges=bool(flags.get("directional_edges")),
        f_mae=f_mae,
        writeback=writeback,
        f_src=f_src,
        f_srcedge=f_srcedge,
        srcedge_gated=bool(flags.get("source_edge_gated")),
    ).to(device)


def bundle_mae_stats(gdir: Path, f_mae: int) -> tuple[dict, list[str], Path]:
    """Replay the trainer's global MAE z-score fit over the bundle allq table.

    The trainer (non-swl branch) fits mean/std over ALL query nodes' embeddings
    and the checkpoints do not persist those stats; the bundle table is frozen,
    so the refit is exact. A bundle can hold several allq tables (MAE-only,
    AEF+MAE); the one whose mae_* width matches the checkpoint's f_mae is the
    one the arm trained on.
    """
    cands = sorted(gdir.glob("*_allq.parquet"))
    if not cands:
        raise SystemExit(f"no *_allq.parquet embedding table in {gdir}")
    for path in cands:
        df = pd.read_parquet(path)
        cols = [c for c in df.columns if c.startswith("mae_")]
        if len(cols) == f_mae:
            log.info("embedding table %s (width %d matches f_mae)", path.name, f_mae)
            return fit_stats(df[cols], cols, None), cols, path
    raise SystemExit(
        f"no allq table in {gdir} has {f_mae} mae_* columns "
        f"(candidates: {[c.name for c in cands]})"
    )


def mae_tensor(
    emb: pd.DataFrame, stats: dict, cols: list[str], f_mae: int, device: str
) -> torch.Tensor:
    missing = [c for c in cols if c not in emb.columns]
    if missing:
        raise SystemExit(f"--mae-embeddings lacks columns: {missing[:4]}...")
    if len(cols) != f_mae:
        raise SystemExit(
            f"embedding width {len(cols)} != checkpoint f_mae {f_mae} -- wrong "
            "embedding table for this arm"
        )
    if not np.isfinite(emb[cols].to_numpy()).all():
        raise SystemExit("--mae-embeddings has non-finite values; investigate")
    return torch.as_tensor(
        apply_stats_df(emb[cols], stats), dtype=torch.float32, device=device
    )


def run_folds_pt(
    models: dict,
    graph: dict,
    frame: pd.DataFrame,
    r_wte: np.ndarray,
    anchors: dict,
    device: str,
    where: str,
    f_mae: int | None,
    writeback: bool,
    src_ctx: dict | None = None,
) -> dict:
    """infer_conus_gnn.run_folds with the extended model ctor (one full batch).

    ``src_ctx`` (--source-obs arms only) supplies the assimilation inputs:
    ``resid`` = frame-aligned observed wte_residual_m (NaN where no obs) and
    ``pool_by_fold(f)`` = the boolean mask of rows whose obs are VISIBLE to fold
    f's forward. Values are standardized per fold by the checkpoint's y_c/y_s,
    exactly the trainer's construction.
    """
    man = models["manifest"]
    dims = dict(man["feature_dims"])
    ck0 = next(iter(models["folds"].values()))
    f_src = ckpt_f_src(ck0, dims["query"])
    f_srcedge = ckpt_f_srcedge(ck0)
    need_src = f_src is not None or f_srcedge is not None
    if need_src != (src_ctx is not None):
        raise SystemExit(
            f"source-arm mismatch ({where}): checkpoint f_src={f_src} "
            f"f_srcedge={f_srcedge} but "
            f"src_ctx {'missing' if src_ctx is None else 'supplied'}"
        )
    model = build_model_pt(
        man, dims, device, f_mae, writeback, f_src=f_src, f_srcedge=f_srcedge
    )
    per_fold = []
    for f, ck in sorted(models["folds"].items()):
        feat = fold_tensors(ck, frame, anchors, device)
        if src_ctx is not None:
            y_c, y_s = float(ck["y_c"]), float(ck["y_s"])
            sel = src_ctx["pool_by_fold"](int(f))
            s = torch.as_tensor(sel.astype("float32"), device=device)
            val = torch.as_tensor(
                np.nan_to_num((src_ctx["resid"] - y_c) / y_s, nan=0.0),
                dtype=torch.float32,
                device=device,
            )
            if f_src is not None:
                feat["src_x"] = torch.stack([val * s, s], dim=-1)
            if f_srcedge is not None:
                ed = src_ctx["edges"]
                keep = s[ed["src"]] > 0
                feat["srcedge_ei"] = ed["ei"][:, keep]
                feat["srcedge_ea"] = ed["ea"][keep]
                feat["srcedge_val"] = val
        out = forward_fold(model, ck, graph, feat, r_wte)
        assert_mixture_identity(out, r_wte, anchors, f, where)
        per_fold.append(out)
        del feat
    wte_f = np.stack([o["wte"] for o in per_fold])
    w = np.stack([o["w"] for o in per_fold]).mean(0)
    w /= w.sum(1, keepdims=True)
    agg = {
        "wte": np.median(wte_f, axis=0),
        "head_wte": np.median(np.stack([o["head_wte"] for o in per_fold]), axis=0),
        "fold_spread": np.percentile(wte_f, 90, axis=0)
        - np.percentile(wte_f, 10, axis=0),
        "w": w,
        "wte_by_fold": {f: o["wte"] for f, o in zip(sorted(models["folds"]), per_fold)},
    }
    if "sigma" in per_fold[0]:
        agg["sigma"] = np.median(np.stack([o["sigma"] for o in per_fold]), axis=0)
    return agg


def bundle_graph(models: dict, device: str, mae_path: str | None) -> tuple:
    """Bundle-verbatim graph tensors (incl. writeback/MAE ext) + sorted qn."""
    man = models["manifest"]
    gdir = Path(man["graph_dir"])
    qn = (
        pd.read_parquet(gdir / "query_nodes.parquet")
        .sort_values("query_node_idx")
        .reset_index(drop=True)
    )
    rn = (
        pd.read_parquet(gdir / "reach_nodes.parquet")
        .sort_values("reach_node_idx")
        .reset_index(drop=True)
    )
    ce = pd.read_parquet(gdir / "channel_edges.parquet")
    lat = pd.read_parquet(gdir / "lateral_edges.parquet")
    rn, ce, lat = prune_for_queries(rn, ce, lat, int(man["channel_layers"]))
    dims = dict(man["feature_dims"])
    graph = {
        **reach_tensors(rn, ce, models["shared"], dims, device),
        **lateral_tensors(lat, models["shared"], dims, device),
    }
    f_mae, writeback = ckpt_extensions(models["folds"][0])
    if writeback:
        graph["lat_ei_reversed"] = graph["lat_ei"].flip(0)
    if f_mae is not None:
        stats, cols, table = bundle_mae_stats(gdir, f_mae)
        emb = pd.read_parquet(mae_path or table).set_index("query_node_idx")
        emb = emb.loc[qn["query_node_idx"].to_numpy()].reset_index()
        graph["mae_x"] = mae_tensor(emb, stats, cols, f_mae, device)
    return graph, qn, f_mae, writeback


def oof_check_pt(models: dict, model_dir: Path, device: str, tol_m: float, mae_path):
    """infer_conus_gnn.oof_check extended to writeback/MAE arms."""
    man = models["manifest"]
    graph, qn, f_mae, writeback = bundle_graph(models, device, mae_path)
    f_src = ckpt_f_src(models["folds"][0], man["feature_dims"]["query"])
    f_srcedge = ckpt_f_srcedge(models["folds"][0])
    log.info(
        "oof-check: f_mae=%s writeback=%s f_src=%s f_srcedge=%s",
        f_mae,
        writeback,
        f_src,
        f_srcedge,
    )
    src_ctx = None
    if f_src is not None or f_srcedge is not None:
        # replay the trainer's FINAL-forward semantics: fold f's sources are the
        # eligible (real, finite-target) wells OUTSIDE fold f (= its tr|va pool),
        # so the archived OOF at test rows round-trips exactly.
        resid = qn["wte_residual_m"].to_numpy("float64")
        water = qn["is_water_pseudo"].to_numpy(bool)
        shore = (
            qn["is_shore_pseudo"].to_numpy(bool)
            if "is_shore_pseudo" in qn.columns
            else np.zeros(len(qn), bool)
        )
        eligible = ~water & ~shore & np.isfinite(resid)
        fold_of = qn["cv_fold"].to_numpy("int64")
        src_ctx = {
            "resid": resid,
            "pool_by_fold": lambda f: eligible & (fold_of != f),
        }
        if f_srcedge is not None:
            src_ctx["edges"] = bundle_source_edges(man, device)
    r_wte = qn["regional_wte_idw_oof_m"].to_numpy("float64")
    anchors = build_anchors(
        qn[man["dtw_base_col"]].to_numpy("float64"),
        qn[man["fac_anchor_col"]].to_numpy("float64"),
        qn[man["deep_anchor_col"]].to_numpy("float64"),
        float(man["flags"]["mirror_depth_m"]),
        bool(man["flags"]["mirror_anchor"]),
    )
    agg = run_folds_pt(
        models,
        graph,
        qn,
        r_wte,
        anchors,
        device,
        "oof-check",
        f_mae,
        writeback,
        src_ctx=src_ctx,
    )
    arch = pd.read_parquet(model_dir / "gnn_oof_predictions.parquet")
    arch = arch.set_index("canonical_id").loc[qn["canonical_id"]]
    fold = qn["cv_fold"].to_numpy("int64")
    worst = 0.0
    for f, wte_hat in agg["wte_by_fold"].items():
        te = fold == f
        d = np.max(np.abs(wte_hat[te] - arch["gnn_wte_hat_m"].to_numpy()[te]))
        worst = max(worst, float(d))
        log.info("fold %d: OOF wte_hat max |diff| = %.6f m (n=%d)", f, d, int(te.sum()))
    if worst >= tol_m:
        raise SystemExit(
            f"OOF round-trip FAILED: max |wte_hat - archived| = {worst:.6f} m "
            f">= {tol_m} (assembly/extension skew)"
        )
    log.info("OOF round-trip PASSED: max |diff| %.2e m < %g m", worst, tol_m)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--points", help="parquet with x5070/y5070 (+ id cols)")
    ap.add_argument("--out", help="output parquet path")
    ap.add_argument(
        "--mae-embeddings",
        help="points-keyed embedding parquet from extract_mae_embeddings.py "
        "--coords, row-aligned join on x5070/y5070 rounded to mm",
    )
    ap.add_argument(
        "--dd-self-exclude-m",
        type=float,
        default=0.0,
        help="drilled-depth self-exclusion radius; 0 = deployment semantics "
        "(what a rendered raster gives a cell containing the well), 100 = "
        "training-semantics sensitivity variant",
    )
    ap.add_argument(
        "--source-exclude-m",
        type=float,
        default=0.0,
        help="(--source-edges arms) drop point<-well source edges from wells "
        "within this radius of the point; 0 = deployment semantics (a cell at "
        "a well reads that well), >0 = eval-fairness sensitivity variant",
    )
    ap.add_argument("--dem", default=DEM)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--oof-check", action="store_true")
    ap.add_argument("--oof-tol-m", type=float, default=1e-3)
    args = ap.parse_args()
    model_dir = Path(args.model_dir)
    models = load_models(model_dir)
    man = models["manifest"]

    if args.oof_check:
        oof_check_pt(
            models, model_dir, args.device, args.oof_tol_m, args.mae_embeddings
        )
        return
    if not args.points or not args.out:
        raise SystemExit("pass --points and --out (or --oof-check)")

    f_mae, writeback = ckpt_extensions(models["folds"][0])
    log.info("arm %s: f_mae=%s writeback=%s", model_dir.name, f_mae, writeback)
    if f_mae is not None and not args.mae_embeddings:
        raise SystemExit("this arm has an MAE head; pass --mae-embeddings")
    if ckpt_f_src(models["folds"][0], man["feature_dims"]["query"]) is not None:
        raise SystemExit(
            "--source-obs (rung 0) arm: external-point prediction needs the source "
            "wells inside the query set (their obs travel via src_x + writeback) -- "
            "unsupported (rung 0 is shelved); use --oof-check or a --source-edges arm"
        )
    f_srcedge_pt = ckpt_f_srcedge(models["folds"][0])

    gdir = Path(man["graph_dir"])
    bman = json.loads((gdir / "graph_manifest.json").read_text())
    pts = pd.read_parquet(args.points).reset_index(drop=True)
    # row number in the ORIGINAL points file = the query_node_idx convention of
    # the extract_mae_embeddings.py --coords companion run
    pts["_row"] = np.arange(len(pts), dtype="int64")
    qx = pts["x5070"].to_numpy("float64")
    qy = pts["y5070"].to_numpy("float64")
    z_surf = sample_coarse(args.dem, qx, qy)
    fin = np.isfinite(z_surf)
    if not fin.all():
        log.info("dropping %d/%d points with DEM nodata", int((~fin).sum()), len(fin))
        pts, qx, qy, z_surf = (
            pts[fin].reset_index(drop=True),
            qx[fin],
            qy[fin],
            z_surf[fin],
        )
    n = len(pts)
    qxy = np.c_[qx, qy]
    log.info("%d points", n)

    # R + deep prior: crossfit-field interpolation (leak-free deployment path)
    qn = well_pool(pd.read_parquet(gdir / "query_nodes.parquet"))
    wells_xy = qn[["x5070", "y5070"]].to_numpy("float64")
    wells_z = qn[man["surface_elev_col"]].to_numpy("float64")
    vw = float(bman["r_relief_vw"])
    k, p = int(bman["idw_k"]), float(bman["idw_power"])
    q_co = _relief_coords(qxy, z_surf, vw)
    pool_co = _relief_coords(wells_xy, wells_z, vw)
    r_cross = qn["regional_wte_idw_oof_m"].to_numpy("float64")
    deep_cross = qn["deep_regional_wte_idw_oof_m"].to_numpy("float64")
    r_wte = idw_exclude_at_points(pool_co, r_cross, wells_xy, q_co, qxy, k, p, 0.0)
    deep_wte = idw_exclude_at_points(
        pool_co, deep_cross, wells_xy, q_co, qxy, k, p, 0.0
    )
    fac_dtw = sample_fac_rem(qx, qy)
    base = z_surf - r_wte
    anchors = build_anchors(
        base,
        np.where(np.isfinite(fac_dtw), (z_surf - fac_dtw) - r_wte, np.nan),
        deep_wte - r_wte,
        float(man["flags"]["mirror_depth_m"]),
        bool(man["flags"]["mirror_anchor"]),
    )
    log.info("FAC coverage %.1f%%", 100 * anchors["fac_present"].mean())

    frame = pd.DataFrame(
        {
            "query_node_idx": np.arange(n, dtype="int64"),
            "fac_rem_wte_anom_m": anchors["fac_raw"],
            "deep_regional_wte_anom_m": anchors["deep_raw"],
            "fac_rem_dtw_m": fac_dtw,
        }
    )
    for d_cov in (
        sample_relief_etrm(qx, qy, z_surf),
        sample_terrain_multiscale(qx, qy),
        sample_gridmet(qx, qy),
    ):
        for c, v in d_cov.items():
            frame[c] = v
    if man["flags"].get("water_features"):
        from pyproj import Transformer

        blocks = pd.read_parquet(gdir / WATER_BLOCKS_PARQUET)
        water_xy = blocks[["x5070", "y5070"]].to_numpy("float64")
        lon_q, lat_q = Transformer.from_crs(5070, 4326, always_xy=True).transform(
            qx, qy
        )
        frame["gsw_occ_pct"] = _sample_gsw_occurrence(lon_q, lat_q)
        for c, v in water_query_features(qxy, z_surf, water_xy, args.dem).items():
            frame[c] = v
    if "drilled_depth_idw_m" in man["query_feature_cols"]:
        dd = bman["drilled_depth"]
        for c, v in sample_drilled_depth(
            dd["points"],
            qxy,
            int(dd["k"]),
            float(dd["power"]),
            self_exclude_m=float(args.dd_self_exclude_m),
        ).items():
            frame[c] = v
        dh = bman["dupuit_hang"]
        bnd = build_boundaries(
            str(gdir / "reach_nodes.parquet"),
            bman["sources"]["geom"],
            top_orders=int(dh["top_orders"]),
        )
        hang_wte, d_bnd = hang_interp(
            bnd[["cx", "cy"]].to_numpy("float64"),
            bnd["reach_elev_m"].to_numpy("float64"),
            qxy,
            k=int(dh["idw_k"]),
            power=float(dh["idw_power"]),
        )
        frame["dupuit_hang_dtw_m"] = z_surf - hang_wte
        frame["log1p_dupuit_d_m"] = np.log1p(d_bnd)
    missing = [c for c in man["query_feature_cols"] if c not in frame.columns]
    if missing:
        raise SystemExit(f"point frame lacks query features: {missing}")

    rn_full = (
        pd.read_parquet(gdir / "reach_nodes.parquet")
        .sort_values("reach_node_idx")
        .reset_index(drop=True)
    )
    ce_full = pd.read_parquet(gdir / "channel_edges.parquet")
    geom = gpd.read_parquet(bman["sources"]["geom"])
    comid_to_idx = dict(
        zip(
            rn_full["comid"].to_numpy("int64"),
            rn_full["reach_node_idx"].to_numpy("int64"),
        )
    )
    geom["reach_node_idx"] = geom["comid"].map(comid_to_idx)
    geom = geom[geom["reach_node_idx"].notna()].copy()
    geom["reach_node_idx"] = geom["reach_node_idx"].astype("int64")
    lat = build_lateral_edges(qxy, geom, None, int(bman["knn_lateral"]))
    lat = attach_lateral_attrs(lat, rn_full, z_surf, float(bman["conductance_p"]))
    # pre-prune copy: prune_for_queries REMAPS lat's reach_node_idx to the pruned
    # numbering, but the source-edge basin lookup needs rn_full's original ids.
    lat_full = lat
    rn, ce, lat = prune_for_queries(rn_full, ce_full, lat, int(man["channel_layers"]))
    log.info(
        "pruned subgraph: %d reaches / %d ch edges / %d lat edges",
        len(rn),
        len(ce),
        len(lat),
    )
    dims = dict(man["feature_dims"])
    graph = {
        **reach_tensors(rn, ce, models["shared"], dims, args.device),
        **lateral_tensors(lat, models["shared"], dims, args.device),
    }
    if writeback:
        graph["lat_ei_reversed"] = graph["lat_ei"].flip(0)
    if f_mae is not None:
        stats, cols, _ = bundle_mae_stats(gdir, f_mae)
        emb = pd.read_parquet(args.mae_embeddings).set_index("query_node_idx")
        missing = np.setdiff1d(pts["_row"].to_numpy(), emb.index.to_numpy())
        if len(missing):
            raise SystemExit(
                f"--mae-embeddings misses {len(missing)}/{n} points (join on "
                "query_node_idx == points-file row number)"
            )
        emb = emb.loc[pts["_row"].to_numpy()].reset_index()
        graph["mae_x"] = mae_tensor(emb, stats, cols, f_mae, args.device)

    src_ctx = None
    if f_srcedge_pt is not None:
        src_ctx = point_source_ctx(
            man,
            bman,
            gdir,
            qxy,
            z_surf,
            lat_full,
            rn_full,
            args.device,
            exclude_m=float(args.source_exclude_m),
        )
    agg = run_folds_pt(
        models,
        graph,
        frame,
        r_wte,
        anchors,
        args.device,
        "points",
        f_mae,
        writeback,
        src_ctx=src_ctx,
    )
    out = pts.drop(columns=["_row"]).copy()
    out["pred_wte_m"] = agg["wte"]
    out["pred_dtw_m"] = z_surf - agg["wte"]
    out["z_surf_m"] = z_surf
    out["r_wte_m"] = r_wte
    out["fac_rem_dtw_m"] = fac_dtw
    out["fold_spread_m"] = agg["fold_spread"]
    if "sigma" in agg:
        out["sigma_m"] = agg["sigma"]
    experts = ["fac", "deep"] + (["mirror"] if man["flags"]["mirror_anchor"] else [])
    for i, e in enumerate(experts + ["head"]):
        out[f"gate_w_{e}"] = agg["w"][:, i]
    out.to_parquet(args.out)
    log.info(
        "wrote %s: %d rows; pred_dtw_m median %.2f m",
        args.out,
        len(out),
        float(np.nanmedian(out["pred_dtw_m"])),
    )


if __name__ == "__main__":
    main()
