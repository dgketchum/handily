"""Wall-to-wall CONUS GNN inference: per-basin 100 m lattice -> coarse rasters.

Phases 1+2 of notes/GNN_INFERENCE_10M_PLAN.md. Loads a ``--save-models`` run
(``<model-dir>/models/``: per-fold checkpoints + shared stats + inference
manifest), builds an ephemeral 100 m EPSG:5070 query lattice per basin (snapped
to the canonical CONUS origin), replays the trainer's exact standardization /
anchor / forward contract for all folds, and writes the fold-median coarse
rasters under ``/data/ssd2/handily/huc8/{basin}/gnn/{model_name}/``:

  gnn_wte_100m.tif        fold-median recomposed WTE (m absolute head)
  gnn_dtw_100m.tif        z_surf - WTE (unclamped; %% negative logged)
  gnn_sigma_100m.tif      fold-median Laplace scale b (m)
  gnn_gate_w_100m.tif     mean gate weights (bands = manifest gate_experts)
  gnn_head_wte_100m.tif   fold-median free-head expert WTE
  gnn_deep_wte_100m.tif   deep-IDW expert WTE
  gnn_r_wte_100m.tif      all-well relief-IDW base R
  gnn_fold_spread_100m.tif p90-p10 of per-fold WTE

Every fold's forward is checked against the gate-mixture identity
(wte == sum_i w_i * expert_wte_i, <1e-3 m) before anything is written; the
10 m render (render_gnn_10m.py) then recomposes from these layers exactly.

``--oof-check`` runs the bundle's own wells through the persisted checkpoints
and compares against the archived OOF predictions -- the end-to-end round-trip
gate (stats + tensor assembly + forward) that must pass before lattice output
is trusted.

Usage:
    uv run python utils/infer_conus_gnn.py \
        --model-dir /data/ssd2/handily/conus/wte_gnn/gnn_conus_monitoring_gate_mirror_sigma_prod \
        --state NM
    uv run python utils/infer_conus_gnn.py --model-dir ... --oof-check
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
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conus_graph_inputs import (  # noqa: E402
    DEM,
    _relief_coords,
    attach_lateral_attrs,
    build_lateral_edges,
    deep_well_mask,
    idw_at_points,
    sample_gridmet,
    sample_relief_etrm,
    sample_terrain_multiscale,
)
from build_stacker_features import sample_coarse  # noqa: E402
from fac_rem_registry import sample_fac_rem  # noqa: E402
from train_conus_gnn import _fac_feat, prune_reach_graph  # noqa: E402
from train_wte_gnn import WTEGraphNet, apply_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("infer_conus_gnn")

HUC8_ROOT = Path("/data/ssd2/handily/huc8")
WBD_HU8 = "/nas/hydrography/HUC_Boundaries/wbd_national/wbdhu8_5070.parquet"
# canonical CONUS 100 m lattice origin (GNN_INFERENCE_10M_PLAN.md section 0)
X0, Y0 = -2_540_000.0, 3_258_000.0
RES = 100.0
NODATA = -9999.0
MIX_TOL_M = 1e-3


def write_tif(path: Path, arr: np.ndarray, transform, count: int = 1, descs=None):
    """float32 deflate/tiled EPSG:5070 writer (build_str7_idw_raster profile)."""
    a = np.where(np.isfinite(arr), arr, NODATA).astype("float32")
    if a.ndim == 2:
        a = a[None]
    prof = dict(
        driver="GTiff",
        dtype="float32",
        count=a.shape[0],
        height=a.shape[1],
        width=a.shape[2],
        crs="EPSG:5070",
        transform=transform,
        nodata=NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(a)
        for i, d in enumerate(descs or [], start=1):
            dst.set_band_description(i, d)


def snapped_window(bounds: tuple, res: float) -> tuple:
    """EPSG:5070 bounds -> (transform, width, height) snapped to the canonical origin."""
    minx, miny, maxx, maxy = bounds
    minx = np.floor((minx - X0) / res) * res + X0
    miny = np.floor((miny - Y0) / res) * res + Y0
    maxx = np.ceil((maxx - X0) / res) * res + X0
    maxy = np.ceil((maxy - Y0) / res) * res + Y0
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    return from_origin(minx, maxy, res, res), width, height


def load_models(model_dir: Path) -> dict:
    mdir = model_dir / "models"
    man = json.loads((mdir / "inference_manifest.json").read_text())
    flags = man["flags"]
    for k in ("fac_skip", "fac_gate", "fac_lambda", "pinball", "anchors", "aquifer"):
        if flags.get(k):
            raise SystemExit(
                f"inference runner supports the production prior-gate arms only; "
                f"flag {k!r} is set in {mdir}"
            )
    if not flags["prior_gate"]:
        raise SystemExit("inference runner requires a --prior-gate checkpoint")
    folds = {}
    for f in man["folds"]:
        folds[int(f)] = torch.load(mdir / f"fold_{int(f)}.pt", weights_only=False)
    shared = torch.load(mdir / "shared_stats.pt", weights_only=False)
    return {"manifest": man, "folds": folds, "shared": shared}


def build_model(man: dict, dims: dict, device: str) -> WTEGraphNet:
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
    ).to(device)


def build_anchors(
    base: np.ndarray,
    fac_raw: np.ndarray,
    deep_raw: np.ndarray,
    mirror_depth_m: float,
    mirror_on: bool,
) -> dict:
    """Anchor raw/presence/pred-dtw arrays, exactly the trainer's construction.

    ``base`` is wte_resid_base_m (z_surf - R); each prior's pred_dtw is its own
    DTW estimate ``base - raw`` (trainer L1216/L1231), presence is finiteness of
    the raw anomaly, and the mirror is the constant-depth surface ``base - d``.
    """
    fac_present = np.isfinite(fac_raw)
    deep_present = np.isfinite(deep_raw)
    a = {
        "fac_raw": fac_raw,
        "fac_present": fac_present,
        "fac_pred_dtw": np.where(fac_present, base - fac_raw, np.nan),
        "deep_raw": deep_raw,
        "deep_present": deep_present,
        "deep_pred_dtw": np.where(deep_present, base - deep_raw, np.nan),
    }
    if mirror_on:
        mirror_raw = base - mirror_depth_m
        mirror_present = np.isfinite(mirror_raw)
        a |= {
            "mirror_raw": mirror_raw,
            "mirror_present": mirror_present,
            "mirror_pred_dtw": np.where(mirror_present, mirror_depth_m, np.nan),
        }
    return a


def fold_tensors(ck: dict, frame: pd.DataFrame, anchors: dict, device: str) -> dict:
    """Per-fold query tensor + anchor tensors from the persisted contract."""
    y_c, y_s = float(ck["y_c"]), float(ck["y_s"])
    qx = torch.as_tensor(
        apply_stats(frame, ck["q_stats"]), dtype=torch.float32, device=device
    )
    feat = {"query_x": qx}
    feat |= _fac_feat(
        anchors["fac_raw"],
        anchors["fac_present"],
        y_c,
        y_s,
        device,
        anchors["fac_pred_dtw"],
        float(ck["fac_stats"]["pc"]),
        float(ck["fac_stats"]["ps"]),
    )
    feat |= _fac_feat(
        anchors["deep_raw"],
        anchors["deep_present"],
        y_c,
        y_s,
        device,
        anchors["deep_pred_dtw"],
        float(ck["deep_stats"]["dc"]),
        float(ck["deep_stats"]["ds"]),
        prefix="deep",
    )
    if ck.get("mirror_stats"):
        feat |= _fac_feat(
            anchors["mirror_raw"],
            anchors["mirror_present"],
            y_c,
            y_s,
            device,
            anchors["mirror_pred_dtw"],
            float(ck["mirror_stats"]["c"]),
            float(ck["mirror_stats"]["s"]),
            prefix="mirror",
        )
    return feat


def forward_fold(
    model: WTEGraphNet, ck: dict, graph: dict, feat: dict, r_wte: np.ndarray
) -> dict:
    """One fold's eval forward -> physical (meters) surfaces + gate weights."""
    y_c, y_s = float(ck["y_c"]), float(ck["y_s"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        native = model({**graph, **feat}).cpu().numpy().astype("float64")
    wte = r_wte + native * y_s + y_c
    out = {"wte": wte, "native": native}
    if model.sigma_log_b is not None:
        out["sigma"] = np.exp(model.sigma_log_b.cpu().numpy().reshape(-1)) * y_s
    out["w"] = model.last_prior_gate.cpu().numpy().astype("float64")
    out["head_wte"] = (
        r_wte + model.last_head_out.cpu().numpy().astype("float64") * y_s + y_c
    )
    return out


def assert_mixture_identity(
    out: dict, r_wte: np.ndarray, anchors: dict, fold: int, where: str
):
    """The section-0 identity: wte == sum_i w_i * expert_wte_i, <1e-3 m.

    Each prior expert's WTE surface is ``R + raw`` (its standardized base
    de-standardizes back to exactly that); the free head's is ``head_wte``.
    Absent experts carry softmax weight exactly 0 (the -1e9 mask underflows),
    so their NaN surface is zeroed out of the sum rather than imputed.
    """
    w = out["w"]
    expert_wtes = [r_wte + anchors["fac_raw"], r_wte + anchors["deep_raw"]]
    presents = [anchors["fac_present"], anchors["deep_present"]]
    if "mirror_raw" in anchors:
        expert_wtes.append(r_wte + anchors["mirror_raw"])
        presents.append(anchors["mirror_present"])
    expert_wtes.append(out["head_wte"])
    presents.append(np.ones(len(r_wte), bool))
    if w.shape[1] != len(expert_wtes):
        raise SystemExit(
            f"gate width {w.shape[1]} != {len(expert_wtes)} experts ({where})"
        )
    mix = np.zeros_like(out["wte"])
    for i, (ew, pres) in enumerate(zip(expert_wtes, presents)):
        mix += w[:, i] * np.where(pres, np.nan_to_num(ew), 0.0)
    err = float(np.max(np.abs(mix - out["wte"]))) if len(mix) else 0.0
    if not err < MIX_TOL_M:
        raise SystemExit(
            f"gate-mixture identity FAILED ({where}, fold {fold}): "
            f"max |sum w_i*expert_i - wte| = {err:.6f} m >= {MIX_TOL_M}"
        )


def prune_for_queries(
    rn: pd.DataFrame, ce: pd.DataFrame, lat: pd.DataFrame, channel_layers: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The trainer's exact prune+remap sequence on a fresh lateral attachment."""
    attached = np.unique(lat["reach_node_idx"].to_numpy("int64"))
    kept, edge_keep = prune_reach_graph(len(rn), ce, attached, channel_layers)
    old2new = np.full(len(rn), -1, dtype="int64")
    old2new[kept] = np.arange(len(kept), dtype="int64")
    rn = rn.iloc[kept].reset_index(drop=True)
    ce = ce[edge_keep].copy()
    ce["src_reach_idx"] = old2new[ce["src_reach_idx"].to_numpy("int64")]
    ce["dst_reach_idx"] = old2new[ce["dst_reach_idx"].to_numpy("int64")]
    lat = lat.copy()
    lat["reach_node_idx"] = old2new[lat["reach_node_idx"].to_numpy("int64")]
    assert (lat["reach_node_idx"] >= 0).all(), "attached reach pruned away (bug)"
    return rn, ce, lat


def reach_tensors(
    rn: pd.DataFrame, ce: pd.DataFrame, shared: dict, dims: dict, device: str
) -> dict:
    """Reach-side tensors (constant across query chunks of one pruned subgraph)."""
    reach_x = torch.as_tensor(
        apply_stats(rn, shared["reach_stats"]), dtype=torch.float32, device=device
    )
    ch_ea = torch.as_tensor(
        apply_stats(ce, shared["ch_stats"]), dtype=torch.float32, device=device
    )
    got = {"reach": reach_x.shape[1], "channel_edge": ch_ea.shape[1]}
    bad = {k: (got[k], dims[k]) for k in got if got[k] != dims[k]}
    if bad:
        raise SystemExit(f"train/infer feature-width skew (got, expected): {bad}")
    return {
        "reach_x": reach_x,
        "ch_ea": ch_ea,
        "ch_ei": torch.as_tensor(
            ce[["src_reach_idx", "dst_reach_idx"]].to_numpy().T,
            dtype=torch.long,
            device=device,
        ),
    }


def lateral_tensors(lat: pd.DataFrame, shared: dict, dims: dict, device: str) -> dict:
    lat_ea = torch.as_tensor(
        apply_stats(lat, shared["lat_stats"]), dtype=torch.float32, device=device
    )
    if lat_ea.shape[1] != dims["lateral_edge"]:
        raise SystemExit(
            f"lateral-edge feature-width skew: got {lat_ea.shape[1]}, "
            f"expected {dims['lateral_edge']}"
        )
    return {
        "lat_ea": lat_ea,
        "lat_ei": torch.as_tensor(
            lat[["reach_node_idx", "query_node_idx"]].to_numpy().T,
            dtype=torch.long,
            device=device,
        ),
    }


def run_folds(
    models: dict,
    graph: dict,
    frame: pd.DataFrame,
    r_wte: np.ndarray,
    anchors: dict,
    device: str,
    where: str,
) -> dict:
    """All folds forward + identity check -> fold-aggregated physical fields.

    Per-fold surfaces are de-standardized in meters BEFORE the fold combine
    (each fold has its own y_c/y_s/q_stats); the ensemble WTE/sigma/head are
    fold medians, the gate weights a renormalized fold mean, and fold_spread
    (p90-p10 of per-fold WTE) is the ensemble-disagreement layer.
    """
    man = models["manifest"]
    dims = dict(man["feature_dims"])
    model = build_model(man, dims, device)
    per_fold = []
    for f, ck in sorted(models["folds"].items()):
        feat = fold_tensors(ck, frame, anchors, device)
        if feat["query_x"].shape[1] != dims["query"]:
            raise SystemExit(
                f"query feature-width skew: got {feat['query_x'].shape[1]}, "
                f"expected {dims['query']}"
            )
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


# ---------------------------------------------------------------------------
# OOF round-trip gate
# ---------------------------------------------------------------------------
def oof_check(models: dict, model_dir: Path, device: str, tol_m: float) -> None:
    """Run the bundle's own wells through the checkpoints; compare archived OOF.

    Uses the bundle frames verbatim (features, R, anchors, lateral edges), so any
    disagreement isolates the persistence/assembly/forward path. The archived OOF
    came from train_fold's final eval-mode full-batch forward with best_state --
    the exact computation replayed here.
    """
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
    # bundle columns verbatim -- no recomputation, so a mismatch isolates the
    # persisted-contract replay rather than feature drift.
    r_wte = qn["regional_wte_idw_oof_m"].to_numpy("float64")
    anchors = build_anchors(
        qn[man["dtw_base_col"]].to_numpy("float64"),
        qn[man["fac_anchor_col"]].to_numpy("float64"),
        qn[man["deep_anchor_col"]].to_numpy("float64"),
        float(man["flags"]["mirror_depth_m"]),
        bool(man["flags"]["mirror_anchor"]),
    )
    agg = run_folds(models, graph, qn, r_wte, anchors, device, "oof-check")
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
            f">= {tol_m} (train/infer contract skew)"
        )
    log.info(
        "OOF round-trip PASSED: max |diff| %.2e m < %g m over 8 folds", worst, tol_m
    )


# ---------------------------------------------------------------------------
# Per-basin lattice inference
# ---------------------------------------------------------------------------
def basin_lattice(basin: str) -> tuple:
    """Snapped 100 m grid + inside-boundary cell centers for one basin."""
    boundary = gpd.read_file(HUC8_ROOT / basin / "basin_boundary.fgb").to_crs(5070)
    transform, width, height = snapped_window(tuple(boundary.total_bounds), RES)
    inside = rasterize(
        [(g, 1) for g in boundary.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    rows, cols = np.where(inside)
    qx = transform.c + (cols + 0.5) * RES
    qy = transform.f - (rows + 0.5) * RES
    return transform, width, height, rows, cols, qx, qy


def scatter(
    vals: np.ndarray, rows: np.ndarray, cols: np.ndarray, height: int, width: int
) -> np.ndarray:
    out = np.full((height, width), np.nan, dtype="float64")
    out[rows, cols] = vals
    return out


def infer_basin(
    basin: str,
    models: dict,
    wells: dict,
    rn_full: pd.DataFrame,
    ce_full: pd.DataFrame,
    geom: gpd.GeoDataFrame,
    bman: dict,
    args,
    device: str,
) -> None:
    man = models["manifest"]
    out_dir = HUC8_ROOT / basin / "gnn" / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    transform, width, height, rows, cols, qx, qy = basin_lattice(basin)
    z_surf = sample_coarse(args.dem, qx, qy)
    fin = np.isfinite(z_surf)
    if not fin.all():
        log.info(
            "%s: dropping %d/%d lattice cells with non-finite z_surf (DEM nodata)",
            basin,
            int((~fin).sum()),
            len(fin),
        )
        rows, cols, qx, qy, z_surf = (
            rows[fin],
            cols[fin],
            qx[fin],
            qy[fin],
            z_surf[fin],
        )
    n = len(qx)
    log.info("%s: %d lattice cells (%dx%d window)", basin, n, width, height)
    qxy = np.c_[qx, qy]

    # R: all-well relief-IDW WTE (inference-time counterpart of the crossfit prior)
    vw = float(bman["r_relief_vw"])
    k, p = int(bman["idw_k"]), float(bman["idw_power"])
    r_wte = idw_at_points(
        _relief_coords(wells["xy"], wells["z"], vw),
        wells["wte"],
        _relief_coords(qxy, z_surf, vw),
        k,
        p,
    )
    # deep expert: deep-pool IDW head, no relief lift (training contract)
    dd = bman["deep_datum"]
    deep_wte = idw_at_points(
        wells["xy"][wells["deep"]],
        wells["wte"][wells["deep"]],
        qxy,
        int(dd["idw_k_deep"]),
        p,
    )
    fac_dtw = sample_fac_rem(qx, qy)
    base = z_surf - r_wte
    anchors_all = build_anchors(
        base,
        np.where(np.isfinite(fac_dtw), (z_surf - fac_dtw) - r_wte, np.nan),
        deep_wte - r_wte,
        float(man["flags"]["mirror_depth_m"]),
        bool(man["flags"]["mirror_anchor"]),
    )
    log.info(
        "%s: FAC coverage %.1f%%, R [%.0f, %.0f] m",
        basin,
        100 * anchors_all["fac_present"].mean(),
        float(r_wte.min()),
        float(r_wte.max()),
    )

    # query feature frame -- schema lockstep with the training bundle
    frame = pd.DataFrame(
        {
            "query_node_idx": np.arange(n, dtype="int64"),
            "fac_rem_wte_anom_m": anchors_all["fac_raw"],
            "deep_regional_wte_anom_m": anchors_all["deep_raw"],
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
    missing = [c for c in man["query_feature_cols"] if c not in frame.columns]
    if missing:
        raise SystemExit(f"lattice frame lacks query features: {missing}")

    lat_all = build_lateral_edges(qxy, geom, None, int(bman["knn_lateral"]))
    lat_all = attach_lateral_attrs(
        lat_all, rn_full, z_surf, float(bman["conductance_p"])
    )
    # One prune on the union of attached reaches: the kept subgraph contains the
    # full channel_layers-hop receptive field of every attached reach, so chunked
    # forwards over query subsets are exact (reach states are query-independent
    # in the non-writeback branch).
    rn, ce, lat_all = prune_for_queries(
        rn_full, ce_full, lat_all, int(man["channel_layers"])
    )
    log.info(
        "%s: pruned reach subgraph %d nodes / %d channel edges / %d lateral edges",
        basin,
        len(rn),
        len(ce),
        len(lat_all),
    )
    dims = dict(man["feature_dims"])
    rt = reach_tensors(rn, ce, models["shared"], dims, device)
    if args.save_lattice:
        frame.assign(x5070=qx, y5070=qy, z_surf_m=z_surf, r_wte_m=r_wte).to_parquet(
            out_dir / "lattice_debug.parquet"
        )

    n_gate = 3 + int(bool(man["flags"]["mirror_anchor"]))
    agg = {
        "wte": np.full(n, np.nan),
        "head_wte": np.full(n, np.nan),
        "fold_spread": np.full(n, np.nan),
        "w": np.full((n, n_gate), np.nan),
    }
    if man["flags"]["sigma_head"]:
        agg["sigma"] = np.full(n, np.nan)
    qidx = lat_all["query_node_idx"].to_numpy("int64")
    for i0 in range(0, n, args.query_chunk):
        i1 = min(i0 + args.query_chunk, n)
        sub = frame.iloc[i0:i1].reset_index(drop=True)
        lat_c = lat_all[(qidx >= i0) & (qidx < i1)].copy()
        lat_c["query_node_idx"] = lat_c["query_node_idx"] - i0
        graph = {**rt, **lateral_tensors(lat_c, models["shared"], dims, device)}
        anchors_c = {k: v[i0:i1] for k, v in anchors_all.items()}
        ch = run_folds(
            models, graph, sub, r_wte[i0:i1], anchors_c, device, f"{basin}[{i0}:{i1}]"
        )
        for key in ("wte", "head_wte", "fold_spread", "sigma"):
            if key in ch and key in agg:
                agg[key][i0:i1] = ch[key]
        agg["w"][i0:i1] = ch["w"]
        if i1 < n:
            log.info("%s: %d/%d cells done", basin, i1, n)
    if not np.isfinite(agg["wte"]).all():
        raise SystemExit(f"{basin}: non-finite WTE cells after all chunks (bug)")

    dtw = z_surf - agg["wte"]
    neg = float((dtw < 0).mean())
    log.info(
        "%s: DTW median %.2f m, %.1f%% negative (predicted artesian/inundation)",
        basin,
        float(np.median(dtw)),
        100 * neg,
    )
    tags = {
        "wells": "monitoring_only",
        "bundle": Path(man["graph_dir"]).name,
        "model": args.model_name,
        "surface": "contemporary_ambient_por_mean",
    }
    layers = {
        "gnn_wte_100m.tif": agg["wte"],
        "gnn_dtw_100m.tif": dtw,
        "gnn_head_wte_100m.tif": agg["head_wte"],
        "gnn_deep_wte_100m.tif": deep_wte,
        "gnn_r_wte_100m.tif": r_wte,
        "gnn_fold_spread_100m.tif": agg["fold_spread"],
    }
    if "sigma" in agg:
        layers["gnn_sigma_100m.tif"] = agg["sigma"]
    for name, vals in layers.items():
        write_tif(out_dir / name, scatter(vals, rows, cols, height, width), transform)
        with rasterio.open(out_dir / name, "r+") as dst:
            dst.update_tags(**tags)
    gate = np.stack(
        [
            scatter(agg["w"][:, i], rows, cols, height, width)
            for i in range(agg["w"].shape[1])
        ]
    )
    write_tif(
        out_dir / "gnn_gate_w_100m.tif",
        gate,
        transform,
        count=gate.shape[0],
        descs=[f"w_{e}" for e in man["gate_experts"]],
    )
    with rasterio.open(out_dir / "gnn_gate_w_100m.tif", "r+") as dst:
        dst.update_tags(**tags)
    (out_dir / "infer_run.json").write_text(
        json.dumps(
            {
                "basin": basin,
                "model_dir": str(args.model_dir),
                "n_cells": n,
                "window": [width, height],
                "pct_negative_dtw": neg,
                "mixture_identity_tol_m": MIX_TOL_M,
                **tags,
            },
            indent=2,
        )
    )
    log.info("%s: wrote %d coarse layers -> %s", basin, len(layers) + 1, out_dir)


def nm_basins(state: str) -> list[str]:
    wbd = pd.read_parquet(WBD_HU8, columns=["huc8", "states"])
    codes = set(
        wbd[wbd["states"].astype(str).str.contains(state.upper(), na=False)][
            "huc8"
        ].astype(str)
    )
    have = {d.name for d in HUC8_ROOT.iterdir() if d.is_dir()}
    out = sorted(codes & have)
    if not out:
        raise SystemExit(f"no on-disk basins for state {state}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--basins", nargs="+", help="explicit basin dirs under huc8/")
    ap.add_argument("--state", help="WBD state filter (e.g. NM) over on-disk basins")
    ap.add_argument("--dem", default=DEM)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--oof-check", action="store_true")
    ap.add_argument("--oof-tol-m", type=float, default=1e-3)
    ap.add_argument(
        "--query-chunk",
        type=int,
        default=1_000_000,
        help="lattice cells per forward pass (bounds GPU memory; chunking is exact)",
    )
    ap.add_argument("--save-lattice", action="store_true")
    ap.add_argument(
        "--overwrite", action="store_true", help="re-run basins with existing output"
    )
    args = ap.parse_args()
    args.model_dir = Path(args.model_dir)
    args.model_name = args.model_dir.name
    models = load_models(args.model_dir)
    man = models["manifest"]

    if args.oof_check:
        oof_check(models, args.model_dir, args.device, args.oof_tol_m)
        return

    if not args.basins and not args.state:
        raise SystemExit("pass --basins or --state (or --oof-check)")
    basins = args.basins or nm_basins(args.state)

    gdir = Path(man["graph_dir"])
    bman = json.loads((gdir / "graph_manifest.json").read_text())
    qn = pd.read_parquet(gdir / "query_nodes.parquet")
    wells = {
        "xy": qn[["x5070", "y5070"]].to_numpy("float64"),
        "z": qn[man["surface_elev_col"]].to_numpy("float64"),
        "wte": qn[man["obs_wte_col"]].to_numpy("float64"),
        "deep": deep_well_mask(
            qn,
            float(bman["deep_datum"]["quantile"]),
            str(bman["deep_datum"]["unit"]),
            int(bman["deep_datum"]["min_per_unit"]),
        ),
    }
    log.info(
        "wells: %d (bundle %s), deep pool %d", len(qn), gdir.name, wells["deep"].sum()
    )
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
    log.info(
        "CONUS frame: %d reaches, %d channel edges, %d flowline rep-points",
        len(rn_full),
        len(ce_full),
        len(geom),
    )

    done = skipped = 0
    for b in basins:
        out_dir = HUC8_ROOT / b / "gnn" / args.model_name
        if not args.overwrite and (out_dir / "gnn_dtw_100m.tif").exists():
            skipped += 1
            continue
        infer_basin(b, models, wells, rn_full, ce_full, geom, bman, args, args.device)
        done += 1
    log.info("inference complete: %d basins run, %d skipped (existing)", done, skipped)


if __name__ == "__main__":
    main()
