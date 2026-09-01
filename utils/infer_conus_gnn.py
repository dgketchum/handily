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
  gnn_deep_wte_100m.tif   deep prior WTE (crossfit-field interpolation)
  gnn_r_wte_100m.tif      base R WTE (crossfit-field interpolation)
  gnn_fold_spread_100m.tif p90-p10 of per-fold WTE

R and the deep prior are leak-free by construction (docs/inference_leakage_prevention.md):
the default interpolates the bundle's ARCHIVED leave-fold-out crossfit feature
values (relief-kNN-IDW of regional_wte_idw_oof_m / deep_regional_wte_idw_oof_m),
so no well's own observation can reach its neighborhood, the surface has no
exclusion-disk seams, and at well locations it reproduces the training features
exactly (pin error 0.000 m at the 34,503 bundle wells) -- the map is the
seamless spatial extension of the audited OOF field. All-well IDW of raw obs
here self-pins R to training labels and prints bulls-eyes into the rendered
maps (the 2026-07 leakage incident, notes/LEAKAGE_AUDIT.md L1); the calibrated
leave-radius-out variant (--r-source obs-exclude) is shelved: it matches the
residual distribution but draws arc/annulus seams where wells cross the
exclusion-disk edge (notes/SHELVED_LEVERS.md).

Every fold's forward is checked against the gate-mixture identity
(wte == sum_i w_i * expert_wte_i, <1e-3 m) before anything is written; the
10 m render (render_gnn_10m.py) then recomposes from these layers exactly.
After each basin a LEAK GATE compares the coarse DTW sampled at the bundle's
in-basin wells against the archived OOF predictions: a map that beats its own
OOF MAD at training wells by more than --leak-gate-frac AND departs from the
OOF predictions per-well (median |map-oof| > --leak-gate-track-m) fails the
run; a ratio trip that still tracks OOF passes as "pass_ratio_tripped".

``--oof-check`` runs the bundle's own wells through the persisted checkpoints
and compares against the archived OOF predictions -- the end-to-end round-trip
gate (stats + tensor assembly + forward) that must pass before lattice output
is trusted.

Source-assimilation arms (``--source-edges``, the r1e recipe) render too: every
lattice cell gets fresh cell<-well kNN source edges built with the frozen
bundle's recipe (``deployment_source_ctx``, shared with predict_gnn_at_points.py),
so the map READS the well pool at inference time. Two consequences:

- ``--query-writeback`` arms make reach states depend on the whole query set, so
  the basin is forwarded in ONE batch (``--query-chunk`` is ignored, and logged);
  a chunked writeback forward would be a different computation.
- the LEAK GATE becomes INFORMATIONAL: a cell sitting at a training well reads
  that well by design, so beating OOF there is intended behaviour, not leakage.
  The panel is still computed and written to infer_run.json, never enforced.

``--extra-sources`` admits an external observation table (e.g. the frozen NDWR
admission set) into the kNN source pool alongside the bundle wells.

``--water-flatten-inference`` zeroes the FAC-REM input at verified-water cells
(the V3 mask, /nas/handily/covariates/water_mask_v3) at inference time only,
weights frozen: over water the gate locks onto the FAC expert, so a wrong
FAC-REM depth is passed straight through and the map reads metres deep at real
rivers and lakes. This is a RENDERER-SIDE override, independent of the bundle's
``water_v3`` block (the production r1e weights come from a bundle that has
none, and the retrain that baked the treatment into training was NO-GO --
notes/WATER_V3_EVAL.md); it moves the FAC input only, never R.
``--water-flatten-ramp`` additionally halves FAC on off-mask cells touching the
mask, tapering the one-cell shoreline step the hard flatten leaves.

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
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conus_graph_inputs import (  # noqa: E402
    DEM,
    WATER_BLOCKS_PARQUET,
    WATER_V3_MASK_TIF,
    _relief_coords,
    _sample_gsw_occurrence,
    attach_lateral_attrs,
    build_lateral_edges,
    deep_well_mask,
    sample_drilled_depth,
    sample_gridmet,
    sample_modis_jja_wetness,
    sample_relief_etrm,
    sample_terrain_multiscale,
    water_query_features,
    water_v3_inference_block,
)
from build_dupuit_wte import build_boundaries, hang_interp  # noqa: E402
from build_source_edges import EDGE_COLS as SOURCE_EDGE_COLS  # noqa: E402
from build_water_mask_v3 import RAMP_FACTOR, water_flatten_factor  # noqa: E402
from build_stacker_features import sample_coarse  # noqa: E402
from fac_rem_registry import sample_fac_rem  # noqa: E402
from train_conus_gnn import _fac_feat, prune_reach_graph  # noqa: E402
from train_wte_gnn import WTEGraphNet, apply_stats, fit_stats  # noqa: E402

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


def well_pool(qn: pd.DataFrame) -> pd.DataFrame:
    """Real wells only: water pseudo-rows are labels, never prior sources.

    The training-time crossfit priors (relief-IDW R, deep-datum IDW) pooled on
    wells exclusively; the inference-time all-well IDW counterparts must match,
    or the gate sees R/deep surfaces drawn toward stage — a distribution the
    checkpoints never trained on.
    """
    if "is_water_pseudo" not in qn.columns:
        return qn
    water = qn["is_water_pseudo"].astype(bool)
    if water.any():
        log.info(
            "well pool: dropped %d/%d water pseudo-rows (labels, never prior sources)",
            int(water.sum()),
            len(qn),
        )
    return qn[~water].reset_index(drop=True)


def idw_exclude_at_points(
    pool_coords: np.ndarray,
    pool_val: np.ndarray,
    pool_xy: np.ndarray,
    query_coords: np.ndarray,
    query_xy: np.ndarray,
    k: int,
    power: float,
    exclude_m: float,
    chunk: int = 250_000,
) -> np.ndarray:
    """Leave-radius-out kNN IDW: the inference counterpart of ``crossfit_idw``.

    Wells inside a horizontal disk of ``exclude_m`` around each query point are
    excluded from its neighbor set; ranking and inverse-distance weights run on
    ``pool_coords``/``query_coords`` (relief-lifted or plain, per the trained
    contract), while the exclusion disk is horizontal (``*_xy``) because the
    training-time fold exclusion removed a horizontal HUC12 cluster. Candidates
    are over-queried (k+96, doubling adaptively for dense clusters) and the
    first k survivors per query are kept -- exactly "k nearest of the allowed
    pool". ``exclude_m=0`` reproduces ``idw_at_points`` output.
    """
    tree = cKDTree(pool_coords)
    n_pool = len(pool_val)
    out = np.full(len(query_coords), np.nan)
    for c0 in range(0, len(query_coords), chunk):
        c1 = min(c0 + chunk, len(query_coords))
        todo = np.arange(c0, c1)
        kk = min(n_pool, k + 96)
        while len(todo):
            dist, idx = tree.query(query_coords[todo], k=kk)
            if kk == 1:
                dist, idx = dist[:, None], idx[:, None]
            dx = pool_xy[idx, 0] - query_xy[todo, None, 0]
            dy = pool_xy[idx, 1] - query_xy[todo, None, 1]
            keep = (dx * dx + dy * dy) >= exclude_m * exclude_m
            n_keep = keep.sum(1)
            enough = n_keep >= min(k, n_pool)
            if kk >= n_pool:
                enough = n_keep > 0  # renormalize over whatever survives
                if not enough.all():
                    raise SystemExit(
                        f"idw_exclude_at_points: {int((~enough).sum())} query "
                        f"points have NO wells outside the {exclude_m:.0f} m "
                        "exclusion disk"
                    )
            used = keep & (np.cumsum(keep, axis=1) <= k)
            w = np.where(used, 1.0 / np.maximum(dist, 1.0) ** power, 0.0)
            num = (w * pool_val[idx]).sum(1)
            den = w.sum(1)
            rows = np.where(enough)[0]
            out[todo[rows]] = num[rows] / den[rows]
            todo = todo[~enough]
            if kk >= n_pool:
                break
            kk = min(n_pool, kk * 2)
    return out


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


def build_model(
    man: dict,
    dims: dict,
    device: str,
    f_mae: int | None = None,
    writeback: bool = False,
    f_src: int | None = None,
    f_srcedge: int | None = None,
) -> WTEGraphNet:
    """Model in the checkpoint's shape.

    The extension args default OFF, so a plain prior-gate arm builds exactly the
    network it always did; they are detected from the checkpoint state dict
    (``ckpt_extensions`` / ``ckpt_f_src`` / ``ckpt_f_srcedge``) because the
    inference manifest does not record the MAE head or its width.
    """
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
    ea = apply_stats(se, fit_stats(se, list(SOURCE_EDGE_COLS), None))
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


def load_extra_sources(
    path: str, geom: gpd.GeoDataFrame, rn_full: pd.DataFrame
) -> pd.DataFrame:
    """Admitted external source wells + their controlling-reach FAC basin.

    ``path`` is a frozen admission table (x5070/y5070/wte_residual_m/z_surf_m,
    residual in the bundle's frame: (z_surf - dtw) - r_wte). The basin label is
    the identity the source-edge ``same_basin`` attr compares, so it is attached
    here with a rank-0 lateral attachment against the UNPRUNED reach frame.
    """
    extra = pd.read_parquet(path)
    need = ["x5070", "y5070", "wte_residual_m", "z_surf_m"]
    missing = [c for c in need if c not in extra.columns]
    if missing:
        raise SystemExit(f"--extra-sources missing columns {missing}")
    exy = extra[["x5070", "y5070"]].to_numpy("float64")
    elat = build_lateral_edges(exy, geom, None, 1)
    basin_of_reach = rn_full.set_index("reach_node_idx")["basin"]
    extra = extra[need].copy()
    extra["basin"] = (
        elat[elat["is_controlling"].astype(bool)]
        .drop_duplicates("query_node_idx")
        .set_index("query_node_idx")["reach_node_idx"]
        .reindex(np.arange(len(extra)))
        .map(basin_of_reach)
        .to_numpy()
    )
    log.info(
        "extra sources: %d admitted wells from %s (%.0f%% with basin)",
        len(extra),
        path,
        100.0 * float(pd.notna(extra["basin"]).mean()),
    )
    return extra


def dd_dup_features(
    bman: dict,
    gdir: Path,
    qxy: np.ndarray,
    z_surf: np.ndarray,
    self_exclude_m: float = 0.0,
    boundaries: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """drilled-depth + Dupuit-hang query features (dd/dup arms), shared recipe.

    Both are well-free at inference time: drilled depth is construction
    metadata (kNN IDW-mean + p90 over the GWX unconfined pool, no cross-fit
    needed) and the hang features are a kNN-IDW of the bundle's top-2-Strahler
    boundary reach elevations. ``self_exclude_m`` is the drilled-depth
    self/nest guard: 0 = deployment semantics (a lattice cell is not a well),
    100 = the training-semantics sensitivity variant the point predictor
    exposes. ``boundaries`` lets a multi-basin run build the boundary set once
    (it is a CONUS-wide table).
    """
    dd = bman["drilled_depth"]
    out = dict(
        sample_drilled_depth(
            dd["points"],
            qxy,
            int(dd["k"]),
            float(dd["power"]),
            self_exclude_m=float(self_exclude_m),
        )
    )
    dh = bman["dupuit_hang"]
    bnd = (
        boundaries
        if boundaries is not None
        else build_boundaries(
            str(gdir / "reach_nodes.parquet"),
            bman["sources"]["geom"],
            top_orders=int(dh["top_orders"]),
        )
    )
    hang_wte, d_bnd = hang_interp(
        bnd[["cx", "cy"]].to_numpy("float64"),
        bnd["reach_elev_m"].to_numpy("float64"),
        qxy,
        k=int(dh["idw_k"]),
        power=float(dh["idw_power"]),
    )
    out["dupuit_hang_dtw_m"] = z_surf - hang_wte
    out["log1p_dupuit_d_m"] = np.log1p(d_bnd)
    return out


def deployment_source_ctx(
    man: dict,
    bman: dict,
    gdir: Path,
    qxy: np.ndarray,
    z_surf: np.ndarray,
    lat: pd.DataFrame,
    rn_full: pd.DataFrame,
    device: str,
    exclude_m: float = 0.0,
    extra: pd.DataFrame | None = None,
) -> dict:
    """Deployment source context for query points OUTSIDE the bundle.

    Used identically by the point predictor (a well set) and the renderer (a
    100 m lattice's cell centres). The source side of the read carries ONLY the
    encoded observed residual -- the model never consumes source covariates
    through this path -- so the bundle wells stay OUT of the query set. We build
    fresh query<-well kNN edges with the build_source_edges recipe (same
    relief-lifted metric, same k, same 4 attrs incl. FAC-basin identity),
    standardize them with the trainer's GLOBAL fit over the frozen bundle edge
    table (trainer-verbatim, like bundle_source_edges), and expose every
    eligible well to every fold: deployment leaves no observation on the table,
    and external queries carry no labels into the forward, so the training
    protocol's same-site/masking guards do not apply. ``exclude_m`` > 0 drops
    edges from wells within that radius of the query point (an eval-fairness
    sensitivity knob; 0 = deployment semantics -- a map cell at a well reads
    that well).

    ``extra`` admits sources beyond the bundle (e.g. a frozen NDWR admission
    table): columns x5070/y5070/wte_residual_m/z_surf_m/basin, where
    wte_residual_m is in the bundle's frame ((z_surf - dtw) - r_wte, same R)
    and basin is the controlling-reach FAC basin label (NaN -> never
    same_basin). Extra sources join the kNN pool on equal footing.

    ``lat`` must be the PRE-prune lateral attachment: prune_for_queries remaps
    reach_node_idx to the pruned numbering, but the basin lookup is keyed on
    ``rn_full``'s original ids.
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
    resid = w["wte_residual_m"].to_numpy("float64")
    n_extra = 0
    if extra is not None and len(extra):
        n_extra = len(extra)
        wxy = np.vstack([wxy, extra[["x5070", "y5070"]].to_numpy("float64")])
        wz = np.concatenate(
            [wz, np.nan_to_num(extra["z_surf_m"].to_numpy("float64"), nan=0.0)]
        )
        resid = np.concatenate([resid, extra["wte_residual_m"].to_numpy("float64")])
        src_basin = pd.concat(
            [src_basin.reset_index(drop=True), extra["basin"].reset_index(drop=True)],
            ignore_index=True,
        )
    pctrl = (
        lat[lat["is_controlling"].astype(bool)]
        .drop_duplicates("query_node_idx")
        .set_index("query_node_idx")["reach_node_idx"]
    )
    dest_basin = (
        pd.Series(np.arange(len(qxy), dtype="int64")).map(pctrl).map(basin_of_reach)
    )

    pad = 8 if exclude_m > 0 else 0
    kq = min(k + pad, len(resid))
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
    stats_panel = {
        "n_edges": int(len(ed)),
        "n_query_with_edges": int(deg.size),
        "n_query": int(len(qxy)),
        "k": k,
        "relief_vw": vw,
        "exclude_m": float(exclude_m),
        "n_sources": int(len(resid)),
        "n_sources_bundle": int(len(w)),
        "n_sources_extra": int(n_extra),
        "median_rank0_geo_dist_km": float(
            np.expm1(ed.groupby("dest")["log1p_geo_dist_km"].min().median())
        ),
        "frac_same_basin": float(ed["same_basin"].mean()),
    }
    log.info(
        "source edges: %d edges to %d/%d query points (k=%d, vw=%.0f, "
        "exclude_m=%.0f, %d eligible wells (%d bundle + %d extra), "
        "median rank-0 geo %.2f km, frac_same_basin %.2f)",
        stats_panel["n_edges"],
        stats_panel["n_query_with_edges"],
        stats_panel["n_query"],
        k,
        vw,
        exclude_m,
        stats_panel["n_sources"],
        stats_panel["n_sources_bundle"],
        n_extra,
        stats_panel["median_rank0_geo_dist_km"],
        stats_panel["frac_same_basin"],
    )
    ea = apply_stats(ed, stats)
    src_t = torch.as_tensor(
        ed["src"].to_numpy("int64"), dtype=torch.long, device=device
    )
    dst_t = torch.as_tensor(
        ed["dest"].to_numpy("int64"), dtype=torch.long, device=device
    )
    return {
        "resid": resid,
        "pool_by_fold": lambda f: np.ones(len(resid), dtype=bool),
        "edges": {
            "src": src_t,
            "ei": torch.stack([src_t, dst_t]),
            "ea": torch.as_tensor(ea, dtype=torch.float32, device=device),
        },
        "stats": stats_panel,
    }


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
    f_mae: int | None = None,
    writeback: bool = False,
    src_ctx: dict | None = None,
) -> dict:
    """All folds forward + identity check -> fold-aggregated physical fields.

    Per-fold surfaces are de-standardized in meters BEFORE the fold combine
    (each fold has its own y_c/y_s/q_stats); the ensemble WTE/sigma/head are
    fold medians, the gate weights a renormalized fold mean, and fold_spread
    (p90-p10 of per-fold WTE) is the ensemble-disagreement layer.

    ``src_ctx`` (source-assimilation arms only) supplies the assimilation
    inputs: ``resid`` = the source-indexed observed wte_residual_m (NaN where no
    obs) and ``pool_by_fold(f)`` = the boolean mask of sources VISIBLE to fold
    f's forward. Values are standardized per fold by that checkpoint's y_c/y_s,
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
    model = build_model(
        man, dims, device, f_mae, writeback, f_src=f_src, f_srcedge=f_srcedge
    )
    per_fold = []
    for f, ck in sorted(models["folds"].items()):
        feat = fold_tensors(ck, frame, anchors, device)
        if feat["query_x"].shape[1] != dims["query"]:
            raise SystemExit(
                f"query feature-width skew: got {feat['query_x'].shape[1]}, "
                f"expected {dims['query']}"
            )
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
    _, writeback = ckpt_extensions(models["folds"][0])
    if writeback:
        graph["lat_ei_reversed"] = graph["lat_ei"].flip(0)
    src_ctx = None
    if ckpt_f_srcedge(models["folds"][0]) is not None:
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
            "edges": bundle_source_edges(man, device),
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
    agg = run_folds(
        models,
        graph,
        qn,
        r_wte,
        anchors,
        device,
        "oof-check",
        writeback=writeback,
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


def add_water_flatten_args(ap: argparse.ArgumentParser) -> None:
    """Register the inference-time water-flatten flags (renderer + point path)."""
    ap.add_argument(
        "--water-flatten-inference",
        action="store_true",
        help="zero fac_rem_dtw_m at verified-water cells (V3 mask) at inference "
        "time, frozen weights: the FAC input the model gate-locks onto over "
        "water is wrong there, so the map reads metres deep at real rivers and "
        "lakes. Renderer-side override, independent of the bundle's water_v3 "
        "block; FAC input only, no R pin",
    )
    ap.add_argument(
        "--water-flatten-ramp",
        action="store_true",
        help="--water-flatten-inference seam mode: off-mask cells with a masked "
        f"8-neighbour get fac_rem_dtw_m x {RAMP_FACTOR:g} (one-cell shoreline "
        "taper) instead of the hard mask-edge step",
    )
    ap.add_argument(
        "--water-flatten-mask",
        default=WATER_V3_MASK_TIF,
        help="V3 verified-water mask raster backing --water-flatten-inference",
    )


def water_flatten_override(
    args, qxy: np.ndarray, fac_dtw: np.ndarray
) -> tuple[np.ndarray, dict | None]:
    """Inference-time FAC flatten at verified water; returns (fac_dtw, provenance).

    Applied AFTER ``water_v3_inference_block`` and BEFORE the FAC anomaly / the
    gate's FAC anchor are formed, so the ONE sampled ``fac_dtw`` that feeds both
    the ``fac_rem_dtw_m`` query feature and the FAC expert anchor is flattened --
    the same substitution the frozen-weight probe made by swapping the FAC raster
    (notes/NAIP_WATER_EVIDENCE_PLAN.md section 11, QUALIFIED GO).

    Deliberately NOT bundle-driven: the production r1e weights were trained on
    graph_conus_monitoring_water_v2, which carries no ``water_v3`` block, and the
    retrain that baked the treatment into training was evaluated NO-GO
    (notes/WATER_V3_EVAL.md). Weights stay frozen; only the input moves. No R pin
    -- that was part of the rejected by-fiat mechanism.

    A no-op returning ``(fac_dtw, None)`` unless ``--water-flatten-inference``.
    """
    if not getattr(args, "water_flatten_inference", False):
        return fac_dtw, None
    mask_path = str(args.water_flatten_mask)
    ramp = bool(args.water_flatten_ramp)
    factor = water_flatten_factor(qxy[:, 0], qxy[:, 1], mask_path, ramp=ramp)
    on = factor == 0.0
    tapered = (factor > 0.0) & (factor < 1.0)
    touched = (factor < 1.0) & np.isfinite(fac_dtw)
    prov = {
        "mode": "ramp" if ramp else "hard",
        "mask_raster": mask_path,
        "mask_manifest": str(Path(mask_path).with_name("build_manifest.json")),
        "ramp_factor": RAMP_FACTOR if ramp else None,
        "n_points": int(len(factor)),
        "n_on_mask": int(on.sum()),
        "n_ramp_neighbors": int(tapered.sum()),
        "n_fac_touched": int(touched.sum()),
        "median_fac_pre_flatten_m": (
            float(np.nanmedian(fac_dtw[on]))
            if (on & np.isfinite(fac_dtw)).any()
            else None
        ),
        "pins_r": False,
        "note": "renderer-side inference-time override, frozen weights: "
        "fac_rem_dtw_m only (query feature + FAC anchor), R untouched",
    }
    log.info(
        "water flatten (%s, %s): %d/%d points on mask, %d ramp neighbours, "
        "%d FAC values moved (median on-mask FAC pre-flatten %s m)",
        prov["mode"],
        Path(mask_path).name,
        prov["n_on_mask"],
        prov["n_points"],
        prov["n_ramp_neighbors"],
        prov["n_fac_touched"],
        "n/a"
        if prov["median_fac_pre_flatten_m"] is None
        else f"{prov['median_fac_pre_flatten_m']:.2f}",
    )
    return fac_dtw * factor, prov


def leak_gate(
    basin: str,
    dtw_grid: np.ndarray,
    transform,
    oof: pd.DataFrame,
    frac: float,
    min_wells: int,
    track_tol_m: float,
    informational: bool = False,
) -> dict:
    """Fail loud if the coarse map beats its own OOF at training wells.

    Samples the basin DTW grid at the bundle's in-window wells and compares
    MAD(map - obs) against MAD(oof_pred - obs) from the archived
    gnn_oof_predictions.parquet. A leak-free map sits near the OOF error
    (fold-median ensembling buys a little; the 2026-07 all-well-IDW leak bought
    ~5x). Failure requires BOTH signals: map MAD < frac * OOF MAD AND the map
    departing from the OOF predictions per-well (median |map - oof| >
    track_tol_m). The second condition is the leak signature proper — labels
    pull the map off the model's honest predictions toward obs (5.29 m in the
    2026-07 incident vs ~0.55 m for clean maps). The MAD ratio alone
    false-positives on small bimodal panels where fold-median ensembling beats
    the single held-out checkpoint in the high-spread deep regime (13040100:
    ratio 0.41 with R pinned to 0.010 m and zero obs-collapsed wells). A ratio
    trip that tracks OOF passes with status "pass_ratio_tripped" for review.
    Returns the gate panel for infer_run.json.

    ``informational`` (source-assimilation arms) computes and records the same
    panel but never fails: a source-armed render legitimately reads the well
    pool, so a cell at a training well pins toward that well's observation by
    design (deployment semantics) -- both gate signals fire on the intended
    behaviour and the leak premise no longer holds. The panel is still the
    place to watch how hard the map is pinning.
    """
    x = oof["x5070"].to_numpy("float64")
    y = oof["y5070"].to_numpy("float64")
    c = np.floor((x - transform.c) / RES).astype("int64")
    r = np.floor((transform.f - y) / RES).astype("int64")
    h, w = dtw_grid.shape
    inb = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    obs = oof["obs_dtw_m"].to_numpy("float64")
    pred = oof["gnn_dtw_m"].to_numpy("float64")
    map_dtw = np.full(len(oof), np.nan)
    map_dtw[inb] = dtw_grid[r[inb], c[inb]]
    fin = np.isfinite(map_dtw) & np.isfinite(obs) & np.isfinite(pred)
    n = int(fin.sum())
    if n < min_wells:
        log.info("%s: leak gate skipped (%d in-basin wells < %d)", basin, n, min_wells)
        return {"n_wells": n, "status": "skipped"}
    mad_map = float(np.median(np.abs(map_dtw[fin] - obs[fin])))
    mad_oof = float(np.median(np.abs(pred[fin] - obs[fin])))
    gap = float(np.median(np.abs(map_dtw[fin] - pred[fin])))
    panel = {
        "n_wells": n,
        "mad_map_vs_obs_m": mad_map,
        "mad_oof_vs_obs_m": mad_oof,
        "median_abs_map_minus_oof_m": gap,
        "fail_frac": frac,
        "track_tol_m": track_tol_m,
        "status": "pass",
    }
    log.info(
        "%s: leak gate n=%d, map MAD %.2f m vs OOF MAD %.2f m, |map-oof| median %.2f m",
        basin,
        n,
        mad_map,
        mad_oof,
        gap,
    )
    if informational:
        panel |= {
            "status": "informational_source_arm",
            "note": "source-assimilation arm: the render reads the well pool by "
            "design (a cell at a well reads that well), so the leak-gate premise "
            "-- map must not beat its own OOF at training wells -- does not apply; "
            "panel recorded, never enforced",
        }
        log.warning(
            "%s: leak gate INFORMATIONAL (source-assimilation arm): the render "
            "assimilates the well pool by design, so beating OOF at training "
            "wells is the intended behaviour, not leakage -- panel recorded, "
            "not enforced",
            basin,
        )
        return panel
    if mad_map < frac * mad_oof:
        if gap > track_tol_m:
            raise SystemExit(
                f"{basin}: LEAK GATE FAILED -- map MAD {mad_map:.2f} m beats OOF MAD "
                f"{mad_oof:.2f} m at training wells by more than {frac:.2f}x and the "
                f"map departs from the OOF predictions (|map-oof| median {gap:.2f} m "
                f"> {track_tol_m:.2f} m): training labels are reaching the lattice "
                "features; docs/inference_leakage_prevention.md"
            )
        panel["status"] = "pass_ratio_tripped"
        log.warning(
            "%s: leak gate MAD ratio tripped (%.2f < %.2f x %.2f m) but map tracks "
            "OOF (|map-oof| median %.2f m <= %.2f m) -- ensemble-variance false "
            "positive, passing for review",
            basin,
            mad_map,
            frac,
            mad_oof,
            gap,
            track_tol_m,
        )
    return panel


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
    water_xy: np.ndarray | None = None,
    oof: pd.DataFrame | None = None,
    extra_sources: pd.DataFrame | None = None,
    boundaries: pd.DataFrame | None = None,
) -> None:
    man = models["manifest"]
    f_mae, writeback = ckpt_extensions(models["folds"][0])
    f_srcedge = ckpt_f_srcedge(models["folds"][0])
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

    # R + deep prior, leak-free. Default "crossfit": plain relief-kNN-IDW
    # interpolation of the ARCHIVED leave-fold-out crossfit feature values --
    # never a well's own obs, no exclusion disk (so no seams), and exact at
    # well locations (pin error 0.000 m; |R - obs| median/p90 5.03/30.42 m ==
    # archived, on the 34,503 bundle wells). "obs-exclude" is the shelved
    # leave-radius-out lever on raw obs (calibrated 3.5/4.5 km disks):
    # distribution-matched but draws arc seams where wells cross the disk
    # edge, with smooth annuli around extreme-residual wells.
    vw = float(bman["r_relief_vw"])
    k, p = int(bman["idw_k"]), float(bman["idw_power"])
    q_co = _relief_coords(qxy, z_surf, vw)
    pool_co = _relief_coords(wells["xy"], wells["z"], vw)
    if args.r_source == "crossfit":
        # deep uses the same k as R: this interpolates the archived per-well
        # deep-feature field (defined at every well), not a re-estimate from
        # the deep pool, so idw_k_deep does not apply.
        r_wte = idw_exclude_at_points(
            pool_co, wells["r_cross"], wells["xy"], q_co, qxy, k, p, 0.0
        )
        deep_wte = idw_exclude_at_points(
            pool_co, wells["deep_cross"], wells["xy"], q_co, qxy, k, p, 0.0
        )
    else:
        r_wte = idw_exclude_at_points(
            pool_co,
            wells["wte"],
            wells["xy"],
            q_co,
            qxy,
            k,
            p,
            args.r_exclude_km * 1000.0,
        )
        dd = bman["deep_datum"]
        deep_wte = idw_exclude_at_points(
            _relief_coords(wells["xy"][wells["deep"]], wells["z"][wells["deep"]], vw),
            wells["wte"][wells["deep"]],
            wells["xy"][wells["deep"]],
            q_co,
            qxy,
            int(dd["idw_k_deep"]),
            p,
            args.deep_exclude_km * 1000.0,
        )
    fac_dtw = sample_fac_rem(qx, qy)
    # screened-water treatment (bundle-driven): flatten FAC and pin R on the mask
    # BEFORE the base residual and the gate's FAC anchor are formed, so the
    # lattice sees exactly what the water pseudo-rows saw in training.
    v3_cols, fac_dtw, r_wte, _on_water_v3 = water_v3_inference_block(
        bman, qxy, z_surf, fac_dtw, r_wte, args.dem
    )
    # renderer-side flatten (--water-flatten-inference), frozen weights: same
    # single fac_dtw feeds the query feature and the FAC anchor below.
    fac_dtw, water_flatten = water_flatten_override(args, qxy, fac_dtw)
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
    if water_xy is not None:
        from pyproj import Transformer

        lon_q, lat_q = Transformer.from_crs(5070, 4326, always_xy=True).transform(
            qx, qy
        )
        frame["gsw_occ_pct"] = _sample_gsw_occurrence(lon_q, lat_q)
        for c, v in water_query_features(qxy, z_surf, water_xy, args.dem).items():
            frame[c] = v
    for c, v in v3_cols.items():
        frame[c] = v
    if bman.get("modis_wetness"):
        for c, v in sample_modis_jja_wetness(qx, qy).items():
            frame[c] = v
    if "drilled_depth_idw_m" in man["query_feature_cols"]:
        # deployment semantics: a lattice cell is not a well, so no drilled-depth
        # self/nest exclusion (build_conus_graph_inputs.sample_drilled_depth).
        for c, v in dd_dup_features(
            bman, Path(man["graph_dir"]), qxy, z_surf, boundaries=boundaries
        ).items():
            frame[c] = v
    missing = [c for c in man["query_feature_cols"] if c not in frame.columns]
    if missing:
        raise SystemExit(f"lattice frame lacks query features: {missing}")

    lat_all = build_lateral_edges(qxy, geom, None, int(bman["knn_lateral"]))
    lat_all = attach_lateral_attrs(
        lat_all, rn_full, z_surf, float(bman["conductance_p"])
    )
    # pre-prune copy: prune_for_queries REMAPS lat's reach_node_idx to the pruned
    # numbering, but the source-edge basin lookup needs rn_full's original ids.
    lat_full = lat_all
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

    src_ctx = None
    if f_srcedge is not None:
        # cell<-well kNN source edges over the WHOLE lattice (dest ids are global
        # cell ids; sliced per chunk below), deployment semantics: every eligible
        # well visible to every fold, no exclusion disk.
        src_ctx = deployment_source_ctx(
            man,
            bman,
            Path(man["graph_dir"]),
            qxy,
            z_surf,
            lat_full,
            rn_full,
            device,
            exclude_m=float(args.source_exclude_m),
            extra=extra_sources,
        )
        src_dst = src_ctx["edges"]["ei"][1]

    # With writeback the reach states depend on the ENTIRE query set (queries are
    # written back onto reaches before the channel stack), so a chunked forward is
    # NOT the same computation as the full-batch one: the basin goes through in a
    # single batch. Without writeback, reach states are query-independent and the
    # chunked path is exact.
    chunk = n if writeback else args.query_chunk
    if writeback and args.query_chunk < n:
        log.info(
            "%s: query-writeback arm -- forwarding all %d cells in ONE batch "
            "(--query-chunk %d ignored; chunking would change the computation)",
            basin,
            n,
            args.query_chunk,
        )
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    qidx = lat_all["query_node_idx"].to_numpy("int64")
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        sub = frame.iloc[i0:i1].reset_index(drop=True)
        lat_c = lat_all[(qidx >= i0) & (qidx < i1)].copy()
        lat_c["query_node_idx"] = lat_c["query_node_idx"] - i0
        graph = {**rt, **lateral_tensors(lat_c, models["shared"], dims, device)}
        if writeback:
            graph["lat_ei_reversed"] = graph["lat_ei"].flip(0)
        chunk_src = src_ctx
        if src_ctx is not None and (i0, i1) != (0, n):
            keep = (src_dst >= i0) & (src_dst < i1)
            ei = src_ctx["edges"]["ei"][:, keep].clone()
            ei[1] -= i0
            chunk_src = {
                **src_ctx,
                "edges": {
                    "src": src_ctx["edges"]["src"][keep],
                    "ei": ei,
                    "ea": src_ctx["edges"]["ea"][keep],
                },
            }
        anchors_c = {k: v[i0:i1] for k, v in anchors_all.items()}
        ch = run_folds(
            models,
            graph,
            sub,
            r_wte[i0:i1],
            anchors_c,
            device,
            f"{basin}[{i0}:{i1}]",
            f_mae=f_mae,
            writeback=writeback,
            src_ctx=chunk_src,
        )
        for key in ("wte", "head_wte", "fold_spread", "sigma"):
            if key in ch and key in agg:
                agg[key][i0:i1] = ch[key]
        agg["w"][i0:i1] = ch["w"]
        if i1 < n:
            log.info("%s: %d/%d cells done", basin, i1, n)
    if not np.isfinite(agg["wte"]).all():
        raise SystemExit(f"{basin}: non-finite WTE cells after all chunks (bug)")
    peak_gb = float(torch.cuda.max_memory_allocated() / 2**30) if cuda else None
    if peak_gb is not None:
        log.info(
            "%s: peak GPU tensor allocation %.2f GiB (%d cells per forward)",
            basin,
            peak_gb,
            min(chunk, n),
        )

    dtw = z_surf - agg["wte"]
    neg = float((dtw < 0).mean())
    log.info(
        "%s: DTW median %.2f m, %.1f%% negative (predicted artesian/inundation)",
        basin,
        float(np.median(dtw)),
        100 * neg,
    )
    gate_panel = None
    if oof is not None:
        gate_panel = leak_gate(
            basin,
            scatter(dtw, rows, cols, height, width),
            transform,
            oof,
            args.leak_gate_frac,
            args.leak_gate_min_wells,
            args.leak_gate_track_m,
            informational=f_srcedge is not None,
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
                "r_source": args.r_source,
                "r_exclude_km": args.r_exclude_km,
                "deep_exclude_km": args.deep_exclude_km,
                "query_writeback": bool(writeback),
                "full_batch_forward": bool(chunk >= n),
                "query_chunk": int(chunk),
                "peak_gpu_alloc_gb": peak_gb,
                "source_edges": (src_ctx or {}).get("stats"),
                "extra_sources_path": args.extra_sources,
                "water_flatten": water_flatten,
                "leak_gate": gate_panel,
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
    ap.add_argument(
        "--source-exclude-m",
        type=float,
        default=0.0,
        help="(--source-edges arms) drop cell<-well source edges from wells "
        "within this radius of the cell; 0 = deployment semantics (a cell at a "
        "well reads that well), >0 = eval-fairness sensitivity variant",
    )
    ap.add_argument(
        "--extra-sources",
        help="(--source-edges arms) parquet of admitted external source wells "
        "(x5070/y5070/wte_residual_m/z_surf_m) joining the bundle wells in the "
        "kNN source pool; wte_residual_m must be in the bundle frame "
        "((z_surf - dtw) - r_wte). Controlling-reach basin is attached here.",
    )
    ap.add_argument("--save-lattice", action="store_true")
    ap.add_argument(
        "--overwrite", action="store_true", help="re-run basins with existing output"
    )
    ap.add_argument(
        "--r-source",
        choices=("crossfit", "obs-exclude"),
        default="crossfit",
        help="lattice R/deep prior: interpolate the archived leave-fold-out "
        "crossfit feature values (default; exact at wells, seam-free), or "
        "leave-radius-out IDW of raw obs (shelved lever; arc seams at the "
        "exclusion-disk edge)",
    )
    ap.add_argument(
        "--r-exclude-km",
        type=float,
        default=3.5,
        help="--r-source obs-exclude only: leave-radius-out disk for lattice "
        "R; calibrated to the archived crossfit residual distribution "
        "(LEAKAGE_AUDIT.md remediation)",
    )
    ap.add_argument(
        "--deep-exclude-km",
        type=float,
        default=4.5,
        help="--r-source obs-exclude only: leave-radius-out disk for the "
        "deep-pool IDW (sparser pool, larger calibrated radius)",
    )
    ap.add_argument(
        "--leak-gate-frac",
        type=float,
        default=0.5,
        help="fail a basin whose map MAD at training wells beats the archived "
        "OOF MAD by more than this fraction",
    )
    add_water_flatten_args(ap)
    ap.add_argument("--leak-gate-min-wells", type=int, default=20)
    ap.add_argument(
        "--leak-gate-track-m",
        type=float,
        default=2.0,
        help="second gate signal: fail only if median |map - oof| at training "
        "wells also exceeds this (m); a ratio trip that tracks OOF passes as "
        "pass_ratio_tripped (ensemble-variance false positive)",
    )
    args = ap.parse_args()
    args.model_dir = Path(args.model_dir)
    args.model_name = args.model_dir.name
    models = load_models(args.model_dir)
    man = models["manifest"]
    f_mae, writeback = ckpt_extensions(models["folds"][0])
    f_src = ckpt_f_src(models["folds"][0], man["feature_dims"]["query"])
    f_srcedge = ckpt_f_srcedge(models["folds"][0])
    log.info(
        "arm %s: f_mae=%s writeback=%s f_src=%s f_srcedge=%s",
        args.model_name,
        f_mae,
        writeback,
        f_src,
        f_srcedge,
    )
    if f_mae is not None:
        raise SystemExit(
            f"{args.model_name} has an MAE head (f_mae={f_mae}): the renderer has "
            "no lattice MAE embeddings (they exist only at the bundle's query "
            "nodes / extract_mae_embeddings.py --coords point sets), so this arm "
            "cannot render. Use a non-MAE arm, or predict_gnn_at_points.py "
            "--mae-embeddings for point output."
        )
    if f_src is not None:
        raise SystemExit(
            f"{args.model_name} is a --source-obs (rung 0) arm: the source wells "
            "must live inside the query set for their obs to travel, which a "
            "lattice render cannot do (rung 0 is shelved). Use a --source-edges arm."
        )

    if args.oof_check:
        oof_check(models, args.model_dir, args.device, args.oof_tol_m)
        return

    if not args.basins and not args.state:
        raise SystemExit("pass --basins or --state (or --oof-check)")
    basins = args.basins or nm_basins(args.state)

    gdir = Path(man["graph_dir"])
    bman = json.loads((gdir / "graph_manifest.json").read_text())
    qn = well_pool(pd.read_parquet(gdir / "query_nodes.parquet"))
    oof = well_pool(pd.read_parquet(args.model_dir / "gnn_oof_predictions.parquet"))
    water_xy = None
    if man["flags"].get("water_features"):
        blocks = pd.read_parquet(gdir / WATER_BLOCKS_PARQUET)
        water_xy = blocks[["x5070", "y5070"]].to_numpy("float64")
        log.info(
            "water features ON: %d permanent-water blocks (bundle cache)",
            len(water_xy),
        )
    wells = {
        "xy": qn[["x5070", "y5070"]].to_numpy("float64"),
        "z": qn[man["surface_elev_col"]].to_numpy("float64"),
        "wte": qn[man["obs_wte_col"]].to_numpy("float64"),
        "r_cross": qn["regional_wte_idw_oof_m"].to_numpy("float64"),
        "deep_cross": qn["deep_regional_wte_idw_oof_m"].to_numpy("float64"),
        "deep": deep_well_mask(
            qn,
            float(bman["deep_datum"]["quantile"]),
            str(bman["deep_datum"]["unit"]),
            int(bman["deep_datum"]["min_per_unit"]),
        ),
    }
    for c in ("r_cross", "deep_cross"):
        if not np.isfinite(wells[c]).all():
            raise SystemExit(
                f"bundle crossfit column behind wells[{c!r}] has non-finite "
                "values -- crossfit interpolation needs the full archived field"
            )
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
    boundaries = None
    if "drilled_depth_idw_m" in man["query_feature_cols"]:
        # CONUS-wide boundary set: built once, reused by every basin
        boundaries = build_boundaries(
            str(gdir / "reach_nodes.parquet"),
            bman["sources"]["geom"],
            top_orders=int(bman["dupuit_hang"]["top_orders"]),
        )
    extra_sources = None
    if args.extra_sources:
        if f_srcedge is None:
            raise SystemExit("--extra-sources requires a --source-edges arm")
        extra_sources = load_extra_sources(args.extra_sources, geom, rn_full)

    done = skipped = 0
    for b in basins:
        out_dir = HUC8_ROOT / b / "gnn" / args.model_name
        if not args.overwrite and (out_dir / "gnn_dtw_100m.tif").exists():
            skipped += 1
            continue
        infer_basin(
            b,
            models,
            wells,
            rn_full,
            ce_full,
            geom,
            bman,
            args,
            args.device,
            water_xy=water_xy,
            oof=oof,
            extra_sources=extra_sources,
            boundaries=boundaries,
        )
        done += 1
    log.info("inference complete: %d basins run, %d skipped (existing)", done, skipped)


if __name__ == "__main__":
    main()
