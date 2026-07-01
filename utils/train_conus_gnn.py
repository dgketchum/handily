"""Step 4 of the CONUS WTE/DTW GNN: edge-gated GNN over the national flow graph.

Reuses the RGA model (``EdgeGatedConv`` / ``WTEGraphNet``) and leak-free inductive
protocol from ``train_wte_gnn.py``, scaled to CONUS by *pruning* rather than
neighbor-sampling: a query well is only influenced by reaches within
``channel_layers`` channel hops of its lateral-attached reach, so the reach graph
is pruned once to that n-hop neighborhood and then full-batched. This is exact
(no sampling variance), unlike NeighborLoader, and keeps the constant reach graph
on GPU across folds (reaches carry no labels).

Target = residual over the cross-fit regional IDW prior; final prediction is
``dtw = regional_idw_dtw_oof + residual_hat``. OOF predictions per HUC4-blocked
fold are directly comparable to the regional baseline (and to Ma/Janssen in the
scorer).

    uv run python utils/train_conus_gnn.py \\
        --graph-dir /data/ssd2/handily/conus/wte_gnn/graph \\
        --out-dir   /data/ssd2/handily/conus/wte_gnn/gnn
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
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conus_graph_inputs import (  # noqa: E402
    DEEP_REGIONAL_WTE_COL,
    FAC_REM_WTE_ANOM_COL,
    FAC_REM_WTE_COL,
    HAND_WTE_COL,
    REGIONAL_WTE_COL,
    TARGET_DTW_RESIDUAL,
    TARGET_WTE,
    TARGET_WTE_RESIDUAL,
)

# Head-space target modes (WTE elevation OR residual-over-R): both reconstruct DTW
# as `base - native` and carry obs_wte/z_surf; dtw_residual reconstructs `base + native`.
HEAD_SPACE_MODES = (TARGET_WTE, TARGET_WTE_RESIDUAL)
from train_wte_gnn import (  # noqa: E402
    WTEGraphNet,
    apply_stats,
    fit_stats,
    val_blocks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_conus_gnn")


def prune_reach_graph(
    n_reach: int, channel_edges: pd.DataFrame, attached: np.ndarray, hops: int
) -> tuple[np.ndarray, np.ndarray]:
    """n-hop channel neighborhood of the lateral-attached reaches.

    Reaches outside this neighborhood cannot reach any query within ``hops``
    message-passing layers, so dropping them is exact. Returns the kept reach
    indices (old ids) and a boolean keep-mask over channel edges (both endpoints
    kept).
    """
    down = channel_edges[channel_edges["direction"] == 1]
    s = down["src_reach_idx"].to_numpy("int64")
    d = down["dst_reach_idx"].to_numpy("int64")
    a = csr_matrix((np.ones(len(s)), (s, d)), shape=(n_reach, n_reach))
    a = a + a.T  # undirected channel adjacency
    keep = np.zeros(n_reach, bool)
    keep[attached] = True
    frontier = keep.copy()
    for _ in range(hops):
        nxt = (a @ frontier.astype("float64")) > 0
        frontier = nxt & ~keep
        keep |= nxt
        if not frontier.any():
            break
    kept = np.where(keep)[0]
    src_all = channel_edges["src_reach_idx"].to_numpy("int64")
    dst_all = channel_edges["dst_reach_idx"].to_numpy("int64")
    edge_keep = keep[src_all] & keep[dst_all]
    return kept, edge_keep


def prune_anchor_reach_edges(ar: pd.DataFrame, old2new: np.ndarray) -> pd.DataFrame:
    """Keep anchor->reach edges whose reach survived the query n-hop prune, remapped.

    The retained reach set is every reach within ``channel_layers`` hops of a query-
    attached reach -- which already contains every reach on every <=L-hop path from
    an anchor to a query. An anchor edge to a reach OUTSIDE that set is silently
    neutered (its BC can never reach a well within the message-passing depth), so it
    is dropped here; survivors are remapped to the pruned reach index space.
    """
    new_reach = old2new[ar["reach_node_idx"].to_numpy("int64")]
    keep = new_reach >= 0
    out = ar[keep].copy()
    out["reach_node_idx"] = new_reach[keep]
    assert (out["reach_node_idx"] >= 0).all(), "anchor edge points to a pruned reach"
    return out


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantile (pinball) loss; tau>0.5 penalizes UNDER-prediction harder.

    The documented deep failure is predicting deep wells too shallow (residual too
    low). With tau~0.85 on the residual target, under-prediction (e>0) is penalized
    at tau and over-prediction at 1-tau, biasing the auxiliary head deeper. ``weight``
    (per-sample) gives a weighted mean (depth-aware loss); None = plain mean.
    """
    e = target - pred
    el = torch.maximum(tau * e, (tau - 1.0) * e)
    if weight is None:
        return torch.mean(el)
    return (el * weight).sum() / weight.sum()


def _native_to_dtw(native: np.ndarray, base: np.ndarray, mode: str) -> np.ndarray:
    """Reconstruct DTW from the model's native prediction.

    dtw_residual: dtw = regional_prior + residual_hat (base = regional prior).
    wte:          dtw = z_surf_well - wte_hat         (base = land-surface elev).
    wte_residual: dtw = (z_surf - R) - resid_hat      (base = z_surf - R).
    """
    return base - native if mode in HEAD_SPACE_MODES else base + native


def _fac_feat(
    fac_raw,
    fac_present,
    y_c: float,
    y_s: float,
    device,
    fac_pred_dtw=None,
    pdtw_c: float = 0.0,
    pdtw_s: float = 1.0,
) -> dict:
    """FAC bypass tensors standardized into THIS fold's target space.

    The FAC anomaly is FAC's own estimate of the (head-space) target, so standardizing
    it by the fold's (y_c, y_s) puts it on the same scale as the model's standardized
    output -- the model then predicts ``fac_base + correction``. Absent FAC is set to 0
    (the presence flag carries that info, so the skip contributes nothing there).
    ``fac_pred_dtw`` (FAC's own predicted DTW = base - fac_raw) feeds the confidence
    gate, standardized by its own train-fold (pdtw_c, pdtw_s)."""
    base = np.where(fac_present, (fac_raw - y_c) / y_s, 0.0)
    out = {
        "fac_base": torch.as_tensor(base, dtype=torch.float32, device=device),
        "fac_present": torch.as_tensor(
            fac_present.astype("float32"), dtype=torch.float32, device=device
        ),
    }
    if fac_pred_dtw is not None:
        gsig = np.where(fac_present, (fac_pred_dtw - pdtw_c) / pdtw_s, 0.0)
        out["fac_pred_dtw"] = torch.as_tensor(gsig, dtype=torch.float32, device=device)
    return out


def _huber_delta_std(args, mode: str, y_s: float) -> float:
    """Huber kink in STANDARDIZED units. Residual mode keeps the legacy standardized
    delta; WTE mode maps a physical-meter delta into std space (delta_m / y_s) so the
    robust kink stays meter-scale, not the ~hundreds-of-meters y_s of absolute head
    (which would push every real residual into the quadratic region = plain MSE)."""
    if mode == TARGET_WTE:
        return args.huber_delta_m / y_s
    return args.huber_delta


def _combine_native(out, y_s: float, y_c: float, base: np.ndarray, mode: str, args):
    """De-standardize the head(s) to the native target; regime-gate to the pinball
    head when the primary head's RECONSTRUCTED DTW is deep (the gate is on DTW in
    both modes, never on raw WTE magnitude)."""
    if args.pinball:
        primary, pin = out
        p = primary.detach().cpu().numpy() * y_s + y_c
        q = pin.detach().cpu().numpy() * y_s + y_c
        deep = _native_to_dtw(p, base, mode) >= args.deep_regime_threshold_m
        return np.where(deep, q, p)
    return out.detach().cpu().numpy() * y_s + y_c


def build_train_pairs(
    x5070: np.ndarray,
    y5070: np.ndarray,
    radius_m: float = 1000.0,
    k: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """(n_pairs, 2) int64 query-node index pairs (i<j, deduped, no self-pairs).

    Each well is linked to its <=``k`` nearest neighbours within ``radius_m`` (cKDTree).
    The anti-compression pair loss uses these to penalize a flattened LOCAL WTE gradient
    between nearby wells -- the diagnosed amplitude-compression pathology (pred std 28 vs
    obs 37.6, coherent <500 m patch bias). ``seed`` is accepted for signature stability;
    neighbour order is deterministic so no randomness is drawn.
    """
    xy = np.c_[x5070, y5070]
    n = len(xy)
    tree = cKDTree(xy)
    # k+1 to allow for the self-match at distance 0; missing neighbours come back with
    # idx == n and dist == inf (distance_upper_bound), filtered below.
    dist, idx = tree.query(xy, k=k + 1, distance_upper_bound=radius_m)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    src = np.repeat(np.arange(n), idx.shape[1])
    dst = idx.ravel()
    dd = dist.ravel()
    valid = (dst < n) & np.isfinite(dd) & (dst != src)
    a = np.minimum(src[valid], dst[valid])
    b = np.maximum(src[valid], dst[valid])
    if a.size == 0:
        return np.empty((0, 2), dtype="int64")
    return np.unique(np.stack([a, b], axis=1), axis=0).astype("int64")


def train_fold(
    model,
    feat,
    y_std,
    tr,
    va,
    base,
    obs_dtw,
    y_c,
    y_s,
    mode,
    eff_tau,
    args,
    device,
    sample_w_t=None,
    pair_idx=None,
    pair_w=0.0,
):
    """Train one fold; early-stop on val DTW-MAD; return native_hat over all queries.

    ``sample_w_t`` (full-length, optional) re-weights the per-well training loss
    (depth-aware loss weighting) so the shallow band is not swamped by deep wells.
    """
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    weighted = sample_w_t is not None
    pair_delta = _huber_delta_std(args, mode, y_s)
    huber = nn.HuberLoss(
        delta=pair_delta,
        reduction="none" if weighted else "mean",
    )
    tr_t = torch.as_tensor(tr, device=device)
    w_tr = sample_w_t[tr_t] if weighted else None
    use_pairs = pair_w > 0.0 and pair_idx is not None and pair_idx.numel() > 0
    best_mad, best_state, best_epoch, since = np.inf, None, -1, 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        out = model(feat)
        if args.pinball:
            primary, pin = out
            hl = huber(primary[tr_t], y_std[tr_t])
            hl = (hl * w_tr).sum() / w_tr.sum() if weighted else hl
            loss = hl + args.pinball_weight * pinball_loss(
                pin[tr_t], y_std[tr_t], eff_tau, w_tr
            )
            pred_point = primary
        else:
            hl = huber(out[tr_t], y_std[tr_t])
            loss = (hl * w_tr).sum() / w_tr.sum() if weighted else hl
            pred_point = out
        if use_pairs:
            # Anti-compression pair term: match the LOCAL predicted WTE-residual
            # gradient between nearby train wells (dp) to the observed one (dy). The
            # regional base R cancels in the difference, so standardized-residual
            # space is the correct arena. delta is the POINT-loss Huber knee
            # (_huber_delta_std) -- NOT eff_tau, which is the pinball QUANTILE and
            # dimensionally wrong as a knee; the plan asks for no second tau knob.
            dp = pred_point[pair_idx[0]] - pred_point[pair_idx[1]]
            dy = y_std[pair_idx[0]] - y_std[pair_idx[1]]
            loss = loss + pair_w * nn.functional.huber_loss(dp, dy, delta=pair_delta)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            native = _combine_native(model(feat), y_s, y_c, base, mode, args)
        pred_dtw = _native_to_dtw(native, base, mode)
        val_mad = float(np.nanmedian(np.abs(pred_dtw[va] - obs_dtw[va])))
        if val_mad < best_mad - 1e-4:
            best_mad, best_epoch, since = val_mad, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch >= args.min_epochs:
            since += 1
            if since >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        native = _combine_native(model(feat), y_s, y_c, base, mode, args)
    if use_pairs:
        # Compression ratio at the best epoch: std(pred_dtw)/std(obs_dtw) over val.
        # <1 == the pathology (predictions flattened toward the mean); the pair term
        # is meant to push this toward 1.
        pred_dtw = _native_to_dtw(native, base, mode)
        den = float(np.nanstd(obs_dtw[va]))
        ratio = float(np.nanstd(pred_dtw[va])) / den if den > 0 else float("nan")
        log.info("  compression std(pred_dtw)/std(obs_dtw)=%.3f (val)", ratio)
    return native, best_mad, best_epoch


def _is_oom(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
    )


def pick_working_hidden(make_model, feat, y_std, tr_mask, candidates, device):
    """Largest hidden width (<= requested) that survives one full-batch fwd+bwd.

    Full-batch CONUS (~2.7M reaches + ~3.4M queries + ~15M edges) is memory-bound
    on the per-edge message tensors EdgeGatedConv materializes. Probe once on the
    heaviest case (loss over ALL wells) so an OOM fails in seconds, not 2h into the
    fold loop, and the chosen width is shared by every fold for a coherent result.
    """
    tr_t = torch.as_tensor(tr_mask, device=device)
    loss_fn = nn.HuberLoss()
    for h in candidates:
        try:
            m = make_model(h)
            opt = torch.optim.Adam(m.parameters())
            opt.zero_grad()
            out = m(feat)
            primary = out[0] if isinstance(out, tuple) else out
            loss = loss_fn(primary[tr_t], y_std[tr_t])
            loss.backward()
            opt.step()
            del m, opt, loss
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            log.info("hidden=%d fits full-batch", h)
            return h
        except Exception as e:  # noqa: BLE001 - only OOM is recoverable; re-raise else
            if not _is_oom(e):
                raise
            log.warning("hidden=%d OOM; trying smaller", h)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    raise SystemExit(
        "no candidate hidden width fits in GPU memory (restrict reaches or run on CPU)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", default="/data/ssd2/handily/conus/wte_gnn/graph")
    p.add_argument("--out-dir", default="/data/ssd2/handily/conus/wte_gnn/gnn")
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--channel-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--min-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="standardized Huber delta for --target dtw_residual (legacy)",
    )
    p.add_argument(
        "--huber-delta-m",
        type=float,
        default=1.0,
        help="PHYSICAL-meter Huber delta for --target wte; mapped to delta_m/y_s "
        "per fold so the robust kink is meter-scale (not the std of absolute head)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    # v2: anchor BC + pinball deep head ---------------------------------------
    p.add_argument(
        "--no-anchors",
        action="store_true",
        help="ignore anchor tensors even if the bundle has them (ablation)",
    )
    p.add_argument(
        "--pinball",
        action="store_true",
        help="add a tau-quantile auxiliary head; regime-gate deep wells to it",
    )
    p.add_argument("--pinball-tau", type=float, default=0.85)
    p.add_argument("--pinball-weight", type=float, default=0.3)
    p.add_argument(
        "--deep-regime-threshold-m",
        type=float,
        default=15.0,
        help="predicted DTW above which the pinball head supersedes Huber",
    )
    p.add_argument(
        "--fac-skip",
        action="store_true",
        help="anchor the head-space output on FAC's standardized target-estimate "
        "(raw-FAC bypass to the head) so message passing corrects, not erases, FAC's "
        "sharp shallow signal; head-space targets (wte / wte_residual) only",
    )
    p.add_argument(
        "--fac-gate",
        action="store_true",
        help="learned confidence gate on the FAC anchor (requires --fac-skip): the gate "
        "sees FAC's own predicted DTW and RELEASES the anchor in the deep-regional regime "
        "where FAC saturates, letting the regional prior + graph carry deep wells",
    )
    p.add_argument(
        "--shallow-weight",
        type=float,
        default=1.0,
        help="depth-aware loss weight applied to training wells with obs_dtw < "
        "--shallow-thresh-m (1.0 = off); protects the shallow band from being swamped "
        "by the more numerous deep wells in the Huber average",
    )
    p.add_argument("--shallow-thresh-m", type=float, default=5.0)
    p.add_argument(
        "--directional-edges",
        action="store_true",
        help="flow-direction-conditioned message passing: channel edges route by their "
        "+1/-1 direction and lateral edges by sign(well_surf - reach_elev) through "
        "SEPARATE message+gate weights, so direction gets a dedicated parameter path "
        "instead of one concatenated feature the single transform must disentangle. "
        "Uses columns already in the bundle (no rebuild); the production arm is the "
        "no-flag default.",
    )
    p.add_argument(
        "--mainstem-read",
        action="store_true",
        help="(item 2) add the query->downstream-datum read conv: each well attends to "
        "its basin's discharge-datum reach's LEARNED state in ONE hop, bypassing the "
        "med-7/p90-28-hop receptive-field gap. Requires a bundle built with "
        "build_conus_graph_inputs.py --mainstem-read (mainstem_edges.parquet). The datum "
        "reaches (+ their 2-hop channel context) are unioned into the prune set so they "
        "actually enter the GPU graph. See notes/GNN_TOPOLOGY_PLAN.md item 2.",
    )
    # --- anti-compression pair loss (item 4): regularize the LOCAL WTE gradient -----
    p.add_argument(
        "--pair-loss-weight",
        type=float,
        default=0.0,
        help="(item 4) weight on the anti-compression pair term (0 = off, the "
        "production default). Nearby train wells (<= --pair-k neighbours within "
        "--pair-radius-m) get their predicted residual DIFFERENCE matched to the "
        "observed one via a Huber on (dp - dy). Targets the amplitude-compression "
        "pathology (std(pred) << std(obs), coherent <500 m patch bias) that the "
        "point loss + median early-stop does not penalize. Only train-train pairs "
        "are used (leak-free). See notes/GNN_TOPOLOGY_PLAN.md item 4.",
    )
    p.add_argument("--pair-radius-m", type=float, default=1000.0)
    p.add_argument("--pair-k", type=int, default=3)
    # --- regional-aquifer substrate (Phase 1): optional gated correction branch -----
    p.add_argument(
        "--aquifer",
        action="store_true",
        help="enable the aquifer graph tensors if present in graph_manifest.json; "
        "adds a GATED residual-correction branch over the stream/FAC-residual head "
        "(exact no-op under --aquifer-route fixed_stream)",
    )
    p.add_argument("--aquifer-layers", type=int, default=4)
    p.add_argument(
        "--aquifer-route",
        choices=["fixed_stream", "learned"],
        default="fixed_stream",
        help="fixed_stream: branch short-circuited (no-op identity vs baseline); "
        "learned: gated aquifer delta is added to the stream primary",
    )
    p.add_argument("--aquifer-gate-init", type=float, default=-6.0)
    p.add_argument(
        "--aquifer-delta-init-zero",
        action="store_true",
        default=True,
        help="zero the aquifer delta head at init (branch starts as an exact no-op)",
    )
    p.add_argument(
        "--no-aquifer-delta-init-zero",
        dest="aquifer_delta_init_zero",
        action="store_false",
    )
    args = p.parse_args()

    gdir = Path(args.graph_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = (
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("device: %s (torch %s)", device, torch.__version__)

    man = json.loads((gdir / "graph_manifest.json").read_text())
    reach_cols = man["reach_feature_cols"]
    query_cols = man["query_feature_cols"]
    ch_cols = man["channel_edge_feature_cols"]
    lat_cols = man["lateral_edge_feature_cols"]
    target_col = man["target_col"]
    fold_col = man["cv_fold_col"]
    group_col = man["cv_group_col"]
    # Backwards-compatible: bundles without target_mode are the v1/v2 residual path.
    target_mode = man.get("target_mode", TARGET_DTW_RESIDUAL)
    surface_col = man.get("surface_elev_col")
    obs_wte_col = man.get("obs_wte_col")
    log.info("target_mode=%s", target_mode)
    # Head-space modes: too-shallow means the predicted head is too HIGH, so the deep
    # quantile is the LOW tail. tau>0.5 on the DTW/dtw-residual head pushes deeper;
    # flip for any head-space (wte / wte_residual) head.
    if target_mode in HEAD_SPACE_MODES and args.pinball and args.pinball_tau > 0.5:
        log.warning(
            "head-space pinball tau>0.5 pushes head UP (shallower); using 1-tau=%.2f "
            "for the deep head",
            1.0 - args.pinball_tau,
        )
        eff_tau = 1.0 - args.pinball_tau
    else:
        eff_tau = args.pinball_tau

    rn = (
        pd.read_parquet(gdir / "reach_nodes.parquet")
        .sort_values("reach_node_idx")
        .reset_index(drop=True)
    )
    ce = pd.read_parquet(gdir / "channel_edges.parquet")
    qn = (
        pd.read_parquet(gdir / "query_nodes.parquet")
        .sort_values("query_node_idx")
        .reset_index(drop=True)
    )
    le = pd.read_parquet(gdir / "lateral_edges.parquet")
    assert (rn["reach_node_idx"].to_numpy() == np.arange(len(rn))).all()
    assert (qn["query_node_idx"].to_numpy() == np.arange(len(qn))).all()

    # --- mainstem-read edges (item 2): one query->datum read edge per covered well ----
    use_ms = args.mainstem_read
    me = None
    ms_cols = None
    if use_ms:
        ms_path = gdir / "mainstem_edges.parquet"
        if not ms_path.exists() or not man.get("mainstem_read"):
            raise SystemExit(
                "--mainstem-read set but the bundle has no mainstem_edges.parquet / "
                "mainstem_read manifest block -- rebuild the graph with "
                "build_conus_graph_inputs.py --mainstem-read"
            )
        me = pd.read_parquet(ms_path)
        ms_cols = man["mainstem_read"]["ms_edge_feature_cols"]

    # --- prune reach graph to the n-hop neighborhood of attached reaches -------
    # The mainstem datum reaches (+ their own 2-hop channel context) are unioned in so
    # the read edge actually reaches a node in the pruned GPU graph -- this is what puts
    # the mainstem back into the receptive field.
    attached = np.unique(le["reach_node_idx"].to_numpy("int64"))
    if use_ms:
        attached = np.unique(
            np.concatenate([attached, me["reach_node_idx"].to_numpy("int64")])
        )
    kept, edge_keep = prune_reach_graph(len(rn), ce, attached, args.channel_layers)
    old2new = np.full(len(rn), -1, dtype="int64")
    old2new[kept] = np.arange(len(kept), dtype="int64")
    rn = rn.iloc[kept].reset_index(drop=True)
    ce = ce[edge_keep].copy()
    ce["src_reach_idx"] = old2new[ce["src_reach_idx"].to_numpy("int64")]
    ce["dst_reach_idx"] = old2new[ce["dst_reach_idx"].to_numpy("int64")]
    le = le.copy()
    le["reach_node_idx"] = old2new[le["reach_node_idx"].to_numpy("int64")]
    assert (le["reach_node_idx"] >= 0).all(), "attached reach pruned away (bug)"
    if use_ms:
        me = me.copy()
        me["reach_node_idx"] = old2new[me["reach_node_idx"].to_numpy("int64")]
        assert (me["reach_node_idx"] >= 0).all(), "mainstem datum reach pruned (bug)"
        log.info(
            "mainstem-read: %d datum read edges (unioned into the prune set)", len(me)
        )
    log.info(
        "pruned reaches %d -> %d (%.1f%%); channel edges %d -> %d; lateral %d",
        len(old2new),
        len(rn),
        100 * len(rn) / len(old2new),
        len(edge_keep),
        len(ce),
        len(le),
    )

    # --- anchor BC: nodes + anchor->reach (pruned/remapped) + anchor->query ----
    anchor_block = man.get("anchors")
    use_anchors = bool(anchor_block) and not args.no_anchors
    an = ar = aq = None
    anchor_cols = ar_cols = aq_cols = None
    anchor_head_m = None  # TARGET_WTE absolute-head BC (fold-standardized per fold)
    anchor_anom = None  # TARGET_WTE_RESIDUAL per-fold head-anomaly BC {fold: array}
    if use_anchors:
        anchor_cols = anchor_block["anchor_feature_cols"]
        ar_cols = anchor_block["anchor_reach_edge_feature_cols"]
        aq_cols = anchor_block["anchor_query_edge_feature_cols"]
        an = (
            pd.read_parquet(gdir / "anchor_nodes.parquet")
            .sort_values("anchor_node_idx")
            .reset_index(drop=True)
        )
        assert (an["anchor_node_idx"].to_numpy() == np.arange(len(an))).all()
        if target_mode == TARGET_WTE:
            bc_col = anchor_block.get("anchor_bc_col", "head_m")
            anchor_head_m = an[bc_col].to_numpy("float64")
            if not np.isfinite(anchor_head_m).all():
                raise SystemExit(
                    f"{int((~np.isfinite(anchor_head_m)).sum())} non-finite anchor "
                    f"{bc_col} (WTE Dirichlet BC value)"
                )
        elif target_mode == TARGET_WTE_RESIDUAL:
            # Residual-space BC: one head-anomaly column (anchor_head - R_f) per fold.
            anom_cols = anchor_block.get("anchor_bc_anom_cols")
            if anom_cols:
                anchor_anom = {
                    int(c.rsplit("_", 1)[1]): an[c].to_numpy("float64")
                    for c in anom_cols
                }
                for fk, v in anchor_anom.items():
                    if not np.isfinite(v).all():
                        raise SystemExit(
                            f"{int((~np.isfinite(v)).sum())} non-finite anchor BC "
                            f"anomaly (fold {fk})"
                        )
        ar = pd.read_parquet(gdir / "anchor_to_reach_edges.parquet")
        aq = pd.read_parquet(gdir / "anchor_to_query_edges.parquet")
        ar = prune_anchor_reach_edges(ar, old2new)
        log.info(
            "anchors: %d nodes, %d->reach edges (kept after prune), %d->query edges",
            len(an),
            len(ar),
            len(aq),
        )
    elif anchor_block and args.no_anchors:
        log.info("bundle has anchors but --no-anchors set: running v1 (ablation)")
    # Dirichlet BC value present on a dedicated channel (absolute head for TARGET_WTE,
    # per-fold residual anomaly for TARGET_WTE_RESIDUAL); standardized per fold below.
    has_anchor_bc = anchor_head_m is not None or anchor_anom is not None

    # --- constant graph tensors -----------------------------------------------
    reach_stats = fit_stats(rn, reach_cols, None)
    ch_stats = fit_stats(ce, ch_cols, None)
    lat_stats = fit_stats(le, lat_cols, None)
    reach_x = torch.as_tensor(
        apply_stats(rn, reach_stats), dtype=torch.float32, device=device
    )
    ch_ea = torch.as_tensor(
        apply_stats(ce, ch_stats), dtype=torch.float32, device=device
    )
    lat_ea = torch.as_tensor(
        apply_stats(le, lat_stats), dtype=torch.float32, device=device
    )
    ch_ei = torch.as_tensor(
        ce[["src_reach_idx", "dst_reach_idx"]].to_numpy().T,
        dtype=torch.long,
        device=device,
    )
    lat_ei = torch.as_tensor(
        le[["reach_node_idx", "query_node_idx"]].to_numpy().T,
        dtype=torch.long,
        device=device,
    )

    # constant graph tensors shared by the probe + every fold; query_x is added per
    # fold (its scaler is train-only).
    graph_tensors = {
        "reach_x": reach_x,
        "ch_ei": ch_ei,
        "ch_ea": ch_ea,
        "lat_ei": lat_ei,
        "lat_ea": lat_ea,
    }
    # mainstem-read edges: standardized exactly like lat_ea (train-blind edge attrs,
    # median-impute + missingness flag), reach->query direction (datum reach = src).
    ms_ea = None
    if use_ms:
        ms_stats = fit_stats(me, ms_cols, None)
        ms_ea = torch.as_tensor(
            apply_stats(me, ms_stats), dtype=torch.float32, device=device
        )
        ms_ei = torch.as_tensor(
            me[["reach_node_idx", "query_node_idx"]].to_numpy().T,
            dtype=torch.long,
            device=device,
        )
        graph_tensors |= {"ms_ei": ms_ei, "ms_ea": ms_ea}
    # Flow-direction sign per edge (row-aligned with ch_ea/lat_ea), from columns
    # already in the bundle: channel `direction` (+1 down / -1 reverse, never NaN) and
    # lateral sign(well_surf - reach_elev) (well-above-reach +1 / below -1). A rare
    # NaN rel-elev (DEM gap; ~<1% per the build) routes to +1 (the dominant gaining-
    # stream/valley case), the same harmless default the median-imputed feature gets.
    if args.directional_edges:
        ch_direction = ce["direction"].to_numpy("float64")
        rel = le["rel_elev_query_reach_m"].to_numpy("float64")
        lat_sign = np.where(np.nan_to_num(rel, nan=0.0) >= 0, 1.0, -1.0)
        graph_tensors |= {
            "ch_dir": torch.as_tensor(
                np.where(ch_direction > 0, 1.0, -1.0),
                dtype=torch.float32,
                device=device,
            ),
            "lat_dir": torch.as_tensor(lat_sign, dtype=torch.float32, device=device),
        }
        log.info(
            "directional edges ON: channel +%d/-%d; lateral well-above %d / below %d "
            "(%d rel-elev NaN -> +1)",
            int((ch_direction > 0).sum()),
            int((ch_direction < 0).sum()),
            int((lat_sign > 0).sum()),
            int((lat_sign < 0).sum()),
            int((~np.isfinite(rel)).sum()),
        )
    f_anchor = f_ar = f_aq = None
    if use_anchors:
        anchor_x = torch.as_tensor(
            apply_stats(an, fit_stats(an, anchor_cols, None)),
            dtype=torch.float32,
            device=device,
        )
        ar_ea = torch.as_tensor(
            apply_stats(ar, fit_stats(ar, ar_cols, None)),
            dtype=torch.float32,
            device=device,
        )
        aq_ea = torch.as_tensor(
            apply_stats(aq, fit_stats(aq, aq_cols, None)),
            dtype=torch.float32,
            device=device,
        )
        graph_tensors |= {
            "anchor_x": anchor_x,
            "ar_ei": torch.as_tensor(
                ar[["anchor_node_idx", "reach_node_idx"]].to_numpy().T,
                dtype=torch.long,
                device=device,
            ),
            "ar_ea": ar_ea,
            "aq_ei": torch.as_tensor(
                aq[["anchor_node_idx", "query_node_idx"]].to_numpy().T,
                dtype=torch.long,
                device=device,
            ),
            "aq_ea": aq_ea,
        }
        f_anchor, f_ar, f_aq = anchor_x.shape[1], ar_ea.shape[1], aq_ea.shape[1]

    # --- regional-aquifer substrate: optional gated correction branch -------------
    # Loaded before the memory probe so the probe's full-batch forward includes the
    # aquifer tensors. Features are target-blind + constant across folds (no label to
    # hold out), so fit_stats on all rows is leak-safe.
    aquifer_block = man.get("aquifer")
    use_aquifer = bool(args.aquifer and aquifer_block and aquifer_block.get("enabled"))
    f_aquifer = f_aq_edge = f_aq_query = None
    if args.aquifer and not aquifer_block:
        raise SystemExit(
            "--aquifer set but graph_manifest.json has no aquifer block "
            "(run build_regional_aquifer_graph.py on this bundle first)"
        )
    if args.aquifer and aquifer_block and not aquifer_block.get("enabled"):
        # disabled-but-present must not silently fall through to a stream-only run when
        # the user explicitly asked for the aquifer branch.
        raise SystemExit(
            "--aquifer set but the manifest aquifer block is disabled "
            "(enabled=false); rebuild the bundle or drop --aquifer"
        )
    if use_aquifer:
        if args.aquifer_route == "learned" and args.aquifer_layers <= 0:
            raise SystemExit("--aquifer-route learned requires --aquifer-layers > 0")
        if use_anchors:
            raise SystemExit("Phase 1: --aquifer is not supported with anchors")
        if args.fac_skip:
            raise SystemExit(
                "Phase 1: --aquifer is not supported with --fac-skip (the leading "
                "recipe uses --residual-base fac_rem and passes no --fac-skip)"
            )
        aqf_cols = aquifer_block["aquifer_feature_cols"]
        aqe_cols = aquifer_block["aquifer_edge_feature_cols"]
        aqq_cols = aquifer_block["aquifer_query_edge_feature_cols"]
        aqn = (
            pd.read_parquet(gdir / aquifer_block["node_file"])
            .sort_values("aquifer_node_idx")
            .reset_index(drop=True)
        )
        aqe = pd.read_parquet(gdir / aquifer_block["edge_file"])
        aqq = pd.read_parquet(gdir / aquifer_block["query_edge_file"])
        assert (aqn["aquifer_node_idx"].to_numpy() == np.arange(len(aqn))).all()
        assert aqq["query_node_idx"].max() < len(qn), "aquifer->query idx out of range"
        ei = aqe[["src_aquifer_idx", "dst_aquifer_idx"]].to_numpy("int64")
        assert ei.min() >= 0 and ei.max() < len(aqn), "aquifer edge idx out of range"
        aq_x = torch.as_tensor(
            apply_stats(aqn, fit_stats(aqn, aqf_cols, None)),
            dtype=torch.float32,
            device=device,
        )
        aq_node_ea_t = torch.as_tensor(
            apply_stats(aqe, fit_stats(aqe, aqe_cols, None)),
            dtype=torch.float32,
            device=device,
        )
        aq_query_ea_t = torch.as_tensor(
            apply_stats(aqq, fit_stats(aqq, aqq_cols, None)),
            dtype=torch.float32,
            device=device,
        )
        # Collision-free keys: anchor->query already owns aq_ei/aq_ea, so the aquifer
        # graph uses aq_node_* (aquifer<->aquifer) + aq_query_* (aquifer->query). The
        # aquifer node/query index spaces are independent of the reach prune (queries
        # are never pruned), so no remap is needed.
        graph_tensors |= {
            "aquifer_x": aq_x,
            "aq_node_ei": torch.as_tensor(ei.T, dtype=torch.long, device=device),
            "aq_node_ea": aq_node_ea_t,
            "aq_query_ei": torch.as_tensor(
                aqq[["aquifer_node_idx", "query_node_idx"]].to_numpy().T,
                dtype=torch.long,
                device=device,
            ),
            "aq_query_ea": aq_query_ea_t,
        }
        f_aquifer = aq_x.shape[1]
        f_aq_edge = aq_node_ea_t.shape[1]
        f_aq_query = aq_query_ea_t.shape[1]
        log.info(
            "aquifer ON: %d nodes(%df) %d edges(%df) %d query-edges(%df) | route=%s "
            "layers=%d gate_init=%.1f delta_init_zero=%s",
            len(aqn),
            f_aquifer,
            len(aqe),
            f_aq_edge,
            len(aqq),
            f_aq_query,
            args.aquifer_route,
            args.aquifer_layers,
            args.aquifer_gate_init,
            args.aquifer_delta_init_zero,
        )
    elif aquifer_block and not args.aquifer:
        log.info(
            "bundle has an aquifer block but --aquifer not set: stream-only baseline"
        )

    obs_dtw = qn[man["obs_dtw_col"]].to_numpy("float64")
    target = qn[target_col].to_numpy("float64")
    folds = np.array(sorted(qn[fold_col].unique()))
    blocks = qn[group_col].to_numpy()
    # `base` is the DTW-reconstruction term, named by the manifest: regional prior
    # (dtw_residual), land-surface elevation (wte), or z_surf - R (wte_residual).
    # Bundles predating dtw_base_col fall back to the per-mode default.
    base_col = man.get("dtw_base_col") or (
        surface_col if target_mode in HEAD_SPACE_MODES else man["regional_prior_col"]
    )
    base = qn[base_col].to_numpy("float64")
    required = {"base": base, "obs_dtw": obs_dtw, "target": target}
    if target_mode in HEAD_SPACE_MODES:
        obs_wte = qn[obs_wte_col].to_numpy("float64")
        z_surf = qn[surface_col].to_numpy("float64")
        required |= {"obs_wte": obs_wte, "z_surf": z_surf}
    else:
        obs_wte = z_surf = None
    for nm, arr in required.items():
        if not np.isfinite(arr).all():
            raise SystemExit(
                f"{int((~np.isfinite(arr)).sum())} non-finite {nm} in query nodes"
            )

    # --- FAC raw-skip: FAC's own (head-space) target-estimate as the output anchor ---
    fac_skip = args.fac_skip
    fac_gate = args.fac_gate
    fac_raw = fac_present = fac_pred_dtw = None
    if fac_gate and not fac_skip:
        raise SystemExit("--fac-gate requires --fac-skip")
    if fac_skip:
        if target_mode == TARGET_WTE_RESIDUAL:
            fac_base_col = FAC_REM_WTE_ANOM_COL  # (z_surf - fac_rem_dtw) - R
        elif target_mode == TARGET_WTE:
            fac_base_col = FAC_REM_WTE_COL  # fac_rem water-surface ELEVATION
        else:
            raise SystemExit(
                "--fac-skip requires a head-space target (wte / wte_residual); "
                f"got {target_mode}"
            )
        if fac_base_col not in qn.columns:
            raise SystemExit(f"--fac-skip: column {fac_base_col!r} not in query nodes")
        fac_raw = qn[fac_base_col].to_numpy("float64")
        fac_present = np.isfinite(fac_raw)
        log.info(
            "fac-skip ON: anchor col %s, %d/%d wells carry FAC (%.1f%%)%s",
            fac_base_col,
            int(fac_present.sum()),
            len(qn),
            100.0 * fac_present.mean(),
            " | gate ON" if fac_gate else "",
        )
        if fac_gate:
            # FAC's own predicted DTW; in head-space base - fac_raw == fac_rem_dtw.
            # The gate keys on this to release the anchor when FAC predicts deep.
            fac_pred_dtw = np.where(fac_present, base - fac_raw, np.nan)

    # --- depth-aware loss weighting: upweight the shallow band (FAC's gold) ---------
    sample_w_t = None
    if args.shallow_weight != 1.0:
        sample_w = np.where(
            obs_dtw < args.shallow_thresh_m, args.shallow_weight, 1.0
        ).astype("float32")
        sample_w_t = torch.as_tensor(sample_w, dtype=torch.float32, device=device)
        log.info(
            "depth-aware loss: %d/%d wells <%.0fm upweighted x%.2f",
            int((obs_dtw < args.shallow_thresh_m).sum()),
            len(qn),
            args.shallow_thresh_m,
            args.shallow_weight,
        )

    f_ch, f_lat = ch_ea.shape[1], lat_ea.shape[1]
    f_ms = ms_ea.shape[1] if use_ms else None  # width incl. missingness flags
    native_oof = np.full(len(qn), np.nan)
    gate_oof = np.full(len(qn), np.nan) if fac_gate else None
    learned_aquifer = use_aquifer and args.aquifer_route == "learned"
    aquifer_gate_oof = np.full(len(qn), np.nan) if learned_aquifer else None
    fold_log: list[dict] = []
    huber_delta_std_by_fold: list[float] = []

    # Shared aquifer-branch kwargs (off unless --aquifer + manifest block present).
    aquifer_kwargs = dict(
        f_aquifer=f_aquifer if use_aquifer else None,
        f_aquifer_edge=f_aq_edge if use_aquifer else None,
        f_aquifer_query=f_aq_query if use_aquifer else None,
        n_aquifer_layers=args.aquifer_layers if use_aquifer else 0,
        aquifer_route=args.aquifer_route if use_aquifer else "off",
        aquifer_gate_init=args.aquifer_gate_init,
        aquifer_delta_init_zero=args.aquifer_delta_init_zero,
    )

    # All-wells query features for the memory probe (shapes match any fold).
    probe_x = torch.as_tensor(
        apply_stats(qn, fit_stats(qn, query_cols, None)),
        dtype=torch.float32,
        device=device,
    )
    f_query = probe_x.shape[1]

    def make_model(h):
        torch.manual_seed(args.seed)
        return WTEGraphNet(
            reach_x.shape[1],
            f_query,
            f_ch,
            f_lat,
            h,
            args.channel_layers,
            args.dropout,
            f_anchor=f_anchor,
            f_anchor_reach=f_ar,
            f_anchor_query=f_aq,
            f_ms=f_ms,
            pinball=args.pinball,
            fac_skip=fac_skip,
            fac_gate=fac_gate,
            directional_edges=args.directional_edges,
            **aquifer_kwargs,
        ).to(device)

    if device.startswith("cuda"):
        probe_feat = {**graph_tensors, "query_x": probe_x}
        yc0 = float(np.median(target))
        ys0 = float(1.4826 * np.median(np.abs(target - yc0)) or 1.0)
        if fac_skip:
            if fac_gate:
                pc0 = float(np.nanmedian(fac_pred_dtw))
                ps0 = float(1.4826 * np.nanmedian(np.abs(fac_pred_dtw - pc0)) or 1.0)
                probe_feat |= _fac_feat(
                    fac_raw, fac_present, yc0, ys0, device, fac_pred_dtw, pc0, ps0
                )
            else:
                probe_feat |= _fac_feat(fac_raw, fac_present, yc0, ys0, device)
        if has_anchor_bc:
            bc0 = (
                anchor_head_m
                if anchor_head_m is not None
                else anchor_anom[int(folds[0])]
            )
            probe_feat["anchor_value"] = torch.as_tensor(
                (bc0 - yc0) / ys0, dtype=torch.float32, device=device
            )
        y_probe = torch.as_tensor(
            (target - yc0) / ys0, dtype=torch.float32, device=device
        )
        # lower ladder: anchor convs + re-assertion + wider head add per-edge memory.
        ladder = [
            h
            for h in sorted({args.hidden, 32, 24, 16, 12}, reverse=True)
            if h <= args.hidden
        ]
        hidden = pick_working_hidden(
            make_model, probe_feat, y_probe, np.ones(len(qn), bool), ladder, device
        )
        del probe_feat, y_probe
        torch.cuda.empty_cache()
    else:
        hidden = args.hidden
    del probe_x

    # Anti-compression pairs (item 4): built ONCE over all query nodes on the shared
    # EPSG:5070 grid, then per-fold restricted to train-train pairs so no held-out
    # label leaks through the difference term.
    pair_w = float(args.pair_loss_weight)
    all_pairs = None
    if pair_w > 0.0:
        all_pairs = build_train_pairs(
            qn["x5070"].to_numpy("float64"),
            qn["y5070"].to_numpy("float64"),
            radius_m=args.pair_radius_m,
            k=args.pair_k,
        )
        log.info(
            "anti-compression pair loss ON: lambda=%.2f radius=%.0fm k=%d -> "
            "%d global nearby-well pairs",
            pair_w,
            args.pair_radius_m,
            args.pair_k,
            len(all_pairs),
        )

    for f in folds:
        test = qn[fold_col].to_numpy() == f
        trainval = ~test
        va = val_blocks(trainval, blocks, args.val_frac, rng)
        tr = trainval & ~va
        pair_idx = None
        if all_pairs is not None and len(all_pairs):
            both_tr = tr[all_pairs[:, 0]] & tr[all_pairs[:, 1]]
            log.info("fold %d: %d train-train pairs", f, int(both_tr.sum()))
            if both_tr.any():
                pair_idx = torch.as_tensor(
                    all_pairs[both_tr].T, dtype=torch.long, device=device
                )
        q_stats = fit_stats(qn, query_cols, tr)
        query_x = torch.as_tensor(
            apply_stats(qn, q_stats), dtype=torch.float32, device=device
        )
        y_c = float(np.median(target[tr]))
        y_s = float(1.4826 * np.median(np.abs(target[tr] - y_c)) or 1.0)
        y_std = torch.as_tensor(
            (target - y_c) / y_s, dtype=torch.float32, device=device
        )
        feat = {**graph_tensors, "query_x": query_x}
        if fac_skip:
            if fac_gate:
                trp = tr & fac_present
                pc = float(np.median(fac_pred_dtw[trp]))
                ps = float(1.4826 * np.median(np.abs(fac_pred_dtw[trp] - pc)) or 1.0)
                feat |= _fac_feat(
                    fac_raw, fac_present, y_c, y_s, device, fac_pred_dtw, pc, ps
                )
            else:
                feat |= _fac_feat(fac_raw, fac_present, y_c, y_s, device)
        # WTE Dirichlet BC value, standardized in THIS fold's target space (so the
        # injected head sits on the same scale as the standardized model output).
        if has_anchor_bc:
            bc_raw = anchor_head_m if anchor_head_m is not None else anchor_anom[int(f)]
            feat["anchor_value"] = torch.as_tensor(
                (bc_raw - y_c) / y_s, dtype=torch.float32, device=device
            )
        huber_delta_std_by_fold.append(float(_huber_delta_std(args, target_mode, y_s)))

        torch.manual_seed(args.seed + int(f))
        model = WTEGraphNet(
            reach_x.shape[1],
            f_query,
            f_ch,
            f_lat,
            hidden,
            args.channel_layers,
            args.dropout,
            f_anchor=f_anchor,
            f_anchor_reach=f_ar,
            f_anchor_query=f_aq,
            f_ms=f_ms,
            pinball=args.pinball,
            fac_skip=fac_skip,
            fac_gate=fac_gate,
            directional_edges=args.directional_edges,
            **aquifer_kwargs,
        ).to(device)
        native, best_mad, best_epoch = train_fold(
            model,
            feat,
            y_std,
            tr,
            va,
            base,
            obs_dtw,
            y_c,
            y_s,
            target_mode,
            eff_tau,
            args,
            device,
            sample_w_t,
            pair_idx=pair_idx,
            pair_w=pair_w,
        )
        if fac_gate and model.last_fac_gate is not None:
            # last_fac_gate is from train_fold's final full-batch forward (all queries).
            gate_oof[test] = model.last_fac_gate.cpu().numpy().reshape(-1)[test]
        if learned_aquifer and model.last_aquifer_gate is not None:
            # per-query sigmoid gate from the final full-batch forward (all queries).
            aquifer_gate_oof[test] = (
                model.last_aquifer_gate.cpu().numpy().reshape(-1)[test]
            )
        del model, query_x, y_std, feat
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        native_oof[test] = native[test]
        pred_dtw = _native_to_dtw(native, base, target_mode)
        log.info(
            "fold %d: train=%d val=%d test=%d | val DTW-MAD=%.3f @%d | test=%.3f",
            f,
            tr.sum(),
            va.sum(),
            test.sum(),
            best_mad,
            best_epoch,
            float(np.nanmedian(np.abs(pred_dtw[test] - obs_dtw[test]))),
        )
        fold_log.append(
            {
                "fold": int(f),
                "n_train": int(tr.sum()),
                "n_val": int(va.sum()),
                "n_test": int(test.sum()),
                "best_val_dtw_mad_m": best_mad,
                "best_epoch": best_epoch,
            }
        )

    if not np.isfinite(native_oof).all():
        raise SystemExit(
            f"{int((~np.isfinite(native_oof)).sum())} queries got no OOF prediction"
        )

    if fac_gate:
        # Verify the mechanism: the gate should RELEASE (c->0) for deep wells. Log the
        # OOF gate value by observed-depth band (deep release = the intended behavior).
        bands = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
        msg = "; ".join(
            f"{lo}-{hi if hi != np.inf else '+'}m c={np.nanmean(gate_oof[m]):.2f}(n={int(m.sum())})"
            for lo, hi in bands
            if (m := (obs_dtw >= lo) & (obs_dtw < hi) & fac_present).any()
        )
        log.info("fac-gate mean OOF gate by obs-depth: %s", msg)

    if learned_aquifer:
        # The aquifer correction should OPEN (gate->1) where the regional substrate
        # carries signal; log the OOF gate by observed-depth band as the first read on
        # whether it earns its place (deep bands are where the stream head is weakest).
        bands = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
        msg = "; ".join(
            f"{lo}-{hi if hi != np.inf else '+'}m "
            f"g={np.nanmean(aquifer_gate_oof[m]):.3f}(n={int(m.sum())})"
            for lo, hi in bands
            if (m := (obs_dtw >= lo) & (obs_dtw < hi)).any()
        )
        log.info("aquifer-gate mean OOF gate by obs-depth: %s", msg)

    gnn_dtw = _native_to_dtw(native_oof, base, target_mode)
    # Common scoring columns (DTW + the named regional/deep DTW priors + benchmarks),
    # so the scorer's predictor set is identical across modes.
    out_cols = {
        "canonical_id": qn["canonical_id"].to_numpy(),
        "source": qn["source"].to_numpy(),
        "is_nwis": qn["is_nwis"].to_numpy(),
        "x5070": qn["x5070"].to_numpy(),
        "y5070": qn["y5070"].to_numpy(),
        "huc2": qn["huc2"].to_numpy(),
        "cv_fold": qn[fold_col].to_numpy(),
        "obs_dtw_m": obs_dtw,
        "regional_idw_dtw_oof_m": qn["regional_idw_dtw_oof_m"].to_numpy(),
        "regional_deep_idw_dtw_oof_m": qn["regional_deep_idw_dtw_oof_m"].to_numpy(),
        "janssen_dtw_m": qn["janssen_dtw"].to_numpy(),
        "hand_m": qn["hand_m"].to_numpy(),
        "gnn_dtw_m": gnn_dtw,
    }
    if fac_gate:
        out_cols["fac_gate_c"] = gate_oof  # learned anchor confidence (diagnostic)
    if learned_aquifer:
        out_cols["aquifer_gate"] = aquifer_gate_oof  # per-query aquifer-branch gate
    identity_max = None
    if target_mode in HEAD_SPACE_MODES:
        # wte_hat is the absolute head (wte) or R + residual_hat (wte_residual).
        # |WTE err| must equal |DTW err| since z_surf is exact and shared
        # (gnn_dtw = z_surf - wte_hat, obs_dtw = z_surf - obs_wte).
        r_wte = qn[REGIONAL_WTE_COL].to_numpy("float64")
        wte_hat = native_oof if target_mode == TARGET_WTE else r_wte + native_oof
        identity_max = float(
            np.nanmax(np.abs(np.abs(wte_hat - obs_wte) - np.abs(gnn_dtw - obs_dtw)))
        )
        out_cols |= {
            "z_surf_well_m": z_surf,
            "obs_wte_m": obs_wte,
            "regional_wte_idw_oof_m": r_wte,
            "deep_regional_wte_idw_oof_m": qn[DEEP_REGIONAL_WTE_COL].to_numpy(),
            "gnn_wte_hat_m": wte_hat,
        }
        if HAND_WTE_COL in qn.columns:
            out_cols[HAND_WTE_COL] = qn[HAND_WTE_COL].to_numpy()
        if FAC_REM_WTE_COL in qn.columns:
            out_cols[FAC_REM_WTE_COL] = qn[FAC_REM_WTE_COL].to_numpy()
        elif "fac_rem_dtw_m" in qn.columns:
            # wte_residual carries fac_rem_dtw_m (registry); the head-space diagnostic
            # wants fac_rem_wte = z_surf - fac_rem_dtw (NaN where FAC absent).
            out_cols[FAC_REM_WTE_COL] = z_surf - qn["fac_rem_dtw_m"].to_numpy("float64")
        log.info(
            "%s identity check: max |abs(WTE err) - abs(DTW err)| = %.3e m",
            target_mode,
            identity_max,
        )
    else:
        # `base` is whatever regional_prior_col selected (deep datum under Mode B).
        out_cols |= {
            "regional_base_m": base,
            "gnn_residual_hat_m": native_oof,
        }
    pd.DataFrame(out_cols).to_parquet(out_dir / "gnn_oof_predictions.parquet")

    run = {
        "graph_dir": str(gdir),
        "device": device,
        "torch": torch.__version__,
        "hyperparams": vars(args),
        "effective_hidden": int(hidden),
        "feature_dims": {
            "reach": int(reach_x.shape[1]),
            "query": int(f_query),
            "channel_edge": int(f_ch),
            "lateral_edge": int(f_lat),
            "anchor": int(f_anchor) if f_anchor else None,
            "anchor_reach_edge": int(f_ar) if f_ar else None,
            "anchor_query_edge": int(f_aq) if f_aq else None,
        },
        "counts": {
            "reach_nodes_pruned": len(rn),
            "channel_edges_pruned": len(ce),
            "query_nodes": len(qn),
            "lateral_edges": len(le),
            "anchor_nodes": int(len(an)) if use_anchors else 0,
            "anchor_reach_edges": int(len(ar)) if use_anchors else 0,
            "anchor_query_edges": int(len(aq)) if use_anchors else 0,
        },
        "anchors": {
            "used": bool(use_anchors),
            "available_in_bundle": bool(anchor_block),
            "by_class": anchor_block.get("by_class") if use_anchors else None,
            "knn_anchor_query": anchor_block.get("knn_anchor_query")
            if use_anchors
            else None,
        },
        "pinball": {
            "enabled": bool(args.pinball),
            "tau": args.pinball_tau if args.pinball else None,
            "effective_deep_tau": eff_tau if args.pinball else None,
            "weight": args.pinball_weight if args.pinball else None,
            "deep_regime_threshold_m": args.deep_regime_threshold_m
            if args.pinball
            else None,
        },
        "fac_skip": {
            "enabled": bool(fac_skip),
            "anchor_col": fac_base_col if fac_skip else None,
            "wells_with_fac": int(fac_present.sum()) if fac_skip else None,
            "confidence_gate": bool(fac_gate),
        },
        "aquifer": {
            "enabled": bool(use_aquifer),
            "available_in_bundle": bool(aquifer_block and aquifer_block.get("enabled")),
            "route": args.aquifer_route if use_aquifer else None,
            "n_layers": args.aquifer_layers if use_aquifer else None,
            "gate_init": args.aquifer_gate_init if learned_aquifer else None,
            "delta_init_zero": bool(args.aquifer_delta_init_zero)
            if learned_aquifer
            else None,
            "n_nodes": int(len(aqn)) if use_aquifer else None,
            "n_edges": int(len(aqe)) if use_aquifer else None,
            "n_query_edges": int(len(aqq)) if use_aquifer else None,
            "feature_dims": {
                "node": int(f_aquifer),
                "edge": int(f_aq_edge),
                "query_edge": int(f_aq_query),
            }
            if use_aquifer
            else None,
            "mean_oof_gate": float(np.nanmean(aquifer_gate_oof))
            if learned_aquifer
            else None,
        },
        "depth_aware_loss": {
            "shallow_weight": args.shallow_weight,
            "shallow_thresh_m": args.shallow_thresh_m
            if args.shallow_weight != 1.0
            else None,
        },
        "directional_edges": bool(args.directional_edges),
        "pair_loss": {
            "weight": pair_w,
            "radius_m": args.pair_radius_m if pair_w > 0.0 else None,
            "k": args.pair_k if pair_w > 0.0 else None,
            "n_global_pairs": int(len(all_pairs)) if all_pairs is not None else None,
        },
        "mainstem_read": {
            "enabled": bool(use_ms),
            "n_edges": int(len(me)) if use_ms else None,
            "ms_edge_feature_cols": ms_cols if use_ms else None,
            "f_ms": int(f_ms) if use_ms else None,
        },
        "target_mode": target_mode,
        "native_prediction_col": man.get(
            "native_prediction_col",
            "gnn_wte_hat_m"
            if target_mode in HEAD_SPACE_MODES
            else "gnn_residual_hat_m",
        ),
        "dtw_reconstruction": man.get("dtw_reconstruction"),
        "huber": {
            "mode": target_mode,
            "legacy_standardized_delta": args.huber_delta,
            "physical_delta_m": args.huber_delta_m
            if target_mode == TARGET_WTE
            else None,
            "effective_delta_std_by_fold": huber_delta_std_by_fold,
        },
        "wte_identity_check": {
            "max_abs_difference_between_wte_and_dtw_abs_errors_m": identity_max
        }
        if target_mode in HEAD_SPACE_MODES
        else None,
        "folds": fold_log,
        "target_col": target_col,
        "final_dtw_definition": man["final_dtw_definition"],
        "leakage_notes": man.get("leakage_notes", []),
    }
    (out_dir / "gnn_run.json").write_text(json.dumps(run, indent=2, default=str))
    log.info("wrote gnn_oof_predictions.parquet + gnn_run.json -> %s", out_dir)


if __name__ == "__main__":
    main()
