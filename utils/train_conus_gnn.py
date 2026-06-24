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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conus_graph_inputs import (  # noqa: E402
    DEEP_REGIONAL_WTE_COL,
    FAC_REM_WTE_COL,
    HAND_WTE_COL,
    REGIONAL_WTE_COL,
    TARGET_DTW_RESIDUAL,
    TARGET_WTE,
)
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


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Quantile (pinball) loss; tau>0.5 penalizes UNDER-prediction harder.

    The documented deep failure is predicting deep wells too shallow (residual too
    low). With tau~0.85 on the residual target, under-prediction (e>0) is penalized
    at tau and over-prediction at 1-tau, biasing the auxiliary head deeper.
    """
    e = target - pred
    return torch.mean(torch.maximum(tau * e, (tau - 1.0) * e))


def _native_to_dtw(native: np.ndarray, base: np.ndarray, mode: str) -> np.ndarray:
    """Reconstruct DTW from the model's native prediction.

    dtw_residual: dtw = regional_prior + residual_hat (base = regional prior).
    wte:          dtw = z_surf_well - wte_hat        (base = land-surface elev).
    """
    return base - native if mode == TARGET_WTE else base + native


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


def train_fold(
    model, feat, y_std, tr, va, base, obs_dtw, y_c, y_s, mode, eff_tau, args, device
):
    """Train one fold; early-stop on val DTW-MAD; return native_hat over all queries."""
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    huber = nn.HuberLoss(delta=_huber_delta_std(args, mode, y_s))
    tr_t = torch.as_tensor(tr, device=device)
    best_mad, best_state, best_epoch, since = np.inf, None, -1, 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        out = model(feat)
        if args.pinball:
            primary, pin = out
            loss = huber(
                primary[tr_t], y_std[tr_t]
            ) + args.pinball_weight * pinball_loss(pin[tr_t], y_std[tr_t], eff_tau)
        else:
            loss = huber(out[tr_t], y_std[tr_t])
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
    # WTE: too-shallow means wte_hat too HIGH, so the deep quantile is the LOW tail.
    # tau>0.5 on the (residual/DTW) head pushes deeper; flip for the WTE head.
    if target_mode == TARGET_WTE and args.pinball and args.pinball_tau > 0.5:
        log.warning(
            "WTE pinball tau>0.5 pushes head UP (shallower); using 1-tau=%.2f for "
            "the deep head",
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

    # --- prune reach graph to the n-hop neighborhood of attached reaches -------
    attached = np.unique(le["reach_node_idx"].to_numpy("int64"))
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
    anchor_head_m = None  # WTE-mode Dirichlet BC values (fold-standardized per fold)
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

    obs_dtw = qn[man["obs_dtw_col"]].to_numpy("float64")
    target = qn[target_col].to_numpy("float64")
    folds = np.array(sorted(qn[fold_col].unique()))
    blocks = qn[group_col].to_numpy()
    # `base` is the DTW-reconstruction term: regional prior (residual) or land-
    # surface elevation (WTE). Only the active mode's columns are required finite.
    if target_mode == TARGET_WTE:
        base = qn[surface_col].to_numpy("float64")
        obs_wte = qn[obs_wte_col].to_numpy("float64")
        required = {
            "surface_elev": base,
            "obs_dtw": obs_dtw,
            "target": target,
            "obs_wte": obs_wte,
        }
    else:
        base = qn[man["regional_prior_col"]].to_numpy("float64")
        obs_wte = None
        required = {"regional": base, "obs_dtw": obs_dtw, "target": target}
    for nm, arr in required.items():
        if not np.isfinite(arr).all():
            raise SystemExit(
                f"{int((~np.isfinite(arr)).sum())} non-finite {nm} in query nodes"
            )

    f_ch, f_lat = ch_ea.shape[1], lat_ea.shape[1]
    native_oof = np.full(len(qn), np.nan)
    fold_log: list[dict] = []
    huber_delta_std_by_fold: list[float] = []

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
            pinball=args.pinball,
        ).to(device)

    if device.startswith("cuda"):
        probe_feat = {**graph_tensors, "query_x": probe_x}
        yc0 = float(np.median(target))
        ys0 = float(1.4826 * np.median(np.abs(target - yc0)) or 1.0)
        if anchor_head_m is not None:
            probe_feat["anchor_value"] = torch.as_tensor(
                (anchor_head_m - yc0) / ys0, dtype=torch.float32, device=device
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

    for f in folds:
        test = qn[fold_col].to_numpy() == f
        trainval = ~test
        va = val_blocks(trainval, blocks, args.val_frac, rng)
        tr = trainval & ~va
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
        # WTE Dirichlet BC value, standardized in THIS fold's target space (so the
        # injected head sits on the same scale as the standardized model output).
        if anchor_head_m is not None:
            feat["anchor_value"] = torch.as_tensor(
                (anchor_head_m - y_c) / y_s, dtype=torch.float32, device=device
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
            pinball=args.pinball,
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
    identity_max = None
    if target_mode == TARGET_WTE:
        # |WTE err| must equal |DTW err| since z_surf is exact and shared.
        identity_max = float(
            np.nanmax(np.abs(np.abs(native_oof - obs_wte) - np.abs(gnn_dtw - obs_dtw)))
        )
        out_cols |= {
            "z_surf_well_m": base,
            "obs_wte_m": obs_wte,
            "regional_wte_idw_oof_m": qn[REGIONAL_WTE_COL].to_numpy(),
            "deep_regional_wte_idw_oof_m": qn[DEEP_REGIONAL_WTE_COL].to_numpy(),
            "hand_wte_m": qn[HAND_WTE_COL].to_numpy(),
            "gnn_wte_hat_m": native_oof,
        }
        if FAC_REM_WTE_COL in qn.columns:
            out_cols[FAC_REM_WTE_COL] = qn[FAC_REM_WTE_COL].to_numpy()
        log.info(
            "WTE identity check: max |abs(WTE err) - abs(DTW err)| = %.3e m",
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
        "target_mode": target_mode,
        "native_prediction_col": man.get(
            "native_prediction_col",
            "gnn_wte_hat_m" if target_mode == TARGET_WTE else "gnn_residual_hat_m",
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
        if target_mode == TARGET_WTE
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
