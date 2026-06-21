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


def train_fold(model, feat, y_std, tr, va, regional, obs_dtw, y_c, y_s, args, device):
    """Train one fold; early-stop on val DTW-MAD; return residual_hat over all queries."""
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.HuberLoss(delta=args.huber_delta)
    tr_t = torch.as_tensor(tr, device=device)
    best_mad, best_state, best_epoch, since = np.inf, None, -1, 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        pred = model(feat)
        loss = loss_fn(pred[tr_t], y_std[tr_t])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            resid = model(feat).cpu().numpy() * y_s + y_c
        pred_dtw = regional + resid
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
        resid = model(feat).cpu().numpy() * y_s + y_c
    return resid, best_mad, best_epoch


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
            loss = loss_fn(m(feat)[tr_t], y_std[tr_t])
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
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
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

    regional = qn[man["regional_prior_col"]].to_numpy("float64")
    obs_dtw = qn[man["obs_dtw_col"]].to_numpy("float64")
    target = qn[target_col].to_numpy("float64")
    folds = np.array(sorted(qn[fold_col].unique()))
    blocks = qn[group_col].to_numpy()
    for nm, arr in (("regional", regional), ("obs_dtw", obs_dtw), ("target", target)):
        if not np.isfinite(arr).all():
            raise SystemExit(
                f"{int((~np.isfinite(arr)).sum())} non-finite {nm} in query nodes"
            )

    f_ch, f_lat = ch_ea.shape[1], lat_ea.shape[1]
    resid_oof = np.full(len(qn), np.nan)
    fold_log: list[dict] = []

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
            reach_x.shape[1], f_query, f_ch, f_lat, h, args.channel_layers, args.dropout
        ).to(device)

    if device.startswith("cuda"):
        probe_feat = {
            "reach_x": reach_x,
            "query_x": probe_x,
            "ch_ei": ch_ei,
            "ch_ea": ch_ea,
            "lat_ei": lat_ei,
            "lat_ea": lat_ea,
        }
        yc0 = float(np.median(target))
        ys0 = float(1.4826 * np.median(np.abs(target - yc0)) or 1.0)
        y_probe = torch.as_tensor(
            (target - yc0) / ys0, dtype=torch.float32, device=device
        )
        ladder = [
            h
            for h in sorted({args.hidden, 32, 24, 16}, reverse=True)
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
        feat = {
            "reach_x": reach_x,
            "query_x": query_x,
            "ch_ei": ch_ei,
            "ch_ea": ch_ea,
            "lat_ei": lat_ei,
            "lat_ea": lat_ea,
        }
        y_c = float(np.median(target[tr]))
        y_s = float(1.4826 * np.median(np.abs(target[tr] - y_c)) or 1.0)
        y_std = torch.as_tensor(
            (target - y_c) / y_s, dtype=torch.float32, device=device
        )

        torch.manual_seed(args.seed + int(f))
        model = WTEGraphNet(
            reach_x.shape[1],
            f_query,
            f_ch,
            f_lat,
            hidden,
            args.channel_layers,
            args.dropout,
        ).to(device)
        resid, best_mad, best_epoch = train_fold(
            model, feat, y_std, tr, va, regional, obs_dtw, y_c, y_s, args, device
        )
        del model, query_x, y_std, feat
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        resid_oof[test] = resid[test]
        pred_dtw = regional + resid
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

    if not np.isfinite(resid_oof).all():
        raise SystemExit(
            f"{int((~np.isfinite(resid_oof)).sum())} queries got no OOF prediction"
        )

    gnn_dtw = regional + resid_oof
    # Carry both bases by their TRUE names: `regional` here is whatever
    # regional_prior_col selected (the deep datum under Mode B), so reading the
    # named columns straight from qn keeps the scorer's `regional` predictor the
    # all-well floor and exposes the deep datum as its own predictor.
    out = pd.DataFrame(
        {
            "canonical_id": qn["canonical_id"].to_numpy(),
            "source": qn["source"].to_numpy(),
            "is_nwis": qn["is_nwis"].to_numpy(),
            "x5070": qn["x5070"].to_numpy(),
            "y5070": qn["y5070"].to_numpy(),
            "huc2": qn["huc2"].to_numpy(),
            "cv_fold": qn[fold_col].to_numpy(),
            "obs_dtw_m": obs_dtw,
            "regional_base_m": regional,
            "regional_idw_dtw_oof_m": qn["regional_idw_dtw_oof_m"].to_numpy(),
            "regional_deep_idw_dtw_oof_m": qn["regional_deep_idw_dtw_oof_m"].to_numpy(),
            "janssen_dtw_m": qn["janssen_dtw"].to_numpy(),
            "hand_m": qn["hand_m"].to_numpy(),
            "gnn_residual_hat_m": resid_oof,
            "gnn_dtw_m": gnn_dtw,
        }
    )
    out.to_parquet(out_dir / "gnn_oof_predictions.parquet")

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
        },
        "counts": {
            "reach_nodes_pruned": len(rn),
            "channel_edges_pruned": len(ce),
            "query_nodes": len(qn),
            "lateral_edges": len(le),
        },
        "folds": fold_log,
        "target_col": target_col,
        "final_dtw_definition": man["final_dtw_definition"],
        "leakage_notes": man.get("leakage_notes", []),
    }
    (out_dir / "gnn_run.json").write_text(json.dumps(run, indent=2, default=str))
    log.info("wrote gnn_oof_predictions.parquet + gnn_run.json -> %s", out_dir)


if __name__ == "__main__":
    main()
