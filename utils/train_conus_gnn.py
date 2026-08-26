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
    DEEP_REGIONAL_WTE_ANOM_COL,
    DEEP_REGIONAL_WTE_COL,
    DUPUIT_FEATURE_COLS,
    FAC_REM_WTE_ANOM_COL,
    FAC_REM_WTE_COL,
    HAND_WTE_COL,
    REGIONAL_WTE_COL,
    TARGET_DTW_RESIDUAL,
    TARGET_WTE,
    TARGET_WTE_RESIDUAL,
)

# The hang expert's DTW column (--hang-anchor): the well-free Dupuit hang surface
# depth written by the builder's --dupuit-hang-features.
HANG_DTW_COL = DUPUIT_FEATURE_COLS[0]

# Head-space target modes (WTE elevation OR residual-over-R): both reconstruct DTW
# as `base - native` and carry obs_wte/z_surf; dtw_residual reconstructs `base + native`.
HEAD_SPACE_MODES = (TARGET_WTE, TARGET_WTE_RESIDUAL)
from build_analog_edges import analog_fold_keep  # noqa: E402
from build_source_edges import EDGE_COLS as SOURCE_EDGE_COLS  # noqa: E402
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
    prefix: str = "fac",
) -> dict:
    """Prior-anchor tensors standardized into THIS fold's target space.

    The anchor anomaly is that prior's own estimate of the (head-space) target, so
    standardizing it by the fold's (y_c, y_s) puts it on the same scale as the model's
    standardized output -- the model then predicts ``fac_base + correction``. Absent
    anchor is set to 0 (the presence flag carries that info, so the skip contributes
    nothing there). ``fac_pred_dtw`` (the prior's own predicted DTW = base - raw) feeds
    the confidence gate, standardized by its own train-fold (pdtw_c, pdtw_s).
    ``prefix`` renames the keys for the deep expert ("deep_base", ...)."""
    base = np.where(fac_present, (fac_raw - y_c) / y_s, 0.0)
    out = {
        f"{prefix}_base": torch.as_tensor(base, dtype=torch.float32, device=device),
        f"{prefix}_present": torch.as_tensor(
            fac_present.astype("float32"), dtype=torch.float32, device=device
        ),
    }
    if fac_pred_dtw is not None:
        gsig = np.where(fac_present, (fac_pred_dtw - pdtw_c) / pdtw_s, 0.0)
        out[f"{prefix}_pred_dtw"] = torch.as_tensor(
            gsig, dtype=torch.float32, device=device
        )
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
    ordinal_y_t=None,
    ts_prior_t=None,
    src_ctx=None,
):
    """Train one fold; early-stop on val DTW-MAD; return native_hat over all queries.

    ``sample_w_t`` (full-length, optional) re-weights the per-well training loss
    (depth-aware loss weighting) so the shallow band is not swamped by deep wells.
    ``ordinal_y_t`` (full-length (N, n_thresh) float, optional) adds the WP5 ordinal
    BCE term at ``args.ordinal_weight`` on the same train rows/weights.
    ``ts_prior_t`` (full-length float in (0,1), optional; --two-surface only) is the
    privileged P(phreatic) assignment prior for the WP2 membership BCE term.
    ``src_ctx`` (--source-obs / --source-edges) carries the fold's assimilation
    machinery: standardized obs values, the train / train+val eligible source pools,
    the mask fraction range, a fold-seeded rng for the per-epoch source redraw,
    ``has_x`` (whether the rung-0 src_x feature block is on), and -- rung 1 -- the
    fold-static source-edge tensors (``edge_src``/``edge_ei``/``edge_ea``).
    """
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # source-well assimilation (rungs 0/1), the masked-label protocol: each epoch a
    # fresh random subset of the eligible TRAIN pool becomes sources -- their
    # standardized obs becomes VISIBLE (rung 0: feat["src_x"]; rung 1: their edges
    # survive the filter); redrawing every epoch stops the model memorizing which
    # wells are always targets. Loss handling differs by rung: with src_x on, a
    # drawn source carries its OWN obs in its feature row, so its loss is a copy
    # task and is zeroed (and pairs through it dropped). Edges-only arms keep FULL
    # label mass -- self/same-site edges are dropped at build time, so a source
    # never sees its own label and its loss is legitimate supervision (zeroing it
    # anyway cost ~60% of train labels per epoch: rung 1's A2 cold regression).
    # The eval forwards are deterministic: val/early-stop shows ALL train sources;
    # the final forward (whose test rows feed the OOF) shows train+val sources --
    # test rows are never in a pool, so they stay unassisted. With val_mix on
    # (--source-val-mix) early stopping averages the val MAD of a NO-source
    # forward with the all-train-sources one, so model selection cannot favor
    # edge-reliant weights that lose unassisted (cold/far-from-well) skill --
    # rung 1's second protocol defect.
    src_on = src_ctx is not None
    if src_on:
        sv = src_ctx["val_std"]
        src_has_x = src_ctx.get("has_x", True)
        src_has_edges = "edge_ei" in src_ctx
        src_val_mix = src_ctx.get("val_mix", False)

        def _apply_sources(sel: np.ndarray) -> None:
            """Point feat's source tensors at a visible-source mask: the rung-0
            src_x block and/or the rung-1 edge subset (visible sources only)."""
            s = torch.as_tensor(sel.astype("float32"), device=device)
            if src_has_x:
                feat["src_x"] = torch.stack([sv * s, s], dim=-1)
            if src_has_edges:
                keep = s[src_ctx["edge_src"]] > 0
                feat["srcedge_ei"] = src_ctx["edge_ei"][:, keep]
                feat["srcedge_ea"] = src_ctx["edge_ea"][keep]
                feat["srcedge_val"] = sv

        src_pool_idx = np.flatnonzero(src_ctx["tr_pool"])
        src_fmin, src_fmax = src_ctx["frac"]
        src_rng = src_ctx["rng"]
        if sample_w_t is None:
            # the rung-0 protocol zeroes per-row weights, so force the weighted loss
            # path even when depth-aware weighting is off (uniform ones == plain
            # mean for edges-only arms, so forcing is harmless there).
            sample_w_t = torch.ones(len(sv), dtype=torch.float32, device=device)
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
        drawn_t = None
        if src_on:
            k = int(round(src_rng.uniform(src_fmin, src_fmax) * len(src_pool_idx)))
            drawn = np.zeros(len(tr), dtype=bool)
            drawn[src_rng.choice(src_pool_idx, size=k, replace=False)] = True
            _apply_sources(drawn)
            drawn_t = torch.as_tensor(drawn, device=device)
            if src_has_x:
                w_tr = sample_w_t[tr_t] * (~drawn_t[tr_t]).float()
        out = model(feat)
        if args.pinball:
            primary, pin = out
            hl = huber(primary[tr_t], y_std[tr_t])
            hl = (hl * w_tr).sum() / w_tr.sum() if weighted else hl
            loss = hl + args.pinball_weight * pinball_loss(
                pin[tr_t], y_std[tr_t], eff_tau, w_tr
            )
            pred_point = primary
        elif getattr(args, "sigma_head", False):
            # heteroscedastic Laplace NLL: |err|/b + log b with per-well b. The model
            # spends variance-budget on wells it cannot fit (deep-regional), which
            # implicitly downweights them -- and the OOF b is the selective-call score.
            lb = model.sigma_log_b[tr_t]
            nll = torch.abs(out[tr_t] - y_std[tr_t]) * torch.exp(-lb) + lb
            loss = (nll * w_tr).sum() / w_tr.sum() if weighted else nll.mean()
            pred_point = out
        elif getattr(args, "two_surface", False):
            # WP2 mixture Laplace NLL: -log[pi*Lap(y;hp,bp) + (1-pi)*Lap(y;hr,br)]
            # via logsumexp (the log 2 constant is dropped from BOTH components).
            # Soft responsibilities let each well pull only the component that
            # explains it -- the mechanism the plan's separated-estimand argument
            # needs and the single-surface loss cannot express.
            ts = model.ts_out
            ll_p = -(
                torch.abs(ts["hp"][tr_t] - y_std[tr_t]) * torch.exp(-ts["lbp"][tr_t])
                + ts["lbp"][tr_t]
            )
            ll_r = -(
                torch.abs(ts["hr"][tr_t] - y_std[tr_t]) * torch.exp(-ts["lbr"][tr_t])
                + ts["lbr"][tr_t]
            )
            m_tr = ts["m"][tr_t]
            row = -torch.logsumexp(
                torch.stack(
                    [
                        nn.functional.logsigmoid(m_tr) + ll_p,
                        nn.functional.logsigmoid(-m_tr) + ll_r,
                    ]
                ),
                dim=0,
            )
            if ts_prior_t is not None:
                # privileged assignment: confidence-scaled BCE toward the
                # construction-metadata prior a. conf = |2a-1| makes neutral
                # priors (a=0.5, e.g. unmatched wells) impose exactly nothing;
                # the prior enters ONLY through the loss, never as a feature.
                a = ts_prior_t[tr_t]
                bce = nn.functional.binary_cross_entropy_with_logits(
                    m_tr, a, reduction="none"
                )
                row = row + args.assign_weight * (2.0 * a - 1.0).abs() * bce
            loss = (row * w_tr).sum() / w_tr.sum() if weighted else row.mean()
            pred_point = out
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
            pi = pair_idx
            if src_on and src_has_x:
                # drop pairs with a drawn-source member: with src_x on, a source's
                # prediction has seen its own obs, so loss through it leaks the label.
                keep = ~drawn_t[pair_idx[0]] & ~drawn_t[pair_idx[1]]
                pi = pair_idx[:, keep]
            if pi.numel():
                dp = pred_point[pi[0]] - pred_point[pi[1]]
                dy = y_std[pi[0]] - y_std[pi[1]]
                loss = loss + pair_w * nn.functional.huber_loss(
                    dp, dy, delta=pair_delta
                )
        if ordinal_y_t is not None and model.ordinal_logits is not None:
            # WP5 ordinal BCE, mean over thresholds then the SAME per-row weighting
            # as the point loss (water pseudo-rows enter as shallow evidence at
            # their water_label_weight).
            bce = nn.functional.binary_cross_entropy_with_logits(
                model.ordinal_logits[tr_t], ordinal_y_t[tr_t], reduction="none"
            ).mean(dim=1)
            obce = (bce * w_tr).sum() / w_tr.sum() if weighted else bce.mean()
            loss = loss + args.ordinal_weight * obce
        loss.backward()
        opt.step()
        model.eval()
        val_mad0 = None
        if src_on:
            if src_val_mix:
                # unassisted half of the mixed criterion: no sources visible.
                _apply_sources(np.zeros(len(src_ctx["tr_pool"]), dtype=bool))
                with torch.no_grad():
                    nat0 = _combine_native(model(feat), y_s, y_c, base, mode, args)
                dtw0 = _native_to_dtw(nat0, base, mode)
                val_mad0 = float(np.nanmedian(np.abs(dtw0[va] - obs_dtw[va])))
            # deterministic val semantics: every eligible train well is a source.
            _apply_sources(src_ctx["tr_pool"])
        with torch.no_grad():
            native = _combine_native(model(feat), y_s, y_c, base, mode, args)
        pred_dtw = _native_to_dtw(native, base, mode)
        val_mad = float(np.nanmedian(np.abs(pred_dtw[va] - obs_dtw[va])))
        if val_mad0 is not None:
            val_mad = 0.5 * (val_mad + val_mad0)
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
    if src_on:
        # final-forward semantics: train+val eligible wells are sources; test rows
        # are in neither pool, so the OOF read at them is the cold eval.
        _apply_sources(src_ctx["trva_pool"])
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
    p.add_argument(
        "--save-models",
        action="store_true",
        help="persist per-fold weights + the full standardization contract for raster "
        "inference (models/fold_*.pt, models/shared_stats.pt, "
        "models/inference_manifest.json); see notes/GNN_INFERENCE_10M_PLAN.md",
    )
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
        "--depth-weight-scale",
        type=float,
        default=0.0,
        help="continuous depth-aware loss weight w = s/(s + obs_dtw), mean-normalized "
        "(0 = off; exclusive with --shallow-weight). Smooth alternative to the "
        "step weight: a 2m well gets ~4x the weight of a 30m well at s=5",
    )
    p.add_argument(
        "--confidence-weight",
        type=float,
        default=0.0,
        help="depth-STRATIFIED confinement-confidence loss weight, alpha in [0,1] "
        "(0 = off). Per-well raw weight = (1-alpha) + alpha*confinement_confidence, then "
        "mean-normalized WITHIN each obs-depth band so the depth mix is preserved (raw "
        "confidence is depth-confounded; see notes/ERROR_SOURCES.md). Down-weights "
        "low-confidence (probable-mislabel) unconfined wells. Multiplies any depth weight. "
        "Wells with no modeled confidence get neutral weight 1.0 (logged).",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="high-confidence training ablation: zero the loss weight for wells with "
        "confinement_confidence < this (and for wells with no confidence). Excludes them "
        "from the fit while keeping them in the scored OOF footprint. 0 = off.",
    )
    p.add_argument(
        "--water-label-weight",
        type=float,
        default=0.25,
        help="loss weight for water-stage pseudo-rows (is_water_pseudo, bundles built "
        "with --water-pseudo-labels). Applied AFTER the depth/confidence weight paths "
        "so water rows are exempt from both (synthetic ids have no confinement label; "
        "--min-confidence must not zero them). 0.0 = rows are loss-inert but still "
        "OOF-predicted (the features-alone routing arm). Ignored when the bundle has "
        "no water rows.",
    )
    p.add_argument(
        "--shore-label-weight",
        type=float,
        default=0.0,
        help="E6: loss weight for land-side shoreline ring pseudo-rows (is_shore_pseudo, "
        "bundles built with --shoreline-points). Applied LAST (after depth/confidence/"
        "water/swl paths) exactly like --water-label-weight: shore ids have no "
        "confinement label so --min-confidence must not zero them. 0.0 = loss-inert but "
        "still OOF-predicted (isolates the query-writeback node-pool effect from the "
        "label effect). Ignored / no-op when the bundle has no is_shore_pseudo column.",
    )
    p.add_argument(
        "--glr-labels",
        default=None,
        help="GLR: per-fold cross-fit gaining/losing-reach screen parquet for the E6 "
        "shore rows (utils/build_glr_labels.py; cell_id + glr_pass_fold_0..N + "
        "glr_pass_full). When set, each shore row's weight in fold f is "
        "--glr-label-weight if its glr_pass_fold_{f} bit is set else 0 -- a PER-FOLD "
        "inclusion mask replacing the global --shore-label-weight (which must stay 0.0). "
        "The screen consults observed WTE, so it is cross-fit by fold exactly like R "
        "(fold f's mask uses only cv_fold!=f training wells); glr_pass_full is diagnostic "
        "only and NEVER weights a CV fold. Every is_shore_pseudo row must be covered "
        "(fail-loud). Off => byte-identical to the --shore-label-weight path. See "
        "notes/plans/GLR_PLAN.md and docs/inference_leakage_prevention.md.",
    )
    p.add_argument(
        "--glr-label-weight",
        type=float,
        default=0.0,
        help="GLR loss weight for shore rows that PASS the per-fold screen (0.0 = "
        "loss-inert flag-off). Ignored without --glr-labels.",
    )
    p.add_argument(
        "--swl-labels",
        default=None,
        help="E4: standalone additive parquet of driller static-water-level (SWL) "
        "AUXILIARY training-label query nodes (is_swl_aux=True, built by "
        "build_swl_labels.py). The base bundle is NOT modified. When set, the rows are "
        "appended to query_nodes and their lateral edges to lateral_edges before any "
        "array derivation, flow into `real=~water&~swl` (excluded from every "
        "standardization fit + val/test metric + scorer panel), get a neutral "
        "standardized-mean MAE embedding, and are weighted by --swl-label-weight. "
        "Absent = byte-identical to the pre-E4 baseline.",
    )
    p.add_argument(
        "--swl-lateral-edges",
        default=None,
        help="E4: SWL aux lateral-edges parquet (query_node_idx local 0..n-1, offset "
        "onto the base frame on append). Defaults to the --swl-labels path with "
        "'_labels'->'_lateral_edges'.",
    )
    p.add_argument(
        "--swl-label-weight",
        type=float,
        default=0.25,
        help="E4: loss weight for SWL aux label rows (is_swl_aux). Applied LAST (after "
        "depth/confidence/water paths) exactly like --water-label-weight: SWL ids have "
        "no confidence label so --min-confidence must not zero them. Ignored without "
        "--swl-labels.",
    )
    p.add_argument(
        "--confidence-table",
        default="/data/ssd2/handily/conus/wte_gnn/confidence_by_canonical.parquet",
        help="parquet keyed by canonical_id with confinement_confidence (deduped); "
        "source for --confidence-weight / --min-confidence. Built from the GWX product "
        "version the bundle was keyed against (previous/, 8-char canonical_id).",
    )
    p.add_argument(
        "--fac-lambda",
        action="store_true",
        help="learned CONVEX blend head: pred = lam*FAC_anchor + (1-lam)*GNN_head, "
        "lam = sigmoid(mlp(query context, anchor, FAC's own DTW)) masked to 0 where "
        "FAC is absent. Unlike --fac-skip/--fac-gate (ADDITIVE anchor the head can "
        "double-count), convexity forces an interpretable per-well mixing weight -- "
        "the OOF lam column is a shallow terrain-coupled-zone map. Head-space "
        "targets only; exclusive with --fac-skip/--fac-gate/--pinball",
    )
    p.add_argument(
        "--sigma-head",
        action="store_true",
        help="heteroscedastic Laplace scale head: loss becomes the Laplace NLL "
        "|err|/b + log b (per-well b), and OOF gnn_sigma_m enables selective "
        "shallow calls (trust predictions only where sigma is small). "
        "Exclusive with --pinball",
    )
    p.add_argument(
        "--ordinal-head",
        action="store_true",
        help="WP5 monotone ordinal shallow head: P(DTW<2m) <= P(DTW<5m) <= P(DTW<10m) "
        "from a shared deepness score + ordered cutpoints (nested by construction, "
        "NOT a threshold on the regression surface). Trained jointly (BCE at "
        "--ordinal-weight) on the same rows/weights as the point loss (water pseudo-"
        "rows count as shallow evidence); OOF p_dtw_lt_{2,5,10}m columns. Composes "
        "with every arm; the point/sigma losses and early-stop criterion are unchanged",
    )
    p.add_argument(
        "--ordinal-weight",
        type=float,
        default=0.3,
        help="weight of the ordinal BCE term added to the point loss",
    )
    p.add_argument(
        "--ordinal-thresholds",
        type=str,
        default="2,5,10",
        help="comma-separated DTW class thresholds in meters (ascending)",
    )
    p.add_argument(
        "--two-surface",
        action="store_true",
        help="WP2 latent two-surface mixture: phreatic + regional component heads, "
        "each ANCHORED on a different physical prior (phreatic: FAC where present "
        "else terrain-mirror; regional: deep-well IDW datum, absent -> R itself) so "
        "component labels cannot swap (plan 7.3 identifiability). Loss = Laplace "
        "mixture NLL over per-component scales + a confidence-scaled BCE tying the "
        "membership head to the privileged construction-metadata prior "
        "(--two-surface-priors; used ONLY in the loss, never as a query feature). "
        "OOF ts_pi_phreatic / ts_native_{p,r} / ts_sigma_{p,r}_m columns. Head-space "
        "targets only; exclusive with --prior-gate/--fac-*/--pinball/--sigma-head",
    )
    p.add_argument(
        "--two-surface-priors",
        type=str,
        default="/data/ssd2/handily/conus/wte_gnn/v02/wp2/assignment_priors.parquet",
        help="privileged P(phreatic) prior parquet keyed by canonical_id "
        "(build_two_surface_priors.py); unmatched wells fall back to 0.5 (inert)",
    )
    p.add_argument(
        "--assign-prior-col",
        type=str,
        default="p_phreatic_construction",
        help="column of --two-surface-priors used as the privileged prior",
    )
    p.add_argument(
        "--assign-weight",
        type=float,
        default=0.3,
        help="weight of the confidence-scaled BCE(membership, privileged prior) term",
    )
    p.add_argument(
        "--prior-gate",
        action="store_true",
        help="learned 3-way softmax mixture over priors: pred = w_fac*FAC_anchor + "
        "w_deep*deep_regional_anchor + w_head*GNN_head. The gate sees the query "
        "context, both anchor values, each prior's own predicted DTW and their "
        "DISAGREEMENT (the regime signal no single covariate carries); absent priors "
        "are masked out of the softmax. OOF gate_w_fac/gate_w_deep/gate_w_head form "
        "a three-regime map (terrain-coupled / deep-regional / free-head). "
        "Head-space targets only; exclusive with --fac-skip/--fac-gate/--fac-lambda/"
        "--pinball; composes with --sigma-head and the depth weights",
    )
    p.add_argument(
        "--mirror-anchor",
        action="store_true",
        help="add a terrain-MIRROR expert to the prior gate (requires --prior-gate): "
        "the subdued-replica shallow prior WTE = z_surf - d (constant DTW = d). "
        "Head-space anomaly = base - d, built trainer-side from the carried "
        "wte_resid_base -- well-free, target-blind, no rebuild. The gate becomes a "
        "4-way softmax [fac, deep, mirror, head] and additionally sees "
        "(fac_base - mirror_base), the scale-consistent 'FAC deeper than d' regime "
        "signal. Gives the gate a shallow ruler where FAC over-deepens off-channel",
    )
    p.add_argument(
        "--mirror-depth-m",
        type=float,
        default=3.0,
        help="mirror offset d in meters (the small distance the water table sits "
        "below the land surface under the mirror hypothesis)",
    )
    p.add_argument(
        "--hang-anchor",
        action="store_true",
        help="add a Dupuit HANG expert to the prior gate (requires --prior-gate and "
        "a bundle built with --dupuit-hang-features): the well-free boundary-"
        "conditioned WTE hang surface, WTE = z_surf - dupuit_hang_dtw_m (kNN-IDW of "
        "top-2-Strahler reach elevations). Head-space anomaly = base - "
        "dupuit_hang_dtw_m; its own predicted DTW is dupuit_hang_dtw_m. The gate "
        "additionally sees the hang-vs-deep DTW disagreement (boundary datum vs "
        "well-IDW deep datum). Expert order [fac, deep, (mirror), hang, head]. "
        "Gives the gate a regional-datum ruler where the well-IDW priors blow up "
        "(Rathdrum-type basins); leak-free (stream elevations only)",
    )
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
    p.add_argument(
        "--portfolio-read",
        action="store_true",
        help="(Phase 6B) add the reference-site portfolio read: each well attends (segment-"
        "softmax) over its <=4 typed reference reaches (ds_datum / up_head / wet / ho_any) in "
        "ONE hop. Supersedes --mainstem-read (its ds_datum edge is one of the four) and is "
        "MUTUALLY EXCLUSIVE with it and with anchors. Requires a bundle built with "
        "build_conus_graph_inputs.py --portfolio-read (portfolio_edges.parquet). The site "
        "reaches (+ their 2-hop channel context) are unioned into the prune set. Writes "
        "gnn_portfolio_attention.parquet. See notes/GNN_PHASE6_PLAN.md 6B.",
    )
    p.add_argument(
        "--spatial-context",
        action="store_true",
        help="add the multi-scale spatial-context read: each well attends (segment-"
        "softmax) over its lattice-snapped ring cells (2000/10000 m radii x 8 octants) carrying "
        "the target-blind covariate bank, with azimuth/ring/rel-elev edge attrs -- the "
        "direction x scale terrain view the point covariates radially average away. "
        "ADDITIVE to the portfolio/mainstem read (own head slot). Requires a bundle "
        "built with build_conus_graph_inputs.py --spatial-context. Writes "
        "gnn_sc_attention.parquet. See notes/SPATIAL_CONTEXT.md.",
    )
    p.add_argument(
        "--sc-no-azimuth",
        action="store_true",
        help="(control arm) drop the sc_sin_az/sc_cos_az edge channels before "
        "standardization, reducing the spatial-context read to a radial mean (ring "
        "one-hot + distance + rel-elev). The isolating ablation for the direction "
        "bet: full ~= no-azimuth means the context nodes just rebuilt the saturated "
        "point covariates.",
    )
    p.add_argument(
        "--mae-embeddings",
        default=None,
        help="path to a mae_embeddings_<arm>_allq.parquet (utils/extract_mae_embeddings.py "
        "--all-query-nodes) keyed by query_node_idx; its mae_* columns ride in as an "
        "ADDITIVE per-query head slot (a self-supervised, target-blind neighborhood "
        "embedding, encoded then concatenated to the head input, composing with the "
        "gate/mirror/sigma arms). MUST cover EVERY query node -- a missing embedding is "
        "an extraction-scope bug, not data to impute, so the trainer fails loud. Off => "
        "byte-identical to baseline. See notes/MAE_NEIGHBORHOOD_EMBEDDING.md.",
    )
    p.add_argument(
        "--extra-query-features",
        default=None,
        help="path to a parquet keyed by query_node_idx whose remaining columns are "
        "appended to the bundle's query_feature_cols as DIRECT query features (same "
        "footing as fac_rem_dtw_m et al.: per-fold z-score + median fill + missingness "
        "indicator). Lets a target-blind, well-free raster covariate be screened "
        "without a graph rebuild; reach-node placement still needs the builder. Must "
        "cover every base query node (fails loud otherwise). Off => byte-identical to "
        "baseline.",
    )
    p.add_argument(
        "--query-writeback",
        action="store_true",
        help="(Phase 6C) add the query->reach write-back conv: well context is written onto "
        "its lateral reaches (reversed lateral edges) as a residual BEFORE the channel stack, "
        "so it mixes 2 hops outward and returns via the lateral/portfolio reads -- the "
        "mechanism-matched lever for the coherent <500m patch bias (well<->well communication "
        "through shared reaches). No rebuild (reuses lateral_edges). See notes/GNN_PHASE6_PLAN.md 6C.",
    )
    p.add_argument(
        "--analog-edges",
        default=None,
        help="(E3) path to an analog_edges.parquet (utils/build_analog_edges.py): kNN "
        "edges in the AEF+MAE embedding space from each well to its <=k NONLOCAL analog "
        "wells (different HUC4, >= min-dist km). Adds an ADDITIVE segment-softmax read "
        "slot that imports each analog's fold-standardized observed residual. Fold-safe "
        "by construction: edges whose SOURCE well is in the held-out fold are dropped "
        "from that fold's forward (a test well never reads a test-fold label). Off => "
        "byte-identical to baseline. See notes/E3_ANALOG_EDGES.md.",
    )
    # --- source-well assimilation (rung 0): obs-as-input features + masked labels ---
    p.add_argument(
        "--source-obs",
        action="store_true",
        help="(assimilation rung 0) give each real training well's observed target "
        "(fold-standardized) as a 2-col query feature block [value*valid, valid] and "
        "train with masked labels: each epoch a fresh random fraction "
        "~U(--source-frac-min, --source-frac-max) of eligible train wells become "
        "SOURCES (obs visible, loss weight 0); the rest stay targets (obs zeroed, "
        "loss on) -- a row never trains on a label it can see. Val/early-stop "
        "forward shows ALL train sources; the final OOF forward shows train+val "
        "sources (held-out test rows never carry their own obs, so the OOF is the "
        "cold eval). Requires --query-writeback -- the only path an obs can take to "
        "another query at rung 0. See notes/SOURCE_WELL_ASSIMILATION_PLAN.md.",
    )
    p.add_argument(
        "--source-val-mix",
        action="store_true",
        help="early-stop on the MEAN of two val MADs: a no-source forward and the "
        "all-train-sources forward. Guards model selection against edge-reliant "
        "weights that lose unassisted (far-from-well / cold) skill -- rung 1's "
        "second protocol defect (plan section 9).",
    )
    p.add_argument("--source-frac-min", type=float, default=0.3)
    p.add_argument("--source-frac-max", type=float, default=0.9)
    p.add_argument(
        "--source-edges",
        default=None,
        help="(assimilation rung 1) path to a source_edges.parquet "
        "(utils/build_source_edges.py): each query's <=k nearest eligible source "
        "wells as direct spatial edges, attention-read as an additive head slot "
        "('learned IDW'). The masked-label protocol governs edge visibility: per "
        "epoch only the drawn train-pool sources keep their edges; val forward "
        "keeps all-train sources' edges; the final OOF "
        "forward keeps train+val (held-out rows never feed an edge, so the OOF "
        "stays the cold eval). Composes with --source-obs (rung 0 features) but "
        "does not require it or --query-writeback -- the edges are their own "
        "transmission path. Edges-only arms keep full label mass (a source never "
        "sees its own label through an edge, so its loss is legitimate); only "
        "--source-obs arms loss-zero drawn sources. "
        "See SOURCE_WELL_ASSIMILATION_PLAN.md sections 4/8/9.",
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

    # --- E4: append SWL auxiliary training labels (additive; base bundle untouched) ---
    # SWL rows carry local query_node_idx 0..n_aux-1; offset them onto the base frame
    # and offset their lateral edges likewise, then let the invariants below validate
    # the combined frame. Absent --swl-labels this block is a no-op (byte-identical).
    swl_appended = 0
    if args.swl_labels:
        base_n = len(qn)
        sqn = (
            pd.read_parquet(args.swl_labels)
            .sort_values("query_node_idx")
            .reset_index(drop=True)
        )
        sle_path = args.swl_lateral_edges or args.swl_labels.replace(
            "_labels.parquet", "_lateral_edges.parquet"
        )
        sle = pd.read_parquet(sle_path)
        if "is_swl_aux" not in qn.columns:
            qn["is_swl_aux"] = False
        for c in set(qn.columns) - set(sqn.columns):
            sqn[c] = np.nan
        sqn = sqn[qn.columns].copy()
        sqn["is_swl_aux"] = True
        sqn["query_node_idx"] = np.arange(base_n, base_n + len(sqn), dtype="int64")
        sle = sle.copy()
        sle["query_node_idx"] = sle["query_node_idx"].to_numpy("int64") + base_n
        qn = pd.concat([qn, sqn], ignore_index=True)
        le = pd.concat([le[sle.columns], sle], ignore_index=True)
        swl_appended = len(sqn)
        log.info(
            "E4 SWL aux labels: %d appended (weight %.2f); base=%d -> total=%d",
            swl_appended,
            args.swl_label_weight,
            base_n,
            len(qn),
        )

    assert (rn["reach_node_idx"].to_numpy() == np.arange(len(rn))).all()
    assert (qn["query_node_idx"].to_numpy() == np.arange(len(qn))).all()

    # --- additive DIRECT query features from a sidecar parquet ------------------------
    # Same semantics as a builder-added query column (e.g. fac_rem_dtw_m): the columns
    # join into `qn` and extend `query_feature_cols`, so they ride the per-fold
    # fit_stats/apply_stats path (train-fold z-score + median fill + missingness
    # indicator) exactly like every native query feature. This exists so a
    # target-blind, well-free raster covariate can be screened as a query feature
    # without a multi-hour graph rebuild; the reach-node placement still requires the
    # builder. Sidecar columns must be target-blind by construction -- the loader does
    # not and cannot verify that.
    extra_query_cols: list[str] = []
    if args.extra_query_features:
        xq = pd.read_parquet(args.extra_query_features)
        if "query_node_idx" not in xq.columns:
            raise SystemExit(
                f"--extra-query-features {args.extra_query_features} lacks "
                "query_node_idx"
            )
        if xq["query_node_idx"].duplicated().any():
            raise SystemExit(
                f"--extra-query-features {args.extra_query_features} has duplicate "
                "query_node_idx"
            )
        extra_query_cols = [c for c in xq.columns if c != "query_node_idx"]
        if not extra_query_cols:
            raise SystemExit(
                f"--extra-query-features {args.extra_query_features} carries no "
                "feature columns besides query_node_idx"
            )
        clash = sorted(set(extra_query_cols) & set(qn.columns))
        if clash:
            raise SystemExit(
                f"--extra-query-features would shadow existing query_nodes columns: "
                f"{clash}; rename them in the sidecar"
            )
        xq = xq.set_index("query_node_idx")
        qidx_all = qn["query_node_idx"].to_numpy()
        # Aux/pseudo rows (SWL) may legitimately be absent from a sidecar built on the
        # base bundle; a missing BASE row is a build-scope bug, not data to impute.
        base_q = (
            qidx_all[~qn["is_swl_aux"].to_numpy(bool)]
            if ("is_swl_aux" in qn.columns)
            else qidx_all
        )
        missing = np.setdiff1d(base_q, xq.index.to_numpy())
        if len(missing):
            raise SystemExit(
                f"--extra-query-features covers {len(xq)} nodes; {len(missing)} base "
                "query nodes have NO value -- rebuild the sidecar over ALL base query "
                "nodes (missing rows must NOT be imputed here)."
            )
        for c in extra_query_cols:
            qn[c] = xq[c].reindex(qidx_all).to_numpy("float64")
        query_cols = list(query_cols) + extra_query_cols
        log.info(
            "extra query features ON: %s (%d cols: %s; NaN%% %s)",
            args.extra_query_features,
            len(extra_query_cols),
            ", ".join(extra_query_cols),
            ", ".join(
                f"{c}={100.0 * qn[c].isna().mean():.2f}" for c in extra_query_cols
            ),
        )

    # Water-stage pseudo-rows (labels only, never metrics): default-False for
    # bundles predating --water-pseudo-labels. `real` masks every fit/metric.
    water = (
        qn["is_water_pseudo"].to_numpy(bool)
        if "is_water_pseudo" in qn.columns
        else np.zeros(len(qn), bool)
    )
    # E4 SWL aux rows are auxiliary labels too: never a fit source, never a metric.
    swl = (
        qn["is_swl_aux"].to_numpy(bool)
        if "is_swl_aux" in qn.columns
        else np.zeros(len(qn), bool)
    )
    # E6 land-side shoreline ring pseudo-rows: same discipline as water pseudo-rows
    # (never a fit source, never a metric). Default-False for bundles predating
    # --shoreline-points, so flag-off training is byte-identical.
    shore = (
        qn["is_shore_pseudo"].to_numpy(bool)
        if "is_shore_pseudo" in qn.columns
        else np.zeros(len(qn), bool)
    )
    real = ~water & ~swl & ~shore
    if shore.any():
        log.info(
            "shore pseudo-rows: %d/%d query rows (label weight %.2f; excluded from "
            "val/test metrics + standardization fits)",
            int(shore.sum()),
            len(qn),
            args.shore_label_weight,
        )
    if water.any():
        log.info(
            "water pseudo-rows: %d/%d query rows (label weight %.2f; excluded from "
            "val/test metrics + standardization fits)",
            int(water.sum()),
            len(qn),
            args.water_label_weight,
        )

    # --- GLR: per-fold cross-fit shore-label inclusion mask ---------------------------
    # A shore row's fold-f training weight becomes --glr-label-weight iff its
    # glr_pass_fold_{f} bit is set (else 0), replacing the global --shore-label-weight.
    # The screen consults observed WTE, so fold f's mask is derived ONLY from cv_fold!=f
    # training wells (build_glr_labels.py) -- leak-free by the crossfit-R contract. The
    # join is on the canonical 100 m EPSG:5070 lattice cell id (the same grid the shore
    # rows were snapped to); every is_shore_pseudo row must carry a GLR bit (fail-loud).
    glr_active = bool(args.glr_labels)
    glr_pass_by_fold = None
    glr_meta = None
    if glr_active:
        if not shore.any():
            raise SystemExit(
                "--glr-labels set but the bundle has no is_shore_pseudo rows "
                "(rebuild with build_conus_graph_inputs.py --shoreline-points)"
            )
        glr = pd.read_parquet(args.glr_labels)
        lat_x0, lat_y0, lat_res = -2540000.0, 3258000.0, 100.0

        def _cell_id(x, y):
            col = np.floor((x - lat_x0) / lat_res).astype("int64")
            row = np.floor((lat_y0 - y) / lat_res).astype("int64")
            return col * 100_000_000 + row

        qn_cid = _cell_id(
            qn["x5070"].to_numpy("float64"), qn["y5070"].to_numpy("float64")
        )
        fold_vals = sorted(int(f) for f in qn[fold_col].unique())
        glr_row = pd.Series(
            np.arange(len(glr), dtype="int64"), index=glr["cell_id"].to_numpy("int64")
        )
        if glr_row.index.duplicated().any():
            raise SystemExit("--glr-labels has duplicate cell_id (join not 1:1)")
        shore_gpos = glr_row.reindex(qn_cid[shore]).to_numpy()
        if np.isnan(shore_gpos).any():
            raise SystemExit(
                f"{int(np.isnan(shore_gpos).sum())} of {int(shore.sum())} shore rows "
                "have no GLR label (cell_id join gap -- extraction-scope bug, not data "
                "to impute); every is_shore_pseudo row must be covered"
            )
        shore_rows = shore_gpos.astype("int64")
        glr_pass_by_fold = {}
        for f in fold_vals:
            col = glr[f"glr_pass_fold_{f}"].to_numpy(bool)
            passf = np.zeros(len(qn), bool)
            passf[shore] = col[shore_rows]
            glr_pass_by_fold[f] = passf
        per_fold_pass = {f: int(glr_pass_by_fold[f].sum()) for f in fold_vals}
        glr_meta = {
            "labels_path": args.glr_labels,
            "label_weight": float(args.glr_label_weight),
            "n_shore_rows": int(shore.sum()),
            "per_fold_shore_pass": per_fold_pass,
            "n_pass_full": int(glr["glr_pass_full"].to_numpy(bool).sum()),
        }
        if args.shore_label_weight != 0.0:
            log.warning(
                "--glr-labels overrides --shore-label-weight per fold; the global "
                "--shore-label-weight=%.2f is ignored for shore rows",
                args.shore_label_weight,
            )
        log.info(
            "GLR active: %d shore rows, pass-weight %.2f; per-fold PASS counts %s "
            "(cross-fit leave-one-fold-out; glr_pass_full=%d diagnostic only)",
            int(shore.sum()),
            args.glr_label_weight,
            per_fold_pass,
            glr_meta["n_pass_full"],
        )

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

    # --- portfolio-read edges (6B): <=4 typed reference reaches per query --------------
    use_pf = args.portfolio_read
    pf = None
    pf_cols = None
    if use_pf:
        if use_ms:
            raise SystemExit(
                "--portfolio-read and --mainstem-read are mutually exclusive (the "
                "portfolio ds_datum site is a superset of the mainstem read)"
            )
        pf_path = gdir / "portfolio_edges.parquet"
        if not pf_path.exists() or not man.get("portfolio_read"):
            raise SystemExit(
                "--portfolio-read set but the bundle has no portfolio_edges.parquet / "
                "portfolio_read manifest block -- rebuild the graph with "
                "build_conus_graph_inputs.py --portfolio-read"
            )
        pf = pd.read_parquet(pf_path)
        pf_cols = man["portfolio_read"]["feature_cols"]

    # --- spatial-context nodes/edges: rings x 8 octants of lattice cells per query -----
    # SC cells are their OWN node set (indexed by sc_node_idx, independent of reach
    # pruning), so unlike the portfolio sites they need no prune-union or remap.
    use_sc = args.spatial_context
    if args.sc_no_azimuth and not use_sc:
        raise SystemExit("--sc-no-azimuth requires --spatial-context")
    scn = sce = None
    sc_node_cols = sc_edge_cols = None
    if use_sc:
        scn_path = gdir / "spatial_context_nodes.parquet"
        sce_path = gdir / "spatial_context_edges.parquet"
        if (
            not scn_path.exists()
            or not sce_path.exists()
            or not man.get("spatial_context")
        ):
            raise SystemExit(
                "--spatial-context set but the bundle has no spatial_context parquets "
                "/ manifest block -- rebuild the graph with "
                "build_conus_graph_inputs.py --spatial-context"
            )
        scn = pd.read_parquet(scn_path)
        sce = pd.read_parquet(sce_path)
        sc_node_cols = man["spatial_context"]["node_feature_cols"]
        sc_edge_cols = list(man["spatial_context"]["edge_feature_cols"])
        if args.sc_no_azimuth:
            # Arm-2 control: strip the azimuth channels BEFORE standardization so the
            # read degrades to a radial mean -- the isolating direction ablation.
            sc_edge_cols = [
                c for c in sc_edge_cols if c not in ("sc_sin_az", "sc_cos_az")
            ]

    # --- prune reach graph to the n-hop neighborhood of attached reaches -------
    # The mainstem datum reaches (+ their own 2-hop channel context) are unioned in so
    # the read edge actually reaches a node in the pruned GPU graph -- this is what puts
    # the mainstem back into the receptive field.
    attached = np.unique(le["reach_node_idx"].to_numpy("int64"))
    if use_ms:
        attached = np.unique(
            np.concatenate([attached, me["reach_node_idx"].to_numpy("int64")])
        )
    if use_pf:
        attached = np.unique(
            np.concatenate([attached, pf["reach_node_idx"].to_numpy("int64")])
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
    if use_pf:
        pf = pf.copy()
        pf["reach_node_idx"] = old2new[pf["reach_node_idx"].to_numpy("int64")]
        assert (pf["reach_node_idx"] >= 0).all(), "portfolio site reach pruned (bug)"
        log.info(
            "portfolio-read: %d site read edges (unioned into the prune set)", len(pf)
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
    if use_pf and use_anchors:
        raise SystemExit(
            "--portfolio-read is mutually exclusive with anchors (the head reconciles one "
            "extra read context); pass --no-anchors or use a bundle without anchors"
        )
    if args.query_writeback and use_anchors:
        raise SystemExit(
            "--query-writeback is wired only into the non-anchor forward branch; pass "
            "--no-anchors or use a bundle without anchors"
        )
    if args.source_obs and not args.query_writeback:
        raise SystemExit(
            "--source-obs requires --query-writeback: the write-back is the only path "
            "a source well's obs can take to another query (rung 0)"
        )
    if (args.source_obs or args.source_edges) and not (
        0.0 < args.source_frac_min <= args.source_frac_max < 1.0
    ):
        raise SystemExit("--source-frac-min/max must satisfy 0 < min <= max < 1")
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
    # portfolio-read edges (6B): standardized exactly like lat_ea (train-blind edge attrs,
    # median-impute + missingness flag); reach->query direction (site reach = src). The
    # type one-hots are z-scored the same way datum_is_self already is -- harmless.
    pf_ea = None
    if use_pf:
        pf_stats = fit_stats(pf, pf_cols, None)
        pf_ea = torch.as_tensor(
            apply_stats(pf, pf_stats), dtype=torch.float32, device=device
        )
        pf_ei = torch.as_tensor(
            pf[["reach_node_idx", "query_node_idx"]].to_numpy().T,
            dtype=torch.long,
            device=device,
        )
        graph_tensors |= {"pf_ei": pf_ei, "pf_ea": pf_ea}
    # spatial-context tensors: cell covariate bank standardized like reach_x, edge attrs
    # like lat_ea (median-impute + missingness flags); cell->query direction (cell = src).
    # Both are target-blind constants, so a global fit is train-blind by construction.
    sc_x_t = sc_ea_t = None
    if use_sc:
        sc_x_t = torch.as_tensor(
            apply_stats(scn, fit_stats(scn, sc_node_cols, None)),
            dtype=torch.float32,
            device=device,
        )
        sc_ea_t = torch.as_tensor(
            apply_stats(sce, fit_stats(sce, sc_edge_cols, None)),
            dtype=torch.float32,
            device=device,
        )
        sc_ei = torch.as_tensor(
            sce[["sc_node_idx", "query_node_idx"]].to_numpy().T,
            dtype=torch.long,
            device=device,
        )
        graph_tensors |= {"sc_x": sc_x_t, "sc_ei": sc_ei, "sc_ea": sc_ea_t}
        log.info(
            "spatial-context ON%s: %d cells, %d edges, node width %d, edge width %d",
            " (no-azimuth control)" if args.sc_no_azimuth else "",
            len(scn),
            len(sce),
            sc_x_t.shape[1],
            sc_ea_t.shape[1],
        )
    # MAE neighborhood-embedding tensor: a per-query, target-blind vector standardized
    # ONCE globally (z-score) -- like reach_x/sc_x it is a target-blind constant, so a
    # global fit is train-blind by construction. Rides in as an ADDITIVE per-query head
    # slot. It MUST cover every query node: the forward runs over ALL query nodes, and a
    # missing embedding is an extraction-scope bug (e.g. water pseudo-rows omitted), NOT
    # data to impute -- so we fail loud and point at --all-query-nodes.
    f_mae = None
    if args.mae_embeddings:
        mae_df = pd.read_parquet(args.mae_embeddings)
        if "query_node_idx" not in mae_df.columns:
            raise SystemExit(
                f"--mae-embeddings {args.mae_embeddings} lacks query_node_idx"
            )
        if mae_df["query_node_idx"].duplicated().any():
            raise SystemExit(
                f"--mae-embeddings {args.mae_embeddings} has duplicate query_node_idx"
            )
        mae_cols = [c for c in mae_df.columns if c.startswith("mae_")]
        if not mae_cols:
            raise SystemExit(
                f"--mae-embeddings {args.mae_embeddings} has no mae_* columns"
            )
        mae_df = mae_df.set_index("query_node_idx")
        qidx = qn["query_node_idx"].to_numpy()
        if args.swl_labels and swl.any():
            # E4: real embedding for base rows; a neutral (standardized-mean = 0)
            # embedding for SWL aux rows. Extracting real AEF+MAE neighborhood reads
            # for the aux points is the pre-registered follow-on IF an arm shows signal;
            # a weight-0.1-0.25 aux label's value is its target + query features + graph
            # context, not its own raster neighborhood read. Stats are fit on the base
            # (non-aux) rows only, so real-well embeddings are unperturbed.
            base_q = qidx[~swl]
            missing = np.setdiff1d(base_q, mae_df.index.to_numpy())
            if len(missing):
                raise SystemExit(
                    f"--mae-embeddings covers {len(mae_df)} nodes; {len(missing)} base "
                    "query nodes have NO embedding -- re-extract with --all-query-nodes."
                )
            base_emb = mae_df.loc[base_q, mae_cols].to_numpy("float64")
            if not np.isfinite(base_emb).all():
                raise SystemExit("--mae-embeddings has non-finite values; investigate.")
            emb = np.empty((len(qn), len(mae_cols)), "float64")
            emb[~swl] = base_emb
            emb[swl] = base_emb.mean(0, keepdims=True)  # -> standardizes to 0
            mae_ord = pd.DataFrame(emb, columns=mae_cols)
            mae_stats = fit_stats(
                mae_ord.loc[~swl].reset_index(drop=True), mae_cols, None
            )
            mae_x_t = torch.as_tensor(
                apply_stats(mae_ord, mae_stats), dtype=torch.float32, device=device
            )
        else:
            missing = np.setdiff1d(qidx, mae_df.index.to_numpy())
            if len(missing):
                raise SystemExit(
                    f"--mae-embeddings covers {len(mae_df)}/{len(qn)} query nodes; "
                    f"{len(missing)} have NO embedding (likely water pseudo-rows). The "
                    "forward runs over every query node -- re-extract with "
                    "extract_mae_embeddings.py --all-query-nodes (missing rows must NOT "
                    "be imputed)."
                )
            mae_ord = mae_df.loc[qidx, mae_cols].reset_index(drop=True)
            if not np.isfinite(mae_ord.to_numpy()).all():
                raise SystemExit(
                    "--mae-embeddings has non-finite values; investigate before use "
                    "(embeddings must be finite everywhere)"
                )
            mae_x_t = torch.as_tensor(
                apply_stats(mae_ord, fit_stats(mae_ord, mae_cols, None)),
                dtype=torch.float32,
                device=device,
            )
        graph_tensors["mae_x"] = mae_x_t
        f_mae = mae_x_t.shape[1]
        log.info(
            "MAE embeddings ON: %s (%d query nodes x %d dims, global z-score)",
            args.mae_embeddings,
            mae_x_t.shape[0],
            f_mae,
        )
    # --- analog edges (E3): nonlocal embedding-similarity well->well read ----------
    # Edge INDEX + edge ATTRS are target-blind constants (built + standardized once); the
    # per-edge SOURCE-FOLD is carried so the fold loop can drop edges whose source is in
    # the held-out fold (leak-free). The per-node SOURCE VALUE (fold-standardized target)
    # is built per fold in the loop -- it is the only fold-dependent piece.
    f_analog = None
    analog_src_t = analog_dst_t = analog_ea_t = analog_src_fold = None
    if args.analog_edges:
        ae = pd.read_parquet(args.analog_edges)
        need = {"query_node_idx", "src_query_node_idx", "src_cv_fold"}
        if not need.issubset(ae.columns):
            raise SystemExit(
                f"--analog-edges {args.analog_edges} missing {need - set(ae.columns)}"
            )
        ae_cols = ["cos_dist", "geo_dist_km", "rel_elev_m"]
        if not set(ae_cols).issubset(ae.columns):
            raise SystemExit(
                f"--analog-edges {args.analog_edges} missing edge-feature cols {ae_cols}"
            )
        dst = ae["query_node_idx"].to_numpy("int64")
        src = ae["src_query_node_idx"].to_numpy("int64")
        if (
            dst.max() >= len(qn)
            or src.max() >= len(qn)
            or min(dst.min(), src.min()) < 0
        ):
            raise SystemExit("--analog-edges references out-of-range query_node_idx")
        if (dst == src).any():
            raise SystemExit("--analog-edges has self-edges (source == dest)")
        # sources must be real wells (they carry the observed residual payload); a
        # water-pseudo source would import a stage pseudo-label, not a well observation.
        if water[src].any():
            raise SystemExit(
                f"{int(water[src].sum())} analog edges source from water-pseudo nodes; "
                "rebuild with build_analog_edges.py (real-well sources only)"
            )
        ae_ea_np = apply_stats(ae, fit_stats(ae, ae_cols, None))
        if not np.isfinite(ae_ea_np).all():
            raise SystemExit("--analog-edges has non-finite edge features")
        analog_src_t = torch.as_tensor(src, dtype=torch.long, device=device)
        analog_dst_t = torch.as_tensor(dst, dtype=torch.long, device=device)
        analog_ea_t = torch.as_tensor(ae_ea_np, dtype=torch.float32, device=device)
        analog_src_fold = ae["src_cv_fold"].to_numpy("int64")
        f_analog = analog_ea_t.shape[1]
        deg = pd.Series(dst).value_counts()
        log.info(
            "analog edges ON: %s (%d edges, %d dest wells, mean degree %.1f, %d edge "
            "features, fold-masked per fold)",
            args.analog_edges,
            len(ae),
            deg.size,
            float(deg.mean()),
            f_analog,
        )

    def analog_feat(fold: int | None, y_c: float, y_s: float) -> dict:
        """Per-fold analog tensors: edges with source NOT in ``fold`` (leak-free), and the
        per-node source value = fold-standardized target. ``fold=None`` keeps every edge
        (memory-probe sizing only). Empty dict when analog edges are off."""
        if f_analog is None:
            return {}
        keep = torch.as_tensor(
            analog_fold_keep(analog_src_fold, fold), dtype=torch.bool, device=device
        )
        val = torch.as_tensor((target - y_c) / y_s, dtype=torch.float32, device=device)
        return {
            "analog_ei": torch.stack([analog_src_t[keep], analog_dst_t[keep]]),
            "analog_ea": analog_ea_t[keep],
            "analog_src_val": val,
        }

    # query->reach write-back (6C): reversed lateral edges (query src -> reach dst), reusing
    # lat_ea unchanged (same attrs, opposite direction). No new file / no prune change.
    if args.query_writeback:
        graph_tensors |= {"lat_ei_reversed": lat_ei.flip(0)}
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

    # --- FAC raw-skip / lambda blend / prior gate: (head-space) anchor tensors ------
    fac_skip = args.fac_skip
    fac_gate = args.fac_gate
    fac_lambda = args.fac_lambda
    prior_gate = args.prior_gate
    two_surface = args.two_surface
    fac_raw = fac_present = fac_pred_dtw = None
    deep_raw = deep_present = deep_pred_dtw = deep_anchor_col = None
    if fac_gate and not fac_skip:
        raise SystemExit("--fac-gate requires --fac-skip")
    if fac_lambda and (fac_skip or fac_gate):
        raise SystemExit("--fac-lambda is exclusive with --fac-skip/--fac-gate")
    if prior_gate and (fac_skip or fac_gate or fac_lambda):
        raise SystemExit(
            "--prior-gate is exclusive with --fac-skip/--fac-gate/--fac-lambda"
        )
    if (fac_lambda or prior_gate) and args.pinball:
        raise SystemExit("--fac-lambda/--prior-gate are exclusive with --pinball")
    if args.sigma_head and args.pinball:
        raise SystemExit("--sigma-head is exclusive with --pinball")
    if two_surface and (
        prior_gate or fac_skip or fac_gate or fac_lambda or args.pinball
    ):
        # the two-component mixture REPLACES every other output mixture/anchor path.
        raise SystemExit(
            "--two-surface is exclusive with --prior-gate/--fac-*/--pinball"
        )
    if two_surface and args.sigma_head:
        # the mixture already carries per-component Laplace scales; a third
        # point-scale head would double-count the NLL.
        raise SystemExit("--two-surface is exclusive with --sigma-head")
    use_fac_anchor = fac_skip or fac_lambda or prior_gate or two_surface
    if use_fac_anchor:
        flag = (
            "--fac-skip"
            if fac_skip
            else (
                "--fac-lambda"
                if fac_lambda
                else ("--two-surface" if two_surface else "--prior-gate")
            )
        )
        if target_mode == TARGET_WTE_RESIDUAL:
            fac_base_col = FAC_REM_WTE_ANOM_COL  # (z_surf - fac_rem_dtw) - R
        elif target_mode == TARGET_WTE:
            fac_base_col = FAC_REM_WTE_COL  # fac_rem water-surface ELEVATION
        else:
            raise SystemExit(
                f"{flag} requires a head-space target (wte / wte_residual); "
                f"got {target_mode}"
            )
        if fac_base_col not in qn.columns:
            raise SystemExit(f"{flag}: column {fac_base_col!r} not in query nodes")
        fac_raw = qn[fac_base_col].to_numpy("float64")
        fac_present = np.isfinite(fac_raw)
        log.info(
            "%s ON: anchor col %s, %d/%d wells carry FAC (%.1f%%)%s",
            flag,
            fac_base_col,
            int(fac_present.sum()),
            len(qn),
            100.0 * fac_present.mean(),
            " | gate ON" if fac_gate else "",
        )
        if fac_gate or fac_lambda or prior_gate:
            # FAC's own predicted DTW; in head-space base - fac_raw == fac_rem_dtw.
            # The gate/lambda keys on this to release the anchor where FAC predicts
            # deep (the saturation regime).
            fac_pred_dtw = np.where(fac_present, base - fac_raw, np.nan)
    if prior_gate or two_surface:
        # Deep expert: crossfit regional-deep well IDW, the only prior that wins the
        # 30+m band. Same head-space convention as the FAC anchor. The two-surface
        # arm anchors its REGIONAL component on it.
        deep_anchor_col = (
            DEEP_REGIONAL_WTE_ANOM_COL
            if target_mode == TARGET_WTE_RESIDUAL
            else DEEP_REGIONAL_WTE_COL
        )
        if deep_anchor_col not in qn.columns:
            raise SystemExit(
                f"--prior-gate: column {deep_anchor_col!r} not in query nodes"
            )
        deep_raw = qn[deep_anchor_col].to_numpy("float64")
        deep_present = np.isfinite(deep_raw)
        deep_pred_dtw = np.where(deep_present, base - deep_raw, np.nan)
        log.info(
            "--prior-gate deep expert: anchor col %s, %d/%d wells carry deep (%.1f%%)",
            deep_anchor_col,
            int(deep_present.sum()),
            len(qn),
            100.0 * deep_present.mean(),
        )
    mirror_raw = mirror_present = mirror_pred_dtw = None
    if args.mirror_anchor and not prior_gate:
        raise SystemExit("--mirror-anchor requires --prior-gate")
    if args.mirror_anchor or two_surface:
        # Terrain-mirror expert: WTE = z_surf - d, i.e. head-space anomaly = base - d
        # (base is z_surf - R in wte_residual mode, z_surf in wte mode -- both work).
        # Its own predicted DTW is the constant d. Well-free + target-blind. The
        # two-surface arm falls back to it as the PHREATIC anchor where FAC is absent.
        mirror_raw = base - args.mirror_depth_m
        mirror_present = np.isfinite(mirror_raw)
        mirror_pred_dtw = np.where(mirror_present, args.mirror_depth_m, np.nan)
        log.info(
            "--mirror-anchor ON: terrain-mirror expert WTE = z_surf - %.1f m "
            "(%d/%d wells carry it, %.1f%%)",
            args.mirror_depth_m,
            int(mirror_present.sum()),
            len(qn),
            100.0 * mirror_present.mean(),
        )
    hang_raw = hang_present = hang_pred_dtw = None
    if args.hang_anchor:
        if not prior_gate:
            raise SystemExit("--hang-anchor requires --prior-gate")
        if HANG_DTW_COL not in qn.columns:
            raise SystemExit(
                f"--hang-anchor: column {HANG_DTW_COL!r} not in query nodes -- "
                "rebuild the bundle with --dupuit-hang-features"
            )
        # Dupuit hang expert: WTE = z_surf - hang_dtw -> head-space anomaly =
        # base - hang_dtw (same convention as the deep expert). Well-free
        # (stream elevations only), target-blind, no cross-fit needed.
        hang_dtw = qn[HANG_DTW_COL].to_numpy("float64")
        hang_raw = base - hang_dtw
        hang_present = np.isfinite(hang_raw)
        hang_pred_dtw = np.where(hang_present, hang_dtw, np.nan)
        log.info(
            "--hang-anchor ON: Dupuit hang expert WTE = z_surf - %s "
            "(%d/%d wells carry it, %.1f%%; hang DTW median %.1f m)",
            HANG_DTW_COL,
            int(hang_present.sum()),
            len(qn),
            100.0 * hang_present.mean(),
            float(np.nanmedian(hang_pred_dtw)),
        )

    # --- depth-aware loss weighting: upweight the shallow band (FAC's gold) ---------
    sample_w_t = None
    if args.depth_weight_scale > 0.0 and args.shallow_weight != 1.0:
        raise SystemExit("--depth-weight-scale is exclusive with --shallow-weight")
    if args.depth_weight_scale > 0.0:
        s = args.depth_weight_scale
        sample_w = (s / (s + np.maximum(obs_dtw, 0.0))).astype("float32")
        sample_w /= sample_w.mean()
        sample_w_t = torch.as_tensor(sample_w, dtype=torch.float32, device=device)
        log.info(
            "continuous depth-aware loss: w = %.1f/(%.1f + obs_dtw), mean-normalized "
            "(w at 0m=%.2f, 5m=%.2f, 30m=%.2f)",
            s,
            s,
            float(sample_w.max()),
            float(np.median(sample_w[np.abs(obs_dtw - 5.0) < 1.0]))
            if (np.abs(obs_dtw - 5.0) < 1.0).any()
            else float("nan"),
            float(np.median(sample_w[obs_dtw > 30.0]))
            if (obs_dtw > 30.0).any()
            else float("nan"),
        )
    elif args.shallow_weight != 1.0:
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

    conf_meta = None
    if args.confidence_weight > 0.0 or args.min_confidence > 0.0:
        ctab = pd.read_parquet(
            args.confidence_table, columns=["canonical_id", "confinement_confidence"]
        ).drop_duplicates("canonical_id")
        cmap = dict(
            zip(
                ctab["canonical_id"].to_numpy(),
                ctab["confinement_confidence"].to_numpy("float64"),
            )
        )
        conf = qn["canonical_id"].map(cmap).astype("float64").to_numpy()
        present = np.isfinite(conf)
        n_null = int((~present).sum())
        # start from the existing depth weight (or ones), multiply in confidence
        base_w = (
            sample_w_t.detach().cpu().numpy().astype("float64")
            if sample_w_t is not None
            else np.ones(len(qn), "float64")
        )
        w = base_w.copy()
        if args.confidence_weight > 0.0:
            a = args.confidence_weight
            raw = (1.0 - a) + a * conf  # present-only; NaN where absent
            bands = np.digitize(obs_dtw, [2.0, 5.0, 10.0, 30.0])
            cw = np.ones(len(qn), "float64")  # null wells stay neutral 1.0
            for bi in np.unique(bands):
                m = (bands == bi) & present
                if m.any():
                    cw[m] = raw[m] / raw[m].mean()  # band mean -> 1.0
            w *= cw
        n_excluded = 0
        if args.min_confidence > 0.0:
            keep = present & (conf >= args.min_confidence)
            n_excluded = int((~keep).sum())
            w *= keep.astype("float64")
        sample_w = w.astype("float32")
        sample_w_t = torch.as_tensor(sample_w, dtype=torch.float32, device=device)
        conf_meta = {
            "confidence_table": args.confidence_table,
            "confidence_weight_alpha": args.confidence_weight,
            "min_confidence": args.min_confidence,
            "wells_with_confidence": int(present.sum()),
            "wells_null_confidence": n_null,
            "median_confidence_present": float(np.nanmedian(conf[present]))
            if present.any()
            else float("nan"),
            "wells_excluded_min_confidence": n_excluded,
        }
        log.info(
            "confidence weighting: alpha=%.2f min=%.2f | %d/%d wells have confidence "
            "(median %.2f), %d null->neutral, %d excluded by min-confidence; "
            "depth-stratified band-normalized",
            args.confidence_weight,
            args.min_confidence,
            int(present.sum()),
            len(qn),
            float(np.nanmedian(conf[present])) if present.any() else float("nan"),
            n_null,
            n_excluded,
        )

    # Water-row weight LAST: overrides the depth path (obs_dtw=0 would max-weight
    # them) and the confidence path (no confinement label -> --min-confidence would
    # zero them). At 0.0 the rows are loss-inert but still OOF-predicted.
    if water.any():
        w_np = (
            sample_w_t.detach().cpu().numpy().astype("float64")
            if sample_w_t is not None
            else np.ones(len(qn), "float64")
        )
        w_np[water] = args.water_label_weight
        sample_w_t = torch.as_tensor(
            w_np.astype("float32"), dtype=torch.float32, device=device
        )
        log.info(
            "water label weight: %d rows -> %.2f (post depth/confidence paths)",
            int(water.sum()),
            args.water_label_weight,
        )

    # E4 SWL aux weight LAST (mirrors the water path): aux ids carry no confinence
    # label, so --min-confidence must not zero them; obs_dtw>0 would mis-weight them
    # on the depth path. Loss-inert at 0.0 but still OOF-predicted.
    if args.swl_labels and swl.any():
        w_np = (
            sample_w_t.detach().cpu().numpy().astype("float64")
            if sample_w_t is not None
            else np.ones(len(qn), "float64")
        )
        w_np[swl] = args.swl_label_weight
        sample_w_t = torch.as_tensor(
            w_np.astype("float32"), dtype=torch.float32, device=device
        )
        log.info(
            "SWL aux label weight: %d rows -> %.2f (post depth/confidence/water paths)",
            int(swl.sum()),
            args.swl_label_weight,
        )

    # E6 shoreline ring weight LAST (mirrors the water/swl paths): shore ids carry no
    # confinement label so --min-confidence must not zero them; obs_dtw=0 would
    # max-weight them on the depth path. Loss-inert at 0.0 but still OOF-predicted.
    if shore.any():
        w_np = (
            sample_w_t.detach().cpu().numpy().astype("float64")
            if sample_w_t is not None
            else np.ones(len(qn), "float64")
        )
        w_np[shore] = args.shore_label_weight
        sample_w_t = torch.as_tensor(
            w_np.astype("float32"), dtype=torch.float32, device=device
        )
        log.info(
            "shore label weight: %d rows -> %.2f (post depth/confidence/water/swl paths)",
            int(shore.sum()),
            args.shore_label_weight,
        )

    f_ch, f_lat = ch_ea.shape[1], lat_ea.shape[1]
    f_ms = ms_ea.shape[1] if use_ms else None  # width incl. missingness flags
    f_pf = pf_ea.shape[1] if use_pf else None  # portfolio edge-attr width (6B)
    f_sc_node = sc_x_t.shape[1] if use_sc else None  # SC widths incl. missing flags
    f_sc_edge = sc_ea_t.shape[1] if use_sc else None
    native_oof = np.full(len(qn), np.nan)
    gate_oof = np.full(len(qn), np.nan) if fac_gate else None
    lambda_oof = np.full(len(qn), np.nan) if fac_lambda else None
    sigma_oof = np.full(len(qn), np.nan) if args.sigma_head else None
    # WP5 ordinal head: class labels from observed DTW (water pseudo-rows are DTW=0
    # shallow positives); thresholds in meters, ascending.
    ord_thresh, ordinal_oof, ordinal_y_t = None, None, None
    if args.ordinal_head:
        ord_thresh = np.array(
            [float(t) for t in args.ordinal_thresholds.split(",")], dtype="float64"
        )
        if not (np.diff(ord_thresh) > 0).all():
            raise SystemExit("--ordinal-thresholds must be strictly ascending")
        if not np.isfinite(obs_dtw).all():
            raise SystemExit(
                f"{int((~np.isfinite(obs_dtw)).sum())} non-finite obs_dtw rows; "
                "ordinal labels require finite observed DTW everywhere"
            )
        ordinal_oof = np.full((len(qn), len(ord_thresh)), np.nan)
        ordinal_y_t = torch.as_tensor(
            (obs_dtw[:, None] < ord_thresh[None, :]).astype("float32"), device=device
        )
    # WP2 two-surface: privileged P(phreatic) assignment priors, loss-only (never a
    # query feature). Unmatched wells fall back to 0.5, which the confidence-scaled
    # BCE makes exactly inert; water pseudo-rows are open-water surface expressions,
    # phreatic by construction.
    ts_prior_t = None
    ts_pi_oof = ts_p_oof = ts_r_oof = ts_bp_oof = ts_br_oof = None
    ts_prior_meta = None
    if two_surface:
        pri = pd.read_parquet(args.two_surface_priors)
        if args.assign_prior_col not in pri.columns:
            raise SystemExit(
                f"--assign-prior-col {args.assign_prior_col!r} not in "
                f"{args.two_surface_priors} (has: {sorted(pri.columns)})"
            )
        pri = pri.drop_duplicates("canonical_id")
        pv = pri[args.assign_prior_col].to_numpy("float64")
        if np.isfinite(pv).any() and (np.nanmin(pv) < 0.0 or np.nanmax(pv) > 1.0):
            raise SystemExit(
                f"{args.assign_prior_col} outside [0,1]; not a probability"
            )
        a_pri = (
            qn["canonical_id"]
            .map(pri.set_index("canonical_id")[args.assign_prior_col])
            .astype("float64")
            .to_numpy()
        )
        n_match = int((np.isfinite(a_pri) & real).sum())
        a_pri = np.where(np.isfinite(a_pri), a_pri, 0.5)
        a_pri[water] = 0.98
        a_pri[shore] = 0.98  # shoreline rings are phreatic (DTW=0) like water rows
        ts_prior_meta = {
            "priors_path": args.two_surface_priors,
            "prior_col": args.assign_prior_col,
            "assign_weight": args.assign_weight,
            "n_real_matched": n_match,
            "n_real": int(real.sum()),
            "n_water_at_098": int(water.sum()),
        }
        log.info(
            "--two-surface ON: assignment prior %s (%d/%d real wells matched, "
            "unmatched -> 0.5 neutral; %d water rows -> 0.98); lambda_assign=%.2f",
            args.assign_prior_col,
            n_match,
            int(real.sum()),
            int(water.sum()),
            args.assign_weight,
        )
        ts_prior_t = torch.as_tensor(
            np.clip(a_pri, 0.01, 0.99).astype("float32"), device=device
        )
        ts_pi_oof = np.full(len(qn), np.nan)
        ts_p_oof = np.full(len(qn), np.nan)
        ts_r_oof = np.full(len(qn), np.nan)
        ts_bp_oof = np.full(len(qn), np.nan)
        ts_br_oof = np.full(len(qn), np.nan)
    gate_experts = None
    if prior_gate:
        # expert order must match the model's mixture; the head is always LAST.
        gate_experts = (
            ["fac", "deep"]
            + (["mirror"] if args.mirror_anchor else [])
            + (["hang"] if args.hang_anchor else [])
            + ["head"]
        )
    prior_gate_oof = (
        np.full((len(qn), len(gate_experts)), np.nan) if prior_gate else None
    )
    learned_aquifer = use_aquifer and args.aquifer_route == "learned"
    aquifer_gate_oof = np.full(len(qn), np.nan) if learned_aquifer else None
    # per-edge portfolio attention, collected on each fold's test rows (6B OOF dump).
    pf_attn_rows: list[pd.DataFrame] = [] if use_pf else []
    # per-edge spatial-context attention (which ring/octant is read) -- the learned
    # direction x scale map, valuable regardless of the MAD outcome.
    sc_attn_rows: list[pd.DataFrame] = []
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

    # source-well assimilation (rungs 0/1): the eligible source pool = real wells with
    # a finite target (water/shore/swl pseudo-rows never donate an obs). The rung-0 src
    # block is [standardized obs value * valid, valid]; standardization is per-fold
    # (y_c/y_s) so it happens inside the fold loop, not here. Both rungs share the
    # masked-label protocol (drawn sources visible + loss-zeroed, per-epoch redraw).
    f_src = 2 if args.source_obs else None
    src_protocol = bool(args.source_obs or args.source_edges)
    src_eligible = None
    if src_protocol:
        src_eligible = real & np.isfinite(target)
        log.info(
            "source protocol ON (obs=%s edges=%s): %d eligible source wells, "
            "per-epoch mask frac ~U(%.2f, %.2f)",
            bool(args.source_obs),
            bool(args.source_edges),
            int(src_eligible.sum()),
            args.source_frac_min,
            args.source_frac_max,
        )

    # source edges (assimilation rung 1): direct spatial query<-source-well edges
    # ("learned IDW"). Like the analog block: edge INDEX + ATTRS are target-blind
    # constants standardized once; src_cv_fold carries the fold guard; the per-node
    # source VALUE is fold-standardized inside the loop. UNLIKE analog, visibility is
    # additionally filtered per forward by the masked-label protocol (train_fold).
    f_srcedge = None
    se_src_t = se_dst_t = se_ea_t = se_src_fold = None
    if args.source_edges:
        se = pd.read_parquet(args.source_edges)
        need = {"query_node_idx", "src_query_node_idx", "src_cv_fold"}
        if not need.issubset(se.columns):
            raise SystemExit(
                f"--source-edges {args.source_edges} missing {need - set(se.columns)}"
            )
        if not set(SOURCE_EDGE_COLS).issubset(se.columns):
            raise SystemExit(
                f"--source-edges {args.source_edges} missing edge-feature cols "
                f"{SOURCE_EDGE_COLS}"
            )
        se_dst = se["query_node_idx"].to_numpy("int64")
        se_src = se["src_query_node_idx"].to_numpy("int64")
        if (
            se_dst.max() >= len(qn)
            or se_src.max() >= len(qn)
            or min(se_dst.min(), se_src.min()) < 0
        ):
            raise SystemExit("--source-edges references out-of-range query_node_idx")
        if (se_dst == se_src).any():
            raise SystemExit("--source-edges has self-edges (source == dest)")
        if not src_eligible[se_src].all():
            raise SystemExit(
                f"{int((~src_eligible[se_src]).sum())} source edges source from "
                "ineligible nodes (pseudo or non-finite target); rebuild with "
                "build_source_edges.py"
            )
        se_ea_np = apply_stats(se, fit_stats(se, list(SOURCE_EDGE_COLS), None))
        if not np.isfinite(se_ea_np).all():
            raise SystemExit("--source-edges has non-finite edge features")
        se_src_t = torch.as_tensor(se_src, dtype=torch.long, device=device)
        se_dst_t = torch.as_tensor(se_dst, dtype=torch.long, device=device)
        se_ea_t = torch.as_tensor(se_ea_np, dtype=torch.float32, device=device)
        se_src_fold = se["src_cv_fold"].to_numpy("int64")
        f_srcedge = se_ea_t.shape[1]
        deg = pd.Series(se_dst).value_counts()
        log.info(
            "source edges ON: %s (%d edges, %d dest nodes, mean degree %.1f, "
            "%d edge features; protocol-masked per forward)",
            args.source_edges,
            len(se),
            deg.size,
            float(deg.mean()),
            f_srcedge,
        )

    def source_edge_ctx(fold: int | None) -> dict:
        """Fold-static source-edge tensors for src_ctx: edges whose source is NOT in
        the held-out fold (the analog leak guard); train_fold then narrows per
        forward to the visible-source protocol. Empty dict when edges are off."""
        if f_srcedge is None:
            return {}
        keep = torch.as_tensor(
            analog_fold_keep(se_src_fold, fold), dtype=torch.bool, device=device
        )
        return {
            "edge_src": se_src_t[keep],
            "edge_ei": torch.stack([se_src_t[keep], se_dst_t[keep]]),
            "edge_ea": se_ea_t[keep],
        }

    # All-wells query features for the memory probe (shapes match any fold).
    # Stats fit on real wells only (water pseudo-rows never shape a fit).
    probe_x = torch.as_tensor(
        apply_stats(qn, fit_stats(qn, query_cols, real)),
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
            f_pf=f_pf,
            f_sc_node=f_sc_node,
            f_sc_edge=f_sc_edge,
            f_mae=f_mae,
            f_analog=f_analog,
            f_src=f_src,
            f_srcedge=f_srcedge,
            writeback=args.query_writeback,
            pinball=args.pinball,
            fac_skip=fac_skip,
            fac_gate=fac_gate,
            fac_lambda=fac_lambda,
            sigma=args.sigma_head,
            ordinal=args.ordinal_head,
            two_surface=two_surface,
            prior_gate=prior_gate,
            mirror_anchor=args.mirror_anchor,
            hang_anchor=args.hang_anchor,
            directional_edges=args.directional_edges,
            **aquifer_kwargs,
        ).to(device)

    if device.startswith("cuda"):
        probe_feat = {**graph_tensors, "query_x": probe_x}
        yc0 = float(np.median(target))
        ys0 = float(1.4826 * np.median(np.abs(target - yc0)) or 1.0)
        probe_feat |= analog_feat(None, yc0, ys0)  # full edge set (sizing only)
        if f_src is not None:
            probe_feat["src_x"] = torch.zeros(
                (len(qn), f_src), dtype=torch.float32, device=device
            )
        if f_srcedge is not None:
            # full edge set (sizing only) + zero source values
            se_probe = source_edge_ctx(None)
            probe_feat["srcedge_ei"] = se_probe["edge_ei"]
            probe_feat["srcedge_ea"] = se_probe["edge_ea"]
            probe_feat["srcedge_val"] = torch.zeros(
                len(qn), dtype=torch.float32, device=device
            )
        if use_fac_anchor:
            if fac_pred_dtw is not None:
                pc0 = float(np.nanmedian(fac_pred_dtw))
                ps0 = float(1.4826 * np.nanmedian(np.abs(fac_pred_dtw - pc0)) or 1.0)
                probe_feat |= _fac_feat(
                    fac_raw, fac_present, yc0, ys0, device, fac_pred_dtw, pc0, ps0
                )
            else:
                probe_feat |= _fac_feat(fac_raw, fac_present, yc0, ys0, device)
        if prior_gate or two_surface:
            dc0 = float(np.nanmedian(deep_pred_dtw))
            ds0 = float(1.4826 * np.nanmedian(np.abs(deep_pred_dtw - dc0)) or 1.0)
            probe_feat |= _fac_feat(
                deep_raw,
                deep_present,
                yc0,
                ys0,
                device,
                deep_pred_dtw,
                dc0,
                ds0,
                prefix="deep",
            )
        if args.mirror_anchor or two_surface:
            # mirror_pred_dtw is the constant d -> its standardized signal is 0
            # everywhere (centered on itself); the informative tensor is mirror_base.
            probe_feat |= _fac_feat(
                mirror_raw,
                mirror_present,
                yc0,
                ys0,
                device,
                mirror_pred_dtw,
                float(args.mirror_depth_m),
                1.0,
                prefix="mirror",
            )
        if args.hang_anchor:
            hc0 = float(np.nanmedian(hang_pred_dtw))
            hs0 = float(1.4826 * np.nanmedian(np.abs(hang_pred_dtw - hc0)) or 1.0)
            probe_feat |= _fac_feat(
                hang_raw,
                hang_present,
                yc0,
                ys0,
                device,
                hang_pred_dtw,
                hc0,
                hs0,
                prefix="hang",
            )
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
        if (water.any() or shore.any()) and len(all_pairs):
            # pair loss compares label differences at full weight -- real wells only
            # (real excludes water + shore + swl pseudo-rows).
            all_pairs = all_pairs[real[all_pairs[:, 0]] & real[all_pairs[:, 1]]]
        log.info(
            "anti-compression pair loss ON: lambda=%.2f radius=%.0fm k=%d -> "
            "%d global nearby-well pairs",
            pair_w,
            args.pair_radius_m,
            args.pair_k,
            len(all_pairs),
        )

    # GLR: base weight vector (post depth/confidence/water/swl/shore paths) that the
    # per-fold shore mask overrides. Shore rows sit at --shore-label-weight here (0.0);
    # in fold f the passing shore rows are lifted to --glr-label-weight below.
    glr_base_w = None
    if glr_active:
        glr_base_w = (
            sample_w_t.detach().cpu().numpy().astype("float32")
            if sample_w_t is not None
            else np.ones(len(qn), "float32")
        )

    for f in folds:
        test = qn[fold_col].to_numpy() == f
        trainval = ~test
        va = val_blocks(trainval, blocks, args.val_frac, rng)
        va &= real  # early stop on wells only; water val-block rows fall to train
        tr = trainval & ~va
        trr = tr & real  # standardization/anchor fits on real wells only
        # GLR: this fold's shore-label weight vector -- passing shore rows (cross-fit on
        # cv_fold!=f training wells) get --glr-label-weight, the rest stay 0. Shore rows
        # whose own cv_fold==f are already held out of `tr` (in `test`), so the weight
        # only bites the shore rows fold f actually trains on.
        sample_w_fold_t = sample_w_t
        if glr_active:
            w_fold = glr_base_w.copy()
            w_fold[glr_pass_by_fold[int(f)]] = args.glr_label_weight
            sample_w_fold_t = torch.as_tensor(
                w_fold, dtype=torch.float32, device=device
            )
        pair_idx = None
        if all_pairs is not None and len(all_pairs):
            both_tr = tr[all_pairs[:, 0]] & tr[all_pairs[:, 1]]
            log.info("fold %d: %d train-train pairs", f, int(both_tr.sum()))
            if both_tr.any():
                pair_idx = torch.as_tensor(
                    all_pairs[both_tr].T, dtype=torch.long, device=device
                )
        q_stats = fit_stats(qn, query_cols, trr)
        query_x = torch.as_tensor(
            apply_stats(qn, q_stats), dtype=torch.float32, device=device
        )
        y_c = float(np.median(target[trr]))
        y_s = float(1.4826 * np.median(np.abs(target[trr] - y_c)) or 1.0)
        y_std = torch.as_tensor(
            (target - y_c) / y_s, dtype=torch.float32, device=device
        )
        feat = {**graph_tensors, "query_x": query_x}
        # analog edges with source in the held-out fold f are dropped (leak-free), and
        # the source value is standardized in this fold's target space.
        feat |= analog_feat(f, y_c, y_s)
        # source-obs (rung 0): this fold's assimilation machinery -- obs values
        # standardized in THIS fold's target space (the target itself), the train /
        # train+val eligible pools, and a fold-seeded rng for per-epoch source redraws.
        # feat["src_x"] itself is set (and re-set each epoch) inside train_fold.
        src_ctx = None
        if src_protocol:
            src_ctx = {
                "val_std": torch.as_tensor(
                    np.nan_to_num((target - y_c) / y_s, nan=0.0),
                    dtype=torch.float32,
                    device=device,
                ),
                "has_x": bool(args.source_obs),
                "val_mix": bool(args.source_val_mix),
                "tr_pool": tr & src_eligible,
                "trva_pool": (tr | va) & src_eligible,
                "frac": (args.source_frac_min, args.source_frac_max),
                "rng": np.random.default_rng(10_000 * (args.seed + 1) + int(f)),
            }
            # rung 1: this fold's static edge tensors (source not in fold f);
            # train_fold narrows them per forward to the visible-source protocol.
            src_ctx |= source_edge_ctx(int(f))
        # anchor pred-dtw standardization stats, hoisted so --save-models can persist
        # them for any flag combination (None where the anchor is off).
        pc = ps = dc = ds = hc = hs = None
        if use_fac_anchor:
            if fac_pred_dtw is not None:
                trp = trr & fac_present
                pc = float(np.median(fac_pred_dtw[trp]))
                ps = float(1.4826 * np.median(np.abs(fac_pred_dtw[trp] - pc)) or 1.0)
                feat |= _fac_feat(
                    fac_raw, fac_present, y_c, y_s, device, fac_pred_dtw, pc, ps
                )
            else:
                feat |= _fac_feat(fac_raw, fac_present, y_c, y_s, device)
        if prior_gate or two_surface:
            trd = trr & deep_present
            dc = float(np.median(deep_pred_dtw[trd]))
            ds = float(1.4826 * np.median(np.abs(deep_pred_dtw[trd] - dc)) or 1.0)
            feat |= _fac_feat(
                deep_raw,
                deep_present,
                y_c,
                y_s,
                device,
                deep_pred_dtw,
                dc,
                ds,
                prefix="deep",
            )
        if args.mirror_anchor or two_surface:
            feat |= _fac_feat(
                mirror_raw,
                mirror_present,
                y_c,
                y_s,
                device,
                mirror_pred_dtw,
                float(args.mirror_depth_m),
                1.0,
                prefix="mirror",
            )
        if args.hang_anchor:
            trh = trr & hang_present
            hc = float(np.median(hang_pred_dtw[trh]))
            hs = float(1.4826 * np.median(np.abs(hang_pred_dtw[trh] - hc)) or 1.0)
            feat |= _fac_feat(
                hang_raw,
                hang_present,
                y_c,
                y_s,
                device,
                hang_pred_dtw,
                hc,
                hs,
                prefix="hang",
            )
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
            f_pf=f_pf,
            f_sc_node=f_sc_node,
            f_sc_edge=f_sc_edge,
            f_mae=f_mae,
            f_analog=f_analog,
            f_src=f_src,
            f_srcedge=f_srcedge,
            writeback=args.query_writeback,
            pinball=args.pinball,
            fac_skip=fac_skip,
            fac_gate=fac_gate,
            fac_lambda=fac_lambda,
            sigma=args.sigma_head,
            ordinal=args.ordinal_head,
            two_surface=two_surface,
            prior_gate=prior_gate,
            mirror_anchor=args.mirror_anchor,
            hang_anchor=args.hang_anchor,
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
            sample_w_fold_t,
            pair_idx=pair_idx,
            pair_w=pair_w,
            ordinal_y_t=ordinal_y_t,
            ts_prior_t=ts_prior_t,
            src_ctx=src_ctx,
        )
        if fac_gate and model.last_fac_gate is not None:
            # last_fac_gate is from train_fold's final full-batch forward (all queries).
            gate_oof[test] = model.last_fac_gate.cpu().numpy().reshape(-1)[test]
        if fac_lambda and model.last_fac_lambda is not None:
            # per-well convex mixing weight from the final full-batch forward.
            lambda_oof[test] = model.last_fac_lambda.cpu().numpy().reshape(-1)[test]
        if args.sigma_head and model.sigma_log_b is not None:
            # Laplace scale b in METERS (de-standardized by this fold's y_s).
            sigma_oof[test] = (
                np.exp(model.sigma_log_b.detach().cpu().numpy().reshape(-1)[test]) * y_s
            )
        if args.ordinal_head and model.ordinal_logits is not None:
            # nested shallow-class probabilities from the final full-batch forward.
            ordinal_oof[test] = (
                torch.sigmoid(model.ordinal_logits.detach()).cpu().numpy()[test]
            )
        if prior_gate and model.last_prior_gate is not None:
            # per-well softmax weights (fac/deep/head) from the final full-batch forward.
            prior_gate_oof[test] = model.last_prior_gate.cpu().numpy()[test]
        if two_surface and model.ts_out is not None:
            # mixture internals from the final full-batch forward, de-standardized:
            # component means in native target units (m), Laplace scales in meters.
            ts = {
                k: v.detach().cpu().numpy().reshape(-1) for k, v in model.ts_out.items()
            }
            ts_pi_oof[test] = 1.0 / (1.0 + np.exp(-ts["m"][test]))
            ts_p_oof[test] = ts["hp"][test] * y_s + y_c
            ts_r_oof[test] = ts["hr"][test] * y_s + y_c
            ts_bp_oof[test] = np.exp(ts["lbp"][test]) * y_s
            ts_br_oof[test] = np.exp(ts["lbr"][test]) * y_s
        if learned_aquifer and model.last_aquifer_gate is not None:
            # per-query sigmoid gate from the final full-batch forward (all queries).
            aquifer_gate_oof[test] = (
                model.last_aquifer_gate.cpu().numpy().reshape(-1)[test]
            )
        if use_pf and model.pf_read.last_attn is not None:
            # per-EDGE attention from the final full-batch forward, row-aligned with pf; keep
            # only this fold's TEST-query edges for the OOF diagnostic (which site is read?).
            attn = model.pf_read.last_attn.cpu().numpy().reshape(-1)
            q_of_edge = pf["query_node_idx"].to_numpy("int64")
            in_test = test[q_of_edge]
            pf_attn_rows.append(
                pd.DataFrame(
                    {
                        "query_node_idx": q_of_edge[in_test],
                        "site_type": pf["site_type"].to_numpy()[in_test],
                        "attn": attn[in_test],
                        "fold": int(f),
                    }
                )
            )
        if use_sc and model.sc_read.last_attn is not None:
            # per-edge SC attention from the final full-batch forward, row-aligned with
            # sce; TEST-query edges only (OOF). ring/octant are the nominal diagnostics
            # carried by the builder -- this is the learned direction/scale map.
            sc_attn = model.sc_read.last_attn.cpu().numpy().reshape(-1)
            q_of_sc = sce["query_node_idx"].to_numpy("int64")
            in_test_sc = test[q_of_sc]
            sc_attn_rows.append(
                pd.DataFrame(
                    {
                        "query_node_idx": q_of_sc[in_test_sc],
                        "ring": sce["ring"].to_numpy("int64")[in_test_sc],
                        "octant": sce["octant"].to_numpy("int64")[in_test_sc],
                        "attn": sc_attn[in_test_sc],
                        "fold": int(f),
                    }
                )
            )
        if args.save_models:
            # Per-fold checkpoint = weights + the FULL standardization contract this
            # fold's forward depends on (query scaler, target center/scale, anchor
            # pred-dtw stats). The inference runner replays _fac_feat/apply_stats
            # from these verbatim -- train/infer skew is a schema error, never drift.
            mdir = out_dir / "models"
            mdir.mkdir(exist_ok=True)
            torch.save(
                {
                    "fold": int(f),
                    "state_dict": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "y_c": y_c,
                    "y_s": y_s,
                    "q_stats": q_stats,
                    "fac_stats": {"pc": pc, "ps": ps},
                    "deep_stats": {"dc": dc, "ds": ds},
                    "mirror_stats": (
                        {"c": float(args.mirror_depth_m), "s": 1.0}
                        if (args.mirror_anchor or two_surface)
                        else None
                    ),
                    "hang_stats": ({"hc": hc, "hs": hs} if args.hang_anchor else None),
                },
                mdir / f"fold_{int(f)}.pt",
            )
        del model, query_x, y_std, feat
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        native_oof[test] = native[test]
        pred_dtw = _native_to_dtw(native, base, target_mode)
        ter = test & real  # headline test MAD on wells only (water rows excluded)
        log.info(
            "fold %d: train=%d val=%d test=%d | val DTW-MAD=%.3f @%d | test=%.3f",
            f,
            tr.sum(),
            va.sum(),
            test.sum(),
            best_mad,
            best_epoch,
            float(np.nanmedian(np.abs(pred_dtw[ter] - obs_dtw[ter]))),
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

    if args.save_models:
        # Fold-independent pieces: the reach/edge scalers (fit on the PRUNED training
        # frames, T-side; inference applies them to its own pruned subgraph -- same
        # physical quantities, same z-scoring) + a self-contained manifest so the
        # runner reconstructs the exact WTEGraphNet ctor and feature schema without
        # re-deriving anything from hyperparams.
        mdir = out_dir / "models"
        torch.save(
            {"reach_stats": reach_stats, "ch_stats": ch_stats, "lat_stats": lat_stats},
            mdir / "shared_stats.pt",
        )
        (mdir / "inference_manifest.json").write_text(
            json.dumps(
                {
                    "graph_dir": str(gdir),
                    "target_mode": target_mode,
                    "dtw_base_col": base_col,
                    "surface_elev_col": surface_col,
                    "obs_wte_col": obs_wte_col,
                    "effective_hidden": int(hidden),
                    "channel_layers": int(args.channel_layers),
                    "dropout": float(args.dropout),
                    "seed": int(args.seed),
                    "folds": [int(f) for f in folds],
                    "flags": {
                        "prior_gate": bool(prior_gate),
                        "mirror_anchor": bool(args.mirror_anchor),
                        "mirror_depth_m": float(args.mirror_depth_m),
                        "hang_anchor": bool(args.hang_anchor),
                        "sigma_head": bool(args.sigma_head),
                        "fac_skip": bool(fac_skip),
                        "fac_gate": bool(fac_gate),
                        "fac_lambda": bool(fac_lambda),
                        "directional_edges": bool(args.directional_edges),
                        "query_writeback": bool(args.query_writeback),
                        "source_obs": bool(args.source_obs),
                        "f_src": int(f_src) if f_src is not None else None,
                        "source_edges": str(args.source_edges)
                        if args.source_edges
                        else None,
                        "f_srcedge": int(f_srcedge) if f_srcedge is not None else None,
                        "mainstem_read": bool(use_ms),
                        "portfolio_read": bool(use_pf),
                        "spatial_context": bool(use_sc),
                        "sc_no_azimuth": bool(args.sc_no_azimuth),
                        "anchors": bool(use_anchors),
                        "aquifer": bool(use_aquifer),
                        "pinball": bool(args.pinball),
                        "water_features": bool(
                            (man.get("water") or {}).get("features_enabled")
                        ),
                        "water_label_weight": float(args.water_label_weight),
                        "water_pseudo_rows": int(water.sum()),
                        "shore_label_weight": float(args.shore_label_weight),
                        "shore_pseudo_rows": int(shore.sum()),
                    },
                    "gate_experts": gate_experts,
                    "fac_anchor_col": fac_base_col if use_fac_anchor else None,
                    "deep_anchor_col": deep_anchor_col,
                    "hang_dtw_col": HANG_DTW_COL if args.hang_anchor else None,
                    "feature_dims": {
                        "reach": int(reach_x.shape[1]),
                        "query": int(f_query),
                        "channel_edge": int(f_ch),
                        "lateral_edge": int(f_lat),
                    },
                    "query_feature_cols": query_cols,
                    "reach_feature_cols": reach_cols,
                    "channel_edge_feature_cols": ch_cols,
                    "lateral_edge_feature_cols": lat_cols,
                },
                indent=2,
            )
        )
        log.info("saved %d fold checkpoints + shared stats -> %s", len(folds), mdir)

    if fac_gate:
        # Verify the mechanism: the gate should RELEASE (c->0) for deep wells. Log the
        # OOF gate value by observed-depth band (deep release = the intended behavior).
        bands = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
        msg = "; ".join(
            f"{lo}-{hi if hi != np.inf else '+'}m c={np.nanmean(gate_oof[m]):.2f}(n={int(m.sum())})"
            for lo, hi in bands
            if (m := (obs_dtw >= lo) & (obs_dtw < hi) & fac_present & real).any()
        )
        log.info("fac-gate mean OOF gate by obs-depth: %s", msg)

    if fac_lambda:
        # Mechanism read: lambda should sit HIGH shallow (FAC's regime) and RELEASE
        # deep. The OOF lambda column is the shallow terrain-coupled-zone map.
        bands = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
        msg = "; ".join(
            f"{lo}-{hi if hi != np.inf else '+'}m "
            f"lam={np.nanmean(lambda_oof[m]):.2f}(n={int(m.sum())})"
            for lo, hi in bands
            if (m := (obs_dtw >= lo) & (obs_dtw < hi) & fac_present & real).any()
        )
        log.info("fac-lambda mean OOF blend by obs-depth: %s", msg)

    if args.sigma_head:
        # Mechanism read: sigma should GROW with depth (the unlearnable deep-regional
        # regime) -- that ordering is what makes selective shallow calls possible.
        bands = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
        msg = "; ".join(
            f"{lo}-{hi if hi != np.inf else '+'}m "
            f"sig={np.nanmedian(sigma_oof[m]):.2f}m(n={int(m.sum())})"
            for lo, hi in bands
            if (m := (obs_dtw >= lo) & (obs_dtw < hi) & real).any()
        )
        log.info("sigma-head median OOF sigma by obs-depth: %s", msg)

    if prior_gate:
        # Mechanism read: w_fac should dominate shallow (FAC's regime) and w_deep
        # should RISE with depth toward the deep-IDW expert (the only prior that wins
        # 30+m). A flat w_head~1 everywhere means the gate never engaged.
        bands = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
        msg = "; ".join(
            f"{lo}-{hi if hi != np.inf else '+'}m "
            + "/".join(
                f"{nm}={np.nanmean(prior_gate_oof[m, j]):.2f}"
                for j, nm in enumerate(gate_experts)
            )
            + f"(n={int(m.sum())})"
            for lo, hi in bands
            if (m := (obs_dtw >= lo) & (obs_dtw < hi) & real).any()
        )
        log.info(
            "prior-gate mean OOF weights (%s) by obs-depth: %s",
            "/".join(gate_experts),
            msg,
        )

    if two_surface:
        # Mechanism read: pi (P(phreatic)) should sit HIGH shallow and FALL with
        # depth toward the regional component; pi flat ~1 or ~0 everywhere is the
        # component-collapse failure mode the WP2 gate checks.
        bands = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
        msg = "; ".join(
            f"{lo}-{hi if hi != np.inf else '+'}m "
            f"pi={np.nanmean(ts_pi_oof[m]):.2f}/"
            f"bp={np.nanmedian(ts_bp_oof[m]):.2f}m/"
            f"br={np.nanmedian(ts_br_oof[m]):.2f}m(n={int(m.sum())})"
            for lo, hi in bands
            if (m := (obs_dtw >= lo) & (obs_dtw < hi) & real).any()
        )
        log.info(
            "two-surface mean OOF pi + median component scales by obs-depth: %s", msg
        )

    if learned_aquifer:
        # The aquifer correction should OPEN (gate->1) where the regional substrate
        # carries signal; log the OOF gate by observed-depth band as the first read on
        # whether it earns its place (deep bands are where the stream head is weakest).
        bands = [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
        msg = "; ".join(
            f"{lo}-{hi if hi != np.inf else '+'}m "
            f"g={np.nanmean(aquifer_gate_oof[m]):.3f}(n={int(m.sum())})"
            for lo, hi in bands
            if (m := (obs_dtw >= lo) & (obs_dtw < hi) & real).any()
        )
        log.info("aquifer-gate mean OOF gate by obs-depth: %s", msg)

    gnn_dtw = _native_to_dtw(native_oof, base, target_mode)
    water_panel = None
    if water.any():
        # The generalize-to-unseen-water readout: |pred DTW| on water rows in their
        # held-out folds (truth is 0 on permanent water). Gate weights show routing.
        wd = np.abs(gnn_dtw[water])
        water_panel = {
            "n": int(water.sum()),
            "abs_dtw_median_m": float(np.nanmedian(wd)),
            "abs_dtw_p90_m": float(np.nanpercentile(wd, 90)),
            "frac_within_0p5m": float(np.nanmean(wd <= 0.5)),
            "frac_within_1m": float(np.nanmean(wd <= 1.0)),
        }
        log.info(
            "water OOF panel: n=%d |DTW| median=%.2f m p90=%.2f m "
            "frac<=0.5m=%.2f frac<=1m=%.2f",
            water_panel["n"],
            water_panel["abs_dtw_median_m"],
            water_panel["abs_dtw_p90_m"],
            water_panel["frac_within_0p5m"],
            water_panel["frac_within_1m"],
        )
        if prior_gate:
            water_panel["gate_w_mean"] = {
                nm: float(np.nanmean(prior_gate_oof[water, j]))
                for j, nm in enumerate(gate_experts)
            }
            log.info(
                "water OOF gate weights (mean): %s",
                " ".join(f"{k}={v:.2f}" for k, v in water_panel["gate_w_mean"].items()),
            )
        if two_surface:
            # water rows should route phreatic; a low mean here is the collapse tell.
            water_panel["ts_pi_phreatic_mean"] = float(np.nanmean(ts_pi_oof[water]))
            log.info(
                "water OOF two-surface pi_phreatic mean=%.2f",
                water_panel["ts_pi_phreatic_mean"],
            )
    # E6 shore panel: the generalize-to-unseen-shoreline readout -- |pred DTW| on shore
    # rows in their held-out folds (truth is 0 at the shore). Mirrors the water panel.
    shore_panel = None
    if shore.any():
        sd = np.abs(gnn_dtw[shore])
        shore_panel = {
            "n": int(shore.sum()),
            "abs_dtw_median_m": float(np.nanmedian(sd)),
            "abs_dtw_p90_m": float(np.nanpercentile(sd, 90)),
            "frac_within_0p5m": float(np.nanmean(sd <= 0.5)),
            "frac_within_1m": float(np.nanmean(sd <= 1.0)),
        }
        log.info(
            "shore OOF panel: n=%d |DTW| median=%.2f m p90=%.2f m "
            "frac<=0.5m=%.2f frac<=1m=%.2f",
            shore_panel["n"],
            shore_panel["abs_dtw_median_m"],
            shore_panel["abs_dtw_p90_m"],
            shore_panel["frac_within_0p5m"],
            shore_panel["frac_within_1m"],
        )
        if prior_gate:
            shore_panel["gate_w_mean"] = {
                nm: float(np.nanmean(prior_gate_oof[shore, j]))
                for j, nm in enumerate(gate_experts)
            }
            log.info(
                "shore OOF gate weights (mean): %s",
                " ".join(f"{k}={v:.2f}" for k, v in shore_panel["gate_w_mean"].items()),
            )
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
        "is_water_pseudo": water,
        "is_shore_pseudo": shore,
        "regional_idw_dtw_oof_m": qn["regional_idw_dtw_oof_m"].to_numpy(),
        "regional_deep_idw_dtw_oof_m": qn["regional_deep_idw_dtw_oof_m"].to_numpy(),
        "janssen_dtw_m": qn["janssen_dtw"].to_numpy(),
        "hand_m": qn["hand_m"].to_numpy(),
        "gnn_dtw_m": gnn_dtw,
    }
    if args.swl_labels:
        # E4 diagnostic: aux rows never join the frozen panel (canonical_id absent),
        # but flag them so any downstream read can exclude them explicitly.
        out_cols["is_swl_aux"] = swl
    if fac_gate:
        out_cols["fac_gate_c"] = gate_oof  # learned anchor confidence (diagnostic)
    if fac_lambda:
        out_cols["fac_lambda"] = lambda_oof  # convex FAC blend weight (zone map)
    if args.sigma_head:
        out_cols["gnn_sigma_m"] = sigma_oof  # per-well Laplace scale (meters)
    if args.ordinal_head:
        # WP5 nested shallow-class probabilities P(DTW < t), dimensionless 0-1.
        for j, t in enumerate(ord_thresh):
            out_cols[f"p_dtw_lt_{t:g}m"] = ordinal_oof[:, j]
    if prior_gate:
        # regime-map deliverable: which expert carries each well OOF.
        for j, nm in enumerate(gate_experts):
            out_cols[f"gate_w_{nm}"] = prior_gate_oof[:, j]
    if two_surface:
        # WP2 deliverables: membership (dimensionless 0-1) + per-component means
        # (native target units, m) + per-component Laplace scales (m). In
        # wte_residual mode a component's DTW is z_surf - (R + ts_native_*_m).
        out_cols["ts_pi_phreatic"] = ts_pi_oof
        out_cols["ts_native_p_m"] = ts_p_oof
        out_cols["ts_native_r_m"] = ts_r_oof
        out_cols["ts_sigma_p_m"] = ts_bp_oof
        out_cols["ts_sigma_r_m"] = ts_br_oof
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

    if use_pf and pf_attn_rows:
        # OOF per-edge portfolio attention (which reference site each test well reads).
        attn_df = pd.concat(pf_attn_rows, ignore_index=True)
        attn_df.to_parquet(out_dir / "gnn_portfolio_attention.parquet")
        by_type = attn_df.groupby("site_type")["attn"].mean().to_dict()
        log.info(
            "portfolio attention (OOF mean by site): %s",
            " ".join(
                f"{t}={by_type.get(t, float('nan')):.3f}"
                for t in man["portfolio_read"]["site_types"]
            ),
        )

    if use_sc and sc_attn_rows:
        # OOF per-edge spatial-context attention: the tells are (a) whether attention
        # collapsed to one ring (-> per-ring pools fallback) and (b) any azimuthal
        # asymmetry at the 10 km ring (the one narrow deep hypothesis).
        sc_attn_df = pd.concat(sc_attn_rows, ignore_index=True)
        sc_attn_df.to_parquet(out_dir / "gnn_sc_attention.parquet")
        by_ring = sc_attn_df.groupby("ring")["attn"].mean()
        by_oct = sc_attn_df.groupby("octant")["attn"].mean()
        log.info(
            "sc attention (OOF mean): rings %s | octants %s",
            " ".join(f"r{i}={v:.4f}" for i, v in by_ring.items()),
            " ".join(f"o{i}={v:.4f}" for i, v in by_oct.items()),
        )

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
        "fac_lambda": {
            "enabled": bool(fac_lambda),
            "anchor_col": fac_base_col if fac_lambda else None,
            "wells_with_fac": int(fac_present.sum()) if fac_lambda else None,
            "mean_oof_lambda": float(np.nanmean(lambda_oof)) if fac_lambda else None,
        },
        "sigma_head": {
            "enabled": bool(args.sigma_head),
            "median_oof_sigma_m": float(np.nanmedian(sigma_oof))
            if args.sigma_head
            else None,
        },
        "ordinal_head": {
            "enabled": bool(args.ordinal_head),
            "thresholds_m": ord_thresh.tolist() if args.ordinal_head else None,
            "weight": float(args.ordinal_weight) if args.ordinal_head else None,
            "median_oof_p_by_threshold": (
                {
                    f"{t:g}m": float(np.nanmedian(ordinal_oof[real, j]))
                    for j, t in enumerate(ord_thresh)
                }
                if args.ordinal_head
                else None
            ),
        },
        "water": {
            "n_pseudo_rows": int(water.sum()),
            "label_weight": float(args.water_label_weight),
            "bundle_block": man.get("water"),
            "oof_panel": water_panel,
        },
        "shore": {
            "n_pseudo_rows": int(shore.sum()),
            "label_weight": float(args.shore_label_weight),
            "bundle_block": man.get("shore"),
            "oof_panel": shore_panel,
        },
        "glr": {
            "enabled": glr_active,
            "labels_path": args.glr_labels,
            "label_weight": float(args.glr_label_weight),
            "crossfit_note": (
                "per-fold shore-label inclusion mask; fold f uses only cv_fold!=f "
                "training wells (leave-one-fold-out, like R); glr_pass_full is "
                "diagnostic/deployment only and never weights a CV fold"
            )
            if glr_active
            else None,
            **(glr_meta or {}),
        },
        "prior_gate": {
            "enabled": bool(prior_gate),
            "fac_anchor_col": fac_base_col if prior_gate else None,
            "deep_anchor_col": deep_anchor_col if prior_gate else None,
            "wells_with_fac": int(fac_present.sum()) if prior_gate else None,
            "wells_with_deep": int(deep_present.sum()) if prior_gate else None,
            "experts": gate_experts,
            "mirror_anchor": bool(args.mirror_anchor),
            "mirror_depth_m": args.mirror_depth_m if args.mirror_anchor else None,
            "hang_anchor": bool(args.hang_anchor),
            "hang_dtw_col": HANG_DTW_COL if args.hang_anchor else None,
            "wells_with_hang": int(hang_present.sum()) if args.hang_anchor else None,
            "mean_oof_w": {
                nm: float(np.nanmean(prior_gate_oof[:, j]))
                for j, nm in enumerate(gate_experts)
            }
            if prior_gate
            else None,
        },
        "two_surface": {
            "enabled": bool(two_surface),
            "priors": ts_prior_meta,
            "fac_anchor_col": fac_base_col if two_surface else None,
            "deep_anchor_col": deep_anchor_col if two_surface else None,
            "mirror_depth_m": args.mirror_depth_m if two_surface else None,
            # anti-collapse diagnostics: pi is the OOF P(phreatic) on real wells;
            # a component used <5% globally fails the WP2 collapse gate.
            "mean_oof_pi_phreatic": (
                float(np.nanmean(ts_pi_oof[real])) if two_surface else None
            ),
            "mean_oof_pi_by_depth_band": (
                {
                    f"{lo:g}-{hi:g}m": float(
                        np.nanmean(ts_pi_oof[real & (obs_dtw >= lo) & (obs_dtw < hi)])
                    )
                    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 30), (30, np.inf)]
                }
                if two_surface
                else None
            ),
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
            "depth_weight_scale": args.depth_weight_scale
            if args.depth_weight_scale > 0.0
            else None,
        },
        "confidence_weighting": conf_meta,
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
        "portfolio_read": {
            "enabled": bool(use_pf),
            "n_edges": int(len(pf)) if use_pf else None,
            "site_types": man["portfolio_read"]["site_types"] if use_pf else None,
            "coverage_by_type": man["portfolio_read"]["coverage_by_type"]
            if use_pf
            else None,
            "pf_edge_feature_cols": pf_cols if use_pf else None,
            "f_pf": int(f_pf) if use_pf else None,
        },
        "spatial_context": {
            "enabled": bool(use_sc),
            "no_azimuth": bool(args.sc_no_azimuth) if use_sc else None,
            "n_nodes": int(len(scn)) if use_sc else None,
            "n_edges": int(len(sce)) if use_sc else None,
            "radii_m": man["spatial_context"]["radii_m"] if use_sc else None,
            "n_per_ring": man["spatial_context"]["n_per_ring"] if use_sc else None,
            "sc_node_feature_cols": sc_node_cols if use_sc else None,
            "sc_edge_feature_cols": sc_edge_cols if use_sc else None,
            "f_sc_node": int(f_sc_node) if use_sc else None,
            "f_sc_edge": int(f_sc_edge) if use_sc else None,
        },
        "query_writeback": {"enabled": bool(args.query_writeback)},
        "source_obs": {
            "enabled": bool(args.source_obs),
            "f_src": int(f_src) if f_src is not None else None,
            "frac_min": float(args.source_frac_min) if src_protocol else None,
            "frac_max": float(args.source_frac_max) if src_protocol else None,
            "n_eligible": int(src_eligible.sum()) if src_protocol else None,
            # drawn sources are loss-zeroed only when src_x carries their own obs
            # (rung 0); edges-only arms keep full label mass (rung 1b).
            "loss_masked": bool(args.source_obs) if src_protocol else None,
            "val_mix": bool(args.source_val_mix) if src_protocol else None,
        },
        "source_edges": {
            "enabled": bool(args.source_edges),
            "path": str(args.source_edges) if args.source_edges else None,
            "f_srcedge": int(f_srcedge) if f_srcedge is not None else None,
            "edge_cols": list(SOURCE_EDGE_COLS) if args.source_edges else None,
        },
        "mae_embeddings": {
            "enabled": bool(args.mae_embeddings),
            "path": args.mae_embeddings if args.mae_embeddings else None,
            "f_mae": int(f_mae) if f_mae is not None else None,
        },
        "extra_query_features": {
            "enabled": bool(args.extra_query_features),
            "path": args.extra_query_features if args.extra_query_features else None,
            "cols": extra_query_cols or None,
        },
        "analog_edges": {
            "enabled": bool(args.analog_edges),
            "path": args.analog_edges if args.analog_edges else None,
            "f_analog": int(f_analog) if f_analog is not None else None,
            "n_edges": int(analog_src_t.numel()) if f_analog is not None else None,
            "fold_masked": bool(args.analog_edges),
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
