"""Edge-gated GNN over the FAC flow graph: predict residual WTE -> hybrid DTW.

Loads the framework-agnostic hetero-graph bundle from
``build_wte_graph_inputs.py`` and trains an edge-gated message-passing network:

  reach nodes  --channel edges (reach->reach, +1 down / -1 reverse)-->  reach
  reach nodes  --lateral edges (reach->query, k-nearest controlling)-->  query

Message passing is edge-gated: ``msg_ij = sigmoid(gate(x_i, x_j, e_ij)) *
transform(x_j, e_ij)``. The learned lateral gate is exported as a QA layer (how
much each query trusts its controlling reach).

Target is the residual WTE over the cross-fit regional prior
(``target_residual_wte_m = obs_wte - regional_wte_oof``); the final prediction is
``hybrid_dtw = dem - (regional_wte_oof + residual_hat)``. This is the same
target/regional base the tabular fusion uses, so the GNN OOF is directly
comparable to Fusion (the bar it must clear).

Leak-free inductive protocol:
  * Reaches carry NO labels; there are NO query->query edges, so a held-out
    query's label never reaches a training query's computation.
  * Train loss is taken on train-fold queries only; OOF prediction per fold uses
    the SAME ``cv_fold`` blocking as the fusion model (apples-to-apples OOF).
  * Query feature scaler + target scaler are fit on TRAIN rows only.
  * Validation (early-stop) is on held-out train BLOCKS, scored by DTW-MAD on our
    own GWX wells -- never against Ma.

    uv run python utils/train_wte_gnn.py \\
        --graph-dir .../hybrid/gwx/graph \\
        --out-dir   .../hybrid/gwx/gnn
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax

log = logging.getLogger("train_wte_gnn")


# ---------------------------------------------------------------------------
# Feature standardization (median-impute + z-score + missingness flags)
# ---------------------------------------------------------------------------
def fit_stats(df: pd.DataFrame, cols: list[str], mask: np.ndarray | None) -> dict:
    """Fit median/mean/std on ``mask`` rows; flag cols that are ever NaN."""
    rows = mask if mask is not None else np.ones(len(df), bool)
    sub = df.loc[rows, cols].astype("float64")
    med = sub.median()
    filled = sub.fillna(med)
    mean = filled.mean()
    std = filled.std(ddof=0).replace(0.0, 1.0)
    # A feature that is all-NaN on the fit rows (e.g. a sparse WTE/FAC-rem column
    # absent from one fold's TRAIN split) leaves med/mean/std NaN. Two failure
    # modes follow if left alone: (1) NaN propagates through apply_stats and
    # poisons the tensor; (2) held-out rows that DO have a finite value sail
    # through as raw, unscaled magnitudes (since med=0/mean=0/std=1 is identity)
    # while the model's first-layer weights for that column saw only zeros during
    # training -- so the OOF prediction rides on random init. Record these columns
    # and force their whole z-block to 0 in apply_stats; only the missingness flag
    # (nan_cols, whole-df below -- always includes these) carries the absence.
    all_nan_fit_cols = [c for c in cols if sub[c].isna().all()]
    med = med.fillna(0.0)
    mean = mean.fillna(0.0)
    std = std.fillna(1.0)
    nan_cols = [c for c in cols if df[c].isna().any()]
    return {
        "cols": cols,
        "med": med,
        "mean": mean,
        "std": std,
        "nan_cols": nan_cols,
        "all_nan_fit_cols": all_nan_fit_cols,
    }


def apply_stats(df: pd.DataFrame, stats: dict) -> np.ndarray:
    """Z-scored feature block with missingness-indicator columns appended."""
    cols = stats["cols"]
    sub = df[cols].astype("float64")
    miss = sub.isna()
    z = (sub.fillna(stats["med"]) - stats["mean"]) / stats["std"]
    # Columns with no finite value on the fit rows got no training signal; zero
    # them for ALL rows so held-out finite magnitudes can't ride random-init
    # weights.
    if stats.get("all_nan_fit_cols"):
        z[stats["all_nan_fit_cols"]] = 0.0
    parts = [z.to_numpy("float64")]
    if stats["nan_cols"]:
        ind = miss[stats["nan_cols"]].astype("float64")
        # An all-NaN-on-fit column was missing on every train row, so its
        # indicator is a constant 1 during training -- the weight is degenerate
        # with the bias. Held-out rows can be finite (indicator 0), which would
        # have the model extrapolate that unidentifiable weight to an unseen
        # value. Pin the indicator to its train-observed constant (1) for ALL
        # rows so the column is a harmless constant, never a held-out surprise.
        pin = [c for c in stats.get("all_nan_fit_cols", []) if c in ind.columns]
        if pin:
            ind[pin] = 1.0
        parts.append(ind.to_numpy())
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Edge-gated message passing
# ---------------------------------------------------------------------------
class EdgeGatedConv(MessagePassing):
    """msg_ij = sigmoid(gate(x_i, x_j, e_ij)) * transform(x_j, e_ij); update(x_i, agg).

    ``directional``: give the two flow senses their OWN message+gate weights instead
    of a single transform that must disentangle one concatenated ``direction`` feature
    (the "hope the gate learns it" regime that under-delivered). Each edge carries a
    ``dir_sign`` (+1 forward / -1 reverse); forward edges use msg_mlp/gate_mlp, reverse
    edges use msg_mlp_rev/gate_mlp_rev. Non-directional construction is byte-identical
    to before (no rev weights, dir_sign ignored), so every existing path is untouched.
    """

    def __init__(
        self,
        in_src: int,
        in_dst: int,
        edge_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        aggr: str = "mean",
        directional: bool = False,
    ) -> None:
        super().__init__(aggr=aggr, flow="source_to_target")
        self.directional = directional
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_src + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_src + in_dst + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        if directional:
            # reverse-sense edges (upstream-of / well-below-reach) get a dedicated
            # parameter path; the aggregation + update stay shared (direction shapes
            # only what message each neighbor sends, not how they are combined).
            self.msg_mlp_rev = nn.Sequential(
                nn.Linear(in_src + edge_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            self.gate_mlp_rev = nn.Sequential(
                nn.Linear(in_src + in_dst + edge_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, 1),
            )
        self.upd_mlp = nn.Sequential(
            nn.Linear(in_dst + out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.last_gate: torch.Tensor | None = None

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        dir_sign: torch.Tensor | None = None,
    ) -> torch.Tensor:
        agg = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            edge_attr=edge_attr,
            dir_sign=dir_sign,
            size=(x_src.size(0), x_dst.size(0)),
        )
        return self.upd_mlp(torch.cat([x_dst, agg], dim=-1))

    def message(
        self,
        x_j: torch.Tensor,
        x_i: torch.Tensor,
        edge_attr: torch.Tensor,
        dir_sign: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gin = torch.cat([x_i, x_j, edge_attr], dim=-1)
        min_ = torch.cat([x_j, edge_attr], dim=-1)
        if self.directional and dir_sign is not None:
            fwd = (dir_sign > 0).view(-1, 1).to(min_.dtype)
            g = fwd * torch.sigmoid(self.gate_mlp(gin)) + (1.0 - fwd) * torch.sigmoid(
                self.gate_mlp_rev(gin)
            )
            m = fwd * self.msg_mlp(min_) + (1.0 - fwd) * self.msg_mlp_rev(min_)
        else:
            g = torch.sigmoid(self.gate_mlp(gin))
            m = self.msg_mlp(min_)
        self.last_gate = g.detach()
        return g * m


class PortfolioReadConv(MessagePassing):
    """Segment-softmax ATTENTION read over a query's <=4 typed reference-site edges (6B).

    Unlike ``EdgeGatedConv`` (independent sigmoid gates over HOMOGENEOUS lateral senders),
    the portfolio senders are HETEROGENEOUS reference reaches (ds_datum / up_head / wet /
    ho_any) COMPETING for one query's read budget -- exactly where softmax attention earns
    its complexity: ``alpha = softmax(score([x_i, x_j, e_ij]), dst)`` normalizes over each
    query's incoming edges, and the aggregate is ``sum_e alpha_e * msg([x_j, e_ij])`` (a
    convex combination). A query with NO portfolio edge aggregates to zero (finite -- the
    ``portfolio_missing_*`` query features carry the absence signal to the head). Same
    ``flow="source_to_target"`` + ``upd_mlp`` shape as ``EdgeGatedConv``; captures
    ``last_attn`` (per-edge alpha) exactly as ``EdgeGatedConv`` captures ``last_gate``.
    """

    def __init__(
        self,
        in_src: int,
        in_dst: int,
        edge_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(aggr="add", flow="source_to_target")
        self.score_mlp = nn.Sequential(
            nn.Linear(in_src + in_dst + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_src + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(in_dst + out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.last_attn: torch.Tensor | None = None

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        agg = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            edge_attr=edge_attr,
            size=(x_src.size(0), x_dst.size(0)),
        )
        return self.upd_mlp(torch.cat([x_dst, agg], dim=-1))

    def message(
        self,
        x_j: torch.Tensor,
        x_i: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor | None,
        size_i: int | None,
    ) -> torch.Tensor:
        score = self.score_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))  # (E, 1)
        alpha = pyg_softmax(
            score, index, ptr, size_i
        )  # segment softmax over dst (query)
        self.last_attn = alpha.detach()
        m = self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))
        return alpha * m


class WTEGraphNet(nn.Module):
    """Edge-gated relaxation over the flow network.

    v1 (no anchors): channel diffusion among reaches, then a lateral read to each
    query. v2 (anchors, CONUS_GNN_V2_PLAN.md): springs/water/wetlands are Dirichlet/
    soft boundary conditions injected to reaches BEFORE the channel layers and
    RE-ASSERTED each layer (a weight-shared ``anchor_to_reach`` conv = holding the BC
    value while the interior relaxes), plus a direct ``anchor_to_query`` BC to off-
    network wells. Anchor support and the pinball auxiliary head are optional so the
    FAC-graph trainer (``train_wte_gnn.main``) keeps its v1 construction unchanged.
    """

    def __init__(
        self,
        f_reach: int,
        f_query: int,
        f_ch: int,
        f_lat: int,
        hidden: int,
        n_channel_layers: int,
        dropout: float,
        *,
        f_anchor: int | None = None,
        f_anchor_reach: int | None = None,
        f_anchor_query: int | None = None,
        f_ms: int | None = None,
        f_pf: int | None = None,
        f_sc_node: int | None = None,
        f_sc_edge: int | None = None,
        f_mae: int | None = None,
        f_analog: int | None = None,
        f_src: int | None = None,
        f_srcedge: int | None = None,
        srcedge_gated: bool = False,
        writeback: bool = False,
        pinball: bool = False,
        fac_skip: bool = False,
        fac_gate: bool = False,
        fac_lambda: bool = False,
        sigma: bool = False,
        ordinal: bool = False,
        n_ordinal: int = 3,
        two_surface: bool = False,
        prior_gate: bool = False,
        mirror_anchor: bool = False,
        hang_anchor: bool = False,
        directional_edges: bool = False,
        f_aquifer: int | None = None,
        f_aquifer_edge: int | None = None,
        f_aquifer_query: int | None = None,
        n_aquifer_layers: int = 0,
        aquifer_route: str = "off",
        aquifer_gate_init: float = -6.0,
        aquifer_delta_init_zero: bool = True,
    ) -> None:
        super().__init__()
        if fac_gate and not fac_skip:
            raise ValueError("fac_gate requires fac_skip")
        if fac_lambda and (fac_skip or fac_gate):
            # the convex blend REPLACES the additive anchor; running both would let
            # FAC enter the output twice and make lambda uninterpretable.
            raise ValueError("fac_lambda is exclusive with fac_skip/fac_gate")
        if fac_lambda and pinball:
            raise ValueError("fac_lambda with the pinball head is not supported")
        if sigma and pinball:
            raise ValueError("sigma head with the pinball head is not supported")
        if prior_gate and (fac_skip or fac_gate or fac_lambda):
            # the K-way mixture REPLACES the 2-way blend / additive anchor; combining
            # them would let FAC enter the output twice.
            raise ValueError(
                "prior_gate is exclusive with fac_skip/fac_gate/fac_lambda"
            )
        if prior_gate and pinball:
            raise ValueError("prior_gate with the pinball head is not supported")
        if two_surface and (prior_gate or fac_skip or fac_gate or fac_lambda):
            # the two-surface mixture is its own output structure over the anchors;
            # combining with another anchor pathway would double-count the priors.
            raise ValueError(
                "two_surface is exclusive with prior_gate/fac_skip/fac_gate/fac_lambda"
            )
        if two_surface and pinball:
            raise ValueError("two_surface with the pinball head is not supported")
        if two_surface and sigma:
            # the mixture already carries per-component Laplace scales.
            raise ValueError("two_surface is exclusive with the sigma head")
        if mirror_anchor and not prior_gate:
            # the mirror is an EXPERT of the gate, not a standalone anchor pathway.
            raise ValueError("mirror_anchor requires prior_gate")
        if hang_anchor and not prior_gate:
            # the Dupuit hang surface is likewise an EXPERT of the gate only.
            raise ValueError("hang_anchor requires prior_gate")
        self.has_anchor = f_anchor is not None
        self.has_ms = f_ms is not None
        self.has_pf = f_pf is not None
        if (f_sc_node is None) != (f_sc_edge is None):
            raise ValueError("f_sc_node and f_sc_edge must be set together")
        self.has_sc = f_sc_node is not None
        self.has_mae = f_mae is not None
        self.has_analog = f_analog is not None
        self.writeback = writeback
        # source-obs (assimilation rung 0): extra per-query obs feature block
        # [standardized obs value * valid, valid, ...] concatenated onto query_x
        # BEFORE query_enc, so a source well's observation rides the 6C write-back
        # onto its reaches and reaches nearby queries through the channel stack.
        self.has_src = f_src is not None
        if self.has_src and not writeback:
            # without the write-back a source obs can influence nothing but its own
            # (loss-masked) row -- a silent no-op arm, so fail loud instead.
            raise ValueError("source-obs features (f_src) require query-writeback (6C)")
        # source-well edges (assimilation rung 1): each query attention-reads its
        # <=k nearest source wells' observed residuals over direct spatial edges,
        # bypassing the diluted writeback->reach->lateral path rung 0 died on.
        # Independent of writeback/f_src (edges ARE the transmission path).
        self.has_srcedge = f_srcedge is not None
        if self.has_ms and self.has_anchor:
            # anchors are off in prod; keep the head bookkeeping ([q, ctx_reach, ctx_*])
            # single-branch so we never have to reconcile two hidden*3 read contexts.
            raise ValueError("mainstem-read (f_ms) is not supported with anchors")
        # portfolio-read (6B) supersedes the single-purpose mainstem read and, like it,
        # occupies the ONE extra hidden*3 read-context slot -- so it is mutually exclusive
        # with BOTH mainstem-read and anchors (the head reconciles only one extra context).
        if self.has_pf and (self.has_ms or self.has_anchor):
            raise ValueError(
                "portfolio-read (f_pf) is exclusive with mainstem-read + anchors"
            )
        if self.writeback and self.has_anchor:
            # 6C is wired only into the non-anchor forward branch (the anchor branch already
            # owns the pre-channel reach injection); combining them is unsupported.
            raise ValueError("query-writeback (6C) is not supported with anchors")
        self.pinball = pinball
        self.fac_skip = fac_skip
        self.fac_gate = fac_gate
        self.fac_lambda = fac_lambda
        self.sigma = sigma
        self.ordinal = ordinal
        self.two_surface = two_surface
        self.prior_gate = prior_gate
        self.mirror_anchor = mirror_anchor
        self.hang_anchor = hang_anchor
        self.directional_edges = directional_edges
        self.last_fac_gate: torch.Tensor | None = None
        self.last_fac_lambda: torch.Tensor | None = None
        self.last_prior_gate: torch.Tensor | None = None
        # free-head expert output (std units) from the LAST forward; the inference
        # renderer needs it to decompose the prior-gate mixture into expert WTE
        # surfaces (wte_hat = sum w_i * expert_wte_i), so it is captured detached
        # alongside last_prior_gate.
        self.last_head_out: torch.Tensor | None = None
        # heteroscedastic log-scale (Laplace b) from the LAST forward; kept WITH grad
        # so the trainer's NLL can backprop through it (unlike the detached last_* QA
        # captures above).
        self.sigma_log_b: torch.Tensor | None = None
        # monotone ordinal shallow-class logits (N, n_ordinal) from the LAST forward,
        # kept WITH grad so the trainer's BCE can backprop through them.
        self.ordinal_logits: torch.Tensor | None = None
        # two-surface mixture internals from the LAST forward (component means hp/hr,
        # log-scales lbp/lbr, membership logit m; all std target units), kept WITH
        # grad so the trainer's mixture NLL + assignment BCE can backprop.
        self.ts_out: dict[str, torch.Tensor] | None = None
        # Regional-aquifer substrate (Phase 1): an OPTIONAL gated residual correction
        # branch over the proven stream/FAC-residual head, never a wider head. "off"
        # / "fixed_stream" make it an EXACT no-op (the branch is short-circuited so no
        # aquifer dropout is drawn -> stream RNG/output is byte-identical to baseline);
        # only "learned" runs message passing + a gated delta. See regional_aquifer_graph.md.
        if aquifer_route not in ("off", "fixed_stream", "learned"):
            raise ValueError(f"bad aquifer_route: {aquifer_route!r}")
        self.has_aquifer = (
            f_aquifer is not None and n_aquifer_layers > 0 and aquifer_route != "off"
        )
        self.aquifer_route = aquifer_route
        self.last_aquifer_gate: torch.Tensor | None = None
        if self.has_aquifer and pinball:
            raise ValueError("--pinball with the aquifer branch is not supported yet")
        if self.has_aquifer and self.has_anchor:
            raise ValueError("the aquifer branch is not supported with anchors yet")
        self.reach_enc = nn.Sequential(
            nn.Linear(f_reach, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        # in_dim widens by the source-obs block when present (f_src=None -> unchanged,
        # byte-identical to baseline).
        self.query_enc = nn.Sequential(
            nn.Linear(f_query + (f_src or 0), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # Channel + lateral message passing is flow-direction-conditioned when
        # directional_edges is set (the trainer supplies ch_dir/lat_dir). Anchor convs
        # stay non-directional (an anchor BC has no up/down flow sense).
        self.channel = nn.ModuleList(
            [
                EdgeGatedConv(
                    hidden,
                    hidden,
                    f_ch,
                    hidden,
                    dropout=dropout,
                    directional=directional_edges,
                )
                for _ in range(n_channel_layers)
            ]
        )
        self.lateral = EdgeGatedConv(
            hidden,
            hidden,
            f_lat,
            hidden,
            dropout=dropout,
            directional=directional_edges,
        )
        head_in = hidden * 2
        if self.has_anchor:
            self.anchor_enc = nn.Sequential(
                nn.Linear(f_anchor, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            # WTE mode: the anchor's head is the Dirichlet BC VALUE, injected on a
            # dedicated value channel (fold-standardized in target space by the
            # caller) -- NOT a generic anchor_x covariate. Additive so the metadata
            # encoder still learns class/source/uncertainty; absent (residual mode)
            # leaves v2 anchor behavior unchanged.
            self.anchor_value_enc = nn.Linear(1, hidden, bias=False)
            # weight-shared across channel layers: one param set re-applied each step.
            self.anchor_to_reach = EdgeGatedConv(
                hidden, hidden, f_anchor_reach, hidden, dropout=dropout
            )
            self.anchor_to_query = EdgeGatedConv(
                hidden, hidden, f_anchor_query, hidden, dropout=dropout
            )
            head_in = hidden * 3  # [q, ctx_reach, ctx_anchor]
        if self.has_ms:
            # mainstem-read (item 2): one query->downstream-datum read conv over the
            # POST-channel-stack reach states, so the query attends to its basin's
            # discharge-datum reach's LEARNED state (seed evidence, drainage, 2-hop
            # context) in a single hop -- bypassing the med-7/p90-28-hop receptive-field
            # gap. Queries with no ms edge get a zero context (mean-agg default).
            self.ms_read = EdgeGatedConv(hidden, hidden, f_ms, hidden, dropout=dropout)
            head_in = hidden * 3  # [q, ctx_reach, ctx_ms]
        if self.has_pf:
            # portfolio-read (6B): one segment-softmax attention conv over each query's <=4
            # typed reference-site edges (ds_datum / up_head / wet / ho_any) on the POST-
            # channel-stack reach states, so the query attends to whichever heterogeneous
            # reference reach's LEARNED state matters -- a missing site is simply absent from
            # the softmax. Supersedes ms_read (superset of its ds_datum edge + 3 more sites).
            self.pf_read = PortfolioReadConv(
                hidden, hidden, f_pf, hidden, dropout=dropout
            )
            head_in = hidden * 3  # [q, ctx_reach, ctx_pf]
        if self.has_sc:
            # spatial-context read: segment-softmax attention over each query's
            # lattice-snapped ring-cell edges (radii x 8 octants), competing senders
            # exactly like the portfolio sites -- same conv, its own instance. SC cells
            # are RAW covariate nodes (no message passing among them), so they get
            # their own encoder. Deliberately an ADDITIVE slot, not a tenant of the
            # exclusive hidden*3 read-context slot above: the terrain read must
            # compose with the production anchor/ms/pf configurations.
            self.sc_enc = nn.Sequential(
                nn.Linear(f_sc_node, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            self.sc_read = PortfolioReadConv(
                hidden, hidden, f_sc_edge, hidden, dropout=dropout
            )
            head_in += hidden  # [.., ctx_sc]
        if self.has_mae:
            # MAE neighborhood embedding (self-supervised, target-blind): a per-query
            # mean-pooled ViT-MAE encoder vector over the query's multi-scale covariate
            # neighborhood. Like the SC read, a deliberately ADDITIVE head slot (not a
            # tenant of the exclusive hidden*3 read-context slot), so it composes with the
            # production anchor/gate/mirror/sigma configuration. Unlike SC it needs NO
            # graph read -- the embedding is already a per-query dense vector -- so it is
            # just an encoder MLP (no attention conv). Off => no module, no head widening,
            # so the run is byte-identical to baseline. See MAE_NEIGHBORHOOD_EMBEDDING.md.
            self.mae_enc = nn.Sequential(
                nn.Linear(f_mae, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            head_in += hidden  # [.., ctx_mae]
        if self.has_analog:
            # Analog-edge read (E3): each query attends (segment-softmax, the 6B/SC
            # PortfolioReadConv precedent) over its <=k NONLOCAL analog wells in the
            # AEF+MAE embedding space, importing each analog's fold-standardized observed
            # residual (`analog_src_val`, a per-node scalar encoded by analog_enc) as the
            # message payload, modulated by the target-blind edge attrs (cosine/geo/rel-
            # elev). A deliberately ADDITIVE head slot (like SC/MAE), NOT a tenant of the
            # exclusive hidden*3 read-context slot, so it composes with gate/mirror/sigma/
            # writeback/MAE. A query with no surviving analog edge (all analogs in the
            # held-out fold) aggregates to zero -- finite. Off => no module, no head
            # widening => byte-identical to baseline. See notes/E3_ANALOG_EDGES.md.
            self.analog_enc = nn.Sequential(
                nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            self.analog_read = PortfolioReadConv(
                hidden, hidden, f_analog, hidden, dropout=dropout
            )
            head_in += hidden  # [.., ctx_analog]
        if self.has_srcedge:
            # Source-edge read (assimilation rung 1): the analog slot's twin over
            # LOCAL spatial kNN edges to visible source wells ("learned IDW"). Each
            # query attends (PortfolioReadConv) over its surviving source edges,
            # importing the source's fold-standardized observed residual
            # (`srcedge_val`, per-node scalar via source_enc) modulated by the
            # target-blind spatial edge attrs (log-distances / rel-elev /
            # same-basin). Leakage control is the TRAINER's job: it filters the
            # edge set per forward to the visible-source protocol (per-epoch drawn
            # masks in training, all-train at val, train+val at final/OOF), so a
            # well never reads its own site and a test row never feeds an edge. A
            # query with no surviving edge aggregates to zero (finite). Additive
            # head slot; off => no module, no head widening => byte-identical.
            self.source_enc = nn.Sequential(
                nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            # srcedge_gated swaps softmax attention for independent per-edge
            # sigmoid gates: softmax can only REDISTRIBUTE trust among the <=k
            # neighbors (weights sum to 1), while sigmoid gates can close every
            # edge and abstain outright -- the missing off-switch when all
            # visible sources are uninformative (the cold/far-from-well regime).
            self.source_read = (
                EdgeGatedConv(hidden, hidden, f_srcedge, hidden, dropout=dropout)
                if srcedge_gated
                else PortfolioReadConv(
                    hidden, hidden, f_srcedge, hidden, dropout=dropout
                )
            )
            head_in += hidden  # [.., ctx_srcedge]
        if self.writeback:
            # query->reach write-back (6C): one bipartite gated conv (query as src, reach as
            # dst) applied as a residual BEFORE the channel stack, so well context mixes 2
            # hops outward along the channels and returns via the lateral/portfolio reads
            # (mirrors the anchor BC pre-channel injection). Off => the reordered query_enc
            # is a pure no-op (identical compute graph to baseline).
            self.writeback_conv = EdgeGatedConv(
                hidden, hidden, f_lat, hidden, dropout=dropout
            )
        if self.fac_skip:
            # raw-FAC bypass: the standardized FAC target-estimate + its presence flag
            # ride straight to the head (un-smoothed), and the output is anchored on
            # that estimate so message passing can only CORRECT FAC, never erase its
            # sharp shallow signal (the diagnosed over-smoothing failure).
            if self.fac_gate:
                # confidence gate c in (0,1) on the FAC anchor: the gate sees the query
                # context (incl. the deep regional prior) + FAC's own predicted DTW, so
                # it can RELEASE the anchor in the deep-regional regime where FAC
                # saturates and let the head lean on the regional prior.
                self.fac_gate_mlp = nn.Sequential(
                    nn.Linear(
                        head_in + 3, hidden
                    ),  # h | fac_base, present, fac_pred_dtw
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
                # Start FIRMLY anchored (c~0.95): epoch-0 behavior == plain fac-skip (the
                # proven shallow win), so release is opt-in with evidence rather than the
                # default-0.5 gate halving the shallow anchor from the start.
                nn.init.constant_(self.fac_gate_mlp[-1].bias, 3.0)
                head_in += 4  # h | fac_base, present, fac_pred_dtw, gate
            else:
                head_in += 2  # h | fac_base, present
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        if self.pinball:
            # asymmetric (deep) auxiliary head; primary Huber head leaves the bulk.
            self.pin_head = nn.Sequential(
                nn.Linear(head_in, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        # --- aquifer branch: CONSTRUCTED LAST so every baseline module above consumes
        # identical RNG whether or not the branch exists. Same seed => byte-identical
        # stream weights, which (with the fixed_stream short-circuit) is what makes the
        # no-op run match baseline to <1e-5 m DTW.
        if self.has_aquifer:
            self.aquifer_enc = nn.Sequential(
                nn.Linear(f_aquifer, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            self.aquifer_layers = nn.ModuleList(
                [
                    EdgeGatedConv(
                        hidden, hidden, f_aquifer_edge, hidden, dropout=dropout
                    )
                    for _ in range(n_aquifer_layers)
                ]
            )
            self.aquifer_to_query = EdgeGatedConv(
                hidden, hidden, f_aquifer_query, hidden, dropout=dropout
            )
            # delta + gate read [q, ctx_reach, ctx_aquifer] (3*hidden); independent of
            # the stream head width, so it never perturbs the stream head.
            self.aquifer_delta_head = nn.Sequential(
                nn.Linear(hidden * 3, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            self.aquifer_gate_mlp = nn.Sequential(
                nn.Linear(hidden * 3, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            # Gate pinned to sigmoid(gate_init) for ALL queries at init (final weight
            # zeroed, bias = gate_init): epoch-0 gate ~= sigmoid(-6) ~= 0.0025, so the
            # stream side dominates and the aquifer correction is strictly opt-in.
            nn.init.zeros_(self.aquifer_gate_mlp[-1].weight)
            nn.init.constant_(self.aquifer_gate_mlp[-1].bias, aquifer_gate_init)
            if aquifer_delta_init_zero:
                # delta == 0 at init regardless of gate: the learned branch starts as
                # an exact no-op over the stream head, then earns its correction.
                nn.init.zeros_(self.aquifer_delta_head[-1].weight)
                nn.init.zeros_(self.aquifer_delta_head[-1].bias)
        # --- lambda blend / sigma head / prior gate: constructed LAST (after every
        # baseline module) so a run with the flag off draws identical RNG to baseline.
        if self.fac_lambda:
            # convex blend: pred = lam * fac_anchor + (1 - lam) * head(h). Unlike
            # fac_skip's ADDITIVE anchor (head can double-count FAC), convexity forces
            # an interpretable per-well mixing weight -- lambda IS the shallow
            # terrain-coupled-zone map. The mlp sees the query context + the anchor
            # value + FAC's own predicted DTW (the release key), like fac_gate.
            self.fac_lambda_mlp = nn.Sequential(
                nn.Linear(head_in + 3, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            # start FIRMLY on FAC (lam~0.95): epoch-0 behavior is the proven sharp
            # shallow anchor; releasing to the GNN side is opt-in with evidence.
            nn.init.constant_(self.fac_lambda_mlp[-1].bias, 3.0)
        if self.sigma:
            # heteroscedastic Laplace scale head over the SAME input as the point
            # head; log_b clamped in forward. Enables selective shallow calls
            # (use predictions only where sigma is small).
            self.sigma_head = nn.Sequential(
                nn.Linear(head_in, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        if self.prior_gate:
            # K-way softmax mixture generalizing fac_lambda: pred = w_fac*FAC_anchor +
            # w_deep*deep_regional_anchor [+ w_mirror*terrain_mirror] + w_head*GNN_head.
            # The gate sees the query context, anchor values/presences, each prior's
            # own predicted DTW and their DISAGREEMENT -- where a shallow terrain prior
            # and a deep aquifer prior diverge is itself the regime signal no single
            # covariate carries. Absent priors are masked out of the softmax (weight
            # -> 0, remaining experts renormalize). The optional mirror expert
            # (mirror_anchor) is the subdued-replica shallow prior WTE = z_surf - d;
            # its inputs add mirror_base, mirror_pred_dtw and (fac_base - mirror_base)
            # -- the two bases share the fold's target standardization, so their
            # difference IS (d - fac_dtw)/y_s, the "FAC deeper than d" signal.
            # Expert order: [fac, deep, (mirror), (hang), head] -- head always LAST.
            # The optional hang expert (hang_anchor) is the well-free Dupuit hang
            # surface WTE = z_surf - dupuit_hang_dtw (boundary-conditioned datum);
            # its inputs add hang_base, hang_present, hang_pred_dtw and its DTW
            # disagreement with the deep-IDW expert -- where the stream-boundary
            # datum and the well-based deep datum diverge is the regime signal.
            n_experts = 3 + int(self.mirror_anchor) + int(self.hang_anchor)
            gate_in = (
                head_in
                + 7
                + (3 if self.mirror_anchor else 0)
                + (4 if self.hang_anchor else 0)
            )
            self.prior_gate_mlp = nn.Sequential(
                nn.Linear(gate_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_experts),
            )
            # start mostly on the FREE HEAD (w_head ~ 0.79 at K=3): epoch-0 behavior
            # is near baseline, and handing off to an anchor is opt-in with evidence --
            # the K-way analog of fac_lambda's anchored start, inverted because
            # starting on FAC would poison the deep majority. The head is always the
            # LAST expert.
            nn.init.zeros_(self.prior_gate_mlp[-1].bias)
            with torch.no_grad():
                self.prior_gate_mlp[-1].bias[-1] = 2.0
        if self.ordinal:
            # WP5 monotone ordinal shallow head, constructed AFTER every other module
            # (same RNG discipline as the aquifer branch: flag off => baseline modules
            # draw identical init RNG). One scalar "deepness" score over the SAME head
            # input as the point/sigma heads, plus ordered scalar cutpoints
            # c_0 < c_1 < ... (base + cumulative softplus). logit_j = c_j - score, so
            # P(DTW < t_0) <= P(DTW < t_1) <= ... is nested BY CONSTRUCTION -- the
            # head is not a threshold on the regression surface.
            self.ordinal_score = nn.Sequential(
                nn.Linear(head_in, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            self.ordinal_cut_raw = nn.Parameter(torch.zeros(n_ordinal))
        if self.two_surface:
            # WP2 latent two-surface mixture, constructed AFTER every other module
            # (same RNG discipline). The components are anchored on DIFFERENT
            # physical priors -- phreatic on the terrain-coupled shallow anchor
            # (FAC where present, else the terrain mirror), regional on the
            # deep-well IDW datum (absent -> 0, which in wte_residual space IS the
            # regional prior R) -- so component labels cannot swap across folds
            # (identifiability by parameterization, not by penalty). The mean
            # correction heads are zero-initialized: epoch-0 components ARE their
            # anchors, and departures from them are learned, not initial noise.

            def _ts_mlp() -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(head_in, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )

            self.ts_head_p = _ts_mlp()
            self.ts_head_r = _ts_mlp()
            self.ts_logb_p = _ts_mlp()
            self.ts_logb_r = _ts_mlp()
            self.ts_member = _ts_mlp()
            for mod in (self.ts_head_p, self.ts_head_r):
                nn.init.zeros_(mod[-1].weight)
                nn.init.zeros_(mod[-1].bias)

    def _augment_for_skip(self, h: torch.Tensor, g: dict):
        """Append the FAC raw-skip / confidence-gate pieces to the head input.

        Returns (head_input, additive_skip). Behaviour is identical to the prior inline
        block; factored out so the aquifer delta can be added on top of the stream
        primary without duplicating the fac_skip/fac_gate logic.
        """
        if not self.fac_skip:
            return h, 0.0
        fb = g["fac_base"].view(-1, 1)
        pres = g["fac_present"].view(-1, 1)
        if self.fac_gate:
            fpd = g["fac_pred_dtw"].view(-1, 1)
            c = torch.sigmoid(self.fac_gate_mlp(torch.cat([h, fb, pres, fpd], dim=-1)))
            self.last_fac_gate = c.detach()
            return torch.cat([h, fb, pres, fpd, c], dim=-1), (pres * c * fb).squeeze(-1)
        return torch.cat([h, fb, pres], dim=-1), (pres * fb).squeeze(-1)

    def _aquifer_delta(self, q: torch.Tensor, ctx_reach: torch.Tensor, g: dict):
        """ctx_aquifer read + gated correction delta (learned route only).

        The aquifer smoothing path deliberately does NOT use the channel branch's outer
        residual ``a = a + layer(a, ...)``: that residual keeps the stream layers sharp,
        but the aquifer substrate's job is a SMOOTH long-range field, so each layer fully
        replaces the node state (the update MLP still sees the prior state via its
        internal [x_dst, agg] concat). Returns (gate, delta), both shape (N,).
        """
        a = self.aquifer_enc(g["aquifer_x"])
        for layer in self.aquifer_layers:
            a = layer(a, a, g["aq_node_ei"], g["aq_node_ea"])
        ctx_aq = self.aquifer_to_query(a, q, g["aq_query_ei"], g["aq_query_ea"])
        h_aq = torch.cat([q, ctx_reach, ctx_aq], dim=-1)
        delta = self.aquifer_delta_head(h_aq).squeeze(-1)
        gate = torch.sigmoid(self.aquifer_gate_mlp(h_aq)).squeeze(-1)
        return gate, delta

    def forward(self, g: dict):
        # flow-direction sign per channel/lateral edge (None unless directional_edges);
        # keyed access errors loudly if the flag is on but the trainer omitted them.
        ch_dir = g["ch_dir"] if self.directional_edges else None
        lat_dir = g["lat_dir"] if self.directional_edges else None
        # source-obs block rides in through query_enc (keyed access errors loudly if
        # the flag is on but the trainer omitted src_x).
        qx = (
            torch.cat([g["query_x"], g["src_x"]], dim=-1)
            if self.has_src
            else g["query_x"]
        )
        r = self.reach_enc(g["reach_x"])
        if self.has_anchor and "anchor_x" in g:
            a = self.anchor_enc(g["anchor_x"])
            if "anchor_value" in g:
                a = a + self.anchor_value_enc(g["anchor_value"].view(-1, 1))
            # set the Dirichlet/soft BC first, then relax it along the network,
            # re-asserting it after each channel diffusion step.
            r = r + self.anchor_to_reach(a, r, g["ar_ei"], g["ar_ea"])
            for layer in self.channel:
                r = r + layer(r, r, g["ch_ei"], g["ch_ea"], dir_sign=ch_dir)
                r = r + self.anchor_to_reach(a, r, g["ar_ei"], g["ar_ea"])
            q = self.query_enc(qx)
            ctx_reach = self.lateral(r, q, g["lat_ei"], g["lat_ea"], dir_sign=lat_dir)
            ctx_anchor = self.anchor_to_query(a, q, g["aq_ei"], g["aq_ea"])
            h = torch.cat([q, ctx_reach, ctx_anchor], dim=-1)
        else:
            # 6C: query_enc runs BEFORE the channel loop so well context can be written back
            # onto reaches as a residual (mirrors the anchor BC pre-channel injection). When
            # writeback is off this reorder is a pure no-op (query_enc reads only query_x),
            # so the compute graph is byte-identical to baseline.
            q = self.query_enc(qx)
            if self.writeback:
                # reversed lateral edges (query src -> reach dst); residual, so reaches with
                # no incident well are unchanged. Context then mixes 2 hops out via channels.
                r = r + self.writeback_conv(q, r, g["lat_ei_reversed"], g["lat_ea"])
            for layer in self.channel:
                # residual channel update (direction-conditioned when enabled)
                r = r + layer(r, r, g["ch_ei"], g["ch_ea"], dir_sign=ch_dir)
            ctx_reach = self.lateral(r, q, g["lat_ei"], g["lat_ea"], dir_sign=lat_dir)
            if self.has_ms:
                # read the datum reach's post-channel-stack state; no ms edge -> zeros.
                ctx_ms = self.ms_read(r, q, g["ms_ei"], g["ms_ea"])
                h = torch.cat([q, ctx_reach, ctx_ms], dim=-1)
            elif self.has_pf:
                # attention-read whichever of the <=4 typed reference sites matters, over
                # the post-channel-stack reach states; no portfolio edge -> zero context.
                ctx_pf = self.pf_read(r, q, g["pf_ei"], g["pf_ea"])
                h = torch.cat([q, ctx_reach, ctx_pf], dim=-1)
            else:
                h = torch.cat([q, ctx_reach], dim=-1)
        if self.has_sc:
            # direction x scale terrain read: attend over the query's ring-cell
            # edges (encoded raw covariate cells; azimuth/ring/rel-elev on the edge).
            # Additive to whichever branch built h, so it composes with anchor/ms/pf.
            ctx_sc = self.sc_read(self.sc_enc(g["sc_x"]), q, g["sc_ei"], g["sc_ea"])
            h = torch.cat([h, ctx_sc], dim=-1)
        if self.has_mae:
            # additive per-query MAE context; composes with whichever branch built h
            # (anchor/ms/pf/sc) and rides through to every output head unchanged.
            h = torch.cat([h, self.mae_enc(g["mae_x"])], dim=-1)
        if self.has_analog:
            # additive per-query analog context: attention-read the fold-standardized
            # observed residual of each dest well's <=k nonlocal embedding analogs. The
            # trainer supplies analog_ei/analog_ea already masked to source-fold != this
            # fold (leak-free), and analog_src_val standardized in this fold's target
            # space. Zero-edge queries -> zero context (finite).
            ctx_analog = self.analog_read(
                self.analog_enc(g["analog_src_val"].view(-1, 1)),
                q,
                g["analog_ei"],
                g["analog_ea"],
            )
            h = torch.cat([h, ctx_analog], dim=-1)
        if self.has_srcedge:
            # additive per-query source-well context (rung 1): attention-read the
            # fold-standardized observed residual of the query's visible spatial
            # source wells. The trainer supplies srcedge_ei/srcedge_ea already
            # filtered to the visible-source protocol; zero-edge queries -> zero
            # context (finite).
            ctx_srcedge = self.source_read(
                self.source_enc(g["srcedge_val"].view(-1, 1)),
                q,
                g["srcedge_ei"],
                g["srcedge_ea"],
            )
            h = torch.cat([h, ctx_srcedge], dim=-1)
        # FAC raw-skip / confidence-gate pieces (no-op when fac_skip is off).
        h, skip = self._augment_for_skip(h, g)
        if self.sigma:
            # clamp keeps b in [e^-4, e^4] std units: bounded NLL, no collapse to
            # zero-variance on easy wells.
            self.sigma_log_b = torch.clamp(self.sigma_head(h).squeeze(-1), -4.0, 4.0)
        if self.fac_lambda:
            fb = g["fac_base"].view(-1, 1)
            pres = g["fac_present"].view(-1, 1)
            fpd = g["fac_pred_dtw"].view(-1, 1)
            lam = pres * torch.sigmoid(
                self.fac_lambda_mlp(torch.cat([h, fb, pres, fpd], dim=-1))
            )
            self.last_fac_lambda = lam.detach()
            # convex blend; absent FAC -> lam=0 -> pure GNN head (fb is 0 there).
            primary = (lam * fb + (1.0 - lam) * self.head(h)).squeeze(-1)
        elif self.prior_gate:
            fb = g["fac_base"].view(-1, 1)
            fpres = g["fac_present"].view(-1, 1)
            fpd = g["fac_pred_dtw"].view(-1, 1)
            db = g["deep_base"].view(-1, 1)
            dpres = g["deep_present"].view(-1, 1)
            dpd = g["deep_pred_dtw"].view(-1, 1)
            # prior disagreement (std units), defined only where BOTH priors exist.
            dis = (fpd - dpd) * fpres * dpres
            gate_in = [h, fb, fpres, fpd, db, dpres, dpd, dis]
            ones = torch.ones_like(fpres)
            expert_vals = [fb, db]
            mask_parts = [fpres, dpres]
            if self.mirror_anchor:
                mb = g["mirror_base"].view(-1, 1)
                mpd = g["mirror_pred_dtw"].view(-1, 1)
                # fac and mirror bases share the fold's target standardization, so
                # (fb - mb) IS (d - fac_dtw)/y_s -- the "FAC deeper than the mirror
                # offset" regime signal, scale-consistent by construction.
                dis_fm = (fb - mb) * fpres
                gate_in += [mb, mpd, dis_fm]
                expert_vals.append(mb)
                mask_parts.append(ones)
            if self.hang_anchor:
                hb = g["hang_base"].view(-1, 1)
                hpres = g["hang_present"].view(-1, 1)
                hpd = g["hang_pred_dtw"].view(-1, 1)
                # hang-vs-deep DTW disagreement: the boundary-conditioned datum
                # against the well-IDW deep datum, defined where both exist.
                dis_hd = (hpd - dpd) * hpres * dpres
                gate_in += [hb, hpres, hpd, dis_hd]
                expert_vals.append(hb)
                mask_parts.append(hpres)
            logits = self.prior_gate_mlp(torch.cat(gate_in, dim=-1))
            # mask absent experts out of the softmax; mirror + head always present.
            mask = torch.cat(mask_parts + [ones], dim=-1)
            w = torch.softmax(logits + (1.0 - mask) * -1e9, dim=-1)
            self.last_prior_gate = w.detach()
            hv = self.head(h)
            self.last_head_out = hv.detach().squeeze(-1)
            expert_vals.append(hv)
            primary = (w * torch.cat(expert_vals, dim=-1)).sum(-1)
        elif self.two_surface:
            fb = g["fac_base"].view(-1, 1)
            fpres = g["fac_present"].view(-1, 1)
            mb = g["mirror_base"].view(-1, 1)
            db = g["deep_base"].view(-1, 1)
            dpres = g["deep_present"].view(-1, 1)
            # phreatic component = terrain-coupled shallow anchor (FAC where
            # present, else the mirror) + learned correction; regional component
            # = deep-IDW datum (absent -> 0 = the regional prior R itself in
            # wte_residual space) + learned correction. Zero-init corrections
            # make epoch-0 components exactly their anchors.
            hp = (fpres * fb + (1.0 - fpres) * mb).squeeze(-1) + self.ts_head_p(
                h
            ).squeeze(-1)
            hr = (dpres * db).squeeze(-1) + self.ts_head_r(h).squeeze(-1)
            # clamp keeps each component's Laplace b in [e^-4, e^4] std units,
            # same bounds as the sigma head.
            lbp = torch.clamp(self.ts_logb_p(h).squeeze(-1), -4.0, 4.0)
            lbr = torch.clamp(self.ts_logb_r(h).squeeze(-1), -4.0, 4.0)
            m = self.ts_member(h).squeeze(-1)
            self.ts_out = {"hp": hp, "hr": hr, "lbp": lbp, "lbr": lbr, "m": m}
            pi = torch.sigmoid(m)
            # point prediction (early stop + legacy scoring) is the mixture mean.
            primary = pi * hp + (1.0 - pi) * hr
        else:
            hv = self.head(h).squeeze(-1)
            self.last_head_out = hv.detach()
            primary = hv + skip
        # Regional-aquifer correction: gated additive delta over the stream primary.
        # "fixed_stream"/"off" short-circuit (no branch executed, no aquifer dropout
        # drawn) so the run is an EXACT no-op vs baseline; only "learned" contributes.
        if self.has_aquifer and self.aquifer_route == "learned":
            gate, aq_delta = self._aquifer_delta(q, ctx_reach, g)
            self.last_aquifer_gate = gate.detach()
            primary = primary + gate * aq_delta
        elif self.has_aquifer:
            self.last_aquifer_gate = torch.zeros_like(primary)
        if self.ordinal:
            # computed LAST so every shared module above consumes identical forward
            # RNG (dropout draws) with or without the ordinal head in the loss.
            s = self.ordinal_score(h)  # (N, 1) deepness score
            base = self.ordinal_cut_raw[0:1]
            steps = nn.functional.softplus(self.ordinal_cut_raw[1:])
            cuts = torch.cat([base, base + torch.cumsum(steps, dim=0)])
            self.ordinal_logits = cuts.view(1, -1) - s
        if self.pinball:
            return primary, self.pin_head(h).squeeze(-1) + skip
        return primary


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def val_blocks(
    trainval: np.ndarray, blocks: np.ndarray, frac: float, rng: np.random.RandomState
) -> np.ndarray:
    """Hold out whole blocks summing to ~``frac`` of TRAIN-VAL wells as a val set.

    Selection is by whole spatial block (keeps val spatially separated from
    train), but accumulated by well count -- blocks vary by >100x in density, so
    picking a fixed fraction of blocks would wildly over/under-shoot the row
    fraction. Deterministic given ``rng`` (seeded).
    """
    idx = np.where(trainval)[0]
    bl = blocks[idx]
    ub, counts = np.unique(bl, return_counts=True)
    order = rng.permutation(len(ub))
    target = frac * idx.size
    chosen: set = set()
    acc = 0
    for j in order:
        if acc >= target:
            break
        chosen.add(ub[j])
        acc += counts[j]
    return np.isin(blocks, list(chosen)) & trainval


def train_fold(
    model: WTEGraphNet,
    feat: dict,
    y_std: torch.Tensor,
    tr: np.ndarray,
    va: np.ndarray,
    dem: np.ndarray,
    reg_wte: np.ndarray,
    obs_dtw: np.ndarray,
    y_mean: float,
    y_scale: float,
    args,
    device: str,
) -> tuple[np.ndarray, float, int]:
    """Train one fold; early-stop on val DTW-MAD; return (residual_hat_all, best_mad, best_epoch)."""
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
            resid_hat = model(feat).cpu().numpy() * y_scale + y_mean
        pred_dtw = dem - (reg_wte + resid_hat)
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
        resid_hat = model(feat).cpu().numpy() * y_scale + y_mean
    return resid_hat, best_mad, best_epoch


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--graph-dir", required=True, help="bundle from build_wte_graph_inputs.py"
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--channel-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--min-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=80)
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

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device: %s (torch %s)", device, torch.__version__)

    manifest = json.loads((gdir / "graph_manifest.json").read_text())
    reach_cols = manifest["reach_feature_cols"]
    query_cols = manifest["query_feature_cols"]
    ch_cols = manifest["channel_edge_feature_cols"]
    lat_cols = manifest["lateral_edge_feature_cols"]
    target_col = manifest["target_col"]
    fold_col = manifest["cv_fold_col"]
    group_col = manifest["cv_group_col"]
    method = manifest["residual_method"]

    rn = gpd.read_parquet(gdir / "reach_nodes.parquet").sort_values("reach_node_idx")
    qn = gpd.read_parquet(gdir / "query_nodes.parquet").sort_values("query_node_idx")
    ce = pd.read_parquet(gdir / "channel_edges.parquet")
    le = pd.read_parquet(gdir / "lateral_edges.parquet")
    rn = rn.reset_index(drop=True)
    qn = qn.reset_index(drop=True)
    assert (rn["reach_node_idx"].to_numpy() == np.arange(len(rn))).all()
    assert (qn["query_node_idx"].to_numpy() == np.arange(len(qn))).all()
    log.info(
        "reaches=%d queries=%d channel_edges=%d lateral_edges=%d",
        len(rn),
        len(qn),
        len(ce),
        len(le),
    )

    # --- constant graph tensors (reaches/edges carry no labels) -----------
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
    # lateral edges are query->reach in the bundle; reverse to reach->query.
    lat_ei = torch.as_tensor(
        le[["reach_node_idx", "query_node_idx"]].to_numpy().T,
        dtype=torch.long,
        device=device,
    )

    dem = qn["dem_m"].to_numpy("float64")
    reg_wte = qn["regional_wte_oof_m"].to_numpy("float64")
    obs_dtw = qn["obs_dtw_m"].to_numpy("float64")
    target = qn[target_col].to_numpy("float64")
    folds = np.array(sorted(qn[fold_col].unique()))
    blocks = qn[group_col].to_numpy()
    for nm, arr in (
        ("dem_m", dem),
        ("regional_wte_oof_m", reg_wte),
        ("obs_dtw_m", obs_dtw),
    ):
        if not np.isfinite(arr).all():
            raise SystemExit(
                f"{int((~np.isfinite(arr)).sum())} non-finite {nm} in query nodes"
            )

    f_ch = ch_ea.shape[1]
    f_lat = lat_ea.shape[1]
    resid_hat_oof = np.full(len(qn), np.nan)
    gate_records: list[pd.DataFrame] = []
    fold_log: list[dict] = []

    for f in folds:
        test = qn[fold_col].to_numpy() == f
        trainval = ~test
        va = val_blocks(trainval, blocks, args.val_frac, rng)
        tr = trainval & ~va
        log.info("fold %d: train=%d val=%d test=%d", f, tr.sum(), va.sum(), test.sum())

        q_stats = fit_stats(qn, query_cols, tr)
        query_x = torch.as_tensor(
            apply_stats(qn, q_stats), dtype=torch.float32, device=device
        )
        f_query = query_x.shape[1]
        feat = {
            "reach_x": reach_x,
            "query_x": query_x,
            "ch_ei": ch_ei,
            "ch_ea": ch_ea,
            "lat_ei": lat_ei,
            "lat_ea": lat_ea,
        }

        # Robust (median / MAD) target scaling: the residual target has a heavy
        # deep-well tail that inflates std and squashes the shallow bulk below
        # the Huber knee, starving the bulk signal. Robust scale keeps the bulk
        # near unit scale so Huber clips the tail instead of being ruled by it.
        y_center = float(np.median(target[tr]))
        y_scale = float(1.4826 * np.median(np.abs(target[tr] - y_center)) or 1.0)
        y_std = torch.as_tensor(
            (target - y_center) / y_scale, dtype=torch.float32, device=device
        )

        torch.manual_seed(args.seed + int(f))
        model = WTEGraphNet(
            reach_x.shape[1],
            f_query,
            f_ch,
            f_lat,
            args.hidden,
            args.channel_layers,
            args.dropout,
        ).to(device)
        resid_hat, best_mad, best_epoch = train_fold(
            model,
            feat,
            y_std,
            tr,
            va,
            dem,
            reg_wte,
            obs_dtw,
            y_center,
            y_scale,
            args,
            device,
        )
        resid_hat_oof[test] = resid_hat[test]
        pred_dtw_all = dem - (reg_wte + resid_hat)
        tr_mad = float(np.nanmedian(np.abs(pred_dtw_all[tr] - obs_dtw[tr])))
        te_mad = float(np.nanmedian(np.abs(pred_dtw_all[test] - obs_dtw[test])))
        log.info(
            "fold %d: val DTW-MAD=%.3f @epoch %d | train=%.3f test=%.3f",
            f,
            best_mad,
            best_epoch,
            tr_mad,
            te_mad,
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

        # learned lateral gate for this fold's TEST queries (QA layer).
        model.eval()
        with torch.no_grad():
            model(feat)
        gate = model.lateral.last_gate.squeeze(-1).cpu().numpy()  # aligned to le rows
        test_edge = test[le["query_node_idx"].to_numpy()]
        gate_records.append(
            pd.DataFrame(
                {
                    "query_node_idx": le.loc[test_edge, "query_node_idx"].to_numpy(),
                    "reach_node_idx": le.loc[test_edge, "reach_node_idx"].to_numpy(),
                    "rank": le.loc[test_edge, "rank"].to_numpy(),
                    "is_controlling": le.loc[test_edge, "is_controlling"].to_numpy(),
                    "lateral_dist_m": le.loc[test_edge, "lateral_dist_m"].to_numpy(),
                    "lateral_gate": gate[test_edge],
                    "cv_fold": int(f),
                }
            )
        )

    if not np.isfinite(resid_hat_oof).all():
        raise SystemExit(
            f"{int((~np.isfinite(resid_hat_oof)).sum())} queries got no OOF prediction"
        )

    gnn_wte = reg_wte + resid_hat_oof
    gnn_dtw = dem - gnn_wte
    oof = gpd.GeoDataFrame(
        {
            "canonical_id": qn["canonical_id"].to_numpy(),
            "query_node_idx": qn["query_node_idx"].to_numpy(),
            "cv_fold": qn[fold_col].to_numpy(),
            "source": qn["source"].to_numpy(),
            "dem_m": dem,
            "obs_dtw_m": obs_dtw,
            "obs_wte_m": qn["obs_wte_m"].to_numpy(),
            "regional_wte_oof_m": reg_wte,
            "residual_wte_hat_gnn_m": resid_hat_oof,
            "gnn_wte_m": gnn_wte,
            "gnn_dtw_m": gnn_dtw,
            "geometry": qn.geometry.to_numpy(),
        },
        geometry="geometry",
        crs=qn.crs,
    )
    oof.to_parquet(out_dir / "gnn_oof_predictions.parquet")
    gates = pd.concat(gate_records, ignore_index=True)
    gates.to_parquet(out_dir / "gnn_lateral_gates.parquet")

    run = {
        "graph_dir": str(gdir),
        "residual_method": method,
        "device": device,
        "torch": torch.__version__,
        "hyperparams": {
            "hidden": args.hidden,
            "channel_layers": args.channel_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "val_frac": args.val_frac,
            "huber_delta": args.huber_delta,
            "seed": args.seed,
        },
        "feature_dims": {
            "reach": int(reach_x.shape[1]),
            "query": int(f_query),
            "channel_edge": int(f_ch),
            "lateral_edge": int(f_lat),
        },
        "counts": {
            "reach_nodes": len(rn),
            "query_nodes": len(qn),
            "channel_edges": len(ce),
            "lateral_edges": len(le),
        },
        "folds": fold_log,
        "target_col": target_col,
        "leakage_notes": manifest.get("leakage_notes", []),
    }
    (out_dir / "gnn_run.json").write_text(json.dumps(run, indent=2))

    controlling = gates[gates["is_controlling"] == 1.0]
    log.info(
        "OOF residual_hat: median=%.2f mean=%.2f | controlling gate: median=%.3f mean=%.3f",
        float(np.median(resid_hat_oof)),
        float(np.mean(resid_hat_oof)),
        float(controlling["lateral_gate"].median()),
        float(controlling["lateral_gate"].mean()),
    )
    log.info(
        "wrote gnn_oof_predictions.parquet, gnn_lateral_gates.parquet, gnn_run.json -> %s",
        out_dir,
    )


if __name__ == "__main__":
    main()
