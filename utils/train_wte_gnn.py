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
        writeback: bool = False,
        pinball: bool = False,
        fac_skip: bool = False,
        fac_gate: bool = False,
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
        self.has_anchor = f_anchor is not None
        self.has_ms = f_ms is not None
        self.has_pf = f_pf is not None
        self.writeback = writeback
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
        self.directional_edges = directional_edges
        self.last_fac_gate: torch.Tensor | None = None
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
        self.query_enc = nn.Sequential(
            nn.Linear(f_query, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
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
            q = self.query_enc(g["query_x"])
            ctx_reach = self.lateral(r, q, g["lat_ei"], g["lat_ea"], dir_sign=lat_dir)
            ctx_anchor = self.anchor_to_query(a, q, g["aq_ei"], g["aq_ea"])
            h = torch.cat([q, ctx_reach, ctx_anchor], dim=-1)
        else:
            # 6C: query_enc runs BEFORE the channel loop so well context can be written back
            # onto reaches as a residual (mirrors the anchor BC pre-channel injection). When
            # writeback is off this reorder is a pure no-op (query_enc reads only query_x),
            # so the compute graph is byte-identical to baseline.
            q = self.query_enc(g["query_x"])
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
        # FAC raw-skip / confidence-gate pieces (no-op when fac_skip is off).
        h, skip = self._augment_for_skip(h, g)
        primary = self.head(h).squeeze(-1) + skip
        # Regional-aquifer correction: gated additive delta over the stream primary.
        # "fixed_stream"/"off" short-circuit (no branch executed, no aquifer dropout
        # drawn) so the run is an EXACT no-op vs baseline; only "learned" contributes.
        if self.has_aquifer and self.aquifer_route == "learned":
            gate, aq_delta = self._aquifer_delta(q, ctx_reach, g)
            self.last_aquifer_gate = gate.detach()
            primary = primary + gate * aq_delta
        elif self.has_aquifer:
            self.last_aquifer_gate = torch.zeros_like(primary)
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
