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
    # absent from one fold's TRAIN split) leaves med/mean/std NaN, which would
    # propagate NaN through apply_stats and poison the tensor. Neutralize to
    # med=0/mean=0/std=1 so its z-scored column is a constant 0 -- the missingness
    # indicator (nan_cols, computed whole-df below) still carries the absence.
    med = med.fillna(0.0)
    mean = mean.fillna(0.0)
    std = std.fillna(1.0)
    nan_cols = [c for c in cols if df[c].isna().any()]
    return {"cols": cols, "med": med, "mean": mean, "std": std, "nan_cols": nan_cols}


def apply_stats(df: pd.DataFrame, stats: dict) -> np.ndarray:
    """Z-scored feature block with missingness-indicator columns appended."""
    cols = stats["cols"]
    sub = df[cols].astype("float64")
    miss = sub.isna()
    z = (sub.fillna(stats["med"]) - stats["mean"]) / stats["std"]
    parts = [z.to_numpy("float64")]
    if stats["nan_cols"]:
        parts.append(miss[stats["nan_cols"]].astype("float64").to_numpy())
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Edge-gated message passing
# ---------------------------------------------------------------------------
class EdgeGatedConv(MessagePassing):
    """msg_ij = sigmoid(gate(x_i, x_j, e_ij)) * transform(x_j, e_ij); update(x_i, agg)."""

    def __init__(
        self,
        in_src: int,
        in_dst: int,
        edge_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        aggr: str = "mean",
    ) -> None:
        super().__init__(aggr=aggr, flow="source_to_target")
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
    ) -> torch.Tensor:
        agg = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            edge_attr=edge_attr,
            size=(x_src.size(0), x_dst.size(0)),
        )
        return self.upd_mlp(torch.cat([x_dst, agg], dim=-1))

    def message(
        self, x_j: torch.Tensor, x_i: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        g = torch.sigmoid(self.gate_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1)))
        self.last_gate = g.detach()
        m = self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))
        return g * m


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
        pinball: bool = False,
    ) -> None:
        super().__init__()
        self.has_anchor = f_anchor is not None
        self.pinball = pinball
        self.reach_enc = nn.Sequential(
            nn.Linear(f_reach, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.query_enc = nn.Sequential(
            nn.Linear(f_query, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.channel = nn.ModuleList(
            [
                EdgeGatedConv(hidden, hidden, f_ch, hidden, dropout=dropout)
                for _ in range(n_channel_layers)
            ]
        )
        self.lateral = EdgeGatedConv(hidden, hidden, f_lat, hidden, dropout=dropout)
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

    def forward(self, g: dict):
        r = self.reach_enc(g["reach_x"])
        if self.has_anchor and "anchor_x" in g:
            a = self.anchor_enc(g["anchor_x"])
            if "anchor_value" in g:
                a = a + self.anchor_value_enc(g["anchor_value"].view(-1, 1))
            # set the Dirichlet/soft BC first, then relax it along the network,
            # re-asserting it after each channel diffusion step.
            r = r + self.anchor_to_reach(a, r, g["ar_ei"], g["ar_ea"])
            for layer in self.channel:
                r = r + layer(r, r, g["ch_ei"], g["ch_ea"])
                r = r + self.anchor_to_reach(a, r, g["ar_ei"], g["ar_ea"])
            q = self.query_enc(g["query_x"])
            ctx_reach = self.lateral(r, q, g["lat_ei"], g["lat_ea"])
            ctx_anchor = self.anchor_to_query(a, q, g["aq_ei"], g["aq_ea"])
            h = torch.cat([q, ctx_reach, ctx_anchor], dim=-1)
        else:
            for layer in self.channel:
                r = r + layer(r, r, g["ch_ei"], g["ch_ea"])  # residual channel update
            q = self.query_enc(g["query_x"])
            ctx = self.lateral(r, q, g["lat_ei"], g["lat_ea"])
            h = torch.cat([q, ctx], dim=-1)
        primary = self.head(h).squeeze(-1)
        if self.pinball:
            return primary, self.pin_head(h).squeeze(-1)
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
