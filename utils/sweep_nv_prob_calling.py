"""Unconditional probability calling surfaces on the NV primary panel (E4 of
``notes/SHALLOW_LEVER_EXPERIMENTS.md``).

Reads ``sites_scored.parquet`` from a ``score_nv_shallow_panel.py`` run and,
for every arm in it, sweeps a probability cut over the whole panel (no
``pred < t`` precondition, unlike the depth-gated sweep in the scorer) for two
probability variables:

- ``laplace``: P(DTW < t) = Laplace CDF of (t - pred) with scale sigma_m, the
  sigma-head probability every arm carries;
- ``ordinal``: the arm's ``p_dtw_lt_{t}m`` column when the arm has an ordinal
  head.

Per (arm, variable, threshold, stratum) it reports the precision / recall /
F1 curve over cuts 0.05..0.95, PR-AUC (average precision, dimensionless),
Brier score and Brier skill = 1 - Brier / Brier_climatology (dimensionless,
positive = better than the stratum base rate), each with a CI95 from the HUC8
block bootstrap (blocks = ``huc8_block`` in the scored sites; the bounded
quantity is the statistic on the resampled panel), and two operating points:
the best precision at recall >= 0.55 (the plan's unconditional gate) and the
best recall at precision >= 0.71 (the gated gate).

Run::

    uv run --directory /home/dgketchum/code/handily python \
        /home/dgketchum/code/handily/utils/sweep_nv_prob_calling.py \
        --sites /data/ssd2/handily/nv/regional/shallow_levers/tier2_e2_e4/sites_scored.parquet \
        --out-dir /data/ssd2/handily/nv/regional/shallow_levers/tier2_e2_e4/prob_sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "utils"))

from score_nv_shallow_panel import BlockBoot, f1, laplace_cdf, log, pr_at  # noqa: E402

CUTS = np.round(np.arange(0.05, 0.96, 0.05), 2)
DEFAULT_THRESHOLDS = "2,5"
GATE_RECALL = 0.55
GATE_PRECISION = 0.71


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the precision-recall curve by the step rule (sklearn's AP)."""
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, ys.size + 1)
    return float((prec * ys).sum() / ys.sum())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def brier_skill(y: np.ndarray, p: np.ndarray) -> float:
    clim = y.mean()
    b0 = float(np.mean((clim - y) ** 2))
    return 1.0 - brier(y, p) / b0 if b0 > 0 else np.nan


def strata_masks(ss: pd.DataFrame) -> dict[str, np.ndarray]:
    obs = ss["obs_dtw_m"].to_numpy("float64")
    irr = ss["dist_irr_m"].to_numpy("float64") <= 500.0
    wt = ss["confinement_class"].isin(["unconfined", "unconfined_marginal"]).to_numpy()
    return {
        "all": np.ones(len(ss), bool),
        "water-table classes": wt,
        "irr500": irr,
        "irr500 obs<10 m": irr & (obs < 10.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        help="comma-separated shallow thresholds (m) to call at; the ordinal "
        "variable is swept only where the arm emits p_dtw_lt_{t}m",
    )
    args = ap.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ss = pd.read_parquet(args.sites)
    arms = sorted(
        c[: -len("__pred_dtw_m")] for c in ss.columns if c.endswith("__pred_dtw_m")
    )
    obs = ss["obs_dtw_m"].to_numpy("float64")
    blocks = ss["huc8_block"].astype(str).to_numpy()
    boot = BlockBoot(blocks, args.n_boot, args.seed)
    strata = strata_masks(ss)
    log(f"{len(ss)} sites, arms {arms}, {boot.n_blocks} HUC8 blocks")

    probs: dict[tuple[str, str, float], np.ndarray] = {}
    for a in arms:
        pred = ss[f"{a}__pred_dtw_m"].to_numpy("float64")
        sig = ss[f"{a}__sigma_m"].to_numpy("float64")
        for t in thresholds:
            probs[(a, "laplace", t)] = laplace_cdf(t - pred, sig)
            col = f"{a}__p_dtw_lt_{t:g}m"
            if col in ss.columns:
                probs[(a, "ordinal", t)] = ss[col].to_numpy("float64")

    curve, summary = [], []
    for (a, var, t), p in probs.items():
        y = (obs < t).astype("float64")
        for label, m in strata.items():
            n_shallow = int(y[m].sum())
            rows = []
            for c in CUTS:
                called = (p >= c) & m
                pr, rc = pr_at(obs[m], called[m], t)
                lo, hi = boot.ci(lambda i: pr_at(obs[i], (p[i] >= c), t)[0], m)
                rlo, rhi = boot.ci(lambda i: pr_at(obs[i], (p[i] >= c), t)[1], m)
                rows.append(
                    {
                        "arm": a,
                        "variable": var,
                        "threshold_m": t,
                        "stratum": label,
                        "cut": float(c),
                        "n_sites": int(m.sum()),
                        "n_obs_shallow": n_shallow,
                        "n_called": int(called.sum()),
                        "precision": pr,
                        "precision_lo": lo,
                        "precision_hi": hi,
                        "recall": rc,
                        "recall_lo": rlo,
                        "recall_hi": rhi,
                        "f1": f1(pr, rc),
                    }
                )
            curve += rows
            df = pd.DataFrame(rows)
            ok_r = df[df["recall"] >= GATE_RECALL]
            ok_p = df[df["precision"] >= GATE_PRECISION]
            best_p = ok_r.loc[ok_r["precision"].idxmax()] if len(ok_r) else None
            best_r = ok_p.loc[ok_p["recall"].idxmax()] if len(ok_p) else None
            ap_lo, ap_hi = boot.ci(lambda i: average_precision(y[i], p[i]), m)
            bs_lo, bs_hi = boot.ci(lambda i: brier_skill(y[i], p[i]), m)
            summary.append(
                {
                    "arm": a,
                    "variable": var,
                    "threshold_m": t,
                    "stratum": label,
                    "n_sites": int(m.sum()),
                    "n_obs_shallow": n_shallow,
                    "base_rate": n_shallow / m.sum(),
                    "pr_auc": average_precision(y[m], p[m]),
                    "pr_auc_lo": ap_lo,
                    "pr_auc_hi": ap_hi,
                    "brier": brier(y[m], p[m]),
                    "brier_skill": brier_skill(y[m], p[m]),
                    "brier_skill_lo": bs_lo,
                    "brier_skill_hi": bs_hi,
                    "best_prec_at_rec_ge_0.55": None
                    if best_p is None
                    else best_p["precision"],
                    "best_prec_at_rec_ge_0.55_lo": None
                    if best_p is None
                    else best_p["precision_lo"],
                    "best_prec_at_rec_ge_0.55_hi": None
                    if best_p is None
                    else best_p["precision_hi"],
                    "cut_for_rec_ge_0.55": None if best_p is None else best_p["cut"],
                    "recall_at_that_cut": None if best_p is None else best_p["recall"],
                    "best_rec_at_prec_ge_0.71": None
                    if best_r is None
                    else best_r["recall"],
                    "cut_for_prec_ge_0.71": None if best_r is None else best_r["cut"],
                    "precision_at_that_cut": None
                    if best_r is None
                    else best_r["precision"],
                }
            )
        log(f"  {a} {var} {t:g} m done")
    pd.DataFrame(curve).to_csv(out / "prob_curve.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "prob_summary.csv", index=False)

    cdf = pd.DataFrame(curve)
    for t in thresholds:
        fig, axes = plt.subplots(
            1, len(strata), figsize=(4.2 * len(strata), 4), sharey=True
        )
        for ax, label in zip(axes, strata):
            sub = cdf[(cdf["threshold_m"] == t) & (cdf["stratum"] == label)]
            for (a, var), g in sub.groupby(["arm", "variable"]):
                g = g.sort_values("recall")
                ax.plot(g["recall"], g["precision"], marker=".", label=f"{a} {var}")
            ax.axhline(GATE_PRECISION, ls=":", c="k", lw=0.8)
            ax.axvline(GATE_RECALL, ls=":", c="k", lw=0.8)
            ax.set_title(f"{label} (<{t:g} m)")
            ax.set_xlabel("recall")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("precision")
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"pr_curves_{t:g}m.png", dpi=130)
        plt.close(fig)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
