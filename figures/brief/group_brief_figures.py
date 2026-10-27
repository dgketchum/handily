"""Two slide figures for the five-minute NV + NM group brief (Oct 2026).

Both read the statewide scorers' outputs only; nothing is recomputed from
rasters. Panel A (held-out admission-eligible sites) throughout.

  brief_scatter.png   observed vs predicted depth to water at held-out wells,
                      rows NV / NM, columns handily (parity render) / Ma
  brief_distance.png  MAD vs distance to the nearest admitted source, with
                      CI95 bars, read from fig06_error_vs_distance_stats.csv

Run::

    uv run python figures/brief/group_brief_figures.py            # both states
    uv run python figures/brief/group_brief_figures.py --state NM  # one state, *_nm.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from utils.figures.fig_common import use_style  # noqa: E402

OUT = Path("/data/ssd2/handily/figures/brief")
STATES = {
    "Nevada": {
        "eval": "/data/ssd2/handily/nv/regional/statewide_eval_round3_parity_sm4",
        "stats": "/data/ssd2/handily/figures/nwi/fig06_error_vs_distance_stats.csv",
        "agency": "NDWR",
    },
    "New Mexico": {
        "eval": "/data/ssd2/handily/nm/regional/ose_lbl/statewide_eval_parity_sm4",
        "stats": "/data/ssd2/handily/figures/nmose/fig06_error_vs_distance_stats.csv",
        "agency": "OSE",
    },
}
PRED = {"handily": "parity__dtw", "Ma et al. (2026)": "ma_resampled_m"}
BLUE = "#2E5C97"
GREY = "#808080"
SCATTER = {"handily": BLUE, "Ma et al. (2026)": GREY}
LIM = (0.1, 1000.0)


def slide_style() -> None:
    use_style()
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def panel_a(path: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d = d[d["is_admission_eligible"].astype(bool)]
    cols = ["obs_dtw_m", *PRED.values()]
    d = d[np.isfinite(d[cols]).all(axis=1)]
    return d


def stats(obs, pred) -> tuple[float, float, float]:
    r = pred - obs
    return float(np.median(np.abs(r))), float(np.sqrt(np.mean(r**2))), float(np.mean(r))


def scatter(states: dict, suffix: str) -> None:
    nrow = len(states)
    fig, axes = plt.subplots(
        nrow, 2, figsize=(7.2, 3.3 * nrow), sharex=True, sharey=True, squeeze=False
    )
    for i, (state, cfg) in enumerate(states.items()):
        d = panel_a(f"{cfg['eval']}/sites_sampled.parquet")
        obs = d["obs_dtw_m"].clip(lower=LIM[0]).to_numpy()
        for j, (name, col) in enumerate(PRED.items()):
            ax = axes[i, j]
            pred_raw = d[col].to_numpy()
            pred = np.clip(pred_raw, LIM[0], LIM[1])
            ax.plot(LIM, LIM, color="#808080", lw=0.8, zorder=1)
            ax.scatter(
                obs,
                pred,
                s=3,
                c=SCATTER[name],
                alpha=0.3,
                lw=0,
                zorder=2,
                rasterized=True,
            )
            mad, rmse, bias = stats(d["obs_dtw_m"].to_numpy(), pred_raw)
            ax.text(
                0.03,
                0.97,
                f"{state} · {name}\nn = {len(d):,}\nMAD {mad:.2f} m\nRMSE {rmse:.1f} m\nbias {bias:+.1f} m",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(LIM)
            ax.set_ylim(LIM)
            ax.set_aspect("equal")
    for ax in axes[-1]:
        ax.set_xlabel("Observed depth to water (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted depth to water (m)")
    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    fig.savefig(OUT / f"brief_scatter{suffix}.png", dpi=250)
    plt.close(fig)


def distance(states: dict, suffix: str) -> None:
    ncol = len(states)
    fig, axes = plt.subplots(
        1, ncol, figsize=(4.5 * ncol + 0.5, 3.9), sharey=True, squeeze=False
    )
    axes = axes[0]
    style = {
        "handily": dict(marker="o", mfc=BLUE, mec=BLUE, color=BLUE, ls="-"),
        "Ma": dict(marker="o", mfc="white", mec=GREY, color=GREY, ls="--"),
    }
    for ax, (state, cfg) in zip(axes, states.items()):
        s = pd.read_csv(cfg["stats"])
        bands = list(dict.fromkeys(s["band"]))
        x = np.arange(len(bands))
        for k, (pred, st) in enumerate(style.items()):
            p = s[s["predictor"] == pred].set_index("band").loc[bands]
            dx = -0.08 if k == 0 else 0.08
            ax.errorbar(
                x + dx,
                p["mad"],
                yerr=[p["mad"] - p["mad_lo"], p["mad_hi"] - p["mad"]],
                capsize=2.5,
                lw=1.2,
                ms=6,
                **st,
            )
        n = s[s["predictor"] == "handily"].set_index("band").loc[bands, "n"]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\nn = {int(v):,}" for b, v in zip(bands, n)])
        ax.set_xlabel("Distance to nearest well the map was given (km)")
        if ncol > 1:
            ax.set_title(state, loc="left", fontsize=11)
        ax.set_ylim(0, 25)
    axes[0].set_ylabel("MAD, median |error| (m)")
    handles = [
        Line2D([], [], label="handily", **style["handily"]),
        Line2D([], [], label="Ma et al. (2026)", **style["Ma"]),
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=True)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / f"brief_distance{suffix}.png", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state", choices=["NV", "NM"], help="one state only (default both)"
    )
    a = ap.parse_args()
    key = {"NV": "Nevada", "NM": "New Mexico"}
    states = {key[a.state]: STATES[key[a.state]]} if a.state else STATES
    suffix = f"_{a.state.lower()}" if a.state else ""
    OUT.mkdir(parents=True, exist_ok=True)
    slide_style()
    scatter(states, suffix)
    distance(states, suffix)
    print(
        "wrote", OUT / f"brief_scatter{suffix}.png", OUT / f"brief_distance{suffix}.png"
    )
