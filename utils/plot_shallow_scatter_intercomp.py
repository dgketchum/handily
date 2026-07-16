"""Shallow-regime scatter intercomparison (Ma / Janssen / handily) for NM.

Consumes the fair per-well CSVs written by ``score_ma_vs_nm_gwx.py``
(``--drop-well-class monitoring`` + ``--require-finite`` common footprint), so
all three products sit on an IDENTICAL well set, and renders a 1x3 grid
(NWIS/NGWMN wells; cols: products) zoomed to the shallow regime where the
FAC-REM/handily prior is designed to win. This is the shallow companion to the
full-range Nevada-style figures; it does NOT re-sample rasters (footing is
inherited from the CSVs) so it is instant and reproducible. Default row is
NWIS/NGWMN: the full non-NWIS panel is 96% nm_ose pumping wells whose
quantized, drawdown-noisy levels render as striping, not a scatter -- its error
structure lives in the depth-band tables instead. ``--panel``/``--sources``
select other subsets (e.g. ``--panel non_nwis --sources nmbgmr_amp`` for the
NM Bureau of Geology Aquifer Mapping Program wells, the cleanest source that is
independent of all three products); subset runs write tagged, non-clobbering
output filenames.

Each cell is a plain single-color scatter (size/alpha adapted to n), a dashed
1:1, and a red OLS line fit to the SHALLOW (on-scale, pred <= cap) cloud only,
so the line describes the visible points. The box reports n and the off-scale
fraction (pred > cap), then r / slope / RMSE / bias of the on-scale cloud; the
uncensored full-window panel is retained in the report JSON (``window_stats``).
``r`` and ``slope`` are range-restricted in a shallow window (attenuation hits
every product equally) -- full-depth r and the depth-banded MAD/medR/p95/floor
panel live in the report JSON and ``notes/BENCHMARK_INTERCOMPARISON.md``.

Usage:
    uv run python utils/plot_shallow_scatter_intercomp.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwx_wells import resid_stats, shallow_skill  # noqa: E402

PANEL_LABELS = {
    "nwis": "NWIS / NGWMN\n(in-sample for Ma/Janssen)",
    "non_nwis": "non-NWIS\n(out-of-sample, all three)",
}


def cell_stats(obs: np.ndarray, pred: np.ndarray, cap: float) -> dict:
    """Scatter stats on the SHALLOW (on-scale) cloud: pred <= cap only.

    The OLS fit and the box stats (r, slope, RMSE, bias) describe the points the
    reader can actually see; wells the product pushes past the window are counted
    in ``frac_pred_gt_cap`` instead of silently dragging the fit off the cloud.
    The uncensored full-window panel is kept in ``window_stats`` (report JSON
    only) so the censored on-figure numbers are never the only record.
    """
    m = np.isfinite(obs) & np.isfinite(pred)
    o_all, p_all = obs[m], pred[m]
    shown = p_all <= cap
    o, p = o_all[shown], p_all[shown]
    st = resid_stats(p, o) or {"n": 0}
    st["n_window"] = int(o_all.size)
    st["frac_pred_gt_cap"] = float((p_all > cap).mean()) if o_all.size else float("nan")
    st["corr"] = float(np.corrcoef(o, p)[0, 1]) if o.size > 2 else float("nan")
    if o.size > 2:
        slope, intercept = np.polyfit(o, p, 1)
    else:
        slope = intercept = float("nan")
    st["slope"] = float(slope)
    st["intercept"] = float(intercept)
    st["window_stats"] = resid_stats(p_all, o_all)  # uncensored, for the record
    st["pred_p1_m"] = float(np.percentile(p_all, 1)) if o_all.size else float("nan")
    st["pred_p50_m"] = float(np.median(p_all)) if o_all.size else float("nan")
    return st


def box_text(st: dict, cap: float) -> str:
    """n + off-scale fraction, then r/slope/RMSE/bias of the on-scale cloud."""
    return (
        f"n = {st['n_window']:,}   ({st['frac_pred_gt_cap'] * 100:.0f}% pred > {cap:g} m)\n"
        f"r {st['corr']:.2f}   slope {st['slope']:.2f}\n"
        f"RMSE {st['rmse_m']:.2f} m   bias {st['bias_m']:+.2f} m"
    )


def draw_cell(ax, sub, cap, *, show_ylabel, ylabel, show_title, title):
    """One scatter cell: obs vs pred (m), square, dashed 1:1 + OLS fit + box.

    Single-color dots with size/alpha adapted to n so both the ~500-point and
    the ~50k-point panels read as a point cloud rather than a wash.
    """
    n = len(sub)
    size, alpha = (16, 0.5) if n <= 5000 else (4, 0.10)
    ax.scatter(
        sub["mean_dtw"],
        sub["pred_dtw_m"],
        s=size,
        c="#2c5f8a",
        alpha=alpha,
        edgecolors="none",
        zorder=3,
    )
    st = cell_stats(sub["mean_dtw"].to_numpy(), sub["pred_dtw_m"].to_numpy(), cap)
    ax.plot([0, cap], [0, cap], color="0.3", lw=1.0, ls="--", zorder=6, label="1:1")
    if np.isfinite(st["slope"]):  # OLS on the on-scale cloud -> the reported slope
        xs = np.array([0.0, cap])
        ax.plot(
            xs,
            st["slope"] * xs + st["intercept"],
            color="#d62728",
            lw=1.5,
            zorder=7,
            label=f"OLS fit (pred ≤ {cap:g} m)",
        )
    ax.set_xlim(0, cap)
    ax.set_ylim(0, cap)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.text(
        0.04,
        0.96,
        box_text(st, cap),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        zorder=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
    )
    if show_title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel(f"{ylabel}\n\nModeled DTW (m)", fontsize=9)
    ax.set_xlabel("Observed DTW (m)", fontsize=9)
    return st


def build_grid(dfs, products, cap, out_path, panel_rows, subset_note=""):
    """Panel-row x 3-product grid of shallow scatters; obs & axes capped at `cap` m."""
    rows = len(panel_rows)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5.4 * rows + 0.8), squeeze=False)
    report = {}
    for ri, (panel_key, row_label) in enumerate(panel_rows):
        report[panel_key] = {}
        for ci, (label, tag) in enumerate(products):
            ax = axes[ri][ci]
            sub = dfs[tag]
            sub = sub[(sub["panel"] == panel_key) & (sub["mean_dtw"] <= cap)]
            report[panel_key][label] = draw_cell(
                ax,
                sub,
                cap,
                show_ylabel=(ci == 0),
                ylabel=row_label,
                show_title=(ri == 0),
                title=label,
            )
    fig.suptitle(
        f"NM shallow-regime intercomparison — observed DTW ≤ {cap:g} m "
        f"(unconfined, monitoring dropped, identical well set{subset_note})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.03 if rows == 1 else 0, 1, 0.92 if rows == 1 else 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir", default="/data/ssd2/handily/nm/regional/nm_scatter_intercomp"
    )
    ap.add_argument(
        "--products",
        default="Ma=ma_fair,Janssen V2=janssen_v2_fair,handily=handily_fair",
        help="Ordered comma-list of LABEL=tag; tag is the CSV prefix in --in-dir.",
    )
    ap.add_argument(
        "--caps",
        default="10",
        help="Comma-list of obs-DTW ceilings (m). Windows much below 10 m are "
        "off-scale-dominated (most predictions exceed the cap) and their fits are "
        "noise -- use the depth-band table for sub-band stats instead.",
    )
    ap.add_argument(
        "--panel",
        default="nwis",
        choices=sorted(PANEL_LABELS),
        help="Which exposure panel to plot (row of the fair CSVs).",
    )
    ap.add_argument(
        "--sources",
        default="",
        help="Optional comma-list of well sources to keep (e.g. nmbgmr_amp). "
        "Filtered runs write tagged, non-clobbering output filenames.",
    )
    ap.add_argument(
        "--row-label", default=None, help="Override the row label on the figure."
    )
    ap.add_argument(
        "--out-tag",
        default="",
        help="Extra output-filename tag (non-clobbering runs on the same "
        "panel/sources, e.g. 'facfp' for the FAC-REM-footprint CSVs).",
    )
    ap.add_argument("--out-dir", default=None, help="Defaults to --in-dir.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    products = [
        (p.split("=", 1)[0], p.split("=", 1)[1]) for p in args.products.split(",")
    ]
    caps = [float(c) for c in args.caps.split(",")]
    sources = [s for s in args.sources.split(",") if s]
    row_label = args.row_label or (
        f"{', '.join(sources)}\n(non-USGS, non-monitoring)"
        if sources
        else PANEL_LABELS[args.panel]
    )
    panel_rows = ((args.panel, row_label),)
    # Non-clobbering output names: any subset run gets its own tag.
    out_tag = ""
    if sources:
        out_tag += "_" + "-".join(sources)
    if args.panel != "nwis":
        out_tag += f"_{args.panel}"
    if args.out_tag:
        out_tag += f"_{args.out_tag}"
    subset_note = f", sources: {'+'.join(sources)}" if sources else ""

    dfs, full_depth_r = {}, {}
    for label, tag in products:
        df = pd.read_csv(in_dir / f"{tag}_vs_nm_gwx_wells.csv")
        if sources:
            df = df[df["source"].isin(sources)]
        dfs[tag] = df
        nn = df[df["panel"] == args.panel]
        full_depth_r[label] = float(np.corrcoef(nn["mean_dtw"], nn["pred_dtw_m"])[0, 1])

    # Guarantee the footing is actually identical (the whole point of the CSVs).
    ns = {tag: len(dfs[tag]) for _, tag in products}
    if len(set(ns.values())) != 1:
        raise SystemExit(f"CSV well counts differ -- not a common footprint: {ns}")

    report = {
        "well_set_n": next(iter(ns.values())),
        "panel": args.panel,
        "sources": sources or "all",
        f"full_depth_{args.panel}_r": full_depth_r,
        # Shallow-class P/R on the FULL (uncapped) panel -- precision is only
        # meaningful over all wells, so it is reported here, not on the cells.
        f"shallow_skill_{args.panel}": {
            label: shallow_skill(
                dfs[tag][dfs[tag]["panel"] == args.panel]["pred_dtw_m"].to_numpy(),
                dfs[tag][dfs[tag]["panel"] == args.panel]["mean_dtw"].to_numpy(),
            )
            for label, tag in products
        },
        "grids": {},
    }
    figs = []
    for cap in caps:
        out_path = out_dir / f"shallow_intercomp{out_tag}_obs_le_{cap:g}m.png"
        report["grids"][f"obs_le_{cap:g}m"] = build_grid(
            dfs, products, cap, out_path, panel_rows, subset_note
        )
        figs.append(str(out_path))

    with open(out_dir / f"shallow_intercomp{out_tag}_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("\nFigures:\n  " + "\n  ".join(figs))


if __name__ == "__main__":
    main()
