"""Depth-banded grouped-bar intercomparison (MAD + median residual) for NM.

The presentation companion to the scatter grids: reads the fair per-well CSVs
(identical well set across products), computes the UNCENSORED per-band metric
panel (`gwx_wells.resid_stats` over `DEPTH_BANDS`), and renders two stacked bar
panels -- MAD by band (typical error magnitude) and median residual by band
(systematic direction; the benchmarks' one-sided too-deep shallow push vs the
centered handily/FAC-REM errors). Bars are labeled with their values, band
labels carry per-band n, and every number matches the report JSON / doc tables
exactly (no on-scale censoring anywhere). Figure display units are ``--units``
(default m, the project's canonical unit; ``--units ft`` scales the bars and
adds dual ft/m band labels); the report JSON is always meters.

Usage (the two update figures):
    uv run python utils/plot_banded_bar_intercomp.py --sources nmbgmr_amp --out-tag amp
    uv run python utils/plot_banded_bar_intercomp.py \
        --products "Ma=ma_facfp,Janssen V2=janssen_v2_facfp,FAC-REM=fac_rem_facfp,handily=handily_facfp" \
        --sources nmbgmr_amp --out-tag amp_facfp \
        --note "FAC-REM footprint (near-network wells only)"
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
from gwx_wells import DEPTH_BANDS, resid_stats  # noqa: E402

FT_PER_M = 1.0 / 0.3048
PRODUCT_COLORS = {
    "Ma": "#8c8c8c",
    "Janssen V2": "#e08214",
    "handily": "#2c5f8a",
    "FAC-REM": "#35978f",
}


def band_label(lo: float, hi: float, units: str) -> str:
    """Depth bands are DEFINED in meters (project DEPTH_BANDS); feet runs get a
    dual label so the band edges stay recognizably the canonical ones."""
    m = f"{lo:g}–{hi:g} m" if hi < 1e6 else f"{lo:g}+ m"
    if units == "m":
        return m
    lo_ft, hi_ft = lo * FT_PER_M, hi * FT_PER_M
    ft = f"{lo_ft:.0f}–{hi_ft:.0f} ft" if hi < 1e6 else f"{lo_ft:.0f}+ ft"
    return f"{ft}\n({m})"


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
        "--sources", default="", help="Optional comma-list of well sources to keep."
    )
    ap.add_argument(
        "--out-tag", required=True, help="Output filename tag (non-clobbering runs)."
    )
    ap.add_argument(
        "--note", default="", help="Extra footprint/context note for the title."
    )
    ap.add_argument(
        "--units",
        default="m",
        choices=("m", "ft"),
        help="Figure display units (bands stay defined in meters; the report "
        "JSON is always meters).",
    )
    ap.add_argument("--out-dir", default=None, help="Defaults to --in-dir.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    products = [
        (p.split("=", 1)[0], p.split("=", 1)[1]) for p in args.products.split(",")
    ]
    sources = [s for s in args.sources.split(",") if s]

    dfs = {}
    for _, tag in products:
        df = pd.read_csv(in_dir / f"{tag}_vs_nm_gwx_wells.csv")
        if sources:
            df = df[df["source"].isin(sources)]
        dfs[tag] = df
    ns = {tag: len(dfs[tag]) for _, tag in products}
    if len(set(ns.values())) != 1:
        raise SystemExit(f"CSV well counts differ -- not a common footprint: {ns}")
    n_wells = next(iter(ns.values()))

    report = {
        "sources": sources or "all",
        "n_wells": n_wells,
        "products": [label for label, _ in products],
        "figure_units": args.units,
        "report_units": "meters (canonical; band keys are project DEPTH_BANDS)",
        "bands": {},
    }
    scale = FT_PER_M if args.units == "ft" else 1.0
    band_labels = []
    mad = {label: [] for label, _ in products}
    medr = {label: [] for label, _ in products}
    for lo, hi in DEPTH_BANDS:
        key = band_label(lo, hi, "m")  # report keys stay canonical meters
        report["bands"][key] = {}
        n_band = None
        for label, tag in products:
            d = dfs[tag]
            sub = d[(d["mean_dtw"] >= lo) & (d["mean_dtw"] < hi)]
            st = resid_stats(sub["pred_dtw_m"].to_numpy(), sub["mean_dtw"].to_numpy())
            report["bands"][key][label] = st
            mad[label].append(st["mad_m"] * scale)
            medr[label].append(st["median_residual_m"] * scale)
            n_band = st["n"]
        band_labels.append(f"{band_label(lo, hi, args.units)}\n(n={n_band:,})")

    x = np.arange(len(DEPTH_BANDS))
    width = 0.8 / len(products)
    fig, (ax_mad, ax_medr) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, height_ratios=(1, 1)
    )
    for pi, (label, _) in enumerate(products):
        off = (pi - (len(products) - 1) / 2) * width
        color = PRODUCT_COLORS.get(label, f"C{pi}")
        b1 = ax_mad.bar(x + off, mad[label], width, color=color, label=label)
        b2 = ax_medr.bar(x + off, medr[label], width, color=color, label=label)
        fmt = "%.0f" if args.units == "ft" else "%.1f"
        ax_mad.bar_label(b1, fmt=fmt, fontsize=7, padding=1)
        ax_medr.bar_label(b2, fmt="%+" + fmt[1:], fontsize=7, padding=1)

    u = args.units
    ax_mad.set_ylabel(f"MAD ({u})\ntypical error magnitude", fontsize=10)
    ax_medr.set_ylabel(f"median residual ({u})\npred − obs; + = too deep", fontsize=10)
    ax_medr.axhline(0.0, color="0.2", lw=1.0)
    ax_medr.set_xticks(x, band_labels, fontsize=9)
    ax_medr.set_xlabel("observed DTW band (well median)", fontsize=10)
    ax_mad.legend(loc="upper left", fontsize=9, framealpha=0.9)
    for ax in (ax_mad, ax_medr):
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.set_axisbelow(True)
        ax.margins(y=0.15)
    note = f"\n{args.note}" if args.note else ""
    src = "+".join(sources) if sources else "all sources"
    fig.suptitle(
        f"NM depth-banded intercomparison — {src}, n = {n_wells:,} unconfined wells"
        f" (identical footprint, uncensored){note}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = out_dir / f"banded_bars_{args.out_tag}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    with open(out_dir / f"banded_bars_{args.out_tag}_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nFigure: {out_png}")


if __name__ == "__main__":
    main()
