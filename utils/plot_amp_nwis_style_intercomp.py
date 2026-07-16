"""Benchmark intercomparison (Ma / Janssen V2 / handily), nwis-split figure style.

Renders the three depth-to-water (DTW, positive-down, meters) products against the
*same* NM well set in the exact figure style of ``score_ma_vs_nm_gwx.py`` — a
full-range scatter per product (per-source colors, 1:1 + phreatophyte guides)
plus two depth-stratified zoom insets (obs ≤ 250 ft and obs ≤ 50 ft). One panel
(with its two insets) per product, laid out side by side — i.e. one panel of the
original nwis-split figure, tripled across products.

Population comes from the fair per-well CSVs (unconfined-only, ``well_class ==
monitoring`` dropped, common finite footprint across all three rasters), selected
by ``--panel`` (nwis / non_nwis) and ``--sources``. Defaults reproduce the
``nmbgmr_amp`` figure (non-USGS, non-monitoring — independent of all three
products); ``--sources ""`` gives the full panel (for non_nwis: the right-side
panel of the original split figure, nm_ose included). We do NOT re-sample rasters
here; we read the CSVs, filter, and assert row-aligned equality. Non-default runs
write derived, non-clobbering output tags.

Axes/annotations are in ``--units`` (default METERS; insets obs ≤ 75 m / ≤ 15 m —
the 15 m inset is exactly the phreatophyte zone. ``--units ft`` restores the
original 250/50 ft slide style). The JSON report is always METERS with the
project depth bands. Annotations carry the full metric panel
(n / bias / median residual / MAD / RMSE / r) — never MAD alone.

Usage:
    uv run python utils/plot_amp_nwis_style_intercomp.py                # AMP
    uv run python utils/plot_amp_nwis_style_intercomp.py --sources ""   # full non-NWIS
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
# Reuse the original figure-construction primitives verbatim so the visual style
# (scatter, 1:1 + phreatophyte guides, inset squares, feet axes) is identical.
from score_ma_vs_nm_gwx import (  # noqa: E402
    PHREATOPHYTE_FT,
    ZOOMS,
    _scatter,
    panel_stats,
)
from gwx_wells import DEPTH_BANDS, resid_stats  # noqa: E402

M_PER_FT = 0.3048
# Meter analogs of the 250/50 ft insets; 15 m is exactly the phreatophyte-zone
# ceiling, so the second inset zooms the phreatophyte box.
ZOOMS_M = (75.0, 15.0)

# Fair per-well CSVs (identical well set, common finite footprint across products).
INTERCOMP_DIR = Path("/data/ssd2/handily/nm/regional/nm_scatter_intercomp")
PRODUCTS = (
    ("Ma", INTERCOMP_DIR / "ma_fair_vs_nm_gwx_wells.csv"),
    ("Janssen V2", INTERCOMP_DIR / "janssen_v2_fair_vs_nm_gwx_wells.csv"),
    ("handily", INTERCOMP_DIR / "handily_fair_vs_nm_gwx_wells.csv"),
)
# Default population: nmbgmr_amp (non-USGS, non-monitoring — independent of all
# three products). Overridable via --panel / --sources.
DEFAULT_SOURCES = "nmbgmr_amp"
POPULATION_BASE = (
    "unconfined only, well_class 'monitoring' dropped, common finite footprint "
    "across the Ma / Janssen V2 / handily rasters."
)
PANEL_DESC = {
    "nwis": "USGS NWIS/NGWMN wells (in-sample for Ma/Janssen)",
    "non_nwis": "non-NWIS NM-state wells (out-of-sample for Ma/Janssen)",
}
# Same inset placement as score_ma_vs_nm_gwx.draw_panel: obs≤250 upper-left (off
# the diagonal), obs≤50 on the main 1:1 line (upper-right).
INSET_SPECS = ((0.05, 0.42, 0.36, 0.36), (0.59, 0.59, 0.36, 0.36))


def _band_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}m" if hi < 1e9 else f"{lo:g}+m"


def _neutral_stats(st: dict) -> dict:
    """panel_stats keys are ft-named but the math is unit-agnostic; strip the
    suffix so report/annotation keys don't lie in meters mode."""
    return {k.removesuffix("_ft"): v for k, v in st.items()}


def _full_stats_text(st: dict, u: str) -> str:
    """Main-panel metric box — the full panel, never MAD alone."""
    return (
        f"n = {st['n']:,}\n"
        f"MAD = {st['mad']:.1f} {u}\n"
        f"median resid = {st['median_resid']:.1f} {u}\n"
        f"bias = {st['bias']:.1f} {u}\n"
        f"RMSE = {st['rmse']:.1f} {u}\n"
        f"r = {st['corr']:.2f}"
    )


def _inset_stats_text(st: dict, u: str) -> str:
    fmt = ".0f" if u == "ft" else ".1f"
    return (
        f"n = {st['n']:,}\n"
        f"MAD = {st['mad']:{fmt}} {u}\n"
        f"med = {st['median_resid']:{fmt}} {u}\n"
        f"bias = {st['bias']:{fmt}} {u}\n"
        f"RMSE = {st['rmse']:{fmt}} {u}\n"
        f"r = {st['corr']:.2f}"
    )


def draw_product_panel(ax, sub, product_label, main_lim, desc, units):
    """Full-range scatter + two zoom insets for one product; returns display-unit
    stats. ``main_lim`` is in display units."""
    if units == "m":
        # _scatter's column contract is ft-named but unit-agnostic — feed it the
        # meter columns under those names.
        sub = sub.assign(obs_ft=sub["mean_dtw"], pred_ft=sub["pred_dtw_m"])
    phreat = 15.0 if units == "m" else PHREATOPHYTE_FT
    _scatter(
        ax,
        sub,
        main_lim,
        f"{product_label} WTD vs {desc}",
        point_size=6,
        color_by_source=True,
        phreatophyte=phreat,
    )
    ax.set_xlabel(f"Observed depth to water ({units}, well median)")
    ax.set_ylabel(f"Modeled WTD ({units}, {product_label} raster)")

    st_all = _neutral_stats(
        panel_stats(sub["obs_ft"].to_numpy(), sub["pred_ft"].to_numpy())
    )
    ax.text(
        0.03,
        0.97,
        _full_stats_text(st_all, units),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        zorder=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
    )
    n_src = sub["source"].nunique()
    leg = ax.legend(
        loc="lower right",
        fontsize=7 if n_src <= 2 else 6,
        framealpha=0.9,
        markerscale=2.5,
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    zoom_stats = {}
    zooms = ZOOMS_M if units == "m" else ZOOMS
    active_zooms = [z for z in zooms if z < main_lim]
    for zlim, spec in zip(active_zooms, INSET_SPECS):
        zsub = sub[sub["obs_ft"] <= zlim]
        st = _neutral_stats(
            panel_stats(zsub["obs_ft"].to_numpy(), zsub["pred_ft"].to_numpy())
        )
        zoom_stats[f"obs_le_{zlim:g}{units}"] = st
        iax = ax.inset_axes(spec)
        _scatter(
            iax, zsub, zlim, "", point_size=3, color_by_source=True, phreatophyte=phreat
        )
        # The shallowest zoom is the headline (phreatophyte-zone) inset — larger text.
        shallow = zlim == min(active_zooms)
        iax.set_title(f"zoom: obs ≤ {zlim:g} {units}", fontsize=9 if shallow else 7)
        iax.tick_params(labelsize=7 if shallow else 6)
        iax.text(
            0.04,
            0.96,
            _inset_stats_text(st, units),
            transform=iax.transAxes,
            va="top",
            ha="left",
            fontsize=9 if shallow else 6,
            zorder=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
        )
    return {"annotation": st_all, "zoom": zoom_stats}


def meters_report(sub) -> dict:
    """All-depth + depth-banded residual stats in meters (gwx_wells.resid_stats)."""
    pred = sub["pred_dtw_m"].to_numpy()
    obs = sub["mean_dtw"].to_numpy()
    bands = {}
    for lo, hi in DEPTH_BANDS:
        m = (obs >= lo) & (obs < hi)
        bands[_band_label(lo, hi)] = resid_stats(pred[m], obs[m])
    return {"all_depth_m": resid_stats(pred, obs), "depth_bands_m": bands}


def load_products(panel: str, sources: list[str]) -> dict:
    """Load the fair CSVs, filter panel/sources, assert identical row-aligned set."""
    frames = {}
    for label, path in PRODUCTS:
        df = pd.read_csv(path)
        df = df[df["panel"] == panel]
        if sources:
            df = df[df["source"].isin(sources)]
        frames[label] = df.reset_index(drop=True)

    lens = {k: len(v) for k, v in frames.items()}
    if len(set(lens.values())) != 1:
        raise ValueError(f"well counts differ across products: {lens}")
    ref = next(iter(frames.values()))["mean_dtw"].to_numpy()
    for label, df in frames.items():
        if not np.allclose(df["mean_dtw"].to_numpy(), ref, equal_nan=True):
            raise ValueError(
                f"{label} well set not row-aligned to the reference (mean_dtw "
                "mismatch) — the fair CSVs must share an identical well order."
            )
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(INTERCOMP_DIR))
    ap.add_argument("--panel", default="non_nwis", choices=sorted(PANEL_DESC))
    ap.add_argument(
        "--sources",
        default=DEFAULT_SOURCES,
        help='Comma-list of well sources to keep; "" = all sources in the panel.',
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="Output filename tag; derived (non-clobbering) when omitted.",
    )
    ap.add_argument(
        "--max-obs-ft",
        type=float,
        default=500.0,
        help="Population obs ceiling (ft); the fair CSVs are already capped here. "
        "The main axis is this cap in --units.",
    )
    ap.add_argument(
        "--units",
        default="m",
        choices=("m", "ft"),
        help="Figure display units (report JSON is always meters).",
    )
    ap.add_argument(
        "--products",
        default=None,
        help="Comma-list of product labels to draw (subset of Ma / Janssen V2 / "
        "handily); default all three. Fewer panels render larger, with a "
        "derived non-clobbering tag.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [s for s in args.sources.split(",") if s]
    if args.products:
        wanted = [p.strip() for p in args.products.split(",")]
        unknown = set(wanted) - {label for label, _ in PRODUCTS}
        if unknown:
            raise SystemExit(f"unknown product labels: {sorted(unknown)}")
        selected = [(label, path) for label, path in PRODUCTS if label in wanted]
    else:
        selected = list(PRODUCTS)
    prod_slug = (
        ""
        if len(selected) == len(PRODUCTS)
        else "_" + "-".join(label.lower().replace(" ", "") for label, _ in selected)
    )
    if args.tag is not None:
        tag = args.tag
    elif args.panel == "non_nwis" and sources == ["nmbgmr_amp"] and not prod_slug:
        tag = "amp_nwis_style_intercomp"  # the original AMP figure name
    else:
        tag = (
            f"{'-'.join(sources) if sources else 'all'}_{args.panel}"
            f"{prod_slug}_style_intercomp"
        )
    if sources == ["nmbgmr_amp"]:
        desc = "nmbgmr_amp (NM, unconfined)"
        pop_line = (
            "nmbgmr_amp (non-USGS, non-monitoring — independent of all three products)"
        )
    else:
        desc = PANEL_DESC[args.panel].split(" (")[0] + " (unconfined)"
        pop_line = (", ".join(sources) + " — " if sources else "") + PANEL_DESC[
            args.panel
        ]

    # The common footprint is defined across ALL three fair CSVs — always load
    # and cross-check the full set even when drawing a subset.
    frames = load_products(args.panel, sources)
    n_wells = len(next(iter(frames.values())))
    main_lim = args.max_obs_ft * M_PER_FT if args.units == "m" else args.max_obs_ft

    n_panels = len(selected)
    # Keep the proven 3-panel width:height proportion (7.0 : 8.4) so the square
    # equal-aspect axes never swallow the x labels when panels render larger.
    panel_w = 7.0 if n_panels == 3 else 9.5
    fig_h = panel_w * 8.4 / 7.0
    fig, axes = plt.subplots(1, n_panels, figsize=(panel_w * n_panels, fig_h))
    axes = np.atleast_1d(axes)
    fig.suptitle(
        "Depth-to-water benchmark intercomparison — New Mexico\n"
        f"{pop_line}: n = {n_wells:,} unconfined wells, observed DTW ≤ "
        f"{main_lim:g} {args.units}",
        fontsize=11,
        fontweight="bold",
    )
    products_report = {}
    for ax, (label, path) in zip(axes, selected):
        sub = frames[label]
        fig_stats = draw_product_panel(ax, sub, label, main_lim, desc, args.units)
        products_report[label] = {
            "pred_csv": str(path),
            **meters_report(sub),
            "figure_annotation": fig_stats,
        }
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))

    fig_path = out_dir / f"{tag}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    report = {
        "population": f"{pop_line}; {POPULATION_BASE}",
        "sources": sources or "all",
        "panel": args.panel,
        "n_wells": int(n_wells),
        "max_obs_ft": args.max_obs_ft,
        "depth_bands_m": [list(b) for b in DEPTH_BANDS],
        "units": {
            "figure_annotation": "meters" if args.units == "m" else "feet",
            "report": "meters (all-depth + depth-banded)",
        },
        "figure": str(fig_path),
        "products": products_report,
    }
    report_path = out_dir / f"{tag}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nFigure: {fig_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
