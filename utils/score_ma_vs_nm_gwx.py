"""Reproduce the Nevada "Shallow Groundwater Subsidies" slide for New Mexico.

Compares a national WTD product (``--pred``: Ma by default, or Janssen) against
GWX *unconfined* wells in New Mexico, split into two panels emulating
``notes/nwi_ma_comparison_Nevada.png``:

    left  : product vs USGS NWIS wells        (product's training-adjacent source)
    right : product vs non-NWIS NM state wells (independent: nm_ose, nmbgmr_amp, ...)

Both quantities are depth-to-water below land surface (positive-down); the figure
is drawn in feet to match the Nevada slide, with the phreatophyte zone marked at
15 m / 49 ft. Residual = predicted - observed (well); negative = product too
shallow (the deep-arid-basin failure mode).

POPULATION (name it; do NOT conflate with score_janssen_vs_ma.py): confinement
here is the GWX national v2 classifier's ``unconfined`` / ``unconfined_marginal``
classes, so nm_ose IS counted unconfined and the non-NWIS panel runs ~34k wells
(~33.5k nm_ose). The consolidated-layer pipeline (build_nm_validation_layer.py /
score_janssen_vs_ma.py) instead HAND-ASSIGNS confinement and holds OSE driller
logs out as 'unknown' screening, headlining only ~839 study/monitoring wells —
a different scientific claim. Never compare the two n's / MADs without naming the
population. Independence is "non-NWIS", not proven-independent of Ma (see
gwx_wells.py).

Confinement: only ``unconfined`` / ``unconfined_marginal`` enter the figure
(depth-to-water = depth to the *unconfined* table). The NM window is defined by
the state-clipped Ma raster footprint (``--nm-mask``), so out-of-state wells
(AZ/TX/CO) drop out — necessary because the continental Janssen raster is not
state-clipped. Using the same mask keeps the well set identical across products.

Usage:
    # Ma (defaults)
    uv run python utils/score_ma_vs_nm_gwx.py
    # Janssen V1
    uv run python utils/score_ma_vs_nm_gwx.py \
        --pred /nas/gwx/janssen/V1_140.tif --product-label "Janssen V1" \
        --tag janssen_v1 --out-dir /data/ssd2/handily/nm/regional/janssen_nm_gwx
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
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwx_wells import GWX_INDEX, WT_CLASSES, load_window_wells, sample_raster  # noqa: E402

FT_PER_M = 1.0 / 0.3048
PHREATOPHYTE_FT = 15.0 * FT_PER_M  # 15 m saturated-zone reach -> ~49.2 ft
ZOOMS = (250.0, 50.0)  # inset obs-depth ceilings (ft)
# USGS-affiliated sources (Ma/Janssen training) -> "NWIS" side; everything else
# is the independent non-NWIS NM-state population.
NWIS_SOURCES = {"nwis", "ngwmn"}


def panel_stats(obs: np.ndarray, pred: np.ndarray) -> dict:
    """n / bias / RMSE / Pearson r on a finite obs,pred pair (feet)."""
    m = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[m], pred[m]
    if o.size == 0:
        return {"n": 0}
    r = p - o
    return {
        "n": int(o.size),
        "bias_ft": float(np.mean(r)),
        "median_resid_ft": float(np.median(r)),
        "mad_ft": float(np.median(np.abs(r))),
        "rmse_ft": float(np.sqrt(np.mean(r**2))),
        "corr": float(np.corrcoef(o, p)[0, 1]) if o.size > 2 else float("nan"),
    }


# Per-source point styling for the non-NWIS panel. nm_ose dominates (~33k wells)
# so it is drawn first (bottom layer) as a nearly transparent red; the sparse
# state/federal sources draw on top in solid colors.
SOURCE_STYLE = {
    "nm_ose": ("#e53e3e", 0.10),
    "nmbgmr_amp": ("#2b6cb0", 0.75),
    "doe_gems": ("#38a169", 0.85),
    "nm_sta": ("#6b46c1", 0.85),
    "nmose_isc_seven_rivers": ("#dd6b20", 0.95),
    "tx_twdb": ("#718096", 0.85),
    "nwis": ("#2b6cb0", 0.25),
    "ngwmn": ("#2b6cb0", 0.25),
}
DEFAULT_STYLE = ("#2b6cb0", 0.30)

# Readable display names for the GWX source codes (used in legends / notes).
SOURCE_NAMES = {
    "nm_ose": "NM OSE",
    "nmbgmr_amp": "NMBGMR AMP",
    "doe_gems": "DOE GEMS",
    "nm_sta": "NM SensorThings",
    "nmose_isc_seven_rivers": "OSE-ISC Seven Rivers",
    "tx_twdb": "TX TWDB",
    "nwis": "USGS NWIS",
    "ngwmn": "USGS NGWMN",
}


def src_name(code: str) -> str:
    return SOURCE_NAMES.get(code, code)


def _scatter(ax, sub, lim, title, *, point_size, color_by_source):
    """Scatter ``sub`` (obs_ft/pred_ft/source) with 1:1 + phreatophyte guides.

    With ``color_by_source`` the dominant source is drawn first (bottom layer)
    and sparse sources on top, each carrying a legend label.
    """
    if color_by_source:
        for src in sub["source"].value_counts().index:  # dominant first -> bottom
            d = sub[sub["source"] == src]
            color, alpha = SOURCE_STYLE.get(src, DEFAULT_STYLE)
            ax.scatter(
                d["obs_ft"],
                d["pred_ft"],
                s=point_size,
                c=color,
                alpha=alpha,
                edgecolors="none",
                label=f"{src_name(src)} (n={len(d):,})",
            )
    else:
        ax.scatter(
            sub["obs_ft"],
            sub["pred_ft"],
            s=point_size,
            c=DEFAULT_STYLE[0],
            alpha=DEFAULT_STYLE[1],
            edgecolors="none",
        )
    ax.plot([0, lim], [0, lim], color="black", lw=1.2, zorder=6, label="1:1")
    ax.axvline(PHREATOPHYTE_FT, color="#c05621", lw=1.0, ls="--", zorder=5)
    ax.axhline(
        PHREATOPHYTE_FT,
        color="#c05621",
        lw=1.0,
        ls="--",
        zorder=5,
        label="phreatophyte zone (<15 m / 49 ft)",
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title, fontsize=10)


def _stats_text(st: dict, source_note: str) -> str:
    return (
        f"n = {st['n']:,}\n"
        f"bias = {st['bias_ft']:.1f} ft   RMSE = {st['rmse_ft']:.1f} ft\n"
        f"r = {st['corr']:.2f}\n"
        f"({source_note})"
    )


def draw_panel(ax, sub, title, source_note, main_lim, color_by_source, product_label):
    """Main scatter + zoom insets whose 1:1 lines lie on the main 1:1 line."""
    lim = main_lim
    obs = sub["obs_ft"].to_numpy()
    pred = sub["pred_ft"].to_numpy()
    _scatter(ax, sub, lim, title, point_size=6, color_by_source=color_by_source)
    ax.set_xlabel("Observed depth to water (ft, well median)")
    ax.set_ylabel(f"Modeled WTD (ft, {product_label} raster)")

    st_all = panel_stats(obs, pred)
    ax.text(
        0.03,
        0.97,
        _stats_text(st_all, source_note),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        zorder=10,  # keep the box above the phreatophyte guide lines
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
    )
    leg = ax.legend(
        loc="lower right",
        fontsize=6 if color_by_source else 7,
        framealpha=0.9,
        markerscale=2.5,
    )
    for lh in leg.legend_handles:  # opaque legend markers despite transparent points
        lh.set_alpha(1.0)

    # Inset order follows ZOOMS = (250, 50). The obs<=50 inset (second) is a
    # square with opposite corners on the axes diagonal (t,t)->(t+s,t+s); since
    # the main axes is square (equal aspect, equal limits) its own 1:1 line lies
    # exactly on the main 1:1 line (upper-right). The obs<=250 inset sits in the
    # upper-left, deliberately off the diagonal.
    inset_specs = [(0.05, 0.42, 0.36, 0.36), (0.59, 0.59, 0.36, 0.36)]
    zoom_stats = {}
    active_zooms = [z for z in ZOOMS if z < lim]
    for zlim, spec in zip(active_zooms, inset_specs):
        zsub = sub[sub["obs_ft"] <= zlim]
        st = panel_stats(zsub["obs_ft"].to_numpy(), zsub["pred_ft"].to_numpy())
        zoom_stats[f"obs_le_{int(zlim)}ft"] = st
        iax = ax.inset_axes(spec)
        _scatter(iax, zsub, zlim, "", point_size=3, color_by_source=color_by_source)
        iax.set_title(f"zoom: obs ≤ {int(zlim)} ft", fontsize=7)
        iax.tick_params(labelsize=6)
        iax.text(
            0.04,
            0.96,
            f"n = {st['n']:,}\nbias = {st['bias_ft']:.1f}\n"
            f"RMSE = {st['rmse_ft']:.0f}\nr = {st['corr']:.2f}",
            transform=iax.transAxes,
            va="top",
            ha="left",
            fontsize=6,
            zorder=10,  # keep the box above the phreatophyte guide lines
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
        )
    return {"all": st_all, "zooms": zoom_stats, "lim_ft": lim}


def build_figure(wells, non_nwis_note, out_path, max_obs_ft, suptitle, product_label):
    """Two-panel NWIS / non-NWIS figure for `wells`; returns its report dict."""
    panels = {
        "nwis": (
            f"{product_label} WTD vs USGS NWIS wells (NM, unconfined)",
            f"USGS NWIS / NGWMN — {product_label} training-adjacent",
        ),
        "non_nwis": (
            f"{product_label} WTD vs non-NWIS NM-state wells (unconfined)",
            non_nwis_note,
        ),
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    panels_report, src_breakdown = {}, {}
    for ax, (key, (title, note)) in zip(axes, panels.items()):
        sub = wells[wells["panel"] == key]
        panels_report[key] = draw_panel(
            ax,
            sub,
            title,
            note,
            max_obs_ft,
            color_by_source=(key == "non_nwis"),
            product_label=product_label,
        )
        src_breakdown[key] = sub["source"].value_counts().to_dict()
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "figure": str(out_path),
        "panels": panels_report,
        "source_breakdown": src_breakdown,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--nm-mask",
        default="/nas/gwx/wtd_states/wtd_new_mexico.tif",
        help="State-clipped raster defining the in-NM footprint (Ma NM tile).",
    )
    ap.add_argument(
        "--pred",
        default="/nas/gwx/wtd_states/wtd_new_mexico.tif",
        help="WTD product raster plotted on the y-axis (default: Ma NM tile). "
        "For Janssen: /nas/gwx/janssen/V1_140.tif",
    )
    ap.add_argument("--product-label", default="Ma", help="Product name in titles.")
    ap.add_argument("--tag", default="ma", help="Output filename prefix.")
    ap.add_argument("--gwx-index", default=GWX_INDEX)
    ap.add_argument("--out-dir", default="/data/ssd2/handily/nm/regional/ma_nm_gwx")
    ap.add_argument(
        "--max-obs-ft",
        type=float,
        default=500.0,
        help="Restrict to wells with observed DTW <= this (ft); sets the main axis.",
    )
    ap.add_argument(
        "--drop-sources",
        default="tx_twdb",
        help="Comma-list of sources to drop entirely (default: tx_twdb).",
    )
    # NM lon/lat extent (Ma raster footprint); Ma-finite filter does the real clip.
    ap.add_argument("--lon-min", type=float, default=-110.2)
    ap.add_argument("--lon-max", type=float, default=-102.4)
    ap.add_argument("--lat-min", type=float, default=31.1)
    ap.add_argument("--lat-max", type=float, default=37.6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xs, ys = tr.transform(
        [args.lon_min, args.lon_max, args.lon_min, args.lon_max],
        [args.lat_min, args.lat_min, args.lat_max, args.lat_max],
    )
    bbox = (min(xs), min(ys), max(xs), max(ys))

    # All sources, unconfined+marginal, dtw present, within the NM bbox.
    wells = load_window_wells(args.gwx_index, bbox, WT_CLASSES, set(), set())
    # The Ma NM tile is state-clipped, so a finite sample defines "in NM"; the
    # continental Janssen raster has no such clip, hence the separate mask.
    nm = sample_raster(args.nm_mask, wells["longitude"], wells["latitude"])
    wells = wells[np.isfinite(nm)].copy()
    wells["pred_dtw_m"] = sample_raster(
        args.pred, wells["longitude"], wells["latitude"]
    )
    wells = wells[wells["pred_dtw_m"].notna()].copy()  # product coverage
    wells["obs_ft"] = wells["mean_dtw"] * FT_PER_M
    wells["pred_ft"] = wells["pred_dtw_m"] * FT_PER_M
    wells["resid_ft"] = wells["pred_ft"] - wells["obs_ft"]
    wells = wells[wells["obs_ft"] <= args.max_obs_ft].copy()  # comparison domain
    drop = {s for s in args.drop_sources.split(",") if s}
    if drop:
        wells = wells[~wells["source"].isin(drop)].copy()
    wells["panel"] = np.where(wells["source"].isin(NWIS_SOURCES), "nwis", "non_nwis")

    base = (
        "Shallow Groundwater Subsidies — New Mexico "
        f"({args.product_label} WTD vs GWX unconfined wells, "
        f"observed DTW ≤ {int(args.max_obs_ft)} ft"
    )
    # Two figures: the non-NWIS panel with vs without the dominant nm_ose source
    # (left NWIS panel is identical in both).
    fig_with = out_dir / f"{args.tag}_vs_nm_gwx_nwis_split_with_nmose.png"
    fig_without = out_dir / f"{args.tag}_vs_nm_gwx_nwis_split_without_nmose.png"
    report = {
        "product_label": args.product_label,
        "population": (
            "GWX national v2 classifier, confinement_class in "
            "{unconfined, unconfined_marginal}; nm_ose IS counted unconfined "
            "(~33.5k of the non-NWIS panel). DISTINCT from the hand-assigned "
            "consolidated layer (build_nm_validation_layer.py), which holds OSE "
            "out as 'unknown' screening (~839-well headline). 'non-NWIS' here is "
            "not proven-independent of Ma."
        ),
        "pred_raster": args.pred,
        "nm_mask_raster": args.nm_mask,
        "gwx_index": args.gwx_index,
        "dropped_sources": sorted(drop),
        "variants": {
            "with_nmose": build_figure(
                wells,
                "independent: NM OSE + NM-state monitoring",
                fig_with,
                args.max_obs_ft,
                base + ", with NM OSE)",
                args.product_label,
            ),
            "without_nmose": build_figure(
                wells[wells["source"] != "nm_ose"],
                "independent: NM-state monitoring (no NM OSE)",
                fig_without,
                args.max_obs_ft,
                base + ", no NM OSE)",
                args.product_label,
            ),
        },
    }

    keep = [
        "source",
        "panel",
        "confinement_class",
        "confinement_source",
        "well_class",
        "obs_count",
        "mean_dtw",
        "pred_dtw_m",
        "obs_ft",
        "pred_ft",
        "resid_ft",
        "longitude",
        "latitude",
        "geometry",
    ]
    wells[keep].to_file(
        out_dir / f"{args.tag}_vs_nm_gwx_wells.fgb", driver="FlatGeobuf"
    )
    wells[[c for c in keep if c != "geometry"]].to_csv(
        out_dir / f"{args.tag}_vs_nm_gwx_wells.csv", index=False
    )
    with open(out_dir / f"{args.tag}_vs_nm_gwx_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nFigures:\n  {fig_with}\n  {fig_without}")


if __name__ == "__main__":
    main()
