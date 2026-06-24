"""Head-to-head Ma vs Janssen V1 comparison figure on the GWX NM unconfined
wells, with a dedicated shallow (<= 50 ft) section and a depth-band analysis.

Same population/footprint as score_ma_vs_nm_gwx.py (the NWIS / non-NWIS scatter
slides): GWX unconfined + unconfined_marginal wells inside the state-clipped Ma
NM tile, observed DTW <= 500 ft, tx_twdb dropped, split into the training-
adjacent NWIS panel and the non-NWIS NM-state panel. Both products are sampled
at every well and the comparison is restricted to the COMMON footprint (both Ma
and Janssen V1 finite) so it is a true paired head-to-head.

Figure layout (per panel):
  row 1  full range (obs <= 500 ft):  Ma | Janssen V1 scatter
  row 2  SHALLOW (obs <= 50 ft):      Ma | Janssen V1 scatter   <- dedicated
  row 3  depth-band analysis:         bias vs band | MAD vs band (both overlaid)
Each scatter carries the 1:1 line, the 15 m / 49 ft phreatophyte guide, a stats
box, and a binned-median trend line (the conditional bias / regression-to-mean).

Residual = predicted - observed (positive => product too deep). Population is
"non-NWIS", not proven-independent of Ma (see gwx_wells.py).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwx_wells import GWX_INDEX, WT_CLASSES, load_window_wells, sample_raster  # noqa: E402

FT_PER_M = 1.0 / 0.3048
PHREATOPHYTE_FT = 15.0 * FT_PER_M  # ~49.2 ft
NWIS_SOURCES = {"nwis", "ngwmn"}
SHALLOW_FT = 50.0
# Finer bands in the shallow zone, coarser in the deep tail (feet).
BAND_EDGES_FT = [0, 5, 10, 25, 50, 100, 250, 500]
PROD = {
    "Ma": {"color": "#2b6cb0"},
    "Janssen V1": {"color": "#dd6b20"},
}


def stats_ft(obs, pred):
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


def binned_median(obs, pred, lim, nbins=12):
    """Median predicted within observed-depth bins (the conditional-bias curve)."""
    edges = np.linspace(0, lim, nbins + 1)
    mid, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (obs >= lo) & (obs < hi) & np.isfinite(pred)
        if m.sum() >= 5:
            mid.append((lo + hi) / 2)
            med.append(float(np.median(pred[m])))
    return np.array(mid), np.array(med)


def band_table(obs, ma, jv1):
    """Per observed-depth band (ft): n, Ma/V1 bias + MAD."""
    rows = []
    for lo, hi in zip(BAND_EDGES_FT[:-1], BAND_EDGES_FT[1:]):
        m = (obs >= lo) & (obs < hi)
        row = {"band_ft": f"{lo}-{hi}", "lo": lo, "hi": hi, "n": int(m.sum())}
        for label, pred in (("Ma", ma), ("Janssen V1", jv1)):
            s = stats_ft(obs[m], pred[m])
            row[f"{label}_bias"] = s.get("bias_ft")
            row[f"{label}_mad"] = s.get("mad_ft")
        rows.append(row)
    return rows


def _scatter(ax, obs, pred, lim, color, title):
    ax.scatter(obs, pred, s=5, c=color, alpha=0.15, edgecolors="none")
    mid, med = binned_median(obs, pred, lim)
    if mid.size:
        ax.plot(mid, med, color=color, lw=2.0, marker="o", ms=4, label="binned median")
    ax.plot([0, lim], [0, lim], color="black", lw=1.2, zorder=6, label="1:1")
    ax.axvline(PHREATOPHYTE_FT, color="#c05621", lw=1.0, ls="--", zorder=5)
    ax.axhline(PHREATOPHYTE_FT, color="#c05621", lw=1.0, ls="--", zorder=5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Observed DTW (ft)")
    ax.set_ylabel("Modeled WTD (ft)")
    s = stats_ft(obs, pred)
    ax.text(
        0.03,
        0.97,
        f"n = {s['n']:,}\nMAD = {s['mad_ft']:.1f} ft\n"
        f"bias = {s['bias_ft']:.1f}  med = {s['median_resid_ft']:.1f} ft\n"
        f"RMSE = {s['rmse_ft']:.1f} ft\nr = {s['corr']:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        zorder=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
    )
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)


def build_panel_figure(obs, ma, jv1, panel, out_path, max_obs_ft):
    fig = plt.figure(figsize=(12.5, 17))
    gs = fig.add_gridspec(3, 2, hspace=0.28, wspace=0.22)
    fig.suptitle(
        f"Ma vs Janssen V1 — NM {panel} unconfined wells "
        f"(observed DTW ≤ {int(max_obs_ft)} ft, common footprint n={obs.size:,})",
        fontsize=13,
        fontweight="bold",
    )

    # Row 1: full range.
    _scatter(
        fig.add_subplot(gs[0, 0]),
        obs,
        ma,
        max_obs_ft,
        PROD["Ma"]["color"],
        "Ma — full range",
    )
    _scatter(
        fig.add_subplot(gs[0, 1]),
        obs,
        jv1,
        max_obs_ft,
        PROD["Janssen V1"]["color"],
        "Janssen V1 — full range",
    )

    # Row 2: dedicated shallow section.
    sh = obs <= SHALLOW_FT
    _scatter(
        fig.add_subplot(gs[1, 0]),
        obs[sh],
        ma[sh],
        SHALLOW_FT,
        PROD["Ma"]["color"],
        f"Ma — SHALLOW (obs ≤ {int(SHALLOW_FT)} ft)",
    )
    _scatter(
        fig.add_subplot(gs[1, 1]),
        obs[sh],
        jv1[sh],
        SHALLOW_FT,
        PROD["Janssen V1"]["color"],
        f"Janssen V1 — SHALLOW (obs ≤ {int(SHALLOW_FT)} ft)",
    )

    # Row 3: depth-band bias and MAD.
    bands = band_table(obs, ma, jv1)
    x = np.arange(len(bands))
    labels = [b["band_ft"] for b in bands]
    ax_b = fig.add_subplot(gs[2, 0])
    ax_m = fig.add_subplot(gs[2, 1])
    for label in ("Ma", "Janssen V1"):
        c = PROD[label]["color"]
        ax_b.plot(
            x, [b[f"{label}_bias"] for b in bands], marker="o", color=c, label=label
        )
        ax_m.plot(
            x, [b[f"{label}_mad"] for b in bands], marker="o", color=c, label=label
        )
    ax_b.axhline(0, color="black", lw=0.8)
    # Shade the shallow (<=50 ft) bands on both band axes.
    shallow_x = max(i for i, b in enumerate(bands) if b["hi"] <= SHALLOW_FT) + 0.5
    for ax, ttl, yl in (
        (ax_b, "Bias (mean resid) by observed-depth band", "bias (ft)"),
        (ax_m, "MAD by observed-depth band", "MAD (ft)"),
    ):
        ax.axvspan(-0.5, shallow_x, color="#fefcbf", alpha=0.6, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("observed DTW band (ft)")
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=8)
        ax.text(
            0.02,
            0.96,
            "shallow ≤ 50 ft",
            transform=ax.transAxes,
            fontsize=7,
            color="#975a16",
            va="top",
        )
        # annotate per-band n on the MAD axis
    for i, b in enumerate(bands):
        ax_m.annotate(
            f"n={b['n']:,}",
            (i, ax_m.get_ylim()[1]),
            fontsize=6,
            ha="center",
            va="top",
            rotation=90,
            color="0.4",
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def git_commit():
    repo = Path(__file__).resolve().parents[1]
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nm-mask", default="/nas/gwx/wtd_states/wtd_new_mexico.tif")
    ap.add_argument("--ma", default="/nas/gwx/wtd_states/wtd_new_mexico.tif")
    ap.add_argument("--janssen-v1", default="/nas/gwx/janssen/V1_140.tif")
    ap.add_argument("--gwx-index", default=GWX_INDEX)
    ap.add_argument(
        "--out-dir", default="/data/ssd2/handily/nm/regional/ma_janssen_compare"
    )
    ap.add_argument("--max-obs-ft", type=float, default=500.0)
    ap.add_argument("--drop-sources", default="tx_twdb")
    ap.add_argument("--panels", default="non_nwis,nwis")
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

    wells = load_window_wells(args.gwx_index, bbox, WT_CLASSES, set(), set())
    nm = sample_raster(args.nm_mask, wells["longitude"], wells["latitude"])
    wells = wells[np.isfinite(nm)].copy()
    wells["ma"] = sample_raster(args.ma, wells["longitude"], wells["latitude"])
    wells["jv1"] = sample_raster(args.janssen_v1, wells["longitude"], wells["latitude"])
    # Common footprint: both products finite -> a true paired head-to-head.
    wells = wells[wells["ma"].notna() & wells["jv1"].notna()].copy()
    wells["obs_ft"] = wells["mean_dtw"] * FT_PER_M
    wells["ma_ft"] = wells["ma"] * FT_PER_M
    wells["jv1_ft"] = wells["jv1"] * FT_PER_M
    wells = wells[wells["obs_ft"] <= args.max_obs_ft].copy()
    drop = {s for s in args.drop_sources.split(",") if s}
    wells = wells[~wells["source"].isin(drop)].copy()
    wells["panel"] = np.where(wells["source"].isin(NWIS_SOURCES), "nwis", "non_nwis")

    report = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "population": (
                "GWX unconfined+unconfined_marginal in the Ma NM tile, obs<=500 ft, "
                "tx_twdb dropped; common footprint (Ma & Janssen V1 both finite). "
                "non-NWIS = independent NM-state sources (not proven-independent of "
                "Ma); NWIS = training-adjacent."
            ),
            "ma_raster": args.ma,
            "janssen_v1_raster": args.janssen_v1,
            "max_obs_ft": args.max_obs_ft,
            "shallow_ft": SHALLOW_FT,
            "band_edges_ft": BAND_EDGES_FT,
            "residual_convention": "pred - obs (positive => too deep)",
        },
        "panels": {},
    }

    for panel in [p for p in args.panels.split(",") if p]:
        sub = wells[wells["panel"] == panel]
        obs = sub["obs_ft"].to_numpy()
        ma = sub["ma_ft"].to_numpy()
        jv1 = sub["jv1_ft"].to_numpy()
        out_path = out_dir / f"ma_janssen_v1_compare_{panel}.png"
        bands = band_table(obs, ma, jv1)
        build_panel_figure(obs, ma, jv1, panel, out_path, args.max_obs_ft)
        sh = obs <= SHALLOW_FT
        report["panels"][panel] = {
            "figure": str(out_path),
            "n_full": int(obs.size),
            "n_shallow_le_50ft": int(sh.sum()),
            "full_range": {"Ma": stats_ft(obs, ma), "Janssen V1": stats_ft(obs, jv1)},
            "shallow_le_50ft": {
                "Ma": stats_ft(obs[sh], ma[sh]),
                "Janssen V1": stats_ft(obs[sh], jv1[sh]),
            },
            "bands": bands,
        }

    with open(out_dir / "ma_janssen_v1_compare_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("\nFigures:")
    for panel, rec in report["panels"].items():
        print(" ", rec["figure"])


if __name__ == "__main__":
    main()
