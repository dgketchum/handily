"""Figure 7 — held-out error against distance to the nearest assimilated source.

The deployed map reads the admitted agency wells as observations, and the
held-out half of the same panel scores it. Because the two halves are one
seeded coin flip apart on the ground (Figure 5), the scored error is a function
of how far a target sits from the nearest source: this figure plots that
function, band by band, for the shipped handily parity render and for the Ma
benchmark on the same targets.

Six panels on one shared distance axis: (a) targets per band, (b) MAD,
(c) RMSE, (d) median residual, (e) mean residual (bias), (f) p95 |residual|.
Residual r = predicted - observed depth to water (m; positive = map too deep).
CI95 bars are a paired block bootstrap over the state's hydrographic basins,
the same resampling the statewide scorer uses.

Every number is read from the statewide scorer's sampled-site table
(``sites_sampled.parquet``), and the per-band statistics are re-derived here on
the scorer's own distance bands and asserted equal to its ``panel_dist.csv``
rows before the figure's finer bands are drawn.

Written to the house figure rules in ``~/code/style/FIGURE_STYLE_GUIDE.md``:
no gridlines, no descriptive panel titles (the findings live in the caption
sidecar), one framed legend placed with ``loc=`` only, type sizes left to the
shared mplstyle, and a two-series encoding carried by fill and line style
(filled black solid vs open grey dashed) so it survives grayscale.

Run::

    NWI_STATE=NM uv run python figures/nwi/fig06_error_vs_distance.py

Writes ``fig06_error_vs_distance.{pdf,png}`` plus a ``*_caption.txt`` sidecar
to ``/data/ssd2/handily/figures/<project>/`` (``nwi`` for NV, ``nmose`` for NM).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from nwi_common import (
    CFG,
    EVAL_DIR,
    FIG_DIR,
    MM,
    RENDER_ARM,
    WIDTH_2COL,
    outside_letter,
    save,
    use_style,
)

#: Figure bands on distance to the nearest ADMITTED agency source, metres. The
#: first band is the near-duplicate band (the site dedup was a 1 m round).
DIST_EDGES = (0.0, 50.0, 300.0, 1000.0, 5000.0, np.inf)
DIST_LABELS = ("<0.05", "0.05–0.3", "0.3–1", "1–5", "≥5")
#: The scorer's own bands (km), used only for the reproduction check.
SRC_EDGES = (0.0, 0.3, 1.0, 5.0, 20.0, np.inf)
SRC_LABELS = ("0-0.3 km", "0.3-1 km", "1-5 km", "5-20 km", "20+ km")

#: Bootstrap of record: the scorer's settings.
N_BOOT = 2000
SEED = 0
#: Tolerance for the reproduction check (m); the scorer's csv is float64.
TOL_M = 1e-6

#: The two predictors, named as the scorer names them.
PRED = {
    "handily": {"col": "parity__dtw", "csv": "parity raster"},
    "Ma": {"col": "ma_resampled_m", "csv": "Ma raster (resampled here)"},
}
STYLE = {
    "handily": {
        "marker": "o",
        "mfc": "black",
        "mec": "black",
        "ls": "-",
        "color": "black",
    },
    "Ma": {
        "marker": "o",
        "mfc": "white",
        "mec": "#6e6e6e",
        "ls": "--",
        "color": "#6e6e6e",
    },
}
#: Horizontal nudge of the two series about each band centre (band units).
DODGE = 0.10

#: (panel letter, statistic key, y label)
PANELS = (
    ("a", "n", "Held-out targets (n)"),
    ("b", "mad", "MAD, median |r| (m)"),
    ("c", "rmse", "RMSE (m)"),
    ("d", "median", "Median residual r (m)"),
    ("e", "bias", "Mean residual, bias (m)"),
    ("f", "p95", "p95 |r| (m)"),
)
ZERO_LINE = {"median", "bias"}


# --------------------------------------------------------------------------
# Statistics (the scorer's definitions, verbatim)
# --------------------------------------------------------------------------


class BlockBoot:
    """Paired block bootstrap over basins; one resample set reused everywhere.

    Copied verbatim from ``utils/score_nv_shallow_panel.py`` (importing it would
    pull rasterio and the raster pass into a figure script): ``rng =
    default_rng(seed)``; one draw of ``len(blocks)`` block ids with replacement
    per resample; ``ci`` is the 2.5/97.5 percentile of the statistic over the
    finite resamples, NaN if fewer than half are finite.
    """

    def __init__(self, blocks: np.ndarray, n_boot: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        ug = np.unique(blocks)
        gidx = [np.where(blocks == g)[0] for g in ug]
        self.n_blocks = len(ug)
        self.samples = [
            np.concatenate(
                [gidx[j] for j in rng.choice(len(ug), len(ug), replace=True)]
            )
            for _ in range(n_boot)
        ]

    def ci(self, stat, mask: np.ndarray) -> tuple[float, float]:
        vals = []
        for s in self.samples:
            idx = s[mask[s]]
            vals.append(stat(idx) if idx.size else np.nan)
        vals = np.asarray(vals, dtype="float64")
        ok = np.isfinite(vals)
        if ok.sum() < len(vals) * 0.5:
            return np.nan, np.nan
        return float(np.percentile(vals[ok], 2.5)), float(np.percentile(vals[ok], 97.5))


STATS = {
    "n": lambda r: float(r.size),
    "mad": lambda r: float(np.median(np.abs(r))),
    "rmse": lambda r: float(np.sqrt(np.mean(r**2))),
    "median": lambda r: float(np.median(r)),
    "bias": lambda r: float(np.mean(r)),
    "p95": lambda r: float(np.percentile(np.abs(r), 95)),
}


def band_masks(v: np.ndarray, edges, labels) -> dict[str, np.ndarray]:
    return {
        lab: (v >= lo) & (v < hi) for lab, lo, hi in zip(labels, edges[:-1], edges[1:])
    }


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_targets() -> pd.DataFrame:
    """Admission-eligible held-out targets with finite obs and both predictions."""
    s = pd.read_parquet(EVAL_DIR / "sites_sampled.parquet")
    elig = s["is_admission_eligible"].to_numpy(bool)
    fin = np.isfinite(s["obs_dtw_m"].to_numpy("float64"))
    for p in PRED.values():
        fin &= np.isfinite(s[p["col"]].to_numpy("float64"))
    out = s[elig & fin].reset_index(drop=True)
    if not np.isfinite(out["dist_src_km"]).all():
        raise ValueError("non-finite dist_src_km on a scored target")
    return out, int(elig.sum()), int((elig & ~fin).sum())


def verify_against_scorer(t: pd.DataFrame, obs: np.ndarray, res: dict) -> None:
    """Per-band MAD/bias/RMSE/median/p95 must equal the scorer's csv rows."""
    pdist = pd.read_csv(EVAL_DIR / "panel_dist.csv")
    pdist = pdist[pdist["set"] == "A_primary_eligible"]
    masks = band_masks(t["dist_src_km"].to_numpy("float64"), SRC_EDGES, SRC_LABELS)
    checked = 0
    for name, p in PRED.items():
        rows = pdist[pdist["predictor"] == p["csv"]].set_index("stratum")
        for lab, m in masks.items():
            if lab not in rows.index:
                raise ValueError(f"scorer csv lacks band {lab!r} for {p['csv']!r}")
            r = res[name][m]
            want = rows.loc[lab]
            got = {
                "n": r.size,
                "MAD_m": STATS["mad"](r),
                "median_resid_m": STATS["median"](r),
                "bias_m": STATS["bias"](r),
                "RMSE_m": STATS["rmse"](r),
                "p95_abs_resid_m": STATS["p95"](r),
            }
            if int(want["n"]) != got["n"]:
                raise ValueError(
                    f"{p['csv']} {lab}: n {got['n']} here vs {int(want['n'])} in csv"
                )
            for k in ("MAD_m", "median_resid_m", "bias_m", "RMSE_m", "p95_abs_resid_m"):
                if abs(float(want[k]) - got[k]) > TOL_M:
                    raise ValueError(
                        f"{p['csv']} {lab} {k}: {got[k]:.6f} here vs "
                        f"{float(want[k]):.6f} in {EVAL_DIR / 'panel_dist.csv'}"
                    )
            checked += 1
    if checked != len(PRED) * len(SRC_LABELS):
        raise ValueError("reproduction check did not cover every band")


def compute(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-band statistics with CI95 for both predictors, on the figure bands."""
    obs = t["obs_dtw_m"].to_numpy("float64")
    res = {k: t[p["col"]].to_numpy("float64") - obs for k, p in PRED.items()}
    verify_against_scorer(t, obs, res)

    blocks = t[CFG["block_col"]].fillna("__null__").astype(str).to_numpy()
    boot = BlockBoot(blocks, N_BOOT, SEED)
    masks = band_masks(
        t["dist_src_km"].to_numpy("float64") * 1000.0, DIST_EDGES, DIST_LABELS
    )
    if sum(int(m.sum()) for m in masks.values()) != len(t):
        raise ValueError("distance bands do not partition the targets")
    rows = []
    for name in PRED:
        r = res[name]
        for lab, m in masks.items():
            row = {"predictor": name, "band": lab, "n": int(m.sum())}
            for key, fn in STATS.items():
                if key == "n":
                    continue
                row[key] = fn(r[m])
                lo, hi = boot.ci(lambda i, fn=fn: fn(r[i]), m)
                row[f"{key}_lo"], row[f"{key}_hi"] = lo, hi
            rows.append(row)
    meta = {
        "n_blocks": boot.n_blocks,
        "n_null_block": int(t[CFG["block_col"]].isna().sum()),
        "res": res,
    }
    return pd.DataFrame(rows), meta


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def draw(stats: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(WIDTH_2COL, 105 * MM),
        gridspec_kw={"wspace": 0.42, "hspace": 0.38},
    )
    x = np.arange(len(DIST_LABELS), dtype=float)
    for ax, (letter, key, ylab) in zip(axes.ravel(), PANELS):
        for j, name in enumerate(PRED):
            d = (
                stats[stats["predictor"] == name]
                .set_index("band")
                .loc[list(DIST_LABELS)]
            )
            st = STYLE[name]
            xj = x + (j - 0.5) * 2 * DODGE
            if key == "n":
                if name != "handily":
                    continue  # the same targets score both predictors
                ax.bar(
                    x,
                    d["n"].to_numpy(float),
                    width=0.62,
                    color="#bdbdbd",
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=2,
                )
                continue
            y = d[key].to_numpy(float)
            lo = d[f"{key}_lo"].to_numpy(float)
            hi = d[f"{key}_hi"].to_numpy(float)
            ax.errorbar(
                xj,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                fmt="none",
                ecolor=st["color"],
                elinewidth=0.6,
                capsize=1.5,
                capthick=0.6,
                zorder=3,
            )
            ax.plot(
                xj,
                y,
                marker=st["marker"],
                ms=3.2,
                mfc=st["mfc"],
                mec=st["mec"],
                mew=0.6,
                ls=st["ls"],
                lw=0.8,
                color=st["color"],
                zorder=4,
            )
        if key in ZERO_LINE:
            ax.axhline(0.0, color="black", lw=0.5, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(DIST_LABELS)
        ax.set_xlim(-0.6, len(DIST_LABELS) - 0.4)
        ax.set_ylabel(ylab)
        if letter in ("d", "e", "f"):
            ax.set_xlabel("Distance to nearest source (km)")
        if key == "n":
            top = stats.loc[stats["predictor"] == "handily", "n"].max()
            step = next(s for s in (500, 1000, 2000, 5000) if top / s <= 5)
            ax.yaxis.set_major_locator(MultipleLocator(step))
        outside_letter(ax, letter, x=-0.30)

    handles = [
        Line2D(
            [],
            [],
            marker=STYLE[k]["marker"],
            ms=3.2,
            mfc=STYLE[k]["mfc"],
            mec=STYLE[k]["mec"],
            mew=0.6,
            ls=STYLE[k]["ls"],
            lw=0.8,
            color=STYLE[k]["color"],
            label=lab,
        )
        for k, lab in (("handily", "handily (parity render)"), ("Ma", "Ma et al."))
    ]
    axes[0, 1].legend(handles=handles, loc="upper left")
    return fig


# --------------------------------------------------------------------------
# Caption
# --------------------------------------------------------------------------


def _caption(stats: pd.DataFrame, meta: dict, n_elig: int, n_dropped: int) -> str:
    ag, name = CFG["agency"], CFG["name"]
    h = stats[stats["predictor"] == "handily"].set_index("band")
    m = stats[stats["predictor"] == "Ma"].set_index("band")
    n_tot = int(h["n"].sum())
    per_band = "; ".join(
        f"{lab} km: n {int(h.loc[lab, 'n']):,}, MAD {h.loc[lab, 'mad']:.2f} vs "
        f"{m.loc[lab, 'mad']:.2f} m, median r {h.loc[lab, 'median']:+.2f} vs "
        f"{m.loc[lab, 'median']:+.2f} m, bias {h.loc[lab, 'bias']:+.2f} vs "
        f"{m.loc[lab, 'bias']:+.2f} m, RMSE {h.loc[lab, 'rmse']:.1f} vs "
        f"{m.loc[lab, 'rmse']:.1f} m"
        for lab in DIST_LABELS
    )
    near = int(h.loc[DIST_LABELS[0], "n"] + h.loc[DIST_LABELS[1], "n"])
    null_txt = (
        f" {meta['n_null_block']:,} targets carry no basin code and are pooled "
        "as one extra block, as the scorer pools them."
        if meta["n_null_block"]
        else ""
    )
    return (
        f"Figure 7. How does the held-out error of the deployed {name} map "
        "depend on how far a scored well sits from the nearest well the map "
        "was given? Error grows with distance for both products; handily's MAD "
        "point estimate sits below Ma's in every band, with the CI95 bars "
        "separating in the near bands and overlapping beyond about 1 km, where "
        "few targets remain.\n\n"
        f"The {n_tot:,} held-out {ag} targets of the admission split (Figure 5; "
        f"{n_elig:,} eligible, {n_dropped} with no finite prediction or "
        "observation at the sampled cell are excluded), scored at their own "
        "100 m cell against the shipped handily parity render "
        f"({RENDER_ARM}; filled circles, solid) and the Ma et al. depth-to-water "
        "product resampled onto the same grid (open circles, dashed), binned by "
        "the planar EPSG:5070 distance from the target to the nearest ADMITTED "
        f"{ag} source — the sibling half of the same split, re-supplied to the "
        "model as observations at render time. Bundle monitoring wells are also "
        "sources but are not in this distance (Figure 5a uses both). The first "
        f"band, under 50 m, is the near-duplicate band: {near:,} targets "
        f"({near / n_tot:.0%}) lie within 300 m of a source.\n\n"
        "Residual r = predicted minus observed depth to water, in metres; "
        "positive means the map puts the water table deeper than the well "
        "found it. (a) Targets per band; the same targets score both products. "
        "(b) MAD = median |r|, the typical error. (c) RMSE = sqrt(mean r²), "
        "which the catastrophic tail dominates. (d) Median r and (e) mean r "
        "(bias), side by side: where the mean sits far from the median the "
        "error is a skewed tail, not an offset. (f) p95 |r|, the 95th "
        "percentile of |r|. Bars are CI95 bounds on each statistic from a "
        f"paired block bootstrap: {N_BOOT:,} resamples (numpy default_rng, seed "
        f"{SEED}) drawing the {meta['n_blocks']} {CFG['block_name']} blocks "
        "with replacement and taking the 2.5th and 97.5th percentiles of the "
        "statistic, both products evaluated on the same resample so a visual "
        "overlap of two bars is conservative for the paired difference."
        f"{null_txt} Per band, handily vs Ma: {per_band}.\n\n"
        "The 50/50 by-site split was random, so the near bands are not a "
        "biased subset: they are what a dense driller-record panel looks like "
        "when half of it is handed to the map. Statistics are re-derived here "
        "from the statewide scorer's sampled-site table and asserted equal to "
        "its panel_dist.csv on the scorer's own 0.3/1/5/20 km bands before the "
        "finer bands are drawn. Units: distances km, depths and residuals m.\n"
    )


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    use_style()
    t, n_elig, n_dropped = load_targets()
    stats, meta = compute(t)

    fig = draw(stats)
    paths = save(fig, "fig06_error_vs_distance", FIG_DIR)
    plt.close(fig)

    cap_path = FIG_DIR / "fig06_error_vs_distance_caption.txt"
    cap_path.write_text(_caption(stats, meta, n_elig, n_dropped))
    stats.to_csv(FIG_DIR / "fig06_error_vs_distance_stats.csv", index=False)

    cols = [
        "predictor",
        "band",
        "n",
        "mad",
        "mad_lo",
        "mad_hi",
        "bias",
        "median",
        "rmse",
        "p95",
    ]
    print(stats[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(
        f"\n{len(t):,} targets of {n_elig:,} eligible ({n_dropped} dropped non-finite); "
        f"{meta['n_blocks']} blocks ({meta['n_null_block']} null-block targets); "
        f"scorer reproduction on {len(SRC_LABELS)} bands x {len(PRED)} predictors OK"
    )
    for ext, p in paths.items():
        print(f"{ext}: {p}")
    print(f"caption: {cap_path}")


if __name__ == "__main__":
    main()
