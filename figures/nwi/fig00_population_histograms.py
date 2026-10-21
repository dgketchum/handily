"""Figure 0 — the four well populations behind the state evaluation.

Every well that touches the state package enters through one of four doors,
and the four differ in depth distribution, not just in size:

(a) the training bundle the model was fitted on, by label class: the national
    unconfined monitoring wells and the appended state-agency wells;
(b) the same footprint wells, split into the fold-0
    train / val / test roles;
(c) the state-agency water levels admitted as deployment data-assimilation
    sources (the same wells as the appended labels, before the append funnel);
(d) the held-out state-agency sites the deployment render is scored against.

All four panels are histograms of the same quantity — observed depth to water,
metres below land surface — on one shared log x axis with one shared set of bin
edges, so the shapes are directly comparable. The first bin also collects the
handful of observations at or above land surface (depth <= 0.1 m). Counts are
NOT shared: the national panel holds several times the rows of any other, so
each panel carries its own y scale (guide section 6: per-panel scaling,
declared in the caption).

Written to the house figure rules in ``~/code/style/FIGURE_STYLE_GUIDE.md``:
four-sided box, no gridlines, no descriptive panel titles (the naming lives in
the caption, emitted alongside the images), one framed legend placed with
``loc=`` only so the frame-to-spine inset comes from the shared mplstyle, type
sizes left to that style, and a colour-vision-deficiency-safe palette with hatch
redundancy so the grouped classes survive grayscale.

The fold-0 split is not re-derived here: it is imported from
``fig03_fold_split``, which reproduces the trainer's split verbatim and asserts
it against the per-fold row counts logged by the run of record.

Run::

    NWI_STATE=NM uv run --directory /home/dgketchum/code/handily \
        python /home/dgketchum/code/handily/figures/nwi/fig00_population_histograms.py

Writes ``fig00_population_histograms.{pdf,png}`` plus a ``*_caption.txt``
sidecar to ``/data/ssd2/handily/figures/<project>/`` (``nwi`` for NV, ``nmose`` for NM).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    MultipleLocator,
    NullLocator,
)

import fig03_fold_split as f3
from nwi_common import (
    BUNDLE_DIR,
    CFG,
    FIG_DIR,
    MM,
    RENDER_ARM,
    SOURCE_COLORS,
    SOURCE_HATCHES,
    SOURCES_PATH,
    TARGETS_PATH,
    WELL_CLASSES,
    WIDTH_2COL,
    outside_letter,
    rendered_basins,
    save,
    use_style,
)

#: Shared bin edges: log-spaced from 0.1 m to the deepest observation in any of
#: the four populations. Five bins per decade reads at slide width; finer bins
#: leave the grouped sub-bars below the mark-acuity floor (guide section 6).
BIN_FLOOR_M = 0.1
BINS_PER_DECADE = 5

#: The fold-0 roles, in descending magnitude so legend order matches the bars.
ROLE_ORDER = ("train", "val", "test")
ROLE_COLORS = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}
ROLE_HATCHES = {"train": "", "val": "///", "test": "xxx"}

#: The single-population panels (c, d) and the monitoring class in (a) share one
#: ink: they show the same quantity for different populations, so a per-panel
#: hue would encode nothing. The appended classes take their Figure 1 colours.
WELL_INK = "#0072B2"
CLASS_SOURCE = {"monitoring": None, "ndwr": "nv_ndwr", "ose": "nm_ose"}
CLASS_LABELS = {
    "monitoring": "monitoring networks",
    "ndwr": "NV NDWR (appended)",
    "ose": "NM OSE (appended)",
}


def class_style(cls: str) -> dict:
    src = CLASS_SOURCE[cls]
    if src is None:
        return {"color": WELL_INK, "hatch": None}
    return {"color": SOURCE_COLORS[src], "hatch": SOURCE_HATCHES[src] or None}


# --------------------------------------------------------------------------
# Populations
# --------------------------------------------------------------------------


#: Two sites this close in EPSG:5070 metres are treated as possibly one borehole.
COLOCATION_TOL_M = 30.0


def _colocated(xy_a: np.ndarray, xy_b: np.ndarray) -> int:
    """Rows of ``xy_b`` within :data:`COLOCATION_TOL_M` of any row of ``xy_a``."""
    from scipy.spatial import cKDTree

    d, _ = cKDTree(xy_a).query(xy_b)
    return int((d < COLOCATION_TOL_M).sum())


def expected_counts() -> dict:
    """Population sizes of record, read from the manifests rather than typed.

    A mismatch between these and the tables means the bundle or the admission
    split moved under the figure, which must raise rather than draw a stale
    ledger.
    """
    man = json.loads((BUNDLE_DIR / "graph_manifest.json").read_text())
    app = man["appended_wells"]
    if app["source_name"] != CFG["source_name"]:
        raise ValueError(
            f"bundle appended {app['source_name']!r}, not {CFG['source_name']!r}"
        )
    runs = sorted(
        glob.glob(f"/data/ssd2/handily/huc8/*/gnn/{RENDER_ARM}/infer_run.json")
    )
    if not runs:
        raise FileNotFoundError(f"no render manifests for {RENDER_ARM}")
    se = {
        (d["source_edges"]["n_sources_bundle"], d["source_edges"]["n_sources_extra"])
        for d in (json.loads(Path(p).read_text()) for p in runs)
    }
    if len(se) != 1:
        raise ValueError(f"basins disagree on the deployment source counts: {se}")
    n_bundle, n_extra = se.pop()
    return {
        "monitoring": int(n_bundle),
        "appended_state": int(app["n_appended"]),
        "appended_all": int(man["parity_rewrite"]["appended_r"]["n"]),
        "sources": int(n_extra),
        "append_funnel": app["funnel"],
    }


def load_populations() -> dict:
    """The four depth-to-water populations, each as a plain float array.

    Depth is metres below land surface: ``mean_dtw`` for the bundle rows (the
    column the model is trained against; for the appended state wells it is the
    single admitted level) and ``obs_dtw_m`` for the two agency tables.
    Water-stage pseudo-rows are excluded from the bundle populations — they are
    surface-water stage labels, not wells.
    """
    exp = expected_counts()
    rows = f3.load_rows()
    dtw = pd.read_parquet(
        BUNDLE_DIR / "query_nodes.parquet", columns=["query_node_idx", "mean_dtw"]
    )
    rows = rows.merge(dtw, on="query_node_idx", how="left", validate="one_to_one")
    if not np.isfinite(rows["mean_dtw"]).all():
        raise ValueError("non-finite mean_dtw in the bundle label set")

    split = f3.split_all_folds(rows)
    f3.verify_against_run(split)

    is_well = rows["well_class"].isin(WELL_CLASSES) & ~rows["is_water_pseudo"].astype(
        bool
    )
    national = rows[is_well]
    classes = {
        cls: national[national["well_class"] == cls]["mean_dtw"].to_numpy(float)
        for cls in WELL_CLASSES
        if (national["well_class"] == cls).any()
    }
    if len(classes["monitoring"]) != exp["monitoring"]:
        raise ValueError(
            f"{len(classes['monitoring']):,} monitoring wells in the bundle but the "
            f"render exposes {exp['monitoring']:,} bundle sources"
        )
    n_app = sum(len(v) for c, v in classes.items() if c != "monitoring")
    if n_app != exp["appended_all"]:
        raise ValueError(
            f"{n_app:,} appended rows in the bundle, manifest says {exp['appended_all']:,}"
        )
    if len(classes[CFG["well_class"]]) != exp["appended_state"]:
        raise ValueError(
            f"{len(classes[CFG['well_class']]):,} {CFG['agency']} rows in the bundle, "
            f"manifest says {exp['appended_state']:,}"
        )

    basins = rendered_basins()
    foot = rows[is_well & rows["huc8"].isin(basins)]
    # Panel (a) draws only the bundle labels inside the render footprint; the
    # national counts above are checked against the manifests but not drawn,
    # so a class appended for another state's package never appears here.
    foot_by_class = {
        cls: foot[foot["well_class"] == cls]["mean_dtw"].to_numpy(float)
        for cls in WELL_CLASSES
        if (foot["well_class"] == cls).any()
    }
    roles = {
        role: foot["mean_dtw"].to_numpy(float)[split[0][role][foot.index]]
        for role in ROLE_ORDER
    }
    n_roles = sum(len(v) for v in roles.values())
    if n_roles != len(foot):
        raise ValueError(f"fold-0 roles cover {n_roles} of {len(foot)} footprint wells")
    foot_classes = foot["well_class"].value_counts().to_dict()

    src = pd.read_parquet(SOURCES_PATH, columns=["x5070", "y5070", "obs_dtw_m"])
    tgt = pd.read_parquet(
        TARGETS_PATH, columns=["x5070", "y5070", "obs_dtw_m", "is_admission_eligible"]
    )
    tgt = tgt[tgt["is_admission_eligible"].astype(bool)]
    for name, frame in (("sources", src), ("targets", tgt)):
        if not np.isfinite(frame["obs_dtw_m"]).all():
            raise ValueError(f"non-finite obs_dtw_m in the {CFG['agency']} {name}")
    if len(src) != exp["sources"]:
        raise ValueError(
            f"{len(src):,} frozen sources but the render logged {exp['sources']:,}"
        )

    pops = {
        "bundle": foot["mean_dtw"].to_numpy(float),
        "classes": foot_by_class,
        "footprint": foot["mean_dtw"].to_numpy(float),
        "footprint_classes": foot_classes,
        "n_bundle_national": len(national),
        "sources": src["obs_dtw_m"].to_numpy(float),
        "targets": tgt["obs_dtw_m"].to_numpy(float),
        "roles": roles,
        "expected": exp,
        "n_basins": len(basins),
    }
    # The populations are described as distinct in the caption, so the claim is
    # measured rather than asserted: how many agency sites sit on top of a
    # monitoring well, and do the two agency sets share any site at all.
    mon_xy = national.loc[national["well_class"] == "monitoring", ["x5070", "y5070"]]
    mon_xy = mon_xy.to_numpy(float)
    pops["colocated"] = {
        "sources": _colocated(mon_xy, src[["x5070", "y5070"]].to_numpy(float)),
        "targets": _colocated(mon_xy, tgt[["x5070", "y5070"]].to_numpy(float)),
    }
    pops["shared_sites"] = len(
        set(map(tuple, np.round(src[["x5070", "y5070"]].to_numpy(float), 2)))
        & set(map(tuple, np.round(tgt[["x5070", "y5070"]].to_numpy(float), 2)))
    )
    return pops


POP_KEYS = ("bundle", "footprint", "sources", "targets")


def shared_edges(pops: dict) -> np.ndarray:
    """One set of log-spaced edges, 0.1 m to the deepest observation of all four."""
    dmax = max(float(pops[k].max()) for k in POP_KEYS)
    n_dec = np.log10(dmax / BIN_FLOOR_M)
    n_bins = int(round(n_dec * BINS_PER_DECADE))
    return np.logspace(np.log10(BIN_FLOOR_M), np.log10(dmax), n_bins + 1)


def binned(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Counts per shared bin, with everything at or below 0.1 m in the first bin."""
    v = np.clip(values, edges[0], edges[-1])
    counts, _ = np.histogram(v, bins=edges)
    if counts.sum() != len(values):
        raise ValueError("binning dropped rows")
    return counts


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def _bars(ax, left, right, counts, **kw) -> None:
    ax.bar(left, counts, width=right - left, align="edge", linewidth=0, **kw)


def panel_single(ax, values, edges) -> None:
    """One population, one series of bars on the shared edges."""
    _bars(ax, edges[:-1], edges[1:], binned(values, edges), color=WELL_INK, zorder=2)


def panel_grouped(ax, groups: dict, edges, styles: dict, labels: dict, title) -> None:
    """Several series as grouped bars inside each shared bin.

    Grouped, never stacked: a stacked histogram gives the upper series an
    ambiguous baseline (guide section 15/16). The sub-bars split each bin
    geometrically, so they stay equal-width on the log axis.
    """
    ratio = edges[1:] / edges[:-1]
    keys = list(groups)
    handles = []
    for k, key in enumerate(keys):
        left = edges[:-1] * ratio ** (k / len(keys))
        right = edges[:-1] * ratio ** ((k + 1) / len(keys))
        st = styles[key]
        _bars(ax, left, right, binned(groups[key], edges), zorder=2, **st)
        handles.append(
            Patch(
                facecolor=st["color"],
                edgecolor="none",
                hatch=st["hatch"],
                label=f"{labels[key]}, n = {len(groups[key]):,}",
            )
        )
    ax.legend(handles=handles, loc="upper right", handlelength=1.0, title=title)


#: Round count-axis steps; the smallest one giving at most five majors wins, so
#: every panel ends on a round top with a round interval (guide section 6).
Y_STEPS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def style_axes(axes, edges) -> None:
    """Shared log x axis, per-panel count axis, round ticks on both."""
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xlim(edges[0], edges[-1])
        ax.xaxis.set_major_locator(FixedLocator([0.1, 1.0, 10.0, 100.0]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10", "100"]))
        # A log scale installs its own minor locator, which the style file's
        # xtick.minor.visible = False does not reach (guide section 6: majors only).
        ax.xaxis.set_minor_locator(NullLocator())

        tallest = max(p.get_height() for p in ax.patches)
        step = next(s for s in Y_STEPS if tallest / s <= 5)
        ax.set_ylim(0, step * np.ceil(tallest / step))
        ax.yaxis.set_major_locator(MultipleLocator(step))


# --------------------------------------------------------------------------
# Caption
# --------------------------------------------------------------------------


def _caption(pops: dict, edges: np.ndarray) -> str:
    roles = pops["roles"]
    exp = pops["expected"]
    fun = exp["append_funnel"]
    n = {k: len(pops[k]) for k in POP_KEYS}
    med = {k: float(np.median(pops[k])) for k in POP_KEYS}
    cls = pops["classes"]
    cls_txt = "; ".join(
        f"{CLASS_LABELS[c]} n = {len(v):,}, median {np.median(v):.1f} m"
        for c, v in cls.items()
    )
    return (
        f"Figure 0. The well populations behind the {CFG['name']} package "
        "differ in depth distribution, not only in size.\n\n"
        "Histograms of observed depth to water, metres below land surface, for "
        f"every population the {CFG['name']} evaluation rests on. All four panels "
        f"share one log x axis and one set of {len(edges) - 1} bin edges, "
        f"log-spaced from {edges[0]:g} m to {edges[-1]:.1f} m, the deepest "
        f"observation in any population ({BINS_PER_DECADE} bins per decade). The "
        f"first bin also collects observations at or above land surface, that is "
        f"depth <= {edges[0]:g} m, of which there are "
        f"{sum(int((pops[k] <= edges[0]).sum()) for k in POP_KEYS)} in total; no "
        "population holds a negative depth. Counts are not shared between "
        "panels: each panel is scaled to its own maximum from a common zero "
        "baseline.\n\n"
        f"(a) The n = {n['bundle']:,} well labels of the training bundle of the "
        f"model of record ({RENDER_ARM}) that lie inside the "
        f"{pops['n_basins']}-basin {CFG['name']} render footprint, drawn as "
        f"grouped bars by label class — {cls_txt}. The monitoring class is the "
        "footprint's share of the national set of unconfined monitoring wells "
        "the model is trained on; depth is the well's mean observed depth to "
        "water over its period of record. The appended "
        f"{CFG['agency']} class is the admitted half of the {CFG['agency_long']} "
        f"({CFG['agency']}) split (panel c) appended to the bundle as ordinary "
        f"labels at weight 1: {fun['wells_in']:,} admitted sites in, "
        f"{fun['dropped_collocated_le_1m']} dropped as collocated within 1 m of "
        f"a bundle well, {fun['dropped_nonfinite_fac_rem']} dropped for lacking "
        f"a finite FAC-REM prior, {exp['appended_state']:,} kept; each carries "
        "its single admitted level. Bundle labels outside the footprint are "
        "not drawn. Water-stage pseudo-rows, which supply surface-water stage "
        "labels at loss weight 0.25, are excluded, as are confined and "
        "likely-confined monitoring wells, which measure a potentiometric "
        "surface rather than the water table. Classes are encoded by colour "
        "and hatch and read from a common zero baseline; each bin's sub-bars "
        "divide the bin geometrically and are equal-width on the log axis.\n\n"
        f"(b) The same n = {n['footprint']:,} footprint wells, split into the "
        "three roles of cross-validation fold 0: train "
        f"(n = {len(roles['train']):,}), val (n = {len(roles['val']):,}) and "
        f"test (n = {len(roles['test']):,}). Test is every well in a HUC12 dealt "
        "to fold 0; val is every remaining well inside a set of whole 40 km grid "
        "blocks drawn until they hold about 15% of the train-val rows, and is "
        "used only to early-stop training, never scored; train is the rest. "
        "Appended wells take the fold of their HUC12 where the base bundle had "
        "one, and otherwise the fold of the nearest base well, so they ride the "
        "same fold guard. The split is reproduced from utils/train_conus_gnn.py "
        "and asserted against the per-fold row counts logged by the run of "
        f"record ({RENDER_ARM}, val_frac 0.15, seed 0). Median depth "
        f"{med['footprint']:.1f} m.\n\n"
        f"(c) The n = {n['sources']:,} {CFG['agency']} water levels admitted as "
        "deployment data-assimilation sources, the set the deployed surface is "
        "conditioned on at inference and, after the append funnel of (a), the "
        f"set it was also trained on. Median depth {med['sources']:.1f} m.\n\n"
        f"(d) The n = {n['targets']:,} held-out {CFG['agency']} sites, the "
        "admission-eligible targets the deployment render is scored against. "
        "The target sites in (d) and the source sites in (c) are disjoint: "
        f"{pops['shared_sites']} site coordinates appear in both. Neither "
        f"{CFG['agency']} set is drawn from the monitoring networks in (a), "
        f"though {pops['colocated']['sources']} sources and "
        f"{pops['colocated']['targets']} targets sit within "
        f"{COLOCATION_TOL_M:g} m of a monitoring well and may be the same "
        f"borehole under two identifiers. Median depth {med['targets']:.1f} m.\n\n"
        "Depth to water throughout is depth to the unconfined water table. "
        f"Bundle depths are from {BUNDLE_DIR.name}/query_nodes.parquet, "
        f"{CFG['agency']} depths from {SOURCES_PATH.name} and "
        f"{TARGETS_PATH.name} under {CFG['well_dir']}.\n"
    )


# --------------------------------------------------------------------------


def main() -> None:
    use_style()
    mpl.rcParams["hatch.linewidth"] = 0.4
    # The grid carries one shared axis title per direction; take its size from
    # the style file's axis-label size rather than matplotlib's oversized default.
    mpl.rcParams["figure.labelsize"] = mpl.rcParams["axes.labelsize"]

    pops = load_populations()
    edges = shared_edges(pops)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(WIDTH_2COL, 108 * MM),
        sharex=True,
        layout="constrained",
    )
    cls = pops["classes"]
    panel_grouped(
        axes[0, 0],
        cls,
        edges,
        {c: class_style(c) for c in cls},
        CLASS_LABELS,
        "Label class",
    )
    panel_grouped(
        axes[0, 1],
        pops["roles"],
        edges,
        {
            r: {"color": ROLE_COLORS[r], "hatch": ROLE_HATCHES[r] or None}
            for r in ROLE_ORDER
        },
        {r: r for r in ROLE_ORDER},
        "Fold-0 role",
    )
    panel_single(axes[1, 0], pops["sources"], edges)
    panel_single(axes[1, 1], pops["targets"], edges)
    style_axes(axes, edges)

    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.05, wspace=0.02, hspace=0.06)
    fig.supxlabel("Observed depth to water (m below land surface)")
    fig.supylabel("Wells (n)")
    for ax, letter in zip(axes.ravel(), "abcd"):
        outside_letter(ax, letter, x=-0.02)

    paths = save(fig, "fig00_population_histograms", FIG_DIR)
    plt.close(fig)

    cap_path = FIG_DIR / "fig00_population_histograms_caption.txt"
    cap_path.write_text(_caption(pops, edges))

    for key in POP_KEYS:
        v = pops[key]
        print(f"{key:9s} n = {len(v):7,d}  median {np.median(v):7.2f} m")
    for c, v in cls.items():
        print(f"  class {c:10s} n = {len(v):6,d}  median {np.median(v):7.2f} m")
    for role in ROLE_ORDER:
        print(f"  fold-0 {role:5s} n = {len(pops['roles'][role]):5,d}")
    print(f"bins: {len(edges) - 1} edges {edges[0]:g}-{edges[-1]:.1f} m")
    for ext, p in paths.items():
        print(f"{ext}: {p}")
    print(f"caption: {cap_path}")


if __name__ == "__main__":
    main()
