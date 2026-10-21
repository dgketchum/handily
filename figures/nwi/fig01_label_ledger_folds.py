"""Figure 1 — the well ledger and the fold partition.

(a) Every well label of the training bundle inside the state render footprint,
    binned by observed depth to water and stacked by contributing network: the
    unconfined monitoring networks and the appended state-agency wells. The
    whole national bundle is available with ``--scope conus``. Water-stage pseudo-rows are excluded: they
    are surface-water stage labels carried at weight 0.25, not wells.
(b) The state render footprint, tiled by its HUC12s and coloured by
    cross-validation fold. Folds are dealt to whole HUC12s, so a held-out fold
    removes whole HUC12s at once. The HUC12s tile the footprint completely; only
    those holding a well label carry a fold.

Written to the house figure rules in ``~/code/style/FIGURE_STYLE_GUIDE.md``:
no gridlines, no descriptive panel titles (the findings live in the caption,
emitted alongside the images), framed interior legends placed with ``loc=`` only
so the frame-to-spine inset comes from the shared mplstyle, type sizes left to
that style, and a colour-vision-deficiency-safe palette with hatch redundancy
for grayscale.

Run::

    NWI_STATE=NM uv run python figures/nwi/fig01_label_ledger_folds.py               # footprint ledger
    NWI_STATE=NM uv run python figures/nwi/fig01_label_ledger_folds.py --scope conus # whole bundle

Writes ``fig01_label_ledger_folds[_conus].{pdf,png}`` plus a ``*_caption.txt``
sidecar to ``/data/ssd2/handily/figures/<project>/`` (``nwi`` for NV, ``nmose`` for NM).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from nwi_common import (
    BAND_LABELS,
    CFG,
    FIG_DIR,
    FOLD_COLORS,
    HUC8_POLYS,
    MM,
    NO_FOLD_COLOR,
    NO_FOLD_EDGE,
    SOURCE_COLORS,
    SOURCE_HATCHES,
    SOURCE_LABELS,
    SOURCE_ORDER,
    WIDTH_2COL,
    huc12_folds,
    load_bundle,
    north_arrow,
    outside_letter,
    rendered_basins,
    save,
    scale_bar,
    state_outline,
    use_style,
)


def _ledger_counts(qn, scope: str, basins: list[str]):
    """band x network count table for well labels, plus the rows used.

    Networks with no well in the chosen population are dropped from the table
    (and so from the legend), so a footprint ledger does not list the other
    state's appended class.
    """
    wells = qn[qn["is_well"]]
    rows = wells if scope == "conus" else wells[wells["huc8"].isin(basins)]
    tab = (
        rows.pivot_table(
            index="band",
            columns="source_class",
            aggfunc="size",
            observed=False,
            fill_value=0,
        )
        .reindex(index=list(BAND_LABELS))
        .reindex(columns=list(SOURCE_ORDER), fill_value=0)
        .fillna(0)
        .astype(int)
    )
    tab = tab.loc[:, tab.sum(axis=0) > 0]
    return tab, rows


def panel_ledger(ax, tab) -> None:
    y = np.arange(len(BAND_LABELS))
    left = np.zeros(len(BAND_LABELS))
    for src in tab.columns:
        w = tab[src].to_numpy(float)
        ax.barh(
            y,
            w,
            left=left,
            height=0.72,
            color=SOURCE_COLORS[src],
            edgecolor="white",
            linewidth=0.4,
            hatch=SOURCE_HATCHES[src] or None,
            zorder=2,
        )
        left += w

    ax.set_yticks(y)
    ax.set_yticklabels(BAND_LABELS)
    ax.invert_yaxis()  # shallow at the top, deepening downward
    ax.set_ylabel("Observed depth to water (m below ground)")
    ax.set_xlabel("Well labels (n)")
    top = tab.sum(axis=1).max()
    ax.set_xlim(0, top * 1.05)
    # 4-5 majors at a round interval; MaxNLocator alone bottoms out at two.
    step = next(s for s in (100, 200, 500, 1000, 2000, 5000) if top / s <= 5)
    ax.xaxis.set_major_locator(MultipleLocator(step))

    handles = [
        Patch(
            facecolor=SOURCE_COLORS[s],
            edgecolor="white",
            linewidth=0.4,
            hatch=SOURCE_HATCHES[s] or None,
            label=SOURCE_LABELS[s],
        )
        for s in tab.columns
    ]
    ax.legend(handles=handles, loc="upper right", title="Label network")


def panel_folds(ax, h12, basin_polys, footprint, state) -> None:
    # Every HUC12 in the footprint is drawn; only those holding a well label
    # carry a fold, so the flat fill IS the tiling, not a gap.
    withwell = h12[h12["n_wells"] > 0].copy()
    nolabel = h12[h12["n_wells"] == 0]

    # The thousands-polygon tiling is rasterized (at savefig.dpi) so the PDF
    # stays under publisher size caps; boundaries, legend and text remain vector.
    nolabel.plot(
        ax=ax,
        color=NO_FOLD_COLOR,
        edgecolor=NO_FOLD_EDGE,
        linewidth=0.15,
        rasterized=True,
    )
    basin_polys.boundary.plot(ax=ax, color="#8c8c8c", linewidth=0.3, zorder=2)
    withwell["_c"] = [FOLD_COLORS[int(f)] for f in withwell["cv_fold"]]
    withwell.plot(
        ax=ax,
        color=withwell["_c"],
        edgecolor="white",
        linewidth=0.15,
        zorder=3,
        rasterized=True,
    )
    footprint.boundary.plot(ax=ax, color="#4d4d4d", linewidth=0.4, zorder=4)
    state.boundary.plot(ax=ax, color="black", linewidth=0.7, zorder=5)

    ax.set_axis_off()
    ax.set_aspect("equal")
    scale_bar(ax)
    north_arrow(ax)

    handles = [
        Patch(facecolor=FOLD_COLORS[f], edgecolor="none", label=f"{f}")
        for f in range(len(FOLD_COLORS))
    ] + [
        Patch(
            facecolor=NO_FOLD_COLOR,
            edgecolor=NO_FOLD_EDGE,
            linewidth=0.4,
            label="none",
        )
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=3,
        title="HUC12 fold",
        handlelength=1.0,
    )


def _caption(
    scope: str, tab, rows, h12, withwell_n: int, n_basins: int, n_pseudo
) -> str:
    per_src = tab.sum(axis=0)
    bands = ", ".join(
        f"{lab} m, {int(n):,}" for lab, n in zip(BAND_LABELS, tab.sum(axis=1))
    )
    nets = "; ".join(f"{SOURCE_LABELS[s]} {int(per_src[s]):,}" for s in tab.columns)
    pop = (
        "the training bundle"
        if scope == "conus"
        else f"the {n_basins}-basin {CFG['name']} render footprint"
    )
    return (
        "Figure 1. Water-table labels are supplied by one dominant monitoring "
        "network plus the appended state-agency wells, and are split for "
        "cross-validation by whole HUC12.\n\n"
        f"(a) The {len(rows):,} well labels in {pop}, binned by observed depth "
        "to water (metres below ground) and stacked by contributing network: "
        f"{nets}. Wells per band: {bands}. Depth to water is land-surface "
        "elevation minus observed water-table elevation. Monitoring-network "
        "rows are unconfined wells carrying their period-of-record mean level; "
        "appended rows are the admitted half of a state agency's admission "
        "split (Figure 5), each carrying its single admitted level and appended "
        f"as an ordinary label at weight 1. Water-stage pseudo-rows (n = "
        f"{n_pseudo:,} continentally), which supply surface-water stage labels "
        "at loss weight 0.25, are excluded; so are confined and likely-confined "
        "monitoring wells, which measure a potentiometric surface rather than "
        "the water table.\n\n"
        f"(b) The render footprint, {n_basins} HUC8 basins tiled by {len(h12):,} "
        "HUC12s, with each HUC12 coloured by its cross-validation fold. Folds "
        "are dealt round-robin to whole HUC12s from a seeded permutation, so "
        "train and test never share a HUC12, and a held-out HUC12's neighbours "
        "stay in train; a HUC12 that gained its first well from the appended "
        "rows takes, as a whole, the fold of the nearest base well. Only "
        f"{withwell_n:,} HUC12s ({withwell_n / len(h12):.0%}) hold a well label "
        "and therefore carry a fold; the remainder supply no well labels and "
        "are drawn in the flat fill keyed 'none'. Hairlines are HUC12 "
        "boundaries, mid-grey lines HUC8 boundaries, the dark grey line the "
        f"render footprint, and the black line the {CFG['name']} state boundary "
        "— the footprint extends past it because HUC8 basins straddle the "
        "border. Albers equal-area projection (EPSG:5070); state boundary from "
        "the 2022 US Census cartographic boundary file, hydrologic units from "
        "the WBD. Fold identity is nominal and is the one encoding in this "
        "figure that does not survive grayscale conversion.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scope",
        choices=("state", "conus"),
        default="state",
        help="well ledger population: only wells inside the state render "
        "footprint (default) or the whole training bundle",
    )
    args = ap.parse_args()

    import geopandas as gpd

    use_style()

    basins = rendered_basins()
    qn = load_bundle()
    tab, rows = _ledger_counts(qn, args.scope, basins)
    n_pseudo = int(qn["is_water_pseudo"].astype(bool).sum())

    h12 = huc12_folds(basins)
    h8 = gpd.read_parquet(HUC8_POLYS, columns=["huc8", "geometry"])
    h8["huc8"] = h8["huc8"].astype(str).str.zfill(8)
    basin_polys = h8[h8["huc8"].isin(basins)]
    footprint = basin_polys.dissolve()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_2COL, 88 * MM),
        gridspec_kw={"width_ratios": [1.0, 0.78], "wspace": 0.10},
    )
    panel_ledger(axes[0], tab)
    panel_folds(axes[1], h12, basin_polys, footprint, state_outline())
    outside_letter(axes[0], "a", x=-0.16)
    outside_letter(axes[1], "b")

    stem = "fig01_label_ledger_folds"
    if args.scope != "state":
        stem += f"_{args.scope}"
    paths = save(fig, stem, FIG_DIR)
    plt.close(fig)

    withwell_n = int((h12["n_wells"] > 0).sum())
    cap_path = FIG_DIR / f"{stem}_caption.txt"
    cap_path.write_text(
        _caption(args.scope, tab, rows, h12, withwell_n, len(basins), n_pseudo)
    )

    print(tab.to_string())
    print(f"\nscope={args.scope}  wells={len(rows):,}")
    for ext, p in paths.items():
        print(f"{ext}: {p}")
    print(f"caption: {cap_path}")


if __name__ == "__main__":
    main()
