"""F4 -- FAC-REM: the shallow terrain prior.

Run as::

    uv run python -m utils.figures.fig04_fac_rem

Writes into ``/data/ssd2/handily/figures/toy/``:

``fig04_fac_rem_paper.{png,pdf}``
    Row 1: the pipeline as five maps on shared geometry (DEM, log flow
    accumulation, stream network with channel heads, FAC water surface, FAC-REM
    depth). Row 2: section B-B' with the FAC water surface against the truth.
    Row 3: median absolute DTW error by distance to stream, at water-table wells
    and over the grid, with the Ma-like and 3 m mirror comparators.
``fig04_fac_rem_3d_a.png``
    A 5 km slab of the basin cut open on B-B', seen broadside from the west: the
    translucent FAC water surface and the true water table under the land skin,
    both traced on the cut face.
``fig04_fac_rem_3d_b.png``
    The whole basin under its land skin with the same slab of water surfaces
    beneath it, from the series' standard oblique bearing.

The FAC-REM prior sees no well labels at all, so every well glyph in this figure
is hollow (the visibility ledger's "label hidden" state).

Comparator honesty
------------------
The Ma-like benchmark interpolates *observed* DTW, so evaluating the all-well
field at a well returns that well's own observation. The at-wells panel therefore
scores a leave-one-fold-out Ma-like field (fold ``k`` wells scored against a field
fitted without fold ``k``; the buffered-holdout wells against the six-fold field),
exactly the construction ``toy_priors.build_R`` uses for R. The grid panel uses
the all-well field from ``toy_priors.ma_like``, which is what the foundation's
verification numbers report. FAC-REM and the 3 m mirror use no wells, so neither
construction changes them.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.figures import render3d as r3
from utils.figures.fig_common import (
    CELL,
    CMAPS,
    COLORS,
    EXTENT,
    LAND_3D,
    LINESTYLES,
    MM,
    NX,
    SECTION_B,
    TOY_DIR,
    VE,
    WIDTH_2COL,
    add_ticks,
    annotate_ve,
    draw_section_lines,
    draw_wells_section,
    map_axes,
    panel_letter,
    profile_along,
    save,
    use_style,
    well_ledger_handles,
    wells_near_line,
)
from utils.figures.toy_priors import (
    _idw_planar,
    fac_rem,
    ma_like,
    training_pool,
    water_table_wells,
)
from utils.figures.toy_world import N_FOLDS, load_world

#: Distance-to-stream bands (m); the last one is open-ended.
BINS = [(0, 100), (100, 300), (300, 600), (600, 1000), (1000, 2000), (2000, np.inf)]
BIN_LABELS = ["0-100", "100-300", "300-600", "600-1000", "1-2 km", "> 2 km"]

MIRROR_DEPTH = 3.0  # m, the constant-depth mirror expert
STRIP_RADIUS_M = 1000.0  # m, the FAC strip-fill decay length (toy_priors default)


# --------------------------------------------------------------------------
# Quantities this figure needs that the shared modules do not provide
# --------------------------------------------------------------------------


def channel_heads(world: dict) -> np.ndarray:
    """Boolean ``(NY, NX)``: stream cells with no stream cell draining into them.

    A cell is a head when it carries enough contributing area to be a stream but
    no upstream neighbour does, i.e. the top of a first-order reach.
    """
    streams = np.asarray(world["streams"], bool)
    receiver = np.asarray(world["receiver"], np.int64).ravel()
    flat = streams.ravel()
    fed = np.zeros(flat.size, bool)
    src = np.flatnonzero(flat)
    dst = receiver[src]
    fed[dst[dst >= 0]] = True
    return (flat & ~fed).reshape(streams.shape)


def stream_segments(world: dict) -> np.ndarray:
    """``(n, 2, 2)`` array of stream-cell -> D8-receiver segments in metres."""
    streams = np.asarray(world["streams"], bool)
    receiver = np.asarray(world["receiver"], np.int64).ravel()
    xf = np.asarray(world["X"], float).ravel()
    yf = np.asarray(world["Y"], float).ravel()
    src = np.flatnonzero(streams.ravel())
    dst = receiver[src]
    ok = dst >= 0
    src, dst = src[ok], dst[ok]
    return np.stack(
        [np.column_stack([xf[src], yf[src]]), np.column_stack([xf[dst], yf[dst]])],
        axis=1,
    )


def ma_crossfit_at_wells(wells, world, k: int = 16) -> np.ndarray:
    """Leave-one-fold-out Ma-like DTW at every water-table well (m).

    Fold ``f < N_FOLDS`` wells are scored against a field fitted without fold
    ``f``; the buffered-holdout wells against the field fitted on all six folds.
    """
    wt = water_table_wells(wells)
    folds = wt["fold"].to_numpy(int)
    rows = wt["row"].to_numpy(int)
    cols = wt["col"].to_numpy(int)
    out = np.empty(len(wt), float)
    for f in np.unique(folds):
        pool = training_pool(wells, exclude_fold=int(f) if f < N_FOLDS else None)
        field = _idw_planar(pool.x, pool.y, pool.dtw_obs, world, k=k)
        m = folds == f
        out[m] = field[rows[m], cols[m]]
    return out


def mad_by_band(pred, obs, dist) -> tuple[np.ndarray, np.ndarray]:
    """Median |pred - obs| (m) and n in each distance-to-stream band."""
    err = np.abs(np.asarray(pred, float) - np.asarray(obs, float))
    dist = np.asarray(dist, float)
    mad = np.empty(len(BINS))
    n = np.empty(len(BINS), int)
    for b, (lo, hi) in enumerate(BINS):
        m = (dist >= lo) & (dist < hi)
        n[b] = int(m.sum())
        mad[b] = float(np.median(err[m]))
    return mad, n


# --------------------------------------------------------------------------
# Paper register
# --------------------------------------------------------------------------


def _map_panel(fig, ax, arr, cmap, label, vmin=None, vmax=None, first: bool = False):
    im = ax.imshow(
        np.asarray(arr, float),
        origin="lower",
        extent=EXTENT,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    map_axes(
        ax,
        xlabel="Easting (km)",
        ylabel="Northing (km)" if first else None,
        x_step=10.0,
        y_step=10.0,
    )
    if not first:
        ax.set_yticklabels([])
    cax = ax.inset_axes([0.0, -0.42, 1.0, 0.07])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=6, labelpad=1.5)
    cb.ax.tick_params(labelsize=5, length=1.8, width=0.5, pad=1.0)
    cb.outline.set_linewidth(0.5)
    return im


def paper_figure(world, wells, fac_ws, fac_depth, bands) -> dict:
    fig = plt.figure(figsize=(WIDTH_2COL, 138 * MM))
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[40.0, 45.0, 45.0],
        hspace=0.30,
        left=0.055,
        right=0.985,
        top=0.99,
        bottom=0.06,
    )
    gs_maps = outer[0].subgridspec(1, 5, wspace=0.10)
    ax_sec = fig.add_subplot(outer[1])
    gs_skill = outer[2].subgridspec(1, 2, wspace=0.22)

    dem = np.asarray(world["dem"], float)
    axes = [fig.add_subplot(gs_maps[0, c]) for c in range(5)]

    # (a) DEM
    _map_panel(fig, axes[0], dem, "terrain", "Elevation (m)", first=True)
    # (b) log flow accumulation
    _map_panel(
        fig,
        axes[1],
        np.log1p(world["flow_acc"]),
        "viridis",
        "log (1 + contributing cells)",
    )
    # (c) stream network and channel heads
    ax = axes[2]
    heads = channel_heads(world)
    ax.add_collection(
        LineCollection(stream_segments(world), colors="#3b3b3b", linewidths=0.35)
    )
    ax.plot(
        world["X"][heads],
        world["Y"][heads],
        marker="o",
        ls="none",
        ms=1.8,
        mfc="none",
        mec=COLORS["fac"],
        mew=0.5,
    )
    map_axes(ax, xlabel="Easting (km)", ylabel=None, x_step=10.0, y_step=10.0)
    ax.set_yticklabels([])
    ax.legend(
        handles=[
            Line2D([], [], color="#3b3b3b", lw=0.6, label="Stream network"),
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=2.2,
                mfc="none",
                mec=COLORS["fac"],
                mew=0.5,
                label=f"Channel head (n = {int(heads.sum())})",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        frameon=False,
        fontsize=5,
        handlelength=1.2,
        labelspacing=0.4,
        borderpad=0.25,
    )
    # (d) FAC water surface as an elevation
    _map_panel(
        fig,
        axes[3],
        fac_ws,
        CMAPS["wte"],
        "FAC water-surface elevation (m)",
        vmin=1500.0,
        vmax=2300.0,
    )
    # (e) FAC-REM depth
    _map_panel(
        fig,
        axes[4],
        fac_depth,
        CMAPS["dtw"],
        "FAC-REM depth (m); dark = shallow",
        vmin=0.0,
        vmax=90.0,
    )

    for ax, letter in zip(axes, "abcde"):
        draw_section_lines(ax)
        panel_letter(ax, letter, x=0.04, y=0.96, fontsize=7)

    # ---------------- Row 2: section B-B' --------------------------------
    prof = profile_along(
        SECTION_B,
        {
            "dem": dem,
            "wte_true": world["wte_true"],
            "fac_ws": fac_ws,
            "dist": world["dist_to_stream"],
        },
        n=600,
    )
    d_km = prof["d"] / 1000.0
    near_stream = prof["dist"] < 1000.0
    ax_sec.fill_between(
        d_km,
        0.0,
        1.0,
        where=near_stream,
        transform=ax_sec.get_xaxis_transform(),
        color="#e9e9e9",
        lw=0.0,
        zorder=0,
    )
    ax_sec.plot(d_km, prof["dem"], color=COLORS["land"], lw=0.5, zorder=4)
    ax_sec.plot(d_km, prof["fac_ws"], color=COLORS["fac"], lw=1.1, zorder=3)
    ax_sec.plot(d_km, prof["wte_true"], color=COLORS["truth"], lw=0.8, zorder=3)

    near = wells_near_line(water_table_wells(wells), SECTION_B, 600.0)
    draw_wells_section(
        ax_sec, near, np.zeros(len(near), bool), tick_half_width=0.10, lw=0.5
    )
    ax_sec.set_xlim(0.0, d_km[-1])
    lo = min(
        prof["wte_true"].min(),
        prof["fac_ws"].min(),
        float((near["dem"] - near["drilled_depth"]).min()),
    )
    hi = max(prof["dem"].max(), prof["wte_true"].max())
    ax_sec.set_ylim(np.floor((lo - 25.0) / 50.0) * 50.0, hi + 130.0)
    ax_sec.set_xlabel("Distance along B–B′ (km)")
    ax_sec.set_ylabel(f"Elevation (m), {VE:g}× vertical")
    ax_sec.set_aspect(VE / 1000.0)
    add_ticks(ax_sec, x_step=2.0, y_step=200.0)
    annotate_ve(ax_sec)
    panel_letter(ax_sec, "f", x=0.008, y=0.96, fontsize=7)
    ax_sec.legend(
        handles=[
            Line2D([], [], color=COLORS["land"], lw=0.5, label="Land surface"),
            Line2D([], [], color=COLORS["fac"], lw=1.1, label="FAC-REM water surface"),
            Line2D([], [], color=COLORS["truth"], lw=0.8, label="True water table"),
            Patch(
                facecolor="#e9e9e9", edgecolor="none", label="Within 1 km of a stream"
            ),
        ]
        + well_ledger_handles(
            visible_label="", hidden_label=f"Well, label hidden (n = {len(near)})"
        )[1:],
        loc="upper center",
        ncol=3,
        fontsize=5.5,
        handlelength=1.4,
        borderpad=0.3,
    )

    # ---------------- Row 3: error against distance to stream -------------
    ax_w = fig.add_subplot(gs_skill[0, 0])
    ax_g = fig.add_subplot(gs_skill[0, 1])
    xs = np.arange(len(BINS), dtype=float)
    ymax = 1.35 * max(
        bands["wells"]["mirror"][0].max(), bands["grid"]["mirror"][0].max()
    )

    for ax, key, letter, title in (
        (ax_w, "wells", "g", "At water-table wells"),
        (ax_g, "grid", "h", "Over grid cells"),
    ):
        b = bands[key]
        for name, color, ls, lw, label in (
            ("mirror", COLORS["mirror"], LINESTYLES["mirror"], 0.8, "3 m mirror"),
            ("ma", COLORS["ma"], LINESTYLES["ma"], 0.9, "Ma-like DTW interpolation"),
            ("fac", COLORS["fac"], "-", 1.3, "FAC-REM"),
        ):
            mad, _ = b[name]
            ax.plot(
                xs,
                mad,
                color=color,
                ls=ls,
                lw=lw,
                marker="o" if name == "fac" else ("s" if name == "ma" else "^"),
                ms=3.0,
                mfc=color,
                mec=color,
                label=label,
            )
        n = b["fac"][1]
        for xi, ni in zip(xs, n):
            ax.annotate(
                f"{ni:,}",
                (xi, ymax * 0.80),
                ha="center",
                va="top",
                fontsize=5,
            )
        ax.annotate(
            "n =",
            (-0.40, ymax * 0.80),
            ha="left",
            va="top",
            fontsize=5,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(BIN_LABELS, fontsize=5.5)
        ax.set_xlim(-0.45, len(BINS) - 0.55)
        ax.set_ylim(0.0, ymax)
        add_ticks(ax, y_step=10.0)
        ax.set_xlabel("Distance to nearest stream (m)")
        ax.set_title(title, fontsize=6.5)
        panel_letter(ax, letter, x=0.025, y=0.975, fontsize=7)
    ax_w.set_ylabel("Median |error| in depth to water (m)")
    ax_g.set_yticklabels([])
    ax_w.legend(
        loc="upper center", ncol=2, fontsize=5.5, handlelength=1.8, borderpad=0.3
    )

    return save(fig, "fig04_fac_rem_paper", TOY_DIR)


# --------------------------------------------------------------------------
# Presentation register
# --------------------------------------------------------------------------


TRUTH_3D = "#4d4d4d"  # opaque dark grey for the true water table in 3-D
UNSAT_3D = "#d8c9a4"  # cut-face fill above the water table
SAT_3D = "#c9dcec"  # cut-face fill below the water table
SLAB_X0 = 8000.0  # m: the cut face, on B-B'
SLAB_X1 = 13000.0  # m: the slab's far side


def _overlay_legend(png_path, entries, dpi: int = 200):
    """Paste a style-compliant legend (black text, framed) onto a 3-D render.

    ``vtkLegendBoxActor`` paints its label text in the swatch colour, which the
    style guide forbids, so the legend is drawn in matplotlib and composited into
    the top-left of the screenshot.
    """
    img = plt.imread(png_path)
    fig = plt.figure(figsize=(2.45, 0.16 * len(entries) + 0.14), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.legend(
        handles=[
            Patch(facecolor=c, edgecolor="black", lw=0.4, label=lab)
            for lab, c in entries
        ],
        loc="center left",
        fontsize=7,
        frameon=True,
        borderpad=0.4,
        handlelength=1.4,
        labelspacing=0.35,
    )
    fig.canvas.draw()
    leg = np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0
    plt.close(fig)
    out = np.array(img[..., :3], dtype=float)
    h, w = leg.shape[:2]
    out[30 : 30 + h, 30 : 30 + w] = leg
    plt.imsave(png_path, np.clip(out, 0.0, 1.0))
    return png_path


def _profile_tube(pl, line, z, color, radius: float = 90.0, n: int = 400):
    """A tube tracing one surface along a section line -- the cut-face trace."""
    x0, y0, x1, y1 = (float(v) for v in line)
    t = np.linspace(0.0, 1.0, n)
    pts = np.column_stack(
        [x0 + t * (x1 - x0), y0 + t * (y1 - y0), r3.zs(np.asarray(z, float))]
    )
    cells = np.column_stack(
        [np.full(n - 1, 2), np.arange(n - 1), np.arange(1, n)]
    ).ravel()
    poly = pv.PolyData(pts, lines=cells).tube(radius=radius, n_sides=10)
    return pl.add_mesh(
        poly, color=color, show_scalar_bar=False, ambient=0.5, diffuse=0.5
    )


def render_3d(world, fac_ws) -> list:
    """Two block diagrams: the B-B' cutaway, and the two sheets over the basin."""
    dem = np.asarray(world["dem"], float)
    truth = np.asarray(world["wte_true"], float)
    prof = profile_along(SECTION_B, {"dem": dem, "wte": truth, "fac": fac_ws}, n=400)
    X = np.asarray(world["X"], float)
    slab = (X >= SLAB_X0) & (X <= SLAB_X1)
    base = float(truth.min()) - 90.0
    paths = []

    # (a) A 5 km slab of the basin cut open on B-B': the ground skin above, the
    # two water surfaces inside it, and both traced on the cut face.
    pl = r3.standard_plotter(scale_bar=False)
    r3.add_vertical_scale_bar(pl, at=(SLAB_X0 - 1400.0, -1200.0, 1520.0))
    r3.cutaway(
        pl,
        dem,
        slab,
        color=LAND_3D,
        opacity=0.35,
        ambient=0.55,
        diffuse=0.55,
        specular=0.0,
    )
    r3.fence_section(
        pl,
        SECTION_B,
        prof["dem"],
        prof["wte"],
        color=UNSAT_3D,
        opacity=1.0,
        ambient=0.75,
        diffuse=0.25,
        specular=0.0,
    )
    r3.fence_section(
        pl,
        SECTION_B,
        prof["wte"],
        np.full(prof["wte"].shape, base),
        color=SAT_3D,
        opacity=1.0,
        ambient=0.75,
        diffuse=0.25,
        specular=0.0,
    )
    r3.cutaway(pl, truth, slab, color=TRUTH_3D, opacity=1.0, ambient=0.35, specular=0.0)
    r3.cutaway(
        pl, fac_ws, slab, color=COLORS["fac"], opacity=0.6, ambient=0.4, specular=0.0
    )
    _profile_tube(pl, SECTION_B, prof["wte"], TRUTH_3D, radius=150.0)
    _profile_tube(pl, SECTION_B, prof["fac"], COLORS["fac"], radius=150.0)
    r3.set_camera(pl, azimuth_deg=248.0, elevation_deg=18.0, zoom=1.5)
    p = r3.shoot(pl, "fig04_fac_rem_3d_a", camera=False)
    _overlay_legend(
        p,
        [
            ("Land surface", LAND_3D),
            ("Unsaturated zone (cut face)", UNSAT_3D),
            ("Saturated zone (cut face)", SAT_3D),
            ("FAC-REM water surface", COLORS["fac"]),
            ("True water table", TRUTH_3D),
        ],
    )
    paths.append(p)

    # (b) The whole basin under its land skin, with the same slab of water
    # surfaces beneath it, from the series' standard bearing.
    pl = r3.standard_plotter()
    r3.land_surface(pl, world, opacity=0.30)
    r3.cutaway(pl, truth, slab, color=TRUTH_3D, opacity=1.0, ambient=0.35, specular=0.0)
    r3.cutaway(
        pl, fac_ws, slab, color=COLORS["fac"], opacity=0.6, ambient=0.4, specular=0.0
    )
    r3.fence_section(
        pl,
        SECTION_B,
        prof["dem"],
        prof["wte"],
        color=UNSAT_3D,
        opacity=1.0,
        ambient=0.75,
        diffuse=0.25,
        specular=0.0,
    )
    r3.fence_section(
        pl,
        SECTION_B,
        prof["wte"],
        np.full(prof["wte"].shape, base),
        color=SAT_3D,
        opacity=1.0,
        ambient=0.75,
        diffuse=0.25,
        specular=0.0,
    )
    r3.stream_tubes(pl, world, radius=55.0)
    _profile_tube(pl, SECTION_B, prof["wte"], TRUTH_3D, radius=150.0)
    _profile_tube(pl, SECTION_B, prof["fac"], COLORS["fac"], radius=150.0)
    r3.set_camera(pl, elevation_deg=20.0, zoom=1.35)
    p = r3.shoot(pl, "fig04_fac_rem_3d_b", camera=False)
    _overlay_legend(
        p,
        [
            ("Land surface", LAND_3D),
            ("Unsaturated zone (cut face)", UNSAT_3D),
            ("Saturated zone (cut face)", SAT_3D),
            ("FAC-REM water surface", COLORS["fac"]),
            ("True water table", TRUTH_3D),
            ("Stream network", COLORS["r"]),
        ],
    )
    paths.append(p)
    return paths


# --------------------------------------------------------------------------


def main() -> None:
    use_style()
    world, wells = load_world()
    fac_ws, fac_depth = fac_rem(world, strip_radius_m=STRIP_RADIUS_M)

    wt = water_table_wells(wells)
    rows = wt["row"].to_numpy(int)
    cols = wt["col"].to_numpy(int)
    obs = wt["dtw_obs"].to_numpy(float)
    d_wells = wt["dist_to_stream"].to_numpy(float)

    bands = {
        "wells": {
            "fac": mad_by_band(fac_depth[rows, cols], obs, d_wells),
            "ma": mad_by_band(ma_crossfit_at_wells(wells, world), obs, d_wells),
            "mirror": mad_by_band(np.full(len(wt), MIRROR_DEPTH), obs, d_wells),
        },
        "grid": {},
    }
    truth_dtw = np.asarray(world["dtw_true"], float).ravel()
    d_grid = np.asarray(world["dist_to_stream"], float).ravel()
    ma_grid = ma_like(wells, world)
    bands["grid"] = {
        "fac": mad_by_band(fac_depth.ravel(), truth_dtw, d_grid),
        "ma": mad_by_band(ma_grid.ravel(), truth_dtw, d_grid),
        "mirror": mad_by_band(np.full(truth_dtw.size, MIRROR_DEPTH), truth_dtw, d_grid),
    }

    for key in ("wells", "grid"):
        print(f"\n|error| in DTW by distance to stream -- {key} (m)")
        header = "  band        " + "".join(f"{lab:>11s}" for lab in BIN_LABELS)
        print(header)
        for name in ("fac", "ma", "mirror"):
            mad, n = bands[key][name]
            print(f"  {name:<10s}  " + "".join(f"{v:11.2f}" for v in mad))
        print("  n           " + "".join(f"{v:11d}" for v in bands[key]["fac"][1]))

    heads = channel_heads(world)
    print(
        f"\nchannel heads: {int(heads.sum())} of {int(np.asarray(world['streams']).sum())} stream cells"
    )
    print(f"cell {CELL:g} m, grid {NX} columns")

    paths = paper_figure(world, wells, fac_ws, fac_depth, bands)
    print(f"\nwrote {paths['png']}")
    print(f"wrote {paths['pdf']}")
    for p in render_3d(world, fac_ws):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
