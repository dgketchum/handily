"""F1 -- the hidden surface: the target is a continuous surface observed only
where wells puncture it, and depth to water is the gap between two surfaces.

Run as::

    uv run python -m utils.figures.fig01_surface

Writes into ``/data/ssd2/handily/figures/toy/``:

``fig01_surface_paper.{png,pdf}``
    Two stacked sections (A-A' along the valley, B-B' across it) with the land
    surface, the true water table, the saturated zone, the depth-to-water gap
    and the well ledger, plus a locator map of true DTW with both section lines.
``fig01_surface_3d_a.png``
    Presentation block: translucent land skin over the water table coloured by
    true DTW, stream tubes, well sticks.
``fig01_surface_3d_b.png``
    The same block cut away along B-B' (western half of the land skin kept), so
    the gap between the ground and the water table is a slab on the cut face.

Nothing here writes or rebuilds a cache and no shared module is modified.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from utils.figures import render3d as r3
from utils.figures.fig_common import (
    CMAPS,
    COLORS,
    EXTENT,
    LAND_3D,
    SECTION_A,
    SECTION_B,
    TOY_DIR,
    VE,
    WIDTH_2COL,
    add_ticks,
    draw_section_lines,
    draw_wells_section,
    map_axes,
    panel_letter,
    profile_along,
    save,
    use_style,
    wells_near_line,
)
from utils.figures.toy_priors import water_table_wells
from utils.figures.toy_world import load_world

#: Half-width of the swath of wells projected onto each section line (m).
#: Chosen so both sections carry 15-30 wells (A-A': 15, B-B': 27).
WELL_BUFFER_M = 400.0

#: Samples along each section profile.
N_PROFILE = 900

#: Fills. Saturated zone = below the water table; the gap = the unsaturated
#: column between the ground and the water table, i.e. depth to water itself.
SAT_FILL = "#b9d3ea"
GAP_FILL = "#e6eff8"
WET_COLOR = "#0072B2"

#: Vertical exaggeration per section. B-B' keeps the series value (VE = 4); the
#: 24 km valley-axis section A-A' carries only ~160 m of relief over 23 km, so at
#: 4x it is a 5 mm ribbon on a 190 mm page and the gap it exists to show is
#: invisible. Its own exaggeration is printed on its vertical axis title.
VE_A = 25.0
VE_B = VE

#: DTW colour limits shared by the map and the 3-D water table (m).
DTW_CLIM = (0.0, 100.0)


# --------------------------------------------------------------------------
# Paper register
# --------------------------------------------------------------------------


def _section_data(world, wells, line):
    """Profiles, wells and vertical limits for one section, before any drawing."""
    p = profile_along(
        line,
        {
            "dem": world["dem"],
            "wte": world["wte_true"],
            "wet": np.asarray(world["wet"], float),
        },
        n=N_PROFILE,
    )
    sec = wells_near_line(wells, line, WELL_BUFFER_M)
    bottoms = (sec["dem"] - sec["drilled_depth"]).to_numpy(float)
    floor = float(min(p["wte"].min(), bottoms.min())) - 12.0
    ceil = float(p["dem"].max()) + 12.0
    return p, sec, floor, ceil


def _section(ax, data, letter, label_lo, label_hi, ve):
    """Draw one land-surface / water-table section with the saturated fill."""
    p, sec, floor, ceil = data
    x = p["d"] / 1000.0
    dem = p["dem"]
    wte = p["wte"]

    ax.fill_between(x, floor, wte, color=SAT_FILL, lw=0, zorder=1)
    ax.fill_between(x, wte, dem, where=dem > wte, color=GAP_FILL, lw=0, zorder=1)

    ax.plot(x, dem, color=COLORS["land"], lw=0.5, zorder=4)
    ax.plot(x, wte, color=COLORS["truth"], lw=0.9, zorder=4)

    # Where the two surfaces meet the water table is at the ground: the stream.
    wet = p["wet"] > 0.5
    ax.plot(
        np.where(wet, x, np.nan),
        np.where(wet, dem, np.nan),
        color=WET_COLOR,
        lw=1.8,
        solid_capstyle="butt",
        zorder=5,
    )

    draw_wells_section(
        ax,
        sec,
        np.ones(len(sec), dtype=bool),
        tick_half_width=0.011 * (x.max() - x.min()),
        lw=0.55,
    )

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(floor, ceil)
    ax.set_aspect(ve / 1000.0)
    ax.set_ylabel(f"Elevation (m), {ve:g}× vertical")
    ax.set_xlabel(f"Distance along {label_lo}–{label_hi} (km)")
    add_ticks(ax, x_step=5.0 if x.max() > 20 else 2.0)
    panel_letter(ax, letter)
    for lab, xf, ha in ((label_lo, 0.0, "left"), (label_hi, 1.0, "right")):
        ax.annotate(
            lab,
            (xf, 1.0),
            xycoords="axes fraction",
            xytext=(0, 2),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=7.0,
        )
    ax.annotate(
        f"n = {len(sec)} wells within {WELL_BUFFER_M:.0f} m of the line",
        (0.5, 1.0),
        xycoords="axes fraction",
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6.0,
    )
    return float(np.nanmin(dem - wte)), float(np.nanmax(dem - wte))


def _locator(ax, world, wells):
    im = ax.imshow(
        np.asarray(world["dtw_true"], float),
        origin="lower",
        extent=EXTENT,
        cmap=CMAPS["dtw"],
        vmin=DTW_CLIM[0],
        vmax=DTW_CLIM[1],
        interpolation="nearest",
    )
    ax.plot(
        wells["x"].to_numpy(float),
        wells["y"].to_numpy(float),
        marker="o",
        ls="none",
        ms=1.5,
        mfc="black",
        mec="black",
        mew=0.3,
        zorder=6,
    )
    draw_section_lines(ax)
    map_axes(ax)
    add_ticks(ax)
    panel_letter(ax, "c")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.032, pad=0.02, extend="max")
    cb.set_label("True depth to water (m); dark = shallow")
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.6)
    return im


def _legend_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return [
        Line2D([], [], color=COLORS["land"], lw=0.5, label="Land surface"),
        Line2D([], [], color=COLORS["truth"], lw=0.9, label="True water table"),
        Patch(
            facecolor=GAP_FILL,
            edgecolor="black",
            lw=0.4,
            label="Depth to water (the gap)",
        ),
        Patch(facecolor=SAT_FILL, edgecolor="black", lw=0.4, label="Saturated zone"),
        Line2D([], [], color=WET_COLOR, lw=1.8, label="Surfaces meet (stream)"),
        Line2D(
            [],
            [],
            color="black",
            lw=0.55,
            marker="o",
            ms=2.6,
            mfc="black",
            mec="black",
            mew=0.5,
            label="Well, tick at observed level",
        ),
    ]


#: Fraction of the figure width given to the locator map's own column.
MAP_COL_FRAC = 0.44


def _drawn_height(data, ve, width_frac=1.0):
    """Height a section occupies, as a fraction of the figure width, at ``ve``."""
    p, _, floor, ceil = data
    span_m = float(p["d"].max() - p["d"].min())
    return width_frac * ve * (ceil - floor) / span_m


def paper_figure(world, wells):
    use_style()
    da = _section_data(world, wells, SECTION_A)
    db = _section_data(world, wells, SECTION_B)

    # Row heights follow the drawn aspect of each panel, so the equal-aspect
    # axes fill their slots instead of floating in white space.
    h_a = _drawn_height(da, VE_A)
    h_b = _drawn_height(db, VE_B)
    h_map = MAP_COL_FRAC * (EXTENT[3] - EXTENT[2]) / (EXTENT[1] - EXTENT[0])

    left, right, top, bottom = 0.070, 0.985, 0.955, 0.075
    plot_w = WIDTH_2COL * (right - left)
    pad_in = 0.34  # per-gap allowance for the x label + section end labels
    fig_h = ((h_a + h_b + h_map) * plot_w + 2 * pad_in + 0.45) / (top - bottom)

    fig = plt.figure(figsize=(WIDTH_2COL, fig_h))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[h_a, h_b, h_map],
        width_ratios=[MAP_COL_FRAC, 1.0 - MAP_COL_FRAC],
        hspace=pad_in / (((h_a + h_b + h_map) / 3.0) * plot_w),
        wspace=0.10,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_m = fig.add_subplot(gs[2, 0])
    ax_l = fig.add_subplot(gs[2, 1])

    a_lo, a_hi = _section(ax_a, da, "a", "A", "A′", VE_A)
    b_lo, b_hi = _section(ax_b, db, "b", "B", "B′", VE_B)
    _locator(ax_m, world, wells)

    ax_l.axis("off")
    ax_l.legend(handles=_legend_handles(), loc="upper left", frameon=True)

    paths = save(fig, "fig01_surface_paper", TOY_DIR)
    plt.close(fig)
    return paths, da[1], db[1], (a_lo, a_hi), (b_lo, b_hi)


# --------------------------------------------------------------------------
# Presentation register
# --------------------------------------------------------------------------


def _stick_selection(wells, n_extra=34, seed=3):
    """Wells on the two section swaths plus a scatter of others, for the block."""
    on_a = wells_near_line(wells, SECTION_A, WELL_BUFFER_M).index
    on_b = wells_near_line(wells, SECTION_B, WELL_BUFFER_M).index
    keep = set(on_a) | set(on_b)
    rest = wells.drop(index=list(keep))
    extra = rest.sample(min(n_extra, len(rest)), random_state=seed).index
    return wells.loc[sorted(keep | set(extra))]


def _masked_streams(world, keep):
    """A shallow copy of ``world`` whose stream network is clipped to ``keep``."""
    out = dict(world)
    out["streams"] = np.asarray(world["streams"], bool) & np.asarray(keep, bool)
    return out


def _scale_bar_outside(pl):
    """Put the exaggeration bar off the block: the SW corner ground is at ~1660 m,
    so the shared default position buries the bar and its label inside the DEM."""
    r3.add_vertical_scale_bar(pl, at=(EXTENT[0] - 2200.0, EXTENT[2] + 1200.0, 1500.0))


#: Base of the rendered earth block (true m), below the lowest water table.
BLOCK_FLOOR_M = 1340.0

#: Half-cell inset so a wall hung on a domain edge stays inside the grid.
_IN = 0.6 * 100.0


def _edge(side: str, x_lo: float | None = None, x_hi: float | None = None):
    """A domain-edge line for a block wall; ``x_lo``/``x_hi`` clip the E-W edges."""
    x0 = (EXTENT[0] + _IN) if x_lo is None else x_lo
    x1 = (EXTENT[1] - _IN) if x_hi is None else x_hi
    y0, y1 = EXTENT[2] + _IN, EXTENT[3] - _IN
    return {
        "south": (x0, y0, x1, y0),
        "north": (x0, y1, x1, y1),
        "west": (x0, y0, x0, y1),
        "east": (x1, y0, x1, y1),
    }[side]


def _wall(pl, world, line, floor=BLOCK_FLOOR_M, sand=True):
    """Hang one block wall: pale sand above the water table, blue below it.

    ``sand=False`` draws only the saturated part, for edges of the block where
    the ground has been cut away and the water table is the exposed top.

    This is what makes the register read in one second -- on a bare pair of
    surfaces a ~20 m depth to water is 8 % of the block's 946 m of relief and
    the two sheets look coplanar, whereas a wall turns the same gap into a
    two-tone slab whose lower half is unmistakably water-bearing ground.
    """
    p = profile_along(
        line, {"dem": world["dem"], "wte": world["wte_true"]}, n=N_PROFILE
    )
    common = dict(opacity=1.0, ambient=0.5, diffuse=0.55, specular=0.0)
    if sand:
        r3.fence_section(pl, line, p["dem"], p["wte"], color=LAND_3D, **common)
    r3.fence_section(
        pl,
        line,
        p["wte"],
        np.full_like(p["wte"], float(floor)),
        color=SAT_FILL,
        **common,
    )


def _add_water_table(pl, world, show_bar=True):
    return r3.surface(
        pl,
        world["wte_true"],
        scalars=np.asarray(world["dtw_true"], float),
        scalar_name="Depth to water (m)",
        cmap=CMAPS["dtw"],
        clim=DTW_CLIM,
        show_scalar_bar=show_bar,
        scalar_bar_title="Depth to water (m)\ndark = shallow",
        ambient=0.30,
        diffuse=0.75,
        specular=0.0,
    )


def _sticks(pl, sel):
    r3.well_sticks(
        pl,
        sel,
        np.ones(len(sel), dtype=bool),
        radius=70.0,
        level_radius=140.0,
        level_thickness=32.0,
    )


def block_figure(world, wells, sticks, elevation_deg=26.0):
    pl = r3.standard_plotter(scale_bar=False)
    _scale_bar_outside(pl)
    r3.land_surface(pl, world)
    _add_water_table(pl, world)
    for side in ("south", "east", "west", "north"):
        _wall(pl, world, _edge(side))
    r3.stream_tubes(pl, world, radius=60.0)
    _sticks(pl, sticks)
    r3.set_camera(pl, elevation_deg=elevation_deg)
    return r3.shoot(pl, "fig01_surface_3d_a", camera=False)


def _water_table_trace(pl, world, line, radius=55.0):
    """A black tube along the water table on a cut face -- the 3-D counterpart of
    the section's solid black water-table line, so the gap stays legible where
    the fill is only a few metres thick."""
    p = profile_along(line, {"wte": world["wte_true"]}, n=200)
    pts = np.column_stack([p["x"], p["y"], r3.zs(p["wte"])])
    poly = r3.pv.Spline(pts, 400).tube(radius=radius, n_sides=8)
    return pl.add_mesh(poly, color=COLORS["truth"], show_scalar_bar=False)


def cutaway_figure(world, wells, sticks, elevation_deg=25.0, azimuth_deg=145.0):
    x_cut = SECTION_B[0]
    keep = r3.half_mask(SECTION_B, side="left")  # western half of the land skin
    pl = r3.standard_plotter(scale_bar=False)
    _scale_bar_outside(pl)

    r3.cutaway(
        pl,
        world["dem"],
        keep,
        color=LAND_3D,
        opacity=0.62,
        ambient=0.55,
        diffuse=0.55,
        specular=0.0,
    )
    _add_water_table(pl, world)
    # Intact block, west of the cut: ground over water.
    for side in ("south", "north"):
        _wall(pl, world, _edge(side, x_hi=x_cut))
    _wall(pl, world, _edge("west"))
    _wall(pl, world, SECTION_B)  # the cut face along B-B'
    _water_table_trace(pl, world, SECTION_B)
    # Cut block, east of it: the water table is the exposed top of a solid body.
    for side in ("south", "north"):
        _wall(pl, world, _edge(side, x_lo=x_cut), sand=False)
    _wall(pl, world, _edge("east"), sand=False)
    r3.stream_tubes(pl, _masked_streams(world, keep), radius=60.0)

    # Sticks on both halves: east of the cut the ground is gone, so each stick
    # hangs from where the land surface was down through the water table and the
    # length above the water table is the depth to water, drawn as a rod.
    _sticks(pl, sticks)
    r3.set_camera(pl, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)
    return r3.shoot(pl, "fig01_surface_3d_b", camera=False)


# --------------------------------------------------------------------------


def main() -> None:
    world, all_wells = load_world()
    wells = water_table_wells(all_wells)

    paths, sec_a, sec_b, gap_a, gap_b = paper_figure(world, wells)
    print("paper :", paths["png"])
    print("paper :", paths["pdf"])

    sticks = _stick_selection(wells)
    print("3-D   :", block_figure(world, wells, sticks))
    print("3-D   :", cutaway_figure(world, wells, sticks))

    dtw = np.asarray(world["dtw_true"], float)
    print(f"buffer            {WELL_BUFFER_M:.0f} m")
    print(f"wells on A-A'     {len(sec_a)}")
    print(f"wells on B-B'     {len(sec_b)}")
    print(f"grid true DTW     {dtw.min():.1f} to {dtw.max():.1f} m")
    print(f"A-A' gap range    {gap_a[0]:.1f} to {gap_a[1]:.1f} m")
    print(f"B-B' gap range    {gap_b[0]:.1f} to {gap_b[1]:.1f} m")
    print(f"colour limits     {DTW_CLIM[0]:.0f} to {DTW_CLIM[1]:.0f} m")


if __name__ == "__main__":
    main()
