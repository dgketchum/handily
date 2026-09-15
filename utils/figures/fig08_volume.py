"""F8: the capstone volume -- one cutaway block carrying the whole series.

Run as::

    uv run python -m utils.figures.fig08_volume            # both registers
    uv run python -m utils.figures.fig08_volume --only 3d  # presentation only

Outputs (all in ``/data/ssd2/handily/figures/toy/``)

``fig08_volume_3d.png``
    the title-slide image: translucent land skin cut away over the south-east
    quadrant, the assimilated water table beneath it coloured by depth to water,
    stream tubes, a smoothed permanent-water patch, the well ledger, and a
    textured cut face along B-B' carrying every layer the series introduced.
``fig08_volume_3d_fence.png``
    the B-B' cut face on its own, large: on the block it is a narrow ribbon, so
    a slide carries the same section beside the volume at a readable size.
``fig08_volume_3d_legend.png``
    a compact matplotlib legend a slide can carry beside the volume.
``fig08_volume_paper.png`` / ``.pdf``
    the 190 mm companion: the F3 leak proof, the F6 gate, and the F7
    assimilation difference and density curve, on the series' conventions.

The Laplace scale
-----------------
The +/-b band is the model's own per-cell Laplace scale, ``fields["sigma"]`` from
``toy_model.load_stack()``. It is drawn only on the B-B' cut face and in the
fence PNG: as a pair of full-grid sheets in the block it was invisible at any
opacity that did not also hide the water table.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.figures import render3d as r3
from utils.figures.fig_common import (
    CELL,
    CMAPS,
    LAND_3D_OPACITY,
    NX,
    NY,
    COLORS,
    EXTENT,
    SECTION_A,
    SECTION_B,
    TOY_DIR,
    WIDTH_2COL,
    add_ticks,
    draw_section_lines,
    draw_wells_map,
    draw_wells_section,
    map_axes,
    panel_letter,
    profile_along,
    resid_norm,
    save,
    use_style,
    well_ledger_handles,
    wells_near_line,
)
from utils.figures.toy_model import (
    EXPERT_NAMES,
    assimilate,
    density_curve,
    load_stack,
    score,
)
from utils.figures.toy_priors import residual_hist_data, water_table_wells

#: The base (pre-assimilation) prediction, a muted tint of the handily orange, so
#: the two prediction lines are read as the same quantity at two stages.
COLOR_BASE = "#F2C46B"
#: The +/-b band on the cut face: the prediction's hue, no edge, very pale.
COLOR_ENVELOPE = "#F7D9A0"
#: Permanent water -- the dark end of the DTW ramp.
COLOR_WATER = "#08306B"

#: DTW colour limits shared by the volume and the legend ramp (m).
DTW_CLIM = (0.0, 100.0)

#: Buffer either side of B-B' for the wells drawn on the cut face (m).
FENCE_BUFFER_M = 900.0

#: Elevation of the block's flat base (m). Below every layer the fence carries.
FLOOR_M = 1450.0
#: Cut rock on the block's walls, and the base plate: the land sand, darkened, so
#: a wall is read as a cut surface and not as more ground.
COLOR_WALL = "#DCCFAE"
COLOR_FLOOR = "#C4B692"
#: Inset of the block walls from the domain edge (m), so a wall samples real cells.
EDGE_M = 60.0
#: Eastern limit of the excavated corner (m); the playa sub-basin lies beyond it.
NOTCH_X_MAX = 16000.0
TRENCH_X_MAX = 12500.0


# --------------------------------------------------------------------------
# The cut face, rendered in matplotlib and mapped onto the fence as a texture
# --------------------------------------------------------------------------


def _fence_layers(world: dict, priors: dict, fields: dict, asm: dict, b_field, n=400):
    """Every section layer along B-B', plus the profile geometry."""
    grids = {
        "dem": world["dem"],
        "truth": world["wte_true"],
        "r": priors["r_inference"],
        "fac": priors["fac_ws"],
        "deep": fields["deep_wte"],
        "base": fields["wte"],
        "asm": asm["wte"],
        "b": b_field,
    }
    return profile_along(SECTION_B, grids, n=n)


def fence_texture(prof: dict, wells, bottom: float, dpi: int = 300) -> pv.Texture:
    """Render the B-B' section into a texture for the 3-D cut face.

    The fence quad is parameterised by distance along the line and by the
    fraction of the way from ``bottom`` (a flat floor) up to the land surface,
    so every elevation is drawn at ``(z - bottom) / (top - bottom)`` rather than
    at ``z``. Layers above the land surface (the deep prior does sit above ground
    in the valley) are clipped by the axes, which is the honest thing to do: the
    cut face only exists below the ground.
    """
    s = prof["d"] / 1000.0
    top = prof["dem"]
    span = top - bottom

    def frac(z):
        return (np.asarray(z, float) - bottom) / span

    fig = plt.figure(figsize=(13.0, 5.2), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(s[0], s[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    ax.set_facecolor(COLOR_WALL)
    ax.add_patch(
        plt.Rectangle((s[0], 0.0), s[-1] - s[0], 1.0, color=COLOR_WALL, zorder=0)
    )

    lo = frac(prof["asm"] - prof["b"])
    hi = frac(prof["asm"] + prof["b"])
    ax.fill_between(s, lo, hi, color=COLOR_ENVELOPE, lw=0, zorder=1)

    layers = [
        ("deep", prof["deep"], COLORS["deep"], "-", 3.0),
        ("fac", prof["fac"], COLORS["fac"], "-", 3.0),
        ("r", prof["r"], COLORS["r"], "-", 3.0),
        ("base", prof["base"], COLOR_BASE, (0, (5, 2)), 3.0),
        ("asm", prof["asm"], COLORS["pred"], "-", 4.5),
        ("truth", prof["truth"], COLORS["truth"], "-", 3.0),
    ]
    for _, z, col, ls, lw in layers:
        ax.plot(s, frac(z), color=col, ls=ls, lw=lw, zorder=3, solid_capstyle="round")
    ax.plot(s, frac(top), color="black", lw=3.5, zorder=5)

    # Well sticks on the face, in the ledger convention: from the land surface
    # down to the observed water level, filled for sources, hollow for validation.
    # The stretch factor is the well's own, interpolated along the section --
    # ``frac`` above is vectorised over the whole profile.
    xw = wells["d_along"].to_numpy(float) / 1000.0
    top_w = np.interp(xw, s, top)
    tw = (wells["dem"].to_numpy(float) - bottom) / (top_w - bottom)
    lw_ = (wells["wte_obs"].to_numpy(float) - bottom) / (top_w - bottom)
    for xi, ti, li, vis in zip(xw, tw, lw_, wells["is_source"].to_numpy(bool)):
        ax.plot([xi, xi], [ti, li], color="black", lw=3.0 if vis else 5.0, zorder=6)
        if not vis:
            ax.plot([xi, xi], [ti, li], color="white", lw=2.6, zorder=7)
        ax.plot([xi - 0.13, xi + 0.13], [li, li], color="black", lw=3.0, zorder=8)
        ax.plot(
            [xi],
            [li],
            marker="o",
            ms=6,
            mfc="black" if vis else "white",
            mec="black",
            mew=1.6,
            zorder=9,
        )

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return pv.Texture(np.ascontiguousarray(img))


def textured_fence(pl, line, top, bottom_z: float, texture: pv.Texture, x_shift=0.0):
    """A vertical quad along ``line`` from a flat floor to ``top``, texture-mapped."""
    top = np.asarray(top, float)
    n = top.size
    x0, y0, x1, y1 = (float(v) for v in line)
    t = np.linspace(0.0, 1.0, n)
    px = x0 + t * (x1 - x0) + x_shift
    py = y0 + t * (y1 - y0)
    m = 2
    frac = np.linspace(0.0, 1.0, m)[None, :]
    zz = bottom_z + frac * (top - bottom_z)[:, None]
    xx = np.repeat(px[:, None], m, axis=1)
    yy = np.repeat(py[:, None], m, axis=1)
    grid = pv.StructuredGrid(xx, yy, r3.zs(zz))
    uu = np.repeat(t[:, None], m, axis=1)
    vv = np.repeat(frac, n, axis=0)
    grid.active_texture_coordinates = np.column_stack(
        [uu.ravel(order="F"), vv.ravel(order="F")]
    ).astype(np.float32)
    return pl.add_mesh(
        grid,
        texture=texture,
        lighting=True,
        ambient=0.85,
        diffuse=0.25,
        specular=0.0,
        show_scalar_bar=False,
    )


# --------------------------------------------------------------------------
# The volume
# --------------------------------------------------------------------------


def _wall(pl, line, world: dict, color=COLOR_WALL):
    """An opaque curtain along ``line`` from :data:`FLOOR_M` up to the land surface."""
    prof = profile_along(line, {"dem": np.asarray(world["dem"], float)}, n=260)
    return r3.fence_section(
        pl,
        line,
        prof["dem"],
        np.full_like(prof["dem"], FLOOR_M),
        color=color,
        opacity=1.0,
        lighting=True,
        ambient=0.55,
        diffuse=0.5,
        specular=0.0,
        show_scalar_bar=False,
    )


def block_walls(pl, world: dict, cut: str) -> None:
    """Close the kept block into a solid volume with a cut-away corner.

    The whole corner is removed -- ground, unsaturated zone and water table alike
    -- so the notch is an empty step down to the base plate and the two cut faces
    (A-A' plain, B-B' textured) stand at their full thickness. Walling the notch
    only up to the water table instead would leave the saturated block in place
    and the water surface would hide the faces, which is what made the first two
    passes of this figure read as a draped sheet.
    """
    x0, x1 = EXTENT[0] + EDGE_M, EXTENT[1] - EDGE_M
    y0, y1 = EXTENT[2] + EDGE_M, EXTENT[3] - EDGE_M
    bx, ay, nx, tx = SECTION_B[0], SECTION_A[1], NOTCH_X_MAX, TRENCH_X_MAX

    _wall(pl, (x0, y0, x0, y1), world)  # west
    _wall(pl, (x0, y0, bx, y0), world)  # south, west of B-B'
    if cut == "trench":
        _wall(pl, (x0, y1, bx, y1), world)  # north, west of B-B'
        _wall(pl, (tx, y0, tx, y1), world)  # far wall of the trench
        _wall(pl, (tx, y0, x1, y0), world)  # south, east of the trench
        _wall(pl, (tx, y1, x1, y1), world)  # north, east of the trench
        _wall(pl, (x1, y0, x1, y1), world)  # east
    elif cut == "notch":
        _wall(pl, (x0, y1, x1, y1), world)  # north
        _wall(pl, (x1, y0, x1, y1), world)  # east
        _wall(pl, (nx, y0, x1, y0), world)  # south, east of the bite
        _wall(pl, (bx, ay, nx, ay), world)  # A-A' cut face
        _wall(pl, (nx, y0, nx, ay), world)  # far wall of the bite
    else:
        _wall(pl, (x0, y1, x1, y1), world)  # north
        _wall(pl, (x1, ay, x1, y1), world)  # east, north of A-A'
        _wall(pl, (bx, ay, x1, ay), world)  # A-A' cut face


def keep_mask(mode: str) -> np.ndarray:
    """Which cells of the block survive the cutaway.

    ``quadrant`` removes everything south of A-A' and east of B-B'; ``notch``
    stops that bite at :data:`NOTCH_X_MAX`, so the closed playa sub-basin in the
    east stays in the block and the excavated floor does not take over the image.
    Both leave the B-B' cut face turned toward the south-east camera.
    """
    west = r3.half_mask(SECTION_B, "left")
    north = r3.half_mask(SECTION_A, "left")
    if mode == "quadrant":
        return west | north
    xs = (np.arange(NX) + 0.5) * CELL
    if mode == "notch":
        far_east = np.repeat((xs > NOTCH_X_MAX)[None, :], NY, axis=0)
        return west | north | far_east
    if mode == "trench":
        far_east = np.repeat((xs > TRENCH_X_MAX)[None, :], NY, axis=0)
        return west | far_east
    raise ValueError(f"unknown cut mode {mode!r}")


def build_volume(
    stack: dict,
    asm: dict,
    b_field: np.ndarray,
    stem: str,
    cut: str = "notch",
    elevation_deg: float = 24.0,
    azimuth_deg: float = 152.0,
    html: bool = False,
) -> dict:
    """Render the capstone volume. Returns the written paths."""
    world, wells = stack["world"], stack["wells"]
    priors, fields = stack["priors"], stack["fields"]
    dem = np.asarray(world["dem"], float)

    pl = r3.standard_plotter(window_size=(2600, 1750), scale_bar=False)

    keep = keep_mask(cut)
    r3.surface(
        pl,
        np.full_like(dem, FLOOR_M),
        color=COLOR_FLOOR,
        opacity=1.0,
        ambient=0.5,
        diffuse=0.5,
        specular=0.0,
    )
    block_walls(pl, world, cut)
    r3.cutaway(
        pl, dem, keep, color=None, opacity=LAND_3D_OPACITY, ambient=0.6, diffuse=0.5
    )

    # Permanent water, smoothed: the raw wet mask is a cell-stepped tangle of
    # one-cell stringers along every channel, which out-shouted the water table
    # itself. A short Gaussian on the mask keeps the lakes and the playa and
    # drops the stringers, which the stream tubes carry instead.
    wet_raw = np.asarray(world["wet"], bool)
    wet = (gaussian_filter(wet_raw.astype(float), 1.2) > 0.55) & keep
    r3.cutaway(
        pl,
        dem + 3.0,
        wet,
        color=COLOR_WATER,
        opacity=1.0,
        ambient=0.5,
        diffuse=0.5,
        specular=0.0,
    )
    r3.stream_tubes(
        pl,
        world | {"streams": wet_raw & np.asarray(world["streams"], bool) & keep},
        radius=55.0,
        color=COLOR_WATER,
    )

    # The +/-b band lives on the cut face and in the fence PNG only: as full-grid
    # sheets it was invisible at any opacity that did not also hide the water
    # table, and it cost two more meshes over the whole domain.

    r3.cutaway(
        pl,
        asm["wte"],
        keep,
        scalars=np.asarray(asm["dtw"], float),
        scalar_name="Depth to water (m), dark = shallow",
        cmap=CMAPS["dtw"],
        clim=DTW_CLIM,
        opacity=1.0,
        show_scalar_bar=True,
        ambient=0.35,
        diffuse=0.7,
        specular=0.0,
    )

    # Sticks: land surface down to the observed water level (well_sticks measures
    # the stick with ``drilled_depth``, so hand it the observed depth instead).
    wt = water_table_wells(wells)
    on_ground = keep[wt["row"].to_numpy(int), wt["col"].to_numpy(int)]
    off_face = np.abs(wt["x"].to_numpy(float) - SECTION_B[0]) > FENCE_BUFFER_M
    pool = wt.loc[on_ground & off_face]
    sel = pool.sample(min(85, len(pool)), random_state=5).sort_index().copy()
    sel["drilled_depth"] = sel["dtw_obs"].to_numpy(float).clip(min=8.0)
    r3.well_sticks(
        pl,
        sel,
        sel["is_source"].to_numpy(bool),
        radius=80.0,
        level_radius=145.0,
        level_thickness=30.0,
    )

    prof = _fence_layers(world, priors, fields, asm, b_field)
    face_wells = wells_near_line(water_table_wells(wells), SECTION_B, FENCE_BUFFER_M)
    tex = fence_texture(prof, face_wells, FLOOR_M)
    textured_fence(pl, SECTION_B, prof["dem"], FLOOR_M, tex, x_shift=30.0)

    r3.add_vertical_scale_bar(
        pl, at=(EXTENT[0] + 2200.0, EXTENT[2] - 300.0, FLOOR_M + 20.0)
    )
    # Name the cut face, so the volume and the paper sections are the same line.
    bx = SECTION_B[0]
    ends = np.array(
        [
            [bx, EXTENT[2] + 250.0, r3.zs(world["dem"][2, int(bx / CELL)] + 120.0)],
            [bx, EXTENT[3] - 250.0, r3.zs(world["dem"][-3, int(bx / CELL)] + 120.0)],
        ]
    )
    pl.add_point_labels(
        ends,
        ["B", "B'"],
        font_size=26,
        text_color="black",
        shape=None,
        show_points=False,
        always_visible=True,
    )

    r3.set_camera(pl, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, zoom=1.55)
    out = {"png": r3.shoot(pl, stem, camera=False, close=not html)}
    if html:
        out["html"] = r3.export_html(pl, stem)
        pl.close()
    return out


# --------------------------------------------------------------------------
# The slide legend
# --------------------------------------------------------------------------


def build_legend(stem: str = "fig08_volume_3d_legend"):
    """A compact standalone legend: series colours, ledger glyphs, DTW ramp."""
    fig = plt.figure(figsize=(3.05, 2.30))
    ax = fig.add_axes([0.02, 0.28, 0.96, 0.70])
    ax.set_axis_off()

    handles = [
        Line2D([], [], color="black", lw=2.2, label="Land surface"),
        Line2D([], [], color=COLORS["truth"], lw=1.0, label="True water table"),
        Line2D([], [], color=COLORS["r"], lw=1.0, label="Regional base R"),
        Line2D([], [], color=COLORS["fac"], lw=1.0, label="FAC-REM water surface"),
        Line2D([], [], color=COLORS["deep"], lw=1.0, label="Deep prior"),
        Line2D(
            [], [], color=COLOR_BASE, lw=1.2, ls=(0, (4, 2)), label="Base prediction"
        ),
        Line2D([], [], color=COLORS["pred"], lw=1.6, label="Assimilated prediction"),
        Patch(
            facecolor=COLOR_ENVELOPE,
            edgecolor="none",
            label="±b (model Laplace scale, cut face)",
        ),
        Patch(facecolor=COLOR_WATER, edgecolor="none", label="Permanent water"),
    ]
    handles += well_ledger_handles(
        visible_label="Source well (assimilated)",
        hidden_label="Validation well (held out)",
    )
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=True,
        handlelength=2.0,
        fontsize=6.5,
    )

    cax = fig.add_axes([0.12, 0.155, 0.76, 0.055])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(*DTW_CLIM), cmap=CMAPS["dtw"]),
        cax=cax,
        orientation="horizontal",
    )
    cb.set_label("Depth to water (m), dark = shallow", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    path = TOY_DIR / f"{stem}.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path


def build_fence_png(
    world, priors, fields, asm, b_field, wells, stem="fig08_volume_3d_fence"
):
    """The B-B' cut face on its own, large enough to read on a slide.

    On the block the face is a narrow ribbon (15 km long, ~850 m of section), so
    the same layers are drawn here on a real elevation axis beside the volume.
    """
    prof = _fence_layers(world, priors, fields, asm, b_field)
    s = prof["d"] / 1000.0

    fig = plt.figure(figsize=(7.6, 2.9))
    ax = fig.add_axes([0.085, 0.175, 0.895, 0.80])
    ax.fill_between(
        s,
        prof["asm"] - prof["b"],
        prof["asm"] + prof["b"],
        color=COLOR_ENVELOPE,
        lw=0,
        zorder=1,
        label="±b (model Laplace scale)",
    )
    ax.fill_between(s, FLOOR_M, prof["dem"], color=COLOR_WALL, lw=0, zorder=0)
    ax.plot(s, prof["deep"], color=COLORS["deep"], lw=0.9, label="Deep prior", zorder=3)
    ax.plot(
        s,
        prof["fac"],
        color=COLORS["fac"],
        lw=0.9,
        label="FAC-REM water surface",
        zorder=3,
    )
    ax.plot(s, prof["r"], color=COLORS["r"], lw=0.9, label="Regional base R", zorder=3)
    ax.plot(
        s,
        prof["base"],
        color=COLOR_BASE,
        lw=1.0,
        ls=(0, (5, 2)),
        label="Base prediction",
        zorder=3,
    )
    ax.plot(
        s,
        prof["asm"],
        color=COLORS["pred"],
        lw=1.4,
        label="Assimilated prediction",
        zorder=4,
    )
    ax.plot(
        s,
        prof["truth"],
        color=COLORS["truth"],
        lw=0.8,
        label="True water table",
        zorder=5,
    )
    ax.plot(s, prof["dem"], color="black", lw=1.1, label="Land surface", zorder=6)

    near = wells_near_line(water_table_wells(wells), SECTION_B, FENCE_BUFFER_M)
    draw_wells_section(ax, near, near["is_source"].to_numpy(bool))

    ax.set_xlim(s[0], s[-1])
    ax.set_ylim(FLOOR_M, float(np.max(prof["dem"])) + 40.0)
    ax.set_xlabel("Distance along B–B′ (km)")
    ax.set_ylabel("Elevation (m), 4× vertical")
    add_ticks(ax, x_step=3.0, y_step=200.0)
    handles, labels = ax.get_legend_handles_labels()
    handles += well_ledger_handles("Source well", "Validation well")
    ax.legend(
        handles=handles,
        labels=labels + ["Source well", "Validation well"],
        loc="upper left",
        fontsize=5,
        ncol=3,
    )
    path = TOY_DIR / f"{stem}.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# The 190 mm paper companion
# --------------------------------------------------------------------------


def _leak_fold(wells) -> int:
    """The fold with the most water-table wells on the B-B' section."""
    near = wells_near_line(water_table_wells(wells), SECTION_B, 900.0)
    counts = near["fold"].value_counts()
    return int(counts.index[0])


def panel_leak(ax, ax_inset, world, wells, priors):
    fold = _leak_fold(wells)
    prof = profile_along(
        SECTION_B,
        {
            "dem": world["dem"],
            "truth": world["wte_true"],
            "leaky": priors["r_leaky"],
            "cf": priors["r_crossfit"][fold],
        },
    )
    s = prof["d"] / 1000.0
    ax.plot(s, prof["dem"], color=COLORS["land"], lw=0.7, label="Land surface")
    ax.plot(s, prof["truth"], color=COLORS["truth"], lw=0.8, label="True water table")
    ax.plot(
        s,
        prof["leaky"],
        color=COLORS["dd"],
        lw=0.9,
        ls=(0, (4, 1.6)),
        label="R, all wells (leaky)",
    )
    ax.plot(
        s, prof["cf"], color=COLORS["r"], lw=0.9, label=f"R, cross-fit (fold {fold})"
    )

    near = wells_near_line(water_table_wells(wells), SECTION_B, 900.0)
    draw_wells_section(ax, near, near["fold"].to_numpy(int) != fold)
    ax.set_xlabel("Distance along B–B′ (km)")
    ax.set_ylabel("Elevation (m), 4× vertical")
    ax.set_xlim(0, s[-1])
    ax.set_ylim(1480, 2300)
    add_ticks(ax, x_step=3.0, y_step=200.0)
    handles, labels = ax.get_legend_handles_labels()
    handles += well_ledger_handles("well seen by R", "well held out of R")
    ax.legend(
        handles=handles,
        labels=labels + ["well seen by R", "well held out of R"],
        loc="lower left",
        fontsize=5,
        ncol=2,
    )

    rl, rc = residual_hist_data(wells, priors["r_leaky"], priors["r_well_values"])
    bins = np.linspace(-60, 60, 49)
    ax_inset.hist(
        rl, bins=bins, color=COLORS["dd"], label=f"leaky, sd {rl.std():.2f} m"
    )
    ax_inset.hist(
        rc,
        bins=bins,
        histtype="step",
        color=COLORS["r"],
        lw=0.8,
        label=f"cross-fit, sd {rc.std():.2f} m",
    )
    ax_inset.set_yscale("log")
    ax_inset.set_xlabel("WTE − R at wells (m)", fontsize=5)
    ax_inset.set_ylabel(f"Wells (count), n = {rl.size}", fontsize=5)
    ax_inset.tick_params(labelsize=5)
    ax_inset.legend(loc="upper left", fontsize=4.5)
    return fold


def panel_gates(axes, fields):
    labels = {"fac": "FAC-REM", "deep": "deep", "head": "free head"}
    for ax, key in zip(axes, ("fac", "deep", "head")):
        j = EXPERT_NAMES.index(key)
        im = ax.imshow(
            np.asarray(fields["gate_w"][j], float),
            origin="lower",
            extent=EXTENT,
            cmap=CMAPS["gate"],
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        map_axes(ax)
        draw_section_lines(ax)
        ax.set_title(f"Gate weight: {labels[key]}", fontsize=6)
    return im


def panel_difference(ax, world, wells, fields, asm):
    _ = world
    im = ax.imshow(
        np.asarray(asm["dtw"], float) - np.asarray(fields["dtw"], float),
        origin="lower",
        extent=EXTENT,
        cmap=CMAPS["resid"],
        norm=resid_norm(30.0),
        interpolation="nearest",
    )
    map_axes(ax)
    draw_section_lines(ax)
    wt = water_table_wells(wells)
    draw_wells_map(ax, wt, wt["is_source"].to_numpy(bool), ms=1.8)
    ax.set_title("Assimilated − base DTW", fontsize=6)
    return im


def panel_density(ax, curve, base_mad, n_wells):
    ax.errorbar(
        curve["n_sources"],
        curve["mad_m"],
        yerr=curve["mad_sd_m"],
        marker="o",
        ms=2.4,
        lw=0.9,
        capsize=1.5,
        color=COLORS["pred"],
        label="assimilated",
    )
    ax.axhline(
        base_mad,
        color=COLOR_BASE,
        lw=0.9,
        ls=(0, (4, 2)),
        label="base model (0 sources)",
    )
    ax.set_xlabel("Source wells revealed (count)")
    ax.set_ylabel("Held-out MAD (m)")
    ax.set_title(f"Density curve (n = {n_wells} held-out wells)", fontsize=6)
    ax.legend(loc="upper right", fontsize=4.6, borderpad=0.35)


def build_paper(stack, asm, stem="fig08_volume_paper"):
    world, wells = stack["world"], stack["wells"]
    priors, fields = stack["priors"], stack["fields"]

    fig = plt.figure(figsize=(WIDTH_2COL, 6.45))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.30, 1.0, 1.02],
        hspace=0.34,
        wspace=0.46,
        left=0.075,
        right=0.965,
        top=0.975,
        bottom=0.055,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_inset = ax_a.inset_axes([0.255, 0.575, 0.325, 0.40])
    fold = panel_leak(ax_a, ax_inset, world, wells, priors)
    panel_letter(ax_a, "a")

    axes_b = [fig.add_subplot(gs[1, j]) for j in range(3)]
    im_g = panel_gates(axes_b, fields)
    panel_letter(axes_b[0], "b")
    cb = fig.colorbar(im_g, ax=axes_b, fraction=0.020, pad=0.012)
    cb.set_label("Gate weight (dimensionless)", fontsize=6)
    cb.ax.tick_params(labelsize=5)

    ax_c = fig.add_subplot(gs[2, 0:2])
    im_c = panel_difference(ax_c, world, wells, fields, asm)
    panel_letter(ax_c, "c")
    cbc = fig.colorbar(im_c, ax=ax_c, fraction=0.028, pad=0.022)
    cbc.set_label("Change in DTW (m)", fontsize=6)
    cbc.ax.tick_params(labelsize=5)

    ax_d = fig.add_subplot(gs[2, 2])
    curve = density_curve(fields, wells, world, tau=asm["tau"], length=asm["length"])
    panel_density(
        ax_d,
        curve,
        float(curve["mad_m"].iloc[0]),
        int(curve["n_wells"].iloc[0]),
    )
    panel_letter(ax_d, "d")

    paths = save(fig, stem, TOY_DIR)
    plt.close(fig)
    return paths, fold, curve


# --------------------------------------------------------------------------


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=("3d", "paper"), default=None)
    ap.add_argument("--azimuth", type=float, default=152.0)
    ap.add_argument("--cut", choices=("trench", "notch", "quadrant"), default="notch")
    ap.add_argument("--elevation", type=float, default=24.0)
    ap.add_argument("--stem", default="fig08_volume_3d")
    ap.add_argument("--html", action="store_true")
    args = ap.parse_args()

    use_style()
    stack = load_stack()
    world, wells = stack["world"], stack["wells"]
    fields, oof = stack["fields"], stack["oof"]

    src = wells["is_source"].to_numpy(bool) & ~wells["is_confined_flag"].to_numpy(bool)
    asm = assimilate(fields, wells, world, src)
    print(f"assimilation: tau {asm['tau']:.0f} m, length {asm['length']:.0f} m")

    b_field = np.asarray(fields["sigma"], float)
    b_wells = oof["b"].to_numpy(float)
    print(
        f"Laplace scale b = fields['sigma'] (m): grid p5/p50/p95 "
        f"{np.percentile(b_field, 5):.2f} / {np.percentile(b_field, 50):.2f} / "
        f"{np.percentile(b_field, 95):.2f}; at wells median {np.median(b_wells):.2f}"
    )
    resid = np.abs(oof["pred_dtw"].to_numpy() - oof["dtw_obs"].to_numpy())
    print(f"  coverage |residual| <= b: {float((resid <= b_wells).mean()):.3f}")

    held = water_table_wells(wells)
    held = held.loc[~held["is_source"].to_numpy(bool)]
    hr, hc = held["row"].to_numpy(int), held["col"].to_numpy(int)
    hobs = held["dtw_obs"].to_numpy(float)
    print("  held-out base      ", score(np.asarray(fields["dtw"])[hr, hc], hobs))
    print("  held-out assimilated", score(asm["dtw"][hr, hc], hobs))

    if args.only != "paper":
        out = build_volume(
            stack,
            asm,
            b_field,
            args.stem,
            cut=args.cut,
            elevation_deg=args.elevation,
            azimuth_deg=args.azimuth,
            html=args.html,
        )
        print("wrote", out)
        print("wrote", build_legend())
        print(
            "wrote",
            build_fence_png(world, stack["priors"], fields, asm, b_field, wells),
        )

    if args.only != "3d":
        paths, fold, curve = build_paper(stack, asm)
        print(f"paper: leak panel uses fold {fold}")
        print(curve.to_string(index=False))
        print("wrote", paths)


if __name__ == "__main__":
    _main()
