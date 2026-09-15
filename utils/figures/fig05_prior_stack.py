"""F5 -- the prior stack: complementary by regime, none best everywhere.

Run as::

    uv run python -m utils.figures.fig05_prior_stack

Writes into ``/data/ssd2/handily/figures/toy``:

* ``fig05_prior_stack_paper.{png,pdf}`` -- five section small multiples on B-B'
  (one prior each, identical limits) over the argmin-of-|error| regime map and
  the win fractions by distance-to-stream band.
* ``fig05_prior_stack_3d.png`` -- the presentation register: the fence of
  hypotheses along B-B', three translucent prior sheets under a cutaway land
  skin, the two remaining priors and the truth as lines on the fence face.

Prior definitions used here (all compared in **DTW space**, m below ground, so
elevation priors and depth priors are commensurable):

===================  ============================  =====================
prior                surface                        DTW
===================  ============================  =====================
regional base R      ``priors["r_inference"]``      ``dem - r``
FAC-REM              ``priors["fac_ws"]``           ``priors["fac_depth"]``
deep                 fold-median of ``deep[f]``     ``dem - deep``
3 m mirror           ``dem - 3``                    ``3``
drilled depth        ``dem - dd_idw``               ``dd_idw``
===================  ============================  =====================

The drilled-depth prior is the *behavioural* one: ``drilled_depth_prior``
returns depths, and the water surface drawn here is ``DEM - dd_idw``, i.e. it
places the water table at the interpolated borehole depth. It is included
because that is what the production stack feeds the gate, not because a driller
stops at the water table.

Ledger: every well is drawn hollow. R, the deep prior and the drilled-depth
prior are cross-fitted or self-excluded, so no well on a panel was seen by the
surface on that panel; FAC-REM and the mirror use no wells at all.
"""

from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pyvista as pv

from utils.figures import fig_common as fc
from utils.figures import render3d as r3
from utils.figures.toy_model import load_stack
from utils.figures.toy_priors import water_table_wells
from utils.figures.toy_world import N_FOLDS

# --------------------------------------------------------------------------
# Series definition: order is fixed for every panel, legend and table here.
# --------------------------------------------------------------------------

PRIORS = ("r", "fac", "deep", "mirror", "dd")

LABELS = {
    "r": "Regional base R",
    "fac": "FAC-REM",
    "deep": "Deep prior",
    "mirror": "3 m mirror",
    "dd": "Drilled-depth prior",
}

#: Distance-to-stream bands, in m; the same bins F4 uses.
BANDS = ((0, 100), (100, 300), (300, 600), (600, 1000), (1000, 2000), (2000, np.inf))
BAND_LABELS = ("0-100", "100-300", "300-600", "600-1k", "1-2k", ">2k")

#: vtkLegendBoxActor draws each legend label in its own entry colour, so the
#: pale-sand land skin needs a darker sibling in the 3-D key to stay legible.
LAND_KEY_3D = "#8a7a52"

SECTION_BUFFER_M = 400.0

YLIM = (1490.0, 2240.0)


def _band_masks(dist):
    return [((dist >= lo) & (dist < hi)) for lo, hi in BANDS]


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------


def assemble(stack: dict) -> dict:
    """Build the five priors as water-surface elevations and as DTW, plus argmin.

    Returns a dict with ``wte`` (elevation fields), ``dtw`` (depth fields),
    ``win`` (index into :data:`PRIORS` of the smallest |error| per cell),
    ``area`` (win fraction of the grid, %), ``band`` (win fraction per band, %),
    ``wells`` (water-table wells with per-prior DTW columns) and ``well_win``.
    """
    world, wells, pri = stack["world"], stack["wells"], stack["priors"]
    dem = np.asarray(world["dem"], float)
    truth = np.asarray(world["dtw_true"], float)

    # The deep prior is fold-fitted; the grid render takes the fold median, the
    # same arrangement toy_model.render_base uses, so no single fold's wells
    # dominate the mapped surface.
    deep_grid = np.median(
        np.stack([np.asarray(pri["deep"][f], float) for f in range(N_FOLDS)]), axis=0
    )

    wte = {
        "r": np.asarray(pri["r_inference"], float),
        "fac": np.asarray(pri["fac_ws"], float),
        "deep": deep_grid,
        "mirror": np.asarray(pri["mirror"], float),
        "dd": dem - np.asarray(pri["dd_idw"], float),
    }
    dtw = {k: dem - v for k, v in wte.items()}

    err = np.stack([np.abs(dtw[k] - truth) for k in PRIORS])
    win = err.argmin(axis=0)
    area = {k: 100.0 * float((win == i).mean()) for i, k in enumerate(PRIORS)}

    dist = np.asarray(world["dist_to_stream"], float)
    masks = _band_masks(dist)
    band = {
        k: np.array([100.0 * float((win[m] == i).mean()) for m in masks])
        for i, k in enumerate(PRIORS)
    }
    band_n = np.array([int(m.sum()) for m in masks])

    # Per-well argmin: each well gets the fold-honest value of every prior that
    # uses wells at all -- its own fold's cross-fit R and its own fold's deep
    # prior; the drilled-depth field already carries a 100 m self-exclusion.
    wt = water_table_wells(wells).copy()
    row = wt["row"].to_numpy(int)
    col = wt["col"].to_numpy(int)
    fold = wt["fold"].to_numpy(int)
    wdem = wt["dem"].to_numpy(float)

    deep_w = np.empty(len(wt))
    for f, field in pri["deep"].items():
        m = fold == f
        if m.any():
            deep_w[m] = np.asarray(field, float)[row[m], col[m]]

    wt["dtw_r"] = wdem - pri["r_well_values"].loc[wt.index].to_numpy(float)
    wt["dtw_fac"] = np.asarray(pri["fac_depth"], float)[row, col]
    wt["dtw_deep"] = wdem - deep_w
    wt["dtw_mirror"] = 3.0
    wt["dtw_dd"] = np.asarray(pri["dd_idw"], float)[row, col]

    obs = wt["dtw_obs"].to_numpy(float)
    ew = np.stack([np.abs(wt[f"dtw_{k}"].to_numpy(float) - obs) for k in PRIORS])
    well_win = ew.argmin(axis=0)

    return {
        "world": world,
        "wte": wte,
        "dtw": dtw,
        "win": win,
        "area": area,
        "band": band,
        "band_n": band_n,
        "wells": wt,
        "well_win": well_win,
        "well_mad": {
            k: float(np.median(np.abs(wt[f"dtw_{k}"].to_numpy(float) - obs)))
            for k in PRIORS
        },
        "grid_mad": {k: float(np.median(np.abs(dtw[k] - truth))) for k in PRIORS},
    }


# --------------------------------------------------------------------------
# Paper register
# --------------------------------------------------------------------------


def _section_axes(fig, x0, w, y_top, gap, n_panel, xrange_km, yrange_m):
    """Place ``n_panel`` stacked section axes whose aspect is exactly ``VE``."""
    fig_w, fig_h = fig.get_size_inches()
    h_in = (fig_w * w) * (yrange_m * fc.VE / 1000.0) / xrange_km
    h = h_in / fig_h
    axes = []
    for i in range(n_panel):
        y = y_top - (i + 1) * h - i * gap
        axes.append(fig.add_axes((x0, y, w, h)))
    return axes


def paper_figure(data: dict):
    """Draw and save the paper register. Returns the saved-paths dict."""
    world = data["world"]
    dem = np.asarray(world["dem"], float)
    wells = data["wells"]

    prof = fc.profile_along(
        fc.SECTION_B,
        {"dem": dem, "truth": np.asarray(world["wte_true"], float), **data["wte"]},
        n=700,
    )
    d_km = prof["d"] / 1000.0
    wsec = fc.wells_near_line(wells, fc.SECTION_B, SECTION_BUFFER_M)
    hidden = np.zeros(len(wsec), bool)

    fig = plt.figure(figsize=(fc.WIDTH_2COL, 5.15))

    axs = _section_axes(
        fig,
        x0=0.065,
        w=0.415,
        y_top=0.962,
        gap=0.062,
        n_panel=len(PRIORS),
        xrange_km=float(d_km[-1]),
        yrange_m=YLIM[1] - YLIM[0],
    )

    for i, (ax, key) in enumerate(zip(axs, PRIORS)):
        ax.plot(d_km, prof["dem"], color=fc.COLORS["land"], lw=0.4, zorder=3)
        ax.plot(d_km, prof["truth"], color=fc.COLORS["truth"], lw=0.8, zorder=4)
        ax.plot(
            d_km,
            prof[key],
            color=fc.COLORS[key],
            lw=1.0,
            ls=fc.LINESTYLES["mirror"] if key == "mirror" else "-",
            zorder=5,
        )
        fc.draw_wells_section(ax, wsec, hidden, lw=0.45, tick_half_width=0.16)
        ax.set_xlim(0.0, float(d_km[-1]))
        ax.set_ylim(*YLIM)
        fc.add_ticks(ax, x_step=3.0, y_step=250.0)
        ax.set_title(LABELS[key], pad=2.0)
        fc.panel_letter(ax, "abcde"[i], x=0.012, y=0.94)
        if i < len(PRIORS) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Distance along B–B′ (km)")
        if i == 2:
            ax.set_ylabel(f"Elevation (m), {fc.VE_LABEL}")

    # ---- the regime map -------------------------------------------------
    fig_w, fig_h = fig.get_size_inches()
    map_x0, map_w = 0.565, 0.400
    map_h = (fig_w * map_w) * (16.0 / 24.0) / fig_h
    map_y0 = 0.958 - map_h
    ax_map = fig.add_axes((map_x0, map_y0, map_w, map_h))
    cmap = ListedColormap([fc.COLORS[k] for k in PRIORS])
    ax_map.imshow(
        data["win"],
        origin="lower",
        extent=fc.EXTENT,
        cmap=cmap,
        norm=BoundaryNorm(np.arange(-0.5, len(PRIORS)), cmap.N),
        interpolation="nearest",
    )
    fc.draw_section_lines(ax_map, color="white", lw=0.9)
    fc.map_axes(ax_map)
    fc.panel_letter(ax_map, "f").set_path_effects(
        [pe.withStroke(linewidth=1.6, foreground="white")]
    )

    # ---- win fraction by distance band ----------------------------------
    ax_bar = fig.add_axes((map_x0, 0.118, map_w, 0.182))
    xs = np.arange(len(BANDS), dtype=float)
    bottom = np.zeros(len(BANDS))
    for k in PRIORS:
        v = data["band"][k]
        ax_bar.bar(
            xs, v, bottom=bottom, width=0.74, color=fc.COLORS[k], lw=0.0, zorder=3
        )
        bottom += v
    for x, n in zip(xs, data["band_n"]):
        ax_bar.text(x, 101.5, f"{n:,}", ha="center", va="bottom", fontsize=5.5)
    ax_bar.set_xlim(-1.0, len(BANDS) - 0.4)
    ax_bar.set_ylim(0.0, 100.0)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(BAND_LABELS, rotation=45, ha="right")
    ax_bar.set_yticks([0, 25, 50, 75, 100])
    ax_bar.set_xlabel("Distance to stream (m)")
    ax_bar.set_ylabel("Cells won (%)")
    fc.panel_letter(ax_bar, "g", x=0.02, y=0.97).set_path_effects(
        [pe.withStroke(linewidth=1.6, foreground="white")]
    )

    # ---- one key, below the map, serving the map fill and the section lines
    prior_handles = [
        Patch(
            facecolor=fc.COLORS[k],
            edgecolor="none",
            label=f"{LABELS[k]}  {data['area'][k]:.0f} %",
        )
        for k in PRIORS
    ]
    leg1 = fig.legend(
        handles=prior_handles,
        loc="upper left",
        bbox_to_anchor=(map_x0, map_y0 - 0.095),
        ncol=2,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.5,
        columnspacing=1.0,
        title="Prior, and share of grid area won",
    )
    leg1._legend_box.align = "left"

    ref_handles = [
        Line2D([], [], color=fc.COLORS["land"], lw=0.4, label="Land surface"),
        Line2D([], [], color=fc.COLORS["truth"], lw=0.8, label="True water table"),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=2.6,
            mfc="white",
            mec="black",
            mew=0.5,
            label="Well, label hidden",
        ),
    ]
    fig.legend(
        handles=ref_handles,
        loc="upper left",
        bbox_to_anchor=(map_x0, map_y0 - 0.190),
        ncol=1,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.5,
    )

    paths = fc.save(fig, "fig05_prior_stack_paper", fc.TOY_DIR)
    return paths


# --------------------------------------------------------------------------
# Presentation register
# --------------------------------------------------------------------------


def _fence_line(pl, line, z_profile, color, radius=55.0, n=None):
    """A tube drawn along ``line`` at the true elevations ``z_profile`` (m)."""
    z = np.asarray(z_profile, float)
    n = z.size if n is None else n
    x0, y0, x1, y1 = (float(v) for v in line)
    t = np.linspace(0.0, 1.0, n)
    pts = np.column_stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0), r3.zs(z)])
    poly = pv.lines_from_points(pts).tube(radius=radius, n_sides=10)
    return pl.add_mesh(
        poly, color=color, show_scalar_bar=False, ambient=0.5, diffuse=0.55
    )


def presentation_figure(data: dict):
    """Render the 3-D fence of hypotheses. Returns the PNG path."""
    world = data["world"]
    dem = np.asarray(world["dem"], float)
    wells = data["wells"]

    prof = fc.profile_along(
        fc.SECTION_B,
        {"dem": dem, "truth": np.asarray(world["wte_true"], float), **data["wte"]},
        n=500,
    )

    floor = float(min(prof[k].min() for k in ("truth", "dem", *PRIORS)) - 30.0)

    # The default scale bar sits in the south-west corner, which this camera
    # puts on top of the block; it goes in the empty half of the cut instead.
    pl = r3.standard_plotter(scale_bar=False)
    r3.add_vertical_scale_bar(pl, at=(10500.0, 2200.0, 1560.0))
    keep = r3.half_mask(fc.SECTION_B, "left")

    # The block is cut on B-B' and the western half kept, so the fence is the
    # front face. Everything in the block is cut with it: leaving the prior
    # sheets running on across the removed half fills the cut with what look
    # like ground surfaces (each prior is within ~100 m of a 900 m relief) and
    # the cutaway stops reading.
    r3.cutaway(
        pl,
        dem,
        keep,
        color=fc.LAND_3D,
        opacity=fc.LAND_3D_OPACITY,
        ambient=0.55,
        diffuse=0.55,
        specular=0.0,
    )

    # Three sheets only: more translucent layers than this and the stack stops
    # being separable.
    for key in ("r", "fac", "deep"):
        r3.cutaway(
            pl,
            data["wte"][key],
            keep,
            color=fc.COLORS[key],
            opacity=0.5,
            ambient=0.5,
            diffuse=0.6,
            specular=0.0,
        )

    # The cut face itself: without an opaque wall at B-B' the three sheets sit
    # within ~100 m of a 900 m relief block and the scene reads as one plane.
    r3.fence_section(
        pl,
        fc.SECTION_B,
        prof["dem"],
        np.full_like(prof["dem"], floor),
        color=fc.LAND_3D,
        opacity=0.6,
        ambient=0.6,
        diffuse=0.45,
        specular=0.0,
    )

    # The two remaining priors and the truth live only on the fence face, drawn
    # a little east of it so they are not z-fought by the wall. The land surface
    # needs no line of its own: the top edge of the wall is the land surface,
    # and a second dark line there is indistinguishable from the mirror prior
    # 3 m under it.
    face = (
        fc.SECTION_B[0] + 60.0,
        fc.SECTION_B[1],
        fc.SECTION_B[2] + 60.0,
        fc.SECTION_B[3],
    )
    _fence_line(pl, face, prof["mirror"], fc.COLORS["mirror"], radius=32.0)
    _fence_line(pl, face, prof["dd"], fc.COLORS["dd"], radius=40.0)
    for key in ("r", "fac", "deep"):
        _fence_line(pl, face, prof[key], fc.COLORS[key], radius=40.0)
    _fence_line(pl, face, prof["truth"], fc.COLORS["truth"], radius=60.0)

    wsec = fc.wells_near_line(wells, fc.SECTION_B, 900.0)
    wsec = wsec.loc[wsec["d_offset"].to_numpy(float) > 0.0]
    r3.well_sticks(
        pl,
        wsec,
        np.zeros(len(wsec), bool),
        radius=55.0,
        level_radius=130.0,
        level_thickness=45.0,
    )

    legend = pl.add_legend(
        [
            ["Land surface", LAND_KEY_3D],
            ["True water table", fc.COLORS["truth"]],
            [LABELS["r"], fc.COLORS["r"]],
            [LABELS["fac"], fc.COLORS["fac"]],
            [LABELS["deep"], fc.COLORS["deep"]],
            [LABELS["mirror"], fc.COLORS["mirror"]],
            [LABELS["dd"], fc.COLORS["dd"]],
        ],
        bcolor="white",
        border=True,
        size=(0.135, 0.175),
        loc="upper right",
        face="rectangle",
    )
    # vtkLegendBoxActor colours the label text to match its swatch; the series
    # key is the swatch, so put the words back to black.
    legend.GetEntryTextProperty().SetColor(0.0, 0.0, 0.0)  # vtk still tints per entry

    # Looking WNW from just east of the fence: the land skin is cut away on the
    # near (eastern) side, so the B-B' face is the front of the block and the
    # sheets stand exposed in front of it.
    r3.set_camera(
        pl, azimuth_deg=112.0, elevation_deg=24.0, distance_factor=1.35, zoom=1.5
    )
    return r3.shoot(pl, "fig05_prior_stack_3d", camera=False)


# --------------------------------------------------------------------------


def _report(data: dict) -> None:
    print("grid-wide |error| MAD by prior (m):")
    for k in PRIORS:
        print(f"  {LABELS[k]:22s} {data['grid_mad'][k]:7.2f}")
    print("\nshare of grid area won (%):")
    for k in PRIORS:
        print(f"  {LABELS[k]:22s} {data['area'][k]:6.2f}")
    print("\nshare of cells won (%) by distance-to-stream band:")
    head = "  band          n     " + " ".join(f"{k:>7s}" for k in PRIORS)
    print(head)
    for i, lab in enumerate(BAND_LABELS):
        vals = " ".join(f"{data['band'][k][i]:7.1f}" for k in PRIORS)
        print(f"  {lab:<10s} {data['band_n'][i]:6d}  {vals}")
    n = len(data["wells"])
    print(f"\nargmin at water-table wells (n = {n}):")
    for i, k in enumerate(PRIORS):
        c = int((data["well_win"] == i).sum())
        print(
            f"  {LABELS[k]:22s} n = {c:4d}  ({100 * c / n:5.1f} %)"
            f"   MAD {data['well_mad'][k]:6.2f} m"
        )


def main() -> None:
    fc.use_style()
    data = assemble(load_stack())
    _report(data)
    paths = paper_figure(data)
    for ext, p in paths.items():
        print(f"wrote {p}")
    print(f"wrote {presentation_figure(data)}")


if __name__ == "__main__":
    main()
