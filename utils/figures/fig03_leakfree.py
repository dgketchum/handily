"""F3 -- Leak-free by construction (plan section 2, F3).

Run as::

    uv run python -m utils.figures.fig03_leakfree

Claim: the surface scored at a well never saw that well, and the surface the map
is drawn from at inference is the surface the model was trained against.

Three constructions of the same regional base R are drawn on section B-B' with
identical limits:

``leaky``
    ``build_R(mode="leaky")`` -- one field from every water-table well. Because
    ``relief_idw`` returns a data point's value exactly at its own cell, the
    field threads every observation: residual std 0.000 m.
``crossfit``
    ``build_R(mode="crossfit")["fields"][FOLD_K]`` -- the field fitted with fold
    ``FOLD_K`` held out. Fold-``FOLD_K`` wells (and the buffered-holdout wells)
    are hidden from it; every other well is still threaded exactly, so the only
    gaps on the section are at the hollow sticks.
``inference``
    ``build_R(mode="inference")`` -- ``relief_idw`` of the archived per-well
    cross-fit *values*. At every well it reproduces that well's cross-fit value
    exactly, which is what makes the rendered map the same surface the model
    trained against.

Outputs (``/data/ssd2/handily/figures/toy/``): ``fig03_leakfree_paper.{png,pdf}``
and the presentation renders ``fig03_leakfree_3d_a.png`` (the all-well sheet,
every stick black) and ``fig03_leakfree_3d_b.png`` (the same block on the same
camera with the fold-K sheet, fold-K sticks white, the all-well sheet ghosted).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.figures import render3d as r3
from utils.figures.fig_common import (
    COLORS,
    LAND_3D,
    EXTENT,
    SECTION_B,
    TOY_DIR,
    VE,
    WIDTH_2COL,
    draw_section_lines,
    draw_wells_map,
    draw_wells_section,
    map_axes,
    panel_letter,
    profile_along,
    save,
    use_style,
    wells_near_line,
)
from utils.figures.toy_model import load_stack
from utils.figures.toy_priors import residual_hist_data, water_table_wells
from utils.figures.toy_world import HOLDOUT_REGION, N_FOLDS

#: The fold shown in panels (b)-(d). Chosen because B-B' crosses it with more
#: wells than any other fold (21 of the 65 wells within 900 m of the line).
FOLD_K = 3

#: Half-width of the section's well corridor (m).
SECTION_BUFFER_M = 900.0

#: Section window: 15 km of B-B', 760 m of elevation. Both are held identical
#: across panels (a)-(c) so the three surfaces are directly comparable.
SEC_X_KM = (0.0, 15.0)
SEC_Y_M = (1500.0, 2260.0)

#: Residual histogram bins (m). 4 m is narrow enough that the leaky spike -- all
#: 359 residuals are exactly 0 m -- occupies a single bar.
HIST_BINS = np.arange(-80.0, 124.0, 4.0)

#: Muted categorical fills for folds 0..5 (matplotlib Pastel1, desaturated).
FOLD_FILLS = [
    "#cfd9e8",
    "#d7e4d2",
    "#eadcd0",
    "#e3d6e6",
    "#d5e5e5",
    "#efe6cf",
]
HOLDOUT_FILL = "#ffffff"

#: Muted grey-blue for the leaky sheet in the 3-D register; it is not a data
#: colour anywhere else in the series, so it cannot be confused with a prior.
LEAKY_SHEET = "#8c9cad"

# --------------------------------------------------------------------------
# Layout: every panel box is placed in millimetres on a 190 mm canvas.
# --------------------------------------------------------------------------

FIG_W_MM = 190.0
FIG_H_MM = 101.0

_LEFT_X, _LEFT_W = 13.0, 113.0
_RIGHT_X, _RIGHT_W = 138.0, 49.0
_SEC_H = _LEFT_W * (SEC_Y_M[1] - SEC_Y_M[0]) * VE / 1000.0 / (SEC_X_KM[1] - SEC_X_KM[0])
_SEC_GAP = 4.0
_TOP = 3.0


def _box(x_mm, top_mm, w_mm, h_mm):
    """(left, bottom, width, height) in figure fractions from a mm box."""
    return (
        x_mm / FIG_W_MM,
        (FIG_H_MM - top_mm - h_mm) / FIG_H_MM,
        w_mm / FIG_W_MM,
        h_mm / FIG_H_MM,
    )


# --------------------------------------------------------------------------
# Data helpers (private to this figure)
# --------------------------------------------------------------------------


def _visibility(wells, fold: int | None):
    """Ledger mask: True where the well's label was visible to the field drawn.

    ``fold=None`` is the leaky field (every water-table well in folds 0..5 was
    used). Otherwise the named fold is hidden. Buffered-holdout wells are never
    in any training pool, so they are hidden on every panel.
    """
    f = wells["fold"].to_numpy(int)
    vis = f < N_FOLDS
    if fold is not None:
        vis &= f != int(fold)
    return vis


def _at_wells(field, wells):
    """Sample a world field at the wells' own cells (wells sit on cell centres)."""
    return np.asarray(field, float)[
        wells["row"].to_numpy(int), wells["col"].to_numpy(int)
    ]


def _section_panel(ax, prof, curve, wells, field, visible, letter, title, behind=None):
    """One B-B' section: land, truth, one R construction, and the well ledger.

    ``curve`` is the R construction sampled along the section line; ``field`` is
    the same construction as a world grid, sampled at the wells' own cells.
    """
    x = prof["d"] / 1000.0
    if behind is not None:
        ax.plot(
            x,
            behind,
            color=COLORS["r"],
            lw=2.2,
            alpha=0.30,
            solid_capstyle="round",
            zorder=2,
        )
    ax.plot(x, prof["dem"], color=COLORS["land"], lw=0.4, zorder=4)
    ax.plot(x, prof["truth"], color=COLORS["truth"], lw=0.9, zorder=5)
    ax.plot(x, curve, color=COLORS["r"], lw=0.9, zorder=6)

    draw_wells_section(ax, wells, visible)

    # The value the drawn field actually takes at each well cell, with a
    # connector to the observed level. On a threaded well the ring lands on the
    # tick and the connector has zero length; on a hidden well the gap is the
    # honest residual.
    xs = wells["d_along"].to_numpy(float) / 1000.0
    obs = wells["wte_obs"].to_numpy(float)
    val = _at_wells(field, wells)
    ax.vlines(xs, obs, val, color=COLORS["r"], lw=0.7, zorder=9)
    ax.plot(
        xs,
        val,
        marker="o",
        ls="none",
        ms=3.6,
        mfc="none",
        mec=COLORS["r"],
        mew=0.7,
        zorder=10,
    )

    ax.set_xlim(*SEC_X_KM)
    ax.set_ylim(*SEC_Y_M)
    ax.set_aspect(VE / 1000.0, adjustable="box", anchor="NW")
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([1600, 1800, 2000, 2200])
    ax.text(
        0.012,
        0.94,
        f"({letter}) {title}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
    )


# --------------------------------------------------------------------------
# Paper register
# --------------------------------------------------------------------------


def paper_figure(stack) -> dict:
    world, wells, priors = stack["world"], stack["wells"], stack["priors"]
    wt = water_table_wells(wells)
    near = wells_near_line(wt, SECTION_B, SECTION_BUFFER_M)

    r_leaky = np.asarray(priors["r_leaky"], float)
    r_cf = np.asarray(priors["r_crossfit"][FOLD_K], float)
    r_inf = np.asarray(priors["r_inference"], float)

    prof = profile_along(
        SECTION_B,
        {
            "dem": world["dem"],
            "truth": world["wte_true"],
            "leaky": r_leaky,
            "cf": r_cf,
            "inf": r_inf,
        },
        n=600,
    )

    fig = plt.figure(figsize=(WIDTH_2COL, FIG_H_MM / 25.4))

    # --- (a)-(c): the three constructions on B-B' -------------------------
    tops = [_TOP + i * (_SEC_H + _SEC_GAP) for i in range(3)]
    ax_a = fig.add_axes(_box(_LEFT_X, tops[0], _LEFT_W, _SEC_H))
    ax_b = fig.add_axes(_box(_LEFT_X, tops[1], _LEFT_W, _SEC_H))
    ax_c = fig.add_axes(_box(_LEFT_X, tops[2], _LEFT_W, _SEC_H))

    _section_panel(
        ax_a,
        prof,
        prof["leaky"],
        near,
        r_leaky,
        _visibility(near, None),
        "a",
        "R from all wells",
    )
    _section_panel(
        ax_b,
        prof,
        prof["cf"],
        near,
        r_cf,
        _visibility(near, FOLD_K),
        "b",
        f"R cross-fitted, fold {FOLD_K} held out",
    )
    _section_panel(
        ax_c,
        prof,
        prof["inf"],
        near,
        r_inf,
        _visibility(near, FOLD_K),
        "c",
        "R at inference",
        behind=prof["cf"],
    )
    for a in (ax_a, ax_b):
        a.set_xticklabels([])
    ax_c.set_xlabel("Distance along B–B′ (km)")
    # One shared vertical axis title for the stack (guide section 6): the three
    # sections have identical limits, so the title is printed once.
    fig.text(
        (_LEFT_X - 8.5) / FIG_W_MM,
        (FIG_H_MM - (_TOP + 1.5 * _SEC_H + _SEC_GAP)) / FIG_H_MM,
        f"Elevation (m), {VE:g}× vertical",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.0,
    )

    # --- shared legend row under the section stack ------------------------
    handles = [
        Line2D([], [], color=COLORS["land"], lw=0.4, label="Land surface"),
        Line2D([], [], color=COLORS["truth"], lw=0.9, label="True water table"),
        Line2D([], [], color=COLORS["r"], lw=0.9, label="R (panel construction)"),
        Line2D(
            [],
            [],
            color=COLORS["r"],
            lw=2.2,
            alpha=0.30,
            label=f"R cross-fitted, fold {FOLD_K} (behind, c)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=3.6,
            mfc="none",
            mec=COLORS["r"],
            mew=0.7,
            label="R at the well cell",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=2.6,
            mfc="black",
            mec="black",
            mew=0.5,
            label="Well, label visible",
        ),
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
        Patch(
            facecolor="white",
            edgecolor="0.30",
            hatch="////",
            lw=0.4,
            label="Buffered holdout (d)",
        ),
    ]
    leg_ax = fig.add_axes(
        _box(_LEFT_X, tops[2] + _SEC_H + 9.0, _LEFT_W, 9.0), frameon=False
    )
    leg_ax.set_axis_off()
    leg_ax.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    # --- (d) fold blocks --------------------------------------------------
    ax_d = fig.add_axes(_box(_RIGHT_X, _TOP, _RIGHT_W, _RIGHT_W * 16.0 / 24.0))
    basin = np.asarray(world["basin_id"], int)
    cmap = ListedColormap(FOLD_FILLS + [HOLDOUT_FILL])
    norm = BoundaryNorm(np.arange(-0.5, HOLDOUT_REGION + 1.0), cmap.N)
    ax_d.imshow(
        basin,
        origin="lower",
        extent=EXTENT,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    cs = ax_d.contourf(
        world["X"],
        world["Y"],
        (basin == HOLDOUT_REGION).astype(float),
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////"],
    )
    cs.set_edgecolor("0.30")
    cs.set_linewidth(0.0)
    for f in range(N_FOLDS):
        m = basin == f
        ax_d.text(
            float(np.asarray(world["X"])[m].mean()),
            float(np.asarray(world["Y"])[m].mean()),
            str(f),
            ha="center",
            va="center",
            fontsize=6.0,
            zorder=5,
        )
    draw_wells_map(ax_d, wt, _visibility(wt, FOLD_K), ms=1.7, mew=0.4)
    draw_section_lines(ax_d)
    map_axes(ax_d)
    # A little air outside the world extent so the A/A' and B/B' end labels
    # drawn by draw_section_lines are not clipped by the spines; the pads keep
    # the 24:16 aspect so the equal-aspect box still fills the panel.
    ax_d.set_xlim(EXTENT[0] - 1800.0, EXTENT[1] + 1800.0)
    ax_d.set_ylim(EXTENT[2] - 1200.0, EXTENT[3] + 1200.0)
    ax_d.set_xlabel("Easting (km)")
    ax_d.set_ylabel("Northing (km)")
    panel_letter(ax_d, "d", x=0.02, y=0.97)

    # --- (e) residual histograms ------------------------------------------
    ax_e = fig.add_axes(
        _box(_RIGHT_X, _TOP + _RIGHT_W * 16.0 / 24.0 + 10.0, _RIGHT_W, 24.0)
    )
    resid_leaky, resid_cf = residual_hist_data(wells, r_leaky, priors["r_well_values"])
    ax_e.hist(
        resid_cf,
        bins=HIST_BINS,
        color=COLORS["r"],
        alpha=0.85,
        label=f"Cross-fitted (σ = {resid_cf.std():.2f} m, n = {resid_cf.size})",
    )
    ax_e.hist(
        resid_leaky,
        bins=HIST_BINS,
        color="black",
        label=f"All wells (σ = {resid_leaky.std():.3f} m, n = {resid_leaky.size})",
    )
    ax_e.set_yscale("log")
    ax_e.set_ylim(0.7, 900.0)
    ax_e.yaxis.grid(True, which="major", color="0.75", lw=0.3, zorder=0)
    ax_e.set_axisbelow(True)
    ax_e.set_xlim(-80, 120)
    ax_e.set_xticks([-80, -40, 0, 40, 80, 120])
    ax_e.set_xlabel("R residual at wells, observed − R (m)")
    ax_e.set_ylabel("Wells (count)")
    panel_letter(ax_e, "e", x=0.02, y=0.97)
    ax_e.legend(loc="upper right", fontsize=5.5)

    # --- (f) inference R minus cross-fit R at wells -----------------------
    ax_f = fig.add_axes(
        _box(
            _RIGHT_X, _TOP + _RIGHT_W * 16.0 / 24.0 + 10.0 + 24.0 + 10.0, _RIGHT_W, 14.0
        )
    )
    cf_vals = priors["r_well_values"].loc[wt.index].to_numpy(float)
    inf_vals = _at_wells(r_inf, wt)
    delta = inf_vals - cf_vals
    ax_f.axhline(0.0, color="0.6", lw=0.4, zorder=1)
    ax_f.plot(
        cf_vals,
        delta,
        marker="o",
        ls="none",
        ms=1.8,
        mfc="none",
        mec=COLORS["r"],
        mew=0.4,
        zorder=3,
    )
    ax_f.set_ylim(-1.0, 1.0)
    ax_f.set_yticks([-1, 0, 1])
    ax_f.set_xlim(1500, 2300)
    ax_f.set_xticks([1500, 1700, 1900, 2100, 2300])
    ax_f.set_xlabel("Archived cross-fit R at well (m)")
    ax_f.set_ylabel("Inference R\n− cross-fit R (m)")
    panel_letter(ax_f, "f", x=0.02, y=0.94)
    ax_f.text(
        0.98,
        0.90,
        f"max |Δ| = {np.abs(delta).max():.1e} m\nn = {delta.size}",
        transform=ax_f.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
    )

    paths = save(fig, "fig03_leakfree_paper", TOY_DIR)
    plt.close(fig)

    wtf = wt.loc[wt["fold"].to_numpy(int) < N_FOLDS]
    fold_mask = wtf["fold"].to_numpy(int) == FOLD_K
    stats = {
        "leaky_std_m": float(resid_leaky.std()),
        "leaky_n": int(resid_leaky.size),
        "crossfit_std_m": float(resid_cf.std()),
        "crossfit_n": int(resid_cf.size),
        "crossfit_fold_std_m": float(resid_cf[fold_mask].std()),
        "crossfit_fold_n": int(fold_mask.sum()),
        "max_abs_delta_m": float(np.abs(delta).max()),
        "delta_n": int(delta.size),
        "section_wells": int(len(near)),
        "section_fold_wells": int((near["fold"].to_numpy(int) == FOLD_K).sum()),
    }
    return {"paths": paths, "stats": stats}


# --------------------------------------------------------------------------
# Presentation register
# --------------------------------------------------------------------------

#: Sub-window rendered in 3-D (m), straddling B-B' (x = 8 km) on the fold-K
#: block. Over the whole 24 km domain at VE = 4 the two sheets are never more
#: than a pixel apart; inside this window they differ by 18.8 m on average and
#: by up to 116 m, which is 75 to 460 m of rendered height.
WINDOW_X = (5000.0, 13500.0)
WINDOW_Y = (9500.0, 16000.0)

#: This figure's camera. Lower than the series default (the block reads almost
#: planimetric at 30 deg) and from the south-west, so the B-B' cut face in the
#: second render is presented to the viewer rather than edge-on.
CAM_AZ = 218.0
CAM_EL = 18.0


def _window_mask(world) -> np.ndarray:
    x = np.asarray(world["X"], float)
    y = np.asarray(world["Y"], float)
    return (
        (x >= WINDOW_X[0])
        & (x <= WINDOW_X[1])
        & (y >= WINDOW_Y[0])
        & (y <= WINDOW_Y[1])
    )


def _sticks(wt, keep_x=None):
    """Water-table wells inside the rendered window (optionally one side of B-B')."""
    inw = (
        (wt["x"] >= WINDOW_X[0])
        & (wt["x"] <= WINDOW_X[1])
        & (wt["y"] >= WINDOW_Y[0])
        & (wt["y"] <= WINDOW_Y[1])
    )
    sub = wt.loc[inw]
    if keep_x is not None:
        sub = sub.loc[sub["x"].to_numpy(float) >= keep_x]
    return sub


def _block(world, keep, sticks, sheets):
    """One block diagram: land skin, the named R sheets, the ledger, a scale bar.

    ``sheets`` is a list of ``(field, colour, opacity)`` drawn in order.
    """
    pl = r3.standard_plotter(window_size=(2400, 1600), scale_bar=False)
    r3.cutaway(
        pl,
        world["dem"],
        keep,
        color=LAND_3D,
        opacity=0.34,
        ambient=0.60,
        diffuse=0.50,
        specular=0.0,
    )
    for field, colour, opacity in sheets:
        r3.cutaway(
            pl,
            field,
            keep,
            color=colour,
            opacity=opacity,
            ambient=0.45,
            diffuse=0.60,
            specular=0.0,
        )
    vis = sticks["fold"].to_numpy(int) != FOLD_K
    r3.well_sticks(
        pl, sticks, vis, radius=45.0, level_radius=105.0, level_thickness=30.0
    )
    # The bar stands outside the block's south-west corner so neither it nor its
    # label crosses a surface; its foot is at the lowest land in the window.
    dem = np.asarray(world["dem"], float)
    floor = float(dem[keep].min())
    r3.add_vertical_scale_bar(
        pl, at=(WINDOW_X[0] - 1200.0, WINDOW_Y[0] - 1200.0, floor + 260.0)
    )
    r3.set_camera(
        pl, azimuth_deg=CAM_AZ, elevation_deg=CAM_EL, distance_factor=1.2, zoom=1.7
    )
    return pl


def presentation_figures(stack) -> list:
    """Two frames on one camera: the all-well sheet, then the cross-fitted one.

    Drawing the two sheets in one frame does not survive VE = 4 over a real
    landscape -- they are 19 m apart on average here, which is 1 % of the block's
    relief, and whichever is on top hides the other. Split across two frames on
    an identical camera the comparison is exact and nothing is occluded: in the
    first every stick is black and every level band lies on the sheet; in the
    second the fold-K sticks are white and stand clear of it, with the all-well
    sheet kept as a faint ghost so the two can still be seen together.
    """
    world, wells, priors = stack["world"], stack["wells"], stack["priors"]
    wt = water_table_wells(wells)
    r_leaky = np.asarray(priors["r_leaky"], float)
    r_cf = np.asarray(priors["r_crossfit"][FOLD_K], float)
    keep = _window_mask(world)
    sticks = _sticks(wt)
    out = []

    # (a) the all-well surface: every label visible, every level band on the sheet.
    pl = _block(
        world,
        keep,
        sticks.assign(fold=np.full(len(sticks), -1)),
        [(r_leaky, LEAKY_SHEET, 1.0)],
    )
    out.append(r3.shoot(pl, "fig03_leakfree_3d_a", camera=False))

    # (b) the cross-fitted surface with the all-well sheet ghosted behind it.
    pl = _block(
        world,
        keep,
        sticks,
        [(r_leaky, LEAKY_SHEET, 0.45), (r_cf, COLORS["r"], 1.0)],
    )
    out.append(r3.shoot(pl, "fig03_leakfree_3d_b", camera=False))
    return out


def _main() -> None:
    use_style()
    stack = load_stack()
    res = paper_figure(stack)
    for k, v in res["paths"].items():
        print(f"paper {k}: {v}")
    s = res["stats"]
    print(
        f"R residual at wells: all-wells σ = {s['leaky_std_m']:.3f} m (n = {s['leaky_n']}); "
        f"cross-fit σ = {s['crossfit_std_m']:.3f} m (n = {s['crossfit_n']}); "
        f"fold {FOLD_K} only σ = {s['crossfit_fold_std_m']:.3f} m (n = {s['crossfit_fold_n']})"
    )
    print(
        f"inference R − cross-fit R at wells: max |Δ| = {s['max_abs_delta_m']:.3e} m "
        f"(n = {s['delta_n']})"
    )
    print(
        f"section corridor: {s['section_wells']} wells within {SECTION_BUFFER_M:.0f} m of "
        f"B–B′, {s['section_fold_wells']} of them in fold {FOLD_K}"
    )
    for p in presentation_figures(stack):
        print(f"3-D: {p}")


if __name__ == "__main__":
    _main()
