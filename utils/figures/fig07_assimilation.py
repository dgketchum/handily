"""F7 -- Assimilation pins the surface.

Run as::

    uv run python -m utils.figures.fig07_assimilation

Claim (plan section 2, F7): source observations pull the surface onto the
network, the gain decays with distance from the nearest source, and beyond the
network the base model is what remains.

Outputs (all in ``/data/ssd2/handily/figures/toy``)

* ``fig07_assimilation_paper.{png,pdf}`` -- three rows: residual maps (base,
  assimilated, difference), two sections, and the two skill panels.
* ``fig07_assimilation_3d_a.png`` -- the two water-table sheets under a cut-away
  land skin.
* ``fig07_assimilation_3d_b.png`` -- the assimilated-minus-base difference draped
  on the assimilated water table.

Arms
----
base
    the fold-median base model render, ``stack["fields"]``.
assimilated
    ``toy_model.assimilate`` with every water-table source well visible, with
    ``tau`` and ``length`` left to the module's own parsimony rule (it selects
    tau = 500 m, length = 1000 m).

Validation is the 193 water-table wells that are *not* sources; they are hollow
everywhere in the figure. Confined wells appear nowhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.figures import render3d as r3  # noqa: E402
from utils.figures.fig_common import (  # noqa: E402
    CMAPS,
    COLORS,
    EXTENT,
    MM,
    SECTION_A,
    SECTION_B,
    TOY_DIR,
    WIDTH_2COL,
    add_ticks,
    draw_section_lines,
    draw_wells_section,
    map_axes,
    panel_letter,
    profile_along,
    resid_norm,
    save,
    use_style,
    wells_near_line,
)
from utils.figures.toy_model import (  # noqa: E402
    assimilate,
    density_curve,
    load_stack,
    score,
)
from utils.figures.toy_priors import water_table_wells  # noqa: E402

# --------------------------------------------------------------------------
# Figure constants
# --------------------------------------------------------------------------

#: Symmetric limit shared by the two residual maps (m). The residual field runs
#: past +/- 100 m in the playa; 60 m holds the 95th percentile of |residual| for
#: both arms (37.6 m base, 34.8 m assimilated) with room to spare, and the two
#: maps must share it or the comparison is not a comparison.
RESID_VMAX = 60.0
#: Tighter symmetric limit for the difference map (m); |difference| has a 99th
#: percentile of 16.9 m, so 20 m keeps the far-field white and shows the pull.
DIFF_VMAX = 20.0

#: Distance-to-nearest-source bands (m) for the skill panel.
DIST_BANDS = (
    (0.0, 500.0),
    (500.0, 1500.0),
    (1500.0, 3000.0),
    (3000.0, 6000.0),
    (6000.0, np.inf),
)
DIST_LABELS = ("0-0.5", "0.5-1.5", "1.5-3", "3-6", "6+")

#: Per-section vertical exaggeration. The series constant VE = 4 is set by the
#: 3-D register; a 23 km along-valley profile with 150 m of relief is 4 mm tall
#: at 4x and cannot be read, so each 2-D section here carries its own value and
#: prints it on the vertical axis title. Deviation from the series convention,
#: recorded in the report.
VE_A = 60.0
VE_B = 12.0

#: Section window (m) for A-A': the trunk valley floor, which is where the
#: water table and the well sticks are.
YLIM_A = (1500.0, 1650.0)
#: Section window (m) for B-B': the whole crossing, valley floor to mountain top.
YLIM_B = (1520.0, 2240.0)

#: Muted orange used for the un-assimilated sheet in the 3-D register.
BASE_ORANGE_3D = "#c9922e"


def _mm_axes(fig, left, bottom, width, height):
    """Add an axes positioned in millimetres from the figure's lower-left."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes(
        [
            left * MM / fw,
            bottom * MM / fh,
            width * MM / fw,
            height * MM / fh,
        ]
    )


def _draw_wells_two_class(ax, wells, source_mask, ms=1.5, mew=0.32):
    """Map glyphs: sources filled circles, validation hollow triangles.

    ``draw_wells_map`` gives both classes the same marker; the F7 spec needs the
    two well classes to differ in shape as well as in fill, so this is the
    private variant.
    """
    src = np.asarray(source_mask, bool)
    x = wells["x"].to_numpy(float)
    y = wells["y"].to_numpy(float)
    ax.plot(
        x[~src],
        y[~src],
        marker="^",
        ls="none",
        ms=ms + 0.3,
        mfc="white",
        mec="black",
        mew=mew,
        zorder=7,
    )
    ax.plot(
        x[src],
        y[src],
        marker="o",
        ls="none",
        ms=ms,
        mfc="black",
        mec="black",
        mew=mew,
        zorder=8,
    )


def _well_class_handles(n_src, n_val):
    return [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=2.6,
            mfc="black",
            mec="black",
            mew=0.4,
            label=f"source, visible (n = {n_src})",
        ),
        Line2D(
            [],
            [],
            marker="^",
            ls="none",
            ms=3.0,
            mfc="white",
            mec="black",
            mew=0.4,
            label=f"validation, hidden (n = {n_val})",
        ),
    ]


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------


def compute(stack: dict) -> dict:
    """Everything the two registers plot, computed once."""
    world, wells, fields = stack["world"], stack["wells"], stack["fields"]
    src_mask = wells["is_source"].to_numpy(bool) & ~wells["is_confined_flag"].to_numpy(
        bool
    )
    asm = assimilate(fields, wells, world, src_mask)

    base_dtw = np.asarray(fields["dtw"], float)
    truth = np.asarray(world["dtw_true"], float)

    wt = water_table_wells(wells)
    sources = wt.loc[wt["is_source"].to_numpy(bool)]
    val = wt.loc[~wt["is_source"].to_numpy(bool)]
    vr, vc = val["row"].to_numpy(int), val["col"].to_numpy(int)
    vobs = val["dtw_obs"].to_numpy(float)

    panel = {
        "base": score(base_dtw[vr, vc], vobs),
        "assimilated": score(asm["dtw"][vr, vc], vobs),
    }

    vd = asm["nearest_source_dist"][vr, vc]
    bands = []
    for (lo, hi), lab in zip(DIST_BANDS, DIST_LABELS):
        m = (vd >= lo) & (vd < hi)
        if not m.any():
            raise ValueError(f"distance band {lab} km is empty; rebands needed")
        bands.append(
            {
                "label": lab,
                "n": int(m.sum()),
                "base_mad_m": score(base_dtw[vr, vc][m], vobs[m])["mad_m"],
                "asm_mad_m": score(asm["dtw"][vr, vc][m], vobs[m])["mad_m"],
            }
        )

    dn = asm["nearest_source_dist"]
    diff = asm["dtw"] - base_dtw
    change = []
    for (lo, hi), lab in zip(DIST_BANDS, DIST_LABELS):
        m = (dn >= lo) & (dn < hi)
        change.append(
            {
                "label": lab,
                "n_cells": int(m.sum()),
                "median_abs_change_m": float(np.median(np.abs(diff[m]))),
            }
        )

    curve = density_curve(
        fields, wells, world, tau=asm["tau"], length=asm["length"], n_draws=40
    )

    return {
        "asm": asm,
        "src_mask": src_mask,
        "base_dtw": base_dtw,
        "truth": truth,
        "diff": diff,
        "sources": sources,
        "val": val,
        "panel": panel,
        "bands": bands,
        "change": change,
        "curve": curve,
        "n_draws": 40,
    }


# --------------------------------------------------------------------------
# Paper register
# --------------------------------------------------------------------------


def _map_panel(fig, ax, field, norm, wells, src_mask, letter, title):
    im = ax.imshow(
        field,
        origin="lower",
        extent=EXTENT,
        cmap=CMAPS["resid"],
        norm=norm,
        interpolation="nearest",
    )
    map_axes(ax)
    draw_section_lines(ax)
    _draw_wells_two_class(ax, wells, src_mask)
    ax.set_title(title, fontsize=6.0, pad=2.5)
    panel_letter(ax, letter)
    return im


def _section_panel(ax, world, wells, fields, asm, line, buffer_m, ve, ylim, xstep):
    prof = profile_along(
        line,
        {
            "dem": world["dem"],
            "truth": world["wte_true"],
            "base": fields["wte"],
            "asm": asm["wte"],
        },
        n=600,
    )
    d_km = prof["d"] / 1000.0
    ax.plot(d_km, prof["dem"], color=COLORS["land"], lw=0.5, zorder=3)
    ax.plot(d_km, prof["truth"], color=COLORS["truth"], lw=0.8, zorder=4)
    ax.plot(
        d_km,
        prof["base"],
        color=COLORS["pred"],
        lw=0.8,
        ls=(0, (3.0, 1.6)),
        alpha=0.85,
        zorder=4,
    )
    ax.plot(d_km, prof["asm"], color=COLORS["pred"], lw=1.1, zorder=5)

    near = wells_near_line(wells, line, buffer_m)
    near = near.loc[~near["is_confined_flag"].to_numpy(bool)]
    draw_wells_section(ax, near, near["is_source"].to_numpy(bool), tick_half_width=0.16)

    ax.set_xlim(0.0, d_km[-1])
    ax.set_ylim(*ylim)
    ax.set_aspect(ve / 1000.0, adjustable="box")
    add_ticks(ax, x_step=xstep)
    ax.set_ylabel(f"Elevation (m), {ve:g}× vertical")
    return int(near["is_source"].sum()), int((~near["is_source"]).sum())


def paper(stack: dict, num: dict) -> dict:
    world, wells, fields = stack["world"], stack["wells"], stack["fields"]
    asm = num["asm"]

    height_mm = 143.0
    fig = plt.figure(figsize=(WIDTH_2COL, height_mm * MM))

    # ---- row 1: three maps on shared geometry -----------------------------
    map_w, map_h, map_b = 45.0, 30.0, 109.0
    ax_a = _mm_axes(fig, 11.0, map_b, map_w, map_h)
    ax_b = _mm_axes(fig, 60.0, map_b, map_w, map_h)
    cax1 = _mm_axes(fig, 106.5, map_b + 2.0, 2.4, map_h - 4.0)
    ax_c = _mm_axes(fig, 124.0, map_b, map_w, map_h)
    cax2 = _mm_axes(fig, 170.5, map_b + 2.0, 2.4, map_h - 4.0)

    rn = resid_norm(RESID_VMAX)
    _map_panel(
        fig,
        ax_a,
        num["base_dtw"] - num["truth"],
        rn,
        wells,
        num["src_mask"],
        "a",
        "Base residual (no sources)",
    )
    im_b = _map_panel(
        fig,
        ax_b,
        np.asarray(asm["dtw"]) - num["truth"],
        rn,
        wells,
        num["src_mask"],
        "b",
        "Assimilated residual",
    )
    im_c = _map_panel(
        fig,
        ax_c,
        num["diff"],
        resid_norm(DIFF_VMAX),
        wells,
        num["src_mask"],
        "c",
        "Assimilated − base",
    )
    ax_b.set_ylabel("")
    ax_b.set_yticklabels([])
    ax_c.set_ylabel("")
    ax_c.set_yticklabels([])

    cb1 = fig.colorbar(im_b, cax=cax1, extend="both")
    cb1.set_label("Predicted − true DTW (m)")
    cb1.set_ticks([-RESID_VMAX, -30, 0, 30, RESID_VMAX])
    cb2 = fig.colorbar(im_c, cax=cax2, extend="both")
    cb2.set_label("Assimilated − base DTW (m)")
    cb2.set_ticks([-DIFF_VMAX, -10, 0, 10, DIFF_VMAX])

    ax_a.legend(
        handles=_well_class_handles(len(num["sources"]), len(num["val"])),
        loc="lower left",
        fontsize=4.4,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.0,
    )

    # ---- row 2: sections --------------------------------------------------
    sec_b = 56.0
    h_a = 100.0 * (YLIM_A[1] - YLIM_A[0]) * VE_A / 1000.0 / 23.0
    ylim_b = YLIM_B
    h_b = 62.0 * (ylim_b[1] - ylim_b[0]) * VE_B / 1000.0 / 15.0
    ax_sa = _mm_axes(fig, 13.0, sec_b, 100.0, h_a)
    ax_sb = _mm_axes(fig, 124.0, sec_b, 62.0, h_b)

    n_sa, n_va = _section_panel(
        ax_sa, world, wells, fields, asm, SECTION_A, 600.0, VE_A, YLIM_A, 5.0
    )
    n_sb, n_vb = _section_panel(
        ax_sb, world, wells, fields, asm, SECTION_B, 600.0, VE_B, ylim_b, 5.0
    )
    ax_sa.set_xlabel("Distance along A–A′ (km)")
    ax_sb.set_xlabel("Distance along B–B′ (km)")
    ax_sa.set_title(
        f"A–A′ (along valley): {n_sa} source, {n_va} validation wells within 600 m",
        fontsize=6.0,
        pad=2.5,
    )
    ax_sb.set_title(
        f"B–B′ (across valley): {n_sb} source, {n_vb} validation",
        fontsize=6.0,
        pad=2.5,
    )
    panel_letter(ax_sa, "d")
    panel_letter(ax_sb, "e")

    ax_sb.legend(
        handles=[
            Line2D([], [], color=COLORS["truth"], lw=0.8, label="true water table"),
            Line2D(
                [],
                [],
                color=COLORS["pred"],
                lw=0.8,
                ls=(0, (3.0, 1.6)),
                label="base prediction",
            ),
            Line2D(
                [], [], color=COLORS["pred"], lw=1.1, label="assimilated prediction"
            ),
            Line2D([], [], color=COLORS["land"], lw=0.5, label="land surface"),
            Line2D([], [], color="black", lw=0.6, label="source well (solid stick)"),
            Line2D(
                [],
                [],
                color="black",
                lw=1.3,
                markerfacecolor="white",
                label="validation well (hollow stick)",
                marker="o",
                ms=2.4,
                mec="black",
                mew=0.5,
                ls="none",
            ),
        ],
        loc="upper left",
        ncol=1,
        fontsize=4.4,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.22,
        handlelength=1.6,
    )

    # ---- row 3: skill panels ---------------------------------------------
    ax_f = _mm_axes(fig, 15.0, 11.0, 70.0, 32.0)
    ax_g = _mm_axes(fig, 106.0, 11.0, 70.0, 32.0)

    xs = np.arange(len(num["bands"]), dtype=float)
    base_mad = np.array([b["base_mad_m"] for b in num["bands"]])
    asm_mad = np.array([b["asm_mad_m"] for b in num["bands"]])
    ax_f.plot(
        xs,
        base_mad,
        marker="o",
        ms=3.0,
        mfc="white",
        mec=COLORS["pred"],
        color=COLORS["pred"],
        lw=0.9,
        ls=(0, (3.0, 1.6)),
        label="base (no sources)",
    )
    ax_f.plot(
        xs,
        asm_mad,
        marker="o",
        ms=3.0,
        mfc=COLORS["pred"],
        mec=COLORS["pred"],
        color=COLORS["pred"],
        lw=1.1,
        label="assimilated",
    )
    ax_f.set_xticks(xs)
    ax_f.set_xticklabels(DIST_LABELS)
    ax_f.set_xlim(-0.35, len(xs) - 0.65)
    ax_f.set_ylim(0.0, max(base_mad.max(), asm_mad.max()) * 1.32)
    for x, b in zip(xs, num["bands"]):
        ax_f.annotate(
            f"n = {b['n']}",
            (x, 0.0),
            xytext=(0, 2.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=4.8,
        )
    ax_f.set_xlabel("Distance to nearest source well (km)")
    ax_f.set_ylabel("MAD (m) at validation wells")
    ax_f.set_title(
        f"Validation wells only (n = {num['panel']['base']['n']})",
        fontsize=6.0,
        pad=2.5,
    )
    ax_f.legend(loc="upper right", fontsize=5.0, handletextpad=0.5, borderpad=0.35)
    panel_letter(ax_f, "f")

    curve = num["curve"]
    frac = curve["fraction"].to_numpy(float) * 100.0
    mad = curve["mad_m"].to_numpy(float)
    sd = curve["mad_sd_m"].to_numpy(float)
    base_level = float(mad[0])
    ax_g.axhline(
        base_level,
        color=COLORS["mirror"],
        lw=0.8,
        ls="-",
        zorder=2,
        label=f"base arm, no sources ({base_level:.2f} m)",
    )
    ax_g.fill_between(
        frac,
        mad - sd,
        mad + sd,
        color=COLORS["pred"],
        alpha=0.22,
        lw=0.0,
        zorder=3,
        label="±1 sd across draws",
    )
    ax_g.plot(
        frac,
        mad,
        marker="o",
        ms=3.0,
        color=COLORS["pred"],
        mfc=COLORS["pred"],
        lw=1.1,
        zorder=4,
        label="assimilated, mean of draws",
    )
    ax_g.set_xlim(-3.0, 95.0)
    ax_g.set_ylim(2.8, base_level + 1.0)
    ax_g.set_xlabel("Source wells made visible (% of 193 candidates)")
    ax_g.set_ylabel("Held-out MAD (m)")
    ax_g.set_title(
        f"{int(curve['n_wells'].iloc[0])} held-out wells within "
        f"{asm['length'] / 1000:g} km of a candidate source; "
        f"{num['n_draws']} draws per point",
        fontsize=5.4,
        pad=2.5,
    )
    ax_g.legend(loc="upper right", fontsize=4.8, handletextpad=0.5, borderpad=0.35)
    panel_letter(ax_g, "g")

    paths = save(fig, "fig07_assimilation_paper", TOY_DIR)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------
# Presentation register
# --------------------------------------------------------------------------


def _stick_selection(wells, n=70, seed=3):
    wt = water_table_wells(wells)
    return wt.sample(min(n, len(wt)), random_state=seed).sort_index()


def _plotter():
    """Standard plotter with the scale bar moved clear of the frame edge.

    ``standard_plotter``'s default bar sits at the south-west corner of the
    domain, which the fixed camera clips at this zoom; the label is the only
    text baked into a 3-D render and it has to be readable.
    """
    pl = r3.standard_plotter(window_size=(2400, 1600), scale_bar=False)
    r3.add_vertical_scale_bar(pl, at=(EXTENT[0] - 2200.0, EXTENT[2] - 1400.0, 1560.0))
    return pl


def three_d(stack: dict, num: dict) -> list:
    world, wells, fields = stack["world"], stack["wells"], stack["fields"]
    asm = num["asm"]
    out = []
    sel = _stick_selection(wells)

    # (a) two sheets under a cut-away land skin ---------------------------
    # The land is removed on the near (south) side of A-A' so the eye goes
    # under the ground to the two water tables; the camera is dropped to 15 deg
    # so a metres-scale separation between the sheets projects to screen height
    # instead of collapsing into a plan view.
    pl = _plotter()
    # Ground kept everywhere except a 4 km strip along the near (southern) edge,
    # so the block still reads as terrain while the water sheets are exposed in
    # the foreground rather than seen only through a translucent skin.
    keep_land = np.asarray(world["Y"], float) > 4000.0
    r3.cutaway(
        pl,
        world["dem"],
        keep_land,
        color=r3.LAND_3D,
        opacity=0.40,
        ambient=0.55,
        diffuse=0.55,
        specular=0.0,
    )
    r3.stream_tubes(pl, world, radius=55.0)
    r3.surface(
        pl,
        asm["wte"],
        color=COLORS["pred"],
        opacity=1.0,
        ambient=0.4,
        diffuse=0.7,
        specular=0.0,
    )
    # The base sheet is drawn only where it stands clear of the assimilated one
    # (more than 0.5 m above it). Everywhere else the two sheets are within half
    # a metre of each other and drawing both produces z-fighting speckle, which
    # would read as structure that is not there; the pale patches are therefore
    # exactly the places assimilation pulled the water table down, and the
    # uninterrupted orange of the eastern basin is the two sheets being one.
    stands_clear = (
        np.asarray(fields["wte"], float) - np.asarray(asm["wte"], float) > 0.5
    )
    r3.cutaway(
        pl,
        fields["wte"],
        stands_clear,
        color=BASE_ORANGE_3D,
        opacity=1.0,
        ambient=0.55,
        diffuse=0.5,
        specular=0.0,
    )
    prof = profile_along(SECTION_B, {"base": fields["wte"], "asm": asm["wte"]}, n=300)
    lo = np.minimum(prof["base"], prof["asm"])
    hi = np.maximum(prof["base"], prof["asm"])
    r3.fence_section(pl, SECTION_B, hi, lo, color=COLORS["deep"], opacity=1.0)
    r3.well_sticks(
        pl,
        sel,
        sel["is_source"].to_numpy(bool),
        radius=85.0,
        level_radius=110.0,
        level_thickness=40.0,
    )
    r3.set_camera(pl, elevation_deg=22.0, distance_factor=2.2, zoom=1.5)
    out.append(r3.shoot(pl, "fig07_assimilation_3d_a", camera=False))

    # (b) the difference field draped on the assimilated water table -------
    pl = _plotter()
    r3.cutaway(
        pl,
        world["dem"],
        keep_land,
        color=r3.LAND_3D,
        opacity=0.28,
        ambient=0.55,
        diffuse=0.55,
        specular=0.0,
    )
    r3.surface(
        pl,
        asm["wte"],
        scalars=np.asarray(num["diff"], float),
        scalar_name="Assimilated − base DTW (m)",
        cmap=CMAPS["resid"],
        clim=(-DIFF_VMAX, DIFF_VMAX),
        show_scalar_bar=True,
        scalar_bar_title="Assimilated − base DTW (m)",
        opacity=1.0,
        ambient=0.5,
        diffuse=0.55,
        specular=0.0,
    )
    r3.well_sticks(
        pl,
        sel,
        sel["is_source"].to_numpy(bool),
        radius=85.0,
        level_radius=110.0,
        level_thickness=40.0,
    )
    r3.set_camera(pl, elevation_deg=22.0, distance_factor=2.1, zoom=1.5)
    out.append(r3.shoot(pl, "fig07_assimilation_3d_b", camera=False))
    return out


# --------------------------------------------------------------------------


def _main() -> None:
    use_style()
    stack = load_stack()
    num = compute(stack)

    asm = num["asm"]
    print(f"assimilation: tau = {asm['tau']:.0f} m, length = {asm['length']:.0f} m")
    print(f"sources (water-table, visible): n = {len(num['sources'])}")
    print(f"validation (water-table, hidden): n = {len(num['val'])}")
    print("held-out metric panel (m):")
    for arm in ("base", "assimilated"):
        s = num["panel"][arm]
        print(
            f"  {arm:<12s} MAD {s['mad_m']:.3f}  median resid {s['median_resid_m']:+.3f}"
            f"  bias {s['bias_m']:+.3f}  RMSE {s['rmse_m']:.3f}  n {s['n']}"
        )
    print("MAD (m) at validation wells by distance to nearest source (km):")
    for b in num["bands"]:
        print(
            f"  {b['label']:>8s}  n = {b['n']:>3d}   base {b['base_mad_m']:6.3f}"
            f"   assimilated {b['asm_mad_m']:6.3f}"
        )
    print("median |change| in DTW (m) by distance to nearest source (km), grid cells:")
    for c in num["change"]:
        print(
            f"  {c['label']:>8s}  n = {c['n_cells']:>6d} cells"
            f"   {c['median_abs_change_m']:.3f}"
        )
    print("density curve (held-out MAD, m):")
    print(num["curve"].to_string(index=False))

    paths = paper(stack, num)
    print("paper:", paths["png"], paths["pdf"])
    for p in three_d(stack, num):
        print("3-D:  ", p)


if __name__ == "__main__":
    _main()
