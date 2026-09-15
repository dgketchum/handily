"""F6 -- the base model chooses (paper + presentation registers).

Run as::

    uv run python -m utils.figures.fig06_base_model

Writes into ``/data/ssd2/handily/figures/toy/``:

``fig06_base_model_paper.{png,pdf}``
    Row 1: the four gate-weight fields on one shared 0-1 ramp. Row 2: predicted
    depth to water, its residual against the known truth, and the fitted Laplace
    scale b. Row 3: section B-B' with the +/- b band, and a 3 km window of the same
    section showing the four expert surfaces the prediction is a blend of. Row 4:
    out-of-fold |residual| against b at water-table wells, and the median
    |residual| by b decile.
``fig06_base_model_3d.png``
    The predicted water table under the translucent land skin, coloured by the
    ternary FAC/deep/head gate mix.
``fig06_base_model_3d_legend.{png,pdf}``
    The ternary key for that render.

The Laplace scale ``b`` plotted here is the model's own fitted scale, taken
straight from ``fields["sigma"]`` on the grid and ``oof["b"]`` at the wells; no
quantity in this figure is refitted.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

from utils.figures import render3d as r3
from utils.figures.fig_common import (
    CMAPS,
    COLORS,
    EXTENT,
    LINESTYLES,
    SECTION_B,
    TOY_DIR,
    VE,
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
from utils.figures.toy_model import EXPERT_NAMES, load_stack
from utils.figures.toy_priors import water_table_wells
from utils.figures.toy_world import HOLDOUT_REGION

#: Colour ramp for the fitted Laplace scale. A tint ladder of the prediction's own
#: hue (#E69F00), so the map, the section band and the scatter all read as the same
#: quantity; it is used for nothing else in the series.
CMAP_B = "Oranges"

#: Ternary mix corners for the 3-D register (renormalised over these three).
TERNARY = ("fac", "deep", "head")
TERNARY_LABELS = {"fac": "FAC-REM", "deep": "Deep", "head": "Free head"}

#: The window of B-B' opened out in panel (i), in metres along the section.
WINDOW_M = (4500.0, 7500.0)
#: Vertical exaggeration of that window. The series' 4x cannot show 60 m of
#: expert spread across 3 km of section; the panel prints its own value.
WINDOW_VE = 8.0

#: Robust symmetric limit for the residual map, as a percentile of |residual|
#: over the whole grid; the value it produces is printed by ``_main``.
RESID_PCTL = 98.0


# --------------------------------------------------------------------------
# The regime check against F5's argmin map
# --------------------------------------------------------------------------


def prior_dtw(stack: dict) -> dict:
    """The mixture's four experts as depth-to-water fields (m), for the argmin map."""
    world, priors, fields = stack["world"], stack["priors"], stack["fields"]
    dem = np.asarray(world["dem"], float)
    return {
        "fac": np.asarray(priors["fac_depth"], float),
        "deep": dem - np.asarray(priors["deep"][HOLDOUT_REGION], float),
        "mirror": dem - np.asarray(priors["mirror"], float),
        "head": dem - np.asarray(fields["head_wte"], float),
    }


def regime_agreement(stack: dict, keys=EXPERT_NAMES) -> dict:
    """Agreement between the gate's argmax class and the argmin-|error| prior class."""
    truth = np.asarray(stack["world"]["dtw_true"], float)
    cand = prior_dtw(stack)
    gate = stack["fields"]["gate_w"]
    idx = [EXPERT_NAMES.index(k) for k in keys]
    err = np.stack([np.abs(cand[k] - truth) for k in keys])
    argmin = err.argmin(axis=0)
    argmax = np.stack([gate[i] for i in idx]).argmax(axis=0)
    table = np.zeros((len(keys), len(keys)))
    for i in range(len(keys)):
        m = argmin == i
        for j, gj in enumerate(idx):
            table[i, j] = gate[gj][m].mean()
    return {
        "keys": list(keys),
        "agreement": float(np.mean(argmin == argmax)),
        "chance": 1.0 / len(keys),
        "argmin_share": {k: float(np.mean(argmin == i)) for i, k in enumerate(keys)},
        "argmax_share": {k: float(np.mean(argmax == i)) for i, k in enumerate(keys)},
        "mean_gate_by_winner": table,
    }


# --------------------------------------------------------------------------
# Paper register
# --------------------------------------------------------------------------


def _place_cbars(fig, pairs, pad: float = 0.006, width: float = 0.009):
    """Shrink every colorbar to the height its equal-aspect map actually occupies.

    A gridspec cell is taller than the map drawn inside it once ``set_aspect`` has
    run, so a colorbar that fills the cell towers over its own panel. The aspect is
    only applied at draw time, hence the explicit draw before the boxes are read.
    """
    fig.canvas.draw()
    for cax, ax in pairs:
        box = ax.get_position(original=False)
        cax.set_position([box.x1 + pad, box.y0, width, box.height])


def _map_panel(ax, arr, cmap, letter, title, **kw):
    im = ax.imshow(
        np.asarray(arr, float),
        origin="lower",
        extent=EXTENT,
        cmap=cmap,
        interpolation="nearest",
        **kw,
    )
    map_axes(ax, xlabel="", ylabel="")
    draw_section_lines(ax, lw=0.5, fontsize=5.5)
    ax.set_title(f"({letter}) {title}", loc="left")
    return im


def paper_figure(stack: dict, b_wells: np.ndarray, b_grid: np.ndarray):
    world, wells, priors, oof, fields = (
        stack["world"],
        stack["wells"],
        stack["priors"],
        stack["oof"],
        stack["fields"],
    )
    dem = np.asarray(world["dem"], float)
    truth_dtw = np.asarray(world["dtw_true"], float)
    resid = np.asarray(fields["dtw"], float) - truth_dtw
    vmax = float(np.ceil(np.percentile(np.abs(resid), RESID_PCTL) / 10.0) * 10.0)

    fig = plt.figure(figsize=(WIDTH_2COL, 158 / 25.4))
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.00, 1.22, 0.78, 1.00],
        hspace=0.40,
        left=0.055,
        right=0.955,
        top=0.975,
        bottom=0.055,
    )
    cbar_pairs = []

    # ---- Row 1: the four gate fields on one ramp --------------------------
    g1 = outer[0].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.045], wspace=0.12)
    titles = {
        "fac": "FAC-REM",
        "deep": "Deep",
        "mirror": "3 m mirror",
        "head": "Free head",
    }
    im_gate = None
    ax_last = None
    for k, (name, letter) in enumerate(zip(EXPERT_NAMES, "abcd")):
        ax = fig.add_subplot(g1[0, k])
        ax_last = ax
        im_gate = _map_panel(
            ax,
            fields["gate_w"][k],
            CMAPS["gate"],
            letter,
            titles[name],
            vmin=0.0,
            vmax=1.0,
        )
        add_ticks(ax, x_step=10000, y_step=10000)
        ax.set_xticklabels([f"{v / 1000:g}" for v in ax.get_xticks()])
        ax.set_yticklabels([f"{v / 1000:g}" for v in ax.get_yticks()])
        if k == 0:
            ax.set_ylabel("Northing (km)")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Easting (km)")
    cax = fig.add_subplot(g1[0, 4])
    cb = fig.colorbar(im_gate, cax=cax)
    cb.set_label("Gate weight (dimensionless)")
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar_pairs.append((cax, ax_last))

    # ---- Row 2: prediction, residual, scale -------------------------------
    g2 = outer[1].subgridspec(
        1, 6, width_ratios=[1, 0.045, 1, 0.045, 1, 0.045], wspace=0.34
    )
    ax = fig.add_subplot(g2[0, 0])
    im = _map_panel(
        ax,
        fields["dtw"],
        CMAPS["dtw"],
        "e",
        "Predicted depth to water",
        vmin=0.0,
        vmax=100.0,
    )
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    cax = fig.add_subplot(g2[0, 1])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label("Depth to water (m), dark = shallow")
    cbar_pairs.append((cax, ax))

    ax = fig.add_subplot(g2[0, 2])
    im = _map_panel(
        ax,
        resid,
        CMAPS["resid"],
        "f",
        "Residual against the true water table",
        norm=resid_norm(vmax),
    )
    wt = water_table_wells(wells)
    draw_wells_map(ax, wt, np.zeros(len(wt), bool), ms=1.5, mew=0.35)
    ax.set_xlabel("Easting (km)")
    cax = fig.add_subplot(g2[0, 3])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label("Predicted − true depth to water (m)")
    cbar_pairs.append((cax, ax))

    ax = fig.add_subplot(g2[0, 4])
    im = _map_panel(
        ax,
        b_grid,
        CMAP_B,
        "g",
        "Laplace scale",
        vmin=0.0,
        vmax=float(np.ceil(np.percentile(b_grid, 99))),
    )
    ax.set_xlabel("Easting (km)")
    cax = fig.add_subplot(g2[0, 5])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("Laplace scale $b$ (m)")
    cbar_pairs.append((cax, ax))

    # ---- Row 3: section B-B' and its 3 km window --------------------------
    g3 = outer[2].subgridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.13)
    prof = profile_along(
        SECTION_B,
        {
            "dem": dem,
            "truth": world["wte_true"],
            "pred": fields["wte"],
            "b": b_grid,
            "fac": priors["fac_ws"],
            "deep": priors["deep"][HOLDOUT_REGION],
            "mirror": priors["mirror"],
            "head": fields["head_wte"],
        },
        n=600,
    )
    d_km = prof["d"] / 1000.0
    near = wells_near_line(wells, SECTION_B, 500.0)
    near = near.loc[~near["is_confined_flag"].to_numpy(bool)]

    ax = fig.add_subplot(g3[0, 0])
    ax.fill_between(
        d_km,
        prof["pred"] - prof["b"],
        prof["pred"] + prof["b"],
        color=COLORS["pred"],
        alpha=0.30,
        lw=0.0,
        label="± $b$, Laplace scale",
    )
    ax.plot(d_km, prof["dem"], color=COLORS["land"], lw=0.4, label="Land surface")
    ax.plot(
        d_km, prof["truth"], color=COLORS["truth"], lw=0.8, label="True water table"
    )
    ax.plot(d_km, prof["pred"], color=COLORS["pred"], lw=1.0, label="Prediction")
    draw_wells_section(
        ax, near, np.zeros(len(near), bool), tick_half_width=0.16, lw=0.5
    )
    ax.set_xlim(0, d_km[-1])
    ax.set_ylim(1500, 2280)
    ax.set_aspect(VE / 1000.0)
    ax.set_xlabel("Distance along B–B′ (km)")
    ax.set_ylabel(f"Elevation (m), {VE:g}× vertical")
    add_ticks(ax, x_step=3.0, y_step=200.0)
    panel_letter(ax, "h", y=0.16)
    ax.add_patch(
        Rectangle(
            (WINDOW_M[0] / 1000.0, 1500),
            (WINDOW_M[1] - WINDOW_M[0]) / 1000.0,
            780,
            fill=False,
            ec="black",
            lw=0.4,
            zorder=9,
        )
    )
    handles, labels = ax.get_legend_handles_labels()
    handles += well_ledger_handles("well, label visible", "well, label hidden")[1:]
    labels += ["Well, label hidden"]
    ax.legend(handles=handles, labels=labels, loc="upper left", ncol=2)

    ax = fig.add_subplot(g3[0, 1])
    m = (prof["d"] >= WINDOW_M[0]) & (prof["d"] <= WINDOW_M[1])
    ax.plot(d_km[m], prof["dem"][m], color=COLORS["land"], lw=0.4)
    ax.plot(
        d_km[m], prof["head"][m], color=COLORS["r"], lw=0.6, label="Free head (over R)"
    )
    ax.plot(d_km[m], prof["fac"][m], color=COLORS["fac"], lw=0.6, label="FAC-REM")
    ax.plot(d_km[m], prof["deep"][m], color=COLORS["deep"], lw=0.6, label="Deep")
    ax.plot(
        d_km[m],
        prof["mirror"][m],
        color=COLORS["mirror"],
        lw=0.6,
        ls=LINESTYLES["mirror"],
        label="3 m mirror",
    )
    ax.plot(d_km[m], prof["pred"][m], color=COLORS["pred"], lw=1.0)
    lo = np.min(
        [prof[k][m].min() for k in ("dem", "pred", "fac", "deep", "mirror", "head")]
    )
    hi = np.max(
        [prof[k][m].max() for k in ("dem", "pred", "fac", "deep", "mirror", "head")]
    )
    pad = 0.06 * (hi - lo)
    ax.set_xlim(WINDOW_M[0] / 1000.0, WINDOW_M[1] / 1000.0)
    ax.set_ylim(lo - pad, hi + pad + 0.80 * (hi - lo))
    ax.set_aspect(WINDOW_VE / 1000.0)
    ax.set_xlabel("Distance along B–B′ (km)")
    ax.set_ylabel(f"Elevation (m), {WINDOW_VE:g}× vertical")
    add_ticks(ax, x_step=1.0, y_step=20.0)
    panel_letter(ax, "i", y=0.13)
    ax.legend(loc="upper left", ncol=2, title="Expert surfaces")

    # ---- Row 4: does b rank the error? ------------------------------------
    g4 = outer[3].subgridspec(1, 2, width_ratios=[1, 1.35], wspace=0.02)
    r = oof["pred_dtw"].to_numpy(float) - oof["dtw_obs"].to_numpy(float)
    a = np.abs(r)
    lim = (0.01, 100.0)

    ax = fig.add_subplot(g4[0, 0])
    for v in (0.1, 1, 10):
        ax.axvline(v, color="0.75", lw=0.3, zorder=0)
        ax.axhline(v, color="0.75", lw=0.3, zorder=0)
    ax.plot(lim, lim, color="0.45", lw=0.6, ls=(0, (4, 2)), zorder=1)
    ax.plot(
        b_wells,
        a,
        marker="o",
        ls="none",
        ms=2.0,
        mfc="none",
        mec="black",
        mew=0.35,
        zorder=2,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Laplace scale $b$ (m)")
    ax.set_ylabel("|out-of-fold residual| (m)")
    ax.minorticks_off()
    panel_letter(ax, "j")
    ax.text(
        0.975,
        0.045,
        f"$n$ = {a.size}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )

    ax = fig.add_subplot(g4[0, 1])
    q = np.quantile(b_wells, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(q, b_wells) - 1, 0, 9)
    bm = np.array([np.median(b_wells[dec == d]) for d in range(10)])
    am = np.array([np.median(a[dec == d]) for d in range(10)])
    nn = np.array([int((dec == d).sum()) for d in range(10)])
    ax.plot(
        np.arange(1, 11),
        am,
        marker="o",
        ms=2.5,
        lw=0.9,
        color="black",
        label="Median |residual|",
    )
    ax.plot(
        np.arange(1, 11),
        bm,
        marker="s",
        ms=2.5,
        lw=0.9,
        color=COLORS["pred"],
        label="Median $b$",
    )
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, max(am.max(), bm.max()) * 1.25)
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlabel("Decile of $b$ (1 = smallest)")
    ax.set_ylabel("Depth (m)")
    panel_letter(ax, "k")
    ax.text(
        0.975,
        0.045,
        f"$n$ = {nn.min()}–{nn.max()} per decile",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )
    ax.legend(loc="lower right")

    _place_cbars(fig, cbar_pairs)
    paths = save(fig, "fig06_base_model_paper", TOY_DIR)
    plt.close(fig)
    return paths, {
        "resid_vmax_m": vmax,
        "decile_b_m": bm,
        "decile_absresid_m": am,
        "decile_n": nn,
    }


# --------------------------------------------------------------------------
# Presentation register
# --------------------------------------------------------------------------


def ternary_rgb(gate: np.ndarray) -> np.ndarray:
    """(NY, NX, 3) RGB of the FAC/deep/head gate mix, renormalised over those three."""
    idx = [EXPERT_NAMES.index(k) for k in TERNARY]
    w = np.stack([np.asarray(gate[i], float) for i in idx], axis=-1)
    w = w / w.sum(axis=-1, keepdims=True)
    cols = np.array([to_rgb(COLORS[k if k != "head" else "pred"]) for k in TERNARY])
    return np.clip(w @ cols, 0.0, 1.0)


def render_3d(stack: dict) -> str:
    world, wells, fields = stack["world"], stack["wells"], stack["fields"]
    rgb = ternary_rgb(fields["gate_w"])

    pl = r3.standard_plotter(window_size=(2400, 1600))
    grid = r3.structured_grid(np.asarray(fields["wte"], float))
    grid.point_data["gate_mix"] = (
        (rgb * 255).astype(np.uint8).transpose(1, 0, 2).reshape(-1, 3)
    )
    pl.add_mesh(
        grid,
        scalars="gate_mix",
        rgb=True,
        smooth_shading=True,
        show_scalar_bar=False,
        ambient=0.35,
        diffuse=0.75,
        specular=0.0,
    )
    r3.land_surface(pl, world)
    r3.stream_tubes(pl, world, color=COLORS["r"], radius=55.0)

    wt = water_table_wells(wells)
    sel = wt.sample(70, random_state=3).sort_index()
    r3.well_sticks(
        pl,
        sel,
        np.zeros(len(sel), bool),
        radius=70.0,
        level_radius=200.0,
        level_thickness=45.0,
    )
    r3.set_camera(pl, elevation_deg=27.0, distance_factor=2.05, zoom=1.55)
    return str(r3.shoot(pl, "fig06_base_model_3d", camera=False))


def ternary_legend():
    """The ternary key for the 3-D render, as its own small figure."""
    fig = plt.figure(figsize=(52 / 25.4, 48 / 25.4))
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.86])
    n = 140
    cols = np.array([to_rgb(COLORS[k if k != "head" else "pred"]) for k in TERNARY])
    xs, ys, cs = [], [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            w = np.array([i, j, k], float) / n
            # barycentric: fac at top, deep lower-left, head lower-right
            xs.append(0.5 * w[0] + 0.0 * w[1] + 1.0 * w[2])
            ys.append(np.sqrt(3) / 2 * w[0])
            cs.append(w @ cols)
    ax.scatter(xs, ys, c=np.clip(cs, 0, 1), s=3.0, marker="s", linewidths=0)
    tri = np.array(
        [[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]]
    )
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=0.6)
    for x, y, ha, va, key in (
        (0.5, np.sqrt(3) / 2 + 0.05, "center", "bottom", "fac"),
        (-0.02, -0.03, "right", "top", "deep"),
        (1.02, -0.03, "left", "top", "head"),
    ):
        ax.text(x, y, TERNARY_LABELS[key], ha=ha, va=va, fontsize=7)
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.14, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Gate weight (dimensionless),\nrenormalised over three experts",
        loc="center",
        fontsize=6.5,
    )
    paths = save(fig, "fig06_base_model_3d_legend", TOY_DIR)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------


def _main() -> None:
    use_style()
    stack = load_stack()
    b_wells = stack["oof"]["b"].to_numpy(float)
    b_grid = np.asarray(stack["fields"]["sigma"], float)

    oof = stack["oof"]
    r = oof["pred_dtw"].to_numpy(float) - oof["dtw_obs"].to_numpy(float)
    a = np.abs(r)
    rho = spearmanr(b_wells, a)
    print(
        f"b at wells: {b_wells.min():.2f}-{b_wells.max():.2f} m, "
        f"median {np.median(b_wells):.2f} m"
    )
    print(f"Spearman rho(b, |residual|) = {rho.statistic:.3f} (p = {rho.pvalue:.2g})")
    print(f"coverage of ±b = {np.mean(a <= b_wells):.3f} (Laplace nominal 0.632)")

    for keys in (EXPERT_NAMES, ("fac", "deep", "mirror")):
        ag = regime_agreement(stack, keys)
        print(
            f"regime agreement over {ag['keys']}: {ag['agreement']:.3f} "
            f"(chance {ag['chance']:.3f})"
        )
        print("  argmin share", {k: round(v, 3) for k, v in ag["argmin_share"].items()})
        print("  argmax share", {k: round(v, 3) for k, v in ag["argmax_share"].items()})
        print("  mean gate weight by winning prior (rows = winner):")
        print("    " + "".join(f"{k:>9s}" for k in ag["keys"]))
        for i, k in enumerate(ag["keys"]):
            print(
                f"    {k:>6s}"
                + "".join(f"{v:9.3f}" for v in ag["mean_gate_by_winner"][i])
            )

    paths, extra = paper_figure(stack, b_wells, b_grid)
    print("wrote", paths["png"], paths["pdf"])
    print(
        f"residual map symmetric limit ±{extra['resid_vmax_m']:.0f} m "
        f"(p{RESID_PCTL:g} of |residual| over the grid)"
    )
    print("b decile, median b (m), median |residual| (m), n:")
    for i in range(10):
        print(
            f"  {i + 1:2d}  {extra['decile_b_m'][i]:6.2f}  "
            f"{extra['decile_absresid_m'][i]:6.2f}  {extra['decile_n'][i]:3d}"
        )
    print("wrote", render_3d(stack))
    print("wrote", ternary_legend()["png"])


if __name__ == "__main__":
    _main()
