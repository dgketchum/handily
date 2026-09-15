"""F2 -- Elevation, not depth: building the regional base R.

Run as::

    uv run python -m utils.figures.fig02_r_elevation

Writes the paper figure (``fig02_r_elevation_paper.{png,pdf}``) and the two
presentation renders (``fig02_r_elevation_3d_a.png``, ``..._3d_b.png``) into
``/data/ssd2/handily/figures/toy``.

Claim: the water-table *elevation* is smooth where the *depth* is rough, so the
regional base is built in elevation space and the model only has to learn a small
residual on top of it.

Two surfaces are compared throughout, and they do not carry the same ledger:

* **R** is the leak-free inference surface -- ``build_R(..., "inference")``, the
  relief-IDW of the archived per-fold cross-fit values. Every well is hidden from
  the value the surface takes at its own location, so section wells are drawn
  hollow on the R panels.
* **Ma-like** is ``toy_priors.ma_like``, a plain (x, y) IDW of observed DTW built
  from every water-table well, which is how the real direct-DTW product is built
  (no cross-fit). It therefore threads every observation, and its section wells
  are drawn filled. Its grid-wide error is flattered by exactly that; the fair,
  cross-fit-both comparison at wells is printed by ``_main`` and belongs in the
  caption, not inside the figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

from utils.figures import fig_common as fc
from utils.figures import render3d as r3
from utils.figures import toy_priors as tp
from utils.figures.toy_model import load_stack, score
from utils.figures.toy_world import N_FOLDS

STEM_PAPER = "fig02_r_elevation_paper"
STEM_3D_A = "fig02_r_elevation_3d_a"
STEM_3D_B = "fig02_r_elevation_3d_b"

SECTION_BUFFER_M = 800.0  #: half-width of the B-B' well swath
DTW_CLIM = (0.0, 100.0)  #: shared depth-to-water colour limits (m)
RESID_VMAX = 60.0  #: shared residual limit (m), symmetric about 0
HIST_BIN_M = 20.0  #: shared bin width for the two row-3 histograms (m)

_LEVEL_TICK_KM = 0.085  #: half-width of a well's water-level tick, in km


# --------------------------------------------------------------------------
# Private helpers (nothing here may live in a shared module: seven other figure
# agents are consuming those concurrently).
# --------------------------------------------------------------------------


def _idw_at_points(px, py, values, qx, qy, k: int = 16, p: float = 2.0):
    """Plain (x, y) k-NN inverse-distance interpolation evaluated at points.

    Mirrors the construction of :func:`toy_priors.ma_like` (same k, same power,
    same metric) but returns values at scattered query points rather than on the
    grid, which is what a cross-fit evaluation at wells needs.
    """
    tree = cKDTree(np.column_stack([np.asarray(px, float), np.asarray(py, float)]))
    d, idx = tree.query(
        np.column_stack([np.asarray(qx, float), np.asarray(qy, float)]), k=k
    )
    vals = np.asarray(values, float)
    w = 1.0 / np.maximum(d, 1e-9) ** p
    return (w * vals[idx]).sum(axis=1) / w.sum(axis=1)


def _ma_crossfit_at_wells(wells: pd.DataFrame) -> pd.Series:
    """Cross-fit Ma-like DTW at every water-table well in folds 0..N_FOLDS-1.

    A fold-k well is predicted from the pool with fold k removed, so this is the
    fair counterpart of ``priors["r_well_values"]``. The Ma-like field drawn in
    the figure is *not* this -- the real product is built from every well -- but
    the honest head-to-head at wells needs both sides held out.
    """
    wt = tp.water_table_wells(wells)
    wt = wt.loc[wt["fold"].to_numpy(int) < N_FOLDS]
    folds = wt["fold"].to_numpy(int)
    out = np.empty(len(wt))
    for fold in range(N_FOLDS):
        pool = tp.training_pool(wells, exclude_fold=fold)
        m = folds == fold
        if not m.any():
            continue
        out[m] = _idw_at_points(
            pool["x"].to_numpy(float),
            pool["y"].to_numpy(float),
            pool["dtw_obs"].to_numpy(float),
            wt["x"].to_numpy(float)[m],
            wt["y"].to_numpy(float)[m],
        )
    return pd.Series(out, index=wt.index, name="ma_crossfit_dtw")


def _draw_wells_depth(ax, wells, visible: bool, lw: float = 0.6):
    """The ledger glyph in *depth* space: stick from 0 m to the drilled depth.

    The depth axis runs downward, so a stick is drawn from 0 (land surface) to
    the borehole bottom and the observed depth to water is the horizontal tick.
    """
    xs = wells["d_along"].to_numpy(float) * 1e-3
    bot = wells["drilled_depth"].to_numpy(float)
    lev = wells["dtw_obs"].to_numpy(float)
    for xi, bi, li in zip(xs, bot, lev):
        if visible:
            ax.plot(
                [xi, xi],
                [0.0, bi],
                color="black",
                lw=lw,
                solid_capstyle="butt",
                zorder=6,
            )
        else:
            ax.plot(
                [xi, xi],
                [0.0, bi],
                color="black",
                lw=lw + 0.7,
                solid_capstyle="butt",
                zorder=5,
            )
            ax.plot(
                [xi, xi],
                [0.0, bi],
                color="white",
                lw=lw,
                solid_capstyle="butt",
                zorder=6,
            )
        ax.plot(
            [xi - _LEVEL_TICK_KM, xi + _LEVEL_TICK_KM],
            [li, li],
            color="black",
            lw=lw + 0.2,
            solid_capstyle="butt",
            zorder=7,
        )
        ax.plot(
            [xi],
            [li],
            marker="o",
            ms=2.2,
            mfc="black" if visible else "white",
            mec="black",
            mew=0.5,
            ls="none",
            zorder=8,
        )


def _spread(a) -> tuple[float, float]:
    """(standard deviation, inter-quartile range) of ``a``, both in metres."""
    a = np.asarray(a, float)
    q25, q75 = np.percentile(a, [25.0, 75.0])
    return float(a.std(ddof=1)), float(q75 - q25)


def _stats_block(ax, lines, x=0.012, y=0.955, fontsize=5.5, **kw):
    """Plain unboxed left-aligned stats text at the interior top-left."""
    return ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color="black",
        linespacing=1.35,
        zorder=12,
        **kw,
    )


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------


def compute(stack: dict) -> dict:
    """Every number and array the figure draws or prints."""
    world, wells, pri = stack["world"], stack["wells"], stack["priors"]

    dem = np.asarray(world["dem"], float)
    dtw_true = np.asarray(world["dtw_true"], float)
    r_wte = np.asarray(pri["r_inference"], float)
    r_dtw = dem - r_wte
    ma_dtw = np.asarray(pri["ma_dtw"], float)

    wt = tp.water_table_wells(wells)
    wt = wt.loc[wt["fold"].to_numpy(int) < N_FOLDS]
    rows, cols = wt["row"].to_numpy(int), wt["col"].to_numpy(int)
    obs_dtw = wt["dtw_obs"].to_numpy(float)
    obs_wte = wt["wte_obs"].to_numpy(float)

    r_at_wells = pri["r_well_values"].loc[wt.index].to_numpy(float)
    resid_wte = obs_wte - r_at_wells
    ma_cf = _ma_crossfit_at_wells(wells).loc[wt.index].to_numpy(float)

    sd_sig, iqr_sig = _spread(obs_wte)
    sd_res, iqr_res = _spread(resid_wte)

    return {
        "dem": dem,
        "dtw_true": dtw_true,
        "r_wte": r_wte,
        "r_dtw": r_dtw,
        "ma_dtw": ma_dtw,
        "err_r": r_dtw - dtw_true,
        "err_ma": ma_dtw - dtw_true,
        "wt": wt,
        "obs_wte": obs_wte,
        "obs_dtw": obs_dtw,
        "resid_wte": resid_wte,
        "sd_signal": sd_sig,
        "iqr_signal": iqr_sig,
        "sd_resid": sd_res,
        "iqr_resid": iqr_res,
        "ratio_sd": sd_sig / sd_res,
        "ratio_iqr": iqr_sig / iqr_res,
        "n_wells": int(len(wt)),
        "grid_r": score(r_dtw.ravel(), dtw_true.ravel()),
        "grid_ma": score(ma_dtw.ravel(), dtw_true.ravel()),
        "well_r": score(dem[rows, cols] - r_at_wells, obs_dtw),
        "well_ma_cf": score(ma_cf, obs_dtw),
        "well_ma_all": score(ma_dtw[rows, cols], obs_dtw),
    }


# --------------------------------------------------------------------------
# Paper register
# --------------------------------------------------------------------------


def _panel_section_elevation(ax, world, num, wsec):
    prof = fc.profile_along(
        fc.SECTION_B, {"dem": world["dem"], "wte": world["wte_true"], "r": num["r_wte"]}
    )
    d = prof["d"] * 1e-3
    ax.plot(d, prof["dem"], color=fc.COLORS["land"], lw=0.4, zorder=4)
    ax.plot(d, prof["wte"], color=fc.COLORS["truth"], lw=0.8, zorder=5)
    ax.plot(d, prof["r"], color=fc.COLORS["r"], lw=0.9, zorder=5)
    fc.draw_wells_section(
        ax, wsec, np.zeros(len(wsec), bool), tick_half_width=_LEVEL_TICK_KM
    )
    ax.set_xlim(0.0, d[-1])
    ax.set_ylim(1490.0, 2270.0)
    ax.set_aspect(fc.VE / 1000.0)
    ax.set_ylabel(f"Elevation (m), {fc.VE_LABEL}")
    fc.add_ticks(ax, x_step=3.0, y_step=200.0)
    # The x axis is shared with the depth panel below, so hide only this axes'
    # labels; clearing the tick label list would empty both.
    ax.tick_params(labelbottom=False)
    handles = [
        Line2D([], [], color=fc.COLORS["truth"], lw=0.8, label="true water table"),
        Line2D(
            [], [], color=fc.COLORS["r"], lw=0.9, label="R (regional base, elevation)"
        ),
        Line2D([], [], color=fc.COLORS["land"], lw=0.4, label="land surface"),
    ] + fc.well_ledger_handles(
        visible_label="well, label seen by the fit",
        hidden_label="well, hidden from the fit (cross-fit)",
    )[1:]
    ax.legend(
        handles=handles, loc="upper left", ncols=2, handlelength=1.6, fontsize=5.5
    )
    ax.set_title("(a) Elevation space", fontsize=7.0, loc="left", pad=2.0)


def _panel_section_depth(ax, world, num, wsec):
    prof = fc.profile_along(
        fc.SECTION_B, {"dtw": world["dtw_true"], "ma": num["ma_dtw"]}
    )
    d = prof["d"] * 1e-3
    ax.plot(d, prof["dtw"], color=fc.COLORS["truth"], lw=0.8, zorder=5)
    ax.plot(
        d,
        prof["ma"],
        color=fc.COLORS["ma"],
        lw=0.9,
        ls=fc.LINESTYLES["ma"],
        zorder=5,
    )
    _draw_wells_depth(ax, wsec, visible=True)
    ax.set_xlim(0.0, d[-1])
    ax.set_ylim(130.0, -8.0)
    ax.set_xlabel("Distance along B–B′ (km)")
    ax.set_ylabel("Depth to water (m)")
    fc.add_ticks(ax, x_step=3.0)
    ax.set_yticks([0, 40, 80, 120])
    handles = [
        Line2D([], [], color=fc.COLORS["truth"], lw=0.8, label="true depth to water"),
        Line2D(
            [],
            [],
            color=fc.COLORS["ma"],
            lw=0.9,
            ls=fc.LINESTYLES["ma"],
            label="Ma-like direct-DTW IDW (depth)",
        ),
    ] + fc.well_ledger_handles(
        visible_label="well, label seen by the fit", hidden_label=""
    )[:1]
    ax.legend(handles=handles, loc="lower left", handlelength=1.6, fontsize=5.5)
    ax.set_title(
        "(b) Depth space, same wells and same limits", fontsize=7.0, loc="left", pad=2.0
    )


def _panel_map(ax, field, cmap, label, clim=None, norm=None, xlabel="Easting (km)"):
    im = ax.imshow(
        field,
        origin="lower",
        extent=fc.EXTENT,
        cmap=cmap,
        vmin=None if norm is not None else clim[0],
        vmax=None if norm is not None else clim[1],
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )
    fc.draw_section_lines(ax, lw=0.5, fontsize=5.0)
    fc.map_axes(ax, xlabel=xlabel, x_step=10.0)
    ax.set_title(label, fontsize=6.5, loc="left", pad=2.0, linespacing=1.3)
    return im


def _panel_hist(ax, num):
    sig = num["obs_wte"] - num["obs_wte"].mean()
    res = num["resid_wte"]
    lo = np.floor(min(sig.min(), res.min()) / HIST_BIN_M) * HIST_BIN_M
    hi = np.ceil(max(sig.max(), res.max()) / HIST_BIN_M) * HIST_BIN_M
    bins = np.arange(lo, hi + HIST_BIN_M, HIST_BIN_M)
    ax.hist(
        sig,
        bins=bins,
        histtype="stepfilled",
        facecolor="#d9d9d9",
        edgecolor="black",
        lw=0.5,
        zorder=3,
        label="observed water-table elevation at wells, centred on its mean",
    )
    ax.hist(
        res,
        bins=bins,
        histtype="step",
        edgecolor=fc.COLORS["r"],
        lw=0.9,
        zorder=4,
        label="residual, observed WTE − R at the same wells (cross-fit)",
    )
    ax.axvline(0.0, color="black", lw=0.4, ls=(0, (2.5, 1.5)), zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Water-table elevation centred on its mean, and R residual (m)")
    ax.set_ylabel("Wells (count)")
    fc.add_ticks(ax, x_step=100.0)
    ax.legend(loc="upper right", handlelength=1.6, fontsize=5.5)
    _stats_block(
        ax,
        [
            f"signal    sd = {num['sd_signal']:6.1f} m   IQR = {num['iqr_signal']:6.1f} m",
            f"residual  sd = {num['sd_resid']:6.1f} m   IQR = {num['iqr_resid']:6.1f} m",
            f"sd ratio = {num['ratio_sd']:.1f}×   IQR ratio = {num['ratio_iqr']:.1f}×   "
            f"$n$ = {num['n_wells']}",
        ],
        x=0.012,
        y=0.955,
        fontsize=6.0,
    )
    ax.set_title(
        "(g) The signal R carries, and what is left for the model",
        fontsize=7.0,
        loc="left",
        pad=2.0,
    )


def paper_figure(stack: dict, num: dict):
    fc.use_style()
    import matplotlib.pyplot as plt

    world, wells = stack["world"], stack["wells"]
    wsec = fc.wells_near_line(
        tp.water_table_wells(wells), fc.SECTION_B, SECTION_BUFFER_M
    )

    fig = plt.figure(figsize=(fc.WIDTH_2COL, 162 * fc.MM))
    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[1.28, 0.62, 1.08, 0.72],
        width_ratios=[1.0, 1.0, 0.26, 1.0, 1.0, 0.26],
        hspace=0.42,
        wspace=0.28,
        left=0.060,
        right=0.975,
        top=0.975,
        bottom=0.058,
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :], sharex=ax_a)
    _panel_section_elevation(ax_a, world, num, wsec)
    _panel_section_depth(ax_b, world, num, wsec)

    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])
    cax_dtw = fig.add_subplot(gs[2, 2])
    ax_e = fig.add_subplot(gs[2, 3])
    ax_f = fig.add_subplot(gs[2, 4])
    cax_res = fig.add_subplot(gs[2, 5])

    n_cells = num["dtw_true"].size
    im_dtw = _panel_map(
        ax_c,
        num["r_dtw"],
        fc.CMAPS["dtw"],
        "(c) R as depth, DEM − R\ncross-fit",
        clim=DTW_CLIM,
    )
    _panel_map(
        ax_d,
        num["ma_dtw"],
        fc.CMAPS["dtw"],
        "(d) Ma-like DTW\nall wells",
        clim=DTW_CLIM,
        xlabel="",
    )
    norm = fc.resid_norm(RESID_VMAX)
    im_res = _panel_map(
        ax_e,
        num["err_r"],
        fc.CMAPS["resid"],
        f"(e) R error\nMAD {num['grid_r']['mad_m']:.2f} m, $n$ = {n_cells:,} cells",
        norm=norm,
    )
    _panel_map(
        ax_f,
        num["err_ma"],
        fc.CMAPS["resid"],
        f"(f) Ma-like error\nMAD {num['grid_ma']['mad_m']:.2f} m, $n$ = {n_cells:,} cells",
        norm=norm,
        xlabel="",
    )
    for ax in (ax_d, ax_e, ax_f):
        ax.set_ylabel("")
        ax.set_yticklabels([])

    cb1 = fig.colorbar(im_dtw, cax=cax_dtw)
    cb1.set_label("Depth to water (m)\ndark = shallow")
    cb1.outline.set_linewidth(0.5)
    cb2 = fig.colorbar(im_res, cax=cax_res, extend="both")
    cb2.set_label("Predicted − true\ndepth to water (m)")
    cb2.outline.set_linewidth(0.5)

    ax_g = fig.add_subplot(gs[3, :])
    _panel_hist(ax_g, num)

    # Equal-aspect map panels shrink inside their gridspec cells, so the two
    # shared colorbars have to be re-hung on the drawn geometry of the panel they
    # serve; otherwise they run the full row height and collide with the maps.
    fig.canvas.draw()
    for cax, ax in ((cax_dtw, ax_d), (cax_res, ax_f)):
        p, q = ax.get_position(), cax.get_position()
        cax.set_position([q.x0, p.y0, min(q.width, 0.008), p.height])

    return fig


# --------------------------------------------------------------------------
# Presentation register
# --------------------------------------------------------------------------

_CAM_ELEV = 25.0
_CAM_ZOOM = 1.42
_N_STICKS = 80
_STICK_KW = dict(radius=55.0, level_radius=135.0, level_thickness=30.0)
_SCALE_BAR_AT = (1200.0, -1800.0, 1430.0)  # off the block, so no text on a surface


def _sticks_subset(wells, n=_N_STICKS):
    wt = tp.water_table_wells(wells)
    wt = wt.loc[wt["fold"].to_numpy(int) < N_FOLDS]
    return wt.sample(min(n, len(wt)), random_state=2).sort_index()


def _plotter():
    """The series plotter with the scale bar moved off the block."""
    pl = r3.standard_plotter(scale_bar=False)
    r3.add_vertical_scale_bar(pl, at=_SCALE_BAR_AT)
    return pl


def _shoot_low(pl, stem):
    """Screenshot from a lower oblique than the series default (25° elevation).

    The foundation camera at 30° reads almost planimetric on this block; dropping
    it, and tightening the zoom, puts the water table visibly *below* the land.
    """
    r3.set_camera(pl, elevation_deg=_CAM_ELEV, zoom=_CAM_ZOOM)
    return r3.shoot(pl, stem, camera=False)


def _vadose_fence(pl, world, r_wte, **kw):
    """The A–A′ cut face: the unsaturated slab between the ground and R.

    ``R`` sits above the ground in a few wet valley cells, so the fence bottom is
    clamped to the land surface there; the curtain is the thickness of the
    unsaturated zone, never a negative one.
    """
    prof = fc.profile_along(fc.SECTION_A, {"dem": world["dem"], "r": r_wte})
    return r3.fence_section(
        pl,
        fc.SECTION_A,
        prof["dem"],
        np.minimum(prof["r"], prof["dem"]),
        color=fc.LAND_3D,
        **kw,
    )


def render_3d(stack: dict, num: dict) -> dict:
    world, wells = stack["world"], stack["wells"]
    sel = _sticks_subset(wells)
    hidden = np.zeros(len(sel), bool)

    # (a) Elevation space: the smooth R sheet as a water-table elevation, under
    # the translucent land skin, with the A–A′ cut face showing the gap.
    pl = _plotter()
    r3.land_surface(pl, world)
    _vadose_fence(pl, world, num["r_wte"], opacity=0.9)
    r3.surface(
        pl,
        num["r_wte"],
        scalars=num["r_wte"],
        scalar_name="R, water-table elevation (m)",
        cmap=fc.CMAPS["wte"],
        clim=(float(num["r_wte"].min()), float(num["r_wte"].max())),
        show_scalar_bar=True,
        scalar_bar_title="R, water-table elevation (m)",
    )
    r3.well_sticks(pl, sel, hidden, **_STICK_KW)
    path_a = _shoot_low(pl, STEM_3D_A)

    # (b) Depth space: the same R, as DEM - R draped on the ground so it ripples
    # with the terrain, with the ground cut away south of A–A′ to expose the
    # smooth sheet it was computed from.
    pl = _plotter()
    north = r3.half_mask(fc.SECTION_A, side="left")
    r3.cutaway(
        pl,
        world["dem"],
        north,
        scalars=np.clip(num["r_dtw"], *DTW_CLIM),
        scalar_name="R as depth to water (m)",
        cmap=fc.CMAPS["dtw"],
        clim=DTW_CLIM,
        show_scalar_bar=True,
    )
    # South of A–A′ the ground is left as the pale translucent skin, so the same
    # R shows twice in one frame: rippling as a depth on the ground to the north,
    # smooth as an elevation sheet under the skin to the south.
    r3.cutaway(
        pl,
        world["dem"],
        ~north,
        color=fc.LAND_3D,
        opacity=fc.LAND_3D_OPACITY,
        ambient=0.55,
        diffuse=0.55,
        specular=0.0,
    )
    _vadose_fence(pl, world, num["r_wte"], opacity=1.0)
    r3.surface(pl, num["r_wte"], color=fc.COLORS["r"], opacity=0.85)
    r3.well_sticks(pl, sel, hidden, **_STICK_KW)
    path_b = _shoot_low(pl, STEM_3D_B)
    return {"a": path_a, "b": path_b}


# --------------------------------------------------------------------------


def _main() -> None:
    stack = load_stack()
    num = compute(stack)

    fig = paper_figure(stack, num)
    paths = fc.save(fig, STEM_PAPER, fc.TOY_DIR)
    print("paper:", paths["png"], paths["pdf"])

    p3 = render_3d(stack, num)
    print("3-D  :", p3["a"], p3["b"])

    print()
    print(f"n water-table wells (folds 0..{N_FOLDS - 1}) = {num['n_wells']}")
    print(
        "signal  (observed WTE at wells): "
        f"sd {num['sd_signal']:.2f} m, IQR {num['iqr_signal']:.2f} m"
    )
    print(
        "residual (observed WTE − R, cross-fit): "
        f"sd {num['sd_resid']:.2f} m, IQR {num['iqr_resid']:.2f} m"
    )
    print(f"spread ratio: sd {num['ratio_sd']:.2f}×, IQR {num['ratio_iqr']:.2f}×")
    print()
    for name, key in (
        ("grid-wide R as DTW (cross-fit) ", "grid_r"),
        ("grid-wide Ma-like (all wells)  ", "grid_ma"),
        ("at wells  R (cross-fit)        ", "well_r"),
        ("at wells  Ma-like (cross-fit)  ", "well_ma_cf"),
        ("at wells  Ma-like (all wells)  ", "well_ma_all"),
    ):
        s = num[key]
        print(
            f"{name} MAD {s['mad_m']:7.3f} m  median {s['median_resid_m']:+7.3f} m  "
            f"bias {s['bias_m']:+7.3f} m  RMSE {s['rmse_m']:7.3f} m  n {s['n']:,}"
        )


if __name__ == "__main__":
    _main()
