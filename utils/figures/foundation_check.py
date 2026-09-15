"""Validation contact sheets for the figure-series foundation.

Run as::

    uv run python -m utils.figures.foundation_check

Writes four PNGs into ``/data/ssd2/handily/figures/toy/``:

``foundation_check_world.png``
    terrain, streams, HAND, true DTW/WTE, wells, folds, the two section lines.
``foundation_check_priors.png``
    every level-0 prior as a map plus the leak diagnostics.
``foundation_check_model.png``
    gate weights, base prediction, residuals, assimilation, density curve.
``foundation_check_3d.png``
    the 3-D register primitives (land, water table, streams, sticks, fence, cutaway).

These are diagnostics, not publication figures: they are deliberately dense and
they carry a title per panel so the reviewer can tell what they are looking at.
The eight figure scripts should *not* copy their layout -- they exist so the
foundation can be checked by eye before anyone builds on it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from utils.figures import render3d as r3
from utils.figures.fig_common import (
    CMAPS,
    COLORS,
    EXTENT,
    SECTION_A,
    SECTION_B,
    TOY_DIR,
    draw_section_lines,
    draw_wells_map,
    draw_wells_section,
    map_axes,
    profile_along,
    resid_norm,
    use_style,
    well_ledger_handles,
    wells_near_line,
)
from utils.figures.toy_model import (
    assimilate,
    density_curve,
    load_stack,
    score,
)
from utils.figures.toy_priors import residual_hist_data, water_table_wells


def _imshow(
    ax, arr, title, cmap="viridis", norm=None, vmin=None, vmax=None, label=None
):
    im = ax.imshow(
        np.asarray(arr, float),
        origin="lower",
        extent=EXTENT,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    map_axes(ax)
    ax.set_title(title, fontsize=6)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    if label:
        cb.set_label(label, fontsize=6)
    cb.ax.tick_params(labelsize=5)
    return im


def sheet_world(world, wells) -> str:
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 9.0))
    ax = axes.ravel()

    _imshow(ax[0], world["dem"], "DEM", cmap="terrain", label="Elevation (m)")
    draw_section_lines(ax[0])

    _imshow(
        ax[1],
        np.log1p(world["flow_acc"]),
        "log1p flow accumulation",
        label="log1p (cells)",
    )
    ax[1].contour(
        np.asarray(world["streams"], float),
        levels=[0.5],
        colors=[COLORS["r"]],
        linewidths=0.4,
        origin="lower",
        extent=EXTENT,
    )

    _imshow(
        ax[2],
        world["hand"],
        "HAND",
        cmap="magma_r",
        vmin=0,
        vmax=120,
        label="Height above drainage (m)",
    )
    _imshow(ax[3], world["dist_to_stream"], "Distance to stream", label="Distance (m)")
    _imshow(
        ax[4],
        world["wte_true"],
        "True water-table elevation",
        cmap=CMAPS["wte"],
        label="WTE (m)",
    )
    _imshow(
        ax[5],
        world["dtw_true"],
        "True depth to water",
        cmap=CMAPS["dtw"],
        vmin=0,
        vmax=100,
        label="DTW (m)",
    )

    _imshow(
        ax[6],
        world["basin_id"],
        "Cross-validation folds",
        cmap="tab10",
        vmin=0,
        vmax=9,
        label="Fold",
    )
    draw_wells_map(ax[6], wells, ~wells["is_confined_flag"].to_numpy(bool), ms=1.6)

    ax[7].imshow(
        np.asarray(world["dtw_true"], float),
        origin="lower",
        extent=EXTENT,
        cmap=CMAPS["dtw"],
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )
    map_axes(ax[7])
    ax[7].set_title("Wells: filled = source, hollow = held out", fontsize=6)
    draw_wells_map(ax[7], wells, wells["is_source"].to_numpy(bool))
    draw_section_lines(ax[7])
    ax[7].legend(
        handles=well_ledger_handles("source", "held out"), loc="lower left", fontsize=5
    )

    prof = profile_along(
        SECTION_A,
        {"dem": world["dem"], "wte": world["wte_true"], "hand": world["hand"]},
    )
    a = ax[8]
    a.plot(
        prof["d"] / 1000,
        prof["dem"],
        color=COLORS["land"],
        lw=0.6,
        label="Land surface",
    )
    a.plot(
        prof["d"] / 1000,
        prof["wte"],
        color=COLORS["truth"],
        lw=0.9,
        ls=(0, (3, 1.5)),
        label="True water table",
    )
    near = wells_near_line(wells, SECTION_A, 600.0)
    draw_wells_section(a, near, near["is_source"].to_numpy(bool))
    a.set_xlabel("Distance along A-A' (km)")
    a.set_ylabel("Elevation (m)")
    a.set_title("Section A-A' (true aspect, no exaggeration)", fontsize=6)
    a.legend(loc="upper right", fontsize=5)

    fig.tight_layout()
    p = TOY_DIR / "foundation_check_world.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return str(p)


def sheet_priors(world, wells, priors) -> str:
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 9.0))
    ax = axes.ravel()
    dem = np.asarray(world["dem"], float)
    truth = np.asarray(world["dtw_true"], float)

    _imshow(
        ax[0],
        dem - priors["r_leaky"],
        "R (leaky, all wells) as DTW",
        cmap=CMAPS["dtw"],
        vmin=0,
        vmax=100,
        label="DTW (m)",
    )
    _imshow(
        ax[1],
        dem - priors["r_inference"],
        "R (cross-fit values, inference) as DTW",
        cmap=CMAPS["dtw"],
        vmin=0,
        vmax=100,
        label="DTW (m)",
    )
    _imshow(
        ax[2],
        priors["fac_depth"],
        "FAC-REM depth prior",
        cmap=CMAPS["dtw"],
        vmin=0,
        vmax=100,
        label="DTW (m)",
    )
    # Shown as an elevation: the deep prior is a WTE surface fitted to the deepest
    # quartile of wells and legitimately sits above ground in the valley, which is
    # what the gate is there to suppress.
    _imshow(
        ax[3],
        priors["deep"][0],
        "Deep prior (fold 0), WTE",
        cmap=CMAPS["wte"],
        label="WTE (m)",
    )
    _imshow(
        ax[4],
        priors["dd_idw"],
        "Drilled-depth prior (IDW)",
        cmap=CMAPS["dtw"],
        vmin=0,
        vmax=150,
        label="Drilled depth (m)",
    )
    _imshow(
        ax[5],
        priors["ma_dtw"],
        "Ma-like direct-DTW interpolation",
        cmap=CMAPS["dtw"],
        vmin=0,
        vmax=100,
        label="DTW (m)",
    )

    err = np.asarray(priors["fac_depth"], float) - truth
    _imshow(
        ax[6],
        err,
        "FAC-REM error (predicted - true DTW)",
        cmap=CMAPS["resid"],
        norm=resid_norm(60.0),
        label="Residual (m)",
    )

    rl, rc = residual_hist_data(wells, priors["r_leaky"], priors["r_well_values"])
    a = ax[7]
    bins = np.linspace(-60, 60, 61)
    a.hist(rl, bins=bins, color=COLORS["dd"], label=f"Leaky (std {rl.std():.2f} m)")
    a.hist(
        rc,
        bins=bins,
        histtype="step",
        color=COLORS["r"],
        lw=0.8,
        label=f"Cross-fit (std {rc.std():.2f} m)",
    )
    a.set_xlabel("R residual at wells (m)")
    a.set_ylabel("Wells (count)")
    a.set_yscale("log")
    a.set_title("R residual at wells: leaky vs cross-fit", fontsize=6)
    a.legend(loc="upper right", fontsize=5)

    d = np.asarray(world["dist_to_stream"], float)
    bands = [(0, 100), (100, 300), (300, 600), (600, 1000), (1000, 2000), (2000, 1e9)]
    labels = ["0-100", "100-300", "300-600", "600-1k", "1k-2k", "2k+"]
    fac_mad, mir_mad, ma_mad = [], [], []
    for lo, hi in bands:
        m = (d >= lo) & (d < hi)
        fac_mad.append(np.median(np.abs(np.asarray(priors["fac_depth"])[m] - truth[m])))
        mir_mad.append(
            np.median(np.abs((dem[m] - np.asarray(priors["mirror"])[m]) - truth[m]))
        )
        ma_mad.append(np.median(np.abs(np.asarray(priors["ma_dtw"])[m] - truth[m])))
    a = ax[8]
    xx = np.arange(len(bands))
    a.plot(xx, fac_mad, marker="o", ms=2.5, color=COLORS["fac"], label="FAC-REM")
    a.plot(
        xx,
        mir_mad,
        marker="s",
        ms=2.5,
        color=COLORS["mirror"],
        ls=(0, (2.5, 1.5)),
        label="3 m mirror",
    )
    a.plot(xx, ma_mad, marker="^", ms=2.5, color=COLORS["ma"], label="Ma-like")
    a.set_xticks(xx)
    a.set_xticklabels(labels)
    a.set_xlabel("Distance to stream (m)")
    a.set_ylabel("MAD of DTW (m)")
    a.set_title("Prior skill vs distance to stream", fontsize=6)
    a.legend(loc="upper left", fontsize=5)

    fig.tight_layout()
    p = TOY_DIR / "foundation_check_priors.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return str(p)


def sheet_model(world, wells, priors, oof, fields) -> str:
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 9.0))
    ax = axes.ravel()
    _ = priors
    truth = np.asarray(world["dtw_true"], float)

    names = ("fac", "deep", "mirror", "head")
    for n, (nm, a) in enumerate(zip(names, ax[:4])):
        _imshow(
            a,
            fields["gate_w"][n],
            f"Gate weight: {nm}",
            cmap=CMAPS["gate"],
            vmin=0,
            vmax=1,
            label="Weight (dimensionless)",
        )

    _imshow(
        ax[4],
        fields["dtw"],
        "Base prediction",
        cmap=CMAPS["dtw"],
        vmin=0,
        vmax=100,
        label="DTW (m)",
    )
    _imshow(
        ax[5],
        fields["dtw"] - truth,
        "Base error (predicted - true DTW)",
        cmap=CMAPS["resid"],
        norm=resid_norm(60.0),
        label="Residual (m)",
    )

    src = wells["is_source"].to_numpy(bool) & ~wells["is_confined_flag"].to_numpy(bool)
    asm = assimilate(fields, wells, world, src)
    _imshow(
        ax[6],
        asm["dtw"] - np.asarray(fields["dtw"], float),
        f"Assimilation change (tau {asm['tau']:.0f} m, L {asm['length']:.0f} m)",
        cmap=CMAPS["resid"],
        norm=resid_norm(30.0),
        label="Change in DTW (m)",
    )
    draw_wells_map(ax[6], wells, src, ms=1.4)

    a = ax[7]
    dn = asm["nearest_source_dist"]
    diff = np.abs(asm["dtw"] - np.asarray(fields["dtw"], float))
    bands = [(0, 500), (500, 1500), (1500, 3000), (3000, 6000), (6000, 1e9)]
    med = [
        np.median(diff[(dn >= lo) & (dn < hi)])
        if ((dn >= lo) & (dn < hi)).any()
        else np.nan
        for lo, hi in bands
    ]
    a.plot(np.arange(len(bands)), med, marker="o", ms=2.5, color=COLORS["pred"])
    a.set_xticks(np.arange(len(bands)))
    a.set_xticklabels(["0-0.5", "0.5-1.5", "1.5-3", "3-6", "6+"])
    a.set_xlabel("Distance to nearest source well (km)")
    a.set_ylabel("Median |change| in DTW (m)")
    a.set_title("Assimilation decays to the base surface", fontsize=6)
    a.set_ylim(bottom=0)

    curve = density_curve(fields, wells, world, tau=asm["tau"], length=asm["length"])
    a = ax[8]
    a.errorbar(
        curve["n_sources"],
        curve["mad_m"],
        yerr=curve["mad_sd_m"],
        marker="o",
        ms=2.5,
        color=COLORS["pred"],
        lw=0.8,
        capsize=1.5,
        label="MAD",
    )
    a.plot(
        curve["n_sources"],
        curve["rmse_m"],
        marker="s",
        ms=2.5,
        color=COLORS["r"],
        lw=0.8,
        label="RMSE",
    )
    a.set_xlabel("Source wells revealed (count)")
    a.set_ylabel("Held-out error (m)")
    a.set_title(
        f"Density curve ({int(curve['n_wells'].iloc[0])} held-out wells within "
        f"{asm['length'] / 1000:g} km of a\ncandidate source; mean of 40 draws)",
        fontsize=6,
    )
    a.legend(loc="upper right", fontsize=5)

    held = water_table_wells(wells)
    held = held.loc[~held["is_source"].to_numpy(bool)]
    hr, hc = held["row"].to_numpy(int), held["col"].to_numpy(int)
    hobs = held["dtw_obs"].to_numpy(float)
    sb = score(np.asarray(fields["dtw"])[hr, hc], hobs)
    sa = score(asm["dtw"][hr, hc], hobs)
    so = score(oof["pred_dtw"].to_numpy(), oof["dtw_obs"].to_numpy())

    fig.tight_layout()
    p = TOY_DIR / "foundation_check_model.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"  base OOF          {so}")
    print(f"  held-out base     {sb}")
    print(f"  held-out assim.   {sa}")
    return str(p)


def sheet_3d(world, wells, fields) -> str:
    pl = r3.standard_plotter(window_size=(2400, 1600))
    r3.land_surface(pl, world)

    keep = r3.half_mask(SECTION_A, "right")
    r3.cutaway(
        pl,
        world["wte_true"],
        keep,
        scalars=np.asarray(world["dtw_true"], float),
        scalar_name="DTW (m)",
        cmap=CMAPS["dtw"],
        clim=(0.0, 100.0),
        show_scalar_bar=True,
    )
    # Both halves carry the same DTW ramp and the same limits, so the split reads
    # as one water-table surface -- truth on one side of A-A', prediction on the
    # other -- rather than as two differently coloured objects.
    r3.cutaway(
        pl,
        fields["wte"],
        ~keep,
        scalars=np.asarray(fields["dtw"], float),
        scalar_name="DTW (m)",
        cmap=CMAPS["dtw"],
        clim=(0.0, 100.0),
    )

    r3.stream_tubes(pl, world)

    prof = profile_along(
        SECTION_B, {"dem": world["dem"], "wte": world["wte_true"]}, n=200
    )
    r3.fence_section(
        pl, SECTION_B, prof["dem"], prof["wte"], color=COLORS["truth"], opacity=0.30
    )

    sel = wells.loc[wells["x"] < 14000].sample(70, random_state=1).sort_index()
    r3.well_sticks(pl, sel, sel["is_source"].to_numpy(bool))

    p = r3.shoot(pl, "foundation_check_3d")
    return str(p)


def _main() -> None:
    use_style()
    stack = load_stack()
    world, wells = stack["world"], stack["wells"]
    priors, oof, fields = stack["priors"], stack["oof"], stack["fields"]
    print("wrote", sheet_world(world, wells))
    print("wrote", sheet_priors(world, wells, priors))
    print("wrote", sheet_model(world, wells, priors, oof, fields))
    print("wrote", sheet_3d(world, wells, fields))


if __name__ == "__main__":
    _main()
