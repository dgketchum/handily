"""Figure 5 — the deployment-side state-agency split: what the map read, what scored it.

(a) The state render footprint with every held-out agency target, coloured by
    planar distance to the nearest source of either kind — an admitted agency
    site or a bundle monitoring well. Both are sources at inference, so this is
    the distance that decides how much assimilated information a scored site
    sits next to.
(b) The densest 4 km x 4 km window of admission-eligible sites, where the three
    populations resolve individually: admitted sources, held-out targets, and
    the bundle monitoring wells that were sources everywhere.

The admission funnel and the 50/50 site split are rebuilt here from the frozen
artefacts and verified: the seeded draw is replayed and must reproduce the
frozen admitted-sources file site for site, or the figure raises. The state
(``NWI_STATE``) selects the agency: Nevada NDWR (pumping and era filters) or
New Mexico OSE (use, confinement class, era, basin, relief and not-training
filters, as recorded in the admission manifest).

Written to the house figure rules in ``~/code/style/FIGURE_STYLE_GUIDE.md``:
no gridlines, no descriptive panel titles (the findings live in the caption
sidecar), framed interior legends placed with ``loc=`` only, type sizes left to
the shared mplstyle, and a perceptually uniform, luminance-ordered sequential
ramp for the distance classes so they survive grayscale.

Run::

    NWI_STATE=NM uv run python figures/nwi/fig05_ndwr_split.py

Writes ``fig05_ndwr_split.{pdf,png}`` plus a ``*_caption.txt`` sidecar to
``/data/ssd2/handily/figures/<project>/`` (``nwi`` for NV, ``nmose`` for NM).
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

from nwi_common import (
    BUNDLE_DIR,
    CFG,
    FIG_DIR,
    HUC8_POLYS,
    MM,
    RENDER_ARM,
    SITES_PATH,
    SOURCES_PATH,
    TARGETS_PATH,
    WELL_DIR,
    WIDTH_2COL,
    emptiest_corner,
    north_arrow,
    outside_letter,
    rendered_basins,
    save,
    scale_bar,
    state_outline,
    use_style,
)

#: Admission filters shared by both states.
ERA_MIN_YEAR = 1980
#: Seed of the frozen site draw (replayed in :func:`verify_split`).
SPLIT_SEED = 0

#: Distance classes, metres. The first class is the near-duplicate class: the
#: site dedup was a 1 m coordinate round, so a source and a target can be metres
#: apart.
DIST_EDGES = (0.0, 50.0, 300.0, 1000.0, 5000.0, np.inf)
DIST_LABELS = ("< 0.05", "0.05–0.3", "0.3–1", "1–5", "≥ 5")
#: viridis sampled at five levels, light for near and dark for far, so the ramp
#: is luminance-ordered (survives grayscale) and the sparse far classes are the
#: dark, visible ones over empty basin floor.
DIST_COLORS = tuple(colormaps["viridis"](v) for v in (0.72, 0.555, 0.39, 0.225, 0.06))

#: Boundary simplification tolerance for drawing panel (a) (metres).
SIMPLIFY_M = 250.0
#: The pair radius called out in panel (b) and the caption.
NEAR_DUP_M = 50.0
#: Panel (b) window: side length, and the search is over windows centred on an
#: admission-eligible site (Chebyshev radius = half the side).
WINDOW_M = 4000.0

#: Panel (b) role glyphs. Filled = the observation was visible to the map
#: (the series' well-ledger glyph), hollow = held out.
SOURCE_STYLE = {"marker": "o", "mfc": "black", "mec": "black", "ms": 2.0, "mew": 0.4}
TARGET_STYLE = {"marker": "o", "mfc": "white", "mec": "black", "ms": 2.0, "mew": 0.4}
BUNDLE_STYLE = {"marker": "+", "mec": "#0072B2", "ms": 3.2, "mew": 0.7}
#: The reserved accent, used only for annotation drawn on data: the near-duplicate
#: halo in (b) and the panel-(b) window on (a).
ACCENT = "#D55E00"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_bundle_monitoring() -> tuple[pd.DataFrame, int]:
    """Bundle monitoring wells with coordinates, and the appended-row check.

    ``nwi_common.load_bundle`` drops the coordinates, so the query nodes are
    read directly here (as Figure 2 does). In the parity bundle the admitted
    agency half IS present as training rows (``source == CFG['source_name']``);
    the count is asserted against the bundle manifest because the caption's
    "labels and sources" statement turns on it.
    """
    qn = pd.read_parquet(
        BUNDLE_DIR / "query_nodes.parquet",
        columns=["source", "well_class", "huc8", "x5070", "y5070"],
    )
    man = json.loads((BUNDLE_DIR / "graph_manifest.json").read_text())
    app = man["appended_wells"]
    n_app = int((qn["source"] == CFG["source_name"]).sum())
    if app["source_name"] != CFG["source_name"] or n_app != int(app["n_appended"]):
        raise ValueError(
            f"bundle carries {n_app:,} {CFG['source_name']} rows; manifest says "
            f"{app['n_appended']:,} appended {app['source_name']}"
        )
    mon = qn[qn["well_class"] == "monitoring"].reset_index(drop=True)
    if (mon["source"] == CFG["source_name"]).any():
        raise ValueError("an appended agency row is classed as monitoring")
    for col in ("x5070", "y5070"):
        if not np.isfinite(mon[col]).all():
            raise ValueError(f"non-finite {col} in the bundle monitoring pool")
    mon["huc8"] = mon["huc8"].astype(str).str.zfill(8)
    return mon, n_app


def load_agency() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The site panel, the admitted sources, and the targets (a partition)."""
    sites = pd.read_parquet(SITES_PATH)
    src = pd.read_parquet(SOURCES_PATH)
    tgt = pd.read_parquet(TARGETS_PATH)
    skeys = set(zip(src["sx"], src["sy"]))
    tkeys = set(zip(tgt["sx"], tgt["sy"]))
    if skeys & tkeys:
        raise ValueError(f"{len(skeys & tkeys)} sites are both source and target")
    if (skeys | tkeys) != set(zip(sites["sx"], sites["sy"])):
        raise ValueError("sources + targets are not a partition of the site panel")
    if len(skeys) != len(src) or len(tkeys) != len(tgt):
        raise ValueError("duplicate (sx, sy) keys in the frozen tables")
    return sites, src, tgt


def funnel_nv(sites: pd.DataFrame, src: pd.DataFrame, tgt: pd.DataFrame) -> dict:
    """NDWR funnel, rebuilt from the frozen tables and checked against the flag.

    The era column lives on the two halves, not on the site panel, so they are
    stacked back together first. Non-pumping AND era >= 1980 must land exactly
    on the eligibility flag, or this raises.
    """
    era = pd.concat(
        [
            src[["sx", "sy", "por_end_yr"]].assign(is_pumping=False, eligible=True),
            tgt[["sx", "sy", "por_end_yr", "is_pumping"]].assign(
                eligible=tgt["is_admission_eligible"].to_numpy(bool)
            ),
        ],
        ignore_index=True,
    )
    pumping = era["is_pumping"].to_numpy(bool)
    yr = era["por_end_yr"].to_numpy(float)
    no_era = ~pumping & ~np.isfinite(yr)
    old = ~pumping & np.isfinite(yr) & (yr < ERA_MIN_YEAR)
    keep = ~pumping & np.isfinite(yr) & (yr >= ERA_MIN_YEAR)
    if not np.array_equal(keep, era["eligible"].to_numpy(bool)):
        raise ValueError(
            "non-pumping AND era >= 1980 does not reproduce is_admission_eligible"
        )
    n_elig = int(keep.sum())
    return {
        "n_wells": int(sites["n_wells"].sum()),
        "n_sites": len(era),
        "n_multi_well_sites": int((sites["n_wells"] > 1).sum()),
        "steps": [
            ("pumping wells", int(pumping.sum())),
            (f"period of record ending before {ERA_MIN_YEAR}", int(old.sum())),
            ("period-of-record end year missing", int(no_era.sum())),
        ],
        "n_eligible": n_elig,
        "n_payload_dropped": n_elig
        - (len(src) + int(tgt["is_admission_eligible"].sum())),
        "n_admitted": len(src),
        "n_held_out": int(tgt["is_admission_eligible"].sum()),
        "split_order": "row",
    }


def funnel_nm(sites: pd.DataFrame, src: pd.DataFrame, tgt: pd.DataFrame) -> dict:
    """OSE funnel: the manifest's sequential filters replayed on the input flags.

    The site panel carries one boolean per filter; applied in the manifest's
    order each step's removal count must equal the manifest's, and the survivor
    set must equal ``is_admission_eligible_pre`` on the panel and the targets'
    ``is_admission_eligible`` (plus the frozen sources) after the payload check.
    """
    man = json.loads((WELL_DIR / CFG["manifest"]).read_text())
    mf = man["funnel"]
    if mf["n_sites_in_footprint"] != len(sites):
        raise ValueError("site panel size differs from the admission manifest")
    order = (
        ("use_ok", "dropped_use_ok", f"use code not in {{{man['use_codes']}}}"),
        (
            "class_ok",
            "dropped_class_ok",
            f"confinement class not in {{{man['classes']}}}",
        ),
        (
            "era_ok",
            "dropped_era_ok",
            f"period of record ending before {man['era_min_year']} or missing",
        ),
        (
            "basin_ok",
            "dropped_basin_ok",
            f"OSE basin {man['exclude_pod_basins']} or HUC8 {man['exclude_huc8']}",
        ),
        (
            "relief_ok",
            "dropped_relief_ok",
            f"DEM range over the accuracy-class window above {man['relief_max_m']:g} m",
        ),
        ("not_training", "dropped_not_training", "already a training well"),
    )
    alive = np.ones(len(sites), bool)
    steps = []
    for col, key, label in order:
        ok = sites[col].to_numpy(bool)
        dropped = int((alive & ~ok).sum())
        if dropped != int(mf[key]):
            raise ValueError(f"{col}: replay drops {dropped}, manifest says {mf[key]}")
        alive &= ok
        steps.append((label, dropped))
    if not np.array_equal(alive, sites["is_admission_eligible_pre"].to_numpy(bool)):
        raise ValueError(
            "sequential filters do not reproduce is_admission_eligible_pre"
        )
    if int(alive.sum()) != int(mf["n_eligible_pre_payload"]):
        raise ValueError("eligible-pre count differs from the manifest")
    elig_keys = set(zip(sites.loc[alive, "sx"], sites.loc[alive, "sy"]))
    held = tgt[tgt["is_admission_eligible"].to_numpy(bool)]
    post = set(zip(src["sx"], src["sy"])) | set(zip(held["sx"], held["sy"]))
    if len(elig_keys - post) != int(mf["dropped_payload"]) or not post <= elig_keys:
        raise ValueError("payload check does not reproduce the eligible set")
    if len(src) != int(mf["n_admitted"]) or len(held) != int(mf["n_held_out_eligible"]):
        raise ValueError("admitted / held-out counts differ from the manifest")
    if man["split"]["seed"] != SPLIT_SEED or man["split"]["fraction"] != 0.5:
        raise ValueError(f"manifest split rule {man['split']} is not the one drawn")
    return {
        "n_wells": int(mf["n_wells"]),
        "n_sites": len(sites),
        "n_multi_well_sites": int((sites["n_wells"] > 1).sum()),
        "steps": steps,
        "n_eligible": int(alive.sum()),
        "n_payload_dropped": int(mf["dropped_payload"]),
        "n_admitted": len(src),
        "n_held_out": len(held),
        "split_order": "sorted",
        "manifest": man,
    }


def funnel(sites, src, tgt) -> dict:
    return (funnel_nm if CFG["manifest"] else funnel_nv)(sites, src, tgt)


def verify_split(sites: pd.DataFrame, src: pd.DataFrame, tgt: pd.DataFrame, f) -> int:
    """Replay the seeded draw; return the number of sites it reproduces.

    The draw is ``default_rng(0).choice(n_eligible, n_admitted, replace=False)``
    over the admission-eligible sites — in the site panel's own row order (NV)
    or in (sx, sy)-sorted order (NM, as its manifest records). Reproducing the
    frozen file exactly is the proof that this figure's source/target labels
    are the ones the deployed map used.
    """
    held = tgt[tgt["is_admission_eligible"].to_numpy(bool)]
    elig_keys = set(zip(src["sx"], src["sy"])) | set(zip(held["sx"], held["sy"]))
    panel = sites[["sx", "sy"]]
    if f["split_order"] == "sorted":
        panel = panel.sort_values(["sx", "sy"], kind="stable")
    key = list(zip(panel["sx"], panel["sy"]))
    elig_rows = [k for k in key if k in elig_keys]
    rng = np.random.default_rng(SPLIT_SEED)
    pick = rng.choice(len(elig_rows), f["n_admitted"], replace=False)
    drawn = {elig_rows[i] for i in pick}
    return len(drawn & set(zip(src["sx"], src["sy"])))


def nearest_source_m(targets: pd.DataFrame, pools: list[np.ndarray]) -> np.ndarray:
    """Planar EPSG:5070 distance (m) from each target to the nearest pool point."""
    pool = np.vstack(pools)
    d = cKDTree(pool).query(
        targets[["x5070", "y5070"]].to_numpy("float64"), workers=-1
    )[0]
    if not np.isfinite(d).all():
        raise ValueError("non-finite nearest-source distance")
    return d


def choose_window(elig_xy: np.ndarray) -> tuple[float, float, int]:
    """Densest ``WINDOW_M`` square: centred on the eligible site with the most
    eligible sites inside a Chebyshev radius of half the side.

    Deterministic — ties resolve to the first such site in the array order
    (sources first, then held-out targets, each in file row order).
    """
    tree = cKDTree(elig_xy)
    counts = np.array(
        [
            len(i)
            for i in tree.query_ball_point(
                elig_xy, r=WINDOW_M / 2, p=np.inf, workers=-1
            )
        ]
    )
    best = int(np.argmax(counts))
    return float(elig_xy[best, 0]), float(elig_xy[best, 1]), int(counts[best])


def emptiest_slot(fx, fy, w: float, h: float, blocked, step: float = 0.01) -> tuple:
    """Lowest well-free slot of size ``w`` x ``h``, in axes fractions.

    Map furniture has to sit somewhere on a panel that is data everywhere, so
    its position is searched rather than fixed: the panel is scanned bottom-up
    on a ``step`` grid, skipping anything that touches an already-placed box in
    ``blocked``, and the first slot enclosing no plotted well wins. Deterministic
    (first minimum in scan order), and it returns the count so the caller can
    report how many symbols the furniture hides when no clear slot exists.
    """
    fx = np.asarray(fx, float)
    fy = np.asarray(fy, float)
    best = None
    for y in np.arange(0.02, 0.96 - h, step):
        for x in np.arange(0.02, 0.98 - w, step):
            bx = (x, x + w, y, y + h)
            if any(
                not (bx[1] < b[0] or bx[0] > b[1] or bx[3] < b[2] or bx[2] > b[3])
                for b in blocked
            ):
                continue
            n = int(
                ((fx >= bx[0]) & (fx <= bx[1]) & (fy >= bx[2]) & (fy <= bx[3])).sum()
            )
            if best is None or n < best[0]:
                best = (n, float(x), float(y), bx)
                if n == 0:
                    return best
    if best is None:
        raise ValueError("no furniture slot fits; every candidate is blocked")
    return best


def zoom_scale_bar(ax, length_m: float, x_frac: float, y_frac: float) -> None:
    """Two-segment scale bar labelled in metres, for the sub-kilometre panel.

    ``nwi_common.scale_bar`` writes whole-kilometre numerals, which round a
    500 m bar to ``0``, so the zoom panel carries its own.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + x_frac * (x1 - x0)
    y = y0 + y_frac * (y1 - y0)
    h = 0.010 * (y1 - y0)
    for i in range(2):
        ax.add_patch(
            Rectangle(
                (x + i * length_m / 2, y),
                length_m / 2,
                h,
                facecolor="black" if i == 0 else "white",
                edgecolor="black",
                linewidth=0.5,
                zorder=6,
            )
        )
    ax.text(x, y + 1.3 * h, "0", ha="center", va="bottom", zorder=6)
    ax.text(
        x + length_m,
        y + 1.3 * h,
        f"{length_m:.0f} m",
        ha="left",
        va="bottom",
        zorder=6,
    )


def in_box(df: pd.DataFrame, box: tuple[float, float, float, float]) -> pd.DataFrame:
    x0, x1, y0, y1 = box
    m = (
        (df["x5070"] >= x0)
        & (df["x5070"] <= x1)
        & (df["y5070"] >= y0)
        & (df["y5070"] <= y1)
    )
    return df[m]


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def panel_footprint(
    ax, tgt_xy, dist_m, basin_polys, footprint, state, box
) -> np.ndarray:
    """Held-out targets over the render footprint, classed by source distance."""
    # Boundaries are simplified to SIMPLIFY_M before drawing: at this scale one
    # printed millimetre is about 12 km, so the full WBD vertex set is
    # invisible detail that only inflates the PDF.
    basin_polys.simplify(SIMPLIFY_M).plot(
        ax=ax, facecolor="white", edgecolor="#bdbdbd", linewidth=0.2
    )
    footprint.simplify(SIMPLIFY_M).boundary.plot(
        ax=ax, color="#4d4d4d", linewidth=0.4, zorder=2
    )
    state.simplify(SIMPLIFY_M).boundary.plot(
        ax=ax, color="black", linewidth=0.7, zorder=3
    )

    cls = np.digitize(dist_m, DIST_EDGES[1:-1], right=False)
    counts = np.bincount(cls, minlength=len(DIST_LABELS))
    # One rasterized artist for all points (five would embed five full-panel
    # images and blow the PDF size cap). The far classes are drawn first: they
    # sit in the empty basin floor where nothing overplots them, while the
    # crowded near class would otherwise be buried under them.
    order = np.argsort(-cls, kind="stable")
    ax.scatter(
        tgt_xy[order, 0],
        tgt_xy[order, 1],
        s=2.4,
        c=[DIST_COLORS[c] for c in cls[order]],
        linewidths=0.0,
        zorder=4,
        rasterized=True,
    )

    x0, x1, y0, y1 = box
    # The window's true extent is 4 km, a third of a printed millimetre here, so
    # the true rectangle is drawn and an open square marker of legible size is
    # centred on it (the caption says the symbol is larger than the extent).
    ax.plot(
        [(x0 + x1) / 2],
        [(y0 + y1) / 2],
        marker="s",
        ls="none",
        ms=5.0,
        mfc="none",
        mec=ACCENT,
        mew=0.6,
        zorder=10,
    )
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="none",
            edgecolor=ACCENT,
            linewidth=0.6,
            zorder=10,
        )
    )

    ax.set_axis_off()
    ax.set_aspect("equal")
    scale_bar(ax, length_m=200_000.0, x_frac=0.03, y_frac=0.02)
    north_arrow(ax, x_frac=0.93, y_frac=0.05)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=2.6,
            mfc=DIST_COLORS[c],
            mec="none",
            label=DIST_LABELS[c],
        )
        for c in range(len(DIST_LABELS))
    ]
    ax.legend(
        handles=handles,
        loc=emptiest_corner(
            ax, tgt_xy[:, 0], tgt_xy[:, 1], ("upper left", "upper right")
        ),
        title="Distance to nearest source (km)",
    )
    return counts


def panel_zoom(ax, box, src_in, tgt_in, mon_in, near_dup) -> dict:
    """The three populations inside the densest eligible-site window."""
    x0, x1, y0, y1 = box
    ax.plot(
        tgt_in.loc[near_dup, "x5070"],
        tgt_in.loc[near_dup, "y5070"],
        marker="o",
        ls="none",
        ms=4.0,
        mfc="none",
        mec=ACCENT,
        mew=0.5,
        zorder=4,
    )
    ax.plot(src_in["x5070"], src_in["y5070"], ls="none", zorder=5, **SOURCE_STYLE)
    ax.plot(tgt_in["x5070"], tgt_in["y5070"], ls="none", zorder=6, **TARGET_STYLE)
    ax.plot(mon_in["x5070"], mon_in["y5070"], ls="none", zorder=7, **BUNDLE_STYLE)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)

    px = np.concatenate([src_in["x5070"], tgt_in["x5070"], mon_in["x5070"]])
    py = np.concatenate([src_in["y5070"], tgt_in["y5070"], mon_in["y5070"]])
    fx = (px - x0) / (x1 - x0)
    fy = (py - y0) / (y1 - y0)

    ag = CFG["agency"]
    handles = [
        Line2D([], [], ls="none", label=f"admitted {ag} source", **SOURCE_STYLE),
        Line2D([], [], ls="none", label=f"held-out {ag} target", **TARGET_STYLE),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=4.0,
            mfc="none",
            mec=ACCENT,
            mew=0.5,
            label=f"target within {NEAR_DUP_M:.0f} m of a source",
        ),
        Line2D([], [], ls="none", label="bundle monitoring well", **BUNDLE_STYLE),
    ]
    loc = emptiest_corner(
        ax, px, py, ("lower right", "lower left", "upper left", "upper right")
    )
    leg = ax.legend(handles=handles, loc=loc)

    # No corner of a dense 4 km window is empty, so the furniture is placed by
    # search, not by hand: the legend takes the emptiest corner and the scale
    # bar and north arrow take the lowest slots that clear both it and each
    # other and enclose no well at all. The legend's own frame is measured, not
    # guessed, so the furniture keeps a real 2 % margin from it.
    ax.figure.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    blocked = [(bb.x0 - 0.02, bb.x1 + 0.02, bb.y0 - 0.02, bb.y1 + 0.02)]
    n_bar, bx, by, bar_box = emptiest_slot(fx, fy, 0.24, 0.09, blocked)
    # the bar starts inside the slot, leaving room for the centred '0' numeral
    zoom_scale_bar(ax, 500.0, x_frac=bx + 0.035, y_frac=by + 0.015)
    blocked.append(bar_box)
    n_arrow, ax_f, ay_f, _ = emptiest_slot(fx, fy, 0.07, 0.13, blocked)
    north_arrow(ax, x_frac=ax_f + 0.035, y_frac=ay_f + 0.01)
    return {
        "legend": leg,
        "loc": loc,
        "px": px,
        "py": py,
        "n_bar": n_bar,
        "n_arrow": n_arrow,
    }


# --------------------------------------------------------------------------
# Caption
# --------------------------------------------------------------------------


def _funnel_text(f: dict) -> str:
    ag, agl, name = CFG["agency"], CFG["agency_long"], CFG["name"]
    n = f["n_sites"]
    steps = []
    for label, dropped in f["steps"]:
        n -= dropped
        steps.append(f"{label} out ({dropped:,} sites removed, leaving {n:,})")
    seq = "; ".join(steps)
    if CFG["manifest"]:
        man = f["manifest"]
        win = ", ".join(f"{k} {v}" for k, v in man["relief_window_cells"].items())
        return (
            f"The funnel. The {agl} ({ag}) panel starts at {f['n_wells']:,} "
            f"well records with a plausible static water level in the {name} "
            f"render footprint (HUC8 prefixes {man['huc8_prefixes']}), collapsed "
            f"to {f['n_sites']:,} sites by rounding EPSG:5070 coordinates to 1 m "
            f"and taking the site median ({f['n_multi_well_sites']:,} sites hold "
            f"more than one well record). Six admission filters then run, in "
            f"this order: {seq}. The relief filter reads the DEM range over a "
            f"square window whose side, in 10 m cells, is set by the site's "
            f"horizontal-accuracy class ({win}), so a PLSS-section coordinate is "
            f"judged on the relief of the whole section it could fall in. A "
            f"finite-payload check on the assimilated quantity, the water-table "
            f"elevation residual (land-surface elevation minus observed depth to "
            f"water, minus the regional base R), removes {f['n_payload_dropped']} "
            f"further sites. The surviving {f['n_eligible']:,} admission-eligible "
            f"sites are split 50/50 BY SITE by a single seeded draw (numpy "
            f"default_rng, seed {SPLIT_SEED}; {f['n_admitted']:,} of "
            f"{f['n_eligible']:,} sites drawn without replacement over the sites "
            f"in (x, y)-sorted order): {f['n_admitted']:,} became admitted "
            f"sources, frozen and handed to the deployed model as extra "
            f"observations, and {f['n_held_out']:,} stayed held out as the "
            f"primary scoring panel. Every admitted site is an unconfined or "
            f"marginally unconfined {', '.join(man['use_codes'].split(','))} "
            f"well, so the scored panel is a water-table panel; most are "
            f"domestic wells carrying one static level recorded at completion."
        )
    return (
        f"The funnel. The {agl} ({ag}) panel starts at {f['n_wells']:,} non-NWIS "
        f"{name} well records outside the continental training bundle with a "
        f"plausible water level, collapsed to {f['n_sites']:,} sites by rounding "
        f"EPSG:5070 coordinates to 1 m and taking the site median "
        f"({f['n_multi_well_sites']:,} sites hold more than one well record). "
        f"Three admission filters then run, in this order: {seq}. A "
        f"finite-payload check on the assimilated quantity, the water-table "
        f"elevation residual (land-surface elevation minus observed depth to "
        f"water, minus the regional base R), removes {f['n_payload_dropped']} "
        f"further sites. The surviving {f['n_eligible']:,} admission-eligible "
        f"sites are split 50/50 BY SITE by a single seeded draw (numpy "
        f"default_rng, seed {SPLIT_SEED}; {f['n_admitted']:,} of "
        f"{f['n_eligible']:,} sites drawn without replacement in the panel's row "
        f"order): {f['n_admitted']:,} became admitted sources, frozen and handed "
        f"to the deployed model as extra observations, and {f['n_held_out']:,} "
        f"stayed held out as the primary scoring panel. No confinement screen "
        f"ran at any stage of the funnel: the sites are {ag} driller-report and "
        f"monitoring records of every confinement class, and each carries one "
        f"static level recorded at completion at a PLSS quarter-quarter "
        f"coordinate, not a period-of-record mean."
    )


def _caption(f, counts, n_basins, n_mon_st, n_mon_conus, win, matched, app) -> str:
    ag, name = CFG["agency"], CFG["name"]
    bins = "; ".join(f"{lab} km, {int(n):,}" for lab, n in zip(DIST_LABELS, counts))
    n_held = f["n_held_out"]
    return (
        f"Figure 5. Which {name} {ag} wells did the deployed map assimilate as "
        "sources, which were held out to score it, and how are the two sets "
        "arranged in space? The two halves are one seeded coin flip apart and "
        "sit on top of each other on the ground.\n\n"
        + _funnel_text(f)
        + f" Replaying that draw here reproduces the frozen admitted file "
        f"{matched:,} sites out of {f['n_admitted']:,}. Deduplication before the "
        "split was by coordinate alone — sites are distinct whenever their "
        "coordinates differ by more than 1 m — so two wells a few metres apart "
        "are two sites and can fall on opposite sides of the split; this is why "
        "the scored error must be read against distance to the nearest source "
        "(Figure 7) and not as a single pooled number.\n\n"
        f"(a) The {n_basins}-basin {name} render footprint, with each of the "
        f"{n_held:,} held-out {ag} targets drawn as a dot coloured by the planar "
        "EPSG:5070 distance to the nearest source of either kind — an admitted "
        f"{ag} site or a bundle monitoring well, whichever is closer, both being "
        f"sources at inference. Targets per class: {bins}. {counts[0]:,} targets "
        f"({counts[0] / n_held:.0%}) lie within {NEAR_DUP_M:.0f} m of a source "
        f"and {counts[0] + counts[1]:,} ({(counts[0] + counts[1]) / n_held:.0%}) "
        "within 300 m. The distance pool is the full deployment pool: all "
        f"{n_mon_conus:,} bundle monitoring wells (of which {n_mon_st:,} fall "
        f"inside the footprint) plus the {f['n_admitted']:,} admitted {ag} "
        f"sites. Neither source population is drawn here — "
        f"{n_mon_st + f['n_admitted']:,} further points would bury the classes — "
        "and the colour ramp is viridis sampled at five levels, light near to "
        "dark far, so the classes are ordered by lightness alone. Hairlines are "
        "HUC8 boundaries, the dark grey line the render footprint and the black "
        f"line the {name} state boundary; the footprint extends past it because "
        f"HUC8 basins straddle the border. The small square is the "
        f"{WINDOW_M / 1000:.0f} km x {WINDOW_M / 1000:.0f} km window of panel "
        "(b).\n\n"
        f"(b) That window: the {WINDOW_M / 1000:.0f} km square, centred on an "
        "admission-eligible site, that contains the most admission-eligible "
        f"sites ({win['n_elig']:,} of them; ties break to the first such site in "
        "file order). It holds "
        f"{win['n_src']:,} admitted sources (filled circles), {win['n_tgt']:,} "
        "held-out targets (hollow circles — the series' well-ledger glyph: "
        "filled means the observation was visible to the map, hollow means it "
        f"was held back) and {win['n_mon']:,} bundle monitoring wells (crosses), "
        f"which were sources here as everywhere. The two {ag} halves are "
        "interleaved at every scale, as a by-site random split must be: "
        f"{win['n_near']:,} of the window's held-out targets sit within "
        f"{NEAR_DUP_M:.0f} m of an admitted source and carry a ring, and their "
        f"median separation is {win['med_near_m']:.0f} m — below about "
        f"{win['res_m']:.0f} m a pair is narrower than its own plotted marker "
        "and the ring is the only thing that shows it. "
        + (
            f"A further {win['n_ineligible']:,} {ag} sites in the window are not "
            "drawn: they failed admission and are scored only as secondary panels."
            if win["n_ineligible"]
            else f"Every {ag} site in the window passed admission, so none is "
            "left undrawn."
        )
        + (
            f" The panel-(b) legend sits in the emptiest corner and still covers "
            f"{win['n_hidden']} of the window's {win['n_sym']:,} symbols."
            if win["n_hidden"]
            else ""
        )
        + "\n\n"
        f"In the shipped render ({RENDER_ARM}) the admitted half was also "
        f"appended to the training bundle as ordinary labels ({app['n_appended']:,} "
        f"rows kept of {f['n_admitted']:,}: {app['nonfinite']} lacked a finite "
        f"FAC-REM prior and {app['colocated']} sat within 1 m of a bundle well; "
        "Figure 1a) and then handed to the model again at render time as "
        "observations, so the filled circles were both labels and sources; the "
        "hollow circles were neither. Coordinates and all distances are in the "
        "Albers equal-area projection (EPSG:5070); hydrologic units are from the "
        "WBD and the state boundary from the 2022 US Census cartographic "
        "boundary file.\n"
    )


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    import geopandas as gpd

    use_style()

    sites, src, tgt = load_agency()
    f = funnel(sites, src, tgt)
    matched = verify_split(sites, src, tgt, f)
    if matched != f["n_admitted"]:
        raise ValueError(
            f"the seed-{SPLIT_SEED} replay reproduces {matched} of "
            f"{f['n_admitted']} admitted sites; the split rule drawn here is not "
            "the one that was frozen"
        )

    mon, n_app = load_bundle_monitoring()
    man = json.loads((BUNDLE_DIR / "graph_manifest.json").read_text())
    fun = man["appended_wells"]["funnel"]
    if fun["wells_in"] != f["n_admitted"]:
        raise ValueError("the appended-label funnel did not start from the frozen file")
    app = {
        "n_appended": n_app,
        "nonfinite": int(fun["dropped_nonfinite_fac_rem"]),
        "colocated": int(fun["dropped_collocated_le_1m"]),
    }
    basins = rendered_basins()
    mon_st = mon[mon["huc8"].isin(basins)]

    held = tgt[tgt["is_admission_eligible"].to_numpy(bool)].reset_index(drop=True)
    src_xy = src[["x5070", "y5070"]].to_numpy("float64")
    mon_xy = mon[["x5070", "y5070"]].to_numpy("float64")
    dist_m = nearest_source_m(held, [src_xy, mon_xy])
    dist_ag_m = nearest_source_m(held, [src_xy])

    elig_xy = np.vstack([src_xy, held[["x5070", "y5070"]].to_numpy("float64")])
    cx, cy, n_elig = choose_window(elig_xy)
    half = WINDOW_M / 2
    box = (cx - half, cx + half, cy - half, cy + half)

    src_in = in_box(src, box)
    tgt_in = in_box(held, box).reset_index(drop=True)
    mon_in = in_box(mon, box)
    inelig_in = in_box(tgt[~tgt["is_admission_eligible"].to_numpy(bool)], box)
    d_in = cKDTree(src_in[["x5070", "y5070"]].to_numpy("float64")).query(
        tgt_in[["x5070", "y5070"]].to_numpy("float64"), workers=-1
    )[0]
    near_dup = d_in < NEAR_DUP_M

    h8 = gpd.read_parquet(HUC8_POLYS, columns=["huc8", "geometry"])
    h8["huc8"] = h8["huc8"].astype(str).str.zfill(8)
    basin_polys = h8[h8["huc8"].isin(basins)]
    footprint = basin_polys.dissolve()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_2COL, 95 * MM),
        gridspec_kw={"width_ratios": [0.70, 1.0], "wspace": 0.06},
    )
    counts = panel_footprint(
        axes[0],
        held[["x5070", "y5070"]].to_numpy("float64"),
        dist_m,
        basin_polys,
        footprint,
        state_outline(),
        box,
    )
    zoom = panel_zoom(axes[1], box, src_in, tgt_in, mon_in, near_dup)
    outside_letter(axes[0], "a")
    outside_letter(axes[1], "b")

    # How many wells the panel-(b) legend frame hides, measured from the drawn
    # legend rather than assumed: no corner of the window is empty, so the count
    # is stated in the caption instead of being papered over.
    fig.canvas.draw()
    bb = zoom["legend"].get_window_extent().transformed(axes[1].transData.inverted())
    n_hidden = int(
        (
            (zoom["px"] >= bb.x0)
            & (zoom["px"] <= bb.x1)
            & (zoom["py"] >= bb.y0)
            & (zoom["py"] <= bb.y1)
        ).sum()
    )

    stem = "fig05_ndwr_split"
    paths = save(fig, stem, FIG_DIR)
    plt.close(fig)

    # marker diameter in ground metres, for the caption's resolution statement
    panel_w_in = WIDTH_2COL / 1.70
    res_m = WINDOW_M * (TARGET_STYLE["ms"] / 72.0) / panel_w_in
    win = {
        "n_elig": n_elig,
        "n_src": len(src_in),
        "n_tgt": len(tgt_in),
        "n_mon": len(mon_in),
        "n_ineligible": len(inelig_in),
        "n_near": int(near_dup.sum()),
        "med_near_m": float(np.median(d_in[near_dup])) if near_dup.any() else np.nan,
        "res_m": res_m,
        "n_hidden": n_hidden,
        "n_sym": len(zoom["px"]),
    }
    cap_path = FIG_DIR / f"{stem}_caption.txt"
    cap_path.write_text(
        _caption(f, counts, len(basins), len(mon_st), len(mon), win, matched, app)
    )

    print("funnel:", {k: v for k, v in f.items() if k != "manifest"})
    print(f"seed-{SPLIT_SEED} replay reproduces {matched:,}/{f['n_admitted']:,} sites")
    print("distance to nearest source of either kind (n targets):")
    for lab, n in zip(DIST_LABELS, counts):
        print(f"  {lab:>9} km  {n:6,}")
    print(f"distance to nearest ADMITTED {CFG['agency']} source only:")
    for lab, n in zip(DIST_LABELS, np.histogram(dist_ag_m, bins=list(DIST_EDGES))[0]):
        print(f"  {lab:>9} km  {n:6,}")
    print(
        f"window centre ({cx:.0f}, {cy:.0f}) 5070 m, eligible in window {n_elig:,}; "
        f"sources {win['n_src']:,}, targets {win['n_tgt']:,}, bundle "
        f"{win['n_mon']:,}, ineligible {win['n_ineligible']:,}, "
        f"pairs <{NEAR_DUP_M:.0f} m {win['n_near']:,} "
        f"(median {win['med_near_m']:.0f} m)"
    )
    print(
        f"panel (b) furniture: legend ({zoom['loc']}) hides {n_hidden} of "
        f"{len(zoom['px']):,} symbols; scale bar slot encloses {zoom['n_bar']}, "
        f"north arrow slot {zoom['n_arrow']}"
    )
    for ext, p in paths.items():
        print(f"{ext}: {p}")
    print(f"caption: {cap_path}")


if __name__ == "__main__":
    main()
