"""Figure 2 — the leave-fold-out construction of the regional base R.

(a) One monitoring well in the state footprint and the 32 wells that actually
    set its base R. The pool is every OTHER fold's monitoring wells, so the
    dense cluster of same-fold wells around the target — the ones a naive kNN
    would have used — is excluded, and the neighbour set is pulled out to the
    valley margins.
(b) The same exclusion over every monitoring well in the state render
    footprint: the planar distance out to the 32nd neighbour, with the own fold
    excluded (the actual construction) and with it retained (the counterfactual).

The neighbour metric is the one ``utils.build_conus_graph_inputs.crossfit_idw``
uses: a relief-lifted coordinate ``(x, y, vw * z_surf)`` in EPSG:5070 metres,
with ``vw``, ``k`` and ``power`` read from the bundle manifest, never assumed.
The reported distances are plain planar 5070 distances so they can be read as
ground distance.

The parity bundle this state's model was trained on differs from the base
bundle in two documented ways, both reproduced and checked here: the archived
R at every monitoring well is the base cross-fit value capped at ground minus
``r_ground_cap_m``, and the appended state-agency wells carry a lattice read of
that same archived field rather than a cross-fit of their own. The pool of the
cross-fit is therefore the monitoring wells only, which is why they are the
population mapped and counted.

Written to the house figure rules in ``~/code/style/FIGURE_STYLE_GUIDE.md``:
no gridlines, no descriptive panel titles (findings live in the caption sidecar),
framed interior legends placed with ``loc=`` only, type sizes left to the shared
mplstyle, and a colour-vision-deficiency-safe fold palette shared with Figure 1.

Run::

    NWI_STATE=NM uv run python figures/nwi/fig02_crossfit_r.py
    NWI_STATE=NM uv run python figures/nwi/fig02_crossfit_r.py --well gwx_41ffc1f4

Writes ``fig02_crossfit_r.{pdf,png}`` plus a ``*_caption.txt`` sidecar to
``/data/ssd2/handily/figures/<project>/`` (``nwi`` for NV, ``nmose`` for NM).
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, ScalarFormatter
from scipy.spatial import cKDTree

from nwi_common import (
    BUNDLE_DIR,
    CFG,
    FIG_DIR,
    FOLD_COLORS,
    HUC12_POLYS,
    MM,
    WELL_CLASSES,
    WIDTH_2COL,
    north_arrow,
    outside_letter,
    rendered_basins,
    save,
    round_bar_length_m,
    scale_bar,
    state_outline,
    use_style,
)

#: Colours for the two panel-(b) curves. The actual construction is the focal
#: black series; the counterfactual is grey context (guide §7), and the pair is
#: solid vs dashed so it survives grayscale.
EXCLUDED_STYLE = {"color": "black", "linestyle": "-", "linewidth": 1.0}
RETAINED_STYLE = {"color": "#808080", "linestyle": (0, (3, 1.5)), "linewidth": 1.0}

#: Selection rule thresholds (see :func:`choose_target`).
MIN_SAME_FOLD_10KM = 8
NEAR_RADIUS_M = 10_000.0
MAX_ELONGATION = 2.0


# --------------------------------------------------------------------------
# The construction, reproduced from the bundle
# --------------------------------------------------------------------------


def load_manifest() -> dict:
    """The IDW parameters and the parity contract, read from the bundle manifest."""
    man = json.loads((BUNDLE_DIR / "graph_manifest.json").read_text())
    pr = man["parity_rewrite"]
    par = {
        "vw": float(man["r_relief_vw"]),
        "k": int(man["idw_k"]),
        "power": float(man["idw_power"]),
        "residual_base": man["residual_base"],
        "cv_scheme": man["cv_scheme"],
        "cap_m": float(man["r_ground_cap_m"]),
        "appended_r": pr["appended_r"],
        "exclude_source_name": list(pr["render_contract"]["exclude_source_name"]),
    }
    if par["residual_base"] != "relief_idw":
        raise ValueError(
            f"bundle residual base is {par['residual_base']!r}; this figure "
            "documents the relief-IDW base R"
        )
    ar = par["appended_r"]
    if (int(ar["k"]), float(ar["power"]), float(ar["relief_vw"])) != (
        par["k"],
        par["power"],
        par["vw"],
    ):
        raise ValueError("the appended-well lattice R uses a different kNN recipe")
    return par


def load_monitoring() -> tuple[pd.DataFrame, dict]:
    """Monitoring wells only — the neighbour pool of the base-R cross-fit.

    ``crossfit_idw`` is called with the base bundle's real pool, which is exactly
    the monitoring class; the appended state-agency rows and the water-stage
    pseudo-rows are never in it. Returns the pool and the counts of the rows
    kept out of it, by class.
    """
    qn = pd.read_parquet(
        BUNDLE_DIR / "query_nodes.parquet",
        columns=[
            "canonical_id",
            "x5070",
            "y5070",
            "huc8",
            "cv_unit",
            "cv_fold",
            "well_class",
            "is_water_pseudo",
            "z_surf_well_m",
            "wte_obs_m",
            "regional_wte_idw_oof_m",
        ],
    )
    allowed = set(WELL_CLASSES) | {"water_pseudo"}
    extra = set(qn["well_class"].unique()) - allowed
    if extra:
        raise ValueError(f"unexpected well_class values: {extra}")
    is_mon = qn["well_class"] == "monitoring"
    mon = qn[is_mon].reset_index(drop=True)
    others = qn.loc[~is_mon, "well_class"].value_counts().to_dict()
    if int(qn["is_water_pseudo"].astype(bool).sum()) != others.get("water_pseudo", 0):
        raise ValueError("is_water_pseudo disagrees with well_class == water_pseudo")
    mon["huc8"] = mon["huc8"].astype(str).str.zfill(8)
    for col in ("x5070", "y5070", "z_surf_well_m", "wte_obs_m"):
        if not np.isfinite(mon[col]).all():
            raise ValueError(f"non-finite {col} in the monitoring pool")
    return mon, {str(k): int(v) for k, v in others.items()}


def relief_coords(x, y, z, vw: float) -> np.ndarray:
    """``_relief_coords``: the lifted (x, y, vw*z) space the kNN runs in."""
    return np.column_stack(
        [
            np.asarray(x, "float64"),
            np.asarray(y, "float64"),
            vw * np.asarray(z, "float64"),
        ]
    )


def neighbour_sets(coords: np.ndarray, fold: np.ndarray, k: int):
    """Per-well neighbour indices, own fold excluded and own fold retained.

    Excluded is the construction itself: for each fold, a KD-tree over every
    OTHER fold's pool row. Retained is the counterfactual: the same lifted kNN
    over the whole pool with only the well itself dropped.
    """
    n = len(coords)
    excl = np.empty((n, k), dtype=np.int64)
    for f in np.unique(fold):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        _, idx = cKDTree(coords[tr]).query(coords[te], k=k)
        excl[te] = tr[idx]

    _, idx_all = cKDTree(coords).query(coords, k=k + 1)
    keep = idx_all != np.arange(n)[:, None]
    # A well is its own nearest neighbour unless another row sits at the exact
    # same lifted coordinate; then the tie can order the twin first and the well
    # itself appears later. Dropping the first non-self column keeps k either way.
    if (keep.sum(axis=1) < k).any():
        raise ValueError("a well matched itself more than once in the retained kNN")
    ret = np.array([row[m][:k] for row, m in zip(idx_all, keep)], dtype=np.int64)
    return excl, ret


def verify_reproduction(mon: pd.DataFrame, excl: np.ndarray, par: dict) -> dict:
    """Rebuild the archived R from the neighbour sets; return the deviations.

    The archived column is what the model was actually trained against, so an
    exact match is the proof that this figure's neighbour sets ARE the ones the
    bundle used. In the parity bundle the archived value is the cross-fit R
    lowered to ``z_surf - cap_m`` wherever it sat above that, so the rebuild
    applies the same cap and also reports how many wells it binds.
    """
    val = mon["wte_obs_m"].to_numpy("float64")
    z = mon["z_surf_well_m"].to_numpy("float64")
    coords = relief_coords(mon["x5070"], mon["y5070"], z, par["vw"])
    d = np.linalg.norm(coords[excl] - coords[:, None, :], axis=2)
    w = 1.0 / np.maximum(d, 1.0) ** par["power"]
    r = (w * val[excl]).sum(1) / w.sum(1)
    bound = r > z - par["cap_m"]
    r_cap = np.minimum(r, z - par["cap_m"])
    archived = mon["regional_wte_idw_oof_m"].to_numpy("float64")
    return {
        "max_dev_m": float(np.max(np.abs(r_cap - archived))),
        "n_bound": int(bound.sum()),
        "median_lowering_m": float(np.median((r - r_cap)[bound]))
        if bound.any()
        else 0.0,
    }


def planar_far_km(xy: np.ndarray, nbr: np.ndarray) -> np.ndarray:
    """Planar 5070 distance (km) to the farthest of a well's k neighbours."""
    far = nbr[:, -1]
    return np.hypot(xy[far, 0] - xy[:, 0], xy[far, 1] - xy[:, 1]) / 1000.0


# --------------------------------------------------------------------------
# Target selection — deterministic, and the rule is the code below
# --------------------------------------------------------------------------


def choose_target(mon, xy, fold, excl, ret, in_state) -> int:
    """Pick the mapped well by a fixed rule, so the figure is reproducible.

    Among monitoring wells inside the state render footprint, keep those where
    the exclusion visibly matters and the neighbourhood maps cleanly:

    1. at least ``MIN_SAME_FOLD_10KM`` other same-fold monitoring wells within
       ``NEAR_RADIUS_M`` (so there is a same-fold cluster to exclude);
    2. all k own-fold-retained neighbours are same-fold — the exclusion replaces
       the entire neighbour set, not a corner of it;
    3. the target-plus-neighbours bounding box is no more elongated than
       ``MAX_ELONGATION``, so the neighbourhood fills a square map panel.

    Rank the survivors by the number of distinct folds among the neighbours
    actually used (the richest fold key), then by how far the exclusion pushes
    the 32nd neighbour out, then by canonical id. Ties therefore never depend on
    row order.
    """
    tree = cKDTree(xy)
    far_e = planar_far_km(xy, excl)
    far_r = planar_far_km(xy, ret)
    rows = []
    for q in np.where(in_state)[0]:
        f = fold[q]
        if not (fold[ret[q]] == f).all():
            continue
        near = np.array(tree.query_ball_point(xy[q], NEAR_RADIUS_M), dtype=np.int64)
        near = near[near != q]
        if int((fold[near] == f).sum()) < MIN_SAME_FOLD_10KM:
            continue
        pts = np.vstack([xy[excl[q]], xy[q]])
        w = np.ptp(pts[:, 0])
        h = np.ptp(pts[:, 1])
        if max(w, h) / max(min(w, h), 1.0) > MAX_ELONGATION:
            continue
        rows.append(
            (
                -len(np.unique(fold[excl[q]])),
                -(far_e[q] - far_r[q]),
                str(mon["canonical_id"][q]),
                int(q),
            )
        )
    if not rows:
        raise ValueError(
            f"no monitoring well in the {CFG['name']} footprint met the map rule"
        )
    return sorted(rows)[0][3]


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def panel_map(ax, mon, xy, fold, excl, q, h12, pad_frac: float = 0.14) -> dict:
    """The neighbourhood of one well: used pool, excluded same-fold, context."""
    used = excl[q]
    f_t = int(fold[q])

    pts = np.vstack([xy[used], xy[q]])
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    half = 0.5 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * (1.0 + 2 * pad_frac)
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half

    inbox = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    is_used = np.zeros(len(xy), bool)
    is_used[used] = True
    same = inbox & (fold == f_t)
    same[q] = False
    other = inbox & ~is_used & ~same
    other[q] = False

    win = h12.cx[x0:x1, y0:y1]
    win.boundary.plot(ax=ax, color="#c8c8c8", linewidth=0.2, zorder=1)
    target_unit = str(mon["cv_unit"][q])
    h12[h12["huc12"] == target_unit].boundary.plot(
        ax=ax, color="black", linewidth=0.7, zorder=2
    )

    ax.plot(
        xy[other, 0],
        xy[other, 1],
        marker="o",
        ls="none",
        ms=2.2,
        mfc="none",
        mec="#8c8c8c",
        mew=0.4,
        zorder=3,
    )
    ax.plot(
        xy[same, 0],
        xy[same, 1],
        marker="x",
        ls="none",
        ms=3.0,
        mec=FOLD_COLORS[f_t],
        mew=0.7,
        zorder=4,
    )
    folds_used = sorted({int(v) for v in fold[used]})
    for f in folds_used:
        sel = used[fold[used] == f]
        ax.plot(
            xy[sel, 0],
            xy[sel, 1],
            marker="o",
            ls="none",
            ms=3.2,
            mfc=FOLD_COLORS[f],
            mec="black",
            mew=0.3,
            zorder=5,
        )
    ax.plot(
        xy[q, 0],
        xy[q, 1],
        marker="*",
        ls="none",
        ms=8.0,
        mfc=FOLD_COLORS[f_t],
        mec="black",
        mew=0.6,
        zorder=7,
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    # bar length follows the window: the neighbourhood spans tens of km in New
    # Mexico and a few km in Nevada, so a fixed bar would read as a dot or a rule
    scale_bar(ax, length_m=round_bar_length_m(ax), n_seg=2, x_frac=0.04, y_frac=0.035)
    north_arrow(ax, x_frac=0.94, y_frac=0.05)

    handles = (
        [
            Line2D(
                [],
                [],
                marker="*",
                ls="none",
                ms=6.0,
                mfc=FOLD_COLORS[f_t],
                mec="black",
                mew=0.5,
                label=f"target well, fold {f_t}",
            ),
            Line2D(
                [],
                [],
                marker="x",
                ls="none",
                ms=3.4,
                mec=FOLD_COLORS[f_t],
                mew=0.7,
                label=f"fold {f_t}, excluded from the pool",
            ),
        ]
        + [
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=3.2,
                mfc=FOLD_COLORS[f],
                mec="black",
                mew=0.3,
                label=f"R neighbour, fold {f}",
            )
            for f in folds_used
        ]
        + [
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=2.4,
                mfc="none",
                mec="#8c8c8c",
                mew=0.4,
                label="other monitoring well",
            )
        ]
    )
    ax.legend(handles=handles, loc="upper left")

    return {
        "extent": (x0, x1, y0, y1),
        "n_same_in_box": int(same.sum()),
        "n_other_in_box": int(other.sum()),
        "folds_used": folds_used,
        "used": used,
    }


def panel_locator(ax, extent, state) -> None:
    """Framed locator: the state with the panel-(a) extent boxed (guide §11)."""
    x0, x1, y0, y1 = extent
    state.plot(ax=ax, facecolor="#f2f2f2", edgecolor="black", linewidth=0.4)
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="black",
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def _nice_ceiling(v: float) -> float:
    """Smallest 1/2/3/5 x 10^k at or above ``v`` — a round axis end, no dead decade."""
    k = np.floor(np.log10(v))
    for m in (1.0, 2.0, 3.0, 5.0, 10.0):
        if m * 10**k >= v:
            return float(m * 10**k)
    raise ValueError(f"no round ceiling for {v}")


def panel_ecdf(ax, far_e, far_r) -> None:
    """ECDF of the planar distance out to the 32nd R-neighbour, both pools."""
    for vals, style, lab in (
        (far_e, EXCLUDED_STYLE, "own fold excluded"),
        (far_r, RETAINED_STYLE, "own fold retained"),
    ):
        s = np.sort(vals)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, y, drawstyle="steps-post", **style, label=lab)

    lo = 10 ** np.floor(np.log10(min(far_e.min(), far_r.min())))
    hi = _nice_ceiling(max(far_e.max(), far_r.max()))
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.0)
    ticks = 10.0 ** np.arange(np.log10(lo), np.log10(hi))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Planar distance to the 32nd R-neighbour (km)")
    ax.set_ylabel("Fraction of wells")
    ax.legend(loc="upper left")


# --------------------------------------------------------------------------
# Caption
# --------------------------------------------------------------------------


def _caption(par, mon, q, info, far_e, far_r, st, n_basins, rep, others) -> str:
    f_t = int(mon["cv_fold"][q])
    n_folds = int(mon["cv_fold"].nunique())
    used_fold = mon["cv_fold"].to_numpy()[info["used"]]
    folds = ", ".join(
        f"{int((used_fold == f).sum())} from fold {f}" for f in info["folds_used"]
    )
    me, pe = np.median(far_e[st]), np.percentile(far_e[st], 90)
    mr, pr = np.median(far_r[st]), np.percentile(far_r[st], 90)
    d_used = np.hypot(
        mon["x5070"].to_numpy()[info["used"]] - mon["x5070"][q],
        mon["y5070"].to_numpy()[info["used"]] - mon["y5070"][q],
    )
    ar = par["appended_r"]
    n_app = sum(n for c, n in others.items() if c != "water_pseudo")
    max_dev = rep["max_dev_m"]
    return (
        "Figure 2. Does a held-out well's own fold ever contribute to the base R "
        "it is corrected against? No: the base R at a fold-f monitoring well is "
        f"a relief-weighted inverse-distance average of the k = {par['k']} "
        f"nearest monitoring wells drawn from the other {n_folds - 1} folds; "
        "the appended state-agency wells and the water-stage pseudo-rows are "
        "never in the pool, and the appended wells read that same archived "
        "field rather than a cross-fit of their own.\n\n"
        f"(a) One monitoring well ({mon['canonical_id'][q]}, HUC12 "
        f"{mon['cv_unit'][q]}, fold {f_t}) inside the {n_basins}-basin "
        f"{CFG['name']} render footprint, with the wells that actually set its "
        f"R. Filled circles are the {par['k']} pool wells used, coloured by "
        f"their cross-validation fold ({folds}; the fold colours are Fig. 1's); "
        "they lie "
        f"{d_used.min() / 1000:.1f}-{d_used.max() / 1000:.1f} km away. Crosses "
        f"are the {info['n_same_in_box']} other fold-{f_t} monitoring wells in "
        "view, every one of them excluded from the pool because they share the "
        f"target's fold; all {par['k']} of the target's nearest neighbours in "
        "the lifted metric are fold-"
        f"{f_t} wells, so the exclusion replaces the neighbour set outright and "
        f"pushes the farthest used neighbour from {far_r[q]:.1f} km to "
        f"{far_e[q]:.1f} km. Open grey circles are the {info['n_other_in_box']} "
        "other monitoring wells in view: other-fold wells that the lifted kNN "
        "did not rank inside the nearest 32, mostly because their land surface "
        "sits at a different elevation. Hairlines are HUC12 boundaries and the "
        "black outline is the target's own HUC12 — folds are dealt to whole "
        "HUC12s, so the whole unit goes out together. The star is the target "
        f"well. The map extent is boxed on the {CFG['name']} locator inset.\n\n"
        f"(b) Empirical cumulative distribution, over all {int(st.sum()):,} "
        f"monitoring wells in the {CFG['name']} footprint, of the planar "
        "distance from each well to the farthest (32nd) of its R-neighbours. "
        f"Solid: own fold excluded, the actual construction (median {me:.1f} km, "
        f"90th percentile {pe:.1f} km). Dashed: own fold retained, the "
        "counterfactual in which only the well itself is dropped (median "
        f"{mr:.1f} km, 90th percentile {pr:.1f} km). Both neighbour sets are "
        "found in the same lifted metric; the plotted distance is the plain "
        "planar one, so it reads as ground distance.\n\n"
        "Construction. Neighbours are the k nearest monitoring wells in the "
        f"relief-lifted coordinate (x, y, vw × z_surf) with vw = {par['vw']:g} "
        f"and k = {par['k']}, weighted by 1/max(d, 1 m) to the power "
        f"{par['power']:g} in that same lifted distance; the interpolated "
        "quantity is observed water-table elevation. The pool is the "
        f"{len(mon):,} monitoring wells only — the other bundle label rows "
        f"({n_app:,} appended state-agency wells; "
        f"{others.get('water_pseudo', 0):,} water-stage "
        "pseudo-rows) are labels, never prior sources — and it is split "
        "leave-one-fold-out on the bundle's cross-validation folds "
        f"({par['cv_scheme']}). The archived value is then capped at ground "
        f"minus {par['cap_m']:g} m: where the cross-fit average sat above "
        f"that, it is lowered to it, which binds at {rep['n_bound']:,} of the "
        f"{len(mon):,} monitoring wells (median lowering "
        f"{rep['median_lowering_m']:.1f} m), and the same cap is applied to the "
        "rendered R. Rebuilding R from the neighbour sets drawn here and "
        "applying that cap reproduces the archived regional_wte_idw_oof_m "
        "column exactly (maximum absolute deviation "
        f"{'0' if max_dev == 0 else f'{max_dev:.1e}'} m over all {len(mon):,} "
        f"monitoring wells). The {ar['n']:,} appended state-agency label rows "
        "carry no cross-fit of their own: their R is the lattice read used at "
        "render time, the same relief-lifted kNN inverse-distance average "
        f"(k = {ar['k']}, power {ar['power']:g}, vw = {ar['relief_vw']:g}) of "
        f"the archived values at the {ar['pool_n']:,} monitoring wells, so an "
        "appended well never contributes to its own R either. Coordinates and "
        "all distances are in the Albers equal-area projection (EPSG:5070); "
        "hydrologic units are from the WBD and the state boundary from the "
        "2022 US Census cartographic boundary file. Fold identity is nominal "
        "and is the one encoding in this figure that does not survive "
        "grayscale.\n\n"
        "At render time the same property carries to the map: R at a lattice "
        "pixel is this same relief-lifted kNN inverse-distance average of the "
        f"ARCHIVED leave-fold-out R values at the {ar['pool_n']:,} monitoring "
        "wells, never a well's raw observation (utils/infer_conus_gnn.py, "
        "--r-source crossfit); the appended state-agency sources are excluded "
        "from that pool by name and "
        f"the ground cap of {par['cap_m']:g} m is applied to the rendered "
        "field.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--well",
        default=None,
        help="canonical_id of the well mapped in panel (a); the default is the "
        "deterministic rule in choose_target()",
    )
    args = ap.parse_args()

    import geopandas as gpd

    use_style()

    par = load_manifest()
    mon, others = load_monitoring()
    if len(mon) != int(par["appended_r"]["pool_n"]):
        raise ValueError(
            f"{len(mon):,} monitoring wells but the manifest's lattice-R pool is "
            f"{par['appended_r']['pool_n']:,}"
        )
    xy = mon[["x5070", "y5070"]].to_numpy("float64")
    fold = mon["cv_fold"].to_numpy().astype(int)
    coords = relief_coords(mon["x5070"], mon["y5070"], mon["z_surf_well_m"], par["vw"])
    excl, ret = neighbour_sets(coords, fold, par["k"])

    rep = verify_reproduction(mon, excl, par)
    if rep["max_dev_m"] > 1e-6:
        raise ValueError(
            f"rebuilt R deviates from the archived column by {rep['max_dev_m']:.3e} "
            "m; the neighbour sets here are not the ones the bundle used"
        )

    basins = rendered_basins()
    in_state = mon["huc8"].isin(basins).to_numpy()
    far_e = planar_far_km(xy, excl)
    far_r = planar_far_km(xy, ret)

    if args.well is None:
        q = choose_target(mon, xy, fold, excl, ret, in_state)
    else:
        hit = np.where(mon["canonical_id"].to_numpy() == args.well)[0]
        if len(hit) != 1:
            raise ValueError(f"{args.well!r} matched {len(hit)} monitoring wells")
        q = int(hit[0])

    h12 = gpd.read_parquet(HUC12_POLYS, columns=["huc12", "geometry"])
    h12["huc12"] = h12["huc12"].astype(str).str.zfill(12)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_2COL, 86 * MM),
        gridspec_kw={"width_ratios": [1.0, 0.92], "wspace": 0.16},
    )
    info = panel_map(axes[0], mon, xy, fold, excl, q, h12)
    loc = axes[0].inset_axes([0.70, 0.695, 0.29, 0.29])
    panel_locator(loc, info["extent"], state_outline())
    panel_ecdf(axes[1], far_e[in_state], far_r[in_state])
    outside_letter(axes[0], "a")
    outside_letter(axes[1], "b", x=-0.13)

    stem = "fig02_crossfit_r"
    paths = save(fig, stem, FIG_DIR)
    plt.close(fig)

    cap = _caption(par, mon, q, info, far_e, far_r, in_state, len(basins), rep, others)
    cap_path = FIG_DIR / f"{stem}_caption.txt"
    cap_path.write_text(cap)

    print(
        f"target {mon['canonical_id'][q]}  HUC12 {mon['cv_unit'][q]}  "
        f"fold {int(fold[q])}  32nd nbr {far_r[q]:.2f} -> {far_e[q]:.2f} km"
    )
    print(
        f"{CFG['name']} monitoring wells: {int(in_state.sum()):,}  |R rebuild dev| "
        f"{rep['max_dev_m']:.1e} m  cap binds {rep['n_bound']:,}"
    )
    for ext, p in paths.items():
        print(f"{ext}: {p}")
    print(f"caption: {cap_path}")


if __name__ == "__main__":
    main()
