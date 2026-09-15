"""Toy analogues of the production prior stack (plan section 1.4).

Run as::

    uv run python -m utils.figures.toy_priors     # build all priors, print a summary

Every function returns a 2-D array on the world grid (shape ``(NY, NX)``) unless
its docstring says otherwise. Elevation fields are water-table *elevations* in m;
depth fields are depths below land surface in m.

Water-table wells only. Confined and likely-confined wells measure a
potentiometric surface, not the water table, so ``is_confined_flag`` wells are
excluded from every prior here (project rule; they appear in one diagnostic panel
only).

Two unrelated things called IDW live in this module, as in the real code base:

* **relief IDW** -- :func:`relief_idw`, k-NN inverse-distance interpolation of
  well water-table *elevations* in an (x, y, 100 z) metric. This is the R
  builder.
* **strip-fill** -- the lateral distance-weighted blend of the FAC water surface
  onto the drainage surface inside :func:`fac_rem`, a local channel-strip
  operation.

:func:`fac_rem` is a HAND-based *analog* of ``src/handily/rem_fac.py``, not that
algorithm: it takes height above the *Euclidean* nearest drainage, subtracts a
channel depth, and blends the result onto the isotropic drainage surface near
channels. It reproduces the recipe's structure and its failure mode, not its
numbers. Nothing in it follows a D8 path or a grid column: both stamp streaks
into the depth field and into its error map.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.figures.fig_common import NX, NY  # noqa: E402
from utils.figures.toy_world import HOLDOUT_REGION, N_FOLDS, load_world  # noqa: E402

Z_SCALE = 100.0


def water_table_wells(wells: pd.DataFrame) -> pd.DataFrame:
    """Rows that measure the unconfined water table (drops the confined flag)."""
    return wells.loc[~wells["is_confined_flag"].to_numpy(bool)]


def training_pool(wells: pd.DataFrame, exclude_fold: int | None = None) -> pd.DataFrame:
    """Water-table wells in folds 0..N_FOLDS-1, optionally dropping one fold.

    The buffered-holdout region (fold ``HOLDOUT_REGION``) is never in the pool.
    """
    w = water_table_wells(wells)
    w = w.loc[w["fold"].to_numpy(int) < N_FOLDS]
    if exclude_fold is not None:
        w = w.loc[w["fold"].to_numpy(int) != int(exclude_fold)]
    return w


# --------------------------------------------------------------------------
# Relief-lifted inverse-distance interpolation
# --------------------------------------------------------------------------


def relief_idw(
    x, y, z, values, grid, k: int = 16, p: float = 2.0, z_scale: float = Z_SCALE
):
    """k-NN inverse-distance interpolation in the (x, y, ``z_scale`` * elevation) metric.

    Parameters
    ----------
    x, y : array
        Point coordinates in metres.
    z : array
        Point land-surface elevation in metres (the relief lift).
    values : array
        The quantity to interpolate, one per point (usually well WTE in m).
    grid : dict
        The world dict; ``grid["dem"]`` supplies the grid's elevation lift and
        ``grid["X"]``/``grid["Y"]`` its coordinates.
    k, p, z_scale : int, float, float
        Neighbour count, inverse-distance power, and the vertical scaling that
        makes 1 m of relief cost ``z_scale`` m of horizontal separation.

    Returns
    -------
    ndarray
        Shape ``(NY, NX)``. A grid point that coincides exactly with a data point
        in all three metric coordinates takes that point's value exactly, which is
        what makes the inference-R check in :func:`build_R` exact.

    Note
    ----
    Signature deviation from the plan: the plan wrote
    ``relief_idw(x, y, z, grid, ...)``, which has no argument for the quantity
    being interpolated. ``values`` is inserted as the fourth positional argument.
    """
    pts = np.column_stack(
        [np.asarray(x, float), np.asarray(y, float), z_scale * np.asarray(z, float)]
    )
    vals = np.asarray(values, float)
    if pts.shape[0] != vals.shape[0]:
        raise ValueError("values must align with the points")
    kk = int(min(k, pts.shape[0]))
    tree = cKDTree(pts)
    q = np.column_stack(
        [
            grid["X"].ravel(),
            grid["Y"].ravel(),
            z_scale * np.asarray(grid["dem"], float).ravel(),
        ]
    )
    d, idx = tree.query(q, k=kk)
    if kk == 1:
        d, idx = d[:, None], idx[:, None]
    w = 1.0 / np.maximum(d, 1e-9) ** p
    exact = d < 1e-9
    any_exact = exact.any(axis=1)
    out = (w * vals[idx]).sum(axis=1) / w.sum(axis=1)
    if any_exact.any():
        first = np.argmax(exact, axis=1)
        out[any_exact] = vals[idx[any_exact, first[any_exact]]]
    return out.reshape(NY, NX)


def _idw_planar(
    x, y, values, grid, k: int = 16, p: float = 2.0, exclusion_m: float = 0.0
):
    """Plain (x, y) k-NN IDW, optionally ignoring neighbours closer than ``exclusion_m``."""
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    vals = np.asarray(values, float)
    tree = cKDTree(pts)
    kk = (
        int(min(k + 12, pts.shape[0])) if exclusion_m > 0 else int(min(k, pts.shape[0]))
    )
    q = np.column_stack([grid["X"].ravel(), grid["Y"].ravel()])
    d, idx = tree.query(q, k=kk)
    if kk == 1:
        d, idx = d[:, None], idx[:, None]
    if exclusion_m > 0:
        ok = d >= exclusion_m
        order = np.argsort(~ok, axis=1, kind="stable")
        d = np.take_along_axis(d, order, axis=1)[:, :k]
        idx = np.take_along_axis(idx, order, axis=1)[:, :k]
        okk = np.take_along_axis(ok, order, axis=1)[:, :k]
        if not okk.all():
            n_short = int((~okk).any(axis=1).sum())
            raise ValueError(
                f"{n_short} grid cells have fewer than k={k} neighbours beyond "
                f"the {exclusion_m:g} m exclusion radius; widen the candidate pool"
            )
    w = 1.0 / np.maximum(d, 1e-9) ** p
    out = (w * vals[idx]).sum(axis=1) / w.sum(axis=1)
    return out.reshape(NY, NX)


# --------------------------------------------------------------------------
# R: the regional base, in three constructions
# --------------------------------------------------------------------------


def build_R(wells: pd.DataFrame, world: dict, mode: str, k: int = 16, p: float = 2.0):
    """Build the regional base R (a water-table *elevation* field).

    ``mode``:

    ``"leaky"``
        One field from every water-table well in folds 0..5. It threads every
        observation; this is the surface figure F3(a) shows as the failure case.
    ``"crossfit"``
        Returns ``{"fields": {fold: field}, "well_values": Series}``. ``fields[k]``
        is fitted with fold ``k`` held out; ``fields[HOLDOUT_REGION]`` is fitted on
        all six folds and is what the buffered-holdout wells are scored against.
        ``well_values`` is the cross-fit value at each water-table well (indexed
        like ``wells``): for a fold-k well, ``fields[k]`` at that well.
    ``"inference"``
        ``relief_idw`` of the archived per-well cross-fit *values*. Because wells
        sit on cell centres, the returned field equals ``well_values`` exactly at
        every well -- the property that makes the map the model trained against
        the same surface it is scored on.

    Returns a 2-D field for ``"leaky"`` and ``"inference"``, a dict for
    ``"crossfit"``.
    """
    if mode == "leaky":
        pool = training_pool(wells)
        return relief_idw(pool.x, pool.y, pool.dem, pool.wte_obs, world, k=k, p=p)

    if mode in ("crossfit", "inference"):
        fields: dict[int, np.ndarray] = {}
        for fold in range(N_FOLDS):
            pool = training_pool(wells, exclude_fold=fold)
            fields[fold] = relief_idw(
                pool.x, pool.y, pool.dem, pool.wte_obs, world, k=k, p=p
            )
        pool = training_pool(wells)
        fields[HOLDOUT_REGION] = relief_idw(
            pool.x, pool.y, pool.dem, pool.wte_obs, world, k=k, p=p
        )

        wt = water_table_wells(wells)
        rows = wt["row"].to_numpy(int)
        cols = wt["col"].to_numpy(int)
        folds = wt["fold"].to_numpy(int)
        vals = np.empty(len(wt))
        for fold, field in fields.items():
            m = folds == fold
            if m.any():
                vals[m] = field[rows[m], cols[m]]
        well_values = pd.Series(vals, index=wt.index, name="r_crossfit")
        if mode == "crossfit":
            return {"fields": fields, "well_values": well_values}

        field = relief_idw(wt.x, wt.y, wt.dem, well_values.to_numpy(), world, k=k, p=p)
        return field

    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------
# FAC-REM analog
# --------------------------------------------------------------------------


def fac_rem(
    world: dict,
    channel_depth: float = 0.0,
    strip_radius_m: float = 1000.0,
    max_depth: float = 90.0,
):
    """HAND-based water surface and depth (an analog of ``rem_fac.py``).

    ``water_surface = DEM - (HAND + channel_depth)``. ``channel_depth`` defaults
    to 0 m -- the production convention of burning the channel to zero depth to
    water -- so the analog's near-channel error is exactly the stream stage it
    does not know about (~1 m here), not an imposed offset. Floored at
    ``DEM - max_depth`` for regional sanity, then strip-filled: the channel
    strip is blended onto the drainage surface itself (``world["stream_elev"]``,
    the isotropically smoothed elevation of the Euclidean nearest stream cell),
    with a weight that decays over ``strip_radius_m`` from the network. This is
    the lateral cross-section fill of the real pipeline, not a blur of the whole
    field -- blurring pulls hillslope values into the channel strip and wrecks
    the near-field, which is the prior's whole reason for existing.

    The fill deliberately does **not** interpolate from the k nearest *stream
    cells*. Along a straight reach those k cells all lie on one line, so the fill
    is constant along lines parallel to it and the field comes out streaked; the
    drainage surface is isotropic by construction and carries no such grain.

    Returns ``(water_surface, depth)``, both ``(NY, NX)`` in m.
    """
    dem = np.asarray(world["dem"], float)
    hand = np.asarray(world["hand"], float)
    dist = np.asarray(world["dist_to_stream"], float)
    fill = np.asarray(world["stream_elev"], float)
    raw_depth = np.minimum(hand + channel_depth, max_depth)
    ws = dem - raw_depth

    w = np.exp(-dist / strip_radius_m)
    ws = w * (fill - channel_depth) + (1.0 - w) * ws
    return ws, dem - ws


# --------------------------------------------------------------------------
# The rest of the stack
# --------------------------------------------------------------------------


def deep_prior(wells: pd.DataFrame, world: dict, fold: int | None = None, k: int = 16):
    """The deep regional expert: ``DEM - IDW(deepest-quartile wells' DTW)``.

    The quartile threshold is computed *inside* the training pool, so holding out
    a fold changes which wells count as deep -- this is the fold-purity fix from
    the production deep prior.

    It interpolates the deep wells' **depth** and hangs it off the local land
    surface, rather than interpolating their water-table *elevations* directly.
    Across 900 m of relief a plain elevation IDW of a handful of deep upland wells
    is unusable (MAD ~77 m in the uplands, ~156 m in the valley here) and the gate
    correctly refuses to touch it -- which leaves the mixture with no deep-regime
    expert at all. Interpolating the depth keeps the expert regional while letting
    the terrain carry the elevation, and is what the production deep prior does.
    """
    pool = training_pool(wells, exclude_fold=fold)
    thr = np.quantile(pool["dtw_obs"].to_numpy(float), 0.75)
    deep = pool.loc[pool["dtw_obs"].to_numpy(float) >= thr]
    depth = relief_idw(
        deep.x, deep.y, deep.dem, deep.dtw_obs, world, k=min(k, len(deep))
    )
    return np.asarray(world["dem"], float) - depth


def mirror_prior(world: dict, depth: float = 3.0):
    """The constant-depth mirror expert: a water-table elevation ``DEM - depth``."""
    return np.asarray(world["dem"], float) - float(depth)


def drilled_depth_prior(
    wells: pd.DataFrame, world: dict, exclusion_m: float = 100.0, k: int = 16
):
    """Behavioural drilled-depth features (both *depths* in m, not elevations).

    Returns ``(dd_idw, dd_p90)``: the IDW of drilled depth and the 90th percentile
    of the k neighbours' drilled depth, both computed with a self-exclusion radius
    so a well's own borehole never predicts its own water level.
    """
    pool = training_pool(wells)
    x = pool["x"].to_numpy(float)
    y = pool["y"].to_numpy(float)
    dd = pool["drilled_depth"].to_numpy(float)
    idw = _idw_planar(x, y, dd, world, k=k, exclusion_m=exclusion_m)

    tree = cKDTree(np.column_stack([x, y]))
    q = np.column_stack([world["X"].ravel(), world["Y"].ravel()])
    d, idx = tree.query(q, k=min(k + 12, len(pool)))
    ok = d >= exclusion_m
    order = np.argsort(~ok, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)[:, :k]
    p90 = np.percentile(dd[idx], 90.0, axis=1).reshape(NY, NX)
    return idw, p90


def water_pseudo_obs(
    world: dict, weight: float = 0.25, stride: int = 2
) -> pd.DataFrame:
    """Permanent-water rows: wet cells carried as DTW ~ 0 observations.

    Returns a DataFrame (not a grid) with ``x``, ``y``, ``dem``, ``wte_obs``,
    ``dtw_obs`` and ``weight``, one row per ``stride``-th wet cell. ``dtw_obs`` is
    0 m by construction, so ``wte_obs`` is the land surface.
    """
    wet = np.asarray(world["wet"], bool)
    j, i = np.nonzero(wet)
    j, i = j[::stride], i[::stride]
    dem = np.asarray(world["dem"], float)[j, i]
    return pd.DataFrame(
        {
            "row": j,
            "col": i,
            "x": world["xs"][i],
            "y": world["ys"][j],
            "dem": dem,
            "wte_obs": dem,
            "dtw_obs": np.zeros(j.size),
            "weight": np.full(j.size, float(weight)),
        }
    )


def ma_like(wells: pd.DataFrame, world: dict, k: int = 16):
    """The Ma-like benchmark: plain (x, y) IDW of observed DTW, in *depth* space.

    Returns a depth field in m. Interpolating depth rather than elevation is the
    whole point of the comparator: it inherits none of the terrain and invents
    artifacts where the land surface moves and the wells do not.
    """
    pool = training_pool(wells)
    return _idw_planar(pool.x, pool.y, pool.dtw_obs, world, k=k)


def residual_hist_data(
    wells: pd.DataFrame, R_leaky: np.ndarray, R_crossfit_values: pd.Series
):
    """Residual arrays for figure F3's two histograms, in m.

    Returns ``(resid_leaky, resid_crossfit)``: observed WTE minus the leaky field
    at the well, and observed WTE minus the archived cross-fit value.
    """
    wt = water_table_wells(wells)
    wt = wt.loc[wt["fold"].to_numpy(int) < N_FOLDS]
    obs = wt["wte_obs"].to_numpy(float)
    leaky = np.asarray(R_leaky)[wt["row"].to_numpy(int), wt["col"].to_numpy(int)]
    cf = R_crossfit_values.loc[wt.index].to_numpy(float)
    return obs - leaky, obs - cf


def build_all(wells: pd.DataFrame, world: dict) -> dict:
    """Build the whole prior stack once. Keys used by :mod:`utils.figures.toy_model`.

    ``r_leaky``, ``r_inference`` (fields); ``r_crossfit`` (dict of fields);
    ``r_well_values`` (Series); ``fac_ws``/``fac_depth``; ``deep`` (dict
    fold->field, plus ``HOLDOUT_REGION``); ``mirror``; ``dd_idw``/``dd_p90``;
    ``ma_dtw``; ``pseudo`` (DataFrame).
    """
    cf = build_R(wells, world, "crossfit")
    fac_ws, fac_depth = fac_rem(world)
    deep = {f: deep_prior(wells, world, fold=f) for f in range(N_FOLDS)}
    deep[HOLDOUT_REGION] = deep_prior(wells, world, fold=None)
    dd_idw, dd_p90 = drilled_depth_prior(wells, world)
    return {
        "r_leaky": build_R(wells, world, "leaky"),
        "r_crossfit": cf["fields"],
        "r_well_values": cf["well_values"],
        "r_inference": build_R(wells, world, "inference"),
        "fac_ws": fac_ws,
        "fac_depth": fac_depth,
        "deep": deep,
        "mirror": mirror_prior(world),
        "dd_idw": dd_idw,
        "dd_p90": dd_p90,
        "ma_dtw": ma_like(wells, world),
        "pseudo": water_pseudo_obs(world),
    }


def _main() -> None:
    world, wells = load_world()
    pri = build_all(wells, world)
    wt = water_table_wells(wells)
    rows, cols = wt["row"].to_numpy(int), wt["col"].to_numpy(int)

    at = pri["r_inference"][rows, cols]
    diff = np.abs(at - pri["r_well_values"].to_numpy(float))
    print(
        f"inference R vs cross-fit values at wells: max |diff| = {diff.max():.3e} m (n = {len(wt)})"
    )

    rl, rc = residual_hist_data(wells, pri["r_leaky"], pri["r_well_values"])
    print(
        f"R residual std at wells: leaky {rl.std():.3f} m   cross-fit {rc.std():.3f} m"
    )
    print(
        f"R residual MAD at wells: leaky {np.median(np.abs(rl)):.3f} m   cross-fit {np.median(np.abs(rc)):.3f} m"
    )

    truth = world["dtw_true"]
    err = pri["fac_depth"] - truth
    d = world["dist_to_stream"]
    print("FAC-REM |error| by distance to stream (m):")
    for lo, hi in [
        (0, 100),
        (100, 300),
        (300, 600),
        (600, 1000),
        (1000, 2000),
        (2000, 1e9),
    ]:
        m = (d >= lo) & (d < hi)
        print(
            f"  {lo:>5.0f}-{hi:<8.0f} n = {m.sum():>6d}  MAD {np.median(np.abs(err[m])):8.2f}"
        )
    print(
        f"Ma-like DTW range: {pri['ma_dtw'].min():.1f} to {pri['ma_dtw'].max():.1f} m"
    )
    print(
        f"water pseudo-obs rows: {len(pri['pseudo'])} at weight {pri['pseudo'].weight.iloc[0]}"
    )


if __name__ == "__main__":
    _main()
