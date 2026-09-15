"""Synthetic toy world with a known water table (plan sections 1.1-1.3).

Run as::

    uv run python -m utils.figures.toy_world           # build (or rebuild) the cache
    uv run python -m utils.figures.toy_world --force   # ignore an existing cache

Caches to ``/data/ssd2/handily/figures/toy/toy_world.npz`` (rasters) and
``toy_wells.parquet`` (observations). :func:`load_world` returns
``(world: dict, wells: pandas.DataFrame)`` and builds the cache if it is missing.

Grid: 240 x 160 cells of 100 m, origin at (0, 0), x east and y north. Arrays are
indexed ``arr[j, i]`` with ``j`` ascending northward (row 0 = south edge); draw
with ``origin="lower"`` and ``extent=fig_common.EXTENT``.

Terrain: a trunk valley running west to east at about 0.3 % grade, an asymmetric
cross-section (steep 600 m mountain wall to the north, gentler 400 m piedmont to
the south), one tributary from the north with a low alluvial fan at its mouth, a
closed playa sub-basin in the east. Upland relief is a two-scale isotropic
Gaussian random field so the block reads as rolling ridges. On top of it sits
band-limited **microtopography** -- three isotropic Gaussian random fields at
about 300 m, 1 km and 3 km correlation length, 5.7 m standard deviation on the
valley floor, scaled up with the local slope to ~18 m on the mountain wall. It
is added before pit filling and D8 routing, and it is what makes the network
dendritic: on a smooth 600 m wall D8 has nothing to deflect it and every column
drains straight downhill as a parallel one-cell rill. The trunk and tributary
channels are narrow trenches (sigma 200 m and 320 m) on meandering axes, for the
same reason -- a wide trench has a flat floor and the thalweg wanders inside it
at the noise's own wavelength.

Hydrology: pits are filled by a priority flood seeded from the domain edge *and*
from the playa sink, so the playa stays endorheic. **D8 is used for two things
only: flow accumulation and the stream mask.** Everything else is Euclidean:
``dist_to_stream`` and ``stream_elev`` come from a single
:func:`scipy.ndimage.distance_transform_edt` with ``return_indices=True``.
``stream_elev`` is the drainage surface -- the elevation of the Euclidean nearest
stream cell, smoothed isotropically -- and ``hand = max(DEM - stream_elev, 0)``.
The smoothing goes on the drainage surface, not on the height above it: the raw
nearest-stream elevation is piecewise constant and jumps across the Voronoi
boundary between branches, and smoothing after the difference leaves those jumps
in place. A D8 HAND, or a HAND smoothed after differencing, stamps one-cell
streaks into every surface derived from it, and the truth surface must carry no
such artifacts.

True water table (plan section 1.2)::

    WTE_true  = min(w * (DEM - d_shallow) + (1 - w) * WTE_regional, DEM + stage)
    w         = exp(-dist_to_stream / 800 m)
    stage     = 1.5 m * exp(-dist_to_stream / 200 m)
    d_shallow = min(0.85 * HAND + 0.5 m - stage, 90 m)
    WTE_regional = drainage datum + 0.80 * (smoothed DEM - datum) - 24 m + bumps

Three deliberate deviations from the plan's section 1.2 sketch, all forced by
what the figures have to show:

1. ``d_shallow`` is referenced to the **nearest drainage** (through HAND) rather
   than being a 1-4 m offset below the local land surface. With the plan's
   formula the true water table near a channel sits a fixed 1-4 m below ground
   everywhere, so a constant 3 m mirror is a near-perfect prior and HAND-based
   FAC-REM never wins anything -- which would make figures F4, F5 and F6
   meaningless. Referencing the shallow term to the drainage is also the
   physically standard near-stream picture, and it is exactly the premise
   FAC-REM encodes.
2. A channel stage term lifts the table above the bed inside the trench. Without
   it no cell is ever wet (``d_shallow > 0`` implies ``WTE_true < DEM``
   everywhere), and the toy would have no permanent water for the water-flatten
   and spring stories.
3. The regional surface is referenced to a broad drainage datum (the smoothed
   Euclidean nearest-stream elevation), not to a plane. A plane, or a fixed
   depth below the smoothed DEM, stands above ground across every incised
   hollow once the terrain carries realistic roughness.
"""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.figures.fig_common import (  # noqa: E402
    CELL,
    NX,
    NY,
    TOY_DIR,
    TOY_WELLS_PARQUET,
    TOY_WORLD_NPZ,
)

SEED = 20260907
N_WELLS = 400
CONFINED_FRACTION = 0.035
STREAM_THRESHOLD_CELLS = 400  # 4.0 km^2 contributing area
LAMBDA_BLEND = 800.0  # m, terrain-following decay length
REGIONAL_ALPHA = 0.80  # fraction of the terrain's rise above the drainage datum
REGIONAL_BASE = 24.0  # m, regional water table below that datum
N_FOLDS = 6  # folds 0..5; region 6 is the reserved buffered holdout
HOLDOUT_REGION = 6

# Terrain control points (metres)
PLAYA_CENTER = (19800.0, 3300.0)
PLAYA_RADIUS = 2300.0
TRIB_X = 13000.0

_D8_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# --------------------------------------------------------------------------
# Terrain
# --------------------------------------------------------------------------


def _valley_centerline(x: np.ndarray) -> np.ndarray:
    """North coordinate of the trunk *valley* axis at easting ``x`` (m).

    This positions the valley walls. The channel itself follows
    :func:`_trunk_thalweg`, which meanders inside the valley.
    """
    return (
        6000.0
        + 900.0 * np.sin(np.pi * x / 24000.0)
        + 380.0 * np.sin(2 * np.pi * x / 9300.0 + 1.1)
        + 160.0 * np.sin(2 * np.pi * x / 4100.0 + 2.7)
    )


def _trunk_thalweg(x: np.ndarray) -> np.ndarray:
    """North coordinate of the trunk *channel* axis at easting ``x`` (m).

    The valley walls follow :func:`_valley_centerline`; the channel adds two short
    meander harmonics on top of it. Their amplitude-to-wavelength ratios (~0.5 and
    ~0.35) are what keep the trunk off the grid axes: with the valley axis alone
    the thalweg turned less than one cell of northing per four cells of easting,
    so D8 drew it as a long east-running staircase.
    """
    return _valley_centerline(x) + (
        365.0 * np.sin(2 * np.pi * x / 3000.0 + 0.4)
        + 170.0 * np.sin(2 * np.pi * x / 1550.0 + 1.9)
    )


def _build_dem(rng: np.random.Generator):
    xs = (np.arange(NX) + 0.5) * CELL
    ys = (np.arange(NY) + 0.5) * CELL
    X, Y = np.meshgrid(xs, ys)

    z_floor = 1600.0 - 0.003 * X  # 0.3 % downstream grade, west to east
    dy = Y - _valley_centerline(X)

    # A ~2.7 km flat alluvial floor, then a steep mountain wall to the north and a
    # gentle piedmont to the south.
    north = 600.0 * np.clip((dy - 1200.0) / 4000.0, 0.0, 1.0) ** 1.35
    south = 400.0 * np.clip((-dy - 1500.0) / 7000.0, 0.0, 1.0) ** 1.60
    dem = z_floor + north + south

    # Rolling upland relief. A pure ramp drains as hundreds of parallel one-cell
    # rills, because D8 on a smooth slope sends every column straight downhill.
    # Two-scale isotropic Gaussian random fields, tapered onto the mountain block
    # and (more weakly) the piedmont, give ridges and hollows that collect the flow
    # into a dendritic network without imposing a washboard at a fixed wavelength.
    up_n = np.clip((dy - 800.0) / 3500.0, 0.0, 1.0)
    up_s = np.clip((-dy - 1200.0) / 6000.0, 0.0, 1.0)
    upness = np.maximum(up_n, 0.6 * up_s)
    relief = np.zeros_like(dem)
    for sigma_cells, amp in ((13.0, 45.0), (5.0, 15.0)):
        f = ndimage.gaussian_filter(
            rng.standard_normal(X.shape), sigma_cells, mode="nearest"
        )
        f /= f.std()
        relief += amp * f
    dem += relief * upness

    # Trunk channel: a narrow sinuous trench following the meandering thalweg.
    # The trench has to be narrow (sigma 200 m, two cells) relative to the
    # microtopography below: a wide trench has a nearly flat floor, the thalweg
    # wanders inside it at the noise's own wavelength, and D8 draws a staircase
    # instead of a river.
    dt = Y - _trunk_thalweg(X)
    dem -= 34.0 * np.exp(-((dt / 200.0) ** 2))

    # Tributary: a trench from the north edge down to the trunk, deepening as it
    # goes, on an axis that wanders in x for the same reason the trunk's does.
    trib_x_at = (
        TRIB_X
        + 900.0 * np.sin(np.pi * (Y - 6500.0) / 9000.0)
        + 320.0 * np.sin(2 * np.pi * Y / 3400.0 + 0.7)
        + 140.0 * np.sin(2 * np.pi * Y / 1700.0 + 2.3)
    )
    trib_dist = np.abs(X - trib_x_at)
    trib_active = (Y > _valley_centerline(X)) & (Y < 15200.0)
    trib_depth = 55.0 * np.exp(-((trib_dist / 320.0) ** 2))
    trib_taper = np.clip((Y - _valley_centerline(X)) / 1500.0, 0.0, 1.0)
    dem -= np.where(trib_active, trib_depth * trib_taper, 0.0)

    # Alluvial fan at the tributary mouth.
    fan_r = np.hypot(X - TRIB_X, Y - (_valley_centerline(np.array(TRIB_X)) + 900.0))
    dem += 28.0 * np.exp(-((fan_r / 2600.0) ** 2))

    # Closed playa sub-basin in the east: a rim ring with a flat floor inside.
    pr = np.hypot(X - PLAYA_CENTER[0], Y - PLAYA_CENTER[1])
    rim = 120.0 * np.exp(-(((pr - PLAYA_RADIUS) / 1100.0) ** 2))
    bowl = -110.0 * np.exp(-((pr / (0.85 * PLAYA_RADIUS)) ** 4))
    dem = dem + rim + bowl
    # Flatten the innermost bowl into a true playa floor at one elevation.
    cj, ci = np.unravel_index(int(np.argmin(pr)), pr.shape)
    inside = np.clip((0.65 * PLAYA_RADIUS - pr) / (0.15 * PLAYA_RADIUS), 0.0, 1.0)
    dem = (1.0 - inside) * dem + inside * dem[cj, ci]

    # Band-limited microtopography: three isotropic Gaussian random fields at
    # correlation lengths of about 300 m, 1 km and 3 km, summing to a 5.7 m
    # standard deviation on the valley floor (within the 3-8 m band the plan
    # asks for) and scaled up with the local slope, so the mountain wall carries
    # ~18 m and the flat floor stays a plain.
    #
    # This is what makes the network dendritic. Without it D8 on the smooth
    # 600 m wall has nothing to deflect it: every column runs straight downhill
    # and the map fills with parallel one-cell rills meeting the trunk at right
    # angles. The slope scaling is the necessary part -- the cross-slope gradient
    # the noise supplies has to be a usable fraction of the downhill gradient it
    # is competing with, and on the wall that gradient is ~0.3.
    gy0, gx0 = np.gradient(dem, CELL)
    slope0 = np.hypot(gx0, gy0)
    micro = np.zeros_like(dem)
    for sigma_cells, amp in ((1.5, 3.0), (5.0, 4.0), (15.0, 4.0)):
        s = ndimage.gaussian_filter(
            rng.standard_normal(dem.shape), sigma_cells, mode="nearest"
        )
        s /= s.std()
        micro += amp * s
    micro *= 1.0 + 14.0 * slope0
    # The playa floor is a flat lake bed: damp the microtopography inside the bowl.
    micro *= 1.0 - 0.9 * np.exp(-((pr / (0.62 * PLAYA_RADIUS)) ** 6))
    dem = dem + micro

    return xs, ys, X, Y, dem


# --------------------------------------------------------------------------
# Hydrology: priority-flood fill, D8, accumulation, HAND
# --------------------------------------------------------------------------


def _priority_flood(
    dem: np.ndarray, extra_seeds: list[tuple[int, int]], eps: float = 1e-3
):
    """Fill depressions (Barnes priority flood) seeded from the domain edge.

    ``extra_seeds`` are ``(j, i)`` cells kept at their own elevation so the
    depression they sit in is *not* filled -- this is how the playa stays a
    closed basin.
    """
    ny, nx = dem.shape
    filled = np.full(dem.shape, np.inf)
    closed = np.zeros(dem.shape, dtype=bool)
    heap: list[tuple[float, int, int]] = []

    edge = []
    edge += [(0, i) for i in range(nx)] + [(ny - 1, i) for i in range(nx)]
    edge += [(j, 0) for j in range(1, ny - 1)] + [(j, nx - 1) for j in range(1, ny - 1)]
    for j, i in edge + list(extra_seeds):
        if not closed[j, i]:
            closed[j, i] = True
            filled[j, i] = dem[j, i]
            heapq.heappush(heap, (float(dem[j, i]), j, i))

    while heap:
        z, j, i = heapq.heappop(heap)
        for dj, di in _D8_OFFSETS:
            jj, ii = j + dj, i + di
            if 0 <= jj < ny and 0 <= ii < nx and not closed[jj, ii]:
                closed[jj, ii] = True
                zz = dem[jj, ii] if dem[jj, ii] > z else z + eps
                filled[jj, ii] = zz
                heapq.heappush(heap, (float(zz), jj, ii))
    return filled


def _d8(filled: np.ndarray):
    """Steepest-descent D8 receiver index (flat array), -1 where terminal."""
    ny, nx = filled.shape
    n = ny * nx
    idx = np.arange(n).reshape(ny, nx)
    best_slope = np.zeros((ny, nx))
    receiver = np.full((ny, nx), -1, dtype=np.int64)
    for dj, di in _D8_OFFSETS:
        length = CELL * np.hypot(dj, di)
        shifted = np.full((ny, nx), np.inf)
        sidx = np.full((ny, nx), -1, dtype=np.int64)
        j0, j1 = max(0, -dj), ny - max(0, dj)
        i0, i1 = max(0, -di), nx - max(0, di)
        shifted[j0:j1, i0:i1] = filled[j0 + dj : j1 + dj, i0 + di : i1 + di]
        sidx[j0:j1, i0:i1] = idx[j0 + dj : j1 + dj, i0 + di : i1 + di]
        slope = (filled - shifted) / length
        take = (slope > best_slope) & np.isfinite(shifted)
        best_slope = np.where(take, slope, best_slope)
        receiver = np.where(take, sidx, receiver)
    return receiver.ravel(), filled.ravel()


def _flow_accumulation(
    receiver: np.ndarray, order_desc: np.ndarray, n: int
) -> np.ndarray:
    """Cell counts draining through each cell, processing upstream cells first."""
    acc = np.ones(n, dtype=np.float64)
    for c in order_desc:
        r = receiver[c]
        if r >= 0:
            acc[r] += acc[c]
    return acc


def _rect_folds(wx, wy, gx, gy, parts: int, cols: int = 3) -> np.ndarray:
    """Rectangular spatially blocked folds with equal well counts.

    Splits the domain into ``cols`` columns at the wells' x quantiles, then each
    column into ``parts // cols`` rows at its wells' y quantiles, so every
    block is contiguous and holds ~the same number of wells. ``wx``/``wy`` are the
    well coordinates that must be balanced (m); ``gx``/``gy`` are the flattened
    grid coordinates to label. Returns a fold index per grid point.
    """
    if parts % cols:
        raise ValueError("parts must be divisible by cols")
    rows = parts // cols
    lab = np.zeros(gx.size, dtype=int)
    xq = np.quantile(wx, np.arange(1, cols) / cols)
    wcol = np.searchsorted(xq, wx, side="right")
    gcol = np.searchsorted(xq, gx, side="right")
    for c in range(cols):
        sel_w, sel_g = wcol == c, gcol == c
        yq = np.quantile(wy[sel_w], np.arange(1, rows) / rows)
        lab[sel_g] = c * rows + np.searchsorted(yq, gy[sel_g], side="right")
    return lab


def _label_from_seeds(seeds: dict[int, int], receiver, order_asc, n):
    """Propagate seed labels upstream along the D8 tree (-1 where unreached)."""
    lab = np.full(n, -1, dtype=np.int32)
    for c, v in seeds.items():
        lab[c] = v
    for c in order_asc:
        if lab[c] >= 0:
            continue
        r = receiver[c]
        if r >= 0:
            lab[c] = lab[r]
    return lab


# --------------------------------------------------------------------------
# True water table
# --------------------------------------------------------------------------


def _wte_true(dem, dist_to_stream, hand, stream_elev, playa_mask, X, Y):
    # Regional aquifer surface: a subdued replica of the landscape, referenced to
    # the broad *drainage* datum rather than to the smoothed land surface. It
    # climbs REGIONAL_ALPHA of the height of the smoothed terrain above that datum
    # and sits REGIONAL_BASE below it, plus two broad bumps, near-flat under the
    # playa. Referencing it to the drainage is what keeps it underground: a
    # surface hung a fixed depth below the smoothed DEM pokes out of every
    # incised hollow, because the terrain's own texture swings +/- 60 m about its
    # smoothed self and the regional depth in the valley is far less than that.
    # The alpha < 1 is also what separates this expert from a HAND prior: at
    # alpha = 0 the regional surface would be exactly the drainage datum and
    # FAC-REM could never lose to it.
    dem_s = ndimage.gaussian_filter(dem, 8.0, mode="nearest")
    drain_s = ndimage.gaussian_filter(stream_elev, 12.0, mode="nearest")
    bumps = 14.0 * np.exp(
        -(((X - 5000.0) / 5000.0) ** 2 + ((Y - 4000.0) / 4500.0) ** 2)
    ) - 10.0 * np.exp(-(((X - 16000.0) / 5500.0) ** 2 + ((Y - 11000.0) / 5000.0) ** 2))
    wte_reg = drain_s + REGIONAL_ALPHA * (dem_s - drain_s) - REGIONAL_BASE + bumps

    # Flatten the regional surface over the playa *floor* only -- the playa
    # sub-basin as a whole reaches far up the piedmont, and imposing the floor
    # elevation there would put the water table hundreds of metres down.
    floor_mask = playa_mask & (
        np.hypot(X - PLAYA_CENTER[0], Y - PLAYA_CENTER[1]) < 0.75 * PLAYA_RADIUS
    )
    playa_floor = float(np.percentile(dem[floor_mask], 8.0))
    flat = playa_floor + 0.5
    pw = ndimage.gaussian_filter(floor_mask.astype(float), 4.0, mode="nearest")
    pw = np.clip(pw / max(pw.max(), 1e-9), 0.0, 1.0)
    wte_reg = (1.0 - pw) * wte_reg + pw * flat

    # Keep the regional surface inside a sane depth band -- but measured against
    # the *smoothed* terrain, never the raw DEM. Clipping against the raw DEM
    # pins the surface to ``dem - cap`` wherever the bound binds and copies the
    # terrain's cell-scale texture straight into what is meant to be a smooth
    # aquifer surface; at the series' vertical exaggeration that reads as a
    # washboard across the whole water table in the 3-D register.
    wte_reg = dem_s - np.clip(dem_s - wte_reg, 2.0, 130.0)

    # Terrain-following term, referenced to the nearest drainage rather than to
    # the local land surface: near a channel the table stands a little above the
    # stream stage and climbs only a fraction of the hillslope's height above the
    # drainage. A channel stage term lifts it above the bed inside the trench so
    # stream cells and the playa floor come out wet (see the module docstring).
    stage = 1.5 * np.exp(-dist_to_stream / 200.0)
    d_shallow = np.minimum(0.85 * hand + 0.5 - stage, 90.0)
    wte_tf = dem - d_shallow

    w = np.exp(-dist_to_stream / LAMBDA_BLEND)
    wte = w * wte_tf + (1.0 - w) * wte_reg
    # The table stands at most the channel stage above the ground. Where this
    # binds the cell is wet by definition and its water surface *is* the ground,
    # so the terrain texture it picks up there is the shoreline, not an artifact.
    wte = np.minimum(wte, dem + stage)
    return wte, wte_reg


# --------------------------------------------------------------------------
# Wells
# --------------------------------------------------------------------------


def _sample_well_cells(rng, dist, X, Y, cluster_x=8000.0):
    """Sample well cells: log-normal in distance to stream, plus a dense cluster.

    Wells sit exactly on cell centres. That is deliberate: it makes the
    cross-fit-value interpolation in :mod:`utils.figures.toy_priors` exact at the
    wells, which is the claim figure F3(c) has to demonstrate.
    """
    ny, nx = dist.shape
    flat_dist = dist.ravel()

    # Log-normal density in distance to stream (median 350 m, sigma 1.0 in log space).
    d = np.maximum(flat_dist, 50.0)
    logw = -((np.log(d) - np.log(350.0)) ** 2) / (2 * 1.0**2) - np.log(d)
    p = np.exp(logw - logw.max())
    p /= p.sum()

    n_cluster = 55
    n_upland = 30
    n_playa = 18
    n_main = N_WELLS - n_cluster - n_upland - n_playa

    chosen = list(rng.choice(flat_dist.size, size=n_main, replace=False, p=p))
    taken = set(chosen)

    def _pick(mask, count):
        cand = np.flatnonzero(mask.ravel())
        cand = np.array([c for c in cand if c not in taken], dtype=np.int64)
        if cand.size == 0:
            raise ValueError("no candidate cells left for this well group")
        wsel = np.exp(-np.maximum(flat_dist[cand], 50.0) / 2500.0)
        wsel /= wsel.sum()
        sel = rng.choice(cand, size=min(count, cand.size), replace=False, p=wsel)
        for c in sel:
            taken.add(int(c))
        return list(sel)

    cluster_mask = (np.abs(X - cluster_x) < 1200.0) & (dist < 2500.0)
    chosen += _pick(cluster_mask, n_cluster)

    upland_mask = dist > 3000.0
    chosen += _pick(upland_mask, n_upland)

    pr = np.hypot(X - PLAYA_CENTER[0], Y - PLAYA_CENTER[1])
    playa_mask = pr < PLAYA_RADIUS * 1.15
    chosen += _pick(playa_mask, n_playa)

    chosen = np.array(sorted(set(int(c) for c in chosen)))
    j = chosen // nx
    i = chosen % nx
    return j, i


def _build_wells(rng, world, j, i):
    dem = world["dem"][j, i]
    wte_true = world["wte_true"][j, i]

    noise = rng.normal(0.0, 1.0, size=j.size)
    dtw_obs = dem - wte_true + noise

    n = j.size
    confined = np.zeros(n, dtype=bool)
    n_conf = max(1, int(round(CONFINED_FRACTION * n)))
    confined[rng.choice(n, size=n_conf, replace=False)] = True
    # Confined wells measure a potentiometric surface standing above the table.
    conf_offset = rng.normal(18.0, 7.0, size=n)
    dtw_obs = np.where(confined, dtw_obs - np.abs(conf_offset), dtw_obs)

    wte_obs = dem - dtw_obs
    drilled = dtw_obs + rng.lognormal(np.log(16.0), 0.55, size=n)
    drilled = np.maximum(drilled, np.maximum(dtw_obs + 2.0, 5.0))

    fold = world["basin_id"][j, i].astype(int)
    holdout = fold == HOLDOUT_REGION

    # A frozen 50/50 split of the water-table wells into contributed "source"
    # observations and validation targets, drawn west-weighted so the eastern
    # third of the domain (piedmont and playa) is a genuine far field with no
    # sources in it -- without that there is nothing for figure F7's far-field
    # claim to be tested against.
    src_rng = np.random.default_rng(0)
    is_source = np.zeros(n, dtype=bool)
    cand = np.flatnonzero(~confined)
    p = np.exp(-np.maximum(world["xs"][i][cand] - 7000.0, 0.0) / 1600.0)
    p /= p.sum()
    is_source[src_rng.choice(cand, size=cand.size // 2, replace=False, p=p)] = True

    return pd.DataFrame(
        {
            "row": j.astype(np.int32),
            "col": i.astype(np.int32),
            "x": world["xs"][i],
            "y": world["ys"][j],
            "dem": dem,
            "wte_true": wte_true,
            "wte_obs": wte_obs,
            "dtw_obs": dtw_obs,
            "dtw_true": dem - wte_true,
            "drilled_depth": drilled,
            "fold": fold,
            "is_confined_flag": confined,
            "is_source": is_source,
            "is_holdout_buffered": holdout,
            "dist_to_stream": world["dist_to_stream"][j, i],
            "hand": world["hand"][j, i],
            "slope": world["slope"][j, i],
            "flow_acc": world["flow_acc"][j, i],
        }
    )


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build_world(verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    """Build the toy world from scratch (does not write the cache)."""
    rng = np.random.default_rng(SEED)
    xs, ys, X, Y, dem = _build_dem(rng)

    pr = np.hypot(X - PLAYA_CENTER[0], Y - PLAYA_CENTER[1])
    playa_core = pr < 0.75 * PLAYA_RADIUS
    sink_flat = int(np.argmin(np.where(playa_core, dem, np.inf)))
    sink_j, sink_i = divmod(sink_flat, NX)

    filled = _priority_flood(dem, [(sink_j, sink_i)])
    receiver, filled_flat = _d8(filled)
    receiver[sink_flat] = -1  # the playa sink is terminal

    n = NX * NY
    order_asc = np.argsort(filled_flat, kind="stable")
    order_desc = order_asc[::-1]

    acc = _flow_accumulation(receiver, order_desc, n)
    flow_acc = acc.reshape(NY, NX)
    streams = flow_acc >= STREAM_THRESHOLD_CELLS

    # Distance to, and elevation of, the *Euclidean* nearest stream cell. HAND is
    # deliberately not computed along the D8 path: a D8 walk assigns whole
    # rectangular blocks of hillslope to one drainage cell and stamps one-cell
    # vertical streaks into every surface derived from it, and those artifacts
    # would then live inside the truth (see the module docstring).
    dist_to_stream, idx = ndimage.distance_transform_edt(
        ~streams, sampling=CELL, return_indices=True
    )
    # ``stream_elev`` is the drainage surface: the elevation of the Euclidean
    # nearest stream cell, smoothed isotropically. The raw nearest-stream
    # elevation is piecewise constant and jumps across the Voronoi boundary
    # between branches; smoothing the *height above it* instead leaves those jumps
    # in place and hands the terrain's own cell-scale texture to HAND, and from
    # HAND to the truth surface, where at the series' vertical exaggeration it
    # reads as a washboard across the whole water table.
    stream_elev = ndimage.gaussian_filter(dem[idx[0], idx[1]], 2.0, mode="nearest")
    hand = np.maximum(dem - stream_elev, 0.0)
    hand[streams] = 0.0

    gy, gx = np.gradient(dem, CELL)
    slope = np.hypot(gx, gy)

    # --- sub-catchment folds -------------------------------------------------
    playa_basin_flat = _label_from_seeds({sink_flat: 0}, receiver, order_asc, n) == 0
    playa_basin = playa_basin_flat.reshape(NY, NX)

    well_j, well_i = _sample_well_cells(rng, dist_to_stream, X, Y)
    well_flat = well_j * NX + well_i
    keep = ~playa_basin_flat[well_flat]

    # Deviation from the plan, deliberate: strict D8 sub-catchments of the mainstem
    # are unusable as folds here. One tributary enters the trunk at a single cell
    # and carries ~30 % of the wells, and the trunk's own upstream walk climbs that
    # tributary, so both sub-catchment folds and along-mainstem bands leave folds of
    # single-digit n -- too small to score. Instead the non-playa domain is cut into
    # six rectangular blocks with equal well counts (a two-by-three recursive median
    # split on the well positions). That is standard spatially blocked CV: blocks are
    # contiguous, no fold shares a neighbourhood with another, and every fold is
    # scoreable. The playa sub-basin is still a true D8 basin and is held out whole.
    basin_flat = _rect_folds(
        X.ravel()[well_flat][keep],
        Y.ravel()[well_flat][keep],
        X.ravel(),
        Y.ravel(),
        N_FOLDS,
    ).astype(np.int16)
    basin_flat[playa_basin_flat] = HOLDOUT_REGION
    n_unlabelled = 0
    basin_id = basin_flat.reshape(NY, NX)

    wte_true, wte_regional = _wte_true(
        dem, dist_to_stream, hand, stream_elev, playa_basin, X, Y
    )
    dtw_true = dem - wte_true
    wet = wte_true >= dem

    world = {
        "xs": xs,
        "ys": ys,
        "X": X,
        "Y": Y,
        "dem": dem,
        "dem_filled": filled,
        "wte_true": wte_true,
        "wte_regional": wte_regional,
        "dtw_true": dtw_true,
        "wet": wet,
        "flow_acc": flow_acc,
        "streams": streams,
        "dist_to_stream": dist_to_stream,
        "hand": hand,
        "stream_elev": stream_elev,
        "slope": slope,
        "basin_id": basin_id,
        "playa_basin": playa_basin,
        "receiver": receiver.reshape(NY, NX),
    }

    wells = _build_wells(rng, world, well_j, well_i)

    if verbose:
        _report(world, wells, n_unlabelled)
    return world, wells


def _report(world, wells, n_unlabelled):
    d = world["dtw_true"]
    st = world["streams"]
    up = world["dist_to_stream"] > 3000
    print(f"streams: {st.sum()} cells ({100 * st.mean():.1f} %)")
    print(f"wet:     {world['wet'].sum()} cells ({100 * world['wet'].mean():.2f} %)")
    print(f"  wet on streams: {100 * world['wet'][st].mean():.0f} % of stream cells")
    print(
        f"  wet off streams within 200 m: {100 * world['wet'][(~st) & (world['dist_to_stream'] <= 200)].mean():.0f} %"
    )
    print(f"DTW_true: min {d.min():.1f} med {np.median(d):.1f} max {d.max():.1f} m")
    print(f"  on streams: median {np.median(d[st]):.2f} m")
    print(
        f"  uplands (>3 km from stream): p10 {np.percentile(d[up], 10):.0f} p90 {np.percentile(d[up], 90):.0f} m"
    )
    print(f"DEM: {world['dem'].min():.0f} - {world['dem'].max():.0f} m")
    print(
        f"basin_id counts: {np.bincount(world['basin_id'].ravel())}, unlabelled before fill: {n_unlabelled}"
    )
    print(
        f"wells: {len(wells)}  confined {int(wells.is_confined_flag.sum())}  "
        f"sources {int(wells.is_source.sum())}  holdout {int(wells.is_holdout_buffered.sum())}"
    )
    print(f"wells per fold: {wells.fold.value_counts().sort_index().to_dict()}")
    print(
        f"well DTW_obs: min {wells.dtw_obs.min():.1f} med {wells.dtw_obs.median():.1f} max {wells.dtw_obs.max():.1f} m"
    )


def _write_cache(world: dict, wells: pd.DataFrame) -> None:
    TOY_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TOY_WORLD_NPZ, **{k: v for k, v in world.items() if k not in ("X", "Y")}
    )
    wells.to_parquet(TOY_WELLS_PARQUET, index=False)
    print(f"wrote {TOY_WORLD_NPZ}")
    print(f"wrote {TOY_WELLS_PARQUET}")


def load_world(force: bool = False) -> tuple[dict, pd.DataFrame]:
    """Return ``(world, wells)``, building and caching the world if needed.

    ``world`` keys: ``xs``, ``ys``, ``X``, ``Y``, ``dem``, ``dem_filled``,
    ``wte_true``, ``wte_regional``, ``dtw_true``, ``wet``, ``flow_acc``,
    ``streams``, ``dist_to_stream``, ``hand``, ``stream_elev``, ``slope``,
    ``basin_id``, ``playa_basin``, ``receiver``. Every 2-D array has shape
    ``(NY, NX)``.
    """
    if force or not (TOY_WORLD_NPZ.exists() and TOY_WELLS_PARQUET.exists()):
        world, wells = build_world()
        _write_cache(world, wells)
        return world, wells
    with np.load(TOY_WORLD_NPZ) as z:
        world = {k: z[k] for k in z.files}
    world["X"], world["Y"] = np.meshgrid(world["xs"], world["ys"])
    wells = pd.read_parquet(TOY_WELLS_PARQUET)
    return world, wells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--force", action="store_true", help="rebuild even if the cache exists"
    )
    args = ap.parse_args()
    world, wells = build_world()
    _write_cache(world, wells)
    _ = args


if __name__ == "__main__":
    main()
