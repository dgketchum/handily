"""Topology-derived pin weights for FAC-based REM relaxation.

This module builds a directed graph from the dense FAC stream network, seeds
wet reaches from NDVI / hard support evidence, and propagates that wet
influence along the network with decay in distance and elevation gain. The
result is a reach-scale pin-weight field consumed by the channel-head solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


@dataclass
class FacTopologyResult:
    streams: gpd.GeoDataFrame
    downstream: dict[int, tuple[int, ...]]
    upstream: dict[int, tuple[int, ...]]


def _build_raster_interp(
    da: xr.DataArray, method: str = "linear"
) -> RegularGridInterpolator:
    vals = np.asarray(da.values, dtype=np.float64)
    x = np.asarray(da.x.values, dtype=np.float64)
    y = np.asarray(da.y.values, dtype=np.float64)
    if y[0] > y[-1]:
        y = y[::-1]
        vals = vals[::-1, :]
    return RegularGridInterpolator(
        (y, x), vals, method=method, bounds_error=False, fill_value=np.nan
    )


def _line_s_values(total_m: float, spacing_m: float) -> np.ndarray:
    if total_m <= spacing_m:
        return (
            np.array([0.0, total_m], dtype=np.float64)
            if total_m > 1e-6
            else np.array([0.0], dtype=np.float64)
        )
    vals = np.arange(0.0, total_m, spacing_m, dtype=np.float64)
    if vals.size == 0 or vals[-1] < total_m - 1e-6:
        vals = np.append(vals, total_m)
    return vals


def _quantize_node(x: float, y: float, precision: int) -> tuple[float, float]:
    return (round(float(x), precision), round(float(y), precision))


def _sample_line_values(
    line,
    interp: RegularGridInterpolator,
    spacing_m: float,
    *,
    trim_m: float = 0.0,
) -> np.ndarray:
    total_m = float(line.length)
    if trim_m > 0.0:
        if total_m > 2.0 * trim_m:
            s_vals = _line_s_values(total_m - 2.0 * trim_m, float(spacing_m)) + trim_m
        else:
            s_vals = np.array([total_m / 2.0], dtype=np.float64)
    else:
        s_vals = _line_s_values(total_m, float(spacing_m))
    pts = shapely.get_coordinates(shapely.line_interpolate_point(line, s_vals))
    return np.asarray(interp(pts[:, ::-1]), dtype=np.float64)


def _sample_corridor_values(
    line,
    interp: RegularGridInterpolator,
    spacing_m: float,
    corridor_m: float,
) -> np.ndarray:
    """Sample ``interp`` over a buffered swath spanning the reach laterally.

    At each along-line station the local centerline tangent is estimated, and
    the raster is sampled along the perpendicular transect at lateral offsets
    spaced ``spacing_m`` apart out to ``±corridor_m``. Returns the flattened
    swath values (NaN outside the raster). Used so a reach whose centerline runs
    down a bare channel still sees the wet/vegetated valley floor beside it.
    """
    total_m = float(line.length)
    s_vals = _line_s_values(total_m, float(spacing_m))
    base = shapely.get_coordinates(shapely.line_interpolate_point(line, s_vals))
    if base.shape[0] >= 2:
        # Discrete centerline tangent (direction only; magnitude is normalized).
        tang = np.gradient(base, axis=0)
    else:
        c0 = np.asarray(line.coords[0], dtype=np.float64)
        c1 = np.asarray(line.coords[-1], dtype=np.float64)
        tang = (c1 - c0).reshape(1, 2)
    tlen = np.hypot(tang[:, 0], tang[:, 1])
    tlen[tlen < 1e-9] = 1.0
    # Unit normal = tangent rotated +90 deg: (tx, ty) -> (-ty, tx).
    nx = -tang[:, 1] / tlen
    ny = tang[:, 0] / tlen
    n_side = max(int(round(float(corridor_m) / float(spacing_m))), 1)
    offsets = np.linspace(-float(corridor_m), float(corridor_m), 2 * n_side + 1)
    px = base[:, 0][:, None] + nx[:, None] * offsets[None, :]
    py = base[:, 1][:, None] + ny[:, None] * offsets[None, :]
    pts_yx = np.column_stack([py.ravel(), px.ravel()])
    return np.asarray(interp(pts_yx), dtype=np.float64)


def build_fac_topology(
    streams_gdf: gpd.GeoDataFrame,
    elev_da: xr.DataArray,
    fac_da: xr.DataArray | None = None,
    *,
    node_precision: int = 3,
) -> FacTopologyResult:
    """Orient FAC reaches and derive upstream/downstream adjacency.

    Orientation prefers endpoint flow accumulation when ``fac_da`` is
    given — D8 accumulation strictly increases downstream, so it cannot
    tie on flat water, where elevation flips reaches and severs mainstem
    chains (a flipped reach's nodes join nothing, leaving a full-drainage
    "headwater" mid-river). Falls back to endpoint elevation where FAC is
    missing or equal.
    """
    if "stream_id" not in streams_gdf.columns:
        raise ValueError("streams_gdf must contain stream_id")
    elev_interp = _build_raster_interp(elev_da)
    fac_interp = (
        _build_raster_interp(fac_da, method="nearest") if fac_da is not None else None
    )

    rows: list[dict] = []
    starts_at: dict[tuple[float, float], list[int]] = {}
    ends_at: dict[tuple[float, float], list[int]] = {}

    first_yx = np.array(
        [[g.coords[0][1], g.coords[0][0]] for g in streams_gdf.geometry],
        dtype=np.float64,
    ).reshape(-1, 2)
    last_yx = np.array(
        [[g.coords[-1][1], g.coords[-1][0]] for g in streams_gdf.geometry],
        dtype=np.float64,
    ).reshape(-1, 2)
    z_first = np.asarray(elev_interp(first_yx), dtype=np.float64)
    z_last = np.asarray(elev_interp(last_yx), dtype=np.float64)
    if fac_interp is not None:
        f_first = np.asarray(fac_interp(first_yx), dtype=np.float64)
        f_last = np.asarray(fac_interp(last_yx), dtype=np.float64)

    for i, row in enumerate(streams_gdf.itertuples(index=False)):
        geom = row.geometry
        coords = list(geom.coords)
        z0 = float(z_first[i])
        z1 = float(z_last[i])
        reverse = np.isfinite(z0) and np.isfinite(z1) and (z1 > z0)
        if fac_interp is not None:
            f0 = float(f_first[i])
            f1 = float(f_last[i])
            if np.isfinite(f0) and np.isfinite(f1) and f0 != f1:
                reverse = f0 > f1
        if reverse:
            geom = type(geom)(coords[::-1])
            z_up, z_down = z1, z0
        else:
            z_up, z_down = z0, z1
        oriented = list(geom.coords)
        up_node = _quantize_node(oriented[0][0], oriented[0][1], node_precision)
        down_node = _quantize_node(oriented[-1][0], oriented[-1][1], node_precision)
        stream_id = int(row.stream_id)
        starts_at.setdefault(up_node, []).append(stream_id)
        ends_at.setdefault(down_node, []).append(stream_id)
        length_m = float(getattr(row, "length_m", geom.length))
        relief_m = (
            float(max(z_up - z_down, 0.0))
            if np.isfinite(z_up) and np.isfinite(z_down)
            else np.nan
        )
        rows.append(
            {
                "stream_id": stream_id,
                "strahler": int(getattr(row, "strahler", 0))
                if getattr(row, "strahler", None) is not None
                else 0,
                "length_m": length_m,
                "up_elev_m": z_up,
                "down_elev_m": z_down,
                "relief_m": relief_m,
                "up_node_x": up_node[0],
                "up_node_y": up_node[1],
                "down_node_x": down_node[0],
                "down_node_y": down_node[1],
                "geometry": geom,
            }
        )

    downstream: dict[int, list[int]] = {int(r["stream_id"]): [] for r in rows}
    upstream: dict[int, list[int]] = {int(r["stream_id"]): [] for r in rows}
    for node in set(starts_at) | set(ends_at):
        ups = ends_at.get(node, [])
        downs = starts_at.get(node, [])
        for u in ups:
            for d in downs:
                if u == d:
                    continue
                downstream[u].append(d)
                upstream[d].append(u)

    streams = (
        gpd.GeoDataFrame(rows, geometry="geometry", crs=streams_gdf.crs)
        .sort_values("stream_id")
        .reset_index(drop=True)
    )
    downstream_t = {int(k): tuple(sorted(set(v))) for k, v in downstream.items()}
    upstream_t = {int(k): tuple(sorted(set(v))) for k, v in upstream.items()}
    return FacTopologyResult(
        streams=streams, downstream=downstream_t, upstream=upstream_t
    )


def compute_strahler_from_topology(topology: FacTopologyResult) -> gpd.GeoDataFrame:
    """Recompute Strahler order on the directed reach graph.

    The Whitebox vector labels are assigned by sampling the order raster
    along each line, so a tributary whose junction vertex snaps onto the
    receiving mainstem's cell inherits the mainstem's order — the same
    failure class as the junction drainage contamination. Recomputing
    from topology replaces both that sampling and the old monotone
    downstream repair (which could only ratchet bad labels upward).

    Convention: headwater reaches (no upstream) are order 0; order
    increments only where two or more max-order contributors meet,
    otherwise the max passes through — so a reach split mid-channel
    keeps its order. The incoming label is preserved as ``strahler_raw``.
    """
    streams = topology.streams.copy()
    raw = {
        int(r.stream_id): int(getattr(r, "strahler", 0))
        for r in streams.itertuples(index=False)
    }
    memo: dict[int, int] = {}

    def solve(start: int) -> int:
        if start in memo:
            return memo[start]
        stack = [start]
        on_stack = {start}
        while stack:
            sid = stack[-1]
            ups = [int(u) for u in topology.upstream.get(sid, ())]
            todo = [u for u in ups if u not in memo and u not in on_stack]
            if todo:
                stack.extend(todo)
                on_stack.update(todo)
                continue
            # Cycle members still on the stack contribute order 0.
            vals = [memo.get(u, 0) for u in ups]
            if vals:
                best = max(vals)
                memo[sid] = best + 1 if vals.count(best) >= 2 else best
            else:
                memo[sid] = 0
            stack.pop()
            on_stack.discard(sid)
        return memo[start]

    streams["strahler_raw"] = [raw[int(s)] for s in streams["stream_id"]]
    streams["strahler"] = [solve(int(s)) for s in streams["stream_id"]]
    return streams


def sample_reach_drainage_km2(
    streams_gdf: gpd.GeoDataFrame,
    fac_da: xr.DataArray,
    *,
    sample_spacing_m: float = 20.0,
    junction_trim_cells: float = 1.5,
) -> np.ndarray:
    """Per-reach upstream drainage area (km2) from a D8 flow-accumulation grid.

    Samples cell counts along each reach with nearest-neighbor lookup (FAC
    is not interpolable — averaging channel and hillslope cells is
    meaningless) and takes the max, so samples that fall off-channel do not
    dilute the estimate.

    Samples within ``junction_trim_cells`` FAC cells of either line endpoint
    are excluded: the downstream vertex of a tributary sits on (or NN-snaps
    to) the receiving mainstem's junction cell, so an untrimmed max inherits
    the mainstem's accumulation. Both ends are trimmed because vector line
    orientation is not guaranteed; the upstream end carries the reach's
    minimum FAC, so trimming it never affects the max. Reaches shorter than
    twice the trim fall back to a single midpoint sample.
    """
    x = np.asarray(fac_da.x.values, dtype=np.float64)
    y = np.asarray(fac_da.y.values, dtype=np.float64)
    dx = abs(float(np.median(np.diff(x))))
    dy = abs(float(np.median(np.diff(y))))
    cell_area_m2 = dx * dy
    trim_m = float(junction_trim_cells) * max(dx, dy)
    interp = _build_raster_interp(fac_da, method="nearest")
    out = np.full(len(streams_gdf), np.nan, dtype=np.float64)
    for i, row in enumerate(streams_gdf.itertuples(index=False)):
        vals = _sample_line_values(
            row.geometry, interp, sample_spacing_m, trim_m=trim_m
        )
        good = vals[np.isfinite(vals)]
        if good.size:
            out[i] = float(good.max()) * cell_area_m2 / 1e6
    return out


def estimate_reach_seed_strength(
    streams_gdf: gpd.GeoDataFrame,
    ndvi_da: xr.DataArray,
    support_da: xr.DataArray | None = None,
    *,
    sample_spacing_m: float = 20.0,
    ndvi_quantile: float = 0.9,
    ndvi_mid: float = 0.35,
    ndvi_scale: float = 0.06,
    seed_corridor_m: float = 0.0,
    support_override: float = 1.0,
) -> gpd.GeoDataFrame:
    """Estimate wet seed strength per FAC reach from raw NDVI and hard support.

    The NDVI quantile is taken over the reach centerline when
    ``seed_corridor_m == 0`` (legacy), or over a buffered swath of width
    ``±seed_corridor_m`` when positive — the latter lets valley-floor reaches
    whose channel itself is bare pick up adjacent irrigated/vegetated ground.
    Hard support is always sampled on the centerline only, to avoid pulling in
    off-channel open water.
    """
    if not (0.0 < ndvi_quantile <= 1.0):
        raise ValueError("ndvi_quantile must be in (0, 1]")
    if ndvi_scale <= 0.0:
        raise ValueError("ndvi_scale must be > 0")
    if seed_corridor_m < 0.0:
        raise ValueError("seed_corridor_m must be >= 0")
    ndvi_interp = _build_raster_interp(ndvi_da)
    support_interp = (
        _build_raster_interp(support_da) if support_da is not None else None
    )

    out = streams_gdf.copy()
    ndvi_p = np.full(len(out), np.nan, dtype=np.float64)
    support_hit = np.zeros(len(out), dtype=bool)
    support_fraction = np.zeros(len(out), dtype=np.float64)
    seed = np.zeros(len(out), dtype=np.float64)

    for i, row in enumerate(out.itertuples(index=False)):
        if seed_corridor_m > 0.0:
            vals = _sample_corridor_values(
                row.geometry, ndvi_interp, sample_spacing_m, seed_corridor_m
            )
        else:
            vals = _sample_line_values(row.geometry, ndvi_interp, sample_spacing_m)
        good = np.isfinite(vals)
        if np.any(good):
            q = float(np.nanquantile(vals[good], ndvi_quantile))
            ndvi_p[i] = q
            seed_ndvi = 1.0 / (1.0 + np.exp(-(q - float(ndvi_mid)) / float(ndvi_scale)))
        else:
            seed_ndvi = 0.0
        seed_val = float(seed_ndvi)
        if support_interp is not None:
            svals = _sample_line_values(row.geometry, support_interp, sample_spacing_m)
            # Drop first and last samples (endpoints) to avoid leaking
            # support across shared confluence/reach-break vertices.
            # Reaches with ≤ 2 samples (shorter than sample_spacing) have
            # no true interior points, so seed_support_fraction stays 0 —
            # they are too short to assess support independently and must
            # not be hard-pinned. They can still get a soft anchor via
            # seed_support_hit from the full-line fallback.
            svals_interior = (
                svals[1:-1] if len(svals) > 2 else np.array([], dtype=svals.dtype)
            )
            svals_valid = svals_interior[np.isfinite(svals_interior)]
            svals_all = svals[np.isfinite(svals)]
            if len(svals_valid) > 0:
                frac = float(np.sum(svals_valid > 0.5)) / len(svals_valid)
                support_fraction[i] = frac
                hit = frac > 0.0
            elif len(svals_all) > 0:
                hit = bool(np.any(svals_all > 0.5))
            else:
                hit = False
            support_hit[i] = hit
            if hit:
                seed_val = max(seed_val, float(support_override))
        seed[i] = min(max(seed_val, 0.0), 1.0)

    out["seed_ndvi_q"] = ndvi_p
    out["seed_support_hit"] = support_hit
    out["seed_support_fraction"] = support_fraction
    out["seed_strength"] = seed
    return out


def _propagate_wet_influence(
    topology: FacTopologyResult,
    neighbor_map: dict[int, tuple[int, ...]],
    *,
    distance_scale_m: float,
    elevation_scale_m: float | None,
    strahler_distance_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Max-decay wet influence pulled from ``neighbor_map`` (iterative DFS).

    Influence flows *from* a reach's mapped neighbors: passing
    ``topology.downstream`` propagates seed wetness upstream and vice
    versa. Decay per traversed reach uses its own length (plus relief as
    an uphill barrier when ``elevation_scale_m`` is given); higher
    Strahler order extends the decay length. Iterative so stack depth
    does not bound network chain length. Returns per-reach arrays
    (weight, seed stream_id, distance to seed, elevation gain to seed).
    """
    streams = topology.streams
    seed_strength: dict[int, float] = {}
    length: dict[int, float] = {}
    relief: dict[int, float] = {}
    decay: dict[int, float] = {}
    for r in streams.itertuples(index=False):
        sid = int(r.stream_id)
        seed_strength[sid] = float(getattr(r, "seed_strength", 0.0))
        length[sid] = float(getattr(r, "length_m", 0.0))
        rel = float(getattr(r, "relief_m", np.nan))
        if not np.isfinite(rel):
            rel = 0.0
        relief[sid] = rel
        ld_eff = float(distance_scale_m) * (
            1.0
            + float(strahler_distance_scale)
            * max(int(getattr(r, "strahler", 0)) - 1, 0)
        )
        arg = -length[sid] / ld_eff
        if elevation_scale_m is not None:
            arg = -length[sid] / ld_eff - rel / float(elevation_scale_m)
        decay[sid] = float(np.exp(arg))

    memo: dict[int, tuple[float, int, float, float]] = {}

    def solve(start: int) -> tuple[float, int, float, float]:
        if start in memo:
            return memo[start]
        stack = [start]
        on_stack = {start}
        while stack:
            sid = stack[-1]
            nbrs = neighbor_map.get(sid, ())
            todo = [
                int(n) for n in nbrs if int(n) not in memo and int(n) not in on_stack
            ]
            if todo:
                stack.extend(todo)
                on_stack.update(todo)
                continue
            w0 = seed_strength.get(sid, 0.0)
            best_weight = w0
            best_seed = sid if w0 > 0.0 else -1
            best_dist = 0.0 if best_seed >= 0 else np.nan
            best_gain = 0.0 if best_seed >= 0 else np.nan
            for n in nbrs:
                n = int(n)
                if n in memo:
                    nb_w, nb_seed, nb_dist, nb_gain = memo[n]
                else:
                    # Cycle member still on the stack: provisional own-seed
                    # value, matching one re-entry level of the network.
                    ns = seed_strength.get(n, 0.0)
                    nb_w, nb_seed, nb_dist, nb_gain = (
                        ns,
                        n if ns > 0.0 else -1,
                        0.0,
                        0.0,
                    )
                cand = float(nb_w) * decay[sid]
                if cand > best_weight:
                    best_weight = cand
                    best_seed = int(nb_seed)
                    best_dist = float(nb_dist) + length[sid]
                    best_gain = float(nb_gain) + relief[sid]
            memo[sid] = (best_weight, best_seed, best_dist, best_gain)
            stack.pop()
            on_stack.discard(sid)
        return memo[start]

    n = len(streams)
    weight = np.zeros(n, dtype=np.float64)
    source_seed = np.full(n, -1, dtype=np.int64)
    dist_to_seed = np.full(n, np.nan, dtype=np.float64)
    gain_to_seed = np.full(n, np.nan, dtype=np.float64)
    for i, sid in enumerate(streams["stream_id"]):
        w, s, d, g = solve(int(sid))
        weight[i] = w
        source_seed[i] = int(s)
        dist_to_seed[i] = d
        gain_to_seed[i] = g
    return weight, source_seed, dist_to_seed, gain_to_seed


def propagate_upstream_wet_influence(
    topology: FacTopologyResult,
    *,
    distance_scale_m: float = 1500.0,
    elevation_scale_m: float = 25.0,
    strahler_distance_scale: float = 0.5,
) -> gpd.GeoDataFrame:
    """Propagate wet seed influence upstream with exponential decay."""
    if distance_scale_m <= 0.0:
        raise ValueError("distance_scale_m must be > 0")
    if elevation_scale_m <= 0.0:
        raise ValueError("elevation_scale_m must be > 0")
    streams = topology.streams.copy()
    weight, seed, dist, gain = _propagate_wet_influence(
        topology,
        topology.downstream,
        distance_scale_m=distance_scale_m,
        elevation_scale_m=elevation_scale_m,
        strahler_distance_scale=strahler_distance_scale,
    )
    streams["topo_pin_weight"] = weight
    streams["topo_seed_stream_id"] = seed
    streams["topo_dist_to_seed_m"] = dist
    streams["topo_gain_to_seed_m"] = gain
    return streams


def propagate_downstream_wet_influence(
    topology: FacTopologyResult,
    *,
    distance_scale_m: float = 20000.0,
    strahler_distance_scale: float = 0.5,
) -> gpd.GeoDataFrame:
    """Propagate wet seed influence downstream with distance-only decay.

    Mirror of :func:`propagate_upstream_wet_influence` for the opposite
    direction: water entering a reach from wet upstream seeds does not
    vanish, so wetness persists downstream, attenuated only by channel
    distance (losing reaches, diversions) — there is no elevation-gain
    barrier going downhill. Higher Strahler order extends the decay length
    so large mainstems carry wetness farther.
    """
    if distance_scale_m <= 0.0:
        raise ValueError("distance_scale_m must be > 0")
    streams = topology.streams.copy()
    weight, seed, dist, _ = _propagate_wet_influence(
        topology,
        topology.upstream,
        distance_scale_m=distance_scale_m,
        elevation_scale_m=None,
        strahler_distance_scale=strahler_distance_scale,
    )
    streams["topo_down_weight"] = weight
    streams["topo_down_seed_stream_id"] = seed
    streams["topo_down_dist_m"] = dist
    return streams
