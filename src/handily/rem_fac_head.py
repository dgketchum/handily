"""Longitudinal channel-head solve on the FAC network.

Solves per-reach water-surface elevation on the directed FAC graph before
the cross-section raster workflow, so that dry headwater channels detach
from the bed rather than producing near-zero REM in the prior.

Uses a residual-depth formulation: solves for additional sag ``r`` below
a local per-reach ceiling ``h_upper``, then recovers head as
``h = h_upper - r``.  This keeps terrain-elevation terms local and makes
graph smoothness operate on extra depth, not absolute hydraulic head.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import xarray as xr

from handily.rem_fac_topology import (
    FacTopologyResult,
    build_fac_topology,
    estimate_reach_seed_strength,
    propagate_downstream_wet_influence,
    propagate_upstream_wet_influence,
    compute_strahler_from_topology,
    sample_reach_drainage_km2,
)

LOGGER = logging.getLogger("handily.rem_fac_head")


def _build_residual_targets(
    streams: gpd.GeoDataFrame,
    *,
    distance_scale_m: float = 4000.0,
    elevation_scale_m: float = 25.0,
    down_distance_scale_m: float = 20000.0,
    area_sag_lo_km2: float = 50.0,
    area_sag_hi_km2: float = 500.0,
    alpha_d: float = 0.75,
    alpha_z: float = 1.0,
    gamma: float = 2.0,
    rmax_min_m: float = 2.0,
    rmax_max_m: float = 60.0,
) -> gpd.GeoDataFrame:
    """Derive per-reach sag targets and bounds from propagated wet influence."""
    if not (0.0 < area_sag_lo_km2 < area_sag_hi_km2):
        raise ValueError("require 0 < area_sag_lo_km2 < area_sag_hi_km2")
    out = streams.copy()

    w_wet = out["topo_pin_weight"].values.astype(np.float64)
    dist = out["topo_dist_to_seed_m"].values.astype(np.float64)
    gain = out["topo_gain_to_seed_m"].values.astype(np.float64)

    dry = 1.0 - w_wet
    f_d = 1.0 - np.exp(-dist / max(distance_scale_m, 1.0))
    f_z = 1.0 - np.exp(-gain / max(elevation_scale_m, 1.0))

    # fmax ignores NaN from dist/gain on reaches with no reachable seed,
    # so those fall back to the dryness term instead of collapsing to NaN.
    g = np.clip(np.fmax(dry, np.fmax(alpha_d * f_d, alpha_z * f_z)), 0.0, 1.0)

    if "topo_down_weight" in out.columns:
        # Per-source sag drivers, wettest evidence wins: upstream-propagated
        # influence (above) and downstream-propagated influence each bound
        # the sag independently; fmin takes the lower (wetter) of the two.
        # Downstream of a wet seed there is no elevation barrier, so g_down
        # uses distance attenuation only.
        w_down = out["topo_down_weight"].values.astype(np.float64)
        dist_down = out["topo_down_dist_m"].values.astype(np.float64)
        f_dd = 1.0 - np.exp(-dist_down / max(down_distance_scale_m, 1.0))
        g_down = np.clip(np.fmax(1.0 - w_down, alpha_d * f_dd), 0.0, 1.0)
        out["sag_driver_down"] = g_down
        g = np.fmin(g, g_down)

    if "drainage_km2" in out.columns:
        # Never-runs-dry prior: a river draining >= area_sag_hi_km2 does not
        # disappear underground, so the sag driver is forced to 0 there
        # regardless of imagery evidence; the ceiling ramps log-linearly up
        # to 1 at area_sag_lo_km2. fmin ignores NaN where drainage is unknown.
        area = out["drainage_km2"].values.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            g_area = (np.log(area_sag_hi_km2) - np.log(area)) / (
                np.log(area_sag_hi_km2) - np.log(area_sag_lo_km2)
            )
        g_area = np.clip(g_area, 0.0, 1.0)
        out["sag_driver_area"] = g_area
        g = np.fmin(g, g_area)

    r_max = rmax_min_m + (rmax_max_m - rmax_min_m) * g
    r_target = r_max * np.power(g, gamma)

    out["sag_driver"] = g
    out["r_target_m"] = r_target
    out["r_max_m"] = r_max

    return out


def solve_channel_heads(
    topology: FacTopologyResult,
    *,
    d_min_off_support_m: float = 0.5,
    support_fraction_threshold: float = 0.25,
    strahler_pin_min: int | None = None,
    area_pin_km2: float | None = None,
    below_bed_offset_m: float = 0.0,
    target_weight_base: float = 2.0,
    zero_weight_base: float = 2.0,
    smoothness_weight: float = 1.0,
    neighbor_length_floor_m: float = 200.0,
    max_hydraulic_slope: float = 0.05,
    max_iter: int = 500,
    tol: float = 0.01,
) -> gpd.GeoDataFrame:
    """Solve per-reach channel-head elevation via residual-depth relaxation.

    Solves for ``r_i >= 0`` (additional sag below a local ceiling) then
    recovers ``h_i = h_upper_i - r_i``.

    Hard-pinned reaches (``seed_support_fraction >= threshold``) get
    ``r = 0``.  ``strahler_pin_min`` additionally hard-pins downstream
    mainstems by topology order (headwaters 0), conjunctively guarded by
    ``area_pin_km2`` when given — the water mask systematically misses
    large channels (narrow/shadow/turbid), leaving them at the
    ``d_min_off_support_m`` floor instead of the surface. Other reaches
    are pulled toward ``r_target`` (derived from dryness / distance /
    elevation gain) while being smoothed along the network in residual
    space.
    """
    streams = topology.streams
    n = len(streams)
    empty_cols = (
        "bed_elev_m",
        "h_upper_m",
        "r_target_m",
        "r_max_m",
        "channel_head_m",
        "head_depth_m",
        "target_weight",
        "zero_weight",
        "hard_pin",
    )
    if n == 0:
        out = streams.copy()
        for col in empty_cols:
            out[col] = np.array([], dtype=np.float64)
        return out

    stream_ids = streams["stream_id"].values.astype(int)
    z_up = streams["up_elev_m"].values.astype(np.float64)
    z_down = streams["down_elev_m"].values.astype(np.float64)
    z_mid = (z_up + z_down) / 2.0
    length_m = streams["length_m"].values.astype(np.float64)

    valid = np.isfinite(z_mid)
    n_invalid = int((~valid).sum())
    if n_invalid > 0:
        LOGGER.warning(
            "  %d / %d reaches have non-finite z_mid, excluded", n_invalid, n
        )

    w_wet = (
        streams["topo_pin_weight"].values.astype(np.float64)
        if "topo_pin_weight" in streams.columns
        else streams["seed_strength"].values.astype(np.float64)
    )

    if "seed_support_fraction" in streams.columns:
        hard_pin = streams["seed_support_fraction"].values >= float(
            support_fraction_threshold
        )
    elif "seed_support_hit" in streams.columns:
        hard_pin = streams["seed_support_hit"].values.astype(bool)
    else:
        hard_pin = np.zeros(n, dtype=bool)
    n_support_pin = int(hard_pin.sum())
    support_pin = hard_pin.copy()

    if strahler_pin_min is not None and "strahler" in streams.columns:
        order_pin = streams["strahler"].values.astype(int) >= int(strahler_pin_min)
        if area_pin_km2 is not None and "drainage_km2" in streams.columns:
            order_pin &= streams["drainage_km2"].values.astype(np.float64) >= float(
                area_pin_km2
            )
        LOGGER.info(
            "  %d reaches hard-pinned by order/area (%d new beyond support)",
            int(order_pin.sum()),
            int((order_pin & ~hard_pin).sum()),
        )
        hard_pin = hard_pin | order_pin

    # --- local ceiling and targets ---
    d_min = float(d_min_off_support_m)
    below_bed = max(float(below_bed_offset_m), 0.0)
    # Pinned reaches normally place the water surface at the streambed
    # (head_depth = 0) — the connected/gaining default. In diverted/losing
    # semi-arid valleys the regional table sits ~1-2 m BELOW the bed (Ruby GWX
    # wells; notes/lit synthesis 4c), so order/area-pinned mainstems get a
    # below-bed offset. Imagery-confirmed wet reaches (support pins) are
    # genuinely connected and stay at the bed. Non-pinned reaches keep the
    # d_min floor (the sag relaxation deepens them further).
    offset = np.where(hard_pin, 0.0, d_min)
    if below_bed > 0.0:
        below_bed_mask = hard_pin & ~support_pin
        offset = np.where(below_bed_mask, below_bed, offset)
        LOGGER.info(
            "  below-bed offset %.2f m on %d order/area-pinned reaches",
            below_bed,
            int(below_bed_mask.sum()),
        )
    h_upper = z_mid - offset

    r_target = (
        streams["r_target_m"].values.astype(np.float64)
        if "r_target_m" in streams.columns
        else np.full(n, d_min, dtype=np.float64)
    )
    r_max = (
        streams["r_max_m"].values.astype(np.float64)
        if "r_max_m" in streams.columns
        else np.full(n, 2.0 * d_min, dtype=np.float64)
    )

    # --- weights ---
    w_target = target_weight_base * np.power(1.0 - w_wet, 1.0)
    w_zero = zero_weight_base * w_wet
    w_smooth = float(smoothness_weight)
    L0 = float(neighbor_length_floor_m)

    # --- neighbor graph ---
    id_to_idx = {int(sid): i for i, sid in enumerate(stream_ids)}
    ds_nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    us_nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, sid in enumerate(stream_ids):
        if not valid[i]:
            continue
        for ds_id in topology.downstream.get(int(sid), ()):
            j = id_to_idx.get(int(ds_id))
            if j is not None and valid[j]:
                L = (length_m[i] + length_m[j]) / 2.0
                ds_nbrs[i].append((j, L))
                us_nbrs[j].append((i, L))

    # --- initialize residual ---
    r = np.clip(r_target.copy(), 0.0, r_max)
    r[hard_pin] = 0.0
    r[~valid] = np.nan

    s_max = float(max_hydraulic_slope)
    LOGGER.info(
        "  %d / %d reaches hard-pinned (%d by support)",
        int(hard_pin.sum()),
        n,
        n_support_pin,
    )

    order = np.argsort(z_mid)

    for iteration in range(int(max_iter)):
        max_change = 0.0

        for i in order:
            if hard_pin[i] or not valid[i]:
                continue

            num = w_target[i] * r_target[i] + w_zero[i] * 0.0
            den = w_target[i] + w_zero[i]

            for j, L in ds_nbrs[i]:
                w_ij = w_smooth / max(L, L0)
                num += w_ij * r[j]
                den += w_ij
            for j, L in us_nbrs[i]:
                w_ij = w_smooth / max(L, L0)
                num += w_ij * r[j]
                den += w_ij

            r_new = num / den if den > 1e-12 else r[i]

            # Max hydraulic slope in recovered-head space:
            # h_i - h_j <= s_max * L  =>  r_i >= r_j + h_upper_i - h_upper_j - s_max * L
            # Wet evidence relaxes the allowance (divide by dryness): observed
            # water surfaces do drop steeply (rapids, dams, canyon riffles),
            # so the smoothness prior must not drag an observed-wet reach off
            # the bed above a steep wet-to-wet drop.
            for j, L in ds_nbrs[i]:
                s_allow = s_max * L / max(1.0 - w_wet[i], 1e-6)
                r_floor = r[j] + h_upper[i] - h_upper[j] - s_allow
                r_new = max(r_new, r_floor)

            r_new = max(0.0, min(r_new, r_max[i]))

            max_change = max(max_change, abs(r_new - r[i]))
            r[i] = r_new

        if iteration % 50 == 0 or max_change < float(tol):
            LOGGER.info("  iteration %d: max_change=%.4f m", iteration + 1, max_change)
        if max_change < float(tol):
            LOGGER.info("Channel head solve converged: %d iterations", iteration + 1)
            break
    else:
        LOGGER.warning(
            "Channel head solve did not converge: %d iterations, max_change=%.4f m",
            int(max_iter),
            max_change,
        )

    h = h_upper - r

    out = streams.copy()
    out["bed_elev_m"] = z_mid
    out["h_upper_m"] = h_upper
    out["r_target_m"] = r_target
    out["r_max_m"] = r_max
    out["channel_head_m"] = h
    out["head_depth_m"] = z_mid - h
    out["target_weight"] = w_target
    out["zero_weight"] = w_zero
    out["hard_pin"] = hard_pin
    return out


def build_channel_heads(
    streams_gdf: gpd.GeoDataFrame,
    elev_da: xr.DataArray,
    ndvi_da: xr.DataArray,
    support_da: xr.DataArray | None = None,
    fac_da: xr.DataArray | None = None,
    *,
    node_precision: int = 3,
    sample_spacing_m: float = 20.0,
    ndvi_quantile: float = 0.9,
    ndvi_mid: float = 0.20,
    ndvi_scale: float = 0.06,
    seed_corridor_m: float = 0.0,
    support_override: float = 1.0,
    distance_scale_m: float = 4000.0,
    elevation_scale_m: float = 25.0,
    down_distance_scale_m: float = 20000.0,
    strahler_distance_scale: float = 0.5,
    area_sag_lo_km2: float = 50.0,
    area_sag_hi_km2: float = 500.0,
    d_min_off_support_m: float = 0.5,
    support_fraction_threshold: float = 0.25,
    strahler_pin_min: int | None = None,
    area_pin_km2: float | None = None,
    below_bed_offset_m: float = 0.0,
    target_weight_base: float = 2.0,
    zero_weight_base: float = 2.0,
    smoothness_weight: float = 1.0,
    neighbor_length_floor_m: float = 200.0,
    max_hydraulic_slope: float = 0.05,
    alpha_d: float = 0.75,
    alpha_z: float = 1.0,
    gamma: float = 2.0,
    rmax_min_m: float = 2.0,
    rmax_max_m: float = 60.0,
    max_iter: int = 500,
    tol: float = 0.01,
) -> gpd.GeoDataFrame:
    """End-to-end: topology, wet anchors, upstream propagation, sag targets, graph solve."""
    LOGGER.info("Building FAC topology (%d reaches)", len(streams_gdf))
    topo = build_fac_topology(
        streams_gdf, elev_da, fac_da, node_precision=node_precision
    )

    if "reach_id" in streams_gdf.columns:
        sid_to_rid = dict(
            zip(
                streams_gdf["stream_id"].astype(int),
                streams_gdf["reach_id"].astype(int),
            )
        )
        topo.streams["reach_id"] = np.array(
            [sid_to_rid.get(int(sid), int(sid)) for sid in topo.streams["stream_id"]],
            dtype=np.int64,
        )

    # Whitebox vector labels carry 0s, downstream drops, and junction
    # contamination; the topology-recomputed order (headwaters 0) drives
    # propagation decay scaling and the order/area hard pin below.
    topo.streams = compute_strahler_from_topology(topo)

    if fac_da is not None:
        LOGGER.info("Sampling reach drainage area from flow accumulation")
        topo.streams["drainage_km2"] = sample_reach_drainage_km2(
            topo.streams, fac_da, sample_spacing_m=sample_spacing_m
        )

    LOGGER.info("Estimating wet-anchor strengths")
    topo.streams = estimate_reach_seed_strength(
        topo.streams,
        ndvi_da,
        support_da=support_da,
        sample_spacing_m=sample_spacing_m,
        ndvi_quantile=ndvi_quantile,
        ndvi_mid=ndvi_mid,
        ndvi_scale=ndvi_scale,
        seed_corridor_m=seed_corridor_m,
        support_override=support_override,
    )

    LOGGER.info("Propagating wet influence upstream")
    topo.streams = propagate_upstream_wet_influence(
        topo,
        distance_scale_m=distance_scale_m,
        elevation_scale_m=elevation_scale_m,
        strahler_distance_scale=strahler_distance_scale,
    )

    LOGGER.info("Propagating wet influence downstream")
    topo.streams = propagate_downstream_wet_influence(
        topo,
        distance_scale_m=down_distance_scale_m,
        strahler_distance_scale=strahler_distance_scale,
    )

    LOGGER.info("Building residual sag targets")
    topo.streams = _build_residual_targets(
        topo.streams,
        distance_scale_m=distance_scale_m,
        elevation_scale_m=elevation_scale_m,
        down_distance_scale_m=down_distance_scale_m,
        area_sag_lo_km2=area_sag_lo_km2,
        area_sag_hi_km2=area_sag_hi_km2,
        alpha_d=alpha_d,
        alpha_z=alpha_z,
        gamma=gamma,
        rmax_min_m=rmax_min_m,
        rmax_max_m=rmax_max_m,
    )

    # The solver's wetness weight pulls toward r=0 with the strongest
    # evidence from either direction (g already took the per-source min).
    # The drainage-area prior counts as wetness too — otherwise the
    # hydraulic-slope floor could still drag evidence-free high-area
    # reaches off the bed at steep drops.
    topo.streams["topo_pin_weight_up"] = topo.streams["topo_pin_weight"]
    w_eff = np.fmax(
        topo.streams["topo_pin_weight"].values,
        topo.streams["topo_down_weight"].values,
    )
    if "sag_driver_area" in topo.streams.columns:
        w_eff = np.fmax(w_eff, 1.0 - topo.streams["sag_driver_area"].values)
    topo.streams["topo_pin_weight"] = w_eff

    LOGGER.info("Solving channel heads (residual-depth)")
    return solve_channel_heads(
        topo,
        d_min_off_support_m=d_min_off_support_m,
        support_fraction_threshold=support_fraction_threshold,
        strahler_pin_min=strahler_pin_min,
        area_pin_km2=area_pin_km2,
        below_bed_offset_m=below_bed_offset_m,
        target_weight_base=target_weight_base,
        zero_weight_base=zero_weight_base,
        smoothness_weight=smoothness_weight,
        neighbor_length_floor_m=neighbor_length_floor_m,
        max_hydraulic_slope=max_hydraulic_slope,
        max_iter=max_iter,
        tol=tol,
    )
