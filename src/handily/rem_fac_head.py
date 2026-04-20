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
    propagate_upstream_wet_influence,
)

LOGGER = logging.getLogger("handily.rem_fac_head")


def _build_residual_targets(
    streams: gpd.GeoDataFrame,
    *,
    distance_scale_m: float = 1500.0,
    elevation_scale_m: float = 25.0,
    alpha_d: float = 0.75,
    alpha_z: float = 1.0,
    gamma: float = 1.5,
    rmax_min_m: float = 2.0,
    rmax_max_m: float = 40.0,
) -> gpd.GeoDataFrame:
    """Derive per-reach sag targets and bounds from propagated wet influence."""
    out = streams.copy()

    w_wet = out["topo_pin_weight"].values.astype(np.float64)
    dist = out["topo_dist_to_seed_m"].values.astype(np.float64)
    gain = out["topo_gain_to_seed_m"].values.astype(np.float64)

    dry = 1.0 - w_wet
    f_d = 1.0 - np.exp(-dist / max(distance_scale_m, 1.0))
    f_z = 1.0 - np.exp(-gain / max(elevation_scale_m, 1.0))

    g = np.clip(np.maximum(dry, np.maximum(alpha_d * f_d, alpha_z * f_z)), 0.0, 1.0)

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
    target_weight_base: float = 2.0,
    zero_weight_base: float = 2.0,
    smoothness_weight: float = 1.0,
    neighbor_length_floor_m: float = 200.0,
    max_hydraulic_slope: float = 0.002,
    max_iter: int = 500,
    tol: float = 0.01,
) -> gpd.GeoDataFrame:
    """Solve per-reach channel-head elevation via residual-depth relaxation.

    Solves for ``r_i >= 0`` (additional sag below a local ceiling) then
    recovers ``h_i = h_upper_i - r_i``.

    Hard-pinned reaches (``seed_support_fraction >= threshold``) get
    ``r = 0``.  Other reaches are pulled toward ``r_target`` (derived
    from dryness / distance / elevation gain) while being smoothed along
    the network in residual space.
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

    # --- local ceiling and targets ---
    d_min = float(d_min_off_support_m)
    h_upper = np.where(hard_pin, z_mid, z_mid - d_min)

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
    n_hard = int(hard_pin.sum())
    LOGGER.info("  %d / %d reaches hard-pinned by support", n_hard, n)

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
            for j, L in ds_nbrs[i]:
                r_floor = r[j] + h_upper[i] - h_upper[j] - s_max * L
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
    *,
    node_precision: int = 3,
    sample_spacing_m: float = 20.0,
    ndvi_quantile: float = 0.9,
    ndvi_mid: float = 0.35,
    ndvi_scale: float = 0.06,
    support_override: float = 1.0,
    distance_scale_m: float = 1500.0,
    elevation_scale_m: float = 25.0,
    strahler_distance_scale: float = 0.5,
    d_min_off_support_m: float = 0.5,
    support_fraction_threshold: float = 0.25,
    target_weight_base: float = 2.0,
    zero_weight_base: float = 2.0,
    smoothness_weight: float = 1.0,
    neighbor_length_floor_m: float = 200.0,
    max_hydraulic_slope: float = 0.002,
    alpha_d: float = 0.75,
    alpha_z: float = 1.0,
    gamma: float = 1.5,
    rmax_min_m: float = 2.0,
    rmax_max_m: float = 40.0,
    max_iter: int = 500,
    tol: float = 0.01,
) -> gpd.GeoDataFrame:
    """End-to-end: topology, wet anchors, upstream propagation, sag targets, graph solve."""
    LOGGER.info("Building FAC topology (%d reaches)", len(streams_gdf))
    topo = build_fac_topology(streams_gdf, elev_da, node_precision=node_precision)

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

    LOGGER.info("Estimating wet-anchor strengths")
    topo.streams = estimate_reach_seed_strength(
        topo.streams,
        ndvi_da,
        support_da=support_da,
        sample_spacing_m=sample_spacing_m,
        ndvi_quantile=ndvi_quantile,
        ndvi_mid=ndvi_mid,
        ndvi_scale=ndvi_scale,
        support_override=support_override,
    )

    LOGGER.info("Propagating wet influence upstream")
    topo.streams = propagate_upstream_wet_influence(
        topo,
        distance_scale_m=distance_scale_m,
        elevation_scale_m=elevation_scale_m,
        strahler_distance_scale=strahler_distance_scale,
    )

    LOGGER.info("Building residual sag targets")
    topo.streams = _build_residual_targets(
        topo.streams,
        distance_scale_m=distance_scale_m,
        elevation_scale_m=elevation_scale_m,
        alpha_d=alpha_d,
        alpha_z=alpha_z,
        gamma=gamma,
        rmax_min_m=rmax_min_m,
        rmax_max_m=rmax_max_m,
    )

    LOGGER.info("Solving channel heads (residual-depth)")
    return solve_channel_heads(
        topo,
        d_min_off_support_m=d_min_off_support_m,
        support_fraction_threshold=support_fraction_threshold,
        target_weight_base=target_weight_base,
        zero_weight_base=zero_weight_base,
        smoothness_weight=smoothness_weight,
        neighbor_length_floor_m=neighbor_length_floor_m,
        max_hydraulic_slope=max_hydraulic_slope,
        max_iter=max_iter,
        tol=tol,
    )
