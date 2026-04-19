"""Longitudinal channel-head solve on the FAC network.

Solves per-reach water-surface elevation on the directed FAC graph before
the cross-section raster workflow, so that dry headwater channels detach
from the bed rather than producing near-zero REM in the prior.
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


def solve_channel_heads(
    topology: FacTopologyResult,
    *,
    d_min_off_support_m: float = 0.5,
    wet_anchor_strength: float = 1.0,
    smoothness_weight: float = 1.0,
    max_hydraulic_slope: float = 0.002,
    support_fraction_threshold: float = 0.25,
    max_iter: int = 200,
    tol: float = 0.01,
) -> gpd.GeoDataFrame:
    """Solve per-reach channel-head elevation via iterative projected relaxation.

    Each reach gets a solved water-surface elevation ``h`` that:

    - is hard-pinned to the bed on reaches where the fraction of support
      samples exceeds ``support_fraction_threshold``
    - pins softly near the bed where propagated wet influence is strong
    - rises upstream more slowly than the bed, controlled by
      ``max_hydraulic_slope``
    - stays smooth along the network via the smoothness term

    Off-support reaches use anchor-scaled clearance:
    ``h <= z_mid - d_min * (1 - w_wet)``, so reaches with strong
    propagated wet influence can sit near the bed while dry reaches are
    pushed below.

    Uses ``topo_pin_weight`` (propagated upstream wet influence) when
    available, falling back to raw ``seed_strength``.
    """
    streams = topology.streams
    n = len(streams)
    if n == 0:
        out = streams.copy()
        for col in ("channel_head_m", "head_depth_m", "bed_elev_m"):
            out[col] = np.array([], dtype=np.float64)
        return out

    stream_ids = streams["stream_id"].values.astype(int)
    z_up = streams["up_elev_m"].values.astype(np.float64)
    z_down = streams["down_elev_m"].values.astype(np.float64)
    z_mid = (z_up + z_down) / 2.0
    length_m = streams["length_m"].values.astype(np.float64)

    # Prefer propagated weight (distance/elevation decay from wet seeds)
    if "topo_pin_weight" in streams.columns:
        w_wet = streams["topo_pin_weight"].values.astype(np.float64)
    else:
        w_wet = streams["seed_strength"].values.astype(np.float64)

    # Hard-pin reaches with substantial support evidence.
    # Uses fractional coverage to avoid over-pinning from a single pixel
    # at a confluence or reach break.
    if "seed_support_fraction" in streams.columns:
        hard_pin = streams["seed_support_fraction"].values >= float(
            support_fraction_threshold
        )
    elif "seed_support_hit" in streams.columns:
        hard_pin = streams["seed_support_hit"].values.astype(bool)
    else:
        hard_pin = np.zeros(n, dtype=bool)

    id_to_idx = {int(sid): i for i, sid in enumerate(stream_ids)}

    # Pre-compute neighbor indices and connection lengths
    ds_nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    us_nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, sid in enumerate(stream_ids):
        for ds_id in topology.downstream.get(int(sid), ()):
            j = id_to_idx.get(int(ds_id))
            if j is not None:
                L = (length_m[i] + length_m[j]) / 2.0
                ds_nbrs[i].append((j, L))
                us_nbrs[j].append((i, L))

    # Process downstream-first (ascending bed elevation)
    order = np.argsort(z_mid)

    # Anchor-scaled upper bound: wet reaches can sit at the bed,
    # dry reaches are pushed below by d_min.
    d_min = float(d_min_off_support_m)
    h_upper = z_mid - d_min * (1.0 - w_wet)

    # Initialize: hard-pinned at bed, others at upper bound
    h = h_upper.copy()
    h[hard_pin] = z_mid[hard_pin]

    s_max = float(max_hydraulic_slope)
    w_anchor = float(wet_anchor_strength)
    w_smooth = float(smoothness_weight)
    n_hard = int(hard_pin.sum())
    LOGGER.info("  %d / %d reaches hard-pinned by support", n_hard, n)

    for iteration in range(int(max_iter)):
        max_change = 0.0

        for i in order:
            if hard_pin[i]:
                continue

            # Weighted target: anchor pulls toward bed, smoothness toward
            # neighbors
            w_a = w_anchor * float(w_wet[i])
            num = w_a * z_mid[i]
            den = w_a

            for j, L in ds_nbrs[i]:
                w_s = w_smooth / max(L, 1.0)
                num += w_s * h[j]
                den += w_s
            for j, L in us_nbrs[i]:
                w_s = w_smooth / max(L, 1.0)
                num += w_s * h[j]
                den += w_s

            h_new = num / den if den > 1e-12 else h[i]

            # Upper bound: anchor-scaled clearance
            h_new = min(h_new, h_upper[i])

            # Monotonicity: h >= all downstream neighbors
            for j, _ in ds_nbrs[i]:
                h_new = max(h_new, h[j])

            # Max hydraulic slope
            for j, L in ds_nbrs[i]:
                h_new = min(h_new, h[j] + s_max * L)

            # Hard ceiling at the bed
            h_new = min(h_new, z_mid[i])

            max_change = max(max_change, abs(h_new - h[i]))
            h[i] = h_new

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

    out = streams.copy()
    out["channel_head_m"] = h
    out["head_depth_m"] = z_mid - h
    out["bed_elev_m"] = z_mid
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
    wet_anchor_strength: float = 1.0,
    smoothness_weight: float = 1.0,
    max_hydraulic_slope: float = 0.002,
    support_fraction_threshold: float = 0.25,
    max_iter: int = 200,
    tol: float = 0.01,
) -> gpd.GeoDataFrame:
    """End-to-end: topology, wet anchors, upstream propagation, graph solve.

    Returns a GeoDataFrame with the original reach geometries plus solved
    ``channel_head_m``, ``head_depth_m``, and diagnostics.
    """
    LOGGER.info("Building FAC topology (%d reaches)", len(streams_gdf))
    topo = build_fac_topology(streams_gdf, elev_da, node_precision=node_precision)

    # Preserve reach_id from input (topology build drops it)
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

    LOGGER.info("Solving channel heads")
    return solve_channel_heads(
        topo,
        d_min_off_support_m=d_min_off_support_m,
        wet_anchor_strength=wet_anchor_strength,
        smoothness_weight=smoothness_weight,
        max_hydraulic_slope=max_hydraulic_slope,
        support_fraction_threshold=support_fraction_threshold,
        max_iter=max_iter,
        tol=tol,
    )
