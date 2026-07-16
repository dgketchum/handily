# Walkthrough: `fac_head_depth_rem_5m.tif`

How the depth-to-water raster at `experiments/nv_aoi_0773_archive/experimental_full_idw1k/fac_head_depth_rem_5m.tif` was produced.

**Config:** `configs/rem/0773_idw1k.toml`
**Command:** `uv run python -m handily.rem_fac --config configs/rem/0773_idw1k.toml`
**Code entry point:** `src/handily/rem_fac.py:main()` (line 1373)

---

## Inputs

| Input | Path | Description |
|-------|------|-------------|
| DEM | `/data/ssd2/handily/nv/aoi_0773/dem_bounds_1m.tif` | 1m LiDAR DEM from USGS 3DEP |
| Streams | `experiments/nv_aoi_0773_archive/streams_fac.fgb` | FAC stream network (D8, per-AOI extraction) |
| NAIP | `nv/aoi_0773/naip/ortho_1-1_hm_s_nv007_2022_1.tif` | 1m 4-band NAIP imagery (2022) |
| Support | `experimental_full/fac_normals_ws_relax_support_20m.tif` | Binary surface-water mask from prior NDWI/RF classification |

## Pipeline (10 steps)

### 1. Load DEM and streams
Read 1m LiDAR DEM and FAC stream network. Filter streams by `min_strahler` (0 = keep all). Assign `reach_id` if missing.

### 2. Build orientation field
Resample DEM to 20m (`coarse_res_m=20`), apply Gaussian smooth (`smooth_sigma_m=500`) to get a terrain aspect field. This field defines the direction cross-sections are cast (perpendicular to local downslope).
**Code:** `build_orientation_field()` (line 1396)

### 3. Generate aspect-normal strips
Along each stream reach, cast cross-section rays perpendicular to the smoothed aspect at 50m intervals (`station_spacing_m=50`). Each ray extends from the stream centerline until it hits an adjacent stream or the AOI boundary. In addition, 12 halo spokes (`halo_n=12`) are cast beyond normal strip termination for better coverage in wide valleys. Sample DEM at the thalweg to get `base_elev_m` per strip.
**Code:** `generate_fac_strips()` (line 1411)

### 4. Filter crossing strips
Remove strips that cross more than 800m (`max_crossing_strip_m=800`) of terrain -- these are artifacts from valley-spanning geometry errors.
**Code:** `_remove_long_crossing_strips()` (line 1428)

### 5. Compute NDVI
Compute NDVI from NAIP bands at burn resolution (5m), then max-resample to 20m so the greenest pixel in each 20m cell drives the seed strength. This prevents narrow riparian corridors from being averaged away.
**Code:** `compute_naip_ndvi_match()` + `rio.reproject_match(resampling=max)` (lines 1450-1455)

### 6. Head solve (longitudinal water-surface estimation)

The head solve is the core algorithmic step. It assigns a per-reach water-surface depth (`head_depth_m`) — the distance in meters from the DEM bed to the estimated water surface — using a graph-based residual-depth relaxation on the FAC stream topology. The solve operates in residual space rather than absolute elevation space: it solves for `r` (additional sag below a local ceiling) then recovers `h = h_upper - r`. This formulation keeps terrain-elevation terms local and makes the network smoothness penalty operate on extra depth rather than absolute hydraulic head, which avoids numerical issues from mixing reach-scale elevation differences with meter-scale depth adjustments.

**Code:** `build_channel_heads()` in `src/handily/rem_fac_head.py` (line 240), which calls four sub-routines in sequence:

#### 6a. Build directed topology

`build_fac_topology()` in `src/handily/rem_fac_topology.py` (line 80)

Construct a directed graph from the FAC stream network. For each reach, sample the DEM at both endpoints. If the downstream end is higher than the upstream end, reverse the geometry so all reaches flow downhill. Quantize endpoint coordinates to millimeter precision (`node_precision=3`) and match reaches that share a quantized node to build upstream/downstream adjacency lists. Each reach gets `up_elev_m`, `down_elev_m`, `relief_m` (elevation drop along the reach), and `length_m`.

The output is a `FacTopologyResult` containing the oriented stream GeoDataFrame plus two dicts: `downstream[stream_id] -> (downstream_ids...)` and `upstream[stream_id] -> (upstream_ids...)`.

#### 6b. Estimate reach seed strength

`estimate_reach_seed_strength()` in `src/handily/rem_fac_topology.py` (line 164)

Score each reach on a 0-1 scale representing confidence that it carries surface water. Two evidence sources are combined:

- **NDVI signal:** Sample the 20m max-resampled NDVI along the reach at 20m intervals. Take the 90th percentile value (`ndvi_quantile=0.9`) and pass it through a logistic function: `seed = 1 / (1 + exp(-(ndvi_q90 - ndvi_mid) / ndvi_scale))`. With `ndvi_mid=0.20` and `ndvi_scale=0.06`, a reach with q90 NDVI of 0.20 gets seed 0.5; reaches above 0.30 saturate near 1.0; reaches below 0.10 are near 0.0. The 90th percentile (not the median) is used because a few green pixels along an otherwise dry channel are meaningful — they indicate at least intermittent water or shallow groundwater.

- **Hard support override:** If a binary surface-water support raster is provided, sample it along the reach. To avoid false hard-pins from shared confluence vertices, only interior samples (excluding the first and last point) are considered. If the fraction of interior samples hitting the support mask exceeds `support_fraction_threshold=0.25`, the reach is flagged as `hard_pin=True` and its seed strength is forced to `support_override=1.0`. Hard-pinned reaches are locked to the bed in the solver (step 6e) — their water surface equals their DEM elevation.

Output columns: `seed_strength`, `seed_ndvi_q`, `seed_support_hit`, `seed_support_fraction`.

#### 6c. Propagate wet influence upstream

`propagate_upstream_wet_influence()` in `src/handily/rem_fac_topology.py` (line 235)

Seed strength is a local, per-reach measurement. A wet mainstem should also influence its dry headwater tributaries — otherwise every reach without green vegetation would be treated as maximally dry regardless of its position in the network.

This step propagates wet influence upstream through the directed graph using memoized recursion (depth-first from each reach, cached). For each reach, the algorithm looks downstream and takes the best (highest) wet signal it can find, decayed by an exponential function of network distance and elevation gain:

```
decay = exp(-length_m / Ld_eff - relief_m / elevation_scale_m)
propagated_weight = downstream_weight * decay
```

where `Ld_eff = distance_scale_m * (1 + strahler_distance_scale * (strahler - 1))`. With `distance_scale_m=4000` the e-folding distance is 4 km on first-order streams and longer on higher-order ones (the `strahler_distance_scale=0.5` term extends the reach of influence on larger channels). The `elevation_scale_m=25` term ensures that influence decays faster when climbing steep terrain — a tributary gaining 50m of elevation in 500m of channel distance receives much less downstream influence than a flat tributary of the same length.

The result is `topo_pin_weight`: a reach's effective wetness considering both its own NDVI evidence and the strongest downstream signal it can see. Also recorded: `topo_dist_to_seed_m` (network distance to the source of its best signal) and `topo_gain_to_seed_m` (cumulative elevation gain to that source).

#### 6d. Derive per-reach sag targets

`_build_residual_targets()` in `src/handily/rem_fac_head.py` (line 31)

Convert the propagated wet influence into a per-reach target depth. The key quantity is the "dryness driver" `g`, which combines three signals into a single 0-1 score:

```
dry = 1 - topo_pin_weight
f_d  = 1 - exp(-dist_to_seed / distance_scale_m)
f_z  = 1 - exp(-gain_to_seed / elevation_scale_m)
g    = max(dry, alpha_d * f_d, alpha_z * f_z)
```

`g` is high (near 1) for reaches that are (a) not wet themselves, (b) far from any wet reach, or (c) high above the nearest wet reach. It is low (near 0) for wet reaches close to the mainstem. The `alpha_d=0.75` and `alpha_z=1.0` coefficients control how aggressively distance and elevation override the propagated wetness.

From `g`, two bounds are derived:

- **Sag ceiling:** `r_max = rmax_min + (rmax_max - rmax_min) * g` = 2 + 28*g meters. This is the maximum depth the solver is allowed to push the water surface below the bed. A reach with `g=1` (maximally dry headwater) can sag up to 30m; a reach with `g=0` (wet mainstem) is capped at 2m.
- **Target sag:** `r_target = r_max * g^gamma` = `r_max * g^2.0`. The gamma exponent controls the nonlinearity: with `gamma=2.0`, the target sag grows as the square of dryness, meaning partially-dry reaches (g ~ 0.5) get modest targets (~3.5m) while only very dry reaches (g > 0.8) get deep targets (>15m).

Output columns: `sag_driver`, `r_target_m`, `r_max_m`.

#### 6e. Iterative projected relaxation

`solve_channel_heads()` in `src/handily/rem_fac_head.py` (line 65)

Solve for the per-reach residual `r_i >= 0` that balances three competing objectives:

1. **Target pull** — push `r` toward `r_target` (the depth derived from dryness). Weight: `target_weight_base * (1 - w_wet)` = `2.0 * (1 - topo_pin_weight)`. Dry reaches feel strong pull toward their target; wet reaches feel almost none.

2. **Zero pull** — push `r` toward 0 (water surface at the bed). Weight: `zero_weight_base * w_wet` = `2.0 * topo_pin_weight`. Wet reaches are pulled toward the bed; dry reaches feel almost no pull toward zero.

3. **Neighbor smoothness** — push `r` toward the average of its upstream and downstream neighbors' residuals. Weight: `smoothness_weight / max(L_ij, neighbor_length_floor_m)` = `3.0 / max(L, 200)`. The length-inverse weighting means closely-spaced reaches are more strongly smoothed; the floor prevents division by very short reaches from dominating.

At each iteration, every non-pinned reach is visited in elevation order (lowest first). The new residual is computed as a weighted average:

```
r_new = (w_target * r_target + w_zero * 0 + sum(w_smooth/L * r_neighbor)) / (w_target + w_zero + sum(w_smooth/L))
```

Then two constraints are enforced:

- **Hydraulic gradient limit:** For each downstream neighbor, `r_new >= r_j + h_upper_i - h_upper_j - s_max * L_ij` (where `s_max=0.05`). This prevents the recovered water surface from dropping faster than 5 cm/m along the network — a physical plausibility bound.
- **Box constraint:** `r_new` is clamped to `[0, r_max]`.

Hard-pinned reaches (from step 6b) are locked at `r=0` throughout and act as fixed boundary conditions — the solver fills in around them.

The local ceiling `h_upper` is set to `z_mid` (midpoint bed elevation) for hard-pinned reaches, and `z_mid - d_min_off_support_m` (= z_mid - 0.5m) for all others. This 0.5m offset ensures that even the shallowest non-pinned reach has its water surface slightly below the bed.

Iteration continues until the maximum per-reach change drops below `tol=0.01m` or `max_iter=500` is reached. Final recovery: `head_depth_m = z_mid - h = r + d_min` (for non-pinned) or `0` (for pinned).

**Output:** `fac_channel_heads.fgb` — per-reach attributes including `head_depth_m`, `channel_head_m`, `bed_elev_m`, `h_upper_m`, `topo_pin_weight`, `seed_strength`, `r_target_m`, `r_max_m`, `sag_driver`, `hard_pin`.

### 7. Depth-offset application
Instead of stamping an absolute water-surface elevation (which causes sawtooth artifacts on long reaches), subtract each reach's `head_depth_m` from each strip's DEM-sampled `base_elev_m`:

```
WS_at_station = base_elev_m - head_depth_m
```

The water surface follows the bed slope at constant depth per reach. Wet reaches (depth ~ 0.5m) have WS near the bed; dry headwaters (depth ~ 15-30m) have WS well below.
**Code:** `_attach_fac_strip_head_depth_offset()` (line 708)

### 8. Sparse rasterization
Burn the depth-offset strip elevations to a 5m grid (`burn_res_m=5`). Only pixels at strip station locations receive values; the rest are NaN. Uses min-burn: where multiple strips overlap a cell, the lowest WS wins.
**Code:** `rasterize_sparse_sections_20m()` (line 1490, despite the function name it uses the configured resolution)

### 9. IDW fill
Fill the sparse 5m raster using inverse-distance weighting with 1000m search radius (`idw_radius_m=1000`) and power=2.0 (`idw_power=2`). Implemented as FFT convolution of a pre-computed IDW kernel, so the entire grid is filled in one pass.
**Code:** `fill_sparse_sections_idw()` (line 1512)

### 10. Post-processing
Three operations applied sequentially:

1. **Gaussian smooth** (`post_smooth_m=100`) — feathers strip-edge discontinuities with a 100m-sigma Gaussian blur (NaN-aware)
2. **DEM clamp** — water surface cannot exceed ground elevation. Uses min-resampled DEM (1m DEM resampled to 5m with `Resampling.min`) so narrow channel bottoms are preserved in the clamp surface
3. **REM computation** — `REM = max(DEM_5m - filled_WS, 0)`. Pixels below the water surface get REM = 0

**Output:** `fac_head_depth_rem_5m.tif`

---

## Key parameter choices and why

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `ndvi_mid=0.20` | Lowered from 0.35 | Semi-arid NV: NDVI 0.25 is riparian/irrigated, not marginal. Tuned via grid search on the 1000-489 dry reach chain (see `notes/fac_sag_tuning.md`) |
| `distance_scale_m=4000` | Raised from 1500 | Wet influence must carry 3+ km upstream to reach headwaters in this terrain |
| `gamma=2.0` | Raised from 1.5 | Sharper sag ramp: partially-dry reaches stay shallow, only very-dry reaches go deep |
| `rmax_max_m=30` | Lowered from 60 | Tighter sag ceiling for this AOI (used 60 for initial experiments, found 30 sufficient here) |
| `smoothness_weight=3.0` | Raised from 1.0 | Stronger network smoothing to prevent reach-to-reach jitter |
| `station_spacing_m=50` | Reduced from 200 | 4x denser strips eliminate along-gradient interpolation waves in the IDW fill |
| `idw_radius_m=1000` | Raised from 200 | Wider fill radius bridges gaps between strips in wide valleys |
| `post_smooth_m=100` | New | Gaussian post-smooth feathers discontinuities at strip edges |
| `halo_n=12` | New | Extra spokes beyond strip termination improve coverage in wide valleys |

## Companion outputs (same directory)

| File | Description |
|------|-------------|
| `fac_channel_heads.fgb` | Per-reach solve results (head_depth_m, pin weights, sag targets) |
| `fac_normals_cross_sections.fgb` | All cross-section strips (for QGIS visualization) |
| `fac_normals_streams.fgb` | Input stream network (copy) |
| `fac_normals_aoi.fgb` | AOI boundary polygon |
| `fac_normals_smoothed_dem.tif` | 20m Gaussian-smoothed DEM (orientation field) |
| `fac_head_depth_sparse_5m.tif` | Sparse burn (stations only, before IDW fill) |
| `fac_rem_water_surface_5m.tif` | Strip-fill-IDW water-surface elevation (before REM subtraction) |
