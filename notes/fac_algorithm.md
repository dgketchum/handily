**FAC Algorithm**

This note describes the current best FAC-based REM workflow in [`rem_fac.py`](/home/dgketchum/code/handily/src/handily/rem_fac.py), the topology weighting in [`rem_fac_topology.py`](/home/dgketchum/code/handily/src/handily/rem_fac_topology.py), and the follow-on water-surface relaxation in [`rem_surface_relax.py`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py).

The core idea is:

- use a dense flow-accumulation stream network as a geometric scaffold
- orient cross sections normal to the aspect of a heavily smoothed DEM
- linearly interpolate water-surface elevation along each cross section from its two ends
- burn those sections to a `20 m` raster and fill the gaps quickly
- then relax that water surface upward at evidence-supported snapped thalwegs, while allowing upper dry reaches to sag away from the last downstream wet reaches through a topology-weighted prior pinning field

**Inputs**

- DEM at native resolution:
  - e.g. `/data/ssd2/handily/nv/aoi_0773/dem_bounds_1m.tif`
- dense FAC stream network:
  - e.g. `/data/ssd2/handily/nv/aoi_0773/streams_fac.fgb`
- lower-density snapped thalwegs for hard support:
  - e.g. `/data/ssd2/handily/nv/aoi_0773/experimental_full/resnapped.fgb`
- evidence-based water mask for hard support:
  - e.g. `/data/ssd2/handily/nv/aoi_0773/water_mask_rf.tif`
- NAIP image for NDVI-derived wet-seed detection:
  - e.g. `/data/ssd2/handily/nv/aoi_0773/naip/ortho_1-1_hm_s_nv007_2022_1.tif`

**Stage 1: Build The Orientation Field**

[`build_orientation_field()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L206) does the valley-scale preprocessing.

1. Coarsen the DEM to a working grid.
2. Apply a heavy Gaussian smooth, currently on the order of `500 m`.
3. Compute downhill vectors from the smoothed surface.
4. Convert the downhill vector to an aspect-normal field.
5. Derive an AOI polygon from the DEM footprint.
6. Build interpolators for:
   - downhill `x`
   - downhill `y`
   - slope magnitude

The result is a stable orientation field that reflects broad valley form rather than noisy channel-scale topography.

**Stage 2: Generate FAC Cross Sections**

[`generate_fac_strips()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L554) builds one connector per station and side.

For each FAC stream:

1. Station the line at a fixed spacing.
   - The current dense production run used `20 m`.
2. Estimate local reach tangent.
3. Sample the smoothed DEM aspect-normal at the station.
4. Use that normal as the strip direction.
5. Cast a ray outward until the first hit on:
   - another FAC stream, or
   - the AOI boundary
6. Record the connector as:
   - `interreach` if another stream is hit first
   - `edge` if the AOI boundary is hit first

This path deliberately avoids:

- reach snapping
- water-support gating
- polygonization / closure-edge construction
- ridge-stop side logic

The output is a dense set of cross-valley sections driven entirely by valley-scale orientation and first-hit geometry.

**Stage 3: Attach Endpoint Elevations**

[`_attach_fac_strip_elevations()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L515) samples DEM elevation at:

- the station base on the source stream
- the section endpoint at the opposing stream or AOI edge

The section water-surface elevation is then treated as linear from base elevation to endpoint elevation.

Interpretation:

- `interreach` strips interpolate between source-stream and target-stream elevations
- `edge` strips stay flat only if their endpoint elevation equals the source elevation; otherwise they still carry the linear end-to-end elevation implied by the DEM sample

**Stage 4: Build Wedges**

[`build_fac_wedges()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L660) groups adjacent cross sections into strip polygons.

These wedges are useful for:

- debugging section coherence
- inspecting overlap structure
- later ownership experiments

They are not currently required for the fast raster path.

**Stage 5: Burn Sparse Section Elevations To 20 m**

[`rasterize_sparse_sections_20m()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L709) rasterizes only the section lines, not the wedges.

For each cross section:

1. Sample points along the line.
2. Linearly interpolate elevation from base to endpoint.
3. Burn those sampled elevations into two accumulators:
   - `sum_wz`
   - `sum_w`
4. Average where multiple sections hit the same raster cell.

This produces:

- a sparse `20 m` section-elevation raster
- a count raster showing overlap density

This step is fast. The expensive part is section generation, not the burn.

**Stage 6: Fill Sparse Sections**

Three fast fill options are implemented:

- nearest:
  - [`fill_sparse_sections_nearest()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L774)
- normalized Gaussian:
  - [`fill_sparse_sections_gaussian()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L826)
- finite-radius IDW:
  - [`fill_sparse_sections_idw()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L883)

The current preferred prior for relaxation is the IDW-filled water surface:

- `fac_normals_idw_fill_20m.tif`

This gives a complete, smooth-enough water-surface prior without trying to resolve thousands of overlapping wedges explicitly.

**Stage 7: Convert Prior Water Surface To Prior REM**

[`rem_from_water_surface()`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py#L461) computes:

- `REM = max(DEM - water_surface, 0)`

This prior REM is often too shallow because the dense FAC network overstates where groundwater should be near the surface.

**Stage 8: Build Hard Support From Snapped Thalwegs**

[`build_stream_evidence_support()`](/home/dgketchum/code/handily/src/handily/rem_sag.py) builds hard support pixels where:

- the lower-density snapped-thalweg network overlaps the evidence-based surface-water mask

These are the only pixels where the relaxed solution is allowed to reach:

- `water_surface = DEM`
- `REM = 0`

Everything else must stay above that by at least a small clearance.

**Stage 9: Build Topology-Derived Prior Pin Weights**

The current best formulation no longer uses local pixel NDVI directly as the pin-weight field.

Instead:

1. Compute raw NAIP NDVI on the `20 m` grid with [`compute_naip_ndvi_match()`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py#L63).
2. Build FAC topology and orient reaches upstream-to-downstream from smoothed-DEM endpoint elevations:
   - [`build_fac_topology()`](/home/dgketchum/code/handily/src/handily/rem_fac_topology.py)
3. Estimate wet seed strength per FAC reach from:
   - high NDVI along the reach
   - hard support overlap as an override
   - [`estimate_reach_seed_strength()`](/home/dgketchum/code/handily/src/handily/rem_fac_topology.py)
4. Propagate that wet influence upstream with exponential decay in:
   - network distance
   - elevation gain
   - Strahler-scaled persistence
   - [`propagate_upstream_wet_influence()`](/home/dgketchum/code/handily/src/handily/rem_fac_topology.py)
5. Rasterize the resulting reach weights back to the `20 m` grid:
   - [`rasterize_reach_weights_max()`](/home/dgketchum/code/handily/src/handily/rem_fac_topology.py)

Interpretation:

- lower wet reaches defend shallow groundwater strongly
- their influence persists upstream for some distance
- that influence decays quickly with elevation gain onto dry benches
- the final raster is a reach-topology pinning field, not a local NDVI blur

This is the right abstraction for the current problem.

**Stage 10: Relax The Water Surface**

[`relax_water_surface_ndvi_pins()`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py#L604) is the current best relaxation model.

It solves in water-surface space, not REM space.

Constraints:

- hard pin:
  - `water_surface = DEM` on support pixels
- hard off-support floor:
  - `REM >= min_clearance_off_support`
  - currently `0.1 m`
- soft prior preservation:
  - stronger where topology-derived wet influence is high
  - weaker where reaches are far and high above the last wet downstream seed
- smoothness:
  - controlled by `smoothness_weight`

This is the conceptual model:

- the FAC water surface is the prior
- wet lower reaches resist change
- upper dry benches lose that support with network distance and elevation gain
- the sag is imposed by the membrane solve, not by a hard local ceiling

**Outputs**

The current workflow writes:

- geometry/debug:
  - `fac_normals_cross_sections.fgb`
  - `fac_normals_wedges.fgb`
- sparse/fill rasters:
  - `fac_normals_sparse_sections_20m.tif`
  - `fac_normals_sparse_sections_count_20m.tif`
  - `fac_normals_nearest_fill_20m.tif`
  - `fac_normals_gaussian_fill_20m.tif`
  - `fac_normals_idw_fill_20m.tif`
- relaxation inputs:
  - `fac_normals_naip_ndvi_20m.tif`
  - `fac_topology_pin_streams.fgb`
  - `fac_topology_pin_weight_20m.tif`
- final relaxed products:
  - `fac_normals_idw_fill_ws_20m_relaxed_fac_topology.tif`
  - `fac_normals_idw_fill_rem_20m_from_ws_relax_fac_topology.tif`

**Current Best Standpoint**

The best current FAC REM path is:

1. dense FAC cross sections from aspect normals
2. sparse line burn to `20 m`
3. fast IDW water-surface fill
4. hard thalweg support from snapped reaches + evidence mask
5. topology-derived prior pinning from wet lower reaches
6. continuous water-surface relaxation with a tunable smoothness weight

That keeps the geometric realism of the dense FAC scaffold while letting implausibly shallow prior REM sag continuously away from dry upper reaches that are far and high above the last defended wet reach.
