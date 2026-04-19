**FAC Algorithm**

This note describes the current best FAC-based REM workflow in [`rem_fac.py`](/home/dgketchum/code/handily/src/handily/rem_fac.py) and the follow-on water-surface relaxation in [`rem_surface_relax.py`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py).

The core idea is:

- use a dense flow-accumulation stream network as a geometric scaffold
- orient cross sections normal to the aspect of a heavily smoothed DEM
- linearly interpolate water-surface elevation along each cross section from its two ends
- burn those sections to a `20 m` raster and fill the gaps quickly
- then relax that water surface upward at evidence-supported snapped thalwegs, while allowing dry low-NDVI areas to sag away from implausibly shallow prior REM

**Inputs**

- DEM at native resolution:
  - e.g. `/data/ssd2/handily/nv/aoi_0773/dem_bounds_1m.tif`
- dense FAC stream network:
  - e.g. `/data/ssd2/handily/nv/aoi_0773/streams_fac.fgb`
- lower-density snapped thalwegs for hard support:
  - e.g. `/data/ssd2/handily/nv/aoi_0773/experimental_full/resnapped.fgb`
- evidence-based water mask for hard support:
  - e.g. `/data/ssd2/handily/nv/aoi_0773/water_mask_rf.tif`
- NAIP image for NDVI-derived soft pinning:
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

**Stage 9: Add Soft NDVI Pinning To The Prior**

The current best formulation does not use NDVI as a hard clearance target.

Instead:

1. Compute raw NAIP NDVI on the `20 m` grid with [`compute_naip_ndvi_match()`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py#L63).
2. Convert NDVI to a soft prior-pin weight with [`ndvi_to_prior_pin_weight()`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py#L287).

Interpretation:

- high NDVI:
  - stronger pinning to the prior shallow water surface
- low NDVI:
  - weaker pinning to the prior
  - easier for the solver to let the surface sag downward

This keeps NDVI as guidance, not as a discontinuous ceiling.

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
  - stronger where NDVI is high
  - weaker where NDVI is low
- smoothness:
  - controlled by `smoothness_weight`

This is the conceptual model:

- the FAC water surface is the prior
- wet green areas resist change
- dry low-NDVI areas can sag more
- the sag is imposed by the membrane solve, not by blurring NDVI

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
  - `fac_normals_ndvi_prior_pin_weight_20m.tif`
- final relaxed products:
  - `fac_normals_idw_fill_ws_20m_relaxed_ndvi_pin.tif`
  - `fac_normals_idw_fill_rem_20m_from_ws_relax_ndvi_pin.tif`

**Current Best Standpoint**

The best current FAC REM path is:

1. dense FAC cross sections from aspect normals
2. sparse line burn to `20 m`
3. fast IDW water-surface fill
4. hard thalweg support from snapped reaches + evidence mask
5. raw-NDVI soft prior pinning
6. continuous water-surface relaxation with a tunable smoothness weight

That keeps the geometric realism of the dense FAC scaffold while letting implausibly shallow prior REM sag continuously away from dry low-NDVI ground.
