# REM Algorithm — Handily

## Current Status

This note describes the current best REM model in the repo.

As of 2026-04-19, the best working path is the FAC-based workflow built around:

- [`rem_fac.py`](/home/dgketchum/code/handily/src/handily/rem_fac.py) for dense aspect-normal strip geometry and fast `20 m` water-surface priors
- [`rem_sag.py`](/home/dgketchum/code/handily/src/handily/rem_sag.py) for hard support from snapped thalwegs plus surface-water evidence
- [`rem_fac_topology.py`](/home/dgketchum/code/handily/src/handily/rem_fac_topology.py) for network-topology pin weights derived from wet lower reaches
- [`rem_surface_relax.py`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py) for the final continuous water-surface relaxation

This replaced the earlier `rem_experimental.py` face-based workflow as the best current model for `0773`.

The main reason is structural:

- the dense FAC network gives better valley-scale geometry
- the fast raster prior is operational at full AOI scale
- the final water-surface solve can now incorporate:
  - hard support where water is actually observed at the surface
  - soft pinning from vegetation and network topology
  - continuous sag away from implausibly shallow prior water tables

## Core Equation

The final REM is still:

```python
REM = max(DEM - water_surface, 0)
```

The problem is constructing `water_surface`.

The current method treats it as a two-stage problem:

1. build a physically plausible but too-shallow prior from dense FAC geometry
2. relax that prior upward only where evidence and topology support shallow groundwater, while allowing unsupported areas to sag downward continuously

## Inputs

The current implementation uses:

- DEM:
  - `dem_bounds_1m.tif`
- dense FAC stream network:
  - `streams_fac.fgb`
- lower-density snapped thalwegs:
  - `experimental_full/resnapped.fgb`
- evidence-based surface-water mask:
  - `water_mask_rf.tif`
- NAIP for NDVI:
  - `naip/ortho_1-1_hm_s_nv007_2022_1.tif`

## Stage 1: Build Valley-Scale Orientation

[`build_orientation_field()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L206) coarsens and heavily smooths the DEM.

It then derives:

- downhill direction
- slope magnitude
- aspect-normal strip orientation
- AOI polygon from the DEM footprint

This produces a stable valley-scale orientation field.

The important point is that orientation comes from the smoothed terrain, not from the FAC line tangent alone.

## Stage 2: Generate Dense FAC Cross Sections

[`generate_fac_strips()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L554) builds dense cross-valley sections on the FAC network.

For each station along each FAC stream:

1. estimate local tangent
2. sample the smoothed DEM aspect-normal
3. cast a line outward along that normal
4. stop at the first hit on:
   - another FAC stream, or
   - the AOI boundary

This yields:

- `interreach` connectors when another stream is hit
- `edge` connectors when the AOI boundary is hit

At this stage there is no snapping, polygon-face ownership, or water-support gating.

## Stage 3: Attach Elevations

[`_attach_fac_strip_elevations()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L515) samples DEM elevation at:

- the base on the source stream
- the section endpoint

Each section therefore carries a linear end-to-end water-surface elevation profile.

## Stage 4: Burn A Fast 20 m Prior Water Surface

The current fast prior path is:

1. burn cross sections sparsely to `20 m`
   - [`rasterize_sparse_sections_20m()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L709)
2. fill the sparse raster
   - preferred: [`fill_sparse_sections_idw()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L883)
3. sample DEM to the same `20 m` grid
4. compute the prior REM from the filled water surface

This prior is useful because it is fast and geometrically realistic, but it is often too shallow in arid areas where the dense FAC network overstates plausible shallow groundwater.

## Stage 5: Build Hard Support

[`build_stream_evidence_support()`](/home/dgketchum/code/handily/src/handily/rem_sag.py) builds the hard support mask where:

- snapped thalwegs overlap the evidence-based water mask

Only these cells are allowed to end up at:

- `water_surface = DEM`
- `REM = 0`

Everywhere else the final REM must remain above a minimum off-support clearance.

## Stage 6: Build Topology-Derived Pin Weights

This is the current standout improvement.

[`build_fac_topology_pin_weights()`](/home/dgketchum/code/handily/src/handily/rem_fac_topology.py) does four things:

1. orient the FAC network upstream-to-downstream from endpoint elevations on the smoothed DEM
2. build upstream/downstream adjacency from shared segment endpoints
3. estimate wet seed strength on each FAC reach from:
   - high NDVI along the reach
   - hard support overlap as an override
4. propagate that wet influence upstream with exponential decay in:
   - network distance
   - elevation gain
   - optional Strahler-scaled persistence

This gives a reach-scale topology pin weight:

- lower wet reaches keep strong shallow-water influence
- upper dry benches lose that influence with distance and elevation above the last wet reach

That is the right conceptual model for the current remaining error.

## Stage 7: Relax The Water Surface

The current best solver is:

- [`relax_water_surface_ndvi_pins()`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py#L604)

Inputs:

- prior water surface from the FAC `20 m` raster
- DEM at `20 m`
- hard support mask
- topology-derived prior pin-weight raster

The solver enforces:

- hard support:
  - `water_surface = DEM` on supported snapped-thalweg cells
- hard minimum clearance elsewhere:
  - `REM >= min_clearance_off_support`
  - currently `0.1 m`
- soft preservation of the prior:
  - stronger where topology weight is high
  - weaker where topology weight is low
- spatial continuity:
  - controlled by `smoothness_weight`

This is the current conceptual model:

- the dense FAC raster is only a prior
- shallow water is defended only where downstream wet evidence and network topology justify it
- the rest of the surface is allowed to sag continuously away from that support

## Key Tuning Controls

There are three groups of parameters that matter now.

### Geometry

In [`rem_fac.py`](/home/dgketchum/code/handily/src/handily/rem_fac.py):

- `station_spacing_m`
- `smooth_sigma_m`
- sparse-fill parameters such as IDW radius and power

These control the prior geometry.

### Topology

In [`rem_fac_topology.py`](/home/dgketchum/code/handily/src/handily/rem_fac_topology.py):

- `ndvi_mid`, `ndvi_scale`
  - how reaches become wet seeds
- `distance_scale_m`
  - how far wet influence persists upstream
- `elevation_scale_m`
  - how quickly benches detach with gain above wet reaches
- `strahler_distance_scale`
  - how much larger channels retain influence longer

These control where shallow groundwater remains plausible on the network.

### Relaxation

In [`rem_surface_relax.py`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py):

- `base_fidelity`
  - global attachment to the prior
- `smoothness_weight`
  - continuity / gradient control
- `min_clearance_off_support`
  - hard off-support minimum REM

These control how gradual the final sag is.

## Outputs

The important current products are:

- geometry:
  - `fac_normals_cross_sections.fgb`
  - `fac_normals_wedges.fgb`
- prior rasters:
  - `fac_normals_idw_fill_20m.tif`
  - `fac_normals_idw_fill_rem_20m.tif`
- hard support:
  - `fac_normals_ws_relax_support_20m.tif`
- topology guidance:
  - `fac_topology_pin_streams.fgb`
  - `fac_topology_pin_weight_20m.tif`
- final relaxed products:
  - `fac_normals_idw_fill_ws_20m_relaxed_fac_topology.tif`
  - `fac_normals_idw_fill_rem_20m_from_ws_relax_fac_topology.tif`

## Current Best Standpoint

The current best FAC REM path is:

1. dense FAC aspect-normal strip generation
2. fast `20 m` IDW water-surface prior
3. hard support from snapped thalwegs plus evidence
4. topology-derived prior pinning from wet lower reaches
5. continuous water-surface relaxation with a tunable smoothness term

This is the first version that ties shallow groundwater plausibility to:

- observed wet conditions
- downstream network position
- elevation above the last wet reach

instead of only to local geometry or local NDVI.
