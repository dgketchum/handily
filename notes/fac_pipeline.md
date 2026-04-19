**FAC Pipeline**

This note is the high-level call sequence for the FAC-based REM prototype and the current downstream relaxation path.

**Geometry Path**

Main entry point:

- [`main()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L955)

Call sequence:

1. Open DEM.
2. Build orientation field:
   - [`build_orientation_field()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L206)
3. Read FAC stream network.
4. Generate aspect-normal strips:
   - [`generate_fac_strips()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L554)
5. Attach base/endpoint elevations:
   - inside `generate_fac_strips()` via [`_attach_fac_strip_elevations()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L515)
6. Build wedge polygons for debugging:
   - [`build_fac_wedges()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L660)

High-level flow:

```text
DEM
  -> build_orientation_field
FAC streams
  -> generate_fac_strips
  -> _attach_fac_strip_elevations
  -> build_fac_wedges
```

**Raster Prior Path**

Still inside [`main()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L955):

1. Burn cross-section elevations to sparse `20 m` raster:
   - [`rasterize_sparse_sections_20m()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L709)
2. Sample DEM to `20 m` grid:
   - [`sample_dem_to_grid()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L750)
3. Fill sparse raster:
   - nearest: [`fill_sparse_sections_nearest()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L774)
   - Gaussian: [`fill_sparse_sections_gaussian()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L826)
   - IDW: [`fill_sparse_sections_idw()`](/home/dgketchum/code/handily/src/handily/rem_fac.py#L883)
4. Compute prior REM:
   - `DEM_20m - water_surface_20m`

High-level flow:

```text
fac strips
  -> rasterize_sparse_sections_20m
  -> sample_dem_to_grid
  -> fill_sparse_sections_idw
  -> prior water surface
  -> prior REM
```

**Hard Support Path**

Support is built outside `rem_fac.py`, using the lower-density snapped network and the evidence mask.

Call sequence:

1. Read evidence water mask and reproject to `20 m` match grid.
2. Read snapped thalwegs from:
   - `experimental_full/resnapped.fgb`
3. Build hard support pixels:
   - [`build_stream_evidence_support()`](/home/dgketchum/code/handily/src/handily/rem_sag.py)

High-level flow:

```text
water evidence mask + resnapped thalwegs
  -> build_stream_evidence_support
  -> support mask
```

**NDVI Guidance Path**

Current best guidance path uses raw NDVI only as soft prior pinning.

Call sequence:

1. Compute NAIP NDVI on the `20 m` grid:
   - [`compute_naip_ndvi_match()`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py#L63)
2. Convert NDVI to a soft prior-pin weight:
   - [`ndvi_to_prior_pin_weight()`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py#L287)

High-level flow:

```text
NAIP
  -> compute_naip_ndvi_match
  -> ndvi_to_prior_pin_weight
  -> prior pin-weight raster
```

**Relaxation Path**

Current best relaxation entry point:

- [`relax_water_surface_ndvi_pins()`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py#L604)

Inputs:

- prior water surface:
  - usually `fac_normals_idw_fill_20m.tif`
- DEM at `20 m`
- hard support mask
- NDVI-derived prior pin-weight raster

Then:

1. Solve upward-only water-surface relaxation.
2. Convert relaxed water surface back to REM:
   - [`rem_from_water_surface()`](/home/dgketchum/code/handily/src/handily/rem_surface_relax.py#L461)

High-level flow:

```text
prior water surface
+ DEM_20m
+ hard support
+ NDVI prior pin weights
  -> relax_water_surface_ndvi_pins
  -> relaxed water surface
  -> rem_from_water_surface
  -> relaxed REM
```

**End-To-End Current Best Sequence**

```text
DEM_1m
  -> build_orientation_field

FAC streams
  -> generate_fac_strips
  -> _attach_fac_strip_elevations
  -> build_fac_wedges

fac strips
  -> rasterize_sparse_sections_20m
  -> fill_sparse_sections_idw
  -> prior water surface

prior water surface + DEM_20m
  -> prior REM

water evidence mask + resnapped thalwegs
  -> build_stream_evidence_support

NAIP
  -> compute_naip_ndvi_match
  -> ndvi_to_prior_pin_weight

prior water surface + DEM_20m + support + NDVI pin weights
  -> relax_water_surface_ndvi_pins
  -> relaxed water surface
  -> rem_from_water_surface
  -> final relaxed REM
```

**Tuning Boundary**

The main tunable boundary between “geometry” and “hydrology” is:

- `rem_fac.py`
  - geometry generation
  - sparse raster prior
- `rem_surface_relax.py`
  - evidence support
  - NDVI guidance
  - continuous sag / relaxation

That separation is useful and should be kept:

- geometry can evolve independently
- the relaxed water-surface solve can be tuned without rebuilding the FAC strip logic
