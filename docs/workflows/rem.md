# REM Workflow

This workflow assembles a local DEM mosaic, combines NHD and NDWI into a stream mask, computes REM,
and summarizes REM over fields.

## Inputs

| Input | Source |
| --- | --- |
| field polygons | `fields_path` |
| AOI bounds | `--bounds` or `config.bounds` |
| local NHD shapefiles | `flowlines_local_dir` |
| local NDWI rasters | `ndwi_dir` |
| local 3DEP STAC | `stac_dir` |

## Main Modules

- `handily.pipeline.REMWorkflow`
- `handily.dem`
- `handily.compute`
- `handily.io`

## Command

```bash
handily bounds \
  --bounds -112.418 45.445 -112.353 45.49 \
  --fields /nas/Montana/statewide_irrigation_dataset/statewide_irrigation_dataset_15FEB2024.shp \
  --ndwi-dir /nas/handily/ndwi/beaverhead \
  --flowlines-local-dir /nas/boundaries/wbd/NHD_H_Montana_State_Shape/Shape \
  --stac-dir /nas/handily/stac/3dep_1m \
  --out-dir /nas/handily/handily/beaverhead/outputs
```

## Outputs

- `rem_bounds.tif`
- `streams_bounds.tif`
- `fields_bounds.fgb`

The Beaverhead script also persists:
- `flowlines_bounds.fgb`

## What Happens

1. Bounds are converted into an AOI polygon.
2. Flowlines are loaded from local NHD data.
3. DEM tiles intersecting the AOI are resolved from the local STAC.
4. NDWI rasters are merged and thresholded.
5. NHD and NDWI are combined into a stream mask.
6. REM is computed from DEM and stream mask.
7. Mean REM is computed for each field polygon.

## Failure Modes

- Missing or incomplete local STAC catalog.
- AOI outside the NDWI coverage directory.
- Flowline alignment poor enough that `--flowlines-buffer` is needed.
- Output directory already contains stale rasters and vectors from another AOI.
