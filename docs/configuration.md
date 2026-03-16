# Configuration

`HandilyConfig` is a flat TOML-backed dataclass. The CLI uses it for all config-driven stages.

Required keys:

| Key | Meaning |
| --- | --- |
| `out_dir` | Output directory for rasters, vectors, and QGIS artifacts |
| `flowlines_local_dir` | Local NHD shapefile directory |
| `ndwi_dir` | Directory containing NDWI rasters |
| `stac_dir` | Local 3DEP STAC root |
| `fields_path` | Source field polygons |

## Project and Mirror

| Key | Meaning |
| --- | --- |
| `project_name` | Project namespace under the local mirror and bucket prefix |
| `bucket_prefix` | Prefix below the bucket root. Default `handily` |
| `local_data_root` | Local mirror root |
| `et_bucket` | GCS bucket for ET and IrrMapper exports |

## AOI and Identity

| Key | Meaning |
| --- | --- |
| `bounds` | `[W, S, E, N]` AOI bounds in EPSG:4326 |
| `feature_id` | Field identifier column. Default `FID` |

## REM Workflow

| Key | Meaning |
| --- | --- |
| `ndwi_threshold` | NDWI cutoff used for stream masking |
| `flowlines_buffer_m` | Optional NHD buffer before rasterization |
| `rem_threshold` | REM threshold used for stratification |

## IrrMapper and Pattern

| Key | Meaning |
| --- | --- |
| `ee_fields_asset` | Optional Earth Engine asset path for fields |
| `irrmapper_csv` | Local IrrMapper irrigation-frequency CSV |

## GridMET

| Key | Meaning |
| --- | --- |
| `met_start` | Start date for GridMET download |
| `met_end` | End date for GridMET download |
| `gridmet_parquet_dir` | Field-level GridMET parquet directory |
| `gridmet_centroids_path` | GridMET centroid shapefile |
| `gridmet_centroid_parquet_dir` | Optional centroid cache directory |
| `gridmet_id_col` | GridMET centroid identifier column. Default `GFID` |

## PT-JPL

| Key | Meaning |
| --- | --- |
| `ptjpl_start_yr` | First export year |
| `ptjpl_end_yr` | Last export year |
| `ptjpl_check_dir` | Local directory checked before submitting export tasks |
| `ptjpl_csv_dir` | Local PT-JPL CSV root |
| `ptjpl_csv_template` | Per-field PT-JPL CSV path template |
| `et_join_parquet_dir` | Joined ET output directory |

## Partition

| Key | Meaning |
| --- | --- |
| `partition_joined_parquet_dir` | Input directory for joined daily parquet |
| `partition_out_parquet_dir` | Output directory for monthly partition parquet |
| `partition_strata_col` | Stratification column name. Default `strata` |
| `partition_pattern_col` | Pattern donor flag column name. Default `pattern` |

## QGIS

| Key | Meaning |
| --- | --- |
| `qgis_project` | `.qgs` project path for update workflow |
| `qgis_layer_group` | Target group name in the project |
| `qgis_view_root` | Optional path prefix remap for another machine |

## Minimal Example

```toml
project_name = "beaverhead"
bucket_prefix = "handily"
local_data_root = "~/data/IrrigationGIS/handily"

fields_path = "~/data/IrrigationGIS/Montana/statewide_irrigation_dataset/statewide_irrigation_dataset_15FEB2024.shp"
flowlines_local_dir = "~/data/IrrigationGIS/boundaries/wbd/NHD_H_Montana_State_Shape/Shape"
ndwi_dir = "~/data/IrrigationGIS/handily/ndwi/beaverhead/"
stac_dir = "~/data/IrrigationGIS/handily/stac/3dep_1m/"
out_dir = "~/data/IrrigationGIS/handily/handily/beaverhead/outputs/"
bounds = [-112.418, 45.445, -112.353, 45.49]
feature_id = "FID"
```

## Beaverhead Example

The full reference config lives at `examples/beaverhead/beaverhead_config.toml`.

Not every `run_*` flag in that file is consumed by the package CLI. Some are used only by the
Beaverhead example script.
