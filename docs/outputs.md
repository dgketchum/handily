# Outputs

The pipeline is file-oriented. This page names the main artifacts and the stage that produces them.

## Terrain and REM

| Artifact | Producer | Meaning |
| --- | --- | --- |
| `rem_bounds.tif` | `handily bounds` | REM raster for the AOI |
| `streams_bounds.tif` | `handily bounds` | Rasterized stream mask derived from NHD and NDWI |
| `fields_bounds.fgb` | `handily bounds` | Fields clipped to the AOI with REM stats |
| `flowlines_bounds.fgb` | Beaverhead script | Clipped flowlines retained for inspection and stratification |

## Stratification and Pattern

| Artifact | Producer | Meaning |
| --- | --- | --- |
| `fields_stratified.fgb` | Beaverhead script | Fields with REM and stream-context classes |
| `fields_pattern.fgb` | Beaverhead script | Stratified fields plus donor flag from IrrMapper |
| `*_irr_freq.csv` | IrrMapper export + sync | Irrigation frequency summary by field |

## GridMET

| Artifact | Producer | Meaning |
| --- | --- | --- |
| `gridmet_parquet_dir/<FID>.parquet` | `handily met download` | Daily field meteorology |
| `gridmet_centroid_parquet_dir/<GFID>.parquet` | `handily met download` | Shared centroid cache |

Field-level GridMET parquet typically includes:
- `date`
- `year`
- `month`
- `day`
- `eto`
- `prcp`
- centroid coordinates and elevation

## PT-JPL

| Artifact | Producer | Meaning |
| --- | --- | --- |
| `gs://<bucket>/ptjpl_tables/etf_zonal/<FID>/ptjpl_etf_zonal_<FID>_<start>_<end>.csv` | `handily et export` | Sparse ET fraction observations by Landsat overpass |
| `ptjpl_tables/etf_zonal/...` in local mirror | manual or helper sync | Local copy of the exported PT-JPL CSV tables |

PT-JPL CSV tables contain:
- field identifier
- observation date
- image identifier
- `et_fraction`

## Joined ET

| Artifact | Producer | Meaning |
| --- | --- | --- |
| `et_join_parquet_dir/<FID>.parquet` | `handily et join` | Daily meteorology joined with sparse and interpolated ETf |

Joined parquet adds:
- `ptjpl_etf`
- `ptjpl_etf_interp`
- `aet`

## Partition

| Artifact | Producer | Meaning |
| --- | --- | --- |
| `partition_out_parquet_dir/<FID>.parquet` | `handily partition` | Monthly ET component estimates |

Partition parquet includes:
- `water_year`
- field identifier
- `donor_fid`
- `fdisag`
- `et_gwsm_m`
- `et_irr_m`

## QGIS

| Artifact | Producer | Meaning |
| --- | --- | --- |
| `handily.qlr` | `handily qgis qlr` | QGIS layer definition bundle for discovered outputs |
| `<project>.qgs` | `handily qgis update` | Existing project updated in place |
| `<project>.qgs.bak` | `handily qgis update` | Backup created before project mutation |

## Mirror Layout

The intended mirrored layout is:

```text
{local_data_root}/{bucket_prefix}/{project_name}/{subdir}/...
```

Example:

```text
~/data/IrrigationGIS/handily/handily/beaverhead/irrmapper/
```

PT-JPL export tables are an exception. They are written by Earth Engine under:

```text
gs://{et_bucket}/ptjpl_tables/etf_zonal/<FID>/
```

The generic `handily sync` command does not currently normalize that alternate layout.
