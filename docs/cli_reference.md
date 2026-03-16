# CLI Reference

This page documents the current CLI surface in `src/handily/cli.py`. Commands not listed here are
either example-script-only or not currently documented as supported.

## Global

```bash
handily -v <command> ...
```

`-v` raises the log level to `DEBUG`.

## `bounds`

Build REM products within explicit bounds using local NHD flowlines, NDWI rasters, and a local 3DEP
STAC catalog.

```bash
handily bounds \
  --bounds W S E N \
  --fields /path/to/fields.shp \
  --ndwi-dir /path/to/ndwi \
  --flowlines-local-dir /path/to/nhd/Shape \
  --stac-dir /path/to/stac \
  --out-dir /path/to/output
```

Important flags:

| Flag | Meaning |
| --- | --- |
| `--ndwi-threshold` | NDWI cutoff for stream masking. Default `0.15`. |
| `--flowlines-buffer` | Optional buffer in meters before rasterizing flowlines. |

Writes:
- `rem_bounds.tif`
- `streams_bounds.tif`
- `fields_bounds.fgb`

## `aoi`

Build centroid-buffer AOI tiles from field polygons.

```bash
handily aoi \
  --fields /path/to/fields.shp \
  --out-shp /path/to/aoi_tiles.shp \
  --max-km2 625 \
  --buffer-m 1000
```

Writes:
- shapefile at `--out-shp`

## `stac build`

Build a local STAC catalog for USGS 3DEP 1 m DEM tiles.

```bash
handily stac build --out-dir /path/to/stac/3dep_1m --states MT ID
```

Writes:
- `catalog.json`
- collection metadata
- bbox index files used for AOI lookup

## `met download`

Download GridMET daily time series for each field or for the nearest configured GridMET centroid.

```bash
handily met download --config examples/beaverhead/beaverhead_config.toml
```

Writes:
- one parquet per field under `gridmet_parquet_dir`
- optional centroid cache under `gridmet_centroid_parquet_dir`

## `et export`

Launch PT-JPL ET fraction zonal exports to Google Cloud Storage.

```bash
handily et export --config examples/beaverhead/beaverhead_config.toml
```

Behavior:
- initializes Earth Engine
- iterates fields in the configured AOI
- creates one async export task per field

Writes:
- remote CSV exports under `gs://{et_bucket}/ptjpl_tables/etf_zonal/<FID>/`

Manual wait point:
- downstream sync or join must wait for Earth Engine tasks to finish

## `et join`

Join local PT-JPL CSV tables with field-level GridMET parquet and interpolate ETf to daily cadence.

```bash
handily et join --config examples/beaverhead/beaverhead_config.toml
```

Writes:
- one joined parquet per field under `et_join_parquet_dir`

## `partition`

Partition joined ET into precipitation-effective ET, groundwater/soil-moisture ET, and irrigation ET.

```bash
handily partition --config examples/beaverhead/beaverhead_config.toml
```

Writes:
- one monthly parquet per field under `partition_out_parquet_dir`

Requires:
- fields with `strata` and `pattern` columns
- joined parquet at `partition_joined_parquet_dir`

## `sync`

Mirror a selected bucket subdirectory into the configured local mirror tree.

```bash
handily sync --config examples/beaverhead/beaverhead_config.toml --subdir irrmapper
```

Important flags:

| Flag | Meaning |
| --- | --- |
| `--subdir` | Bucket subdirectory below `{bucket_prefix}/{project_name}`. Default `irrmapper`. |
| `--glob` | Filename filter. Default `*`. |
| `--overwrite` | Replace existing local files. |
| `--dry-run` | Print candidate files without copying. |

Current constraint:
- this command matches the mirrored `handily/<project>/<subdir>/...` layout
- PT-JPL ET exports use `ptjpl_tables/etf_zonal/...` and are typically synced through the Beaverhead example helpers instead of the generic CLI

## `qgis update`

Discover outputs in `out_dir` and add them into an existing `.qgs` project.

```bash
handily qgis update \
  --config examples/beaverhead/beaverhead_config.toml \
  --project ~/data/IrrigationGIS/handily_debug.qgs
```

Notes:
- `update` currently requires a `.qgs` project file
- discovered layers include rasters and common vector outputs in `out_dir`

## `qgis qlr`

Generate a QGIS layer definition file from discovered outputs.

```bash
handily qgis qlr --config examples/beaverhead/beaverhead_config.toml
```

Writes:
- `<out_dir>/handily.qlr` unless `--output` is supplied

## `qgis open`

Open QGIS with a project path.

```bash
handily qgis open --project ~/data/IrrigationGIS/handily_debug.qgz
```

## Beaverhead Sequence

```bash
handily stac build --out-dir ~/data/IrrigationGIS/handily/stac/3dep_1m --states MT
handily bounds \
  --bounds -112.418 45.445 -112.353 45.49 \
  --fields ~/data/IrrigationGIS/Montana/statewide_irrigation_dataset/statewide_irrigation_dataset_15FEB2024.shp \
  --ndwi-dir ~/data/IrrigationGIS/handily/ndwi/beaverhead \
  --flowlines-local-dir ~/data/IrrigationGIS/boundaries/wbd/NHD_H_Montana_State_Shape/Shape \
  --stac-dir ~/data/IrrigationGIS/handily/stac/3dep_1m \
  --out-dir ~/data/IrrigationGIS/handily/handily/beaverhead/outputs

python examples/beaverhead/beaverhead.py --step stratify examples/beaverhead/beaverhead_config.toml
python examples/beaverhead/beaverhead.py --step pattern examples/beaverhead/beaverhead_config.toml
handily met download --config examples/beaverhead/beaverhead_config.toml
handily et export --config examples/beaverhead/beaverhead_config.toml
handily et join --config examples/beaverhead/beaverhead_config.toml
handily partition --config examples/beaverhead/beaverhead_config.toml
handily qgis qlr --config examples/beaverhead/beaverhead_config.toml
```
