# Meteorology and ET

This stage creates daily meteorology by field, exports sparse PT-JPL ET fraction observations, and
joins both on a daily index.

## GridMET

Entry point:

```bash
handily met download --config examples/beaverhead/beaverhead_config.toml
```

Behavior:
- reads fields from `fields_path`
- optionally assigns fields to the nearest GridMET centroid
- downloads daily GridMET records
- writes one parquet per field

Core output columns:
- `eto`
- `prcp`
- `date`
- `year`
- `month`
- `day`

## PT-JPL Export

Entry point:

```bash
handily et export --config examples/beaverhead/beaverhead_config.toml
```

Behavior:
- initializes Earth Engine
- builds a PT-JPL collection over each field polygon
- computes mean `et_fraction` for each overpass image
- submits async table exports to GCS

Remote output path:

```text
gs://{et_bucket}/ptjpl_tables/etf_zonal/<FID>/ptjpl_etf_zonal_<FID>_<start>_<end>.csv
```

Manual wait point:
- exports are asynchronous
- downstream local sync and join must wait for task completion

## Join

Entry point:

```bash
handily et join --config examples/beaverhead/beaverhead_config.toml
```

Behavior:
- reads one GridMET parquet per field
- reads the corresponding PT-JPL CSV
- inserts sparse observed `ptjpl_etf`
- interpolates to `ptjpl_etf_interp`
- computes `aet = ptjpl_etf_interp * eto`

Output:
- one joined parquet per field in `et_join_parquet_dir`

## Main Modules

- `handily.et.gridmet`
- `handily.et.image_export`
- `handily.et.join`
- `handily.bucket`

## Constraints

- `et export` requires working Earth Engine credentials and the `openet` PT-JPL dependency.
- The generic `handily sync` command mirrors only the configured project prefix and is not the full
  PT-JPL export retrieval path.
- `join` expects local PT-JPL CSVs to already exist in the configured template path.
