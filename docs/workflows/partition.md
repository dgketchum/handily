# Partition

Partitioning converts joined daily AET into monthly estimates of effective precipitation,
groundwater/soil-moisture ET, and irrigation ET.

## Entry Point

```bash
handily partition --config examples/beaverhead/beaverhead_config.toml
```

## Inputs

| Input | Source |
| --- | --- |
| joined daily parquet | `et_join_parquet_dir` |
| stratified fields | `fields_path` |
| `strata` column | stratification step |
| `pattern` column | pattern selection step |

## Method

1. Drop `non_partitioned` fields.
2. Build donor and recipient sets within each `strata`.
3. Assign each recipient to the nearest donor in the same stratum.
4. Compute annual `aet`, `eto`, `prcp`, `pe`, and donor `etf_gwsm`.
5. Apply donor `etf_gwsm` to recipient annual `eto`.
6. Compute annual `et_irr = aet - et_gwsm - pe`.
7. Clip negative irrigation to zero and reassign the balance to `et_gwsm`.
8. Disaggregate annual totals into monthly values using positive monthly net ET fractions.

## Outputs

- one parquet per field in `partition_out_parquet_dir`

Columns:
- `water_year`
- field identifier
- `donor_fid`
- `fdisag`
- `et_gwsm_m`
- `et_irr_m`

## Main Module

- `handily.et.partition`

## Failure Modes

- no donor fields available for a stratum
- missing joined parquet for a recipient or donor field
- fields file missing the configured `strata` or `pattern` columns
