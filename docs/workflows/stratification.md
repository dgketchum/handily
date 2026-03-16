# Stratification

Stratification attaches stream context to each field and creates the categories used later by ET
partitioning.

## Inputs

| Input | Source |
| --- | --- |
| `fields_bounds.fgb` or in-memory fields | REM workflow |
| `flowlines_bounds.fgb` or in-memory flowlines | REM workflow |
| REM raster | `rem_bounds.tif` |
| IrrMapper irrigation-frequency CSV | optional, for donor selection |

## Main Modules

- `handily.nhd`
- `handily.stratify`
- `handily.pattern`
- `handily.et.irrmapper`

## Current Entry Points

The package CLI does not currently expose stratification or pattern selection as first-class commands.
The supported reference path is the Beaverhead script:

```bash
python examples/beaverhead/beaverhead.py --step stratify examples/beaverhead/beaverhead_config.toml
python examples/beaverhead/beaverhead.py --step pattern examples/beaverhead/beaverhead_config.toml
```

## What Happens

1. Flowlines are classified into stream categories.
2. Each field receives a nearest-stream type and nearest-stream distance.
3. REM summary values are combined with stream type.
4. A `strata` label is assigned.
5. Optional IrrMapper summaries are joined.
6. Low-irrigation donor candidates are flagged in `pattern`.

## Expected Columns

Stratified fields typically include:
- `rem_mean`
- `stream_category`
- `nearest_dist`
- `strata`

Pattern outputs add:
- `irr_count`
- `irr_freq`
- `irr_mean`
- `pattern`

## Operational Notes

- `non_partitioned` fields are carried for context but excluded by `partition_et`.
- Pattern donors are selected within strata, so missing donors in a stratum will block partitioning.
- IrrMapper export and sync remain example-script workflows rather than top-level CLI commands.
