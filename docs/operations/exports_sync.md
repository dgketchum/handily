# Exports and Sync

Earth Engine exports and bucket mirroring are separate steps. `handily` treats them as an
asynchronous boundary.

## IrrMapper

Reference path:

```bash
python examples/beaverhead/beaverhead.py --step irrmapper examples/beaverhead/beaverhead_config.toml
handily sync --config examples/beaverhead/beaverhead_config.toml --subdir irrmapper
```

Expected remote path:

```text
gs://{et_bucket}/{bucket_prefix}/{project_name}/irrmapper/
```

Expected local path:

```text
{local_data_root}/{bucket_prefix}/{project_name}/irrmapper/
```

## PT-JPL

Reference export path:

```bash
handily et export --config examples/beaverhead/beaverhead_config.toml
```

Remote layout:

```text
gs://{et_bucket}/ptjpl_tables/etf_zonal/<FID>/
```

Current constraint:
- the generic `handily sync` command mirrors the configured project prefix
- PT-JPL exports do not use that prefix
- PT-JPL retrieval is therefore handled by the Beaverhead helper path rather than the generic CLI

## Operational Rules

- submit exports first
- wait for completion in Earth Engine or GCS
- sync or copy the resulting CSV tables locally
- run join or pattern-selection stages only after the local files exist
