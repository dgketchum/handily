# STAC

`handily` expects a local STAC catalog for USGS 3DEP 1 m DEM tiles. The REM workflow does not query
TNM directly at run time.

## Build

```bash
handily stac build --out-dir ~/data/IrrigationGIS/handily/stac/3dep_1m --states MT
```

This scans TNM project metadata, creates a STAC collection, and writes a bbox index for faster AOI
intersection.

## Result

Expected files under the STAC root:
- `catalog.json`
- collection metadata
- item JSON files
- bbox index artifacts written by `handily.stac_3dep`

## Operational Model

- build once for a state or project set
- reuse for repeated AOI runs
- update by rebuilding if the local catalog is stale

## Notes

- The CLI defines a `stac extend` path, but the current codebase does not provide a working
  implementation. This page documents only the validated `build` workflow.
- The builder uses TNM S3 metadata XMLs and stores GeoTIFF and metadata asset links in the STAC items.
