# QGIS

`handily.qgis` discovers output rasters and vectors in `out_dir` and either injects them into a
project or emits a `.qlr`.

## Discoverable Layer Types

Rasters:
- `.tif`
- `.tiff`
- `.vrt`

Vectors:
- `.fgb`
- `.shp`
- `.gpkg`
- `.geojson`

## Update an Existing Project

```bash
handily qgis update \
  --config examples/beaverhead/beaverhead_config.toml \
  --project ~/data/IrrigationGIS/handily_debug.qgs
```

Behavior:
- scans `out_dir`
- creates a backup at `<project>.bak`
- creates or updates the configured layer group
- adds raster and vector subgroups

Constraint:
- `update` currently requires a `.qgs` project file

## Generate a QLR

```bash
handily qgis qlr --config examples/beaverhead/beaverhead_config.toml
```

Default output:
- `<out_dir>/handily.qlr`

## Path Remapping

If `qgis_view_root` is set, discovered layer paths are remapped by replacing the local home prefix.
Use this when the workflow runs on one machine and the layers are viewed from another mount root.
