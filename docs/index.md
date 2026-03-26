# handily

`handily` is a pipeline for deriving terrain context from 3DEP DEMs, attaching stream structure to
field polygons, acquiring climate and ET inputs, and partitioning ET into component fluxes.

The reference deployment is Beaverhead. The package itself is broader than the notebooks: it also
includes STAC build tooling, AOI tiling, bucket mirroring, and QGIS integration.

## Scope

| Capability | What it does | Primary entry point |
| --- | --- | --- |
| 3DEP STAC | Build a local STAC for USGS 1 m DEM tiles | `handily stac build` |
| AOI tiling | Create buffered AOI tiles from field polygons | `handily aoi` |
| REM/HAND | Build DEM, stream mask, REM, and field stats for an AOI | `handily bounds` |
| Stratification | Classify fields by REM and nearest-stream type | `examples/beaverhead/beaverhead.py` |
| Pattern selection | Mark likely groundwater/soil-moisture donor fields | `examples/beaverhead/beaverhead.py` |
| GridMET | Download centroid-based meteorology | `handily met download` |
| PT-JPL export | Launch capture-date ETf zonal exports to GCS | `handily et export` |
| PT-JPL join | Interpolate ETf onto daily GridMET records | `handily et join` |
| ET partition | Partition AET into `pe`, `et_gwsm`, and `et_irr` | `handily partition` |
| Bucket sync | Mirror selected export files to local storage | `handily sync` |
| QGIS integration | Discover outputs and add them to a project or QLR | `handily qgis ...` |

## Quick Links

- Architecture: [overview.md](overview.md)
- Commands: [cli_reference.md](cli_reference.md)
- Config schema: [configuration.md](configuration.md)
- Artifact map: [outputs.md](outputs.md)
- Beaverhead reference: [examples/beaverhead.md](examples/beaverhead.md)

## Canonical Sequence

```bash
handily stac build --out-dir /nas/handily/stac/3dep_1m --states MT
handily bounds \
  --bounds -112.418 45.445 -112.353 45.49 \
  --fields /nas/Montana/statewide_irrigation_dataset/statewide_irrigation_dataset_15FEB2024.shp \
  --ndwi-dir /nas/handily/ndwi/beaverhead \
  --flowlines-local-dir /nas/boundaries/wbd/NHD_H_Montana_State_Shape/Shape \
  --stac-dir /nas/handily/stac/3dep_1m \
  --out-dir /nas/handily/handily/beaverhead/outputs

handily met download --config examples/beaverhead/beaverhead_config.toml
handily et export --config examples/beaverhead/beaverhead_config.toml
handily et join --config examples/beaverhead/beaverhead_config.toml
handily partition --config examples/beaverhead/beaverhead_config.toml
handily qgis qlr --config examples/beaverhead/beaverhead_config.toml
```

## Boundary Conditions

- Main docs cover architecture, CLI behavior, configuration, and file contracts.
- Beaverhead notebooks remain the visual and pedagogical layer.
- Not all implemented workflows are currently exposed as top-level CLI commands.
- Earth Engine export steps are asynchronous and require manual waiting before downstream sync or join.
