# Beaverhead

Beaverhead is the reference AOI for development. It is where the workflow is exercised end to end,
where the notebooks are organized, and where the example config is maintained.

## Files

| File | Role |
| --- | --- |
| `examples/beaverhead/beaverhead.py` | Example orchestration script and dev runner |
| `examples/beaverhead/beaverhead_config.toml` | Reference config |
| `examples/beaverhead/01_introduction.ipynb` | Intro and scope |
| `examples/beaverhead/02_terrain_analysis.ipynb` | REM and terrain workflow |
| `examples/beaverhead/03_field_classification.ipynb` | Stratification and pattern context |
| `examples/beaverhead/04_climate_and_et.ipynb` | GridMET and PT-JPL join |
| `examples/beaverhead/05_et_partitioning.ipynb` | Partition outputs |

## Reference Sequence

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
python examples/beaverhead/beaverhead.py --step irrmapper examples/beaverhead/beaverhead_config.toml
python examples/beaverhead/beaverhead.py --step pattern examples/beaverhead/beaverhead_config.toml
handily met download --config examples/beaverhead/beaverhead_config.toml
handily et export --config examples/beaverhead/beaverhead_config.toml
handily et join --config examples/beaverhead/beaverhead_config.toml
handily partition --config examples/beaverhead/beaverhead_config.toml
handily qgis qlr --config examples/beaverhead/beaverhead_config.toml
```

## Why Beaverhead

- known AOI with existing local inputs
- stable comparison area for iterative algorithm work
- aligned notebooks, config, and output conventions

## Division of Labor

- docs site: architecture, commands, file contracts
- example script: operational orchestration and unsupported-yet-useful steps
- notebooks: visualization and interpretation
