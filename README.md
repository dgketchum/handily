# handily

`handily` is a geospatial workflow for deriving REM/HAND-style terrain products, classifying irrigated
fields by stream context, joining climate and ET data, and partitioning ET into precipitation,
groundwater/soil-moisture, and irrigation components.

The main documentation lives in [`docs/`](docs/index.md). Beaverhead remains the reference AOI for
development, examples, and notebooks.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .
```

## Canonical CLI Path

```bash
handily stac build --out-dir ~/data/IrrigationGIS/handily/stac/3dep_1m --states MT
handily bounds \
  --bounds -112.418 45.445 -112.353 45.49 \
  --fields ~/data/IrrigationGIS/Montana/statewide_irrigation_dataset/statewide_irrigation_dataset_15FEB2024.shp \
  --ndwi-dir ~/data/IrrigationGIS/handily/ndwi/beaverhead \
  --flowlines-local-dir ~/data/IrrigationGIS/boundaries/wbd/NHD_H_Montana_State_Shape/Shape \
  --stac-dir ~/data/IrrigationGIS/handily/stac/3dep_1m \
  --out-dir ~/data/IrrigationGIS/handily/handily/beaverhead/outputs
```

Further stages use a TOML config:

```bash
handily met download --config examples/beaverhead/beaverhead_config.toml
handily et export --config examples/beaverhead/beaverhead_config.toml
handily et join --config examples/beaverhead/beaverhead_config.toml
handily partition --config examples/beaverhead/beaverhead_config.toml
handily qgis qlr --config examples/beaverhead/beaverhead_config.toml
```

## Entry Points

- Main docs: [`docs/index.md`](docs/index.md)
- CLI reference: [`docs/cli_reference.md`](docs/cli_reference.md)
- Beaverhead reference: [`docs/examples/beaverhead.md`](docs/examples/beaverhead.md)
- Beaverhead notebooks: [`examples/beaverhead/`](examples/beaverhead/)
- Examples guide: [`examples/README.md`](examples/README.md)

## Local Docs Build

```bash
pip install -r docs/requirements.txt
mkdocs serve
```
