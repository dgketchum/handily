# Handily Examples

This directory contains example configurations, scripts, and tutorial notebooks for the handily workflow.

## Directory Structure

```
examples/
├── README.md                          # This file
├── beaverhead/                        # Full workflow example
│   ├── beaverhead.py                  # Development script
│   ├── beaverhead_config.toml         # Configuration file
│   ├── 01_introduction.ipynb          # Overview and setup
│   ├── 02_terrain_analysis.ipynb      # REM workflow
│   ├── 03_field_classification.ipynb  # Stratification + IrrMapper + Pattern
│   ├── 04_climate_and_et.ipynb        # GridMET + PT-JPL
│   └── 05_et_partitioning.ipynb       # Final partitioning
└── data/                              # Sample data for notebooks
    └── (bundled sample datasets)
```

---

## Tutorial Notebooks

The notebooks provide a step-by-step introduction to irrigation water balance estimation using handily. Start with notebook 01 and work through in order.

### 01 - Introduction to Irrigation Water Balance

Orients users to the problem, required data, and overall workflow:
- What is irrigation water balance and why it matters
- The handily approach: terrain + remote sensing + climate
- Data requirements overview (DEM, flowlines, NDWI, climate, ET)
- Environment setup and configuration walkthrough

### 02 - Terrain Analysis: Building a Relative Elevation Model

Explains how terrain data identifies fields influenced by shallow groundwater:
- What is a REM vs DEM
- Input data exploration (DEM, flowlines, NDWI)
- Building the stream mask
- Computing the REM
- Field statistics and visualization

### 03 - Field Classification: Stratification and Pattern Selection

Covers how fields are classified for ET modeling:
- Why stratify fields by water source
- REM-based partitioning (threshold concept)
- Stream classification (perennial, intermittent, managed)
- IrrMapper irrigation history
- Pattern field selection criteria

### 04 - Climate and ET Data: GridMET and PT-JPL

Demonstrates how climate and remote sensing data are combined:
- Reference ET concepts
- GridMET climate data access
- PT-JPL remote sensing via Earth Engine
- Understanding ET fraction
- Joining climate + ET data

### 05 - ET Partitioning: Separating Irrigation from Groundwater

Explains the final step of separating applied irrigation from natural water sources:
- The partitioning problem
- Water balance approach (AET = PE + ET_gwsm + ET_irr)
- Using pattern fields as donors
- The partitioning algorithm
- Interpreting and validating results

---

## Beaverhead Example

The `beaverhead/` directory contains a complete workflow example for processing irrigation fields in the Beaverhead area of Montana.

### Running the Example Script

```bash
# Run full workflow (uses config flags)
python examples/beaverhead/beaverhead.py examples/beaverhead/beaverhead_config.toml

# Run specific steps
python examples/beaverhead/beaverhead.py --step rem           # REM workflow only
python examples/beaverhead/beaverhead.py --step stratify      # Stratification only
python examples/beaverhead/beaverhead.py --step irrmapper     # IrrMapper export (EE)
python examples/beaverhead/beaverhead.py --step pattern       # Pattern selection
python examples/beaverhead/beaverhead.py --step met           # GridMET download
python examples/beaverhead/beaverhead.py --step et            # PT-JPL export (EE)
python examples/beaverhead/beaverhead.py --step join          # Join ET data
python examples/beaverhead/beaverhead.py --step partition     # ET partition
python examples/beaverhead/beaverhead.py --step qgis          # Update QGIS project
python examples/beaverhead/beaverhead.py --step all           # Run all steps
```

---

## CLI Command Reference

### 1. STAC Catalog Management

Build a local 3DEP 1m STAC catalog for efficient DEM access:

```bash
# Build new catalog for specific states
handily stac build \
    --out-dir ~/data/IrrigationGIS/handily/stac/3dep_1m/ \
    --states MT ID WY

# Extend existing catalog with more states
handily stac extend \
    --out-dir ~/data/IrrigationGIS/handily/stac/3dep_1m/ \
    --states WA OR
```

### 2. AOI Tile Generation

Generate processing tiles from field boundaries:

```bash
handily aoi \
    --fields ~/data/IrrigationGIS/Montana/statewide_irrigation_dataset.shp \
    --out-shp ~/data/IrrigationGIS/handily/aoi_tiles.shp \
    --max-km2 625 \
    --buffer-m 1000 \
    --bounds -112.5 45.0 -112.0 45.5
```

### 3. REM/HAND Workflow

Build Relative Elevation Model for a bounding box:

```bash
handily bounds \
    --bounds -112.418 45.445 -112.353 45.49 \
    --fields ~/data/IrrigationGIS/Montana/statewide_irrigation_dataset.shp \
    --ndwi-dir ~/data/IrrigationGIS/handily/ndwi/beaverhead/ \
    --flowlines-local-dir ~/data/IrrigationGIS/boundaries/wbd/NHD_H_Montana_State_Shape/Shape \
    --stac-dir ~/data/IrrigationGIS/handily/stac/3dep_1m/ \
    --ndwi-threshold 0.45 \
    --out-dir ~/data/IrrigationGIS/handily/outputs/
```

### 4. Meteorology (GridMET)

Download GridMET time series for field centroids:

```bash
handily met download --config examples/beaverhead/beaverhead_config.toml
```

### 5. ET Workflows (PT-JPL via Earth Engine)

Export PT-JPL evapotranspiration fraction (runs on Earth Engine):

```bash
# Export to GCS bucket (async - check EE task manager)
handily et export --config examples/beaverhead/beaverhead_config.toml

# Join PT-JPL with GridMET after export completes
handily et join --config examples/beaverhead/beaverhead_config.toml
```

### 6. ET Partitioning

Partition ET into subsurface and irrigation components:

```bash
handily partition --config examples/beaverhead/beaverhead_config.toml
```

### 7. Bucket Sync

Sync Earth Engine exports from GCS to local filesystem:

```bash
# Sync IrrMapper exports
handily sync --config examples/beaverhead/beaverhead_config.toml \
    --subdir irrmapper \
    --glob "*irr_freq*"

# Sync PT-JPL exports
handily sync --config examples/beaverhead/beaverhead_config.toml \
    --subdir ptjpl \
    --glob "*.csv"

# Dry run (preview without copying)
handily sync --config examples/beaverhead/beaverhead_config.toml \
    --subdir irrmapper \
    --dry-run
```

### 8. QGIS Integration

Update QGIS project with output layers:

```bash
# Update existing project with layers from config
handily qgis update \
    --config examples/beaverhead/beaverhead_config.toml \
    --project ~/data/IrrigationGIS/handily_debug.qgz \
    --group beaverhead

# Generate QLR file for drag-and-drop import
handily qgis qlr --config examples/beaverhead/beaverhead_config.toml

# Open QGIS with project
handily qgis open --project ~/data/IrrigationGIS/handily_debug.qgz
```

---

## Typical Workflow Sequence

### Initial Setup (one-time)

```bash
# 1. Build STAC catalog for your states
handily stac build --out-dir ~/data/stac/3dep_1m/ --states MT ID

# 2. Generate AOI tiles for processing
handily aoi \
    --fields ~/data/fields.shp \
    --out-shp ~/data/aoi_tiles.shp \
    --max-km2 400
```

### Per-AOI Processing

```bash
# 3. Run REM workflow for an AOI
handily bounds \
    --bounds -112.418 45.445 -112.353 45.49 \
    --fields ~/data/fields.shp \
    --ndwi-dir ~/data/ndwi/ \
    --flowlines-local-dir ~/data/NHD/Shape \
    --stac-dir ~/data/stac/3dep_1m/ \
    --out-dir ~/data/outputs/aoi_001/

# 4. Update QGIS for visualization
handily qgis update \
    --config examples/beaverhead/beaverhead_config.toml \
    --project ~/data/project.qgz
```

### ET Processing (after REM)

```bash
# 5. Download GridMET meteorology
handily met download --config examples/beaverhead/beaverhead_config.toml

# 6. Export PT-JPL (async Earth Engine task)
handily et export --config examples/beaverhead/beaverhead_config.toml

# 7. Wait for EE export, then sync
handily sync --config examples/beaverhead/beaverhead_config.toml --subdir ptjpl

# 8. Join ET with meteorology
handily et join --config examples/beaverhead/beaverhead_config.toml

# 9. Partition ET
handily partition --config examples/beaverhead/beaverhead_config.toml
```

---

## Configuration File

See `beaverhead/beaverhead_config.toml` for a complete example. Key sections:

- **Project identification**: `project_name`, `bucket_prefix`, `local_data_root`
- **Input data paths**: `fields_path`, `flowlines_local_dir`, `ndwi_dir`, `stac_dir`
- **AOI bounds**: `bounds` (W, S, E, N in EPSG:4326)
- **Output directory**: `out_dir`
- **Step flags**: `run_rem`, `run_stratify`, `run_irrmapper`, etc.
- **ET parameters**: `ptjpl_start_yr`, `ptjpl_end_yr`, `met_start`, `met_end`
- **QGIS integration**: `qgis_project`, `qgis_layer_group`

---

## Verbose Output

Add `-v` flag for debug logging:

```bash
handily -v bounds --bounds ...
handily -v qgis update --config ...
```
