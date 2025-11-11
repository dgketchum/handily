# handily
Application of Height Above Nearest Drainage algorithm to estimate groundwater interaction with riparian polygon objects (agricultural fields).

## Installation

This project is packaged with a standard `pyproject.toml`.

### Option A: pip (venv or pipx)

- Python 3.10 is required.
- Create and activate a virtualenv, then install:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -U pip setuptools wheel`
  - `pip install .`

Notes:
- Heavy geospatial wheels (geopandas, rasterio, fiona) install best on Linux/macOS with prebuilt wheels. If your platform lacks wheels, prefer the Conda/Mamba route below.
- We pin `numpy<2` for RichDEM compatibility and include `pybind11` to help build if a wheel isn’t available.

### Option B: uv (fast installer)

- Install uv (see https://docs.astral.sh/uv/ for platform installers), then:
  - Ensure Python 3.10 is available: `uv python install 3.10`
  - Create and activate a venv: `uv venv --python 3.10 .venv && source .venv/bin/activate`
  - Install the package: `uv pip install -U pip setuptools wheel && uv pip install .`

Tool-style install (no venv activation):
- `uv tool install .`  # installs a `handily` command in uv’s tool shim
- Run: `handily --help`

Dev install with extras:
- `uv pip install -e .[dev]`

Notes:
- uv honors the `pyproject.toml` and will use prebuilt wheels when available. If wheels are missing for your platform, prefer Conda/Mamba below for GDAL/Fiona/Rasterio.

### Option C: Conda/Mamba (robust for compiled deps)

- `mamba create -n handily -c conda-forge python=3.10 geopandas rasterio rioxarray xarray scipy rasterstats pynhd numpy=1.26 -y`
- `mamba activate handily`
- `pip install -U pybind11 richdem`

If you maintain a local RichDEM checkout and want to install from source instead of PyPI:
- `pip install -v --no-binary :all: /path/to/richdem/wrappers/pyrichdem`

## 3DEP 1 m STAC

Build a local STAC catalog for USGS 3DEP OPR 1 m DEM tiles directly from the TNM S3 index:

- Build for Montana (MT) only into `stac/3dep_1m`:
  - `handily stac build --out-dir stac/3dep_1m --states MT`
- Extend later with Idaho (ID):
  - `handily stac extend --out-dir stac/3dep_1m --states ID`

The builder crawls project → subproject → metadata, parses FGDC XMLs (for bbox and links), and creates one STAC Item per tile with links to the GeoTIFF and XML.

## Usage

After installation, use the CLI:

- Run stratification (replace `--fields` with your dataset). Required: pass `--stac-dir` pointing to your local STAC:
  - `handily run --huc10 1002000207 --fields /path/to/fields.shp --out-dir ./outputs/beaverhead --wbd-local-dir ~/data/IrrigationGIS/boundaries/wbd/NHD_H_Montana_State_Shape/Shape --stac-dir stac/3dep_1m -v`
  - To use local NHD flowlines from the same state dataset, you can add `--flowlines-local-dir` (defaults to the WBD dir if omitted).
  - DEM is LiDAR-only (~1 m). Runs abort if the service returns > 2 m.
  - Optional: `--stac-download-cache-dir` for tile caching, `--stac-collection-id` to target a custom collection id.

Caching and performance
- The final mosaic is cached to `dem_huc10_<HUC>_1m.tif` in your `--out-dir` and reused unless `--overwrite-dem` is provided.
- Individual STAC GeoTIFF tiles are downloaded once and cached under `--stac-download-cache-dir` (default: `<out-dir>/stac_cache`).

Outputs (written to your `--out-dir`):
- REM GeoTIFF, stratified fields (GPKG/SHP), and an interactive debug map (`debug_map.html`).

## Troubleshooting

- RichDEM build/import issues:
  - Ensure Python 3.10, a C++ toolchain, and `pybind11` are present.
  - If installing from PyPI, use `numpy<2` (e.g., `1.26.*`).
  - As a fallback, consider installing RichDEM from local source as shown above.
- Network access is required to fetch 3DEP DEMs. NHD flowlines can be loaded from the local state shapefiles by setting `--flowlines-local-dir`.
  - Note: WBD boundaries are loaded from a local state shapefile (set `--wbd-local-dir` or env `HANDILY_WBD_DIR`).

## Dev usage (without install)

- You can still run the dev script:
  - `python scripts/run_beaverhead.py --huc10 1002000207 --fields /path/to/fields.shp --out-dir ./outputs/beaverhead -v`
  - The script now imports the in-repo package (`src/handily`) or an installed version.
