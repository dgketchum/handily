# Tutorial Notebooks for Handily

## Summary

Create 6 Jupyter notebooks that introduce new users to irrigation water balance estimation using the handily framework. Move existing beaverhead example into `examples/beaverhead/` subdirectory to make space for notebooks and future examples.

## Directory Structure

```
examples/
├── README.md                          # Updated with notebook descriptions
├── beaverhead/                        # Moved from examples/
│   ├── beaverhead.py
│   └── beaverhead_config.toml
├── data/                              # Bundled sample data for notebooks
│   ├── sample_dem.tif                 # Small DEM clip (~5 fields)
│   ├── sample_flowlines.fgb           # Flowlines for sample AOI
│   ├── sample_fields.fgb              # ~10 sample irrigation fields
│   ├── sample_ndwi.tif                # NDWI for sample AOI
│   ├── sample_gridmet.parquet         # Pre-downloaded climate data
│   ├── sample_ptjpl.csv               # Pre-exported ET fractions
│   └── sample_irrmapper.csv           # Pre-exported irrigation history
└── notebooks/
    ├── 01_introduction.ipynb          # Overview and setup
    ├── 02_terrain_analysis.ipynb      # REM workflow
    ├── 03_points_samplling.ipynb      # AOI point sampling
    ├── 04_field_classification.ipynb  # Stratification + IrrMapper + Pattern
    ├── 05_climate_and_et.ipynb        # GridMET + PT-JPL
    └── 06_et_partitioning.ipynb       # Final partitioning
```

## Files to Modify/Create

| File | Action |
|------|--------|
| `examples/beaverhead/` | **Create** directory and move files |
| `examples/beaverhead/beaverhead.py` | **Move** from `examples/beaverhead.py` |
| `examples/beaverhead/beaverhead_config.toml` | **Move** from `examples/beaverhead_config.toml` |
| `examples/data/` | **Create** directory with bundled sample data |
| `examples/README.md` | **Modify** - Update paths, add notebook descriptions |
| `examples/notebooks/01_introduction.ipynb` | **Create** |
| `examples/notebooks/02_terrain_analysis.ipynb` | **Create** |
| `examples/notebooks/03_points_samplling.ipynb` | **Create** |
| `examples/notebooks/04_field_classification.ipynb` | **Create** |
| `examples/notebooks/05_climate_and_et.ipynb` | **Create** |
| `examples/notebooks/06_et_partitioning.ipynb` | **Create** |

---

## Notebook Outlines

### Notebook 1: Introduction to Irrigation Water Balance

**Goal**: Orient users to the problem, required data, and overall workflow.

**Sections**:

1. **What is Irrigation Water Balance?**
   - The core question: How much water are irrigated fields using?
   - Why this matters: water rights, conservation, agricultural planning
   - The challenge: separating applied irrigation from natural water sources

2. **The Handily Approach**
   - Terrain analysis identifies fields near shallow groundwater
   - Remote sensing provides crop water use estimates (ET)
   - Climate data provides the reference for water demand
   - Pattern fields (non-irrigated) serve as natural baselines

3. **Data Requirements Overview**
   - **DEM**: USGS 3DEP 1-meter LiDAR (via STAC)
   - **Flowlines**: USGS NHD (National Hydrography Dataset)
   - **NDWI**: Normalized Difference Water Index imagery (Landsat/NAIP)
   - **Fields**: Irrigation field boundaries (shapefile)
   - **Climate**: GridMET daily meteorology (THREDDS)
   - **ET**: PT-JPL via Landsat (Earth Engine)
   - **Irrigation history**: IrrMapper (Earth Engine)

4. **Workflow Pipeline**
   ```
   Inputs → REM → Stratification → Pattern Selection → ET → Partition
   ```
   - Diagram showing data flow between steps

5. **Environment Setup**
   - Installation: `pip install handily`
   - Earth Engine authentication
   - Google Cloud credentials (for bucket exports)
   - Directory structure for outputs

6. **Configuration Deep Dive**
   - Walk through `beaverhead_config.toml` section by section
   - Explain each parameter's purpose

**Code cells**:
- Load and display example config
- Check data paths exist
- Verify EE authentication

---

### Notebook 2: Terrain Analysis - Building a Relative Elevation Model

**Goal**: Understand how terrain data identifies fields influenced by shallow groundwater.

**Sections**:

1. **What is a REM?**
   - Digital Elevation Model (DEM): raw surface elevation
   - Relative Elevation Model (REM): elevation relative to water surface
   - Why relative elevation matters: groundwater proximity
   - Relationship to HAND (Height Above Nearest Drainage)

2. **Input Data Exploration**
   - Load and visualize the DEM for the AOI
   - Load and visualize NHD flowlines
   - Load and visualize NDWI imagery
   - Understanding data sources and resolution

3. **The Stream Mask**
   - Rasterizing vector flowlines to DEM grid
   - Using NDWI to identify actual water pixels
   - Combining structural (NHD) with observational (NDWI) data
   - Tuning the NDWI threshold

4. **Computing the REM**
   - Algorithm: Gaussian-smoothed base elevation along streams
   - REM = DEM - base_elevation
   - Visualize REM results: terrain colored by relative elevation
   - Interpret low vs high REM values

5. **Field Statistics**
   - Computing zonal statistics (mean REM per field)
   - Histogram of field REM values
   - Map showing fields colored by mean REM

6. **Hands-on: Run the REM Workflow**
   - Use `REMWorkflow` class directly
   - Inspect intermediate outputs
   - Save results to disk

**Code cells**:
- Load DEM, visualize with hillshade
- Load flowlines, plot on DEM
- Load NDWI, show water detection
- Build streams mask step-by-step
- Compute REM, visualize
- Calculate field statistics, create choropleth map

---

### Notebook 3: Field Classification - Stratification and Pattern Selection

**Goal**: Understand how fields are classified for ET modeling.

**Sections**:

1. **Why Stratify Fields?**
   - Different water sources need different models
   - Groundwater-influenced vs elevated fields
   - Stream proximity affects water availability
   - Stratification enables targeted ET estimation

2. **REM-Based Partitioning**
   - The threshold concept: 2 meters (default)
   - Fields below threshold → "partitioned" (shallow groundwater)
   - Fields above threshold → "non-partitioned" (elevated)
   - Sensitivity analysis: varying the threshold

3. **Stream Classification**
   - NHD FCODE values → meaningful categories
   - Perennial: reliable year-round water
   - Intermittent: seasonal water
   - Managed: canals, ditches (artificial)
   - Visualize stream categories on map

4. **Nearest Stream Assignment**
   - KD-tree spatial search algorithm
   - Assigning stream type to each field
   - Distance calculation and optional cutoffs
   - Map showing field-to-stream associations

5. **Final Strata Categories**
   - Four strata: perennial, intermittent, managed, non_partitioned
   - How strata combine REM + stream type
   - Visualize strata distribution (map + bar chart)

6. **IrrMapper: Irrigation History**
   - What is IrrMapper? Annual irrigation classification (1987-present)
   - Computing irrigation frequency per field
   - Visualizing irrigation history
   - Earth Engine export process (async)

7. **Pattern Field Selection**
   - What are pattern fields? Non-irrigated reference sites
   - Selection criteria: irr_freq ≤ 0.1, irr_mean ≤ 0.05
   - Why pattern fields matter: baseline ET estimation
   - Ensuring minimum patterns per strata

8. **Hands-on: Run Stratification**
   - Load REM outputs from Notebook 2
   - Run stratify() function
   - Inspect results and visualize strata

**Code cells**:
- Load fields with REM statistics
- Apply threshold, visualize partitioned/non-partitioned
- Classify flowlines, show categories
- Assign nearest stream to fields
- Create final strata, visualize
- (If IrrMapper data available) Load and analyze irrigation frequency
- Select pattern fields, highlight on map

---

### Notebook 4: Climate and ET Data - GridMET and PT-JPL

**Goal**: Understand how climate and remote sensing data are combined to estimate actual ET.

**Sections**:

1. **Reference ET Concepts**
   - What is reference ET (ETo)? Potential water demand
   - The Penman-Monteith equation (simplified explanation)
   - Why we need daily climate data
   - ETo as a normalization baseline

2. **GridMET Climate Data**
   - What is GridMET? 4km daily climate dataset
   - Variables: precipitation, ETo, temperature, humidity
   - Accessing via THREDDS/OPeNDAP
   - Downloading for field centroids

3. **Exploring Climate Data**
   - Load sample GridMET data for a field
   - Plot time series: ETo, precipitation, temperature
   - Seasonal patterns in water demand
   - Multi-year variability

4. **PT-JPL Remote Sensing**
   - What is PT-JPL? Priestley-Taylor model from Landsat
   - ET fraction concept: AET / ETo
   - Landsat temporal resolution (sparse observations)
   - Earth Engine export process

5. **Understanding ET Fraction**
   - Load sample PT-JPL data
   - Sparse time series visualization
   - Why values range 0-1 (and occasionally >1)
   - Relationship to crop growth stage

6. **Joining Climate + ET**
   - Time series alignment challenge
   - Linear interpolation for sparse satellite data
   - Computing actual ET: AET = ETf × ETo
   - Visualize: daily ETo, sparse ETf, interpolated ETf, daily AET

7. **Hands-on: Download and Join Data**
   - Download GridMET for example fields
   - (If available) Load PT-JPL exports
   - Run join function
   - Inspect joined dataset

**Code cells**:
- Download GridMET for AOI
- Load parquet, plot time series
- Load PT-JPL CSV, visualize sparse dates
- Demonstrate interpolation
- Compute AET, compare to ETo
- Save joined output

---

### Notebook 5: ET Partitioning - Separating Irrigation from Groundwater

**Goal**: Understand the final step: separating applied irrigation from natural water sources.

**Sections**:

1. **The Partitioning Problem**
   - Irrigated fields receive multiple water inputs:
     - Precipitation
     - Groundwater (shallow water table)
     - Applied irrigation water
   - PT-JPL measures total AET, not individual sources
   - Goal: estimate irrigation component

2. **The Water Balance Approach**
   - Basic equation: AET = PE + ET_gwsm + ET_irr
   - PE = precipitation consumed by plants
   - ET_gwsm = groundwater + soil moisture component
   - ET_irr = irrigation water consumed

3. **Using Pattern Fields as Donors**
   - Pattern fields = non-irrigated reference
   - Same climate/strata → similar ET_gwsm
   - Spatial borrowing: use nearest pattern field
   - Computing donor ET_gwsm fraction

4. **The Partitioning Algorithm**
   - Step 1: For donor (pattern) field, compute annual ET_gwsm
   - Step 2: Express as fraction of ETo
   - Step 3: Apply fraction to irrigated field's ETo
   - Step 4: Compute ET_irr = AET - ET_gwsm - PE
   - Handle edge cases (negative values → 0)

5. **Monthly Disaggregation**
   - Annual values → monthly distribution
   - Using seasonal AET patterns
   - Why monthly resolution matters

6. **Interpreting Results**
   - Load partitioned output for example field
   - Visualize: monthly ET_gwsm vs ET_irr
   - Compare irrigated vs non-irrigated fields
   - Seasonal irrigation patterns

7. **Validation Considerations**
   - Sources of uncertainty
   - Comparison with measured data
   - Sensitivity to pattern field selection
   - Limitations of the approach

8. **Hands-on: Run Full Partition**
   - Load joined ET data
   - Load stratified fields with pattern column
   - Run partition function
   - Analyze results

**Code cells**:
- Load joined ET data from Notebook 4
- Load fields with strata + pattern
- Demonstrate donor selection
- Run partitioning
- Visualize monthly ET components
- Compare multiple fields
- Export final results

---

## Implementation Notes

### Plain Language Guidelines

Each notebook should:
- Start with "why" before "how"
- Use analogies (e.g., "REM is like measuring how high you are above the river")
- Avoid jargon on first mention (define terms)
- Include "Key Takeaway" boxes after complex sections
- Use progressive complexity (simple examples → full workflow)

### Visualization Style

- Consistent color schemes across notebooks
- Maps with context (state boundaries, landmarks)
- Time series with clear axis labels
- Annotated figures explaining key features

### Code Style

- Minimal boilerplate (use helper functions)
- Clear variable names
- Comments explaining non-obvious steps
- Progressive disclosure (show simple version, then full version)

### Data Considerations

- Ship small sample datasets for offline use
- Provide fallback static images if data unavailable
- Mark cells requiring Earth Engine with warnings

---

## Sample Data Preparation

Create clipped sample data from the full beaverhead outputs:

1. **Select sample AOI**: Small bounding box containing ~5-10 fields
2. **Clip rasters**: `sample_dem.tif`, `sample_ndwi.tif` (keep small, <5MB each)
3. **Clip vectors**: `sample_fields.fgb`, `sample_flowlines.fgb`
4. **Extract time series**: Filter GridMET/PT-JPL to sample field FIDs
5. **Pre-compute outputs**: Run REM workflow on sample AOI, save results

Sample data should be:
- Small enough to commit to repo (<20MB total)
- Complete enough to run all notebook cells
- Representative of full workflow outputs

---

## Verification

1. Move beaverhead files to subdirectory
2. Update README.md with new paths and notebook descriptions
3. Create each notebook with outlined structure
4. Run all notebooks end-to-end with beaverhead data
5. Verify visualizations render correctly
6. Test with fresh environment (no cached data)
