# Handily Docs Plan

## Goal

Build a repo-wide documentation system for `handily` that matches the docs stack and tone of
`../obsmet`:

- `mkdocs` + `mkdocs-material`
- Markdown pages under `docs/`
- one short `README.md` as the repo landing page
- terse wording
- developer/operator audience
- command-first examples
- explicit inputs, outputs, and artifact paths

This is not a notebook replacement plan. Notebooks remain the visual and exploratory layer.
Main docs cover architecture, operations, CLI usage, configuration, and outputs.

## Audience

Primary audience:

- sophisticated developer
- technical analyst
- operator running Beaverhead or other AOIs

Assumptions:

- comfortable with Python packaging and CLI workflows
- comfortable reading config files and geospatial artifact names
- does not need introductory GIS or hydrology teaching in the main docs

Not the target:

- beginner GIS users
- notebook-only users
- long-form conceptual hydrology explanation

## Docs Stack

Mirror `obsmet`:

- `mkdocs.yml`
- `docs/index.md`
- `docs/overview.md`
- `docs/cli_reference.md`
- topic pages grouped by workflow and operations

Recommended package additions:

- add `docs/requirements.txt`
- add `mkdocs-material`
- keep docs plain Markdown, no notebook rendering in the main site

## Documentation Model

Split documentation into three layers.

### Layer 1: Repo Entry

Files:

- `README.md`

Purpose:

- state scope
- show minimal install
- point to the docs site
- point to Beaverhead examples and notebooks

Constraint:

- keep short
- do not duplicate the manual

### Layer 2: Main Docs Site

Files:

- `docs/`
- `mkdocs.yml`

Purpose:

- architecture
- CLI reference
- workflow boundaries
- operational tooling
- configuration and outputs

Style:

- terse
- declarative
- artifact-oriented

### Layer 3: Notebooks

Files:

- `examples/beaverhead/*.ipynb`

Purpose:

- visual exploration
- worked example
- pedagogical explanation

Constraint:

- notebooks should not be the only place a capability is described

## Proposed Docs Tree

```text
README.md
mkdocs.yml
docs/
  index.md
  overview.md
  cli_reference.md
  configuration.md
  outputs.md
  workflows/
    rem.md
    stratification.md
    met_et.md
    partition.md
  operations/
    stac.md
    aoi.md
    exports_sync.md
    qgis.md
  examples/
    beaverhead.md
notes/
  DOCS_PLAN.md
```

## Page Plan

### `README.md`

Keep minimal.

Contents:

- one-paragraph scope statement
- install
- one canonical command path
- links to:
  - docs site
  - Beaverhead example
  - notebooks

Remove outdated narrative that no longer matches the actual CLI or package surface.

### `docs/index.md`

Purpose:

- answer what `handily` does
- provide quick links
- expose major capabilities in one table

Include:

- short scope statement
- capability table:
  - 3DEP STAC
  - AOI tiling
  - REM/HAND
  - stratification
  - pattern selection
  - GridMET
  - PT-JPL export/join
  - ET partition
  - bucket sync
  - QGIS integration
- links to workflow and operations pages

### `docs/overview.md`

Purpose:

- define the full pipeline and stage boundaries

Suggested stage model:

1. Data access and tiling
2. Terrain and REM generation
3. Stratification and pattern selection
4. Climate and ET acquisition
5. ET join and partition
6. Delivery interfaces

Include:

- one mermaid flowchart
- for each stage:
  - what it does
  - why it exists
  - inputs
  - outputs
  - main modules

### `docs/cli_reference.md`

Purpose:

- document the actual CLI surface

Document only implemented commands:

- `bounds`
- `aoi`
- `stac build`
- `met download`
- `et export`
- `et join`
- `partition`
- `sync`
- `qgis update`
- `qgis qlr`
- `qgis open`

Include:

- typical Beaverhead workflows
- common flags
- output artifacts by command
- notes on manual wait points for Earth Engine export workflows

Do not document broken or not-yet-implemented commands as supported.

### `docs/configuration.md`

Purpose:

- document `HandilyConfig` as a schema

Group config fields by concern:

- project identity
- input datasets
- REM settings
- IrrMapper/pattern
- meteorology
- PT-JPL
- partition
- QGIS
- bucket/local mirror

Include:

- minimal config example
- full Beaverhead config example
- field table with:
  - key
  - type
  - required
  - default
  - used by

### `docs/outputs.md`

Purpose:

- centralize all produced artifacts

Document:

- filename
- format
- semantics
- producing command or module
- downstream consumer

Include:

- `dem_bounds_1m.tif`
- `ndwi_bounds.tif`
- `streams_bounds.tif`
- `rem_bounds.tif`
- `flowlines_bounds.fgb`
- `fields_bounds.fgb`
- `fields_stratified.fgb`
- `fields_pattern.fgb`
- joined parquet outputs
- partitioned parquet outputs
- `handily.qlr`

### `docs/workflows/rem.md`

Purpose:

- document the terrain path as a software workflow

Cover:

- local STAC-backed DEM acquisition
- NHD flowline selection
- NDWI mosaic and thresholding
- stream mask construction
- REM generation
- field zonal statistics

Modules:

- `pipeline.py`
- `dem.py`
- `io.py`
- `compute.py`

### `docs/workflows/stratification.md`

Purpose:

- document field classification and pattern selection

Cover:

- NHD FCODE mapping
- nearest-stream assignment
- REM thresholding
- strata construction
- IrrMapper irrigation frequency
- pattern candidate selection

Modules:

- `nhd.py`
- `stratify.py`
- `pattern.py`
- `et/irrmapper.py`

### `docs/workflows/met_et.md`

Purpose:

- document the climate and ET acquisition chain

Cover:

- GridMET download
- PT-JPL export
- bucket sync
- ET join

Modules:

- `et/gridmet.py`
- `et/image_export.py`
- `et/join.py`
- `bucket.py`

### `docs/workflows/partition.md`

Purpose:

- document ET partitioning logic and outputs

Cover:

- donor fields by strata
- annual sums
- groundwater/soil moisture ET estimate
- monthly disaggregation
- failure modes when donor coverage is absent

Module:

- `et/partition.py`

### `docs/operations/stac.md`

Purpose:

- document 3DEP STAC as an operational prerequisite

Cover:

- why local STAC exists
- how to build it
- cache behavior
- collection assumptions
- bbox index behavior

### `docs/operations/aoi.md`

Purpose:

- document AOI tiling as a scaling tool

Cover:

- centroid-buffer AOIs
- key flags
- intended use for larger regional processing

Keep the page scoped to actually supported AOI helpers.

### `docs/operations/exports_sync.md`

Purpose:

- document Earth Engine export and local sync conventions

Cover:

- export destinations
- GCS bucket layout
- local mirror layout
- sync semantics
- expected asynchronous wait points

### `docs/operations/qgis.md`

Purpose:

- document delivery into QGIS

Cover:

- output discovery
- project update
- QLR generation
- expected layer naming

### `docs/examples/beaverhead.md`

Purpose:

- define Beaverhead as the reference project

Cover:

- example script
- config file
- notebooks
- canonical run order

Keep concise. This is a reference page, not a tutorial.

## MkDocs Navigation Plan

Follow `obsmet` style:

```yaml
nav:
  - Home: index.md
  - Overview: overview.md
  - Workflows:
    - REM: workflows/rem.md
    - Stratification: workflows/stratification.md
    - Climate and ET: workflows/met_et.md
    - Partition: workflows/partition.md
  - Operations:
    - 3DEP STAC: operations/stac.md
    - AOI Tiling: operations/aoi.md
    - Exports and Sync: operations/exports_sync.md
    - QGIS: operations/qgis.md
  - Configuration: configuration.md
  - Outputs: outputs.md
  - Examples:
    - Beaverhead: examples/beaverhead.md
  - CLI Reference: cli_reference.md
```

## Style Rules

Match `obsmet` tone.

- terse
- explicit
- command-first
- no marketing language
- no broad claims without artifact examples
- define terms once, then reuse them consistently

Prefer these section patterns:

- `What it does`
- `Why it exists`
- `Inputs`
- `Outputs`
- `Commands`
- `Failure modes`

Prefer:

- tables over prose
- exact filenames over generic phrases
- real command examples over pseudocode
- one mermaid diagram per architecture-heavy page

Avoid:

- notebook-style narrative in the main docs
- long hydrology background sections
- API-by-function reference unless the package stabilizes more

## Capability Coverage Rules

Every major implemented capability should appear in the docs site at least once.

Required coverage targets:

- local STAC build and use
- AOI generation
- REM workflow
- flowline classification and strata
- IrrMapper pattern selection
- GridMET acquisition
- PT-JPL export and join
- ET partitioning
- bucket sync
- QGIS integration

Capabilities may be covered in one of three ways:

- primary workflow page
- operations page
- CLI reference

Notebooks can remain the visual companion layer, but should not be the only source of truth.

## Repo / Docs Boundary

### Main Docs Should Cover

- package purpose
- architecture
- config schema
- command usage
- operational sequencing
- artifact contracts

### Notebooks Should Cover

- visual inspection
- Beaverhead exploratory analysis
- parameter interpretation
- richer figures and maps

### README Should Cover

- thin entry point only

## Drift Control

To reduce README / docs / notebook drift:

1. Treat `docs/cli_reference.md` as the canonical command surface.
2. Treat `docs/configuration.md` as the canonical config schema.
3. Treat `docs/outputs.md` as the canonical artifact contract.
4. Keep `README.md` short and link outward.
5. When a workflow changes, update:
   - workflow page
   - CLI reference if command-facing
   - Beaverhead example page if sequence changes

## Implementation Order

1. Add MkDocs scaffolding:
   - `mkdocs.yml`
   - `docs/requirements.txt`
2. Write core entry pages:
   - `docs/index.md`
   - `docs/overview.md`
   - `docs/cli_reference.md`
3. Add contracts:
   - `docs/configuration.md`
   - `docs/outputs.md`
4. Add workflow pages
5. Add operations pages
6. Add Beaverhead reference page
7. Reduce and align `README.md`

## Definition of Done

The docs plan is complete when:

- a developer can infer the full package flow from `index`, `overview`, and `cli_reference`
- every major implemented capability appears in the docs site
- command pages describe actual current behavior, not aspirational behavior
- README, docs, and notebooks have clear role separation
- Beaverhead is documented as the reference project without becoming the entire product surface

## Open Questions

These should be resolved while implementing docs, not before writing this plan:

- whether to document unstable commands that are present but currently broken
- whether to include `.qgs` and `.qgz` together once QGIS handling is fixed
- whether docs should include a generated config schema table or maintain it manually
- whether some dev/example scripts should be demoted from docs if the CLI becomes canonical
