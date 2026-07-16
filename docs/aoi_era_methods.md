# AOI-Era High-Resolution REM — Methods Record

Permanent methods record for the pre-FAC-10m "AOI era" of REM/HAND development.
It is written to stand alone: everything needed to understand what the era built,
why, and how to reproduce it is recorded here, so the record survives even if
`notes/`, `logs/`, and the `/data/ssd2/handily/` data disk are all lost. Every
number carries a unit; where a reproduction record is missing this document says
so rather than guessing.

Facts are compiled from git history, the archive tag, the surviving run records,
the per-state and per-AOI configs, and the contemporaneous development notes.
Provenance detail beyond this file lives in the manifest `notes/AOI_ERA_ARCHIVE.md`
(untracked). The era's own algorithm write-ups were deleted from disk during
cleanup but remain recoverable from git history:
`git show 3893430:notes/algorithm.md` and `git show 3893430:notes/fac_algorithm.md`
(and the abandoned docs-site plan `git show 3893430:notes/DOCS_PLAN.md`).

## Scope and timeline

The AOI era was per-area-of-interest REM development on high-resolution terrain:
1 m and 5 m 3DEP DEMs, seeded from NAIP/NDWI-derived surface-water evidence, run
one AOI at a time. It ran roughly March–June 2026 in three overlapping phases and
was superseded by the current FAC 10 m + Sentinel-2 seasonal-wetness pipeline
(`src/handily/rem_fac.py`), which is the algorithm of record. Nothing below is a
current recommendation; it is the historical method.

| Phase | What | Where | When | Status |
|---|---|---|---|---|
| A | Batch EDT / propagated-mask REM (NHD flowlines + NDWI/RF mask + BFS) | MT/NM/NV per-state | ~late Mar 2026 | Code survives on `main` (`pipeline.py`, `compute.py`); secondary to FAC |
| B | Anisotropic-frame / RF-water experiments | NV AOI 0773 (Upper Humboldt, HUC8 16040101) | 2026-04-10 → 04-21 | Code archived at tag only; derived products lost |
| C | 1 m / 5 m FAC channel-head experiments | MT AOIs 0007/0009/0010, Bitterroot 0020/0024/0025/0026/0030 | 2026-06-10 → 07-02 | Superseded by FAC 10 m; run records preserved (this dir) |

The FAC pipeline itself — a dense flow-accumulation stream network as the
geometric scaffold, plus a per-reach channel-head solve — was the breakthrough
over sparse-NHD methods. DEM resolution was *not* the breakthrough; the 1 m
resolution advantage decomposed into network density and seed evidence, both of
which reproduce at 10 m (see Verdicts).

## Code lineage

| Item | Value |
|---|---|
| Archive tag | `rem-pre-fac-cleanup-2026-06` |
| Tag commit | `78b89288ccd9f9970912367969a84b7941d5267f` (2026-04-21, "feat: FAC REM — min-burn sparse, IDW smooth, DEM clamp, halo spokes") |
| Cleanup commit (removed the experimental modules from `main`) | `44d5e8ba3680718a45864964ce307869f3b69567` (2026-06-10) |
| `naip_rf.py` removal | `229f618` (later than the cleanup; the module is still present at the tag) |
| Salvaged notes recoverable at | `git show 3893430:notes/{algorithm,fac_algorithm,DOCS_PLAN}.md` |

**Durability warning.** The tag *ref* is local-only — it is not on `origin`. Both
`78b89288` and the cleanup commit `44d5e8b` are ancestors of `origin/main`, so the
pre-cleanup tree is recoverable *by commit hash* from origin, but the mnemonic tag
handle is not durable. To make the archive point survive a lost local clone:
`git push origin rem-pre-fac-cleanup-2026-06`.

## Phase A — batch EDT / propagated-mask REM

The oldest path and the only AOI-era code still live on `main`
(`src/handily/pipeline.py`, `src/handily/compute.py`). It is driven per state by
`configs/handily/{mt,nm,nv}_rem.toml` (superseded top-level `configs/{mt,nm,nv}_rem.toml`
were deleted at the cleanup). Algorithm:

1. **Stream skeleton from NHD.** Load state NHD flowlines
   (`NHDFlowline_filtered.fgb`; MT had 1,005,544 features), clip to the AOI,
   buffer by `flowlines_buffer_m = 1.0` m.
2. **Surface-water evidence mask.** Threshold an EE-exported NDWI raster at
   `ndwi_threshold = 0.15` (the Phase-B anisotropic path used 0.25); optionally a
   Random Forest water mask (Phase B) in place of the NDWI threshold.
3. **Mask propagation.** With `rem_propagate_mask = true`, propagate the water
   mask along the flowline network by breadth-first search so seeded reaches
   extend the confirmed channel; this is the "propagated mask."
4. **EDT-based REM.** Compute height above the nearest confirmed drainage using a
   Euclidean distance transform against the propagated channel mask; stratify at
   `rem_threshold = 2.0` m.

Key batch parameters (all three states identical except inputs): `ndwi_threshold
0.15`, `flowlines_buffer_m 1.0` m, `rem_threshold 2.0` m, `rem_propagate_mask
true`, `delete_stac_cache true` (raw 3DEP tiles deleted after each mosaic — saved
~1 TB across 694 MT AOIs). Coverage was gated on `stac_1m == 1` (MT: 857 AOIs
total, 694 with 1 m 3DEP). STAC 1 m items at `/nas/handily/stac/3dep_1m/` (3,381
items, rebuilt 2026-03-23). This path was kept for batch NHD workflows but is
secondary to FAC.

## Phase B — anisotropic-frame / RF experiments (NV AOI 0773)

Phase B built REM on 1 m 3DEP for NV AOI 0773 (Upper Humboldt) and is where the
FAC idea emerged. All of its bespoke code was archived at the tag and deleted from
`main` at the cleanup; its derived products are **lost** (the archive dir
`/data/ssd2/handily/experiments/nv_aoi_0773_archive/` no longer exists — only the
raw `dem_bounds_1m.tif` and `naip/` survive, and both are re-downloadable). No
`fac_rem_run.json` exists anywhere for 0773. The modules, described from their
docstrings at the tag:

- **`rem_frame.py`** (1,622 lines) — *Anisotropic REM via thalweg-frame
  cross-section interpolation.* Takes confirmed NHD flowlines (from
  network-propagated NDWI seeding), decomposes them into simple reaches, snaps each
  reach to the DEM thalweg via dynamic programming, builds a smoothed curvilinear
  frame for stable cross-section normals, samples cross-sections with ridge-stop
  logic, rasterizes an anisotropic water surface from the section interpolation,
  and returns `REM = DEM − water_surface`. Carried dataclasses `ReachMetrics`,
  `SnappedReach`, `FrameReach`, `CrossSectionSet`, `AnisotropicREMResult`, and used
  the (now-removed) `HandilyConfig`.
- **`rem_experimental.py`** (3,252 lines) — *Paired-reach side strips using the
  overall DEM aspect.* For two groups of reaches within a maximum separation,
  composes each side into one ordered LineString (bridging internal gaps), clips
  each composite frame to where distance to the other frame is ≤ `max_frame_sep_m`
  (default 2,000 m), transfers that extent to each composite snapped thalweg,
  closes a polygon from the two clipped thalwegs, computes the overall downhill
  direction of the DEM inside that polygon, and emits strips whose orientation is
  **normal to the bulk DEM aspect** (along contours) rather than to local frame
  tangents. The driver ran on reaches 7 vs 8 (braided bight) and (2,6,12,10) vs (9)
  (open inter-reach zone) of the 0773 debug subset. Key constants: sample spacing
  10 m, strip spacing 20 m, max strip length 3,000 m, ridge prominence 10 m,
  descend-stop 5 m, ray step 2 m; resnap weights elev 0.45 / water 0.40 / prior 0.15.
- **`rem_sag.py`** (265 lines) — *Downward-only REM relaxation from hard support.*
  Treats an existing REM as an upper bound and lets it relax **downward** from hard
  support pixels (where water is known at the surface); it never lifts REM. Solves a
  projected membrane in REM space: pinned to 0 on support, optionally fixed to the
  prior on the AOI boundary, constrained `0 ≤ rem ≤ rem_prior`. `build_stream_evidence_support()`
  builds the hard-support mask where a lower-density snapped-thalweg network overlaps
  the evidence water mask.
- **`rem_surface_relax.py`** (753 lines) — *Upward-only water-surface relaxation
  from hard support.* The complement to `rem_sag`: treats a water-surface raster as
  a lower bound and relaxes it **upward** from support, never lowering the prior and
  never rising above the DEM. Held `compute_naip_ndvi_match()` (resample NAIP
  red/NIR to a match grid → NDVI; supports 4-band R,G,B,NIR and 3-band CIR N,R,G)
  and the `relax_water_surface_ndvi_pins()` solver. Both helpers were later inlined
  into `rem_fac.py`.
- **`naip_rf.py`** (811 lines) — *NAIP Random Forest land-use classifier.*
  Assembles training polygons from compiled IrrMapper labels and NWI wetlands,
  draws stratified random sample points, extracts NAIP bands + SARL class via Earth
  Engine, trains a Random Forest, and exports predictions. Classes
  uncultivated/irrigated/surface_water (binary other/surface_water at train time),
  analysis CRS EPSG:5070, NAIP bands R,G,B,NIR, interior buffers to cut mixed-pixel
  edge effects. This produced the Phase-B RF water mask (`water_mask_rf.tif`) that
  replaced the NDWI threshold in shadow/turbid/partial-canopy conditions.

Phase-B lesson (from `notes/REM_DEV_PROGRESS.md`): NHD is too sparse for continuous
REM in arid landscapes — the dense FAC network (2,144 reaches on 0773 vs ~14 NHD
reaches) was the pivotal change, and raster relaxation alone cannot manufacture the
20–40 m depth-to-water of dry uplands (it only pushes toward the DEM). That drove
the move to a per-reach channel-head solve.

## The FAC method (as it stood in the AOI era)

The salvaged `algorithm.md` and `fac_algorithm.md` (2026-04-19) describe the FAC
water-surface-relaxation formulation, which was itself superseded within the era by
the channel-head solve (`rem_fac_head.py`, Phase 9). The durable core equation is:

```
REM = max(DEM − water_surface, 0)
```

The whole problem is constructing `water_surface`. The FAC stages (from the
salvaged docs and the early `rem_fac.py` at the tag, 1,539 lines):

1. **Orientation field** (`build_orientation_field`) — coarsen the DEM to a working
   grid (20 m), heavily smooth (Gaussian σ ≈ 500 m), derive downhill vectors, and
   convert to an aspect-normal field so cross-section orientation reflects
   valley-scale form, not channel-scale noise.
2. **Dense FAC cross sections** (`generate_fac_strips`) — station each FAC reach at
   a fixed spacing, cast a ray along the smoothed-DEM aspect-normal, and stop at the
   first hit on another FAC stream (`interreach` strip) or the AOI boundary (`edge`
   strip). No snapping, no water gating.
3. **Endpoint elevations** (`_attach_fac_strip_elevations`) — sample DEM elevation
   at the reach base and the strip endpoint; the water surface is linear between them.
4. **Sparse burn to raster** (`rasterize_sparse_sections_20m` → later 5 m) followed
   by a fast fill: nearest, normalized Gaussian, or finite-radius IDW
   (`fill_sparse_sections_idw`, the preferred prior).
5. **Hard support** (`build_stream_evidence_support`) — the only pixels allowed to
   reach `water_surface = DEM` (`REM = 0`), where snapped thalwegs overlap the
   evidence water mask; everywhere else keeps a minimum clearance (0.1 m in the
   relaxation era, 0.5 m `d_min_off_support_m` in the head-solve configs).
6. **Topology-derived pin weights** (`rem_fac_topology.py`, 421 lines) — orient the
   FAC network upstream→downstream from smoothed-DEM endpoint elevations, estimate
   per-reach wet-seed strength from high NDVI plus hard-support override, and
   propagate wet influence upstream with exponential decay in network distance,
   elevation gain, and Strahler-scaled persistence. Lower wet reaches defend shallow
   water; upper dry benches detach with distance and elevation gain.
7. **Channel-head solve** (`rem_fac_head.py`, 330 lines) — the step that replaced
   raster relaxation. Instead of only pushing the water surface upward, it solves a
   per-reach residual-depth relaxation on the FAC graph *before* rasterization, so a
   dry headwater channel gets its water surface deliberately placed below the bed.
   The residual-depth (not absolute-head) formulation subtracts a per-reach
   `head_depth_m` from the DEM-sampled bed, which follows bed slope at constant depth
   and removes the sawtooth artifacts that absolute-head stamps produced on long
   reaches. This is what current `rem_fac.py` runs.

## Phase C — 1 m / 5 m FAC channel-head experiments (MT AOIs)

Phase C applied the FAC + channel-head pipeline to Montana AOIs on 1 m 3DEP, using
the reusable profile `configs/rem/profiles/mt_0009_best.toml` (per-AOI configs
supply only `[paths]` and seed overrides). Each run writes `fac_rem_run.json`.

**AOI 0009 seed/variant ladder** (`/data/ssd2/handily/mt/aoi_0009/experimental_*`):

| Variant dir | Config | What it varied | Run JSON |
|---|---|---|---|
| `experimental_full` | `mt_0009.toml` | NAIP single-date NDVI seed; original Apr-21 build on pre-JSON, pre-down-prop head-solve code | **absent** (predates JSON emission) |
| `experimental_reprod` | `mt_0009_reprod.toml` | reproduction rerun (NAIP seed) | present |
| `experimental_naip_v2` | `mt_0009_naip_v2.toml` | NAIP seed on current code — clean seed-only A/B baseline for the S2 run | present |
| `experimental_s2` | `mt_0009_s2.toml` | S2 seasonal wetness seed (spring NDVI + fall NDWI), sigmoid `ndvi_mid 0.48 / ndvi_scale 0.12` for the [0,1] index | present |
| `experimental_s2_summer` | `mt_0009_s2_summer.toml` | S2 summer-NDVI-only seed (A/B control for `s2`); same sigmoid | present |

Sibling MT AOIs on the same profile: `aoi_0007/experimental_full/`
(`mt_0007.toml`, NAIP seed, run JSON present) and `aoi_0010/experimental_full/`
(**no** run JSON — the dir exists on disk but predates/omits JSON emission).

**Bitterroot 1 m ladder (L2 rung, 2026-07-02)** — five valley-floor AOIs run by
`configs/rem/mt_00{20,24,25,26,30}_bitterroot_s2.toml`, output under
`aoi_00XX/rem_1m_s2/` (not `experimental_*`). All share the `mt_0009_best` profile,
a single 10 m S2 wetness seed sampled by coordinate
(`mt/bitterroot_pilot/s2/s2_wetness_seed_10m.tif`), the S2 sigmoid
(`ndvi_mid 0.48 / ndvi_scale 0.12`), and 1 m DEM / 1 m inflow-injected FAC network /
5 m burn. Run JSONs present for all five.

The only parameter deltas across the ladder are the **seed source** (single-date
NAIP NDVI vs S2 summer NDVI vs S2 spring-NDVI+fall-NDWI wetness) and the sigmoid
recentering that follows it (`ndvi_mid 0.20 / scale 0.06` for raw NAIP NDVI →
`0.48 / 0.12` for the [0,1] combined index); the geometry, burn, fill, propagation,
sag, and solver blocks are the profile's throughout. A second, code-level delta
separates the April `experimental_full`/`reprod` builds (older head-solve, no
downstream propagation) from the June/July S2 and Bitterroot builds, whose
effective configs carry `propagation.down_distance_scale_m = 20000.0` m and a
downstream term the earlier code lacked.

Representative head-solve diagnostics (from the run JSONs, `head_depth_m` per reach):

| AOI / variant | reaches | strips | head_depth mean / max (m) | burn (m) |
|---|---|---|---|---|
| 0007 `experimental_full` (NAIP, Jun-10 code) | 3,638 | 75,020 | 3.77 / 60.5 | 5 |
| 0009 `experimental_s2` (S2 wetness) | 7,448 | 86,403 | 3.31 / 45.1 | 5 |
| 0020 Bitterroot `rem_1m_s2` | 2,669 | 57,042 | 6.48 / 54.1 | 5 |

## Verdicts

- **1 m failed the promotion gate versus 10 m.** The decisive 5-AOI Bitterroot
  panel (`notes/ONE_METER_PILOT_PLAN.md`, 2026-07-02; 5,108 common-footprint wells,
  2,046 IrrMapper-irrigated) had the 1 m rung (L2) *lose to* the production 10 m
  rung (L0) on every shallow axis: irrigated-shallow (obs ≤ 5 m) MAD 1.36 m (L2) vs
  1.20 m (L0); obs 0–2 m MAD 0.88 m (L2) vs 0.67 m (L0); ≤ 2 m precision/recall/F1
  0.17 / 0.52 / 0.26 (L2) vs 0.23 / 0.59 / 0.33 (L0). The gate required L2 to beat
  `max(L0-calibrated, L1)` by ≥ +0.10 in ≤ 2 m precision or ≥ 20 % in MAD; it
  cleared neither. The earlier Beaverhead/AOI-0009 ladder agreed: at wells the 1 m/5 m
  bench (`aoi_0009/experimental_s2`) posted ≤ 2 m precision/recall 0.33 / 0.73 and
  ≤ 5 m MAD 1.13 m vs the 10 m production 0.30 / 0.55 and 0.98 m — 1 m's only edge
  was ≤ 2 m recall on a weak n = 11 positive set. Conclusion: DEM resolution was not
  the binding constraint; ≤ 2 m precision saturated near ~0.22 across the entire
  ladder (the same ceiling the statewide GNN hits), a discrimination limit, not a
  resolution or calibration one.
- **S2 spring-NDVI + fall-NDWI wetness seed matched the benchmark at 10 m.** The
  seasonal index separated the irrigated Big Hole valley floor from forested uplands
  (rank-AUC 0.89) where single-date summer NDVI could not (both read ~0.77, leaving
  the head solve surficial). Reproduced at 10 m, heads tracked terrain (Spearman
  +0.47) and REM < 1 m extent matched the benchmark (11.8 % vs 12.4 %). In the final
  Bitterroot panel the S2-seed 10 m rung (L1) tied L0 on MAD while winning the
  shallow tail (0–2 m P90 |error| 2.5 m vs 3.6 m; best ≤ 2 m recall 0.61).
- **NAIP seeding rejected for scale.** The 20 m single-date NAIP NDVI grid diluted
  the riparian signal (0.152 vs 0.287 for the seasonal S2 index) and per-AOI NAIP
  coverage was too small to build a CONUS-scale seed; the S2 seasonal wetness seed
  replaced it.
- **Superseded by FAC 10 m + S2 wetness.** The adopted CONUS recipe is 10 m
  everywhere with the S2 seasonal-wetness seed as a cheap shallow-tail upgrade where
  available; the 1 m path is retained only as a benchmark tool (no production 1 m
  tier).

## Where the logic lives now — durability map

| Kind of logic | Where |
|---|---|
| Phase-A batch code | `src/handily/pipeline.py`, `src/handily/compute.py` (live on `main`) |
| Phase-B/early-FAC experimental code | tag `rem-pre-fac-cleanup-2026-06` = commit `78b89288` (also reachable by hash from `origin/main`) |
| Salvaged AOI-era algorithm write-ups | `git show 3893430:notes/algorithm.md`, `git show 3893430:notes/fac_algorithm.md`, `git show 3893430:notes/DOCS_PLAN.md` |
| Batch parameters | `configs/handily/{mt,nm,nv}_rem.toml` |
| FAC profile + per-AOI overrides | `configs/rem/profiles/mt_0009_best.toml`, `configs/rem/mt_0009*.toml`, `configs/rem/mt_00{20,24,25,26,30}_bitterroot_s2.toml` |
| Effective run configs (reproduction records) | `configs/rem/archive/aoi_runs/*.json` (10 files, this dir) |
| Run commands / narrative | `logs/` (untracked, zoran-local) and `notes/REM_DEV_PROGRESS.md`, `notes/ONE_METER_PILOT_PLAN.md`, `notes/FAC10_PROGRESS.md` (untracked) |

### The `fac_rem_run.json` schema

Every FAC run writes one, and it is the canonical byte-faithful reproduction
record. Top-level keys: `command`, `config_path`, `profile_path`,
`schema_version` (= 1 on every AOI-era run), `diagnostics`, and `effective_config`
(the fully resolved config *after* profile inheritance). `diagnostics` holds
`heads` (`count`, `head_depth_{min,mean,max}_m`, `runtime_s`), `sparse_burn` and
`idw_fill` (`coverage_fraction`, pixel counts, `runtime_s`), `streams` (`count`),
and `strips` (`count`, `count_before_filter`, `interreach`, `edge`,
`removed_long_crossing`). `effective_config` mirrors the TOML blocks: `paths`,
`strips`, `raster`, `seed`, `propagation`, `sag`, `solver`. Note that
`schema_version` stayed 1 while the *field set* of `effective_config` grew with the
code: the June-10 AOI-0007 run lacks `propagation.down_distance_scale_m`,
`sag.area_sag_{lo,hi}_km2`, `raster.base_{fac_snap_cells,smooth_stations}`,
`seed.seed_corridor_m`, `solver.below_bed_offset_m`, and `strips.write_strip_debug`
that the June/July S2 and Bitterroot runs carry — so a run's field set, not its
schema number, dates the code.

## Reproduction recipe

**A Phase-B product (NV 0773 anisotropic-frame or early FAC).** The code is
gone from `main`; check out the archive tree first. Derived 0773 products are lost
and must be rebuilt from raw inputs.

```
git worktree add /tmp/handily-aoi-era rem-pre-fac-cleanup-2026-06   # or: git worktree add /tmp/handily-aoi-era 78b89288
uv sync --all-extras
uv run python -m handily.rem_fac --config configs/rem/0773_idw1k.toml
```

Requires on disk: `dem_bounds_1m.tif`, `streams_fac.fgb`, and (for the frame path)
`water_mask_rf.tif` + snapped thalwegs — all re-downloadable/re-derivable from 3DEP
1 m (`/nas/handily/stac/3dep_1m/`) and NAIP (USDA Box). The dense FAC network for
0773 was built from the regional Humboldt run (HUC6 160401, 43,834 km², 1.3 billion
pixels at 10 m, ~31 min, 59,808 streams / 73,413 km, threshold 5,000 cells ≈ 0.5 km²).

**A Phase-C product (MT AOI 0009 S2 wetness seed).** Current `main` suffices —
`rem_fac.py` is unchanged in the parts this exercises.

```
uv run python -m handily.rem_fac --config configs/rem/mt_0009_s2.toml
```

Requires on disk (paths are in the config + profile): `mt/aoi_0009/dem_bounds_1m.tif`,
`mt/aoi_0009/streams_fac.fgb` (7,448 reaches, built with 10 m regional inflow
injection), and `mt/aoi_0009/s2_wetness_seed_10m.tif`. The effective config it will
reproduce is archived at
`configs/rem/archive/aoi_runs/mt_0009_experimental_s2_fac_rem_run.json`; that file's
`effective_config` block is the exact parameter set for a byte-faithful rerun.
