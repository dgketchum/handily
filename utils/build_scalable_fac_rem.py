"""Build a scalable, NHD-free FAC-REM for one region or a statewide HUC8 batch.

Runs the accepted regime-general 10 m FAC-REM recipe end to end from
CONUS-available substrate only -- no LiDAR, no NAIP, no NHD. Every input is derived
from sources that exist everywhere in the lower 48, so the same command works for
any HUC8, AOI polygon, bounding box, or whole state. The recipe lives in the default
profile ``configs/rem/profiles/conus_fac_rem_scalable.toml`` (the 2026-06-25 values
verified across montane MT + arid NM/NV); it is the single source of truth, and the
``--ndvi-*`` / ``--*-scale-m`` / etc. flags are a pure override layer (default None =
inherit the profile). It is NOT arid-specific.

Pipeline (each stage is idempotent -- existing outputs are reused unless --force):

  1. DEM      : USGS 3DEP 1/3 arc-second tiles -> merged 10 m DEM (EPSG:5070),
                clipped to the region + halo                  (regional_fac).
  2. STREAMS  : WhiteboxTools D8 fill/pointer/accumulation + extract_streams ->
                dense DEM-derived network (streams_regional.fgb) and
                flow_accumulation.tif                          (regional_fac).
                This replaces sparse NHD flowlines, the failure mode of the
                per-HUC8 NHD CONUS pilot in arid basins.
  3. SEED     : MODIS-JJA NDVI climatology (250 m, MPC MOD13Q1) bilinear-matched
                to the DEM grid. High growing-season NDVI marks the irrigated /
                riparian valley floor; low NDVI is desert/upland. Replaces NAIP.
  4. SUPPORT  : JRC Global Surface Water occurrence (30 m, MPC jrc-gsw),
                thresholded to PERMANENT water (occurrence >= --occ-threshold),
                nearest-matched to the DEM grid. Isolates the perennial mainstem
                and excludes flood-irrigation (the NAIP-NDWI failure mode).
  5. CONFIG   : write a rem_fac TOML inheriting the scalable profile (+ any explicit
                CLI overrides) with [paths] pointing at the built inputs.
  6. RUN      : invoke ``python -m handily.rem_fac`` as a subprocess (skip with
                --no-run) -> fac_head_depth_rem_10m.tif (depth) +
                fac_rem_water_surface_10m.tif (elevation).

Single region -- exactly one of:
  --huc8 CODE          resolve the polygon from the WBD HUC8 layer.
  --aoi PATH           a polygon vector file (any CRS) -> reprojected to 5070.
  --bbox MINX MINY MAXX MAXY [--bbox-crs EPSG]   an explicit box.

Statewide / batch -- either of (switches to batch mode):
  --state NM [MT NV ...]   every HUC8 touching these states (national WBD layer).
  --huc8-list FILE         one HUC8 code per line.
  Each HUC8 -> <out-root>/<huc8>/; a shared <out-root>/dem_tiles/ cache is reused
  across HUC8s; done HUC8s (marker + REM raster) are skipped on rerun; large
  re-derivable intermediates are deleted on success (--keep-intermediates opts out);
  one HUC8 failure never aborts the batch (rerun retries the failures).

Validate the result ONLY against GWX unconfined/marginal wells + NHD springs
(utils/validate_fac_gwx_wells.py); never against Ma. Two knobs widen the filled
footprint when the REM has visual gaps in deep upland inter-channel zones --
--max-crossing-strip-m (longer cross-sections) and --idw-radius-m (longer IDW
reach) -- but note they fill VISUAL gaps without moving well metrics and can
worsen the deep band by extrapolating the shallow valley surface into deep
regional tables (see notes/regional_fac_rem.md).

Examples
--------
Reproduce one basin (bare command = the canonical profile recipe):
  uv run python utils/build_scalable_fac_rem.py --huc8 13030102 --name mesilla_repro

Whole state of New Mexico:
  uv run python utils/build_scalable_fac_rem.py --state NM \
      --out-root /data/ssd2/handily/nm/fac_rem
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import rasterio
import requests
import rioxarray  # noqa: F401 - registers the .rio accessor
from pystac_client import Client
from pystac_client.exceptions import APIError
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from shapely.geometry import box

from handily import regional_fac

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_scalable_fac_rem")

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = REPO / "configs/rem/profiles/conus_fac_rem_scalable.toml"

# CONUS substrate (exists everywhere in the lower 48).
HUC8_POLYS = "/data/ssd2/handily/conus/wte_gnn/huc8_polys.parquet"
# Authoritative national HUC8 layer for batch selection: carries the
# pipe-separated `states` column AND the 5070 geometry, so a per-state batch
# needs no other lookup.
WBD_HUC8 = "/nas/hydrography/HUC_Boundaries/wbd_national/wbdhu8_5070.parquet"
MODIS_JJA = "/data/ssd2/handily/conus/covariates/modis_ndvi_jja_mean.tif"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# Local JRC-GSW occurrence cache (populated once by utils/cache_gsw_tiles.py).
# When the tiles covering a region are present here, build_support reads them
# directly and never touches MPC -- removing the per-HUC8 live-MPC dependency for
# this static dataset. MPC stays the fallback for any region not yet cached.
GSW_CACHE_DIR = Path("/nas/hydrography/gsw/occurrence")

REM_RASTER = "fac_head_depth_rem_10m.tif"  # depth product rem_fac writes
WS_RASTER = "fac_rem_water_surface_10m.tif"  # strip-fill water SURFACE (elevation)
DONE_MARKER = ".fac_rem_done"  # per-HUC8 completion sentinel (JSON)

# Large, fully re-derivable intermediates deleted after a HUC8 succeeds (statewide
# disk control). Products (REM depth + water surface), the generated config,
# fac_rem_run.json, the seed/support evidence rasters, streams_regional.fgb, and
# basin_boundary.fgb are KEPT. The shared dem_tiles/ cache is kept until the whole
# batch finishes (it is not per-HUC8).
_CLEANUP_REGION = (
    # per-HUC8 DEM clip + build_regional_dem scratch
    "dem_10m.tif",
    "dem_10m.vrt",
    "basin_cutline.fgb",
    # WhiteboxTools FAC intermediates (compute_regional_fac)
    "dem_10m_filled.tif",
    "d8_pointer.tif",
    "flow_accumulation.tif",
    "streams_10m.tif",
    "stream_order.tif",
    "streams_raw.shp",
    "streams_raw.shx",
    "streams_raw.dbf",
    "streams_raw.prj",
    "streams_raw.cpg",
)
# rem_fac burn intermediates (live in the rem output dir alongside the products).
_CLEANUP_REM = (
    "fac_head_depth_sparse_10m.tif",
    "fac_normals_smoothed_dem.tif",
    "fac_normals_cross_sections.fgb",
)


# ---------------------------------------------------------------------------
# Region resolution
# ---------------------------------------------------------------------------


def resolve_region(args) -> tuple[object, str]:
    """Return (poly_5070, name) for exactly one of --huc8 / --aoi / --bbox."""
    given = [bool(args.huc8), bool(args.aoi), bool(args.bbox)]
    if sum(given) != 1:
        raise SystemExit("specify exactly one of --huc8, --aoi, --bbox")

    if args.huc8:
        polys = gpd.read_parquet(HUC8_POLYS).set_index("huc8")
        if args.huc8 not in polys.index:
            raise SystemExit(f"HUC8 {args.huc8} not found in {HUC8_POLYS}")
        poly = polys.loc[args.huc8, "geometry"]
        name = args.name or args.huc8
        log.info("region: HUC8 %s", args.huc8)
        return poly, name

    if args.aoi:
        gdf = gpd.read_file(args.aoi).to_crs(5070)
        poly = gdf.geometry.union_all()
        name = args.name or Path(args.aoi).stem
        log.info("region: AOI %s (%d features)", args.aoi, len(gdf))
        return poly, name

    minx, miny, maxx, maxy = args.bbox
    poly = (
        gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=args.bbox_crs)
        .to_crs(5070)
        .iloc[0]
    )
    name = args.name or "bbox_region"
    log.info("region: bbox %s (EPSG:%s)", args.bbox, args.bbox_crs)
    return poly, name


# ---------------------------------------------------------------------------
# Seed + support (generalized from utils/prep_mesilla_modis_gsw.py)
# ---------------------------------------------------------------------------


def _open(path):
    da = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)
    return da.rio.set_spatial_dims(x_dim="x", y_dim="y")


# MPC's STAC API and signed COG reads intermittently time out under load (observed
# failing repeatedly 2026-06-25 -- "request exceeded the maximum allowed time"); the
# same query succeeds on retry. Retry ONLY these known-transient classes so a
# persistent/real error still surfaces rather than being masked.
_TRANSIENT = (APIError, RasterioIOError, requests.exceptions.RequestException)


def _with_retries(fn, *, what, tries=5, base_delay=3.0):
    """Run ``fn()``, retrying transient MPC/network failures with exponential backoff.

    Re-raises the last error after ``tries`` attempts (so a HUC8 with a genuinely
    unreachable input still fails -- and, in batch mode, fails just that HUC8).
    """
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except _TRANSIENT as e:
            if attempt == tries:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            log.warning(
                "%s failed (attempt %d/%d): %s -- retrying in %.0fs",
                what,
                attempt,
                tries,
                e,
                delay,
            )
            time.sleep(delay)


def build_seed(dem_path: Path, out_path: Path, modis_jja: str, force: bool) -> Path:
    """MODIS-JJA NDVI climatology -> bilinear to the DEM grid."""
    if out_path.exists() and not force:
        log.info("seed exists: %s", out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dem = _open(dem_path)
    minx, miny, maxx, maxy = dem.rio.bounds()
    log.info("seed: MODIS-JJA NDVI -> %dm DEM grid", int(dem.rio.resolution()[0]))
    modis = _open(modis_jja).rio.clip_box(minx, miny, maxx, maxy)
    seed = modis.rio.reproject_match(dem, resampling=Resampling.bilinear)
    seed.rio.to_raster(out_path, compress="deflate", tiled=True)
    sv = seed.values[np.isfinite(seed.values)]
    log.info(
        "  NDVI min=%.3f med=%.3f max=%.3f -> %s",
        sv.min(),
        np.median(sv),
        sv.max(),
        out_path,
    )
    return out_path


def _local_gsw_tiles(
    bbox4326: tuple[float, float, float, float],
) -> list[Path] | None:
    """Cached GSW occurrence tiles that FULLY cover ``bbox4326`` (lon/lat).

    Returns the intersecting tile paths only when their union contains the bbox, so
    a partially-cached region falls back to MPC instead of silently dropping water
    support over the uncovered part. Returns None when the cache is absent, empty,
    or incomplete for this bbox.
    """
    if not GSW_CACHE_DIR.is_dir():
        return None
    tiles = sorted(GSW_CACHE_DIR.glob("*.tif"))
    if not tiles:
        return None
    want = box(*bbox4326)
    hits: list[Path] = []
    union = None
    for t in tiles:
        with rasterio.open(t) as ds:
            tb = ds.bounds
        geom = box(tb.left, tb.bottom, tb.right, tb.top)
        if geom.intersects(want):
            hits.append(t)
            union = geom if union is None else union.union(geom)
    # buffer ~0.1 m (1e-6 deg) so float noise at the shared 10-deg tile seams does
    # not spuriously fail the coverage test for a bbox spanning two abutting tiles.
    if not hits or not union.buffer(1e-6).contains(want):
        return None
    return hits


def build_support(
    dem_path: Path, out_path: Path, occ_threshold: int, force: bool
) -> Path:
    """JRC GSW occurrence -> permanent water -> nearest to the DEM grid.

    Each GSW tile is reproject_match'd to the DEM grid independently and combined
    with a per-pixel max, so basins spanning several tiles compose correctly (the
    Mesilla single-tile prep concatenated along x, which only works for one tile).
    """
    if out_path.exists() and not force:
        log.info("support exists: %s", out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dem = _open(dem_path)
    bbox4326 = tuple(dem.rio.transform_bounds("EPSG:4326"))

    def _accumulate(sources):
        # Each GSW occurrence source (0-100 = % of valid months with water; NaN = no
        # valid obs) is clipped, thresholded to permanent water, reproject_match'd to
        # the DEM grid, and combined with a per-pixel max so a region spanning several
        # tiles composes correctly. Shared by the local-cache and MPC paths.
        acc = None
        template = None
        for src in sources:
            occ = _open(src).rio.clip_box(*bbox4326)
            permanent = (occ >= occ_threshold).astype("uint8")
            permanent = permanent.rio.write_crs(occ.rio.crs).rio.write_nodata(0)
            m = permanent.rio.reproject_match(dem, resampling=Resampling.nearest)
            m = m.fillna(0).astype("uint8")
            template = m
            acc = m.values if acc is None else np.maximum(acc, m.values)
        return template, acc

    local = _local_gsw_tiles(bbox4326)
    if local is not None:
        log.info(
            "support: %d local GSW tile(s) cover bbox %s -> %s",
            len(local),
            bbox4326,
            [t.name for t in local],
        )
        template, acc = _accumulate([str(t) for t in local])
    else:
        log.info("support: query MPC jrc-gsw for bbox %s", bbox4326)

        def _fetch():
            # Search + sign + read + reproject as one unit so any transient failure
            # retries from a clean STAC search (signed COG URLs expire; reproject is
            # idempotent so redoing it is harmless).
            items = list(
                Client.open(STAC_URL)
                .search(collections=["jrc-gsw"], bbox=bbox4326)
                .items()
            )
            if not items:
                # GSW tiles cover all land; empty here is a real failure, not transient.
                raise RuntimeError(f"no jrc-gsw items for bbox {bbox4326}")
            log.info("  %d GSW tile(s) [MPC]: %s", len(items), [it.id for it in items])
            return _accumulate([pc.sign(it).assets["occurrence"].href for it in items])

        template, acc = _with_retries(_fetch, what="MPC jrc-gsw fetch")

    support = template.copy(data=acc)
    support.rio.to_raster(out_path, dtype="uint8", compress="deflate", tiled=True)
    n_water = int((acc == 1).sum())
    log.info(
        "  permanent-water (occ>=%d): %d px (%.3f%%) -> %s",
        occ_threshold,
        n_water,
        100.0 * n_water / acc.size,
        out_path,
    )
    return out_path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def write_config(
    cfg_path: Path,
    profile: Path,
    dem_path: Path,
    streams_path: Path,
    fac_path: Path,
    seed_path: Path,
    support_path: Path,
    rem_out: Path,
    args,
) -> Path:
    """Write a rem_fac TOML inheriting the profile, overlaying ONLY explicitly-set
    recipe overrides + the runtime worker count + [paths].

    The profile (default ``conus_fac_rem_scalable.toml``) is the single source of
    truth for the recipe. Each recipe knob's argparse default is ``None``; a block
    or key is emitted here only when the operator passed a value, so an unset knob
    inherits from the profile and the CLI is a pure override layer. (Previously the
    argparse defaults were baked into every generated config, silently overriding
    the profile -- which is how the defaults became the de-facto recipe.)
    """
    # (section, key, value) for every override the operator explicitly set.
    overrides: dict[str, dict[str, object]] = {}

    def put(section: str, key: str, value):
        if value is not None:
            overrides.setdefault(section, {})[key] = value

    put("seed", "ndvi_mid", args.ndvi_mid)
    put("seed", "ndvi_scale", args.ndvi_scale)
    put("strips", "max_crossing_strip_m", args.max_crossing_strip_m)
    put("strips", "naked_fill_m", args.naked_fill_m)
    # workers is a runtime knob (never affects output); always pin it explicitly.
    overrides.setdefault("strips", {})["workers"] = args.workers
    put("raster", "idw_radius_m", args.idw_radius_m)
    put("propagation", "down_distance_scale_m", args.down_distance_scale_m)
    put("propagation", "elevation_scale_m", args.elevation_scale_m)
    put("propagation", "strahler_distance_scale", args.strahler_distance_scale)
    put("solver", "below_bed_offset_m", args.below_bed_offset_m)
    put("solver", "d_min_off_support_m", args.d_min_off_support_m)

    lines = [
        f"# Auto-generated by utils/build_scalable_fac_rem.py for '{args.name}'.",
        "# Scalable / NHD-free FAC-REM (WBT streams + MODIS-JJA seed + JRC GSW",
        "# support). Recipe is defined by the inherited profile; only explicit CLI",
        "# overrides + [paths] appear below. Validate ONLY against GWX unconfined",
        "# wells + NHD springs, never against Ma.",
        "",
        f'profile = "{profile}"',
        "",
    ]
    for section, kv in overrides.items():
        lines.append(f"[{section}]")
        for key, value in kv.items():
            lines.append(f"{key} = {value}")
        lines.append("")
    lines += [
        "[paths]",
        f'dem_path = "{dem_path}"',
        f'streams_path = "{streams_path}"',
        f'fac_path = "{fac_path}"',
        f'ndvi_path = "{seed_path}"',
        f'support_path = "{support_path}"',
        f'out_dir = "{rem_out}"',
        "",
    ]
    cfg_path.write_text("\n".join(lines))
    log.info("wrote config %s", cfg_path)
    return cfg_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _force_clean_region(out_dir: Path) -> None:
    """Wipe a region's derived outputs so a --force rebuild regenerates EVERY stage.

    ``build_regional_dem`` / ``compute_regional_fac`` skip-if-exists and take no force
    flag, so without this a forced rebuild would reuse the stale DEM + streams and
    silently ignore a changed --halo-km / --stream-threshold / profile (only the seed
    and support, which do take force, would rebuild -- on top of the old grid). Remove
    everything except a local ``dem_tiles/`` cache (single-region layout); the batch's
    shared tile cache lives in the out-root, outside out_dir, so it is never touched.
    """
    if not out_dir.is_dir():
        return
    for child in out_dir.iterdir():
        if child.name == "dem_tiles":  # preserve the expensive 3DEP download cache
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def build_one_region(
    poly, name: str, out_dir: Path, args, *, dem_tiles_dir: Path | None = None
) -> Path | None:
    """Build the scalable 10 m FAC-REM for one region.

    Returns the REM depth raster path on success, or ``None`` when ``--no-run``
    (inputs + config prepped, rem_fac skipped). Raises ``RuntimeError`` on any hard
    failure (no DEM tiles, rem_fac non-zero, missing output) so a batch caller can
    catch it with ``except Exception`` and keep going (``SystemExit`` would not be
    caught and would abort the batch).
    """
    args.name = name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("name=%s out_dir=%s", name, out_dir)

    # --force means rebuild from scratch: clear stale derived outputs so a changed
    # halo / threshold / profile actually takes effect (the DEM + FAC stages
    # skip-if-exists and would otherwise reuse the old grid).
    if args.force:
        _force_clean_region(out_dir)

    halo_m = args.halo_km * 1000.0
    halo = poly.buffer(halo_m)
    bbox_wgs84 = tuple(gpd.GeoSeries([halo], crs=5070).to_crs(4326).total_bounds)

    # 1. DEM. In a batch, dem_tiles_dir is the shared out-root cache so adjacent
    # HUC8 halos reuse downloaded 3DEP tiles (download_3dep_10m_tiles is
    # concurrency-safe: per-PID temp + atomic rename + skip-if-exists).
    dem_tiles = (
        Path(dem_tiles_dir) if dem_tiles_dir is not None else out_dir / "dem_tiles"
    )
    tiles = regional_fac.download_3dep_10m_tiles(bbox_wgs84, dem_tiles)
    if not tiles:
        raise RuntimeError(f"no 3DEP tiles for bbox {bbox_wgs84} (border/ocean?)")
    dem_path = out_dir / "dem_10m.tif"
    regional_fac.build_regional_dem(
        tiles,
        gpd.GeoDataFrame({"geometry": [poly]}, crs=5070),
        dem_path,
        target_crs_epsg=5070,
        buffer_m=halo_m,
    )

    # 2. WBT FAC + streams
    regional_fac.compute_regional_fac(
        dem_path, out_dir, threshold=args.stream_threshold, max_procs=args.workers
    )
    streams_path = out_dir / "streams_regional.fgb"
    fac_path = out_dir / "flow_accumulation.tif"

    # 3-4. Seed + support
    evidence = out_dir / "evidence" / "scalable"
    seed_path = build_seed(
        dem_path,
        evidence / f"{name}_modis_jja_ndvi_10m.tif",
        args.modis_jja,
        args.force,
    )
    support_path = build_support(
        dem_path,
        evidence / f"{name}_gsw_permanent_10m.tif",
        args.occ_threshold,
        args.force,
    )

    # 5. Config
    rem_out = out_dir / "rem" / f"{name}_scalable"
    cfg = write_config(
        out_dir / f"{name}_scalable.toml",
        Path(args.profile).resolve(),
        dem_path,
        streams_path,
        fac_path,
        seed_path,
        support_path,
        rem_out,
        args,
    )

    if args.no_run:
        log.info("--no-run: inputs + config ready. Run rem_fac with:")
        log.info("  uv run python -m handily.rem_fac --config %s --no-strip-debug", cfg)
        return None

    # 6. rem_fac (fresh subprocess so one failure is isolated)
    rem_out.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "rem_fac.log"
    log.info("running rem_fac (log: %s) ...", run_log)
    with open(run_log, "w") as fh:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "handily.rem_fac",
                "--config",
                str(cfg),
                "--no-strip-debug",
            ],
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"rem_fac failed (rc={proc.returncode}); see {run_log}")

    rem_path = rem_out / REM_RASTER
    if not rem_path.exists():
        raise RuntimeError(f"rem_fac produced no REM raster at {rem_path}")
    log.info("done -> %s", rem_path)
    log.info(
        "pull for QGIS: rsync -rav zoran:%s ~%s",
        rem_out,
        str(rem_out).replace("/data/ssd2", "/data", 1),
    )
    return rem_path


# ---------------------------------------------------------------------------
# Batch mode (statewide / HUC8-list)
# ---------------------------------------------------------------------------


def resolve_batch_huc8s(args) -> list[tuple[str, object]]:
    """Return [(huc8, poly_5070), ...] for --state and/or --huc8-list.

    Both codes and geometry come from the authoritative national WBD HUC8 layer
    (``WBD_HUC8``), whose ``states`` column is a pipe-separated list of the state
    abbreviations a HUC8 touches.
    """
    gdf = gpd.read_parquet(WBD_HUC8)
    if "huc8" not in gdf.columns:
        raise SystemExit(f"{WBD_HUC8} lacks a 'huc8' column")
    if gdf.crs is None or gdf.crs.to_epsg() != 5070:
        gdf = gdf.to_crs(5070)
    gdf = gdf.drop_duplicates("huc8").set_index("huc8")

    codes: list[str] = []
    if args.huc8_list:
        codes += [
            ln.strip()
            for ln in Path(args.huc8_list).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    if args.state:
        want = {s.upper() for s in args.state}
        touched = gdf["states"].fillna("").str.upper().str.split("|")
        mask = touched.apply(lambda lst: bool(want.intersection(lst)))
        codes += gdf.index[mask].tolist()

    seen: set[str] = set()
    targets: list[tuple[str, object]] = []
    for c in codes:
        if c in seen:
            continue
        seen.add(c)
        if c not in gdf.index:
            log.warning("HUC8 %s not in %s -> skip", c, WBD_HUC8)
            continue
        targets.append((c, gdf.loc[c, "geometry"]))
    if not targets:
        raise SystemExit("no HUC8 targets resolved (check --state / --huc8-list)")
    return targets


def _write_marker(marker: Path, huc8: str, profile: str, rem_path: Path) -> None:
    """Write the per-HUC8 completion sentinel only after the REM raster is on disk."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "huc8": huc8,
                "profile": str(profile),
                "rem_path": str(rem_path),
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )


def _cleanup_region(out_dir: Path, rem_path: Path) -> None:
    """Delete large re-derivable intermediates after a HUC8 succeeds. Keeps the
    products, generated config, fac_rem_run.json, seed/support evidence,
    streams_regional.fgb, and basin_boundary.fgb."""
    removed = 0
    for name in _CLEANUP_REGION:
        p = out_dir / name
        if p.exists():
            p.unlink()
            removed += 1
    rem_dir = rem_path.parent
    for name in _CLEANUP_REM:
        p = rem_dir / name
        if p.exists():
            p.unlink()
            removed += 1
    log.info("cleaned %d intermediates in %s", removed, out_dir)


def run_batch(args) -> None:
    """Iterate the resolved HUC8 set, build each, resume-skip the done ones, and
    tally built/skipped/failed. One HUC8 failure never aborts the batch."""
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    shared_tiles = out_root / "dem_tiles"  # reused across HUC8s
    targets = resolve_batch_huc8s(args)
    log.info("batch: %d HUC8 target(s) -> %s", len(targets), out_root)

    built = skipped = failed = 0
    failed_ids: list[str] = []
    for huc8, poly in targets:
        out_dir = out_root / huc8
        marker = out_dir / DONE_MARKER
        rem_path = out_dir / "rem" / f"{huc8}_scalable" / REM_RASTER
        if not args.force and marker.exists() and rem_path.exists():
            log.info("[%s] done (marker + REM raster) -> skip", huc8)
            skipped += 1
            continue
        # Invalidate the completion state BEFORE any (re)build so an interrupted
        # or failed rebuild can never leave a believable-complete marker behind.
        marker.unlink(missing_ok=True)
        try:
            produced = build_one_region(
                poly, huc8, out_dir, args, dem_tiles_dir=shared_tiles
            )
        except Exception as e:  # noqa: BLE001 - isolate per-HUC8 failures
            log.exception("[%s] build failed: %s", huc8, e)
            failed += 1
            failed_ids.append(huc8)
            continue
        if args.no_run:
            skipped += 1  # prepped, not built
            continue
        if produced is None or not produced.exists():
            log.error("[%s] no REM raster after build", huc8)
            failed += 1
            failed_ids.append(huc8)
            continue
        if not args.keep_intermediates:
            _cleanup_region(out_dir, produced)
        _write_marker(marker, huc8, args.profile, produced)
        built += 1

    log.info(
        "batch complete: built=%d skipped=%d failed=%d (of %d)",
        built,
        skipped,
        failed,
        len(targets),
    )
    if failed_ids:
        log.warning(
            "%d failed HUC8(s) -> rerun to retry: %s", len(failed_ids), failed_ids
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Single region (exactly one of these); ignored in batch mode.
    p.add_argument("--huc8", help="HUC8 code (resolved from the WBD HUC8 layer)")
    p.add_argument("--aoi", help="polygon vector file (any CRS)")
    p.add_argument(
        "--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY")
    )
    p.add_argument(
        "--bbox-crs", default="EPSG:4326", help="CRS of --bbox (default EPSG:4326)"
    )
    p.add_argument("--name", help="region label for output naming")
    p.add_argument(
        "--out-dir",
        help="single-region working dir (default /data/ssd2/handily/scalable_fac_rem/<name>)",
    )

    # Batch mode (statewide / HUC8-list). Either flag switches to batch.
    p.add_argument(
        "--state",
        nargs="*",
        help="build every HUC8 touching these state abbrevs (e.g. --state NM MT NV)",
    )
    p.add_argument("--huc8-list", help="file with one HUC8 code per line")
    p.add_argument(
        "--out-root",
        default="/data/ssd2/handily/scalable_fac_rem",
        help="batch root; each HUC8 -> <out-root>/<huc8>/, shared <out-root>/dem_tiles/",
    )
    p.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="batch: keep large re-derivable intermediates (default: delete on success)",
    )

    # Inputs
    p.add_argument(
        "--profile", default=str(DEFAULT_PROFILE), help="rem_fac profile TOML"
    )
    p.add_argument(
        "--modis-jja", default=MODIS_JJA, help="MODIS-JJA NDVI climatology raster"
    )
    p.add_argument(
        "--halo-km",
        type=float,
        default=10.0,
        help="DEM/streams halo (km). 10 km matches the validated 2026-06-25 MT/NM/NV "
        "runs: a wider halo captures the upland catchment that drains into the valley, "
        "so tributaries reach the stream threshold farther upstream (denser network, "
        "fuller strip-fill coverage) and adjacent HUC8 tiles overlap enough to heal "
        "seams. 5 km under-covers the interior (~11% coverage loss vs the 10 km run).",
    )
    p.add_argument(
        "--stream-threshold", type=int, default=5000, help="WBT extract_streams cells"
    )
    p.add_argument(
        "--occ-threshold",
        type=int,
        default=90,
        help="GSW permanent-water occurrence %%",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=32,
        help="parallelism for WBT FAC + rem_fac strip-gen (worker count never "
        "affects output, only runtime)",
    )

    # Recipe overrides. Default None -> inherit the profile (the single source of
    # truth, conus_fac_rem_scalable.toml). Pass a value only to deviate from it.
    p.add_argument("--ndvi-mid", type=float, default=None)
    p.add_argument("--ndvi-scale", type=float, default=None)
    p.add_argument("--max-crossing-strip-m", type=float, default=None)
    p.add_argument("--naked-fill-m", type=float, default=None)
    p.add_argument("--idw-radius-m", type=float, default=None)
    p.add_argument("--down-distance-scale-m", type=float, default=None)
    p.add_argument("--elevation-scale-m", type=float, default=None)
    p.add_argument("--strahler-distance-scale", type=float, default=None)
    p.add_argument("--below-bed-offset-m", type=float, default=None)
    p.add_argument("--d-min-off-support-m", type=float, default=None)

    # Control
    p.add_argument(
        "--no-run", action="store_true", help="prep inputs + write config, skip rem_fac"
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if present (seed/support; in batch, a done HUC8 too)",
    )
    args = p.parse_args(argv)

    if args.state or args.huc8_list:
        run_batch(args)
        return

    poly, name = resolve_region(args)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(f"/data/ssd2/handily/scalable_fac_rem/{name}")
    )
    try:
        build_one_region(poly, name, out_dir, args)
    except RuntimeError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
