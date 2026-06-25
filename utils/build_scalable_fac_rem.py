"""Build a scalable, NHD-free FAC-REM for any region in CONUS.

Reproduces the accepted Mesilla recipe (configs/rem/nm_mesilla_modis_gsw.toml,
the best regional FAC-REM so far) end to end from CONUS-available substrate only
-- no LiDAR, no NAIP, no NHD. Every input is derived from sources that exist
everywhere in the lower 48, so the same command works for any HUC8, AOI polygon,
or bounding box.

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
  5. CONFIG   : write a rem_fac TOML inheriting the FAC10 profile, with the
                arid-regime overrides (defaults below = the accepted Mesilla
                values) and [paths] pointing at the built inputs.
  6. RUN      : invoke ``python -m handily.rem_fac`` as a subprocess (skip with
                --no-run) -> fac_head_depth_rem_10m.tif (depth) +
                fac_rem_water_surface_10m.tif (elevation).

Region is given exactly one of:
  --huc8 CODE          resolve the polygon from the WBD HUC8 layer.
  --aoi PATH           a polygon vector file (any CRS) -> reprojected to 5070.
  --bbox MINX MINY MAXX MAXY [--bbox-crs EPSG]   an explicit box.

Validate the result ONLY against GWX unconfined/marginal wells + NHD springs
(utils/validate_fac_gwx_wells.py); never against Ma. Two knobs widen the filled
footprint when the REM has visual gaps in deep upland inter-channel zones --
--max-crossing-strip-m (longer cross-sections) and --idw-radius-m (longer IDW
reach) -- but note they fill VISUAL gaps without moving well metrics and can
worsen the deep band by extrapolating the shallow valley surface into deep
regional tables (see notes/regional_fac_rem.md).

Examples
--------
Reproduce Mesilla exactly:
  uv run python utils/build_scalable_fac_rem.py --huc8 13030102 --name mesilla_repro

A new basin from a bbox:
  uv run python utils/build_scalable_fac_rem.py --bbox -107.0 32.0 -106.5 32.6 \
      --name some_basin --out-dir /data/ssd2/handily/scalable_fac_rem/some_basin
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import rioxarray  # noqa: F401 - registers the .rio accessor
from pystac_client import Client
from rasterio.enums import Resampling
from shapely.geometry import box

from handily import regional_fac

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_scalable_fac_rem")

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = REPO / "configs/rem/profiles/ruby_fac10_baseline.toml"

# CONUS substrate (exists everywhere in the lower 48).
HUC8_POLYS = "/data/ssd2/handily/conus/wte_gnn/huc8_polys.parquet"
MODIS_JJA = "/data/ssd2/handily/conus/covariates/modis_ndvi_jja_mean.tif"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

REM_RASTER = "fac_head_depth_rem_10m.tif"  # depth product rem_fac writes


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
    log.info("support: query MPC jrc-gsw for bbox %s", bbox4326)
    items = list(
        Client.open(STAC_URL).search(collections=["jrc-gsw"], bbox=bbox4326).items()
    )
    if not items:
        raise SystemExit(f"no jrc-gsw items for bbox {bbox4326}")
    log.info("  %d GSW tile(s): %s", len(items), [it.id for it in items])

    acc = None
    template = None
    for it in items:
        href = pc.sign(it).assets["occurrence"].href
        # occurrence 0-100 (% of valid months with water); NaN = no valid obs.
        occ = _open(href).rio.clip_box(*bbox4326)
        permanent = (occ >= occ_threshold).astype("uint8")
        permanent = permanent.rio.write_crs(occ.rio.crs).rio.write_nodata(0)
        m = permanent.rio.reproject_match(dem, resampling=Resampling.nearest)
        m = m.fillna(0).astype("uint8")
        template = m
        acc = m.values if acc is None else np.maximum(acc, m.values)

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
    """Write a rem_fac TOML: FAC10 profile + arid-regime overrides + [paths]."""
    raster_block = ""
    if args.idw_radius_m is not None:
        raster_block = f"\n[raster]\nidw_radius_m = {args.idw_radius_m}\n"

    cfg_path.write_text(
        f"# Auto-generated by utils/build_scalable_fac_rem.py for '{args.name}'.\n"
        f"# Scalable / NHD-free FAC-REM (WBT streams + MODIS-JJA seed + JRC GSW\n"
        f"# support). Reproduces the accepted Mesilla recipe. Validate ONLY against\n"
        f"# GWX unconfined wells + NHD springs, never against Ma.\n\n"
        f'profile = "{profile}"\n\n'
        "[seed]\n"
        f"ndvi_mid = {args.ndvi_mid}\n"
        f"ndvi_scale = {args.ndvi_scale}\n\n"
        "[strips]\n"
        f"max_crossing_strip_m = {args.max_crossing_strip_m}\n"
        f"naked_fill_m = {args.naked_fill_m}\n"
        f"{raster_block}\n"
        "[propagation]\n"
        f"down_distance_scale_m = {args.down_distance_scale_m}\n"
        f"elevation_scale_m = {args.elevation_scale_m}\n"
        f"strahler_distance_scale = {args.strahler_distance_scale}\n\n"
        "[solver]\n"
        f"below_bed_offset_m = {args.below_bed_offset_m}\n"
        f"d_min_off_support_m = {args.d_min_off_support_m}\n\n"
        "[paths]\n"
        f'dem_path = "{dem_path}"\n'
        f'streams_path = "{streams_path}"\n'
        f'fac_path = "{fac_path}"\n'
        f'ndvi_path = "{seed_path}"\n'
        f'support_path = "{support_path}"\n'
        f'out_dir = "{rem_out}"\n'
    )
    log.info("wrote config %s", cfg_path)
    return cfg_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Region (exactly one)
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
        help="region working dir (default /data/ssd2/handily/scalable_fac_rem/<name>)",
    )

    # Inputs
    p.add_argument(
        "--profile", default=str(DEFAULT_PROFILE), help="rem_fac profile TOML"
    )
    p.add_argument(
        "--modis-jja", default=MODIS_JJA, help="MODIS-JJA NDVI climatology raster"
    )
    p.add_argument("--halo-km", type=float, default=5.0, help="DEM/streams halo (km)")
    p.add_argument(
        "--stream-threshold", type=int, default=5000, help="WBT extract_streams cells"
    )
    p.add_argument(
        "--occ-threshold",
        type=int,
        default=90,
        help="GSW permanent-water occurrence %%",
    )
    p.add_argument("--workers", type=int, default=32, help="WBT max_procs")

    # Arid-regime overrides (defaults = the accepted Mesilla values)
    p.add_argument("--ndvi-mid", type=float, default=0.23)
    p.add_argument("--ndvi-scale", type=float, default=0.035)
    p.add_argument("--max-crossing-strip-m", type=float, default=1500.0)
    p.add_argument("--naked-fill-m", type=float, default=300.0)
    p.add_argument(
        "--idw-radius-m", type=float, default=None, help="override profile IDW radius"
    )
    p.add_argument("--down-distance-scale-m", type=float, default=8000.0)
    p.add_argument("--elevation-scale-m", type=float, default=15.0)
    p.add_argument("--strahler-distance-scale", type=float, default=1.0)
    p.add_argument("--below-bed-offset-m", type=float, default=1.5)
    p.add_argument("--d-min-off-support-m", type=float, default=1.0)

    # Control
    p.add_argument(
        "--no-run", action="store_true", help="prep inputs + write config, skip rem_fac"
    )
    p.add_argument(
        "--force", action="store_true", help="rebuild seed/support even if present"
    )
    args = p.parse_args(argv)

    poly, name = resolve_region(args)
    args.name = name
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(f"/data/ssd2/handily/scalable_fac_rem/{name}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("name=%s out_dir=%s", name, out_dir)

    halo_m = args.halo_km * 1000.0
    halo = poly.buffer(halo_m)
    bbox_wgs84 = tuple(gpd.GeoSeries([halo], crs=5070).to_crs(4326).total_bounds)

    # 1. DEM
    dem_tiles = out_dir / "dem_tiles"
    tiles = regional_fac.download_3dep_10m_tiles(bbox_wgs84, dem_tiles)
    if not tiles:
        raise SystemExit(f"no 3DEP tiles for bbox {bbox_wgs84} (border/ocean?)")
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
        return

    # 6. rem_fac
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
        raise SystemExit(f"rem_fac failed (rc={proc.returncode}); see {run_log}")

    rem_path = rem_out / REM_RASTER
    if not rem_path.exists():
        raise SystemExit(f"rem_fac produced no REM raster at {rem_path}")
    log.info("done -> %s", rem_path)
    log.info(
        "pull for QGIS: rsync -rav zoran:%s ~%s",
        rem_out,
        str(rem_out).replace("/data/ssd2", "/data", 1),
    )


if __name__ == "__main__":
    main()
