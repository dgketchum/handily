"""Restore the per-basin 10 m DEM (``dem_10m.tif``) the FAC build deletes on success.

``build_scalable_fac_rem`` removes ``dem_10m.tif`` (and other re-derivable intermediates;
see its ``_CLEANUP_REGION``) after each HUC8 FAC solve to control disk. Downstream
products that need a GROUND elevation -- notably the ``str_top2`` regional-WTE prior
(``build_str7_idw_raster``, which samples ``dem_10m.tif`` for stream-anchor elevations and
the 100 m ground grid) -- then have no DEM to read.

This re-derives ``dem_10m.tif`` for every basin that has a FAC product +
``basin_boundary.fgb`` but no DEM, reusing the exact FAC DEM stage
(``regional_fac.build_regional_dem``) and the shared 3DEP tile cache, so it is a
clip/merge from disk -- NOT a re-download (only genuinely-uncached tiles are fetched).
Idempotent: skips basins whose DEM already exists. Per-basin failures are isolated.

See ``notes/HUC8_DATA.md``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handily import regional_fac  # noqa: E402

log = logging.getLogger("restore_huc8_dems")

ROOT = Path("/data/ssd2/handily/huc8")
HALO_KM = (
    10.0  # matches build_scalable_fac_rem --halo-km default (validated MT/NM/NV runs)
)
DEM_NAME = "dem_10m.tif"


def _fac_product(basin: str) -> Path:
    return ROOT / basin / "rem" / f"{basin}_scalable" / "fac_head_depth_rem_10m.tif"


def needs_dem(basin: str) -> bool:
    """A basin we CAN and SHOULD restore: has a FAC product + boundary, lacks a DEM."""
    d = ROOT / basin
    return (
        (d / "basin_boundary.fgb").exists()
        and _fac_product(basin).exists()
        and not (d / DEM_NAME).exists()
    )


def restore_one(basin: str, halo_m: float, tiles_dir: Path) -> Path:
    """Re-derive ``dem_10m.tif`` for one basin from the shared 3DEP tile cache."""
    out_dir = ROOT / basin
    dem_path = out_dir / DEM_NAME
    boundary = gpd.read_file(out_dir / "basin_boundary.fgb").to_crs(5070)
    poly = boundary.union_all()
    halo = poly.buffer(halo_m)
    bbox_wgs84 = tuple(gpd.GeoSeries([halo], crs=5070).to_crs(4326).total_bounds)
    tiles = regional_fac.download_3dep_10m_tiles(bbox_wgs84, tiles_dir)
    if not tiles:
        raise RuntimeError(f"no 3DEP tiles for bbox {bbox_wgs84} (border/ocean?)")
    regional_fac.build_regional_dem(
        tiles,
        gpd.GeoDataFrame({"geometry": [poly]}, crs=5070),
        dem_path,
        target_crs_epsg=5070,
        buffer_m=halo_m,
    )
    return dem_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--basins",
        nargs="*",
        help="basin names under huc8/ (default: every basin needing a DEM)",
    )
    ap.add_argument("--halo-km", type=float, default=HALO_KM)
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    tiles_dir = ROOT / "dem_tiles"
    halo_m = args.halo_km * 1000.0
    if args.basins:
        basins = list(args.basins)
    else:
        basins = sorted(
            d.name for d in ROOT.iterdir() if d.is_dir() and needs_dem(d.name)
        )
    log.info("restoring DEM for %d basin(s) (halo %.0f km)", len(basins), args.halo_km)

    restored = skipped = failed = 0
    failed_ids: list[str] = []
    for basin in basins:
        if (ROOT / basin / DEM_NAME).exists():
            log.info("[%s] DEM already exists -> skip", basin)
            skipped += 1
            continue
        try:
            p = restore_one(basin, halo_m, tiles_dir)
        except Exception as e:  # noqa: BLE001 - isolate per-basin failures
            log.exception("[%s] DEM restore failed: %s", basin, e)
            failed += 1
            failed_ids.append(basin)
            continue
        log.info("[%s] DEM -> %s", basin, p)
        restored += 1

    log.info("done: restored=%d skipped=%d failed=%d", restored, skipped, failed)
    if failed_ids:
        log.warning("%d failed -> rerun to retry: %s", len(failed_ids), failed_ids)


if __name__ == "__main__":
    main()
