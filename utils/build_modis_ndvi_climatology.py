"""Build a MODIS seasonal NDVI climatology (CONUS, 250 m) from MOD13Q1 on MPC.

This is the Tier-3 *real-deep* discriminator upgrade: a static aridity index (P/PET)
flags arid basins, but long-term vegetation vigor / actual water use is a stronger
index of how deep the regional table sits. Low growing-season NDVI in an arid basin
=> deep regional table; high riparian/humid NDVI => shallow. Feeding seasonal-mean
NDVI (and Kcb = NDVI * 1.25 downstream for a soil-water-balance model) gives the
stacker a vegetation-water-use covariate the relief/aridity features cannot supply.

Source: Microsoft Planetary Computer (free, no Earth Engine quota). Collection
`modis-13Q1-061` (250 m, 16-day Maximum-Value Composite) serves BOTH Terra (MOD13Q1)
and Aqua (MYD13Q1) granules; `--platform` selects which (default `terra` = MOD13Q1
only, the single 16-day series; `both` interleaves to an effective 8-day sampling).
`--datetime` selects the window (default 2020-2025). Each composite is already
cloud-minimized (MVC); we further mask `pixel_reliability` (keep 0 good / 1 marginal;
drop 2 snow-ice, 3 cloud, 255 fill) and the NDVI fill (-3000) / out-of-range, then
average per season across the window. We never store the archive -- we stream each
composite, accumulate per-tile sum/count, and discard.

Pipeline:
  1. Search MPC for all MOD13Q1 composites intersecting CONUS over the POR.
  2. Group by MODIS sinusoidal tile (h##v##). Per tile, stream every composite,
     mask, accumulate per-season + annual sum/count, write per-tile mean COGs
     (native sinusoidal grid). Resume = skip tiles whose outputs already exist.
  3. gdalwarp-mosaic the per-tile means -> CONUS EPSG:5070 250 m rasters
     (matching the relief covariates so build_stacker_features samples them at
     well x5070/y5070).

Outputs in <out-dir>:
  modis_ndvi_djf_mean.tif  modis_ndvi_mam_mean.tif  modis_ndvi_jja_mean.tif
  modis_ndvi_son_mean.tif  modis_ndvi_annual_mean.tif

Usage:
    uv run python utils/build_modis_ndvi_climatology.py \
        --out-dir /data/ssd2/handily/conus/covariates --workers 4
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import time
from collections import defaultdict
from datetime import date, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client

log = logging.getLogger("build_modis_ndvi_climatology")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "modis-13Q1-061"
NDVI_ASSET = "250m_16_days_NDVI"
REL_ASSET = "250m_16_days_pixel_reliability"

CONUS_BBOX = (-125.0, 24.0, -66.5, 49.5)
DATETIME = "2020-01-01/2025-12-31"  # window; partial years bias season means slightly
PLATFORM = "terra"  # "terra"=MOD13Q1, "aqua"=MYD13Q1, "both"=interleaved (8-day eff.)

NDVI_FILL = -3000
NDVI_MIN, NDVI_MAX = -2000, 10000  # valid VI range; scale 1e-4
NDVI_SCALE = 1e-4
GOOD_REL = (0, 1)  # good + marginal; 2 snow-ice / 3 cloud / 255 fill are dropped

SEASONS = {
    "djf": (12, 1, 2),
    "mam": (3, 4, 5),
    "jja": (6, 7, 8),
    "son": (9, 10, 11),
}
ANNUAL = "annual"
BANDS = (*SEASONS.keys(), ANNUAL)

OUT_DIR = "/data/ssd2/handily/conus/covariates"
TARGET_CRS = "EPSG:5070"
TARGET_RES = 250.0

_TILE_RE = re.compile(r"\.(h\d\dv\d\d)\.")
_DATE_RE = re.compile(r"\.A(\d{4})(\d{3})\.")
_PRODUCT_RE = re.compile(r"^(MOD|MYD)13Q1\.")


def granule_platform(item_id: str) -> str:
    """Platform from the product token: MOD13Q1 -> 'terra', MYD13Q1 -> 'aqua'.

    The `modis-13Q1-061` collection mixes both; this lets us keep one series.
    """
    m = _PRODUCT_RE.match(item_id)
    if not m:
        raise ValueError(f"no MOD/MYD13Q1 product token in item id: {item_id}")
    return "terra" if m.group(1) == "MOD" else "aqua"


def tile_id(item_id: str) -> str:
    """Parse the MODIS sinusoidal tile (e.g. 'h10v04') from a granule id."""
    m = _TILE_RE.search(item_id)
    if not m:
        raise ValueError(f"no h##v## tile in item id: {item_id}")
    return m.group(1)


def granule_month(item_id: str) -> int:
    """Composite month from the granule date code (e.g. 'A2015209' -> year+DOY).

    MOD13Q1 items carry a 16-day range, so item.datetime is null; the granule id
    is the deterministic, offline source of the composite date.
    """
    m = _DATE_RE.search(item_id)
    if not m:
        raise ValueError(f"no A<year><doy> date in item id: {item_id}")
    yr, doy = int(m.group(1)), int(m.group(2))
    return (date(yr, 1, 1) + timedelta(days=doy - 1)).month


def season_of(month: int) -> str:
    for name, months in SEASONS.items():
        if month in months:
            return name
    raise ValueError(f"month out of range: {month}")


def valid_mask(ndvi: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Pixels that carry a usable NDVI: good/marginal reliability, in-range, not fill."""
    return (
        np.isin(rel, GOOD_REL)
        & (ndvi != NDVI_FILL)
        & (ndvi >= NDVI_MIN)
        & (ndvi <= NDVI_MAX)
    )


def scaled_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """Raw int16 DN -> physical NDVI (float32)."""
    return (ndvi.astype(np.float32)) * np.float32(NDVI_SCALE)


def seasonal_mean(sum_arr: np.ndarray, count_arr: np.ndarray) -> np.ndarray:
    """Mean where any valid sample exists, NaN where the season is never clear."""
    out = np.full(sum_arr.shape, np.nan, dtype=np.float32)
    seen = count_arr > 0
    out[seen] = sum_arr[seen] / count_arr[seen]
    return out


def _read_band(href: str, retries: int = 4):
    """Sign a blob URL and read its single band, retrying transient I/O.

    Returns (array, profile). A failure here is an Azure/network transient, not
    missing data -- we retry with backoff and let the caller count hard skips.
    """
    last = None
    for attempt in range(retries):
        try:
            signed = pc.sign(href)
            with rasterio.open(signed) as ds:
                return ds.read(1), ds.profile
        except Exception as exc:  # transient HTTP/SAS/GDAL I/O
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"read failed after {retries} tries: {href}") from last


def accumulate_tile(tile: str, records: list[dict], tile_dir: str) -> dict:
    """Stream every composite for one tile -> per-season + annual mean COGs.

    records: [{"month": int, "ndvi": href, "rel": href}, ...]
    """
    Path(tile_dir).mkdir(parents=True, exist_ok=True)
    out_paths = {b: f"{tile_dir}/{tile}_{b}.tif" for b in BANDS}
    if all(Path(p).exists() for p in out_paths.values()):
        return {"tile": tile, "status": "skip-exists", "used": 0, "skipped": 0}

    sums = {b: None for b in BANDS}
    counts = {b: None for b in BANDS}
    profile = None
    used = skipped = 0
    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MAX_RETRY="3",
        GDAL_HTTP_RETRY_DELAY="2",
    )
    with env:
        for rec in records:
            try:
                ndvi, prof = _read_band(rec["ndvi"])
                rel, _ = _read_band(rec["rel"])
            except RuntimeError:
                skipped += 1
                continue
            if profile is None:
                profile = prof
                for b in BANDS:
                    sums[b] = np.zeros(ndvi.shape, dtype=np.float32)
                    counts[b] = np.zeros(ndvi.shape, dtype=np.uint16)
            m = valid_mask(ndvi, rel)
            vals = np.where(m, scaled_ndvi(ndvi), np.float32(0))
            inc = m.astype(np.uint16)
            s = season_of(rec["month"])
            sums[s] += vals
            counts[s] += inc
            sums[ANNUAL] += vals
            counts[ANNUAL] += inc
            used += 1

    if profile is None:
        return {"tile": tile, "status": "no-data", "used": 0, "skipped": skipped}

    out_prof = profile.copy()
    out_prof.update(
        dtype="float32", count=1, nodata=np.nan, compress="deflate", tiled=True
    )
    for b in BANDS:
        mean = seasonal_mean(sums[b], counts[b])
        with rasterio.open(out_paths[b], "w", **out_prof) as ds:
            ds.write(mean, 1)
    log.info("tile %s: used %d composites, skipped %d", tile, used, skipped)
    return {"tile": tile, "status": "ok", "used": used, "skipped": skipped}


def list_records(bbox, datetime_str, platform: str = PLATFORM) -> dict:
    """Search MPC and group unsigned (ndvi, rel) href records by MODIS tile.

    platform: "terra"/"aqua" keep only that series; "both" interleaves them.
    """
    cat = Client.open(STAC_URL)  # unsigned; we sign per-read (long run, SAS expiry)
    search = cat.search(collections=[COLLECTION], bbox=bbox, datetime=datetime_str)
    by_tile: dict[str, list[dict]] = defaultdict(list)
    n = dropped = 0
    for it in search.items():
        try:
            t = tile_id(it.id)
        except ValueError:
            continue
        if platform != "both" and granule_platform(it.id) != platform:
            dropped += 1
            continue
        by_tile[t].append(
            {
                "month": granule_month(it.id),
                "ndvi": it.assets[NDVI_ASSET].href,
                "rel": it.assets[REL_ASSET].href,
            }
        )
        n += 1
        if n % 1000 == 0:
            log.info("listed %d composites (%d tiles)", n, len(by_tile))
    log.info(
        "total %d composites over %d tiles (platform=%s, dropped %d other-platform)",
        n,
        len(by_tile),
        platform,
        dropped,
    )
    return dict(by_tile)


def mosaic_to_conus(tile_dir: str, out_dir: str) -> dict[str, str]:
    """gdalwarp the per-tile means for each band -> CONUS EPSG:5070 250 m COGs."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for b in BANDS:
        srcs = sorted(Path(tile_dir).glob(f"*_{b}.tif"))
        if not srcs:
            log.warning("no per-tile rasters for band %s -- skipping mosaic", b)
            continue
        out = f"{out_dir}/modis_ndvi_{b}_mean.tif"
        cmd = [
            "gdalwarp",
            "-t_srs",
            TARGET_CRS,
            "-tr",
            str(TARGET_RES),
            str(TARGET_RES),
            "-r",
            "bilinear",
            "-dstnodata",
            "nan",
            "-of",
            "GTiff",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "TILED=YES",
            "-co",
            "BIGTIFF=YES",
            "-overwrite",
            *[str(s) for s in srcs],
            out,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        paths[b] = out
        log.info("mosaicked %s from %d tiles -> %s", b, len(srcs), out)
    return paths


def build(
    out_dir: str = OUT_DIR,
    *,
    bbox=CONUS_BBOX,
    datetime_str: str = DATETIME,
    platform: str = PLATFORM,
    workers: int = 4,
    tile_dir: str | None = None,
) -> dict[str, str]:
    tile_dir = tile_dir or f"{out_dir}/modis_ndvi_tiles"
    by_tile = list_records(bbox, datetime_str, platform=platform)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(accumulate_tile, t, recs, tile_dir): t
            for t, recs in by_tile.items()
        }
        for fut in as_completed(futs):
            results.append(fut.result())
    ok = sum(r["status"] in ("ok", "skip-exists") for r in results)
    tot_skip = sum(r["skipped"] for r in results)
    log.info(
        "tile pass done: %d/%d tiles ok, %d composite-reads skipped",
        ok,
        len(by_tile),
        tot_skip,
    )
    return mosaic_to_conus(tile_dir, out_dir)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--datetime", default=DATETIME)
    p.add_argument("--platform", choices=["terra", "aqua", "both"], default=PLATFORM)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tile-dir", default=None)
    args = p.parse_args(argv)
    build(
        args.out_dir,
        datetime_str=args.datetime,
        platform=args.platform,
        workers=args.workers,
        tile_dir=args.tile_dir,
    )


if __name__ == "__main__":
    main()
