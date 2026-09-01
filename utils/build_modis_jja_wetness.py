"""Build CONUS 250 m MODIS summer (JJA) wetness composites from MOD13Q1 on MPC.

Lever D of the GNN covariate ladder. The bundle's water features are JRC-occurrence
based, so they only see *open* water. They cannot see a valley that simply stays
green and moist through a dry summer -- a subirrigated meadow, a phreatophyte
corridor, a flood-irrigated bottomland. That persistence is a shallow-water-table
signal. This builder turns it into two static CONUS query features.

Source: Microsoft Planetary Computer (free, no Earth Engine quota), collection
`modis-13Q1-061` (250 m, 16-day Maximum-Value Composite). It mixes Terra
(MOD13Q1) and Aqua (MYD13Q1) granules; we keep Terra only, matching the
established project recipe in `build_modis_ndvi_climatology.py` (whose JJA-mean
NDVI is the MODIS-JJA seed used by the scalable FAC-REM). This module differs in
three ways: (1) **median** rather than mean across the window -- robust to the
residual cloud/aerosol excursions the reliability mask lets through; (2) it also
builds a **wetness index** from the MOD13Q1 reflectance bands; (3) a fixed
7-full-year window.

Indices (both dimensionless, both oriented **higher = wetter/greener**):

  NDVI  = (NIR - red) / (NIR + red)
          taken pre-computed from the `250m_16_days_NDVI` asset (scale 1e-4).

  NDWI  = (NIR - MIR) / (NIR + MIR)
          from `250m_16_days_NIR_reflectance` (MODIS band 2, 841-876 nm) and
          `250m_16_days_MIR_reflectance` (MODIS band 7, 2105-2155 nm), both
          scale 1e-4. This is the Gao-family vegetation-water index computed on
          the bands MOD13Q1 actually carries at 250 m. Note the SWIR leg is band
          7 (2.1 um), not Gao's original band 5 (1.24 um), so this is the
          index often written NDWI(2.1) / NDMI / NDII. It responds to canopy +
          soil water content: wetter -> higher.

Chosen over MOD09 green-band NDWI (McFeeters) because that would drop us to
500 m and halve the spatial detail in exactly the narrow valley bottoms this
feature exists to resolve.

Masking, applied per 16-day composite before it enters the median stack:
  * `250m_16_days_pixel_reliability` in {0 good, 1 marginal}; 2 snow/ice,
    3 cloud and 255 fill are dropped.
  * NDVI fill (-3000) and out-of-range (outside [-2000, 10000]).
  * reflectance fill (-1000) and out-of-range (outside [-100, 16000]), and
    NIR + MIR <= 0 (degenerate denominator).

Pipeline (resumable at tile granularity):
  1. Search MPC for every MOD13Q1 composite intersecting CONUS in the window;
     keep the ones whose composite start date falls in Jun/Jul/Aug.
  2. Per MODIS sinusoidal tile (h##v##): stream each composite, mask, stack the
     surviving DNs as int16, take the per-pixel median, write two float32 GeoTIFFs
     on the native sinusoidal grid plus a JSON sidecar of the read tallies.
     A tile whose three outputs exist is skipped, so a killed run resumes.
  3. gdalwarp-mosaic the per-tile medians -> CONUS EPSG:5070 250 m GeoTIFFs on a
     grid co-registered with the existing MODIS covariates, then write the
     build manifest.

Outputs in <out-dir>:
  modis_jja_ndvi_median.tif  modis_jja_ndwi_median.tif  build_manifest.json

Usage:
    uv run python utils/build_modis_jja_wetness.py \
        --out-dir /nas/handily/covariates/modis_jja_wetness_250m --workers 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
from pystac_client import Client

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_modis_ndvi_climatology import (  # noqa: E402
    GOOD_REL,
    NDVI_FILL,
    NDVI_MAX,
    NDVI_MIN,
    STAC_URL,
    _DATE_RE,
    _read_band,
    granule_platform,
    tile_id,
)

log = logging.getLogger("build_modis_jja_wetness")

COLLECTION = "modis-13Q1-061"
NDVI_ASSET = "250m_16_days_NDVI"
REL_ASSET = "250m_16_days_pixel_reliability"
NIR_ASSET = "250m_16_days_NIR_reflectance"
MIR_ASSET = "250m_16_days_MIR_reflectance"

CONUS_BBOX = (-125.0, 24.0, -66.5, 49.5)
YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)  # 7 full summers
JJA_MONTHS = (6, 7, 8)
PLATFORM = "terra"  # MOD13Q1 only -- one 16-day series, no Aqua interleave

REFL_FILL = -1000
REFL_MIN, REFL_MAX = -100, 16000  # MOD13Q1 reflectance valid range (scale 1e-4)
DN_SCALE = 1e-4
DN_FILL = np.int16(-32768)  # our own stack sentinel, distinct from any MODIS fill

INDICES = ("ndvi", "ndwi")

OUT_DIR = "/nas/handily/covariates/modis_jja_wetness_250m"
TARGET_CRS = "EPSG:5070"
TARGET_RES = 250.0
# CONUS window snapped to the pixel grid of the existing 250 m MODIS covariates
# (/data/ssd2/handily/conus/covariates/modis_ndvi_*_mean.tif) so the new rasters
# are co-registered with them: 19401 x 12601 px.
TARGET_TE = (-2500070.4636, 169850.3143, 2350179.5364, 3320100.3143)

MEDIAN_ROW_CHUNK = 400  # rows per nanmedian pass, caps the float32 working set


# Reading a whole 4800x4800 MOD13Q1 COG band is 100 sequential 512-px block
# fetches; single-threaded that costs ~65 s/band. Multi-threaded block fetch with
# HTTP/2 multiplexing brings it to ~14 s. These must be process environment
# variables, not a rasterio.Env: GDAL config set through Env is thread-local, so
# the band-reading worker threads below would never see it.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_NUM_THREADS": "4",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "268435456",
}


def configure_gdal() -> None:
    """Install the COG-over-HTTP read settings process-wide (idempotent)."""
    for k, v in GDAL_ENV.items():
        os.environ.setdefault(k, v)


def granule_ymd(item_id: str) -> tuple[int, int]:
    """(year, month) of the composite start from the granule's A<year><doy> code.

    MOD13Q1 items span 16 days so `item.datetime` is null; the id is the
    deterministic offline source of the composite date.
    """
    m = _DATE_RE.search(item_id)
    if not m:
        raise ValueError(f"no A<year><doy> date in item id: {item_id}")
    yr, doy = int(m.group(1)), int(m.group(2))
    d = date(yr, 1, 1) + timedelta(days=doy - 1)
    return d.year, d.month


def ndvi_dn(ndvi: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Masked NDVI DNs (int16, scale 1e-4); invalid pixels set to DN_FILL."""
    ok = (
        np.isin(rel, GOOD_REL)
        & (ndvi != NDVI_FILL)
        & (ndvi >= NDVI_MIN)
        & (ndvi <= NDVI_MAX)
    )
    return np.where(ok, ndvi, DN_FILL).astype(np.int16)


def ndwi_dn(nir: np.ndarray, mir: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Masked NDWI = (NIR - MIR)/(NIR + MIR) as int16 DNs (scale 1e-4).

    Higher = wetter. Invalid pixels set to DN_FILL.
    """
    ok = (
        np.isin(rel, GOOD_REL)
        & (nir != REFL_FILL)
        & (mir != REFL_FILL)
        & (nir >= REFL_MIN)
        & (nir <= REFL_MAX)
        & (mir >= REFL_MIN)
        & (mir <= REFL_MAX)
    )
    num = nir.astype(np.float32) - mir.astype(np.float32)
    den = nir.astype(np.float32) + mir.astype(np.float32)
    ok &= den > 0
    val = np.zeros(num.shape, dtype=np.float32)
    np.divide(num, den, out=val, where=ok)
    dn = np.clip(np.rint(val / np.float32(DN_SCALE)), -10000, 10000)
    return np.where(ok, dn, DN_FILL).astype(np.int16)


def stack_median(stack: np.ndarray, row_chunk: int = MEDIAN_ROW_CHUNK) -> np.ndarray:
    """Per-pixel median over axis 0, ignoring DN_FILL, scaled to physical units.

    Row-chunked so the float32 working copy stays small; NaN where a pixel never
    had a valid summer observation.
    """
    n, rows, cols = stack.shape
    out = np.full((rows, cols), np.nan, dtype=np.float32)
    for r0 in range(0, rows, row_chunk):
        r1 = min(r0 + row_chunk, rows)
        chunk = stack[:, r0:r1, :].astype(np.float32)
        chunk[stack[:, r0:r1, :] == DN_FILL] = np.nan
        if np.isnan(chunk).all():
            continue
        with warnings.catch_warnings():
            # rows where a pixel was never clear are legitimately all-NaN -> NaN out
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(chunk, axis=0)
        out[r0:r1, :] = med * np.float32(DN_SCALE)
    return out


def _tile_paths(tile: str, tile_dir: str) -> dict[str, str]:
    p = {i: f"{tile_dir}/{tile}_jja_{i}_median.tif" for i in INDICES}
    p["json"] = f"{tile_dir}/{tile}_jja_tally.json"
    return p


def accumulate_tile(tile: str, records: list[dict], tile_dir: str) -> dict:
    """Stream every JJA composite for one tile -> median NDVI/NDWI GeoTIFFs.

    records: [{"year": int, "month": int, "ndvi": href, "rel": href,
               "nir": href, "mir": href}, ...]
    """
    Path(tile_dir).mkdir(parents=True, exist_ok=True)
    out = _tile_paths(tile, tile_dir)
    if all(Path(v).exists() for v in out.values()):
        return json.loads(Path(out["json"]).read_text())

    ndvi_stack: list[np.ndarray] = []
    ndwi_stack: list[np.ndarray] = []
    profile = None
    used = skipped = 0
    years_used: set[int] = set()
    t0 = time.time()
    configure_gdal()
    keys = ("ndvi", "rel", "nir", "mir")
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        for rec in records:
            # the four band reads are latency-bound HTTP range fetches: overlap them
            futs = {k: pool.submit(_read_band, rec[k]) for k in keys}
            try:
                got = {k: f.result() for k, f in futs.items()}
            except RuntimeError:
                skipped += 1
                continue
            ndvi, prof = got["ndvi"]
            rel, nir, mir = got["rel"][0], got["nir"][0], got["mir"][0]
            if profile is None:
                profile = prof
            ndvi_stack.append(ndvi_dn(ndvi, rel))
            ndwi_stack.append(ndwi_dn(nir, mir, rel))
            years_used.add(rec["year"])
            used += 1

    if profile is None:
        return {"tile": tile, "status": "no-data", "used": 0, "skipped": skipped}

    out_prof = profile.copy()
    out_prof.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    valid_frac = {}
    for name, stack in (("ndvi", ndvi_stack), ("ndwi", ndwi_stack)):
        med = stack_median(np.stack(stack, axis=0))
        stack.clear()
        valid_frac[name] = float(np.isfinite(med).mean())
        with rasterio.open(out[name], "w", **out_prof) as ds:
            ds.write(med, 1)
        del med

    tally = {
        "tile": tile,
        "status": "ok",
        "used": used,
        "skipped": skipped,
        "years": sorted(years_used),
        "valid_fraction": valid_frac,
        "seconds": round(time.time() - t0, 1),
    }
    Path(out["json"]).write_text(json.dumps(tally, indent=2))
    log.info(
        "tile %s: %d composites used, %d read-skips, valid ndvi %.3f ndwi %.3f, %.0fs",
        tile,
        used,
        skipped,
        valid_frac["ndvi"],
        valid_frac["ndwi"],
        tally["seconds"],
    )
    return tally


def list_records(bbox, years, platform: str = PLATFORM) -> dict:
    """Search MPC and group unsigned JJA hrefs by MODIS sinusoidal tile."""
    cat = Client.open(STAC_URL)  # unsigned; we sign per-read (long run, SAS expiry)
    dt = f"{min(years)}-01-01/{max(years)}-12-31"
    search = cat.search(collections=[COLLECTION], bbox=bbox, datetime=dt)
    by_tile: dict[str, list[dict]] = defaultdict(list)
    n = off_platform = off_season = 0
    for it in search.items():
        try:
            t = tile_id(it.id)
        except ValueError:
            continue
        if granule_platform(it.id) != platform:
            off_platform += 1
            continue
        yr, mo = granule_ymd(it.id)
        if yr not in years or mo not in JJA_MONTHS:
            off_season += 1
            continue
        by_tile[t].append(
            {
                "year": yr,
                "month": mo,
                "ndvi": it.assets[NDVI_ASSET].href,
                "rel": it.assets[REL_ASSET].href,
                "nir": it.assets[NIR_ASSET].href,
                "mir": it.assets[MIR_ASSET].href,
            }
        )
        n += 1
    log.info(
        "%d JJA composites over %d tiles (platform=%s; dropped %d other-platform, "
        "%d out-of-season/year)",
        n,
        len(by_tile),
        platform,
        off_platform,
        off_season,
    )
    return dict(by_tile)


def mosaic_to_conus(tile_dir: str, out_dir: str) -> dict[str, str]:
    """gdalwarp the per-tile medians -> CONUS EPSG:5070 250 m GeoTIFFs."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for idx in INDICES:
        srcs = sorted(Path(tile_dir).glob(f"*_jja_{idx}_median.tif"))
        if not srcs:
            log.warning("no per-tile rasters for %s -- skipping mosaic", idx)
            continue
        out = f"{out_dir}/modis_jja_{idx}_median.tif"
        cmd = [
            "gdalwarp",
            "-t_srs",
            TARGET_CRS,
            "-tr",
            str(TARGET_RES),
            str(TARGET_RES),
            "-te",
            *[f"{v:.4f}" for v in TARGET_TE],
            "-r",
            "bilinear",
            "-dstnodata",
            "nan",
            "-of",
            "GTiff",
            "-multi",
            "-wo",
            "NUM_THREADS=ALL_CPUS",
            "-co",
            "COMPRESS=DEFLATE",
            "-co",
            "PREDICTOR=3",
            "-co",
            "TILED=YES",
            "-co",
            "BIGTIFF=YES",
            "-co",
            "NUM_THREADS=ALL_CPUS",
            "-overwrite",
            *[str(s) for s in srcs],
            out,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        paths[idx] = out
        log.info("mosaicked %s from %d tiles -> %s", idx, len(srcs), out)
    return paths


def write_manifest(
    out_dir: str, paths: dict[str, str], tallies: list[dict], seconds: float
) -> str:
    """Record how these rasters were built, for reproduction and provenance."""
    with rasterio.open(next(iter(paths.values()))) as ds:
        grid = {
            "crs": ds.crs.to_string(),
            "res_m": [abs(ds.transform.a), abs(ds.transform.e)],
            "width": ds.width,
            "height": ds.height,
            "bounds": list(ds.bounds),
        }
    manifest = {
        "product": "MODIS summer (JJA) wetness composites, CONUS 250 m",
        "purpose": (
            "static query features for the handily CONUS water-table GNN: summer "
            "greenness/wetness persistence as a shallow-water-table signal the "
            "JRC open-water features cannot see"
        ),
        "build_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_seconds": round(seconds, 1),
        "source": {
            "catalog": STAC_URL,
            "collection": COLLECTION,
            "product": "MOD13Q1 v061 (Terra, 250 m, 16-day Maximum-Value Composite)",
            "platform": PLATFORM,
            "assets": {
                "ndvi": NDVI_ASSET,
                "nir": NIR_ASSET,
                "mir": MIR_ASSET,
                "reliability": REL_ASSET,
            },
        },
        "window": {
            "years": list(YEARS),
            "months": list(JJA_MONTHS),
            "month_source": "composite start date from the granule A<year><doy> code",
            "statistic": "per-pixel median across all retained JJA composites",
        },
        "formulas": {
            "ndvi": {
                "expression": "(NIR - red) / (NIR + red)",
                "note": (
                    "read pre-computed from the 250m_16_days_NDVI asset, DN scale 1e-4"
                ),
                "units": "dimensionless, [-1, 1]",
                "orientation": "higher = greener",
            },
            "ndwi": {
                "expression": "(NIR - MIR) / (NIR + MIR)",
                "bands": (
                    "MODIS band 2 NIR 841-876 nm; MODIS band 7 MIR 2105-2155 nm; "
                    "both DN scale 1e-4"
                ),
                "note": (
                    "Gao-family vegetation-water index on the bands MOD13Q1 carries "
                    "at 250 m; SWIR leg is band 7 (2.1 um) not Gao's band 5 "
                    "(1.24 um), i.e. NDWI(2.1) / NDMI / NDII"
                ),
                "units": "dimensionless, [-1, 1]",
                "orientation": "higher = wetter (more canopy/soil water)",
            },
        },
        "masking": {
            "pixel_reliability_kept": list(GOOD_REL),
            "pixel_reliability_dropped": {
                "2": "snow/ice",
                "3": "cloud",
                "255": "fill",
            },
            "ndvi_dropped": f"fill {NDVI_FILL}, outside [{NDVI_MIN}, {NDVI_MAX}] DN",
            "reflectance_dropped": (
                f"fill {REFL_FILL}, outside [{REFL_MIN}, {REFL_MAX}] DN, "
                "or NIR + MIR <= 0"
            ),
        },
        "grid": grid,
        "grid_note": (
            "co-registered with /data/ssd2/handily/conus/covariates/"
            "modis_ndvi_*_mean.tif (same 250 m EPSG:5070 pixel grid), clipped to "
            "a CONUS window"
        ),
        "tiles": {
            "count": len(tallies),
            "ids": sorted(t["tile"] for t in tallies),
            "composites_used_total": sum(t.get("used", 0) for t in tallies),
            "composite_read_skips_total": sum(t.get("skipped", 0) for t in tallies),
            "per_tile": sorted(tallies, key=lambda t: t["tile"]),
        },
        "outputs": {
            k: {"path": v, "bytes": Path(v).stat().st_size} for k, v in paths.items()
        },
        "nodata": "NaN (float32)",
        "builder": "utils/build_modis_jja_wetness.py",
    }
    path = f"{out_dir}/build_manifest.json"
    Path(path).write_text(json.dumps(manifest, indent=2))
    log.info("wrote manifest -> %s", path)
    return path


def build(
    out_dir: str = OUT_DIR,
    *,
    bbox=CONUS_BBOX,
    years=YEARS,
    platform: str = PLATFORM,
    workers: int = 10,
    tile_dir: str | None = None,
    mosaic_only: bool = False,
) -> dict[str, str]:
    t0 = time.time()
    tile_dir = tile_dir or f"{out_dir}/tiles"
    by_tile = list_records(bbox, years, platform=platform)
    tallies = []
    if mosaic_only:
        for t in by_tile:
            j = Path(_tile_paths(t, tile_dir)["json"])
            if j.exists():
                tallies.append(json.loads(j.read_text()))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(accumulate_tile, t, recs, tile_dir): t
                for t, recs in by_tile.items()
            }
            for fut in as_completed(futs):
                tallies.append(fut.result())
        ok = sum(t["status"] == "ok" for t in tallies)
        log.info("tile pass: %d/%d tiles ok", ok, len(by_tile))
    paths = mosaic_to_conus(tile_dir, out_dir)
    write_manifest(
        out_dir, paths, [t for t in tallies if t["status"] == "ok"], time.time() - t0
    )
    return paths


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--tile-dir", default=None)
    p.add_argument(
        "--mosaic-only",
        action="store_true",
        help="skip the streaming pass, mosaic whatever per-tile medians exist",
    )
    args = p.parse_args(argv)
    build(
        args.out_dir,
        workers=args.workers,
        tile_dir=args.tile_dir,
        mosaic_only=args.mosaic_only,
    )


if __name__ == "__main__":
    main()
