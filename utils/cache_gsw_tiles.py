"""Pre-cache JRC Global Surface Water occurrence tiles to NAS (one-time populate).

JRC-GSW is a STATIC dataset (v1.3, 2020), but the scalable FAC-REM builder used to
query it live from Microsoft Planetary Computer once per HUC8 -- so a statewide batch
made hundreds of live MPC calls and stalled whenever MPC's STAC/COG endpoints had a
slow spell (observed sustained-outage 2026-06-25). This downloads the raw occurrence
COGs that cover the target states ONCE to a local NAS cache; thereafter
``build_scalable_fac_rem.build_support`` reads them directly (see
``_local_gsw_tiles``) and never touches MPC for cached regions.

The cache stores the RAW occurrence COG (0-100 = %% of valid months with water);
the permanent-water threshold is applied at build time, so the cache is
threshold-agnostic. Tiles are 10x10-degree, named by their NW corner
(e.g. ``110W_40N.tif``); NM/MT/NV are covered by four tiles
(110W_40N, 110W_50N, 120W_40N, 120W_50N).

Populate the default NM/MT/NV set:
  uv run python utils/cache_gsw_tiles.py

Whole lower-48, or an explicit box:
  uv run python utils/cache_gsw_tiles.py --conus
  uv run python utils/cache_gsw_tiles.py --bbox -109 31 -103 37

Requires MPC to be reachable (this is the one step that does); build-time reads do not.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import geopandas as gpd
import planetary_computer as pc
import rasterio
import requests
from pystac_client import Client

from build_scalable_fac_rem import GSW_CACHE_DIR, STAC_URL, WBD_HUC8, _with_retries

log = logging.getLogger("cache_gsw_tiles")

# Lower-48 bounding box (lon/lat) -- generous; MPC returns only intersecting tiles.
CONUS_BBOX = (-125.0, 24.0, -66.5, 49.5)
# GSW tile ids look like "110W_40Nv1_3_2020"; the leading "<lon>W_<lat>N" labels the
# 10-deg tile and is what we name the cached file (stable, human-readable).
_TILE_RE = re.compile(r"(\d+[EW]_\d+[NS])")


def _state_bbox(state: str) -> tuple[float, float, float, float]:
    """lon/lat bounds of every HUC8 touching ``state`` (national WBD layer)."""
    g = gpd.read_parquet(WBD_HUC8)
    sel = g[g["states"].fillna("").str.contains(rf"\b{state}\b")]
    if sel.empty:
        raise SystemExit(f"no HUC8s for state {state!r} in {WBD_HUC8}")
    b = sel.to_crs(4326).total_bounds
    return tuple(float(x) for x in b)


def _tile_basename(item_id: str) -> str:
    m = _TILE_RE.match(item_id)
    if not m:
        raise ValueError(f"unrecognized GSW tile id: {item_id!r}")
    return m.group(1)


def _download(item, dst: Path) -> None:
    """Stream the signed occurrence COG to ``dst`` atomically; validate before publish."""
    href = pc.sign(item).assets["occurrence"].href
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with requests.get(href, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        # A truncated download must not masquerade as a cached tile -- confirm it
        # opens as a raster before the atomic rename.
        with rasterio.open(tmp) as ds:
            _ = ds.profile
        tmp.rename(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _collect(bbox: tuple[float, float, float, float], tag: str) -> dict:
    items = _with_retries(
        lambda: list(
            Client.open(STAC_URL).search(collections=["jrc-gsw"], bbox=bbox).items()
        ),
        what=f"MPC jrc-gsw search ({tag})",
    )
    log.info("%s: %d GSW tile(s): %s", tag, len(items), [it.id for it in items])
    return {_tile_basename(it.id): it for it in items}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--state",
        nargs="*",
        default=["NM", "MT", "NV"],
        help="states to cover (default: the FAC-REM target states NM MT NV).",
    )
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="explicit lon/lat box instead of --state.",
    )
    p.add_argument(
        "--conus",
        action="store_true",
        help="cover the whole lower-48 instead of --state.",
    )
    p.add_argument("--cache-dir", type=Path, default=GSW_CACHE_DIR)
    p.add_argument(
        "--force", action="store_true", help="re-download tiles already in the cache."
    )
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    targets: dict = {}
    if args.bbox:
        targets.update(_collect(tuple(args.bbox), "bbox"))
    elif args.conus:
        targets.update(_collect(CONUS_BBOX, "conus"))
    else:
        for st in args.state:
            targets.update(_collect(_state_bbox(st), st))

    log.info("%d unique GSW tile(s) to ensure: %s", len(targets), sorted(targets))
    n_dl = n_skip = 0
    for base, it in sorted(targets.items()):
        dst = args.cache_dir / f"{base}.tif"
        if dst.exists() and not args.force:
            log.info("  cached: %s", dst.name)
            n_skip += 1
            continue
        log.info("  downloading %s -> %s", it.id, dst.name)
        _with_retries(
            lambda it=it, dst=dst: _download(it, dst), what=f"download {base}"
        )
        n_dl += 1
    log.info(
        "done: %d downloaded, %d already cached -> %s", n_dl, n_skip, args.cache_dir
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
