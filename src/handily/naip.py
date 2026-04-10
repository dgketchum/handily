"""NAIP imagery download from USDA Box and NDWI computation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge

LOGGER = logging.getLogger("handily.naip")

FIPS_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"
CENSUS_CACHE_DIR = Path("/tmp/census_counties")
COUNTIES_SHP = CENSUS_CACHE_DIR / "cb_2022_us_county_500k.shp"

STATE_FIPS = {
    "AZ": "04",
    "CA": "06",
    "CO": "08",
    "ID": "16",
    "MT": "30",
    "NV": "32",
    "NM": "35",
    "OR": "41",
    "UT": "49",
    "WA": "53",
    "WY": "56",
}

MRSID_DSDK = os.environ.get(
    "MRSID_DSDK",
    os.path.expanduser("~/.local/opt/MrSID/current/Raster_DSDK"),
)
MRSIDINFO = os.path.join(MRSID_DSDK, "bin", "mrsidinfo")
MRSIDDECODE = os.path.join(MRSID_DSDK, "bin", "mrsiddecode")

BOX_ROOT_FOLDER = "17936490251"
_YEAR_FOLDERS: dict[str, str] | None = None

_FLOAT_PATTERN = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _default_headers() -> dict[str, str]:
    return {"User-Agent": "handily-naip/0.1"}


def _download_file(url: str, dest_path: str | Path, timeout: int = 300) -> None:
    with requests.get(
        url, stream=True, headers=_default_headers(), timeout=timeout
    ) as r:
        r.raise_for_status()
        with Path(dest_path).open("wb") as dst:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    dst.write(chunk)


def _coerce_crs(crs: Any) -> Any:
    if crs is None:
        LOGGER.warning("AOI CRS is undefined; assuming EPSG:4326")
        return "EPSG:4326"
    return crs


def _project_geometry(geometry, src_crs: Any, dst_crs: Any):
    return gpd.GeoSeries([geometry], crs=_coerce_crs(src_crs)).to_crs(dst_crs).iloc[0]


def _aoi_bounds_wsen(aoi_geometry, aoi_crs: Any) -> tuple[float, float, float, float]:
    return _project_geometry(aoi_geometry, aoi_crs, "EPSG:4326").bounds


def _aoi_tag(aoi_geometry, aoi_crs: Any) -> str:
    aoi_wgs84 = _project_geometry(aoi_geometry, aoi_crs, "EPSG:4326")
    return hashlib.sha1(aoi_wgs84.wkb).hexdigest()[:12]


def _ensure_parent(path: str | Path) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def _get_aoi_fips(aoi_gdf: gpd.GeoDataFrame, state: str) -> gpd.GeoDataFrame:
    """Add a ``_fips3`` column to an AOI GeoDataFrame using Census counties."""

    if not COUNTIES_SHP.exists():
        CENSUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = CENSUS_CACHE_DIR / "counties.zip"
        LOGGER.info("Downloading Census county boundaries")
        _download_file(FIPS_URL, zip_path, timeout=120)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(CENSUS_CACHE_DIR)

    counties = gpd.read_file(COUNTIES_SHP)
    state_fips = STATE_FIPS.get(state.upper())
    if state_fips:
        counties = counties[counties["STATEFP"] == state_fips]

    counties = counties.to_crs("EPSG:5070")
    centroids = aoi_gdf.to_crs("EPSG:5070").copy()
    centroids["geometry"] = centroids.geometry.centroid
    joined = gpd.sjoin(
        centroids, counties[["COUNTYFP", "geometry"]], how="left", predicate="within"
    )

    aoi_gdf = aoi_gdf.copy()
    aoi_gdf["_fips3"] = joined["COUNTYFP"].values
    missing = int(aoi_gdf["_fips3"].isna().sum())
    if missing:
        LOGGER.warning("%d AOIs could not be assigned a county FIPS", missing)
    return aoi_gdf


def _box_list_folder(folder_id: str) -> list[dict[str, Any]]:
    """List items in a Box shared folder by scraping the shared-folder page."""

    items: list[dict[str, Any]] = []
    page = 1
    while True:
        url = f"https://nrcs.app.box.com/v/naip/folder/{folder_id}?page={page}"
        response = requests.get(url, headers=_default_headers(), timeout=60)
        response.raise_for_status()
        match = re.search(
            r"Box\.postStreamData\s*=\s*(\{.+?\})\s*;",
            response.text,
            flags=re.DOTALL,
        )
        if not match:
            break
        data = json.loads(match.group(1))
        folder_data = data.get("/app-api/enduserapp/shared-folder", {})
        items.extend(folder_data.get("items", []))
        if page >= int(folder_data.get("pageCount", 1)):
            break
        page += 1
    return items


def _box_download_file(file_id: str, dest_path: str) -> None:
    """Download a file from the shared NAIP Box folder."""

    url = (
        "https://nrcs.app.box.com/index.php?"
        f"rm=box_download_shared_file&vanity_name=naip&file_id=f_{file_id}"
    )
    LOGGER.info("Downloading Box file %s -> %s", file_id, dest_path)
    _download_file(url, dest_path)


def _get_year_folders() -> dict[str, str]:
    """Return ``{year: folder_id}`` for the NAIP root folder."""

    global _YEAR_FOLDERS
    if _YEAR_FOLDERS is None:
        _YEAR_FOLDERS = {
            item["name"]: item["id"]
            for item in _box_list_folder(BOX_ROOT_FOLDER)
            if item["type"] == "folder" and item["name"].isdigit()
        }
    return _YEAR_FOLDERS


def find_county_zip(state: str, year: str, fips: str) -> dict[str, Any] | None:
    """Find the multispectral county ZIP entry for a state/year/FIPS."""

    year_folders = _get_year_folders()
    if year not in year_folders:
        LOGGER.warning("Year %s not found in NAIP Box", year)
        return None

    state_lower = state.lower()
    states = _box_list_folder(year_folders[year])
    state_entry = next(
        (
            item
            for item in states
            if item["type"] == "folder" and item["name"].lower() == state_lower
        ),
        None,
    )
    if state_entry is None:
        LOGGER.warning("State %s not found in NAIP %s", state, year)
        return None

    state_items = _box_list_folder(state_entry["id"])
    # Prefer 4-band multispectral (_m), fall back to CIR (_c) which has NIR
    imagery_folder = None
    for suffix in (f"{state_lower}_m", f"{state_lower}_c"):
        imagery_folder = next(
            (
                item
                for item in state_items
                if item["type"] == "folder" and item["name"] == suffix
            ),
            None,
        )
        if imagery_folder is not None:
            break
    if imagery_folder is None:
        LOGGER.warning("No multispectral or CIR folder for %s %s", state, year)
        return None

    pattern = f"{state_lower}{str(fips).zfill(3)}"
    for entry in _box_list_folder(imagery_folder["id"]):
        name = entry.get("name", "").lower()
        if entry.get("type") == "file" and name.endswith(".zip") and pattern in name:
            return entry

    LOGGER.warning("No ZIP for %s FIPS %s in %s", state, fips, year)
    return None


def _mrsid_env() -> dict[str, str]:
    env = os.environ.copy()
    mrsid_paths = [os.path.join(MRSID_DSDK, "lib"), os.path.join(MRSID_DSDK, "bin")]
    env["LD_LIBRARY_PATH"] = ":".join(
        [path for path in mrsid_paths + [env.get("LD_LIBRARY_PATH", "")] if path]
    )
    return env


def _run_mrsid_command(cmd: list[str]) -> str:
    result = subprocess.run(cmd, env=_mrsid_env(), capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(stderr or f"Command failed: {' '.join(cmd)}")
    return result.stdout


def _read_sid_metadata(
    sid_path: str,
) -> tuple[str | None, tuple[float, float, float, float] | None]:
    output = _run_mrsid_command([MRSIDINFO, "-i", sid_path, "-wkt"])

    # Match the LAST AUTHORITY tag — the outermost PROJCS level, not inner SPHEROID/DATUM
    crs_matches = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', output)
    crs = f"EPSG:{crs_matches[-1]}" if crs_matches else None

    ul_match = re.search(
        rf"upper left:\s+\(({_FLOAT_PATTERN}),\s+({_FLOAT_PATTERN})\)",
        output,
        flags=re.IGNORECASE,
    )
    lr_match = re.search(
        rf"lower right:\s+\(({_FLOAT_PATTERN}),\s+({_FLOAT_PATTERN})\)",
        output,
        flags=re.IGNORECASE,
    )
    bounds = None
    if ul_match and lr_match:
        bounds = (
            float(ul_match.group(1)),
            float(ul_match.group(2)),
            float(lr_match.group(1)),
            float(lr_match.group(2)),
        )
    return crs, bounds


def _get_sid_crs(sid_path: str) -> str | None:
    """Extract the EPSG code from a MrSID file."""

    return _read_sid_metadata(sid_path)[0]


def _get_sid_bounds(
    sid_path: str,
) -> tuple[float, float, float, float] | None:
    """Extract native upper-left / lower-right coordinates from a MrSID file."""

    return _read_sid_metadata(sid_path)[1]


def _clip_decode_bounds(
    requested: tuple[float, float, float, float],
    sid_bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    ulx, uly, lrx, lry = requested
    sid_ulx, sid_uly, sid_lrx, sid_lry = sid_bounds
    clipped = (
        max(ulx, sid_ulx),
        min(uly, sid_uly),
        min(lrx, sid_lrx),
        max(lry, sid_lry),
    )
    if clipped[0] >= clipped[2] or clipped[3] >= clipped[1]:
        raise RuntimeError("AOI does not intersect SID extent")
    return clipped


def decode_sid(
    sid_path: str,
    tif_path: str,
    bounds_wsen: tuple[float, float, float, float] | None = None,
) -> str:
    """Decode a MrSID file to GeoTIFF using ``mrsiddecode``."""

    cmd = [MRSIDDECODE, "-i", sid_path, "-o", tif_path, "-of", "tifg", "-wf"]
    if bounds_wsen is not None:
        from pyproj import Transformer

        sid_crs, sid_bounds = _read_sid_metadata(sid_path)
        if sid_crs is None:
            raise RuntimeError(f"Could not determine CRS for {sid_path}")

        transformer = Transformer.from_crs("EPSG:4326", sid_crs, always_xy=True)
        west, south, east, north = bounds_wsen
        map_ul = transformer.transform(west, north)
        map_lr = transformer.transform(east, south)

        # mrsiddecode -ulxy/-lrxy expects PIXEL coordinates, not map coords.
        # Convert using the SID's origin and resolution.
        if sid_bounds is None:
            raise RuntimeError("Cannot determine SID origin for pixel conversion")
        origin_x, origin_y = sid_bounds[0], sid_bounds[1]  # UL corner in map coords

        # Parse resolution from mrsidinfo output (X res, Y res)
        info_output = _run_mrsid_command([MRSIDINFO, "-i", sid_path, "-wkt"])
        res_match = re.search(r"X res:\s+(" + _FLOAT_PATTERN + r")", info_output)
        res = float(res_match.group(1)) if res_match else 0.6  # default NAIP 0.6m

        # Map coords → pixel coords
        pix_ulx = int((map_ul[0] - origin_x) / res)
        pix_uly = int((origin_y - map_ul[1]) / res)  # Y is inverted (origin is top)
        pix_lrx = int((map_lr[0] - origin_x) / res)
        pix_lry = int((origin_y - map_lr[1]) / res)

        # Clamp to SID pixel dimensions
        pix_ulx = max(0, pix_ulx)
        pix_uly = max(0, pix_uly)

        cmd.extend(
            ["-ulxy", str(pix_ulx), str(pix_uly), "-lrxy", str(pix_lrx), str(pix_lry)]
        )

    _ensure_parent(tif_path)
    LOGGER.info("Decoding %s", os.path.basename(sid_path))
    _run_mrsid_command(cmd)
    return tif_path


def compute_ndwi(naip_path: str, ndwi_path: str) -> str:
    """Compute NDWI from a NAIP GeoTIFF.

    Supports two band arrangements:
    - 4+ band multispectral: R, G, B, NIR → Green=Band2, NIR=Band4
    - 3-band CIR: NIR, R, G → Green=Band3, NIR=Band1
    """

    with rasterio.open(naip_path) as src:
        if src.count >= 4:
            # 4-band multispectral (R, G, B, NIR)
            green = src.read(2).astype("float32")
            nir = src.read(4).astype("float32")
        elif src.count == 3:
            # 3-band CIR (NIR, R, G)
            green = src.read(3).astype("float32")
            nir = src.read(1).astype("float32")
        else:
            raise ValueError(
                f"Expected 3+ band NAIP, got {src.count} bands: {naip_path}"
            )
        denom = green + nir
        ndwi = np.where(denom != 0, (green - nir) / denom, 0).astype("float32")

        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=None)
        with rasterio.open(ndwi_path, "w", **profile) as dst:
            dst.write(ndwi, 1)

    return ndwi_path


def _extract_sid_files(zip_path: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    sid_paths: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        sid_names = sorted(
            name for name in zf.namelist() if name.lower().endswith(".sid")
        )
        LOGGER.info("ZIP contains %d .sid files", len(sid_names))
        for name in sid_names:
            dest = extract_dir / Path(name).name
            if not dest.exists():
                with zf.open(name) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            sid_paths.append(dest)
    return sid_paths


def _clip_dataset_to_aoi(src, out_path: str, aoi_geometry, aoi_crs: Any) -> None:
    if src.crs is None:
        raise ValueError("Raster has no CRS")

    aoi_in_raster_crs = _project_geometry(aoi_geometry, aoi_crs, src.crs)
    clipped, clipped_transform = rio_mask(
        src,
        [aoi_in_raster_crs],
        crop=True,
        nodata=0,
    )
    profile = src.profile.copy()
    profile.update(
        height=clipped.shape[1],
        width=clipped.shape[2],
        transform=clipped_transform,
    )
    _ensure_parent(out_path)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(clipped)


def _write_clipped_ndwi(
    ndwi_tiles: list[Path], out_path: str, aoi_geometry, aoi_crs: Any
) -> None:
    if len(ndwi_tiles) == 1:
        with rasterio.open(ndwi_tiles[0]) as src:
            _clip_dataset_to_aoi(src, out_path, aoi_geometry, aoi_crs)
        return

    with ExitStack() as stack:
        datasets = [stack.enter_context(rasterio.open(tile)) for tile in ndwi_tiles]
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile.copy()

    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
    )
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(mosaic)
            _clip_dataset_to_aoi(dataset, out_path, aoi_geometry, aoi_crs)


def naip_ndwi_for_aoi(
    aoi_geometry,
    state: str,
    fips: str,
    year: str,
    out_path: str,
    cache_dir: str | None = None,
    cleanup: bool = True,
    aoi_crs: Any = "EPSG:4326",
) -> str:
    """Download NAIP for one AOI, compute NDWI, and clip to the AOI."""

    state = state.lower()
    fips = str(fips).zfill(3)
    aoi_crs = _coerce_crs(aoi_crs)
    _ensure_parent(out_path)

    entry = find_county_zip(state, year, fips)
    if entry is None:
        raise FileNotFoundError(f"No NAIP found for {state} FIPS {fips} year {year}")

    temp_cache_root: Path | None = None
    cache_root = (
        Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="naip_cache_"))
    )
    if cache_dir is None:
        temp_cache_root = cache_root
    cache_root.mkdir(parents=True, exist_ok=True)

    zip_path = cache_root / entry["name"]
    if not zip_path.exists():
        _box_download_file(entry["id"], str(zip_path))
    else:
        LOGGER.info("Using cached ZIP: %s", zip_path)

    aoi_tag = _aoi_tag(aoi_geometry, aoi_crs)
    extract_dir = cache_root / f"{state}_{fips}_{year}"
    aoi_bounds = _aoi_bounds_wsen(aoi_geometry, aoi_crs)
    ndwi_tiles: list[Path] = []

    for sid_path in _extract_sid_files(zip_path, extract_dir):
        base = sid_path.stem
        tif_path = extract_dir / f"{base}_{aoi_tag}.tif"
        ndwi_path = extract_dir / f"{base}_{aoi_tag}_ndwi.tif"

        if not tif_path.exists():
            try:
                decode_sid(str(sid_path), str(tif_path), bounds_wsen=aoi_bounds)
            except RuntimeError as exc:
                LOGGER.warning("Skipping %s: %s", sid_path.name, exc)
                continue

        try:
            with rasterio.open(tif_path) as src:
                if src.width == 0 or src.height == 0:
                    continue
        except Exception as exc:
            LOGGER.warning("Skipping unreadable tile %s: %s", tif_path.name, exc)
            continue

        if not ndwi_path.exists():
            compute_ndwi(str(tif_path), str(ndwi_path))
        ndwi_tiles.append(ndwi_path)

    if not ndwi_tiles:
        raise ValueError(f"No NAIP tiles intersect AOI for {state} FIPS {fips} {year}")

    LOGGER.info("Mosaicking %d NDWI tiles for AOI", len(ndwi_tiles))
    _write_clipped_ndwi(ndwi_tiles, out_path, aoi_geometry, aoi_crs)
    LOGGER.info("NDWI written: %s", out_path)

    if cleanup:
        if temp_cache_root is not None:
            shutil.rmtree(temp_cache_root, ignore_errors=True)
        elif extract_dir.is_dir():
            shutil.rmtree(extract_dir, ignore_errors=True)

    return out_path
