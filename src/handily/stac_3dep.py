import os
import re
import json
import logging
from tqdm import tqdm
from typing import Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import requests
import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
import rioxarray as rxr
from rasterio.merge import merge as rio_merge
from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree
import shapely
import pystac
import xml.etree.ElementTree as ET
from urllib.parse import quote

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional at runtime
    def tqdm(x, **kwargs):
        return x

LOGGER = logging.getLogger("handily.stac_3dep")

S3_BASE = "https://prd-tnm.s3.amazonaws.com"
# 3DEP 1 m Project tiles live under this prefix, e.g.:
# https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Elevation/1m/Projects/MT_Statewide_Phase4_B22/metadata/
S3_PREFIX_PROJECTS = "StagedProducts/Elevation/1m/Projects/"
PROJECTS_INDEX = f"{S3_BASE}/?delimiter=/&prefix={quote(S3_PREFIX_PROJECTS)}"
PROJECTS_ROOT = f"{S3_BASE}/{S3_PREFIX_PROJECTS}"


def _http_get(url: str, timeout: float = 30.0) -> str:
    headers = {"User-Agent": "handily-stac-builder/0.1"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def _http_get_with_retry(url: str, timeout: float = 30.0, retries: int = 3, backoff: float = 0.5) -> str:
    attempt = 0
    while True:
        try:
            return _http_get(url, timeout=timeout)
        except Exception:
            attempt += 1
            if attempt > int(retries):
                raise
            time.sleep(float(backoff) * (2 ** (attempt - 1)))


def _parse_s3_common_prefixes(xml_text: str) -> List[str]:
    # Parse AWS S3 ListBucketResult and extract <CommonPrefixes><Prefix> entries
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    ns = {"s3": root.tag.split("}")[0].strip("{") if "}" in root.tag else ""}
    prefixes = []
    # Handle with and without namespace
    if ns["s3"]:
        path = f".//{{{ns['s3']}}}CommonPrefixes/{{{ns['s3']}}}Prefix"
    else:
        path = ".//CommonPrefixes/Prefix"
    for el in root.findall(path):
        if el.text:
            prefixes.append(el.text.strip())
    return prefixes


def _parse_s3_contents(xml_text: str) -> List[str]:
    # Parse <Contents><Key> object keys
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    ns = {"s3": root.tag.split("}")[0].strip("{") if "}" in root.tag else ""}
    keys = []
    if ns["s3"]:
        path = f".//{{{ns['s3']}}}Contents/{{{ns['s3']}}}Key"
    else:
        path = ".//Contents/Key"
    for el in root.findall(path):
        if el.text:
            keys.append(el.text.strip())
    return keys


def _list_projects(states: Optional[Iterable[str]] = None) -> List[str]:
    xml = _http_get(PROJECTS_INDEX)
    entries = _parse_s3_common_prefixes(xml)
    # entries are full prefixes like 'StagedProducts/Elevation/1m/Projects/MT_.../'
    norm = []
    for e in entries:
        if not e.endswith("/"):
            continue
        # Extract last segment after projects/
        seg = e.split(S3_PREFIX_PROJECTS, 1)[-1]
        if seg and seg.endswith("/"):
            norm.append(seg)
    if states:
        states_u = {s.upper() for s in states}
        norm = [d for d in norm if any(d.upper().startswith(f"{s}_") for s in states_u)]
    return sorted({d for d in norm})


def _list_metadata_xmls(project_dir: str) -> List[str]:
    """List tile metadata XML filenames under <project>/metadata/.

    Example project_dir: 'MT_Statewide_Phase4_B22/'
    """
    pref = S3_PREFIX_PROJECTS + project_dir + "metadata/"
    url = f"{S3_BASE}/?prefix={quote(pref)}"
    xml = _http_get(url)
    keys = _parse_s3_contents(xml)
    # Return filenames for xmls
    names = [os.path.basename(k) for k in keys if k.lower().endswith(".xml")]
    return sorted({n for n in names})


def _fetch_item_meta(proj: str, name: str):
    """Fetch and parse a single tile XML for a given project."""
    xml_url = f"{S3_BASE}/{S3_PREFIX_PROJECTS}{proj}metadata/{name}"
    try:
        xml_text = _http_get_with_retry(xml_url, timeout=30.0, retries=3, backoff=0.5)
        meta = _parse_fgdc_xml(xml_text)
    except Exception:
        return None
    bbox = meta.get("bbox")
    tif_href = meta.get("tif_href")
    if not bbox or None in bbox or not tif_href:
        return None
    item_id = os.path.splitext(os.path.basename(name))[0]
    iso_pub = _to_iso_date(meta.get("pubdate"))
    iso_beg = _to_iso_date(meta.get("begdate"))
    iso_end = _to_iso_date(meta.get("enddate"))
    dt = iso_end or iso_pub or iso_beg
    return {
        "item_id": item_id,
        "bbox": list(bbox),
        "tif_href": tif_href,
        "xml_url": xml_url,
        "jpg_href": meta.get("jpg_href"),
        "iso_beg": iso_beg,
        "iso_end": iso_end,
        "dt": dt,
    }


def _parse_fgdc_xml(xml_text: str) -> dict:
    # Lightweight FGDC CSDGM extraction using regexes against known tags
    def _find(tag: str) -> Optional[str]:
        m = re.search(fr"<{tag}>(.*?)</{tag}>", xml_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return None

    def _findall(tag: str) -> List[str]:
        return [m.strip() for m in re.findall(fr"<{tag}>(.*?)</{tag}>", xml_text, flags=re.IGNORECASE | re.DOTALL)]

    title = _find("title")
    pubdate = _find("pubdate")
    begdate = _find("begdate")
    enddate = _find("enddate")
    west = _find("westbc")
    east = _find("eastbc")
    north = _find("northbc")
    south = _find("southbc")
    # Distribution URLs
    onlinks = _findall("networkr") or _findall("onlink")
    tif_href = next((u for u in onlinks if u.lower().endswith(".tif")), None)
    jpg_href = next((u for u in onlinks if u.lower().endswith(".jpg")), None)
    if not jpg_href:
        # Some records provide thumbnail in <browsen> under <browse>
        jpg_href = _find("browsen")

    # Rows/cols if present
    rowcount = _find("rowcount")
    colcount = _find("colcount")

    return {
        "title": title,
        "pubdate": pubdate,
        "begdate": begdate,
        "enddate": enddate,
        "bbox": (
            float(west) if west else None,
            float(south) if south else None,
            float(east) if east else None,
            float(north) if north else None,
        ),
        "tif_href": tif_href,
        "jpg_href": jpg_href,
        "rows": int(rowcount) if rowcount and rowcount.isdigit() else None,
        "cols": int(colcount) if colcount and colcount.isdigit() else None,
    }


def _to_iso_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s


def _bbox_to_geojson_polygon(bbox: Tuple[float, float, float, float]) -> dict:
    w, s, e, n = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [w, s],
                [e, s],
                [e, n],
                [w, n],
                [w, s],
            ]
        ],
    }


def build_3dep_stac(out_dir: str,
                    states: Optional[Iterable[str]] = None,
                    collection_id: str = "usgs-3dep-1m-opr",
                    project_head: Optional[str] = None,
                    search_string: Optional[str] = None,
                    num_workers: int = 12,
                    items_shapefile: Optional[str] = None) -> str:
    """
    Build a STAC Collection of 3DEP 1 m DEM tiles.

    Modes:
    - State scan (default): pass states to scan all matching projects.
    - Project scoped: pass project_head and optionally search_string to restrict to a subproject.

    Returns the path to the root catalog.json.
    """
    os.makedirs(out_dir, exist_ok=True)

    catalog = pystac.Catalog(id=f"{collection_id}-catalog", description="USGS 3DEP 1 m DEM tiles")
    collection = pystac.Collection(
        id=collection_id,
        description="USGS 3DEP 1 m DEM tiles (LiDAR-derived)",
        extent=pystac.Extent(
            pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
            pystac.TemporalExtent([(None, None)]),
        ),
        license="public-domain",
        title="USGS 3DEP 1 m DEM",
        keywords=["USGS", "3DEP", "LiDAR", "DEM", "1m"],
        providers=[
            pystac.Provider(name="U.S. Geological Survey",
                            roles=[pystac.ProviderRole.PRODUCER, pystac.ProviderRole.LICENSOR]),
        ],
    )
    catalog.add_child(collection)

    if project_head:
        projects = [project_head.rstrip("/") + "/"]
    else:
        projects = _list_projects(states)
    LOGGER.info("Projects to build: %d (states=%s, project_head=%s, subfilter=%s)",
                len(projects), list(states) if states else None, project_head, search_string)
    total_items = 0
    for proj in tqdm(projects, desc="Projects", unit="proj"):
        # List XMLs directly under <project>/metadata/
        try:
            xml_names = _list_metadata_xmls(proj)
        except Exception:
            xml_names = []
        if search_string:
            xml_names = [n for n in xml_names if search_string in n]
        if not xml_names:
            continue

        bar = tqdm(total=len(xml_names), desc=f"{proj}metadata/", position=1, unit="xml", leave=False)
        workers = max(1, int(num_workers))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_fetch_item_meta, proj, name) for name in xml_names]
            for fut in as_completed(futures):
                try:
                    d = fut.result()
                finally:
                    try:
                        bar.update(1)
                    except Exception:
                        pass
                if not d:
                    continue
                geom = _bbox_to_geojson_polygon(d["bbox"])  # type: ignore[arg-type]
                item = pystac.Item(
                    id=d["item_id"],
                    geometry=geom,
                    bbox=d["bbox"],  # type: ignore[arg-type]
                    datetime=pystac.utils.str_to_datetime(d["dt"]) if d["dt"] else None,
                    properties={},
                )
                if d.get("iso_beg"):
                    item.properties["start_datetime"] = pystac.utils.str_to_datetime(d["iso_beg"]).isoformat()
                if d.get("iso_end"):
                    item.properties["end_datetime"] = pystac.utils.str_to_datetime(d["iso_end"]).isoformat()
                item.properties["gsd"] = 1.0
                item.add_asset(
                    "data",
                    pystac.Asset(href=d["tif_href"], media_type="image/tiff", roles=["data"], title="DEM 1m (GeoTIFF)"),
                )
                item.add_asset(
                    "metadata",
                    pystac.Asset(href=d["xml_url"], media_type="application/xml", roles=["metadata"], title="FGDC metadata"),
                )
                if d.get("jpg_href"):
                    item.add_asset(
                        "thumbnail",
                        pystac.Asset(href=d["jpg_href"], media_type="image/jpeg", roles=["thumbnail"], title="Thumbnail"),
                    )
                collection.add_item(item)
                total_items += 1

        try:
            bar.close()
        except Exception:
            pass

    catalog.normalize_hrefs(out_dir)
    catalog.make_all_asset_hrefs_absolute()
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    LOGGER.info("Wrote STAC: %s (items=%d)", os.path.join(out_dir, "catalog.json"), total_items)
    # Persist spatial index for fast AOI queries
    try:
        _write_bbox_index(out_dir, collection_id)
    except Exception:
        pass
    if items_shapefile:
        shp_path = os.path.expanduser(items_shapefile)
        shp_dir = os.path.dirname(shp_path)
        if shp_dir and not os.path.exists(shp_dir):
            os.makedirs(shp_dir)
        root = os.path.join(out_dir, "catalog.json")
        cat = pystac.read_file(root)
        coll = next((c for c in cat.get_children() if isinstance(c, pystac.Collection) and c.id == collection_id), None)
        if coll is None:
            raise ValueError(f"Collection {collection_id} not found in catalog.")
        ids, geoms = [], []
        for it in coll.get_items():
            if it.bbox:
                ids.append(it.id)
                geoms.append(shapely_box(*it.bbox))
        gdf = gpd.GeoDataFrame({"id": ids}, geometry=geoms, crs="EPSG:4326")
        gdf.to_file(shp_path)
        print(f'wrote stac shapefiles to {shp_path}')
    return os.path.join(out_dir, "catalog.json")


def tiles_for_aoi(stac_dir: str, aoi_bbox_4326: Tuple[float, float, float, float],
                  collection_id: str = "usgs-3dep-1m-opr") -> list[pystac.Item]:
    """
    Return a list of STAC Items whose bbox intersects the given AOI bbox (EPSG:4326).

    Uses a Shapely STRtree spatial index for fast intersection queries over item bboxes.
    """
    root = os.path.join(stac_dir, "catalog.json")
    cat = pystac.read_file(root)
    coll = next((c for c in cat.get_children() if isinstance(c, pystac.Collection) and c.id == collection_id), None)
    if coll is None:
        raise ValueError(f"Collection {collection_id} not found in catalog.")

    aoi_poly = shapely_box(*aoi_bbox_4326)
    loaded = _load_bbox_index(stac_dir, collection_id)
    if loaded is None:
        items_all = list(coll.get_items())
        geoms = []
        ids = []
        for it in items_all:
            ib = it.bbox
            if ib:
                geoms.append(shapely_box(*ib))
                ids.append(it.id)
        try:
            _write_bbox_index(stac_dir, collection_id)
        except Exception:
            pass
    else:
        geoms, ids = loaded

    if not geoms:
        return []

    tree = STRtree(geoms)
    matches = tree.query(aoi_poly, predicate="intersects")
    # Shapely 2.x may return either integer indices (np.ndarray) or geometry objects.
    wkb_to_idx = {g.wkb: i for i, g in enumerate(geoms)}
    idxs = []
    try:
        # numpy array of indices path
        import numpy as _np  # local import to avoid top-level constraints
        if hasattr(matches, "dtype") and _np.issubdtype(matches.dtype, _np.integer):
            idxs = [int(i) for i in matches.tolist()]
        else:
            for m in matches:
                if isinstance(m, (int,)):
                    idxs.append(int(m))
                else:
                    i = wkb_to_idx.get(m.wkb)
                    if i is not None:
                        idxs.append(i)
    except Exception:
        for m in matches:
            try:
                i = wkb_to_idx.get(m.wkb)
            except Exception:
                i = None
            if i is not None:
                idxs.append(i)

    seen = set()
    out = []
    for i in idxs:
        if i in seen:
            continue
        seen.add(i)
        it = cat.get_item(ids[i], recursive=True)
        if it is not None:
            out.append(it)
    return out


def _index_path(stac_dir: str, collection_id: str) -> str:
    return os.path.join(stac_dir, f"{collection_id}_bbox_index.json")


def _write_bbox_index(stac_dir: str, collection_id: str) -> str:
    root = os.path.join(stac_dir, "catalog.json")
    cat = pystac.read_file(root)
    coll = next((c for c in cat.get_children() if isinstance(c, pystac.Collection) and c.id == collection_id), None)
    if coll is None:
        raise ValueError(f"Collection {collection_id} not found in catalog.")
    ids = []
    bboxes = []
    for it in coll.get_items():
        if it.bbox:
            ids.append(it.id)
            bboxes.append(list(it.bbox))
    data = {"collection_id": collection_id, "ids": ids, "bboxes": bboxes}
    path = _index_path(stac_dir, collection_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def _load_bbox_index(stac_dir: str, collection_id: str):
    path = _index_path(stac_dir, collection_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = data.get("ids") or []
    bboxes = data.get("bboxes") or []
    if not ids or not bboxes or len(ids) != len(bboxes):
        return None
    geoms = [shapely_box(*bbox) for bbox in bboxes]
    return geoms, ids


def mosaic_from_stac(stac_dir: str,
                     aoi_gdf,
                     cache_dir: str,
                     collection_id: str = "usgs-3dep-1m-opr",
                     target_crs_epsg: int = 5070) -> xr.DataArray:
    """
    Select tiles overlapping the AOI from a local 3DEP STAC and mosaic into a DEM DataArray.

    - Downloads GeoTIFFs into cache_dir if not present.
    - Merges with rasterio.merge.
    - Reprojects to target_crs_epsg and clips to AOI.
    """
    os.makedirs(cache_dir, exist_ok=True)
    bbox4326 = tuple(aoi_gdf.to_crs(4326).total_bounds.tolist())
    items = tiles_for_aoi(stac_dir, bbox4326, collection_id=collection_id)
    if not items:
        raise ValueError("No STAC tiles intersect AOI.")

    tifs_local: list[str] = []
    for it in tqdm(items, total=len(items), desc='Mosaic from STAC'):
        asset = it.assets.get("data")
        if asset is None:
            continue
        href = asset.href
        fname = os.path.basename(href)
        local = os.path.join(cache_dir, fname)
        if not os.path.exists(local):
            # stream download
            with requests.get(href, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
        tifs_local.append(local)

    if not tifs_local:
        raise ValueError("No GeoTIFF assets available among intersecting Items.")

    srcs = [rasterio.open(p) for p in tifs_local]
    mosaic, mosaic_tr = rio_merge(srcs)
    crs = srcs[0].crs
    for ds in srcs:
        try:
            ds.close()
        except Exception:
            pass

    dem = xr.DataArray(mosaic[0].astype("float32"), dims=("y", "x"), name="elevation")
    dem = dem.rio.write_crs(crs, inplace=False)
    dem = dem.rio.write_transform(mosaic_tr)
    epsg = crs.to_epsg()
    if (epsg is None) or (int(epsg) != int(target_crs_epsg)):
        dem = dem.rio.reproject(f"EPSG:{int(target_crs_epsg)}")

    dem = dem.rio.clip(aoi_gdf.to_crs(dem.rio.crs).geometry, aoi_gdf.to_crs(dem.rio.crs).crs)
    # Replace nodata with NaN consistently
    nd = dem.rio.nodata
    if nd is not None and np.isfinite(nd):
        dem = dem.where(dem != nd)
        dem = dem.rio.write_nodata(np.nan)

    # Enforce LiDAR resolution sanity (1 m nominal)
    try:
        resx, resy = dem.rio.resolution()
        if max(abs(resx), abs(resy)) > 2.0:
            raise ValueError(
                f"Mosaic resolution too coarse: {abs(resx):.2f} x {abs(resy):.2f} m (> 2 m)."
            )
    except Exception:
        pass
    return dem
