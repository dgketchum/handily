import os
import re
import json
import logging
from typing import Iterable, List, Optional, Tuple

import requests
import numpy as np
import rasterio
import xarray as xr
import rioxarray as rxr
from rasterio.merge import merge as rio_merge
from shapely.geometry import box as shapely_box
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
S3_PREFIX_PROJECTS = "StagedProducts/Elevation/OPR/Projects/"
PROJECTS_INDEX = f"{S3_BASE}/?delimiter=/&prefix={quote(S3_PREFIX_PROJECTS)}"
PROJECTS_ROOT = f"{S3_BASE}/{S3_PREFIX_PROJECTS}"


def _http_get(url: str, timeout: float = 30.0) -> str:
    headers = {"User-Agent": "handily-stac-builder/0.1"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


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
    # entries are full prefixes like 'StagedProducts/Elevation/OPR/Projects/MT_.../'
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


def _list_subprojects(project_dir: str) -> List[str]:
    # project_dir like 'MT_Statewide_Phase4_B22/' → list CommonPrefixes one level down
    pref = S3_PREFIX_PROJECTS + project_dir
    url = f"{S3_BASE}/?delimiter=/&prefix={quote(pref)}"
    xml = _http_get(url)
    prefixes = _parse_s3_common_prefixes(xml)
    # Return only the subdir names (last segment) with trailing slash
    out = []
    for p in prefixes:
        if p.endswith("/") and p.startswith(pref):
            seg = p[len(pref) :]
            if seg:
                out.append(seg)
    return sorted({d for d in out})


def _list_metadata_xmls(project_dir: str, sub_dir: str) -> List[str]:
    # List objects under .../<project>/<sub_dir>/metadata/
    pref = S3_PREFIX_PROJECTS + project_dir + sub_dir + "metadata/"
    url = f"{S3_BASE}/?prefix={quote(pref)}"
    xml = _http_get(url)
    keys = _parse_s3_contents(xml)
    # Return filenames for xmls
    names = [os.path.basename(k) for k in keys if k.lower().endswith(".xml")]
    return sorted({n for n in names})


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


def build_3dep_stac(out_dir: str, states: Optional[Iterable[str]] = None, collection_id: str = "usgs-3dep-1m-opr") -> str:
    """
    Build a single STAC Collection of 3DEP OPR 1m DEM tiles for the given states.

    - Scans the TNM S3 static index for projects.
    - Filters projects by state abbreviation prefix (e.g., "MT_").
    - Traverses subprojects, then the 'metadata' folder to read per-tile FGDC XML.
    - Creates one STAC Item per XML, linking to the GeoTIFF and XML assets.
    - Saves a self-contained STAC catalog under out_dir.

    Returns the path to the root catalog.json.
    """
    os.makedirs(out_dir, exist_ok=True)

    catalog = pystac.Catalog(id=f"{collection_id}-catalog", description="USGS 3DEP OPR 1 m DEM tiles")
    collection = pystac.Collection(
        id=collection_id,
        description="USGS 3DEP Original Product Resolution (OPR) 1 m DEM tiles (LiDAR-derived)",
        extent=pystac.Extent(
            pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
            pystac.TemporalExtent([(None, None)]),
        ),
        license="public-domain",
        title="USGS 3DEP OPR 1 m DEM",
        keywords=["USGS", "3DEP", "LiDAR", "DEM", "OPR", "1m"],
        providers=[
            pystac.Provider(name="U.S. Geological Survey", roles=[pystac.ProviderRole.PRODUCER, pystac.ProviderRole.LICENSOR]),
        ],
    )
    catalog.add_child(collection)

    projects = _list_projects(states)
    LOGGER.info("Found %d projects matching states=%s", len(projects), list(states) if states else None)
    total_items = 0
    for proj in tqdm(projects, desc="Projects", unit="proj"):
        subdirs = _list_subprojects(proj)
        for sub in subdirs:
            try:
                xml_names = _list_metadata_xmls(proj, sub)
            except requests.HTTPError:
                continue
            for name in xml_names:
                xml_url = f"{S3_BASE}/{S3_PREFIX_PROJECTS}{proj}{sub}metadata/{name}"
                try:
                    xml_text = _http_get(xml_url)
                    meta = _parse_fgdc_xml(xml_text)
                except Exception:
                    continue
                bbox = meta.get("bbox")
                tif_href = meta.get("tif_href")
                if not bbox or None in bbox or not tif_href:
                    continue

                item_id = os.path.splitext(os.path.basename(name))[0]
                geom = _bbox_to_geojson_polygon(bbox)  # type: ignore[arg-type]
                iso_pub = _to_iso_date(meta.get("pubdate"))
                iso_beg = _to_iso_date(meta.get("begdate"))
                iso_end = _to_iso_date(meta.get("enddate"))
                dt = iso_end or iso_pub or iso_beg

                item = pystac.Item(
                    id=item_id,
                    geometry=geom,
                    bbox=list(bbox),  # type: ignore[arg-type]
                    datetime=pystac.utils.str_to_datetime(dt) if dt else None,
                    properties={},
                )
                if iso_beg or iso_end:
                    if iso_beg:
                        item.properties["start_datetime"] = pystac.utils.str_to_datetime(iso_beg).isoformat()
                    if iso_end:
                        item.properties["end_datetime"] = pystac.utils.str_to_datetime(iso_end).isoformat()
                item.properties["gsd"] = 1.0

                # Assets
                item.add_asset(
                    "data",
                    pystac.Asset(href=tif_href, media_type="image/tiff", roles=["data"], title="DEM 1m (GeoTIFF)"),
                )
                item.add_asset(
                    "metadata",
                    pystac.Asset(href=xml_url, media_type="application/xml", roles=["metadata"], title="FGDC metadata"),
                )
                if meta.get("jpg_href"):
                    item.add_asset(
                        "thumbnail",
                        pystac.Asset(href=meta["jpg_href"], media_type="image/jpeg", roles=["thumbnail"], title="Thumbnail"),
                    )

                collection.add_item(item)
                total_items += 1

    catalog.normalize_hrefs(out_dir)
    catalog.make_all_asset_hrefs_absolute()
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    LOGGER.info("Wrote STAC: %s (items=%d)", os.path.join(out_dir, "catalog.json"), total_items)
    return os.path.join(out_dir, "catalog.json")


def extend_3dep_stac(out_dir: str, states: Iterable[str], collection_id: str = "usgs-3dep-1m-opr") -> str:
    """
    Extend an existing STAC catalog by adding Items for additional state abbreviations.

    Loads the catalog at out_dir/catalog.json, finds the collection, and adds new Items
    for projects whose names start with the provided state abbreviations.
    """
    root = os.path.join(out_dir, "catalog.json")
    cat = pystac.read_file(root)
    coll = next((c for c in cat.get_children() if isinstance(c, pystac.Collection) and c.id == collection_id), None)
    if coll is None:
        raise ValueError(f"Collection {collection_id} not found in catalog.")

    existing_ids = set(i.id for i in coll.get_items())
    new_count = 0
    projects = _list_projects(states)
    for proj in projects:
        subdirs = _list_subprojects(proj)
        for sub in subdirs:
            try:
                xml_names = _list_metadata_xmls(proj, sub)
            except requests.HTTPError:
                continue
            for name in xml_names:
                item_id = os.path.splitext(os.path.basename(name))[0]
                if item_id in existing_ids:
                    continue
                xml_url = PROJECTS_ROOT + proj + sub + "metadata/" + name
                try:
                    xml_text = _http_get(xml_url)
                    meta = _parse_fgdc_xml(xml_text)
                except Exception:
                    continue
                bbox = meta.get("bbox")
                tif_href = meta.get("tif_href")
                if not bbox or None in bbox or not tif_href:
                    continue
                geom = _bbox_to_geojson_polygon(bbox)  # type: ignore[arg-type]
                iso_pub = _to_iso_date(meta.get("pubdate"))
                iso_beg = _to_iso_date(meta.get("begdate"))
                iso_end = _to_iso_date(meta.get("enddate"))
                dt = iso_end or iso_pub or iso_beg
                item = pystac.Item(
                    id=item_id,
                    geometry=geom,
                    bbox=list(bbox),  # type: ignore[arg-type]
                    datetime=pystac.utils.str_to_datetime(dt) if dt else None,
                    properties={},
                )
                if iso_beg or iso_end:
                    if iso_beg:
                        item.properties["start_datetime"] = pystac.utils.str_to_datetime(iso_beg).isoformat()
                    if iso_end:
                        item.properties["end_datetime"] = pystac.utils.str_to_datetime(iso_end).isoformat()
                item.properties["gsd"] = 1.0
                item.add_asset(
                    "data",
                    pystac.Asset(href=tif_href, media_type="image/tiff", roles=["data"], title="DEM 1m (GeoTIFF)"),
                )
                item.add_asset(
                    "metadata",
                    pystac.Asset(href=xml_url, media_type="application/xml", roles=["metadata"], title="FGDC metadata"),
                )
                if meta.get("jpg_href"):
                    item.add_asset(
                        "thumbnail",
                        pystac.Asset(href=meta["jpg_href"], media_type="image/jpeg", roles=["thumbnail"], title="Thumbnail"),
                    )
                coll.add_item(item)
                existing_ids.add(item_id)
                new_count += 1

    cat.normalize_hrefs(out_dir)
    cat.make_all_asset_hrefs_absolute()
    cat.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    LOGGER.info("Extended STAC with %d new items", new_count)
    return root


def tiles_for_aoi(stac_dir: str, aoi_bbox_4326: Tuple[float, float, float, float], collection_id: str = "usgs-3dep-1m-opr") -> list[pystac.Item]:
    """
    Return a list of STAC Items whose bbox intersects the given AOI bbox (EPSG:4326).
    """
    root = os.path.join(stac_dir, "catalog.json")
    cat = pystac.read_file(root)
    coll = next((c for c in cat.get_children() if isinstance(c, pystac.Collection) and c.id == collection_id), None)
    if coll is None:
        raise ValueError(f"Collection {collection_id} not found in catalog.")
    aoi_poly = shapely_box(*aoi_bbox_4326)
    hits = []
    for it in coll.get_items():
        ib = it.bbox
        if ib and shapely_box(*ib).intersects(aoi_poly):
            hits.append(it)
    return hits


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
    for it in items:
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
    if int(target_crs_epsg) != int(pystac.utils.maybe_int(crs.to_epsg()) or target_crs_epsg):
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


if __name__ == "__main__":
    # Temporary debug: direct function calls without CLI
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    states = ["MT"]
    print(f"Listing projects for states={states}")
    projects = _list_projects(states)
    print(f"Found {len(projects)} projects:")
    for d in projects:
        print(f"  - {d}")
    if projects:
        proj = projects[0]
        print(f"Probing subprojects of {proj}")
        subs = _list_subprojects(proj)
        print(f"  Subprojects: {len(subs)}")
        if subs:
            xmls = _list_metadata_xmls(proj, subs[0])
    out_dir = os.path.expanduser("~/data/IrrigationGIS/handily/stac/3dep_1m_debug")
    print(f"Building STAC into: {out_dir}")
    build_3dep_stac(out_dir, states=states)
    print("Done.")
