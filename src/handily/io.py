import glob
import logging
import os
import shlex
import subprocess
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
import rioxarray as rxr
from pynhd import NHD
from shapely.geometry import box
from rioxarray.merge import merge_arrays

LOGGER = logging.getLogger("handily.io")


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _shapefile_has_spatial_index(shp_path: str) -> bool:
    p = Path(shp_path)
    base = p.with_suffix("")
    qix = base.with_suffix(".qix")
    sbn = base.with_suffix(".sbn")
    sbx = base.with_suffix(".sbx")
    return qix.exists() or (sbn.exists() and sbx.exists())


def _ensure_shapefile_spatial_index(shp_path: str) -> None:
    if _shapefile_has_spatial_index(shp_path):
        return

    p = Path(shp_path)
    layer_name = p.stem
    cmd = ["ogrinfo", str(p), "-sql", f"CREATE SPATIAL INDEX ON {layer_name}"]
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    LOGGER.warning("Missing spatial index for %s", shp_path)
    LOGGER.warning("Attempting to build spatial index with: %s", cmd_str)

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if completed.stdout:
            LOGGER.debug("ogrinfo stdout: %s", completed.stdout.strip())
        if completed.stderr:
            LOGGER.debug("ogrinfo stderr: %s", completed.stderr.strip())
    except FileNotFoundError:
        LOGGER.warning(
            "ogrinfo not found on PATH; cannot build spatial index. Install GDAL or pre-create a .qix index."
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        LOGGER.warning("ogrinfo failed to build spatial index (exit %s): %s", exc.returncode, detail or "(no output)")


def get_huc10_boundary(huc10: str, wbd_local_dir: str | None = None) -> gpd.GeoDataFrame:
    """Get HUC-10 boundary, preferring a local WBD HU10 shapefile."""
    if wbd_local_dir is None:
        raise ValueError(
            "wbd_local_dir is required to load WBDHU10 locally; e.g., "
            "~/data/IrrigationGIS/boundaries/wbd/NHD_H_Montana_State_Shape/Shape"
        )

    path = os.path.expanduser(wbd_local_dir)
    shp_path = None
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, "WBDHU10.shp"),
            os.path.join(path, "Shape", "WBDHU10.shp"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                shp_path = candidate
                break
        if shp_path is None:
            hits = glob.glob(os.path.join(path, "**", "WBDHU10.shp"), recursive=True)
            if hits:
                shp_path = hits[0]
    elif os.path.isfile(path) and path.lower().endswith(".shp"):
        shp_path = path

    if shp_path is None or not os.path.exists(shp_path):
        raise FileNotFoundError(
            "Could not find WBDHU10.shp. Ensure the state WBD zip is extracted and set wbd_local_dir "
            "to the extracted '.../NHD_H_<State>_State_Shape/Shape' folder."
        )

    LOGGER.info("Loading WBDHU10 from local shapefile: %s", shp_path)
    hu10 = gpd.read_file(shp_path)
    col = None
    for c in hu10.columns:
        lc = c.lower()
        if lc == "huc10" or lc == "huc_10":
            col = c
            break
    if col is None:
        raise ValueError("HUC10 attribute not found in WBDHU10 shapefile.")

    gdf = hu10[hu10[col].astype(str) == str(huc10)].copy()
    if gdf.empty:
        raise ValueError(f"HUC10 {huc10} not found in local WBDHU10 shapefile: {shp_path}")
    if col != "huc10":
        gdf = gdf.rename(columns={col: "huc10"})
    return gdf.reset_index(drop=True)


def get_flowlines_within_aoi(
    aoi_gdf: gpd.GeoDataFrame, local_flowlines_dir: str | None = None
) -> gpd.GeoDataFrame:
    """Get NHD flowlines intersecting the AOI, preferring local shapefiles when provided."""
    if local_flowlines_dir:
        path = os.path.expanduser(local_flowlines_dir)
        search_roots = [path]
        if os.path.isdir(path) and os.path.basename(path).lower() != "shape":
            shape_dir = os.path.join(path, "Shape")
            if os.path.isdir(shape_dir):
                search_roots.append(shape_dir)

        shp_paths: list[str] = []
        for root in search_roots:
            shp_paths.extend(glob.glob(os.path.join(root, "NHDFlowline*.shp")))
        if not shp_paths:
            shp_paths = glob.glob(os.path.join(path, "**", "NHDFlowline*.shp"), recursive=True)

        if not shp_paths:
            raise FileNotFoundError(
                f"No NHDFlowline shapefiles found under {path}. Ensure the state NHD zip is extracted."
            )

        LOGGER.info("Loading local NHDFlowline shapefiles (%d files)", len(shp_paths))
        parts = []
        for shp in shp_paths:
            try:
                _ensure_shapefile_spatial_index(shp)
                with fiona.open(shp) as src:
                    src_crs = src.crs_wkt if src.crs_wkt else src.crs
                aoi_bbox = aoi_gdf.to_crs(src_crs).total_bounds.tolist()
                gdf = gpd.read_file(shp, bbox=tuple(aoi_bbox))
                if not gdf.empty:
                    parts.append(gdf)
            except Exception:
                gdf = gpd.read_file(shp)
                if not gdf.empty:
                    parts.append(gdf)
        if not parts:
            raise ValueError("No flowline features found in local shapefiles for the AOI extent.")

        flow = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
        try:
            flow = gpd.clip(flow, aoi_gdf.to_crs(flow.crs))
        except Exception:
            flow = gpd.overlay(flow, aoi_gdf.to_crs(flow.crs), how="intersection")
        flow = flow.reset_index(drop=True)
        LOGGER.info("Local flowlines selected: %d", len(flow))
        return flow

    LOGGER.info("Fetching NHDPlus flowlines within AOI via service")
    geom = aoi_gdf.to_crs(4326).geometry.unary_union
    nhd = NHD("flowline_hr")
    fl = nhd.bygeom(geom, geo_crs=4326)
    fl = gpd.clip(fl, aoi_gdf)
    fl = fl.reset_index(drop=True)
    return fl


def _filter_flowlines_nhd(fl: gpd.GeoDataFrame, natural_perennial: bool = False, exclude_artificial: bool = False):
    if len(fl) == 0:
        return fl
    df = fl.copy()

    fcode_col = next((c for c in ("FCODE", "FCode", "fcode") if c in df.columns), None)
    ftype_col = next((c for c in ("FTYPE", "FType", "ftype") if c in df.columns), None)

    if natural_perennial:
        if fcode_col is not None:
            df = df[df[fcode_col] == 46006]
        elif ftype_col is not None:
            df = df[df[ftype_col].astype(str).str.lower().isin(["streamriver"])]

    if exclude_artificial and ftype_col is not None:
        drop_types = {"artificialpath", "canalditch"}
        df = df[~df[ftype_col].astype(str).str.lower().isin(drop_types)]

    return df.reset_index(drop=True)


def aoi_from_bounds(bounds_wsen) -> gpd.GeoDataFrame:
    w, s, e, n = bounds_wsen
    geom = box(w, s, e, n)
    return gpd.GeoDataFrame([{}], geometry=[geom], crs="EPSG:4326")


def ndwi_files_for_bounds(ndwi_dir: str, bounds_wsen):
    """
    Select NDWI GeoTIFFs in ndwi_dir whose raster bounds intersect bounds_wsen (EPSG:4326).

    This avoids relying on MGRS tile IDs being present in filenames.
    """
    ndwi_dir = os.path.expanduser(ndwi_dir)
    w, s, e, n = bounds_wsen
    target = box(w, s, e, n)

    candidates = sorted(glob.glob(os.path.join(ndwi_dir, "*.tif")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(ndwi_dir, "**", "*.tif"), recursive=True))

    hits: list[str] = []
    for path in candidates:
        try:
            with rasterio.open(path) as ds:
                if ds.crs is None:
                    continue
                bw, bs, be, bn = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
            if box(bw, bs, be, bn).intersects(target):
                hits.append(path)
        except Exception:
            continue

    return hits


def open_ndwi_mosaic_from_paths(paths: list[str], bounds_wsen):
    if not paths:
        raise ValueError("No NDWI rasters provided to mosaic.")
    rasters = []
    for p in paths:
        da = rxr.open_rasterio(p)
        if "band" in da.dims:
            da = da.squeeze("band", drop=True)
        rasters.append(da)
    ndwi = rasters[0] if len(rasters) == 1 else merge_arrays(rasters)
    geom = box(*bounds_wsen)
    aoi_ll = gpd.GeoDataFrame([{}], geometry=[geom], crs="EPSG:4326")
    shapes = [aoi_ll.to_crs(ndwi.rio.crs).geometry.unary_union.__geo_interface__]
    return ndwi.rio.clip(shapes, all_touched=True)


def load_and_clip_fields(fields_path: str, aoi_gdf: gpd.GeoDataFrame, target_crs):
    """Load the irrigation dataset and clip to AOI, reprojecting to target CRS."""
    LOGGER.info("Loading irrigation dataset: %s", fields_path)
    try:
        with fiona.open(fields_path) as src:
            fields_crs = src.crs_wkt if src.crs_wkt else src.crs
        aoi_in_fields = aoi_gdf.to_crs(fields_crs)
        bounds = tuple(aoi_in_fields.total_bounds.tolist())
        fields = gpd.read_file(fields_path, bbox=bounds)
    except Exception:
        fields = gpd.read_file(fields_path)

    fields = fields[~fields.geometry.is_empty & fields.geometry.notnull()].copy()

    LOGGER.info("Clipping irrigation dataset to AOI")
    try:
        clipped = gpd.clip(fields, aoi_gdf)
    except Exception:
        clipped = gpd.overlay(fields, aoi_gdf, how="intersection")

    clipped = clipped.to_crs(target_crs)
    return clipped.reset_index(drop=True)
