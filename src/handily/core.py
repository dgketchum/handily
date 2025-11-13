import glob
import logging
import os

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from pynhd import NHD
from pyproj import CRS as _CRS
from rasterio import features
from rasterstats import zonal_stats
from scipy import ndimage as ndi
from rioxarray.merge import merge_arrays

LOGGER = logging.getLogger("handily.core")


def ensure_dir(path):
    """
    Create a directory if it doesn't exist.
    Uses os.path.* per constraints.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_huc10_boundary(huc10: str, wbd_local_dir: str | None = None) -> gpd.GeoDataFrame:
    """Get HUC-10 boundary, preferring a local WBD HU10 shapefile.

    This implementation avoids network calls. It expects a local state WBD
    dataset such as:
      ~/data/IrrigationGIS/boundaries/wbd/NHD_H_Montana_State_Shape/Shape/WBDHU10.shp

    Parameters
    ----------
    huc10 : str
        HUC-10 ID, e.g., "1002000207".
    wbd_local_dir : str, optional
        Directory containing a WBDHU10 shapefile or a direct path to WBDHU10.shp.

    Parameters
    ----------
    huc10 : str
        HUC-10 ID, e.g., "1002000207".

    Returns
    -------
    geopandas.GeoDataFrame
        Single-row GeoDataFrame for the requested HUC-10.
    """
    if wbd_local_dir is None:
        raise ValueError(
            "wbd_local_dir is required to load WBDHU10 locally; e.g., "
            "~/data/IrrigationGIS/boundaries/wbd/NHD_H_Montana_State_Shape/Shape"
        )

    path = os.path.expanduser(wbd_local_dir)
    shp_path = None
    if os.path.isdir(path):
        # Try common locations first
        candidates = [
            os.path.join(path, "WBDHU10.shp"),
            os.path.join(path, "Shape", "WBDHU10.shp"),
        ]
        for c in candidates:
            if os.path.exists(c):
                shp_path = c
                break
        if shp_path is None:
            # Search recursively as a fallback
            hits = glob.glob(os.path.join(path, "**", "WBDHU10.shp"), recursive=True)
            if hits:
                shp_path = hits[0]
    elif os.path.isfile(path) and path.lower().endswith(".shp"):
        shp_path = path

    if shp_path is None or not os.path.exists(shp_path):
        raise FileNotFoundError(
            "Could not find WBDHU10.shp. Ensure the state WBD zip is extracted, e.g.:\n"
            "  wget https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/State/Shape/NHD_H_Montana_State_Shape.zip\n"
            "  unzip -d ~/data/IrrigationGIS/boundaries/wbd ~/Downloads/NHD_H_Montana_State_Shape.zip\n"
            "and set wbd_local_dir to that '.../NHD_H_Montana_State_Shape/Shape' folder."
        )

    LOGGER.info("Loading WBDHU10 from local shapefile: %s", shp_path)
    hu10 = gpd.read_file(shp_path)
    # Robust column detection for HUC10
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
    # Normalize column name
    if col != "huc10":
        gdf = gdf.rename(columns={col: "huc10"})
    return gdf.reset_index(drop=True)


def get_flowlines_within_aoi(aoi_gdf: gpd.GeoDataFrame, local_flowlines_dir: str | None = None) -> gpd.GeoDataFrame:
    """
    Get NHD flowlines intersecting the AOI.

    Prefer local state shapefiles (NHDFlowline*.shp) if a directory is provided.
    Otherwise, fetch from the USGS NHD ArcGIS service via PyNHD.

    Parameters
    ----------
    aoi_gdf : GeoDataFrame
        AOI polygon, any CRS.
    local_flowlines_dir : str, optional
        Directory containing state NHD shapefiles (expects files like NHDFlowline_*.shp).

    Returns
    -------
    GeoDataFrame
        Flowlines clipped to AOI (in the dataset/native CRS). Caller should reproject to target raster CRS.
    """
    if local_flowlines_dir:
        path = os.path.expanduser(local_flowlines_dir)
        # Common locations
        search_roots = [path]
        if os.path.isdir(path) and os.path.basename(path).lower() != "shape":
            shape_dir = os.path.join(path, "Shape")
            if os.path.isdir(shape_dir):
                search_roots.append(shape_dir)

        shp_paths = []
        for root in search_roots:
            shp_paths.extend(glob.glob(os.path.join(root, "NHDFlowline*.shp")))
        if not shp_paths:
            # Recursive search
            shp_paths = glob.glob(os.path.join(path, "**", "NHDFlowline*.shp"), recursive=True)

        if not shp_paths:
            raise FileNotFoundError(
                f"No NHDFlowline shapefiles found under {path}. Ensure the state NHD zip is extracted."
            )

        LOGGER.info("Loading local NHDFlowline shapefiles (%d files)", len(shp_paths))
        parts = []
        for shp in shp_paths:
            try:
                with fiona.open(shp) as src:
                    src_crs = src.crs_wkt if src.crs_wkt else src.crs
                # Compute AOI bbox in source CRS to minimize IO
                aoi_bbox = aoi_gdf.to_crs(src_crs).total_bounds.tolist()
                gdf = gpd.read_file(shp, bbox=tuple(aoi_bbox))
                if not gdf.empty:
                    parts.append(gdf)
            except Exception:
                # Fallback to full read if bbox fails
                gdf = gpd.read_file(shp)
                if not gdf.empty:
                    parts.append(gdf)
        if not parts:
            raise ValueError("No flowline features found in local shapefiles for the AOI extent.")
        flow = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
        # Clip to AOI to remove bbox overshoot
        try:
            flow = gpd.clip(flow, aoi_gdf.to_crs(flow.crs))
        except Exception:
            flow = gpd.overlay(flow, aoi_gdf.to_crs(flow.crs), how="intersection")
        flow = flow.reset_index(drop=True)
        LOGGER.info("Local flowlines selected: %d", len(flow))
        return flow

    # Remote fallback: Use NHD high-resolution flowlines from ArcGIS REST
    LOGGER.info("Fetching NHDPlus flowlines within AOI via service")
    geom = aoi_gdf.to_crs(4326).geometry.unary_union
    nhd = NHD("flowline_hr")
    fl = nhd.bygeom(geom, geo_crs=4326)
    fl = gpd.clip(fl, aoi_gdf)
    fl = fl.reset_index(drop=True)
    return fl


def _filter_flowlines_nhd(fl, natural_perennial=False, exclude_artificial=False):
    """
    Optionally filter NHD flowlines to natural perennial streams and/or exclude artificial paths.
    Uses robust attribute lookups across common NHD schema variants.
    """
    if len(fl) == 0:
        return fl
    df = fl.copy()

    # Common attribute names
    fcode_col = None
    for c in ("FCODE", "FCode", "fcode"):
        if c in df.columns:
            fcode_col = c
            break
    ftype_col = None
    for c in ("FTYPE", "FType", "ftype"):
        if c in df.columns:
            ftype_col = c
            break

    # Perennial stream/river FCODE in NHD is typically 46006.
    if natural_perennial:
        if fcode_col is not None:
            df = df[df[fcode_col] == 46006]
        elif ftype_col is not None:
            df = df[df[ftype_col].astype(str).str.lower().isin(["streamriver"])]

    if exclude_artificial and ftype_col is not None:
        drop_types = {"artificialpath", "canalditch"}
        df = df[~df[ftype_col].astype(str).str.lower().isin(drop_types)]

    return df.reset_index(drop=True)


def rasterize_lines_to_grid(lines_gdf, template_da, burn_value=1):
    """
    Rasterize line features to match a template DataArray grid.

    Parameters
    ----------
    lines_gdf : GeoDataFrame
        Line geometries in the same CRS as `template_da`.
    template_da : xarray.DataArray
        Raster providing transform, shape, and CRS.
    burn_value : numeric
        Value to burn for stream cells; others get 0.

    Returns
    -------
    xarray.DataArray
        Boolean/int mask DataArray with stream cells burned as `burn_value`.
    """
    transform = template_da.rio.transform()
    shape = template_da.shape

    # Prepare shapes iterable for rasterize: (geometry, value)
    shapes = [(geom, burn_value) for geom in lines_gdf.geometry if geom is not None]
    if len(shapes) == 0:
        raise ValueError("No valid geometries found to rasterize.")

    stream_arr = features.rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    stream_da = xr.DataArray(
        stream_arr,
        dims=template_da.dims,
        coords=template_da.coords,
        name="streams",
    )
    stream_da = stream_da.rio.write_crs(template_da.rio.crs, inplace=False)
    return stream_da

def build_streams_mask_from_nhd_ndwi(flowlines_gdf, dem_da, ndwi_da=None, ndwi_threshold=None):
    """
    Build a streams/water mask on the DEM grid by combining NHD flowlines with an NDWI raster.

    - Rasterize flowlines to the DEM grid.
    - If NDWI is provided, reproject to DEM grid, threshold at ndwi_threshold, and union with flowlines mask.
    """
    if dem_da.rio.crs is None:
        raise ValueError("DEM must have a valid CRS.")
    if flowlines_gdf.crs is None or str(flowlines_gdf.crs) != str(dem_da.rio.crs):
        flowlines_gdf = flowlines_gdf.to_crs(dem_da.rio.crs)

    fl_mask = rasterize_lines_to_grid(flowlines_gdf, dem_da, burn_value=1)
    streams_mask = fl_mask

    if ndwi_da is not None and ndwi_threshold is not None:
        ndwi_match = ndwi_da
        if str(ndwi_da.rio.crs) != str(dem_da.rio.crs) or ndwi_da.shape != dem_da.shape:
            ndwi_match = ndwi_da.rio.reproject_match(dem_da)
        water = (ndwi_match > float(ndwi_threshold)).astype("uint8")
        water = water.rio.write_crs(dem_da.rio.crs, inplace=False)
        combo = xr.DataArray(
            np.maximum(np.asarray(streams_mask.data, dtype="uint8"), np.asarray(water.data, dtype="uint8")),
            dims=dem_da.dims,
            coords=dem_da.coords,
            name="streams",
        )
        combo = combo.rio.write_crs(dem_da.rio.crs, inplace=False)
        streams_mask = combo

    return streams_mask

def compute_rem_quick(dem_da, streams_da):
    """
    Compute a quick Relative Elevation Model using Euclidean nearest-stream height without sink filling.
    """
    if dem_da.rio.crs is None or streams_da.rio.crs is None:
        raise ValueError("Both DEM and streams rasters must have a valid CRS.")
    if str(dem_da.rio.crs) != str(streams_da.rio.crs) or dem_da.shape != streams_da.shape:
        raise ValueError("DEM and streams must share grid shape and CRS.")

    dem_np = np.asarray(dem_da.data)
    streams_np = np.asarray(streams_da.data).astype(bool)
    if streams_np.sum() == 0:
        raise ValueError("Stream mask has no active cells after combination.")

    row_ind, col_ind = ndi.distance_transform_edt(
        ~streams_np, return_distances=False, return_indices=True
    )
    base_elev = dem_np[row_ind, col_ind]
    rem_np = dem_np - base_elev
    rem_np = np.where(rem_np < 0, 0, rem_np)

    rem_da = xr.DataArray(
        rem_np,
        dims=dem_da.dims,
        coords=dem_da.coords,
        name="REM",
        attrs={"description": "Quick REM (euclidean nearest stream; no sink filling)"},
    )
    rem_da = rem_da.rio.write_crs(dem_da.rio.crs, inplace=False)
    return rem_da

def aoi_from_bounds(bounds_wsen):
    w, s, e, n = bounds_wsen
    aoi = gpd.GeoDataFrame([{}], geometry=[gpd.GeoSeries.from_bounds(w, s, e, n).iloc[0]], crs="EPSG:4326")
    return aoi

def tiles_for_bounds(bounds_wsen, mgrs_shp_path):
    w, s, e, n = bounds_wsen
    aoi_ll = gpd.GeoDataFrame([{}], geometry=[gpd.GeoSeries.from_bounds(w, s, e, n).iloc[0]], crs="EPSG:4326")
    mgrs = gpd.read_file(os.path.expanduser(mgrs_shp_path))
    mgrs_aea = mgrs.to_crs("EPSG:5070")
    aoi_aea = aoi_ll.to_crs("EPSG:5070")
    hits = mgrs_aea[mgrs_aea.intersects(aoi_aea.unary_union)].copy()
    if hits.empty:
        raise ValueError("No MGRS tiles intersect bounds.")
    name_col = "MGRS_TILE"
    tiles = list(hits[name_col].astype(str).unique())
    return tiles, hits

def ndwi_files_for_tiles(ndwi_dir, tiles):
    ndwi_dir = os.path.expanduser(ndwi_dir)
    present = {}
    missing = []
    for t in tiles:
        matches = sorted(glob.glob(os.path.join(ndwi_dir, f"*{t}*.tif")))
        if matches:
            present[t] = matches
        else:
            missing.append(t)
    return present, missing

def open_ndwi_mosaic(present_map, bounds_wsen):
    rasters = []
    for paths in present_map.values():
        first = rxr.open_rasterio(paths[0])
        if "band" in first.dims:
            first = first.squeeze("band", drop=True)
        rasters.append(first)
    if len(rasters) == 1:
        ndwi = rasters[0]
    else:
        ndwi = merge_arrays(rasters)
    aoi_ll = gpd.GeoDataFrame([{}], geometry=[gpd.GeoSeries.from_bounds(*bounds_wsen).iloc[0]], crs="EPSG:4326")
    shapes = [aoi_ll.to_crs(ndwi.rio.crs).geometry.unary_union.__geo_interface__]
    ndwi_clip = ndwi.rio.clip(shapes, all_touched=True)
    return ndwi_clip

def run_bounds_rem(bounds_wsen,
                        fields_path,
                        ndwi_dir,
                        stac_dir,
                        flowlines_local_dir,
                        out_dir,
                        ndwi_threshold=0.15,
                        mgrs_shp_path="~/data/IrrigationGIS/boundaries/mgrs/mgrs_aea.shp"):
    ensure_dir(out_dir)
    aoi = aoi_from_bounds(bounds_wsen)
    flowlines = get_flowlines_within_aoi(aoi, local_flowlines_dir=flowlines_local_dir)
    tiles, tiles_gdf = tiles_for_bounds(bounds_wsen, mgrs_shp_path)
    present_map, missing = ndwi_files_for_tiles(ndwi_dir, tiles)
    if missing:
        raise ValueError(f"Missing NDWI tiles: {missing}")
    ndwi_clip = open_ndwi_mosaic(present_map, bounds_wsen)

    dem_cache = os.path.join(out_dir, "dem_bounds_1m.tif")
    dem = get_dem_for_aoi_via_stac(
        aoi_gdf=aoi,
        stac_dir=os.path.expanduser(stac_dir),
        target_crs_epsg=5070,
        cache_path=dem_cache,
        overwrite=False,
        stac_download_cache_dir=os.path.join(out_dir, 'stac_cache'),
        stac_collection_id="usgs-3dep-1m-opr",
    )
    dem_crs = dem.rio.crs
    flowlines_dem = flowlines.to_crs(dem_crs)
    streams = build_streams_mask_from_nhd_ndwi(flowlines_dem, dem, ndwi_da=ndwi_clip, ndwi_threshold=float(ndwi_threshold))
    rem = compute_rem_quick(dem, streams)
    fields = load_and_clip_fields(fields_path, aoi, dem_crs)
    fields_stats = compute_field_rem_stats(fields, rem, stats=("mean",))
    return {
        "aoi": aoi,
        "flowlines": flowlines,
        "ndwi": ndwi_clip,
        "streams": streams,
        "rem": rem,
        "mgrs_tiles": tiles_gdf,
        "fields": fields,
        "fields_stats": fields_stats,
        "summary": {"total_fields": len(fields_stats), "ndwi_threshold": float(ndwi_threshold)},
    }

def load_and_clip_fields(fields_path, aoi_gdf, target_crs):
    """
    Load the statewide irrigation dataset and clip to AOI, reprojecting to target CRS.

    Parameters
    ----------
    fields_path : str
        Path to the statewide irrigation shapefile.
    aoi_gdf : GeoDataFrame
        AOI polygon.
    target_crs : str or dict
        CRS to project the output to (e.g., DEM CRS).

    Returns
    -------
    GeoDataFrame
        Clipped fields in target_crs.
    """
    LOGGER.info("Loading irrigation dataset: %s", fields_path)
    # Assume path exists; if wrong, let it fail upstream per instructions.
    fields = None
    try:
        # Discover file CRS and read with AOI bbox in that CRS to avoid full load
        with fiona.open(fields_path) as src:
            fields_crs = src.crs_wkt if src.crs_wkt else src.crs
        aoi_in_fields = aoi_gdf.to_crs(fields_crs)
        bounds = tuple(aoi_in_fields.total_bounds.tolist())
        fields = gpd.read_file(fields_path, bbox=bounds)
    except Exception:
        fields = gpd.read_file(fields_path)  # fallback to full read
    # Ensure valid geometries; drop empties
    fields = fields[~fields.geometry.is_empty & fields.geometry.notnull()].copy()

    LOGGER.info("Clipping irrigation dataset to AOI")
    try:
        clipped = gpd.clip(fields, aoi_gdf)
    except Exception:
        clipped = gpd.overlay(fields, aoi_gdf, how="intersection")

    clipped = clipped.to_crs(target_crs)
    clipped = clipped.reset_index(drop=True)
    return clipped


def compute_field_rem_stats(fields_gdf, rem_da, stats=("mean",)):
    """
    Compute zonal statistics of REM over irrigation polygons.

    Parameters
    ----------
    fields_gdf : GeoDataFrame
        Polygons reprojected to the same CRS as rem_da.
    rem_da : xarray.DataArray
        REM raster with CRS/transform.
    stats : tuple of str
        Statistics to compute via rasterstats (default: mean).

    Returns
    -------
    GeoDataFrame
        Input fields_gdf with added columns for each requested stat, prefixed by 'rem_'.
    """
    LOGGER.info("Computing zonal statistics over fields (stats: %s)", ",".join(stats))

    # Ensure CRS consistency; reproject fields to REM CRS if needed.
    rem_crs = rem_da.rio.crs
    if rem_crs is None:
        raise ValueError("REM raster has no CRS; cannot compute zonal stats.")
    if fields_gdf.crs is None:
        raise ValueError("Fields GeoDataFrame has no CRS; cannot compute zonal stats.")
    try:
        if not _CRS.from_user_input(fields_gdf.crs).equals(_CRS.from_user_input(rem_crs)):
            fields_gdf = fields_gdf.to_crs(rem_crs)
    except Exception:
        if str(fields_gdf.crs) != str(rem_crs):
            fields_gdf = fields_gdf.to_crs(rem_crs)

    affine = rem_da.rio.transform()
    raster = np.asarray(rem_da.data)
    # Mask NaNs so mean ignores nodata
    raster = np.ma.array(raster, mask=~np.isfinite(raster))

    zs = zonal_stats(
        vectors=fields_gdf.geometry,
        raster=raster,
        affine=affine,
        stats=list(stats),
        nodata=None,
        all_touched=True,
        geojson_out=False,
    )

    df_stats = pd.DataFrame(zs)
    # Prefix columns for clarity
    df_stats = df_stats.rename(columns={s: f"rem_{s}" for s in df_stats.columns})
    result = fields_gdf.reset_index(drop=True).join(df_stats)
    return result


def stratify_fields_by_rem(fields_with_stats_gdf, threshold_m=2.0):
    """
    Add a boolean field 'partitioned' where mean REM < threshold.

    Parameters
    ----------
    fields_with_stats_gdf : GeoDataFrame
        Fields with at least 'rem_mean' column.
    threshold_m : float
        Relative elevation threshold in meters for partitioned fields.

    Returns
    -------
    GeoDataFrame
        Input with added 'partitioned' bool flag.
    """
    if "rem_mean" not in fields_with_stats_gdf.columns:
        raise ValueError("'rem_mean' column not found; compute stats first.")
    out = fields_with_stats_gdf.copy()
    out["partitioned"] = out["rem_mean"] < float(threshold_m)
    return out


def get_dem_for_aoi_via_stac(
        aoi_gdf,
        stac_dir: str,
        target_crs_epsg: int = 5070,
        cache_path: str | None = None,
        overwrite: bool = False,
        stac_download_cache_dir: str | None = None,
        stac_collection_id: str = "usgs-3dep-1m-opr",
):
    """
    Build a DEM mosaic from a local 3DEP STAC for the AOI.

    Parameters
    ----------
    aoi_gdf : GeoDataFrame
        AOI polygon.
    stac_dir : str
        Path to the local STAC directory (containing catalog.json).
    target_crs_epsg : int
        Output DEM CRS.
    cache_path : str, optional
        If provided, write the resulting DEM to this GeoTIFF path and read from it if
        present (unless overwrite=True).
    overwrite : bool
        Overwrite cached DEM if present.
    stac_download_cache_dir : str, optional
        Directory to cache downloaded GeoTIFF tiles referenced by the STAC.
        If not provided, defaults to a sibling folder of cache_path named 'stac_cache'
        or to './stac_cache' in the current working directory.
    stac_collection_id : str
        Collection ID inside the STAC (default: 'usgs-3dep-1m-opr').

    Returns
    -------
    xarray.DataArray
        DEM with CRS/transform set, clipped to AOI, reprojected to target_crs_epsg.
    """
    from handily.stac_3dep import mosaic_from_stac

    target_crs = f"EPSG:{int(target_crs_epsg)}"

    # Cached DEM
    if cache_path and (not overwrite) and os.path.exists(cache_path):
        LOGGER.info("Loading cached DEM (STAC): %s", cache_path)
        dem_cached = rxr.open_rasterio(cache_path)
        if "band" in dem_cached.dims:
            dem_cached = dem_cached.squeeze("band", drop=True)
        return dem_cached

    # Determine download cache for individual STAC tiles
    if stac_download_cache_dir is None:
        if cache_path:
            base = os.path.dirname(os.path.abspath(cache_path))
            stac_download_cache_dir = os.path.join(base, "stac_cache")
        else:
            stac_download_cache_dir = os.path.join(os.getcwd(), "stac_cache")

    LOGGER.info(
        "Mosaicking DEM from STAC at %s (tiles cached in %s)", stac_dir, stac_download_cache_dir
    )
    dem = mosaic_from_stac(
        stac_dir=stac_dir,
        aoi_gdf=aoi_gdf,
        cache_dir=stac_download_cache_dir,
        collection_id=stac_collection_id,
        target_crs_epsg=int(target_crs_epsg),
    )

    if str(dem.rio.crs) != target_crs:
        dem = dem.rio.reproject(target_crs)

    # Optionally persist to cache_path
    if cache_path:
        ensure_dir(os.path.dirname(os.path.abspath(cache_path)))
        LOGGER.info("Saving DEM mosaic to cache: %s", cache_path)
        dem.rio.to_raster(cache_path)
    return dem


# Legacy HUC10/RichDEM path removed; use run_bounds_rem
