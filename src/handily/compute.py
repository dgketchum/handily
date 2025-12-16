import logging

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS as _CRS
from rasterio import features
from rasterstats import zonal_stats
from scipy import ndimage as ndi

LOGGER = logging.getLogger("handily.compute")


def rasterize_lines_to_grid(lines_gdf, template_da, burn_value=1):
    """Rasterize line features to match a template DataArray grid."""
    transform = template_da.rio.transform()
    shape = template_da.shape

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
    return stream_da.rio.write_crs(template_da.rio.crs, inplace=False)


def build_streams_mask_from_nhd_ndwi(flowlines_gdf, dem_da, ndwi_da=None, ndwi_threshold=None):
    """Build a streams/water mask on the DEM grid by combining NHD flowlines with NDWI."""
    if dem_da.rio.crs is None:
        raise ValueError("DEM must have a valid CRS.")
    if flowlines_gdf.crs is None or str(flowlines_gdf.crs) != str(dem_da.rio.crs):
        flowlines_gdf = flowlines_gdf.to_crs(dem_da.rio.crs)

    streams_mask = rasterize_lines_to_grid(flowlines_gdf, dem_da, burn_value=1)

    if ndwi_da is not None and ndwi_threshold is not None:
        ndwi_match = ndwi_da
        if str(ndwi_da.rio.crs) != str(dem_da.rio.crs) or ndwi_da.shape != dem_da.shape:
            ndwi_match = ndwi_da.rio.reproject_match(dem_da)
        water = (ndwi_match > float(ndwi_threshold)).astype("uint8")
        water = water.rio.write_crs(dem_da.rio.crs, inplace=False)
        combo = xr.DataArray(
            np.asarray(streams_mask.data, dtype="uint8") * np.asarray(water.data, dtype="uint8"),
            dims=dem_da.dims,
            coords=dem_da.coords,
            name="streams",
        )
        streams_mask = combo.rio.write_crs(dem_da.rio.crs, inplace=False)

    return streams_mask


def compute_rem_quick(dem_da, streams_da, radius: int = 1000):
    """Compute a quick Relative Elevation Model using a local mean water-surface base elevation."""
    if dem_da.rio.crs is None or streams_da.rio.crs is None:
        raise ValueError("Both DEM and streams rasters must have a valid CRS.")
    if str(dem_da.rio.crs) != str(streams_da.rio.crs) or dem_da.shape != streams_da.shape:
        raise ValueError("DEM and streams must share grid shape and CRS.")

    if int(radius) < 1:
        raise ValueError("radius must be >= 1")

    dem_np = np.asarray(dem_da.data, dtype="float32")
    streams_np = np.asarray(streams_da.data).astype(bool)
    if streams_np.sum() == 0:
        raise ValueError("Stream mask has no active cells after combination.")

    streams_f = streams_np.astype("float32")
    sigma = float(radius) / 3.0
    mean_elev_masked = ndi.gaussian_filter(dem_np * streams_f, sigma=sigma, mode="nearest", truncate=3.0)
    mean_mask = ndi.gaussian_filter(streams_f, sigma=sigma, mode="nearest", truncate=3.0)
    base_fallback = float(np.nanmean(dem_np[streams_np]))
    base_elev = np.divide(
        mean_elev_masked,
        mean_mask,
        out=np.full_like(mean_elev_masked, base_fallback, dtype="float32"),
        where=mean_mask > 0,
    )

    rem_np = dem_np - base_elev
    rem_np = np.where(rem_np < 0, 0, rem_np)

    rem_da = xr.DataArray(
        rem_np,
        dims=dem_da.dims,
        coords=dem_da.coords,
        name="REM",
        attrs={"description": "Quick REM (local mean water base; no sink filling)"},
    )
    return rem_da.rio.write_crs(dem_da.rio.crs, inplace=False)


def compute_field_rem_stats(fields_gdf, rem_da, stats=("mean",)):
    """Compute zonal statistics of REM over polygon features."""
    LOGGER.info("Computing zonal statistics over fields (stats: %s)", ",".join(stats))

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
    df_stats = df_stats.rename(columns={c: f"rem_{c}" for c in df_stats.columns})
    return fields_gdf.reset_index(drop=True).join(df_stats)


def stratify_fields_by_rem(fields_with_stats_gdf, threshold_m=2.0):
    """Add boolean column `partitioned` where mean REM < threshold."""
    if "rem_mean" not in fields_with_stats_gdf.columns:
        raise ValueError("'rem_mean' column not found; compute stats first.")
    out = fields_with_stats_gdf.copy()
    out["partitioned"] = out["rem_mean"] < float(threshold_m)
    return out
