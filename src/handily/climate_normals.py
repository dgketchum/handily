"""gridMET climatological normals for water-table covariates.

gridMET (~4 km daily, CONUS) variables are stored as stacked NetCDF, one file
per variable::

    gridmet_pr.nc    precipitation_amount          mm/day
    gridmet_pet.nc   potential_evapotranspiration  mm/day (grass reference ET)
    gridmet_vpd.nc   mean_vapor_pressure_deficit   kPa
    gridmet_tmmx.nc  air_temperature               K (daily max)

The daily stack is reduced to long-term means: flux variables (pr, pet) are
summed to a mean-annual total, state variables (vpd, temperature) are averaged.

The **aridity index** ``AI = P / PET`` (UNEP convention) is the transferable
physical driver of regional water-table depth: arid basins (AI < 0.2) carry a
deep table far below the land surface, while humid settings track topography.
Deriving AI from P and PET — with no basin-specific calibration — is what lets a
regional water-table model fit in one basin transfer toward statewide and CONUS
scale, rather than memorizing local coordinates.

Note: the local gridMET archive currently spans ~2020-2026, so these are a
short-record climatology, not a 30-year normal. Adequate as a first-order
aridity covariate; revisit if a longer record becomes available.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# key -> (filename, variable, kind). "flux" is summed to an annual total;
# "state" is time-averaged.
_VARS = {
    "pr": ("gridmet_pr.nc", "precipitation_amount", "flux"),
    "pet": ("gridmet_pet.nc", "potential_evapotranspiration", "flux"),
    "vpd": ("gridmet_vpd.nc", "mean_vapor_pressure_deficit", "state"),
    "tmmx": ("gridmet_tmmx.nc", "air_temperature", "state"),
}
_DAYS_PER_YEAR = 365.25


def _climatology_field(gridmet_dir: str, key: str, bbox: tuple) -> xr.DataArray:
    """2-D (lat, lon) long-term mean of one gridMET variable, clipped to bbox.

    bbox is (west, south, east, north) in degrees. Only the spatial hyperslab is
    read across the time axis, so this stays cheap for a small basin window.
    """
    fname, var, kind = _VARS[key]
    path = Path(gridmet_dir) / fname
    if not path.exists():
        raise FileNotFoundError(f"gridMET file not found: {path}")
    ds = xr.open_dataset(path)
    da = ds[var].sortby("lat").sortby("lon")
    w, s, e, n = bbox
    da = da.sel(lon=slice(w, e), lat=slice(s, n))
    if da.size == 0:
        raise ValueError(f"gridMET {key}: no cells in bbox {bbox}")
    time_dim = "day" if "day" in da.dims else da.dims[0]
    if kind == "flux":
        field = da.mean(time_dim) * _DAYS_PER_YEAR  # mm/day -> mm/yr
    else:
        field = da.mean(time_dim)
    return field.load()


def sample_gridmet_normals(
    gdf, gridmet_dir: str, bbox_margin_deg: float = 0.25
) -> pd.DataFrame:
    """Sample gridMET climatological normals at point geometries.

    Returns a DataFrame (indexed like ``gdf``) with columns:
    ``precip_mm_yr``, ``pet_mm_yr``, ``aridity_index`` (= P/PET), ``vpd_kpa``,
    ``tmax_c``. Points are matched to the nearest 4 km gridMET cell.
    """
    pts = gdf.to_crs(4326)
    lon = xr.DataArray(
        np.array([g.x for g in pts.geometry], dtype="float64"), dims="pt"
    )
    lat = xr.DataArray(
        np.array([g.y for g in pts.geometry], dtype="float64"), dims="pt"
    )
    bbox = (
        float(lon.min()) - bbox_margin_deg,
        float(lat.min()) - bbox_margin_deg,
        float(lon.max()) + bbox_margin_deg,
        float(lat.max()) + bbox_margin_deg,
    )

    sampled = {}
    for key in ("pr", "pet", "vpd", "tmmx"):
        field = _climatology_field(gridmet_dir, key, bbox)
        sampled[key] = field.sel(lon=lon, lat=lat, method="nearest").values

    precip = sampled["pr"]
    pet = sampled["pet"]
    out = pd.DataFrame(
        {
            "precip_mm_yr": precip,
            "pet_mm_yr": pet,
            "aridity_index": np.where(pet > 0, precip / pet, np.nan),
            "vpd_kpa": sampled["vpd"],
            "tmax_c": sampled["tmmx"] - 273.15,
        },
        index=gdf.index,
    )
    return out
