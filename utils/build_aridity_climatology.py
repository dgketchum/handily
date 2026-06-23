"""Build a static gridMET aridity climatology (CONUS) -- the Tier-3 real-deep covariate.

The stacker's failure (notes/WTE_ELEVATION_SURFACE.md, Step 8) is that it cannot tell
artifact-deep (the 2 km WTE sagging below a perched well) from a genuinely deep
regional water table; both look like a large wte_dtw. elev_above_coarse_m / slope /
TRI flag the *artifact* side. Aridity flags the *real* side: arid basins (low P/PET)
carry deep regional tables, humid settings shallow ones -- separating the two deep
populations the relief features cannot.

Source: gridMET (4 km, 1/24deg, EPSG:4326), chosen over ERA5 (~10 km) for the ~4x
finer grid. Daily precip and reference ET live as per-year netCDF:
  /data/hdd1/gridmet/pr_raw/pr_{year}.nc   var precipitation_amount        (mm)
  /data/hdd1/gridmet/pet_raw/pet_{year}.nc var potential_evapotranspiration (mm, +)
gridMET PET is a positive grass-reference ET (no sign flip needed, unlike ERA5).

For each year in the normal window we sum the daily bands -> annual total, average
across years -> mean annual P and PET (mm/yr), and form the UNEP aridity index
AI = P / PET (low = arid). Ocean / off-CONUS is a static NaN land mask and stays NaN.

Outputs single-band GeoTIFFs on the native gridMET grid (sample at well lon/lat with
build_stacker_features.sample_coarse, which is CRS-agnostic):
  gridmet_mean_annual_precip_mm.tif, gridmet_mean_annual_pet_mm.tif,
  gridmet_aridity_index.tif

Usage:
    uv run python utils/build_aridity_climatology.py \
        --out-dir /data/ssd2/handily/conus/covariates
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor used to write GeoTIFFs)
import xarray as xr

log = logging.getLogger("build_aridity_climatology")

GRIDMET_DIR = "/data/hdd1/gridmet"
VAR_P = "precipitation_amount"
VAR_PET = "potential_evapotranspiration"
OUT_DIR = "/data/ssd2/handily/conus/covariates"
YEAR0, YEAR1 = 1991, 2020  # WMO 30-year normal


def read_year(path: str, var: str) -> np.ndarray:
    """Load one gridMET year as (day, lat, lon) float64."""
    with xr.open_dataset(path) as ds:
        return ds[var].values.astype(np.float64)


def annual_total(stack: np.ndarray) -> np.ndarray:
    """Sum all daily layers -> annual total. Handles 365/366 days transparently.

    gridMET is masked to CONUS land: ocean / off-domain cells are NaN, an identical
    STATIC mask every day (legitimate -- no land value over the sea), so they stay NaN
    through the sum and are never sampled at the land wells. A per-day-VARYING NaN
    footprint would be real corruption -- guard that rather than averaging over it.
    """
    nan = np.isnan(stack)
    if not (nan == nan[0]).all():
        raise ValueError(
            "non-static NaN mask in gridMET year (per-day gaps -> corruption)"
        )
    return stack.sum(axis=0)  # land -> annual total; ocean stays NaN


def build(
    out_dir: str = OUT_DIR,
    *,
    gridmet_dir: str = GRIDMET_DIR,
    var_p: str = VAR_P,
    var_pet: str = VAR_PET,
    year0: int = YEAR0,
    year1: int = YEAR1,
) -> dict[str, str]:
    years = list(range(year0, year1 + 1))
    p_sum = pet_sum = None
    first_p_path = None
    for yr in years:
        p_path = f"{gridmet_dir}/pr_raw/pr_{yr}.nc"
        pet_path = f"{gridmet_dir}/pet_raw/pet_{yr}.nc"
        for pth in (p_path, pet_path):
            if not Path(pth).exists():
                raise FileNotFoundError(
                    f"missing required year in normal window: {pth}"
                )
        first_p_path = first_p_path or p_path
        p_yr = annual_total(read_year(p_path, var_p))
        pet_yr = np.abs(annual_total(read_year(pet_path, var_pet)))  # gridMET PET is +
        if p_sum is None:
            p_sum = np.zeros_like(p_yr)
            pet_sum = np.zeros_like(pet_yr)
        p_sum += p_yr
        pet_sum += pet_yr
        log.info(
            "accumulated %d (P %.0f mm, PET %.0f mm land-mean)",
            yr,
            np.nanmean(p_yr),
            np.nanmean(pet_yr),
        )

    n = len(years)
    p_mean = (p_sum / n).astype(np.float32)
    pet_mean = (pet_sum / n).astype(np.float32)
    ai = (p_mean / pet_mean).astype(np.float32)  # PET > 0 over land everywhere

    # Coords for georeferencing the outputs (gridMET native grid, EPSG:4326).
    with xr.open_dataset(first_p_path) as ds:
        lat = ds["lat"]
        lon = ds["lon"]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    outs = {
        "gridmet_mean_annual_precip_mm.tif": p_mean,
        "gridmet_mean_annual_pet_mm.tif": pet_mean,
        "gridmet_aridity_index.tif": ai,
    }
    paths = {}
    for name, arr in outs.items():
        da = xr.DataArray(arr, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs(
            "EPSG:4326"
        )
        pth = f"{out_dir}/{name}"
        da.rio.to_raster(pth, compress="deflate")
        paths[name] = pth
        log.info("wrote %s (land-mean %.4f)", pth, float(np.nanmean(arr)))
    log.info(
        "gridMET aridity climatology %d-%d (%d yr): AI land-mean %.3f [%.3f arid .. %.3f humid]",
        year0,
        year1,
        n,
        float(np.nanmean(ai)),
        float(np.nanmin(ai)),
        float(np.nanmax(ai)),
    )
    return paths


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gridmet-dir", default=GRIDMET_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--year0", type=int, default=YEAR0)
    p.add_argument("--year1", type=int, default=YEAR1)
    args = p.parse_args(argv)
    build(
        args.out_dir, gridmet_dir=args.gridmet_dir, year0=args.year0, year1=args.year1
    )


if __name__ == "__main__":
    main()
