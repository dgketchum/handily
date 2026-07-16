"""Anthropogenic stress covariate: county total groundwater-withdrawal density.

Joins USGS 2015 county water-use (`usco2015v2.0.csv`, column TO-WGWTo = total
groundwater withdrawals, fresh+saline, all categories, Mgal/d) to Census 2018
county boundaries on FIPS/GEOID, divides by 5070 county area to get a
withdrawal density (Mgal/d per km2), and rasterizes onto the canonical CONUS
grid (EPSG:5070, 100 m, 49810x31390).

Output: anthropogenic/gw_withdrawal_density.tif (Float32, nodata -9999).
Skip-if-exists.
"""

import os
import time
import subprocess

import pandas as pd
import geopandas as gpd

DL = "/nas/handily/covariates/_download"
COV = "/nas/handily/covariates"
CSV = f"{DL}/usco2015v2.0.csv"
COUNTIES = f"{DL}/ex/counties/cb_2018_us_county_500k.shp"
OUT = f"{COV}/anthropogenic/gw_withdrawal_density.tif"
GPKG = f"{DL}/ex/_gw_withdrawal_5070.gpkg"
TE = ["-2540000", "119000", "2441000", "3258000"]
TR = ["100", "100"]
CRS = "EPSG:5070"
CO = [
    "-co",
    "COMPRESS=DEFLATE",
    "-co",
    "PREDICTOR=2",
    "-co",
    "TILED=YES",
    "-co",
    "BIGTIFF=YES",
]
# AK, HI, and territories (PR/VI/GU/AS/MP) are off the CONUS grid.
NON_CONUS = {"02", "15", "60", "66", "69", "72", "78"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    if os.path.exists(OUT):
        log(f"skip (exists): {OUT}")
        return
    log("water-use groundwater withdrawal density")

    wu = pd.read_csv(CSV, skiprows=1, dtype={"FIPS": str})
    wu = wu[["FIPS", "TO-WGWTo"]].copy()
    wu["gw_mgd"] = pd.to_numeric(wu["TO-WGWTo"], errors="raise")
    log(
        f"  water-use rows: {len(wu)}  gw Mgal/d range {wu.gw_mgd.min():.2f}–{wu.gw_mgd.max():.1f}"
    )

    c = gpd.read_file(COUNTIES)
    c = c[~c["STATEFP"].isin(NON_CONUS)].to_crs(CRS)
    c["area_km2"] = c.geometry.area / 1e6
    log(f"  CONUS counties: {len(c)}")

    m = c.merge(wu, left_on="GEOID", right_on="FIPS", how="left")
    n_unmatched = int(m["gw_mgd"].isna().sum())
    if n_unmatched:
        # boundary vintage drift (2015 water-use FIPS vs 2018 county GEOIDs):
        # these counties have no withdrawal estimate, so they stay nodata holes
        # rather than being fabricated. Report which.
        miss = m.loc[m["gw_mgd"].isna(), ["GEOID", "NAME"]].values.tolist()
        log(
            f"  WARNING: {n_unmatched} CONUS counties unmatched to water-use (left nodata): {miss}"
        )
    m = m[m["gw_mgd"].notna()].copy()
    m["gw_dens"] = (m["gw_mgd"] / m["area_km2"]).astype("float64")
    log(f"  density Mgal/d/km2 range {m.gw_dens.min():.5f}–{m.gw_dens.max():.4f}")

    if os.path.exists(GPKG):
        os.remove(GPKG)
    m[["geometry", "gw_dens"]].to_file(GPKG, layer="data", driver="GPKG")

    tmp = OUT + ".tmp.tif"
    subprocess.run(
        [
            "gdal_rasterize",
            "-l",
            "data",
            "-a",
            "gw_dens",
            "-te",
            *TE,
            "-tr",
            *TR,
            "-a_nodata",
            "-9999",
            "-init",
            "-9999",
            "-ot",
            "Float32",
            *CO,
            GPKG,
            tmp,
        ],
        check=True,
    )
    os.replace(tmp, OUT)
    log(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
