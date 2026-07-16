"""Context covariate: EPA Level IV ecoregions, rasterized onto the canonical
CONUS grid (EPSG:5070, 100 m, 49810x31390).

US_L4CODE is a hierarchical string (e.g. "10.1.8"), not an integer, so it is
factorized to integer codes (1..N) with a legend CSV mapping code -> L4CODE,
L4NAME, L3 parent. Skip-if-exists.

Outputs:
  context/ecoregion_l4_code.tif   (Int32, nodata 0)
  context/ecoregion_l4_legend.csv
"""

import os
import csv
import time
import zipfile
import subprocess

import pandas as pd
import geopandas as gpd

DL = "/nas/handily/covariates/_download"
COV = "/nas/handily/covariates"
ZIP = f"{DL}/us_eco_l4.zip"
EXDIR = f"{DL}/ex/us_eco_l4"
GPKG = f"{DL}/ex/_ecol4_5070.gpkg"
OUT = f"{COV}/context/ecoregion_l4_code.tif"
LEG = f"{COV}/context/ecoregion_l4_legend.csv"
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


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    if os.path.exists(OUT):
        log(f"skip (exists): {OUT}")
        return
    log("EPA Level IV ecoregions")

    shp = None
    for root, _d, files in os.walk(EXDIR):
        for f in files:
            if f.lower().endswith(".shp"):
                shp = os.path.join(root, f)
    if shp is None:
        os.makedirs(EXDIR, exist_ok=True)
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(EXDIR)
        for root, _d, files in os.walk(EXDIR):
            for f in files:
                if f.lower().endswith(".shp"):
                    shp = os.path.join(root, f)
    log(f"  shapefile: {shp}")

    g = gpd.read_file(shp)
    cols = {c.upper(): c for c in g.columns}
    code_f, name_f, l3_f = cols["US_L4CODE"], cols["US_L4NAME"], cols.get("US_L3CODE")
    codes, uniques = pd.factorize(g[code_f], sort=True)
    g["l4code"] = (codes + 1).astype("int32")  # 0 reserved for nodata
    log(f"  {len(uniques)} L4 classes over {len(g)} polygons")

    if os.path.exists(GPKG):
        os.remove(GPKG)
    g.to_crs(CRS)[["geometry", "l4code"]].to_file(GPKG, layer="data", driver="GPKG")

    tmp = OUT + ".tmp.tif"
    subprocess.run(
        [
            "gdal_rasterize",
            "-l",
            "data",
            "-a",
            "l4code",
            "-te",
            *TE,
            "-tr",
            *TR,
            "-a_nodata",
            "0",
            "-init",
            "0",
            "-ot",
            "Int32",
            *CO,
            GPKG,
            tmp,
        ],
        check=True,
    )
    os.replace(tmp, OUT)

    legend = g[["l4code", code_f, name_f] + ([l3_f] if l3_f else [])].drop_duplicates(
        "l4code"
    )
    legend = legend.sort_values("l4code")
    with open(LEG, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["code", "us_l4code", "us_l4name"] + (["us_l3code"] if l3_f else [])
        w.writerow(hdr)
        for _, r in legend.iterrows():
            row = [int(r["l4code"]), r[code_f], r[name_f]] + ([r[l3_f]] if l3_f else [])
            w.writerow(row)
    log(f"  wrote {OUT} and {LEG}")


if __name__ == "__main__":
    main()
