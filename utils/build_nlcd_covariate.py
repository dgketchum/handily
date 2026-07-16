"""Ecohydrology covariate: NLCD land cover (Annual NLCD Collection 1.1, 2024)
warped from native 30 m EPSG:5070 to the canonical CONUS grid (100 m) with
mode (majority) resampling, since land cover is categorical.

Source: USGS ScienceBase Annual NLCD C1.1 (June 2025).
Output: ecohydro/landcover_nlcd_2024.tif (Byte) + legend CSV. Skip-if-exists.

Standard NLCD class codes are preserved (11 water, 12 ice, 21-24 developed,
31 barren, 41-43 forest, 52 shrub, 71 grassland, 81 pasture, 82 cropland,
90/95 wetlands).
"""

import os
import csv
import time
import zipfile
import subprocess

DL = "/nas/handily/covariates/_download"
COV = "/nas/handily/covariates"
# ScienceBase direct-download endpoint (the /manager/download/ URL returns the
# SPA HTML, not the file); catalog/file/get redirects to the S3-backed object.
URL = (
    "https://www.sciencebase.gov/catalog/file/get/6810c1a4d4be022940554075"
    "?name=Annual_NLCD_LndCov_2024_CU_C1V1.zip"
)
ZIP = f"{DL}/Annual_NLCD_LndCov_2024_CU_C1V1.zip"
EXDIR = f"{DL}/ex/nlcd_2024"
OUT = f"{COV}/ecohydro/landcover_nlcd_2024.tif"
LEG = f"{COV}/ecohydro/landcover_nlcd_legend.csv"
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

LEGEND = [
    (11, "Open Water"),
    (12, "Perennial Ice/Snow"),
    (21, "Developed, Open Space"),
    (22, "Developed, Low Intensity"),
    (23, "Developed, Medium Intensity"),
    (24, "Developed, High Intensity"),
    (31, "Barren Land"),
    (41, "Deciduous Forest"),
    (42, "Evergreen Forest"),
    (43, "Mixed Forest"),
    (52, "Shrub/Scrub"),
    (71, "Herbaceous"),
    (81, "Hay/Pasture"),
    (82, "Cultivated Crops"),
    (90, "Woody Wetlands"),
    (95, "Emergent Herbaceous Wetlands"),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    if os.path.exists(OUT):
        log(f"skip (exists): {OUT}")
        return
    os.makedirs(f"{COV}/ecohydro", exist_ok=True)
    log("NLCD land cover (Annual NLCD 2024 C1.1)")

    if not os.path.exists(ZIP):
        log("  downloading zip")
        subprocess.run(["curl", "-sSL", "-o", ZIP, URL], check=True)
    log(f"  zip {os.path.getsize(ZIP) / 1e6:.0f} MB")

    src = None
    if os.path.isdir(EXDIR):
        for r, _d, fs in os.walk(EXDIR):
            for f in fs:
                if f.lower().endswith(".tif"):
                    src = os.path.join(r, f)
    if src is None:
        os.makedirs(EXDIR, exist_ok=True)
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(EXDIR)
        for r, _d, fs in os.walk(EXDIR):
            for f in fs:
                if f.lower().endswith(".tif"):
                    src = os.path.join(r, f)
    log(f"  source tif: {src}")

    tmp = OUT + ".tmp.tif"
    subprocess.run(
        [
            "gdalwarp",
            "-t_srs",
            CRS,
            "-te",
            *TE,
            "-tr",
            *TR,
            "-r",
            "mode",
            "-ot",
            "Byte",
            "-dstnodata",
            "0",
            "-overwrite",
            "-wo",
            "NUM_THREADS=ALL_CPUS",
            "-multi",
            *CO,
            src,
            tmp,
        ],
        check=True,
    )
    os.replace(tmp, OUT)

    with open(LEG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "class"])
        w.writerows(LEGEND)
    log(f"  wrote {OUT} and {LEG}")


if __name__ == "__main__":
    main()
