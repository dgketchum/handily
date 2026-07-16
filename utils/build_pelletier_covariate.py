"""Geology/subsurface covariate: Pelletier et al. (2016) sediment/regolith
thickness (ORNL DAAC, 1 km global, Earthdata-gated) warped to the canonical
CONUS grid (EPSG:5070, 100 m).

Two layers (complementary to the ISRIC SoilGrids depth-to-bedrock already built;
let the model choose between the process-based Pelletier estimate and the
SoilGrids statistical one):
  average_soil_and_sedimentary-deposit_thickness  -> total sediment to bedrock
  upland_valley-bottom_and_lowland_sedimentary_deposit_thickness -> basin fill

Output (Float32, nodata -9999):
  geology/sediment_thickness_avg_m.tif
  geology/sediment_thickness_basinfill_m.tif
Skip-if-exists. Auth via ~/.netrc (machine urs.earthdata.nasa.gov).
"""

import os
import time
import subprocess

DL = "/nas/handily/covariates/_download/ex/pelletier"
COV = "/nas/handily/covariates"
BASE = "https://daac.ornl.gov/daacdata/global_soil/Global_Soil_Regolith_Sediment/data"
COOK = "/nas/handily/covariates/_download/_urs_cookies"
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

LAYERS = [
    (
        "average_soil_and_sedimentary-deposit_thickness.tif",
        f"{COV}/geology/sediment_thickness_avg_m.tif",
    ),
    (
        "upland_valley-bottom_and_lowland_sedimentary_deposit_thickness.tif",
        f"{COV}/geology/sediment_thickness_basinfill_m.tif",
    ),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    os.makedirs(DL, exist_ok=True)
    for fname, out in LAYERS:
        if os.path.exists(out):
            log(f"skip (exists): {out}")
            continue
        src = f"{DL}/{fname}"
        if not os.path.exists(src):
            log(f"download {fname}")
            subprocess.run(
                [
                    "curl",
                    "-sSL",
                    "--netrc",
                    "-c",
                    COOK,
                    "-b",
                    COOK,
                    "-o",
                    src,
                    f"{BASE}/{fname}",
                ],
                check=True,
            )
        log(f"  {fname} {os.path.getsize(src) / 1e6:.0f} MB; warping")
        tmp = out + ".tmp.tif"
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
                "bilinear",
                "-ot",
                "Float32",
                "-dstnodata",
                "-9999",
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
        os.replace(tmp, out)
        log(f"  wrote {out}")
    log("PELLETIER PASS COMPLETE")


if __name__ == "__main__":
    main()
