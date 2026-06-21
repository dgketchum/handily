"""One-time fetch of national NHDPlus V2 flowline geometry for the CONUS graph.

``nhdplus_vaa()`` gives reach topology + attributes but no geometry. Lateral
edges (well -> controlling reach) need a reach location, so this pulls the
seamless NHDFlowline_Network layer from the EPA national NHDPlus dataset (7.3 GB
7z, downloaded + cached once under --data-dir), reprojects to EPSG:5070, and
writes a slim parquet keyed by COMID:

    comid   int64        joins to nhdplus_vaa().comid
    cx, cy  float64      representative-point coords (5070) for cKDTree candidates
    geometry LineString  for exact point-to-line distance refinement

Run via commands.sh (background; the download dominates wall-clock).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pynhd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_conus_nhd_geom")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data/ssd2/handily/conus/cache")
    ap.add_argument(
        "--out", default="/data/ssd2/handily/conus/wte_gnn/nhd_flowline_geom.parquet"
    )
    args = ap.parse_args()
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "downloading/reading NHDFlowline_Network (7.3 GB cached @ %s)...", args.data_dir
    )
    fl = pynhd.nhdplus_l48(layer="NHDFlowline_Network", data_dir=args.data_dir)
    log.info("read %d flowline features (crs=%s)", len(fl), fl.crs)

    comid_col = next(c for c in fl.columns if c.lower() in ("comid", "nhdplusid"))
    fl = fl[[comid_col, "geometry"]].rename(columns={comid_col: "comid"})
    fl["comid"] = fl["comid"].astype("int64")
    fl = fl[fl.geometry.notna() & ~fl.geometry.is_empty].to_crs(5070)

    rep = fl.geometry.representative_point()
    fl["cx"] = rep.x.to_numpy()
    fl["cy"] = rep.y.to_numpy()
    fin = np.isfinite(fl["cx"]) & np.isfinite(fl["cy"])
    log.info("dropping %d features with non-finite rep points", int((~fin).sum()))
    fl = fl[fin].reset_index(drop=True)

    fl[["comid", "cx", "cy", "geometry"]].to_parquet(args.out)
    log.info("wrote %d reach geometries -> %s", len(fl), args.out)


if __name__ == "__main__":
    main()
