#!/usr/bin/env python
"""Well-free str>=7 IDW depth-to-water prior — a shared level-0 product.

The anchors are str>=7 channel-bed DEM **elevations** (NOT well measurements), so the
prior is leakage-free against the GWX validation set: it consumes zero well labels.

For each ConusFAC HUC8 that carries a str>=7 trunk, densify those reaches, sample the
10 m DEM channel-bed elevation along them, pool the anchors across HUC8s, then k-NN
IDW the channel-bed elevation to every well:

    strahler_dtw_m = well_ground_dem - idw(channel_bed_elev)

(the gaining-stream / low-relief base-level assumption — channel bed ≈ water table).
NaN where the nearest str>=7 anchor is farther than --dmax-km (no mainstem support).
Both the well ground elevation and the channel elevation are sampled from the SAME
10 m DEM so the subtraction is on one vertical datum.

Output: `canonical_id, strahler_dtw_m, str7_support_dist_m` — one row per well that
falls in a ConusFAC HUC8. Joined by both the DTW stacker (`build_stacker_features`)
and the WTE GNN builder (`build_conus_graph_inputs`).

Run:
    uv run python utils/build_str7_idw_prior.py \
        --features /data/ssd2/handily/conus/stacker/wte_fac_features.parquet \
        --out      /data/ssd2/handily/conus/stacker/str7_idw_dtw.parquet
"""

import argparse
import glob
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString
from sklearn.neighbors import NearestNeighbors

FAC = Path("/data/ssd2/handily/conus/fac_rem")
FEAT_DEFAULT = "/data/ssd2/handily/conus/stacker/wte_fac_features.parquet"
OUT_DEFAULT = "/data/ssd2/handily/conus/stacker/str7_idw_dtw.parquet"


def densify(geom, step):
    out = []
    geoms = geom.geoms if isinstance(geom, MultiLineString) else [geom]
    for ln in geoms:
        if isinstance(ln, LineString) and ln.length > 0:
            n = max(int(ln.length // step), 1)
            for d in np.linspace(0, ln.length, n + 1):
                p = ln.interpolate(d)
                out.append((p.x, p.y))
    return out


def sample_dem(path, x, y):
    """Sample a DEM (EPSG:5070) at xy; nodata and absurd values -> NaN."""
    with rasterio.open(path) as s:
        v = np.array([t[0] for t in s.sample(np.c_[x, y])], "float64")
        nod = s.nodata
    if nod is not None:
        v[v == nod] = np.nan
    v[(v < -1e3) | (v > 1e4)] = np.nan
    return v


def idw_elev(axy, az, qxy, k, p, eps=1e-6):
    k = min(k, len(axy))
    nn = NearestNeighbors(n_neighbors=k).fit(axy)
    d, i = nn.kneighbors(qxy)
    w = 1.0 / np.power(d + eps, p)
    return (w * az[i]).sum(1) / w.sum(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=FEAT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument(
        "--str-min",
        type=int,
        default=7,
        help="minimum Strahler order for anchor reaches",
    )
    ap.add_argument(
        "--step-m",
        type=float,
        default=150.0,
        help="densify spacing along str>=N reaches",
    )
    ap.add_argument("--idw-k", type=int, default=16)
    ap.add_argument("--idw-power", type=float, default=2.0)
    ap.add_argument(
        "--dmax-km",
        type=float,
        default=50.0,
        help="NaN beyond this distance to nearest anchor",
    )
    args = ap.parse_args()

    df = pd.read_parquet(
        args.features, columns=["canonical_id", "x5070", "y5070", "huc8"]
    ).reset_index(drop=True)
    df["huc8"] = df["huc8"].astype(str).str.zfill(8)
    have = {
        os.path.basename(d.rstrip("/"))
        for d in glob.glob(str(FAC / "*"))
        if (FAC / os.path.basename(d.rstrip("/")) / "streams.fgb").exists()
        and (FAC / os.path.basename(d.rstrip("/")) / "dem_10m.tif").exists()
    }
    df = df[df["huc8"].isin(have)].reset_index(drop=True)
    print(f"wells in {df.huc8.nunique()} ConusFAC HUC8s: {len(df)}")

    well_ground = np.full(len(df), np.nan)
    anchors, n_anchor = [], {}
    for huc8, grp in df.groupby("huc8"):
        demp, strp = FAC / huc8 / "dem_10m.tif", FAC / huc8 / "streams.fgb"
        idx = grp.index.to_numpy()
        well_ground[idx] = sample_dem(
            str(demp), grp.x5070.to_numpy(), grp.y5070.to_numpy()
        )
        st = gpd.read_file(strp)
        if "strahler" not in st.columns:
            n_anchor[huc8] = 0
            continue
        trunk = st[st["strahler"] >= args.str_min]
        pts = (
            [p for g in trunk.geometry for p in densify(g, args.step_m)]
            if len(trunk)
            else []
        )
        if not pts:
            n_anchor[huc8] = 0
            continue
        pa = np.asarray(pts, "float64")
        az = sample_dem(str(demp), pa[:, 0], pa[:, 1])
        ok = np.isfinite(az)
        n_anchor[huc8] = int(ok.sum())
        if ok.any():
            anchors.append(np.c_[pa[ok], az[ok]])

    a = np.vstack(anchors)
    n_with = sum(1 for v in n_anchor.values() if v > 0)
    print(
        f"pooled str>={args.str_min} anchors: {len(a)} from {n_with}/{df.huc8.nunique()} HUC8s"
    )
    print(
        "  per-HUC8 anchor pts:", {k: v for k, v in sorted(n_anchor.items()) if v > 0}
    )
    print(
        "  HUC8s WITHOUT a str>=7 trunk:",
        sorted(k for k, v in n_anchor.items() if v == 0),
    )

    qxy = df[["x5070", "y5070"]].to_numpy("float64")
    nd = cKDTree(a[:, :2]).query(qxy)[0]
    pred = idw_elev(a[:, :2], a[:, 2], qxy, args.idw_k, args.idw_power)
    str7_dtw = well_ground - pred
    far = nd > args.dmax_km * 1000.0
    str7_dtw[far | ~np.isfinite(well_ground)] = np.nan

    out = pd.DataFrame(
        {
            "canonical_id": df["canonical_id"].to_numpy(),
            "strahler_dtw_m": str7_dtw,
            "str7_support_dist_m": np.where(far, np.nan, nd),
        }
    )
    fin = np.isfinite(str7_dtw)
    print(
        f"\nfinite strahler_dtw_m: {int(fin.sum())}/{len(out)} ({100 * fin.mean():.1f}%)"
    )
    print(out.loc[fin, "strahler_dtw_m"].describe().round(2).to_string())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
