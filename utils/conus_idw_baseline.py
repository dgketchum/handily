"""Decisive cheap baseline: can spatially-cross-validated IDW of the GWX well
DTW beat the Janssen CONUS WTD benchmark?

This defines the achievable bar for a CONUS DTW prior BEFORE investing in HAND
sampling + the flow graph. If well-interpolation alone (no terrain, no graph)
can't beat Janssen out-of-block, a HAND-feature-dominated GNN won't either.

Spatial blocked CV: wells binned into ~`block-km` grid cells (EPSG:5070),
cells round-robin assigned to K folds. For each held-out fold, predict DTW at
its wells from training-fold wells via inverse-distance kNN. Compare to Janssen
sampled at the same wells, on the common footprint. Panel: overall, by lon-third
region, by observed-DTW band. Excludes NWIS/NGWMN from the headline (Ma/Janssen
lineage / leakage hygiene), reports them separately.
"""

import argparse

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

WELLS = "/data/ssd2/gwx/products/current/wells.geoparquet"
JANSSEN = "/nas/gwx/janssen/V2_140.tif"
NWIS = {"nwis", "ngwmn"}


def load_wells():
    cols = [
        "canonical_id",
        "source",
        "longitude",
        "latitude",
        "mean_dtw",
        "confinement_class",
        "well_class",
    ]
    df = pd.read_parquet(WELLS, columns=cols)
    df = df[df["confinement_class"].isin(["unconfined", "unconfined_marginal"])]
    df = df[
        np.isfinite(df["mean_dtw"]) & (df["mean_dtw"] >= 0) & (df["mean_dtw"] < 1000)
    ]
    df = df[np.isfinite(df["longitude"]) & np.isfinite(df["latitude"])]
    # CONUS only: FIM/Janssen are CONUS, and EPSG:5070 (CONUS Albers) is invalid
    # off-CONUS (AK/HI/territories project to inf). Drops ~0.12% of wells.
    df = df[df["longitude"].between(-125, -66) & df["latitude"].between(24, 50)]
    tr = Transformer.from_crs(4326, 5070, always_xy=True)
    df["x"], df["y"] = tr.transform(df["longitude"].values, df["latitude"].values)
    return df.reset_index(drop=True)


def sample_janssen(df):
    with rasterio.open(JANSSEN) as ds:
        arr = ds.read(1)
        nd = ds.nodata
        inv = ~ds.transform
        cols, rows = inv * (df["longitude"].values, df["latitude"].values)
        cols = np.floor(cols).astype(int)
        rows = np.floor(rows).astype(int)
        ok = (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
        out = np.full(len(df), np.nan)
        v = arr[np.clip(rows, 0, ds.height - 1), np.clip(cols, 0, ds.width - 1)]
        out[ok] = v[ok]
    out[(out == nd) | ~np.isfinite(out)] = np.nan
    out[out < 0] = np.nan
    return out


def idw_cv(df, block_km, k, folds, power, seed):
    bs = block_km * 1000.0
    bx = np.floor(df["x"].values / bs).astype(np.int64)
    by = np.floor(df["y"].values / bs).astype(np.int64)
    block = bx * 100000 + by
    ublk = np.unique(block)
    rng = np.random.RandomState(seed)
    fold_of = {b: i for i, b in enumerate(rng.permutation(ublk))}
    blk_fold = np.array([fold_of[b] % folds for b in block])

    pred = np.full(len(df), np.nan)
    xy = df[["x", "y"]].values
    dtw = df["mean_dtw"].values
    for f in range(folds):
        te = blk_fold == f
        trn = ~te
        if te.sum() == 0 or trn.sum() == 0:
            continue
        tree = cKDTree(xy[trn])
        dist, idx = tree.query(xy[te], k=k)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        w = 1.0 / np.maximum(dist, 1.0) ** power
        vals = dtw[trn][idx]
        pred[te] = (w * vals).sum(1) / w.sum(1)
    return pred, block


def panel(name, pred, jn, dtw, mask):
    p, j, d = pred[mask], jn[mask], dtw[mask]
    fin = np.isfinite(p) & np.isfinite(j) & np.isfinite(d)
    p, j, d = p[fin], j[fin], d[fin]
    if len(d) == 0:
        print(f"  {name}: no common-footprint rows")
        return
    idw_mad = np.median(np.abs(p - d))
    jn_mad = np.median(np.abs(j - d))
    idw_rmse = np.sqrt(np.mean((p - d) ** 2))
    jn_rmse = np.sqrt(np.mean((j - d) ** 2))
    flag = "IDW<" if idw_mad < jn_mad else "    "
    print(
        f"  {name:22s} n={len(d):7d}  IDW MAD={idw_mad:6.2f} RMSE={idw_rmse:6.1f} | "
        f"Janssen MAD={jn_mad:6.2f} RMSE={jn_rmse:6.1f}  {flag}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-km", type=float, default=40.0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dump", default=None, help="parquet path to dump per-well OOF predictions"
    )
    args = ap.parse_args()

    df = load_wells()
    print(f"loaded {len(df):,} unconfined GWX wells with DTW", flush=True)
    df["janssen"] = sample_janssen(df)
    pred, block = idw_cv(df, args.block_km, args.k, args.folds, args.power, args.seed)
    df["idw"] = pred
    df["block"] = block

    if args.dump:
        df[
            [
                "canonical_id",
                "source",
                "longitude",
                "latitude",
                "x",
                "y",
                "mean_dtw",
                "janssen",
                "idw",
                "block",
            ]
        ].to_parquet(args.dump)
        print(f"dumped per-well OOF predictions -> {args.dump}", flush=True)

    nwis = df["source"].isin(NWIS).values
    jn, dtw, idw = df["janssen"].values, df["mean_dtw"].values, df["idw"].values
    lon = df["longitude"].values

    print(
        f"\n=== spatially-blocked IDW vs Janssen (block={args.block_km}km k={args.k} "
        f"p={args.power} folds={args.folds}) ==="
    )
    print(f"common-footprint Janssen coverage: {np.isfinite(jn).mean() * 100:.1f}%")

    print("\n-- INDEPENDENT (non-NWIS) headline --")
    ind = ~nwis
    panel("ALL independent", idw, jn, dtw, ind)
    print("  by lon-third:")
    panel("  West (lon<-104)", idw, jn, dtw, ind & (lon < -104))
    panel("  Plains(-104..-90)", idw, jn, dtw, ind & (lon >= -104) & (lon < -90))
    panel("  East (lon>=-90)", idw, jn, dtw, ind & (lon >= -90))
    print("  by observed DTW band:")
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 30), (30, 1000)]:
        panel(f"  {lo}-{hi}m", idw, jn, dtw, ind & (dtw >= lo) & (dtw < hi))

    print("\n-- NWIS/NGWMN (diagnostic only) --")
    panel("ALL nwis", idw, jn, dtw, nwis)


if __name__ == "__main__":
    main()
