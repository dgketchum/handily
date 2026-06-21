"""Production CONUS well-node sampler: OWP HAND + Janssen at GWX unconfined wells.

Pipeline:
  1. Cache the 2135 per-HUC8 WBD polygons from the CIROH OWP mirror (also the
     FIM-coverage mask: wells outside these HUC8s have no HAND and are dropped).
  2. sjoin all unconfined GWX wells (finite DTW) to their HUC8.
  3. Per HUC8, sample HAND = min over level-path branch rem rasters via /vsicurl
     (height above nearest drainage). Resumable per-HUC8 parquet shards.
  4. Concat shards, sample the Janssen CONUS WTD benchmark, write the well table.

Output columns: canonical_id, source, longitude, latitude, x5070, y5070,
confinement_class, well_class, mean_dtw, huc8, hand_m, janssen_dtw.

Run via commands.sh (multi-hour; nohup + log). Re-running skips finished shards.
"""

import argparse
import io
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "512")

BASE = "https://ciroh-owp-hand-fim.s3.amazonaws.com/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
WELLS = "/data/ssd2/gwx/products/current/wells.geoparquet"
JANSSEN = "/nas/gwx/janssen/V2_140.tif"
HAND_NODATA = -999999.0


def s3_list(prefix, delimiter=True, max_keys=1000):
    keys, prefixes, token = [], [], None
    while True:
        u = (
            BASE
            + f"?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys={max_keys}"
        )
        if delimiter:
            u += "&delimiter=/"
        if token:
            u += "&continuation-token=" + urllib.parse.quote(token)
        x = urllib.request.urlopen(u, timeout=60).read().decode()
        r = ET.fromstring(x)
        prefixes += [
            p.find("s3:Prefix", NS).text for p in r.findall("s3:CommonPrefixes", NS)
        ]
        keys += [k.find("s3:Key", NS).text for k in r.findall("s3:Contents", NS)]
        if r.find("s3:IsTruncated", NS).text == "true":
            token = r.find("s3:NextContinuationToken", NS).text
        else:
            break
    return prefixes, keys


def all_huc8s():
    prefixes, _ = s3_list("", delimiter=True)
    return sorted(
        p.strip("/")
        for p in prefixes
        if len(p.strip("/")) == 8 and p.strip("/").isdigit()
    )


def fetch_huc8_poly(huc):
    try:
        raw = urllib.request.urlopen(BASE + f"{huc}/wbd8_clp.gpkg", timeout=90).read()
        g = gpd.read_file(io.BytesIO(raw))
        if g.crs is None or g.crs.to_epsg() != 5070:
            g = g.to_crs(5070)
        return huc, g.union_all()
    except Exception as e:
        return huc, f"ERR:{e!r}"


def build_huc8_layer(cache, workers):
    if cache.exists():
        return gpd.read_parquet(cache)
    hucs = all_huc8s()
    print(f"fetching {len(hucs)} HUC8 polygons...", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_huc8_poly, h) for h in hucs]
        for i, f in enumerate(as_completed(futs), 1):
            huc, geom = f.result()
            if isinstance(geom, str):
                print(f"  poly fail {huc}: {geom}", flush=True)
                continue
            rows.append({"huc8": huc, "geometry": geom})
            if i % 200 == 0:
                print(f"  {i}/{len(hucs)} polys", flush=True)
    gdf = gpd.GeoDataFrame(rows, crs=5070)
    gdf.to_parquet(cache)
    print(f"cached {len(gdf)} HUC8 polygons -> {cache}", flush=True)
    return gdf


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
    df["x5070"], df["y5070"] = tr.transform(
        df["longitude"].values, df["latitude"].values
    )
    return gpd.GeoDataFrame(
        df.reset_index(drop=True),
        geometry=gpd.points_from_xy(df["x5070"], df["y5070"]),
        crs=5070,
    )


def sample_hand(huc, xy):
    branches, _ = s3_list(f"{huc}/branches/")
    pts = list(zip(xy[:, 0], xy[:, 1]))
    best = np.full(len(pts), np.inf)
    minx, miny, maxx, maxy = (
        xy[:, 0].min(),
        xy[:, 1].min(),
        xy[:, 0].max(),
        xy[:, 1].max(),
    )
    for b in branches:
        bid = b.rstrip("/").split("/")[-1]
        url = f"/vsicurl/{BASE}{huc}/branches/{bid}/rem_zeroed_masked_{bid}.tif"
        try:
            with rasterio.open(url) as ds:
                bb = ds.bounds
                if (
                    bb.right < minx
                    or bb.left > maxx
                    or bb.top < miny
                    or bb.bottom > maxy
                ):
                    continue
                vals = np.array([v[0] for v in ds.sample(pts)], dtype="float64")
        except Exception:
            continue
        ok = np.isfinite(vals) & (vals != HAND_NODATA) & (vals > -1.0) & (vals < 1e4)
        best = np.minimum(best, np.where(ok, vals, np.inf))
    best[~np.isfinite(best)] = np.nan
    return best


def process_shard(huc, sub, shard_dir):
    out = shard_dir / f"{huc}.parquet"
    if out.exists():
        return huc, "cached", len(sub)
    xy = sub[["x5070", "y5070"]].values
    sub = sub.copy()
    sub["hand_m"] = sample_hand(huc, xy)
    sub.drop(columns="geometry").to_parquet(out)
    return huc, "done", int(np.isfinite(sub["hand_m"]).sum())


def add_janssen(df):
    pts = list(zip(df["longitude"].values, df["latitude"].values))
    with rasterio.open(JANSSEN) as ds:
        nd = ds.nodata
        vals = np.array([v[0] for v in ds.sample(pts)], dtype="float64")
    vals[(vals == nd) | ~np.isfinite(vals)] = np.nan
    df["janssen_dtw"] = vals
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/data/ssd2/handily/conus/wte_gnn")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    shard_dir = out_dir / "hand_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    huc_layer = build_huc8_layer(out_dir / "huc8_polys.parquet", args.workers)
    wells = load_wells()
    print(f"loaded {len(wells):,} unconfined GWX wells with DTW", flush=True)

    joined = gpd.sjoin(wells, huc_layer, predicate="within", how="inner")
    joined = joined.drop(columns="index_right")
    print(
        f"{len(joined):,} wells fall inside FIM HUC8 coverage "
        f"({joined['huc8'].nunique()} HUC8s)",
        flush=True,
    )

    groups = list(joined.groupby("huc8"))
    print(f"sampling HAND across {len(groups)} HUC8s...", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, h, g, shard_dir): h for h, g in groups}
        for f in as_completed(futs):
            huc, status, n = f.result()
            done += 1
            if done % 50 == 0 or status != "cached":
                print(
                    f"  [{done}/{len(groups)}] {huc} {status} hand_ok={n}", flush=True
                )

    shards = [pd.read_parquet(p) for p in sorted(shard_dir.glob("*.parquet"))]
    table = pd.concat(shards, ignore_index=True)
    table = add_janssen(table)
    final = out_dir / "conus_wells_hand.parquet"
    table.to_parquet(final)
    hok = int(np.isfinite(table["hand_m"]).sum())
    print(
        f"\nwrote {final}: {len(table):,} wells, hand_ok={hok:,} "
        f"({100 * hok / len(table):.1f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
