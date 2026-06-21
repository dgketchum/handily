"""Go/no-go probe: does NOAA OWP HAND track observed DTW at CONUS scale?

Samples OWP HAND (CIROH public mirror, min over level-path branches) and the
Janssen CONUS WTD benchmark at GWX unconfined wells across a regime-spread of
HUC8s, then reports whether terrain HAND has predictive signal for the observed
water table BEFORE we invest in the full CONUS graph build.

This is a subset probe (capped wells per HUC8), not the production sampler.
"""

import argparse
import io
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.stats import spearmanr

# /vsicurl tuning for many small range reads against tiled COGs.
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
        x = urllib.request.urlopen(u, timeout=40).read().decode()
        r = ET.fromstring(x)
        for p in r.findall("s3:CommonPrefixes", NS):
            prefixes.append(p.find("s3:Prefix", NS).text)
        for k in r.findall("s3:Contents", NS):
            keys.append(k.find("s3:Key", NS).text)
        if r.find("s3:IsTruncated", NS).text == "true":
            token = r.find("s3:NextContinuationToken", NS).text
        else:
            break
    return prefixes, keys


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
        np.isfinite(df["mean_dtw"]) & (df["mean_dtw"] >= 0) & (df["mean_dtw"] < 300)
    ]
    df = df[np.isfinite(df["longitude"]) & np.isfinite(df["latitude"])]
    tr = Transformer.from_crs(4326, 5070, always_xy=True)
    x, y = tr.transform(df["longitude"].values, df["latitude"].values)
    df["x5070"], df["y5070"] = x, y
    return df.reset_index(drop=True)


def huc8_polygon(huc):
    """Download the per-HUC8 wbd8 clip gpkg and return its union geom in 5070."""
    url = BASE + f"{huc}/wbd8_clp.gpkg"
    raw = urllib.request.urlopen(url, timeout=60).read()
    g = gpd.read_file(io.BytesIO(raw))
    if g.crs is None or g.crs.to_epsg() != 5070:
        g = g.to_crs(5070)
    return g.union_all()


def sample_hand_for_huc(huc, wells_xy):
    """min HAND over branch rem rasters at the given (x,y) 5070 points."""
    branches, _ = s3_list(f"{huc}/branches/")
    pts = list(zip(wells_xy[:, 0], wells_xy[:, 1]))
    best = np.full(len(pts), np.inf)
    minx, miny = wells_xy[:, 0].min(), wells_xy[:, 1].min()
    maxx, maxy = wells_xy[:, 0].max(), wells_xy[:, 1].max()
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
                    continue  # branch raster does not overlap the well cluster
                vals = np.array([v[0] for v in ds.sample(pts)], dtype="float64")
        except Exception:
            continue
        ok = np.isfinite(vals) & (vals != HAND_NODATA) & (vals > -1.0) & (vals < 1e4)
        vals = np.where(ok, vals, np.inf)
        best = np.minimum(best, vals)
    best[~np.isfinite(best)] = np.nan
    return best


def process_huc(huc, wells, cap, seed):
    try:
        poly = huc8_polygon(huc)
    except Exception as e:
        return huc, None, f"poly_err:{e!r}"
    pts = gpd.GeoSeries(gpd.points_from_xy(wells["x5070"], wells["y5070"]), crs=5070)
    inside = pts.within(poly).values
    sub = wells.loc[inside].copy()
    if len(sub) == 0:
        return huc, None, "no_wells"
    if len(sub) > cap:
        sub = sub.sample(cap, random_state=seed)
    xy = sub[["x5070", "y5070"]].values
    sub["hand_m"] = sample_hand_for_huc(huc, xy)
    sub["huc8"] = huc
    return huc, sub, "ok"


def add_janssen(df):
    pts = list(
        zip(df["longitude"].values, df["latitude"].values)
    )  # Janssen is EPSG:4326
    with rasterio.open(JANSSEN) as ds:
        nd = ds.nodata
        vals = np.array([v[0] for v in ds.sample(pts)], dtype="float64")
    vals[(vals == nd) | ~np.isfinite(vals)] = np.nan
    df["janssen_dtw"] = vals
    return df


def robust_hand_calib_mad(hand, dtw):
    """Leave-it-simple HAND->DTW: median ratio + intercept via Theil-like fit.

    Reports the in-sample MAD of a monotone-ish linear HAND->DTW calibration as a
    floor for 'how much does raw HAND explain DTW'. Not a model, just a sniff.
    """
    m = np.isfinite(hand) & np.isfinite(dtw)
    h, d = hand[m], dtw[m]
    if len(h) < 20:
        return np.nan, np.nan, len(h)
    # robust slope/intercept: median of pairwise won't scale; use np.polyfit on
    # ranks-clipped data (clip extreme HAND so the fit is not tail-driven).
    hi = np.clip(h, 0, np.percentile(h, 99))
    A = np.vstack([hi, np.ones_like(hi)]).T
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    pred = A @ coef
    mad = float(np.median(np.abs(pred - d)))
    return mad, float(spearmanr(h, d).statistic), len(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-huc2", type=int, default=3)
    ap.add_argument("--cap", type=int, default=250, help="max wells sampled per HUC8")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--hucs-json",
        default="/tmp/claude-1002/-home-dgketchum-code-handily/"
        "bbbe8296-ffe5-4f10-a7cd-8e7cb124f528/scratchpad/ciroh_hucs.json",
    )
    ap.add_argument(
        "--out",
        default="/tmp/claude-1002/-home-dgketchum-code-handily/"
        "bbbe8296-ffe5-4f10-a7cd-8e7cb124f528/scratchpad/conus_hand_probe.parquet",
    )
    args = ap.parse_args()

    all_hucs = json.load(open(args.hucs_json))
    regions = [
        "01",
        "02",
        "03",
        "05",
        "07",
        "08",
        "10",
        "11",
        "12",
        "13",
        "15",
        "16",
        "17",
        "18",
    ]
    chosen = []
    for h2 in regions:
        pool = sorted(h for h in all_hucs if h.startswith(h2))
        if not pool:
            continue
        idx = np.linspace(0, len(pool) - 1, min(args.per_huc2, len(pool))).astype(int)
        chosen += [pool[i] for i in idx]
    print(f"probing {len(chosen)} HUC8s across {len(regions)} regions", flush=True)

    wells = load_wells()
    print(f"loaded {len(wells):,} unconfined GWX wells with DTW", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(process_huc, h, wells, args.cap, args.seed): h for h in chosen
        }
        for f in as_completed(futs):
            huc, sub, status = f.result()
            n = 0 if sub is None else int(np.isfinite(sub["hand_m"]).sum())
            print(f"  {huc} [{status}] hand_ok={n}", flush=True)
            if sub is not None and n > 0:
                results.append(sub)

    if not results:
        print("NO RESULTS")
        return
    out = pd.concat(results, ignore_index=True)
    out = add_janssen(out)
    out.to_parquet(args.out)

    print("\n==== HAND vs observed DTW (unconfined GWX) ====")
    hand, dtw = out["hand_m"].values, out["mean_dtw"].values
    jn = out["janssen_dtw"].values
    mad, rho, n = robust_hand_calib_mad(hand, dtw)
    print(f"OVERALL n={n}  HAND~DTW spearman={rho:.3f}  HAND-calib MAD={mad:.2f} m")
    mj = np.isfinite(jn) & np.isfinite(dtw)
    print(
        f"Janssen benchmark MAD vs DTW = {np.median(np.abs(jn[mj] - dtw[mj])):.2f} m (n={mj.sum()})"
    )

    print("\n-- by HUC2 region --")
    out["huc2"] = out["huc8"].str[:2]
    for h2, g in out.groupby("huc2"):
        mad, rho, n = robust_hand_calib_mad(g["hand_m"].values, g["mean_dtw"].values)
        gj = g.dropna(subset=["janssen_dtw", "mean_dtw"])
        jmad = (
            np.median(np.abs(gj["janssen_dtw"] - gj["mean_dtw"])) if len(gj) else np.nan
        )
        print(
            f"  HUC2 {h2}: n={n:5d}  spearman={rho:+.3f}  HANDcalibMAD={mad:6.2f}  "
            f"JanssenMAD={jmad:6.2f}  medDTW={np.median(g['mean_dtw']):.1f}"
        )

    print("\n-- by observed DTW band --")
    bands = [(0, 2), (2, 5), (5, 10), (10, 30), (30, 300)]
    for lo, hi in bands:
        g = out[(out["mean_dtw"] >= lo) & (out["mean_dtw"] < hi)]
        if len(g) < 20:
            continue
        rho = spearmanr(g["hand_m"], g["mean_dtw"], nan_policy="omit").statistic
        print(
            f"  {lo:>3}-{hi:<3}m: n={len(g):5d}  spearman(HAND,DTW)={rho:+.3f}  "
            f"medHAND={np.nanmedian(g['hand_m']):.1f}  medDTW={np.median(g['mean_dtw']):.1f}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
