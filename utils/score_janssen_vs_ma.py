"""Head-to-head: Janssen et al. (UBC) 500 m WTD product (V1/V2/V3) vs Ma WTD on
the NWIS-independent NM wells.

Janssen, Tootchi & Ameli, "A critical appraisal of water table depth estimation"
(J. Hydrology, draft) -- three XGBoost static-WTD simulations across the US/Canada:
  V1 = trained on real WTD obs only,
  V2 = real obs + a subset of satellite surface-water proxy obs,
  V3 = real obs + more proxy obs.
Adding surface-water proxy obs pins WTD toward 0 at persistently wet pixels, so
V2/V3 carry a progressively shallower bias than V1.

Janssen, like Ma, trained its real obs on USGS NWIS gwlevels, so the non-NWIS NM
wells in the consolidated validation layer are an independent test set for BOTH
products. We re-sample Ma here (rather than reusing the layer's stored ma_dtw_m,
which has a zero-fill artifact at out-of-grid points) so Ma and Janssen share
identical out-of-bounds + nodata masking -- a true paired comparison.

Both products are depth-to-water in meters, positive-down. Residual = pred - obs
(positive => product too deep; negative => too shallow). Headline accuracy uses
the unconfined water-table subset only; artesian_confined wells are excluded
(diagnostic only) per the project DTW rule. OSE driller wells (confinement
'unknown') are reported separately as screening-grade (single drill-time
measurement, no confinement typing).
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

WELLS = "/data/ssd2/handily/nm/regional/ma_nwis_statewide/nm_independent_validation_wells.geoparquet"
OUT = Path("/data/ssd2/handily/nm/regional/ma_nwis_statewide")
RASTERS = {
    "ma": "/nas/gwx/wtd_states/wtd_new_mexico.tif",
    "janssen_v1": "/nas/gwx/janssen/V1_140.tif",
    "janssen_v2": "/nas/gwx/janssen/V2_140.tif",
    "janssen_v3": "/nas/gwx/janssen/V3_140.tif",
}
PRODUCTS = list(RASTERS)


def sample(path, gdf):
    """Sample a WTD raster at well points with explicit out-of-bounds + nodata
    masking (-> NaN, never a fill value). Reprojects wells to the raster CRS."""
    with rasterio.open(path) as src:
        pts = gdf.to_crs(src.crs)
        xs = np.array([g.x for g in pts.geometry])
        ys = np.array([g.y for g in pts.geometry])
        vals = np.array(
            [v[0] for v in src.sample(zip(xs, ys), indexes=1)], dtype="float64"
        )
        nod, b = src.nodata, src.bounds
    vals[(xs < b.left) | (xs > b.right) | (ys < b.bottom) | (ys > b.top)] = np.nan
    if nod is not None and np.isfinite(nod):
        vals[vals == nod] = np.nan
    vals[(~np.isfinite(vals)) | (np.abs(vals) > 1e30)] = np.nan
    return vals


def metrics(obs, pred):
    valid = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[valid], pred[valid]
    if o.size == 0:
        return {"n": 0}
    r = p - o
    return {
        "n": int(o.size),
        "obs_med_m": round(float(np.median(o)), 1),
        "pred_med_m": round(float(np.median(p)), 1),
        "MAD_m": round(float(np.median(np.abs(r))), 2),
        "bias_m": round(float(np.mean(r)), 2),
        "med_resid_m": round(float(np.median(r)), 2),
        "rmse_m": round(float(np.sqrt(np.mean(r**2))), 1),
        "corr": round(float(np.corrcoef(o, p)[0, 1]), 3) if o.size > 2 else None,
    }


def main():
    g = gpd.read_parquet(WELLS)
    for name, path in RASTERS.items():
        g[name] = sample(path, g)

    zf = int((g["ma"] == 0).sum())
    print(
        f"stored ma==0: {int((g['ma_dtw_m'] == 0).sum())}  fresh ma==0: {zf}  "
        f"fresh ma NaN: {int(g['ma'].isna().sum())}  (re-sample drops the zero-fill artifact)"
    )

    subsets = {
        "unconfined_water_table": g[g["confinement"] == "unconfined"],
        "ose_driller_screening": g[g["confinement"] == "unknown"],
    }
    report = {}
    for sname, sub in subsets.items():
        obs = sub["dtw_m"].to_numpy()
        paired = np.isfinite(obs)
        for prod in PRODUCTS:
            paired &= np.isfinite(sub[prod].to_numpy())
        report[sname] = {
            "n_total": int(len(sub)),
            "n_paired_all_products": int(paired.sum()),
            "per_product": {p: metrics(obs, sub[p].to_numpy()) for p in PRODUCTS},
            "paired": {
                p: metrics(obs[paired], sub[p].to_numpy()[paired]) for p in PRODUCTS
            },
        }

    g.drop(columns="geometry").to_csv(OUT / "janssen_vs_ma_wells.csv", index=False)
    with open(OUT / "janssen_vs_ma_report.json", "w") as f:
        json.dump(report, f, indent=2)

    for sname in subsets:
        rec = report[sname]
        print(
            f"\n=== {sname}  (n_total={rec['n_total']}, "
            f"paired_all_products={rec['n_paired_all_products']}) ==="
        )
        print("paired head-to-head (identical n across products):")
        for prod in PRODUCTS:
            print(f"  {prod:11s} {rec['paired'][prod]}")
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
