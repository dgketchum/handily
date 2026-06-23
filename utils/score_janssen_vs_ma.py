"""Head-to-head: Janssen et al. (UBC) 500 m WTD product (V1/V2/V3) vs Ma WTD on
the consolidated NM non-NWIS validation layer.

POPULATION (name it; do NOT conflate with score_ma_vs_nm_gwx.py): this scores
the layer built by build_nm_validation_layer.py, whose confinement is
HAND-ASSIGNED per source/basin (NOT the GWX v2 classifier). The water-table
headline here is the confinement=='unconfined' subset (ABQ-basin study + ABQ
SensorThings), de-duplicated to unique 6-decimal sites; OSE driller logs are
held out as confinement=='unknown' screening-grade. score_ma_vs_nm_gwx.py uses
the GWX national classifier and counts nm_ose AS unconfined (a ~34k-well
non-NWIS panel) -- a different scientific claim. Never compare the two n's /
MADs without naming which population produced them.

Janssen, Tootchi & Ameli, "A critical appraisal of water table depth estimation"
(J. Hydrology, draft) -- three XGBoost static-WTD simulations across the US/Canada:
  V1 = trained on real WTD obs only,
  V2 = real obs + a subset of satellite surface-water proxy obs,
  V3 = real obs + more proxy obs.
Adding surface-water proxy obs pins WTD toward 0 at persistently wet pixels, so
V2/V3 carry a progressively shallower bias than V1.

INDEPENDENCE (softened): these wells are non-NWIS, but source != nwis does not
prove independence from Ma -- the Ma 2026 product also trained on Fan et al.
wells, Jasechko CA/TX data, and ~20k stream-dummy cells, so independence is
defensible-but-unproven for Ma. It is stronger for Janssen, whose US real-obs
source is USGS gwlevels. Treat as "non-NWIS", not "independent".

Ma==0 NOTE: ~2800 NM points sample Ma==0 (water table at land surface). The Ma
nodata sentinel is +3.4e38, so these zeros are GENUINE surface/stream WTD=0
clamps in the product, NOT an out-of-bounds fill -- they are retained, not
dropped. (Earlier code claimed re-sampling "drops a zero-fill artifact"; that
was wrong. Re-sampling here only guarantees Ma and Janssen share identical
out-of-bounds + nodata->NaN masking, for a true paired comparison.)

PAIRED MASK (#6 fix): each Janssen product is scored against Ma on its OWN
pairwise common footprint finite(obs) & finite(ma) & finite(Pj), so V2/V3 nodata
can never shrink the V1-vs-Ma pair. An all-products-common footprint is reported
separately.

Both products are depth-to-water in meters, positive-down. Residual = pred - obs
(positive => product too deep; negative => too shallow). Per the project rule the
report carries a full panel: depth bands, by-source, shallow recall/precision,
common-footprint coverage, and native-cell aggregation (a 500 m product is not
credited with sub-cell "independent" points).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwx_wells import DEPTH_BANDS  # noqa: E402

DEFAULT_WELLS = "/data/ssd2/handily/nm/regional/ma_nwis_statewide/nm_independent_validation_wells.geoparquet"
DEFAULT_OUT = "/data/ssd2/handily/nm/regional/ma_nwis_statewide"
DEFAULT_PRODUCTS = [
    "ma=/nas/gwx/wtd_states/wtd_new_mexico.tif",
    "janssen_v1=/nas/gwx/janssen/V1_140.tif",
    "janssen_v2=/nas/gwx/janssen/V2_140.tif",
    "janssen_v3=/nas/gwx/janssen/V3_140.tif",
]
SHALLOW_LEVELS = (2.0, 5.0, 10.0)  # m
# Headline / screening subsets, named so the population is never ambiguous.
SUBSETS = {
    "unconfined_water_table": "confinement == 'unconfined' (hand-assigned headline)",
    "ose_driller_screening": "confinement == 'unknown' (OSE driller logs, screening)",
}


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
    """Full per-pair panel (meters): central error, bias vs median residual,
    RMSE, correlation, and shallow recall/precision at <2/<5/<10 m."""
    obs = np.asarray(obs, dtype="float64")
    pred = np.asarray(pred, dtype="float64")
    valid = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[valid], pred[valid]
    if o.size == 0:
        return {"n": 0}
    r = p - o
    out = {
        "n": int(o.size),
        "obs_med_m": round(float(np.median(o)), 1),
        "pred_med_m": round(float(np.median(p)), 1),
        "MAD_m": round(float(np.median(np.abs(r))), 2),
        "bias_m": round(float(np.mean(r)), 2),
        "med_resid_m": round(float(np.median(r)), 2),
        "rmse_m": round(float(np.sqrt(np.mean(r**2))), 1),
        "corr": round(float(np.corrcoef(o, p)[0, 1]), 3) if o.size > 2 else None,
    }
    for lvl in SHALLOW_LEVELS:
        os_, ps_ = o < lvl, p < lvl
        tp = int((os_ & ps_).sum())
        out[f"recall_<{int(lvl)}m"] = (
            round(tp / int(os_.sum()), 3) if os_.sum() else None
        )
        out[f"prec_<{int(lvl)}m"] = round(tp / int(ps_.sum()), 3) if ps_.sum() else None
    return out


def depth_bands(obs, pred):
    """metrics() stratified by observed-depth band (project rule)."""
    obs = np.asarray(obs, dtype="float64")
    pred = np.asarray(pred, dtype="float64")
    out = {}
    for lo, hi in DEPTH_BANDS:
        m = (obs >= lo) & (obs < hi)
        label = f"{lo}-{hi if hi < 1e9 else 'inf'}m"
        out[label] = metrics(obs[m], pred[m])
    return out


def cell_aggregated(path, lon, lat, obs, pred):
    """Aggregate obs+pred to the product's NATIVE cell (median per cell), then
    score -- so a 500 m product is not credited with multiple sub-cell points
    as if they were independent samples. Returns metrics + n_cells."""
    obs = np.asarray(obs, dtype="float64")
    pred = np.asarray(pred, dtype="float64")
    with rasterio.open(path) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = t.transform(np.asarray(lon), np.asarray(lat))
        cols, rows = (~src.transform) * (np.asarray(xs), np.asarray(ys))
    cid = np.floor(cols).astype(np.int64) * 1_000_000 + np.floor(rows).astype(np.int64)
    m = np.isfinite(obs) & np.isfinite(pred)
    if not m.any():
        return {"n": 0, "n_cells": 0}
    agg = (
        pd.DataFrame({"cid": cid[m], "o": obs[m], "p": pred[m]}).groupby("cid").median()
    )
    st = metrics(agg["o"].to_numpy(), agg["p"].to_numpy())
    st["n_cells"] = int(len(agg))
    return st


def raster_meta(path):
    with rasterio.open(path) as src:
        res = src.res
        geographic = bool(src.crs.is_geographic) if src.crs else None
        meta = {
            "path": path,
            "crs": src.crs.to_string() if src.crs else None,
            "native_res": [round(float(res[0]), 6), round(float(res[1]), 6)],
            "native_res_units": "degrees" if geographic else "meters",
            "nodata": float(src.nodata) if src.nodata is not None else None,
            "shape_rows_cols": [int(src.height), int(src.width)],
        }
        if geographic:
            meta["approx_res_m"] = round(float(res[0]) * 111_320, 1)
    return meta


def file_provenance(path):
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": p.stat().st_size,
        "mtime_utc": datetime.fromtimestamp(
            p.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": h.hexdigest(),
    }


def git_provenance():
    repo = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wells", default=DEFAULT_WELLS)
    ap.add_argument(
        "--product",
        action="append",
        default=None,
        help="LABEL=path (repeatable); first product is the reference unless "
        "--reference is given. Default: Ma + Janssen V1/V2/V3.",
    )
    ap.add_argument("--reference", default="ma", help="Reference product label.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--report-name", default="janssen_vs_ma_report.json")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = args.product or DEFAULT_PRODUCTS
    products = {}
    for s in specs:
        label, path = s.split("=", 1)
        products[label] = path
    ref = args.reference
    if ref not in products:
        raise SystemExit(f"reference {ref!r} not among products {list(products)}")
    others = [p for p in products if p != ref]

    g = gpd.read_parquet(args.wells)
    for label, path in products.items():
        g[label] = sample(path, g)

    # Genuine surface clamps (NOT a zero-fill artifact); reported, retained.
    zero_clamp = {label: int((g[label] == 0).sum()) for label in products}

    subsets = {
        "unconfined_water_table": g[g["confinement"] == "unconfined"].copy(),
        "ose_driller_screening": g[g["confinement"] == "unknown"].copy(),
    }

    report = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_provenance(),
            "wells": file_provenance(args.wells),
            "products": {label: raster_meta(path) for label, path in products.items()},
            "reference_product": ref,
            "paired_mask": (
                "head-to-head metrics use the PAIRWISE common footprint "
                "finite(obs) & finite(reference) & finite(product); V2/V3 nodata "
                "never shrinks the V1-vs-reference pair. 'all_products_common' is "
                "the intersection of every product and is reported separately."
            ),
            "population": (
                "consolidated NM non-NWIS layer (build_nm_validation_layer.py); "
                "confinement HAND-ASSIGNED by source/basin (NOT the GWX v2 "
                "classifier); de-duplicated to unique 6-decimal sites. Headline = "
                "confinement=='unconfined'; OSE driller logs = 'unknown' screening. "
                "DISTINCT from score_ma_vs_nm_gwx.py (GWX classifier, nm_ose "
                "counted unconfined, ~34k-well panel)."
            ),
            "independence": (
                "non-NWIS; not proven-independent of Ma (Ma 2026 also trained on "
                "Fan et al. + Jasechko CA/TX + ~20k stream-dummy cells). Stronger "
                "for Janssen (US real-obs = USGS gwlevels)."
            ),
            "zero_clamp_note": (
                "product==0 counts are genuine surface/stream WTD=0 clamps "
                "(nodata is +3.4e38, not 0); retained, not dropped."
            ),
            "product_zero_clamp_count": zero_clamp,
            "shallow_levels_m": list(SHALLOW_LEVELS),
            "depth_bands_m": [
                [lo, (hi if hi < 1e9 else None)] for lo, hi in DEPTH_BANDS
            ],
            "subset_definitions": SUBSETS,
            "residual_convention": "pred - obs (positive => product too deep)",
        },
        "subsets": {},
    }

    for sname, sub in subsets.items():
        obs = sub["dtw_m"].to_numpy()
        s_lon, s_lat = sub["lon"].to_numpy(), sub["lat"].to_numpy()
        finite = {label: np.isfinite(sub[label].to_numpy()) for label in products}

        # All-products-common footprint (reported separately, not the headline).
        all_common = np.isfinite(obs)
        for label in products:
            all_common &= finite[label]

        # Per-product pairwise-vs-reference head-to-head (the #6 fix).
        head = {}
        for label in others:
            pair = np.isfinite(obs) & finite[ref] & finite[label]
            o_p = obs[pair]
            head[label] = {
                "footprint": f"finite(obs) & finite({ref}) & finite({label})",
                "n_common": int(pair.sum()),
                ref: metrics(o_p, sub[ref].to_numpy()[pair]),
                label: metrics(o_p, sub[label].to_numpy()[pair]),
                "depth_bands": {
                    ref: depth_bands(o_p, sub[ref].to_numpy()[pair]),
                    label: depth_bands(o_p, sub[label].to_numpy()[pair]),
                },
                "cell_aggregated": {
                    ref: cell_aggregated(
                        products[ref],
                        s_lon[pair],
                        s_lat[pair],
                        o_p,
                        sub[ref].to_numpy()[pair],
                    ),
                    label: cell_aggregated(
                        products[label],
                        s_lon[pair],
                        s_lat[pair],
                        o_p,
                        sub[label].to_numpy()[pair],
                    ),
                },
            }

        # By-source on each product's own finite footprint.
        by_src = {}
        for src_name, ssub in sub.groupby("source"):
            so = ssub["dtw_m"].to_numpy()
            by_src[str(src_name)] = {
                label: metrics(so, ssub[label].to_numpy()) for label in products
            }

        report["subsets"][sname] = {
            "definition": SUBSETS[sname],
            "n_total": int(len(sub)),
            "coverage_n_finite": {
                label: int((np.isfinite(obs) & finite[label]).sum())
                for label in products
            },
            "native_cells_occupied": {
                label: cell_aggregated(
                    products[label], s_lon, s_lat, obs, sub[label].to_numpy()
                ).get("n_cells", 0)
                for label in products
            },
            "all_products_common": {
                "n": int(all_common.sum()),
                "per_product": {
                    label: metrics(obs[all_common], sub[label].to_numpy()[all_common])
                    for label in products
                },
            },
            "head_to_head_vs_reference": head,
            "by_source": by_src,
        }

    g.drop(columns="geometry").to_csv(out / "janssen_vs_ma_wells.csv", index=False)
    with open(out / args.report_name, "w") as f:
        json.dump(report, f, indent=2)

    # Console summary: pairwise head-to-head on each product's own common footprint.
    for sname in subsets:
        rec = report["subsets"][sname]
        print(f"\n=== {sname} (n_total={rec['n_total']}) — {SUBSETS[sname]} ===")
        print(f"  coverage (finite n): {rec['coverage_n_finite']}")
        print(f"  native cells occupied: {rec['native_cells_occupied']}")
        for label, h in rec["head_to_head_vs_reference"].items():
            print(f"  {label} vs {ref} (pairwise n={h['n_common']}):")
            for who in (ref, label):
                m = h[who]
                print(
                    f"    {who:11s} MAD={m.get('MAD_m')} bias={m.get('bias_m')} "
                    f"med={m.get('med_resid_m')} RMSE={m.get('rmse_m')} "
                    f"r={m.get('corr')}  cell-agg MAD={h['cell_aggregated'][who].get('MAD_m')} "
                    f"(n_cells={h['cell_aggregated'][who].get('n_cells')})"
                )


if __name__ == "__main__":
    main()
