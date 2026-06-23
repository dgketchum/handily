"""Consolidate every non-NWIS NM water-table observation we have into one
validation layer, score Ma against it, and write it ready for FAC-prior scoring.

POPULATION (name it; do NOT conflate with score_ma_vs_nm_gwx.py): confinement
here is HAND-ASSIGNED per source/basin (not the GWX v2 classifier). The
water-table headline is the confinement=='unconfined' subset (ABQ-basin study +
SensorThings); OSE driller logs are held out as confinement='unknown'
screening-grade. score_ma_vs_nm_gwx.py instead uses the GWX national classifier
and counts nm_ose AS unconfined (a ~34k-well non-NWIS panel) -- a different
scientific claim. Never compare the two n's / MADs without naming which
population produced them.

Sources unioned (all non-NWIS, DTW positive-down in meters):
  - ose_driller     : NM OSE POD driller logs, depth_wate (genuine drill-time DTW;
                      static_lev is a redundant dup, dropped). 100k+ wells, no
                      confinement typing -> confinement='unknown' (screening).
  - nm_sensorthings : NM Water Data telemetry (BernCo/CABQ/OSE-Roswell/PVACD/EBID/
                      SanAcacia), per-site median DTW. Read from the PRODUCT-
                      AGNOSTIC reduction (nm_sensorthings_sites.csv) so the
                      inventory is not conditioned on Ma coverage.
                      ABQ-basin = unconfined; Roswell basin = artesian (confined).
  - eastern_abq_*   : USGS ScienceBase deep ABQ-basin water table (KAFB/SNL/CABQ/
                      AECOM source rows only), unconfined.
  - sfgas_2016      : Santa Fe Group deep production-zone wells, unconfined.

DE-DUPLICATION: rows are collapsed to unique 6-decimal lon/lat sites (median
DTW) before sampling, so cross-source re-reports (e.g. a well in both the 2008
and 2016 ABQ tables, or STA + study) do not over-count support.

INDEPENDENCE (softened): these wells are non-NWIS, but source != nwis does not
prove independence from Ma -- the Ma 2026 product also trained on Fan et al.
wells, Jasechko CA/TX data, and ~20k stream-dummy cells. Independence is
defensible-but-unproven for Ma; stronger for Janssen (US real-obs = USGS
gwlevels). Treat as "non-NWIS", not "independent".

Two national WTD baselines are sampled at every site and stored as columns:
  ma_dtw_m / resid_m              : Ma et al. WTD (resid = ma - observed)
  janssen_v1_dtw_m / janssen_v1_resid_m : Janssen et al. (UBC) V1 (real-obs-only
      XGBoost; the deep-basin-best of V1/V2/V3, see score_janssen_vs_ma.py).
Residual = predicted - observed (positive => product too deep). Output:
  nm_independent_validation_wells.{geoparquet,csv} + _summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

FT_TO_M = 0.3048
MA = "/nas/gwx/wtd_states/wtd_new_mexico.tif"
JANSSEN_V1 = "/nas/gwx/janssen/V1_140.tif"
BASELINES = ("ma_dtw_m", "janssen_v1_dtw_m")
STUD = Path("/nas/gwx/studies/nm_independent_wells")
STA_CSV = Path(
    "/data/ssd2/handily/nm/regional/ma_nwis_statewide/nm_sensorthings_sites.csv"
)
OUT = Path("/data/ssd2/handily/nm/regional/ma_nwis_statewide")
NM = dict(lon=(-109.1, -102.9), lat=(31.2, 37.1))

UNCONF = {"BernCo", "CABQ"}  # Albuquerque basin fill
ARTESIAN = {"OSE-Roswell", "PVACD", "CityOfRoswell"}  # Roswell Artesian Basin
VALLEY = {
    "EBID",
    "SanAcaciaReach",
    "EBWPC",
}  # Rio Grande / Mesilla valley, shallow unconfined


def sample_raster(path, gdf):
    """Sample a WTD raster at well points, reprojecting to its CRS; out-of-bounds
    and nodata (either sign of ±3.4e38) -> NaN, never a fill value."""
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


def metrics(df, pred_col="ma_dtw_m"):
    o, p = df["dtw_m"].to_numpy(), df[pred_col].to_numpy()
    valid = np.isfinite(o) & np.isfinite(p)
    o, p = o[valid], p[valid]
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


def ose_driller():
    o = pd.read_parquet(
        "/nas/gwx/nm_ose/pod.parquet",
        columns=["_longitude", "_latitude", "depth_wate", "finish_dat"],
    )
    lon = pd.to_numeric(o["_longitude"], errors="coerce")
    lat = pd.to_numeric(o["_latitude"], errors="coerce")
    dtw = pd.to_numeric(o["depth_wate"], errors="coerce")
    m = (
        lon.between(*NM["lon"])
        & lat.between(*NM["lat"])
        & dtw.between(0, 3000, inclusive="neither")
    )
    o = pd.DataFrame(
        {
            "lon": lon[m],
            "lat": lat[m],
            "dtw_m": dtw[m] * FT_TO_M,
            "date": o["finish_dat"][m],
        }
    )
    o["cell"] = list(zip(o["lon"].round(4), o["lat"].round(4)))
    w = (
        o.groupby("cell")
        .agg(
            lon=("lon", "first"),
            lat=("lat", "first"),
            dtw_m=("dtw_m", "median"),
            n_obs=("dtw_m", "size"),
            date=("date", "max"),
        )
        .reset_index(drop=True)
    )
    w["source"], w["agency"], w["confinement"] = "ose_driller", "NM_OSE", "unknown"
    return w


def sensorthings():
    s = pd.read_csv(STA_CSV)
    s = s.rename(columns={"prop_agency": "agency", "last_obs": "date"})
    s = s[s["dtw_m"].between(0, 600)].copy()
    s["source"] = "nm_sensorthings"
    s["confinement"] = np.where(
        s["agency"].isin(UNCONF | VALLEY),
        "unconfined",
        np.where(s["agency"].isin(ARTESIAN), "artesian_confined", "unknown"),
    )
    return s[
        ["lon", "lat", "dtw_m", "n_obs", "date", "source", "agency", "confinement"]
    ]


def study():
    rows = []
    for yr in ("2016", "2008"):
        d = pd.read_csv(
            STUD / f"eastern_albuquerque_wt/Welldata{yr}.txt", sep="\t", engine="python"
        )
        d = d[~d["Source"].isin(["NWIS"]) & d["Source"].notna()]
        rows.append(
            pd.DataFrame(
                {
                    "lon": pd.to_numeric(d["Long"], errors="coerce"),
                    "lat": pd.to_numeric(d["Lat"], errors="coerce"),
                    "dtw_m": pd.to_numeric(d["DTW_ft"], errors="coerce") * FT_TO_M,
                    "source": f"eastern_abq_{yr}",
                    "agency": d["Source"].astype(str),
                    "confinement": "unconfined",
                }
            )
        )
    sf = pd.read_csv(
        STUD / "albuquerque_sfgas_2016/Wells_and_wls_WY2016_SFGAS.txt",
        sep="\t",
        engine="python",
    )
    sf = sf[~sf["SOURCE_WL"].str.contains("USGS", na=False)]
    g = gpd.GeoDataFrame(
        sf,
        geometry=gpd.points_from_xy(
            pd.to_numeric(sf["X_N83SPCFT"], errors="coerce"),
            pd.to_numeric(sf["Y_N83SPCFT"], errors="coerce"),
        ),
        crs="EPSG:2903",
    ).to_crs(4326)
    rows.append(
        pd.DataFrame(
            {
                "lon": g.geometry.x,
                "lat": g.geometry.y,
                "dtw_m": pd.to_numeric(sf["DTW_FT"], errors="coerce") * FT_TO_M,
                "source": "sfgas_2016",
                "agency": "SFGAS",
                "confinement": "unconfined",
            }
        )
    )
    out = pd.concat(rows, ignore_index=True)
    return out[out["dtw_m"].between(0, 600)]


def dedupe_sites(df):
    """Collapse to unique 6-decimal lon/lat sites (median DTW) so cross-source
    re-reports (2008 vs 2016 ABQ tables, STA vs study) don't over-count support.
    Confinement is resolved by priority -- a typed label (artesian_confined,
    then unconfined) wins over 'unknown'; cross-confinement coincidences are
    logged (verified: 1 such site in the current NM layer)."""
    df = df.copy()
    df["n_obs"] = pd.to_numeric(df.get("n_obs"), errors="coerce").fillna(1)
    df["_k"] = list(zip(df["lon"].round(6), df["lat"].round(6)))
    rank = {"artesian_confined": 0, "unconfined": 1, "unknown": 2}
    df["_r"] = df["confinement"].map(rank).fillna(3)
    n_conflict = int((df.groupby("_k")["confinement"].nunique() > 1).sum())
    df = df.sort_values("_r")  # highest-priority confinement first within a site
    agg = (
        df.groupby("_k", sort=False)
        .agg(
            lon=("lon", "first"),
            lat=("lat", "first"),
            dtw_m=("dtw_m", "median"),
            n_obs=("n_obs", "sum"),
            date=("date", "max"),
            source=("source", "first"),
            agency=("agency", "first"),
            confinement=("confinement", "first"),
            n_sites_merged=("dtw_m", "size"),
        )
        .reset_index(drop=True)
    )
    return agg, n_conflict


def main():
    df = pd.concat([ose_driller(), sensorthings(), study()], ignore_index=True)
    df = df[df["lon"].between(*NM["lon"]) & df["lat"].between(*NM["lat"])].copy()
    df["date"] = df["date"].astype(
        "string"
    )  # OSE int + STA ISO str + study <NA> -> uniform
    n_rows_pre_dedup = len(df)
    df, n_conflict = dedupe_sites(df)
    print(
        f"sites: {n_rows_pre_dedup} rows -> {len(df)} unique 6-dp lon/lat sites "
        f"({n_conflict} cross-confinement coincidence(s) resolved by priority)"
    )
    g = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )
    g["ma_dtw_m"] = sample_raster(MA, g)
    g["janssen_v1_dtw_m"] = sample_raster(JANSSEN_V1, g)
    # Membership = wells with a valid Ma value (its tile is the limiting footprint).
    g = g[g["ma_dtw_m"].notna()].copy()
    g["resid_m"] = g["ma_dtw_m"] - g["dtw_m"]
    g["janssen_v1_resid_m"] = g["janssen_v1_dtw_m"] - g["dtw_m"]
    n_jv1_missing = int(g["janssen_v1_dtw_m"].isna().sum())
    print(f"Ma-valid wells missing Janssen V1 coverage: {n_jv1_missing} / {len(g)}")

    g.to_parquet(OUT / "nm_independent_validation_wells.geoparquet")
    g.drop(columns="geometry").to_csv(
        OUT / "nm_independent_validation_wells.csv", index=False
    )

    def by(subset):
        return {b: metrics(subset, b) for b in BASELINES}

    summary = {
        "population": (
            "consolidated NM non-NWIS layer; confinement HAND-ASSIGNED by "
            "source/basin (NOT the GWX v2 classifier); water-table headline = "
            "confinement=='unconfined'; OSE driller logs held out as 'unknown' "
            "screening. Distinct from score_ma_vs_nm_gwx.py (GWX classifier, "
            "nm_ose counted unconfined)."
        ),
        "independence_note": (
            "non-NWIS, not proven-independent of Ma (Ma 2026 also trained on Fan "
            "et al. + Jasechko CA/TX + stream-dummy cells); stronger for Janssen."
        ),
        "dedup": {
            "rows_pre_dedup": n_rows_pre_dedup,
            "unique_sites_post_dedup": int(len(df)),
            "cross_confinement_resolved": n_conflict,
            "sites_in_ma_footprint": int(len(g)),
        },
        "n_total": len(g),
        "baselines": list(BASELINES),
        "all_non_nwis": by(g),
        "water_table_only": by(g[g["confinement"] == "unconfined"]),
        "by_source": {s: by(sub) for s, sub in g.groupby("source")},
        "by_confinement": {c: by(sub) for c, sub in g.groupby("confinement")},
    }
    with open(OUT / "nm_independent_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
