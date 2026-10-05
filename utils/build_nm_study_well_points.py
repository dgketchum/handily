"""Build the New Mexico study-well point set for the independent check of a
handily arm (``predict_gnn_at_points.py`` input + scoring attributes).

Five on-disk study sets, none of them OSE points of diversion, so none of them
can be a label or an inference-time source of the ``_oselbl`` arm:

* ``urgb_fas`` -- USGS SIR 2021-5035 Upper Rio Grande FAS median water-level
  altitudes, 2,699 wells x 5-year groups; the latest group per well is kept.
  USGS site numbers, so NOT independent of NWIS (and possibly of the CONUS
  bundle; ``dist_train_km`` tells).
* ``eastern_abq_2016`` -- Eastern Albuquerque 2016 water-table map wells,
  non-NWIS sources only (KAFB / SNL / CABQ / AECOM).
* ``sfgas_2016`` -- Santa Fe Group 2016 contour wells, non-USGS sources only.
* ``rgtihm_ibwc`` -- RGTIHM head observations from the International Boundary
  and Water Commission (Mesilla), per-well median DTW.
* ``mesilla_2010`` -- TAAP 2010 median groundwater ELEVATIONS (ft NAVD88); the
  DTW is derived downstream as ``z_surf_m - wt_elev_m`` from the point path's
  100 m land surface (``dtw_from_dem`` flag). Mexico wells are independent of
  NWIS; most fall outside the HUC8 footprint.

Attributes added: EPSG:5070 coordinates, WBD HUC8 and footprint membership
(the 72 NM render basins), distance to the nearest bundle training well and
nearest NWIS well, Ma DTW, and the regional-model DTW surfaces sampled at the
well (URGB 2015 DTW 200 m; MRGB McAda-Barroll predevelopment steady state
500 m; the Mesilla and Albuquerque 250 m steady-state model DTW rasters). All
depths are metres, positive down.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_admission_split import WBD_HU8_PARQUET, footprint_huc8s, log  # noqa: E402
from sample_benchmark_rasters import MA_DIR, sample_ma_tiles  # noqa: E402

FT_TO_M = 0.3048
STUDIES = Path("/nas/gwx/studies")
INDEP = STUDIES / "nm_independent_wells"
URGB_GPKG = (
    STUDIES / "analysis_ready/urgb_fas_wl_altitude/urgb_fas_median_wl_obs_5070.gpkg"
)
MODEL_RASTERS = {
    "urgb_dtw_2015_m": STUDIES
    / "analysis_ready/urgb_fas_wl_altitude/urgb_dtw_2015_m_5070.tif",
    "mrgb_predev_ss_dtw_m": STUDIES
    / "analysis_ready/mrgb_mcada_modflow/mrgb_steadystate_dtw_m_5070.tif",
    "mesilla_model_ss_dtw_m": Path(
        "/data/ssd2/handily/nm/regional/mesilla/modflow_ss_dtw_5070.tif"
    ),
    "abq_model_ss_dtw_m": Path(
        "/data/ssd2/handily/nm/regional/rio_grande_albuquerque/modflow_ss_dtw_5070.tif"
    ),
}
OUT_DIR = Path("/data/ssd2/handily/nm/regional/wells")
COLS = [
    "study",
    "site_id",
    "source_org",
    "year",
    "obs_dtw_m",
    "wt_elev_m",
    "well_depth_m",
    "basin_name",
    "aqfr_cd",
    "independent_of_nwis",
    "geometry",
]


def _frame(df: pd.DataFrame, study: str, geometry, crs) -> gpd.GeoDataFrame:
    g = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)
    g["study"] = study
    for c in COLS:
        if c not in g.columns:
            g[c] = np.nan
    g = g[COLS]
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    return g.to_crs(5070)


def urgb_fas() -> gpd.GeoDataFrame:
    g = gpd.read_file(URGB_GPKG)
    g = g[np.isfinite(g["dtw_m"])]
    g = g.sort_values(["SITE_NO", "YearGrp"]).groupby("SITE_NO").tail(1)
    df = pd.DataFrame(
        {
            "site_id": g["SITE_NO"].astype(str),
            "source_org": "USGS SIR 2021-5035",
            "year": g["YearGrp"].astype(float),
            "obs_dtw_m": g["dtw_m"].astype(float),
            "wt_elev_m": g["wla_median_m"].astype(float),
            "well_depth_m": pd.to_numeric(g["WELL_DEPTH"], errors="coerce") * FT_TO_M,
            "basin_name": g["AlluvialBasin"].astype(str),
            "aqfr_cd": g["AQFR_CD"].astype(str),
            "independent_of_nwis": False,
        }
    )
    return _frame(df, "urgb_fas", g.geometry.values, g.crs)


def eastern_abq_2016() -> gpd.GeoDataFrame:
    df = pd.read_csv(
        INDEP / "eastern_albuquerque_wt/Welldata2016.txt", sep="\t", engine="python"
    )
    df = df[df["Source"].notna() & (df["Source"] != "NWIS")].copy()
    out = pd.DataFrame(
        {
            "site_id": df["SiteID"].astype(str),
            "source_org": df["Source"].astype(str),
            "year": 2016.0,
            "obs_dtw_m": pd.to_numeric(df["DTW_ft"], errors="coerce") * FT_TO_M,
            "wt_elev_m": pd.to_numeric(df["Water_Elev"], errors="coerce") * FT_TO_M,
            "well_depth_m": pd.to_numeric(df["WellDep_ft"], errors="coerce") * FT_TO_M,
            "basin_name": "Middle Rio Grande",
            "independent_of_nwis": True,
        }
    )
    geom = gpd.points_from_xy(
        pd.to_numeric(df["Long"], errors="coerce"),
        pd.to_numeric(df["Lat"], errors="coerce"),
    )
    return _frame(out, "eastern_abq_2016", geom, "EPSG:4326")


def sfgas_2016() -> gpd.GeoDataFrame:
    df = pd.read_csv(
        INDEP / "albuquerque_sfgas_2016/Wells_and_wls_WY2016_SFGAS.txt",
        sep="\t",
        engine="python",
    )
    df = df[~df["SOURCE_WL"].str.contains("USGS", na=False)].copy()
    out = pd.DataFrame(
        {
            "site_id": df["SITE_ID"].astype(str),
            "source_org": df["SOURCE_WL"].astype(str),
            "year": 2016.0,
            "obs_dtw_m": pd.to_numeric(df["DTW_FT"], errors="coerce") * FT_TO_M,
            "wt_elev_m": pd.to_numeric(df["WL_ELV_88"], errors="coerce") * FT_TO_M,
            "well_depth_m": pd.to_numeric(df["WELL_DEPTH"], errors="coerce") * FT_TO_M,
            "basin_name": "Middle Rio Grande",
            "aqfr_cd": df["AQFER_CD"].astype(str),
            "independent_of_nwis": True,
        }
    )
    geom = gpd.points_from_xy(
        pd.to_numeric(df["X_N83SPCFT"], errors="coerce"),
        pd.to_numeric(df["Y_N83SPCFT"], errors="coerce"),
    )
    return _frame(out, "sfgas_2016", geom, "EPSG:2903")


def rgtihm_ibwc() -> gpd.GeoDataFrame:
    hob = pd.read_csv(INDEP / "rgtihm_hob/RGTIHM_HOB.csv")
    ib = hob[hob["WL_source"].str.contains("International Boundary", na=False)].copy()
    ib["yr"] = pd.to_datetime(ib["Date"], errors="coerce").dt.year
    per = (
        ib.groupby("Site_ID")
        .agg(
            dtw_ft=("DTW_ft", "median"),
            wl_ft=("WL_elev_ft", "median"),
            yr=("yr", "median"),
        )
        .reset_index()
    )
    loc = gpd.read_file(f"zip://{INDEP}/rgtihm_hob/RGTIHM_HOB_Locations.zip")
    m = loc.merge(per, on="Site_ID", how="inner")
    if len(m) != len(per):
        raise SystemExit(f"RGTIHM: {len(per)} IBWC wells, {len(m)} matched locations")
    out = pd.DataFrame(
        {
            "site_id": m["Site_ID"].astype(str),
            "source_org": "IBWC (RGTIHM HOB)",
            "year": m["yr"].astype(float),
            "obs_dtw_m": m["dtw_ft"].astype(float) * FT_TO_M,
            "wt_elev_m": m["wl_ft"].astype(float) * FT_TO_M,
            "basin_name": "Mesilla and Conejos-Medanos",
            "independent_of_nwis": True,
        }
    )
    return _frame(out, "rgtihm_ibwc", m.geometry.values, loc.crs)


def mesilla_2010() -> gpd.GeoDataFrame:
    df = pd.read_csv(INDEP / "mesilla_kriged_2010/Control_points.csv")
    out = pd.DataFrame(
        {
            "site_id": df["Site ID"].astype(str),
            "source_org": np.where(
                df["Country"].eq("Mexico"), "SGM (TAAP 2010)", "NWIS (TAAP 2010)"
            ),
            "year": 2010.0,
            "obs_dtw_m": np.nan,
            "wt_elev_m": df["Median Groundwater Elevation in Feet [NAVD 88]"].astype(
                float
            )
            * FT_TO_M,
            "basin_name": "Mesilla and Conejos-Medanos",
            "independent_of_nwis": df["Country"].eq("Mexico").to_numpy(),
        }
    )
    geom = gpd.points_from_xy(
        df["Longitude [NAD 83 UTM Zone 13]"], df["Latitude [NAD 83 UTM Zone 13]"]
    )
    return _frame(out, "mesilla_2010", geom, "EPSG:26913")


def sample_raster(path: Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    with rasterio.open(path) as ds:
        if ds.crs.to_epsg() != 5070:
            raise SystemExit(f"{path} is not EPSG:5070")
        b = ds.bounds
        inside = (x >= b.left) & (x <= b.right) & (y >= b.bottom) & (y <= b.top)
        if inside.any():
            v = np.array(
                [s[0] for s in ds.sample(list(zip(x[inside], y[inside])))], "float64"
            )
            if ds.nodata is not None:
                v[v == ds.nodata] = np.nan
            v[~np.isfinite(v) | (np.abs(v) > 1e4)] = np.nan
            out[inside] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bundle",
        default="/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2_ndwr_ufold_ose",
        help="training bundle of the arm under test (distance-to-training-well attributes)",
    )
    ap.add_argument("--huc8-dir", default="/data/ssd2/handily/huc8")
    ap.add_argument("--huc8-prefixes", default="13,1108,1408,1502")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--prefix", default="nm_study_wells")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = [
        urgb_fas(),
        eastern_abq_2016(),
        sfgas_2016(),
        rgtihm_ibwc(),
        mesilla_2010(),
    ]
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=5070)
    for p in parts:
        log(f"{p['study'].iat[0]}: {len(p):,} wells")
    g["x5070"] = g.geometry.x.astype("float64")
    g["y5070"] = g.geometry.y.astype("float64")
    # a well listed in two studies at the same coordinate keeps both rows; the
    # scorer stratifies by study, so no cross-study dedupe here
    polys = gpd.read_parquet(WBD_HU8_PARQUET)[["huc8", "geometry"]]
    j = gpd.sjoin(g[["geometry"]], polys, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    g["huc8"] = j["huc8"].reindex(g.index).astype(object)
    fp = set(footprint_huc8s(args.huc8_dir, args.huc8_prefixes.split(",")))
    g["in_footprint"] = g["huc8"].isin(fp).to_numpy()
    log(
        f"footprint: {len(fp)} HUC8 basins; in footprint {int(g['in_footprint'].sum()):,} of {len(g):,}"
    )

    qn = pd.read_parquet(
        args.bundle + "/query_nodes.parquet",
        columns=["x5070", "y5070", "is_water_pseudo", "is_nwis", "source"],
    )
    real = ~qn["is_water_pseudo"].to_numpy(bool)
    for c in ("is_shore_pseudo", "is_swl_aux"):
        if c in qn.columns:
            real &= ~qn[c].to_numpy(bool)
    xy = g[["x5070", "y5070"]].to_numpy("float64")
    g["dist_train_km"] = (
        cKDTree(qn.loc[real, ["x5070", "y5070"]].to_numpy("float64")).query(xy)[0] / 1e3
    )
    nw = real & qn["is_nwis"].to_numpy(bool)
    g["dist_nwis_km"] = (
        cKDTree(qn.loc[nw, ["x5070", "y5070"]].to_numpy("float64")).query(xy)[0] / 1e3
    )
    ose = real & qn["source"].astype(str).eq("nm_ose").to_numpy()
    g["dist_ose_label_km"] = (
        cKDTree(qn.loc[ose, ["x5070", "y5070"]].to_numpy("float64")).query(xy)[0] / 1e3
    )
    x, y = g["x5070"].to_numpy("float64"), g["y5070"].to_numpy("float64")
    g["ma_dtw_m"] = sample_ma_tiles(x, y, MA_DIR)
    for name, path in MODEL_RASTERS.items():
        g[name] = sample_raster(path, x, y)
        log(f"  {name}: finite at {int(np.isfinite(g[name]).sum()):,} wells")

    df = pd.DataFrame(g.drop(columns="geometry"))
    df.to_parquet(out_dir / f"{args.prefix}_all.parquet", index=False)
    pts = df[df["in_footprint"]].reset_index(drop=True)
    pts.to_parquet(out_dir / f"{args.prefix}_points.parquet", index=False)
    summ = {
        "n_all": int(len(df)),
        "n_in_footprint": int(len(pts)),
        "per_study": {
            s: {
                "n": int(len(d)),
                "n_in_footprint": int(d["in_footprint"].sum()),
                "n_independent_of_nwis": int(d["independent_of_nwis"].sum()),
                "n_within_50m_of_training_well": int(
                    (d["dist_train_km"] <= 0.05).sum()
                ),
                "obs_dtw_median_m": float(np.nanmedian(d["obs_dtw_m"]))
                if np.isfinite(d["obs_dtw_m"]).any()
                else None,
            }
            for s, d in df.groupby("study")
        },
        "bundle": args.bundle,
        "model_rasters": {k: str(v) for k, v in MODEL_RASTERS.items()},
    }
    (out_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summ, indent=2))
    log(json.dumps(summ["per_study"], indent=1))
    log(f"wrote {out_dir / (args.prefix + '_points.parquet')} ({len(pts):,} rows)")


if __name__ == "__main__":
    main()
