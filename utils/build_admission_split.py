"""Admission split for a state driller-report well pool (the NV NDWR funnel, scripted).

Turns a gwx source's wells into the two frozen tables the data-assimilation
arms read: the ADMITTED SOURCES (seen by the model as labels in training and as
assimilated wells at inference) and the ADMISSION TARGETS (hidden; graded).

Funnel (replays the Nevada draw; every filter is recorded per site):
  1. gwx product rows of ``--source`` joined to the raw POD table on the stable
     record number for the raw use code, OSE basin and artesian code;
  2. 1 m coordinate round -> site; per-site median DTW, ``n_wells``, DTW std;
  3. footprint = the HUC8 render basins under ``--huc8-dir`` with the given
     prefixes; sites outside are dropped from the panel;
  4. eligibility = use code in ``--use-codes`` AND every well at the site in
     ``--classes`` (any-flag policy) AND record year >= ``--era-min-year`` AND
     not in an excluded OSE basin / HUC8 AND within-site DEM range <=
     ``--relief-max-m`` (window = the coordinate's PLSS class) AND not within
     1 m of a bundle training well;
  5. (``--pt-preds``) finite point-path payload (``z_surf_m``, ``r_wte_m``);
     seeded 50/50 draw over the eligible sites in panel row order;
     ``wte_residual_m = (z_surf - obs_dtw) - r_wte`` in the bundle frame.

Two phases because the payload comes from ``predict_gnn_at_points.py`` run on
the phase-1 site panel (that is how the NV frozen tables were built):

  phase 1:  build_admission_split.py --out-dir D            -> D/<prefix>_sites_input.parquet
  predict:  predict_gnn_at_points.py --points D/..._sites_input.parquet --out P
  phase 2:  build_admission_split.py --out-dir D --pt-preds P
            -> D/<prefix>_admitted_sources_frozen.parquet
               D/<prefix>_admission_targets.parquet
               D/<prefix>_admission_manifest.json

The gwx rows read are pinned to ``D/<source>_gwx_<build_id>.parquet`` on the
first run so the draw can be replayed after the product is rebuilt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conus_graph_inputs import DEM, WBD_HU8_PARQUET  # noqa: E402
from sample_benchmark_rasters import MA_DIR, sample_ma_tiles  # noqa: E402

GWX = "/data/ssd2/gwx/products/current/wells.geoparquet"
RAW_POD = "/data/ssd2/gwx/nm_ose/pod.parquet"
BUNDLE = "/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2_ndwr_ufold"
HUC8_DIR = "/data/ssd2/handily/huc8"
#: DEM cells (100 m) on a side of the within-site relief window per gwx
#: horizontal-accuracy class: the PLSS cell the coordinate is a centroid of.
#: ``unknown`` (applicant/driller-sourced, no stated precision) takes the
#: quarter-section window.
RELIEF_WINDOW_CELLS = {
    "gps": 1,
    "survey_gps": 1,
    "map": 3,
    "plss_64": 3,
    "plss_16": 5,
    "plss_4": 9,
    "plss": 9,
    "unknown": 9,
    "plss_1": 17,
}
#: Worst-first ordering used to pick a site's class when its wells disagree.
ACC_RANK = [
    "plss_1",
    "unknown",
    "plss",
    "plss_4",
    "plss_16",
    "plss_64",
    "map",
    "survey_gps",
    "gps",
]
GWX_COLS = [
    "source",
    "source_well_id",
    "canonical_id",
    "mean_dtw",
    "confinement_class",
    "confinement_source",
    "source_confinement",
    "well_class",
    "well_use",
    "h_accuracy_class",
    "por_end",
    "build_id",
    "is_static_only",
    "screen_top",
    "screen_bottom",
    "geometry",
]
RAW_COLS = ["pod_rec_nb", "use_", "pod_basin", "grnd_wtr_s", "utm_source"]


def log(msg: str) -> None:
    print(msg, flush=True)


def majority(s: pd.Series) -> str:
    return str(s.fillna("(none)").astype(str).value_counts().index[0])


def worst_class(s: pd.Series) -> str:
    vals = set(s.fillna("unknown").astype(str))
    for c in ACC_RANK:
        if c in vals:
            return c
    return "unknown"


def footprint_huc8s(huc8_dir: str, prefixes: list[str]) -> list[str]:
    out = sorted(
        p.name
        for p in Path(huc8_dir).iterdir()
        if p.is_dir()
        and any(p.name.startswith(pre) for pre in prefixes)
        and len(p.name) == 8
    )
    if not out:
        raise SystemExit(f"no HUC8 dirs under {huc8_dir} with prefixes {prefixes}")
    return out


def load_gwx(gwx: str, source: str, pin_dir: Path) -> gpd.GeoDataFrame:
    g = gpd.read_parquet(gwx, columns=GWX_COLS, filters=[("source", "==", source)])
    if g.empty:
        raise SystemExit(f"no rows for source={source} in {gwx}")
    build = sorted(set(g["build_id"].astype(str)))
    if len(build) != 1:
        raise SystemExit(f"{source} rows span several build_ids: {build}")
    pin = pin_dir / f"{source}_gwx_{build[0]}.parquet"
    if pin.exists():
        prev = gpd.read_parquet(pin, columns=["source_well_id", "mean_dtw"])
        if len(prev) != len(g):
            raise SystemExit(f"pinned {pin} has {len(prev)} rows, product has {len(g)}")
        log(f"gwx rows match the pin {pin.name} ({len(g):,} rows)")
    else:
        g.to_parquet(pin)
        log(f"pinned {len(g):,} {source} rows to {pin}")
    return g


def site_panel(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    g = load_gwx(args.gwx, args.source, out_dir)
    build_id = str(g["build_id"].iloc[0])
    raw = pd.read_parquet(args.raw_pod, columns=RAW_COLS)
    raw["source_well_id"] = raw["pod_rec_nb"].astype(str)
    if not raw["source_well_id"].is_unique:
        raise SystemExit("raw pod_rec_nb is not unique")
    g = g.merge(raw.drop(columns=["pod_rec_nb"]), on="source_well_id", how="left")
    n_unjoined = int(g["use_"].isna().sum())
    if n_unjoined:
        raise SystemExit(
            f"{n_unjoined} gwx rows did not join the raw POD table on pod_rec_nb"
        )
    g = g.to_crs(5070)
    g["x5070"] = g.geometry.x
    g["y5070"] = g.geometry.y
    ok = np.isfinite(g["x5070"]) & np.isfinite(g["y5070"]) & np.isfinite(g["mean_dtw"])
    log(
        f"wells: {len(g):,} ({int((~ok).sum())} with non-finite coordinate or DTW dropped)"
    )
    g = g[ok].reset_index(drop=True)
    g["sx"] = np.round(g["x5070"].to_numpy("float64"))
    g["sy"] = np.round(g["y5070"].to_numpy("float64"))
    g["por_end_yr"] = pd.to_datetime(g["por_end"], utc=True).dt.year.astype("float64")
    allowed = set(args.classes.split(","))
    g["class_ok"] = g["confinement_class"].isin(allowed)
    use_codes = set(args.use_codes.split(","))
    g["use_ok"] = g["use_"].isin(use_codes)

    grp = g.groupby(["sx", "sy"], sort=True)
    s = grp.agg(
        obs_dtw_m=("mean_dtw", "median"),
        dtw_std_m=("mean_dtw", "std"),
        n_wells=("mean_dtw", "size"),
        n_class_flagged=("class_ok", lambda v: int((~v).sum())),
        n_use_ok=("use_ok", "sum"),
        por_end_yr=("por_end_yr", "max"),
        use_code=("use_", majority),
        well_use=("well_use", majority),
        well_class=("well_class", majority),
        confinement_class=("confinement_class", majority),
        pod_basin=("pod_basin", majority),
        grnd_wtr_s=("grnd_wtr_s", majority),
        h_accuracy_class=("h_accuracy_class", worst_class),
        source_well_ids=("source_well_id", lambda v: ",".join(sorted(v.astype(str)))),
    ).reset_index()
    s["x5070"] = s["sx"].astype("float64")
    s["y5070"] = s["sy"].astype("float64")
    s["dtw_std_m"] = s["dtw_std_m"].fillna(0.0)
    log(
        f"sites: {len(s):,} (1 m round + site median; {int((s['n_wells'] > 1).sum()):,} multi-well)"
    )

    # HUC8 + footprint
    polys = gpd.read_parquet(WBD_HU8_PARQUET)[["huc8", "geometry"]]
    pts = gpd.GeoDataFrame(
        {"_i": np.arange(len(s))},
        geometry=gpd.points_from_xy(s["x5070"], s["y5070"]),
        crs=5070,
    )
    j = (
        gpd.sjoin(pts, polys, how="left", predicate="within")
        .drop_duplicates("_i")
        .sort_values("_i")
    )
    s["huc8"] = j["huc8"].astype(object).to_numpy()
    fp = footprint_huc8s(args.huc8_dir, args.huc8_prefixes.split(","))
    in_fp = s["huc8"].isin(fp).to_numpy(bool)
    log(
        f"footprint: {len(fp)} HUC8 basins; {int((~in_fp).sum()):,} sites outside dropped"
    )
    s = s[in_fp].reset_index(drop=True)

    # within-site relief (DEM range over the PLSS window of the coordinate class)
    s["dem_range_m"] = dem_range(s)
    # distance to bundle training wells / NWIS wells
    qn = pd.read_parquet(
        args.bundle + "/query_nodes.parquet",
        columns=["x5070", "y5070", "is_water_pseudo", "is_nwis", "source"],
    )
    real = ~qn["is_water_pseudo"].to_numpy(bool)
    for c in ("is_shore_pseudo", "is_swl_aux"):
        if c in qn.columns:
            real &= ~qn[c].to_numpy(bool)
    if qn.loc[real, "source"].astype(str).eq(args.source).any():
        raise SystemExit(f"bundle {args.bundle} already carries {args.source} rows")
    xy = s[["x5070", "y5070"]].to_numpy("float64")
    s["dist_train_km"] = (
        cKDTree(qn.loc[real, ["x5070", "y5070"]].to_numpy("float64")).query(xy)[0] / 1e3
    )
    nw = real & qn["is_nwis"].to_numpy(bool)
    s["dist_nwis_km"] = (
        cKDTree(qn.loc[nw, ["x5070", "y5070"]].to_numpy("float64")).query(xy)[0] / 1e3
    )
    s["ma_dtw_m"] = sample_ma_tiles(
        s["x5070"].to_numpy("float64"), s["y5070"].to_numpy("float64"), MA_DIR
    )

    # eligibility flags (each a reason a site is held out of the draw)
    excl_basins = (
        set(args.exclude_pod_basins.split(",")) if args.exclude_pod_basins else set()
    )
    excl_huc8 = set(args.exclude_huc8.split(",")) if args.exclude_huc8 else set()
    s["use_ok"] = s["use_code"].isin(use_codes)
    s["class_ok"] = s["n_class_flagged"] == 0
    s["era_ok"] = np.isfinite(s["por_end_yr"]) & (s["por_end_yr"] >= args.era_min_year)
    s["basin_ok"] = ~s["pod_basin"].isin(excl_basins) & ~s["huc8"].isin(excl_huc8)
    s["relief_ok"] = s["dem_range_m"] <= args.relief_max_m
    s["not_training"] = s["dist_train_km"] > 1e-3
    flags = ["use_ok", "class_ok", "era_ok", "basin_ok", "relief_ok", "not_training"]
    s["is_admission_eligible_pre"] = s[flags].all(axis=1)
    s.attrs["build_id"] = build_id
    funnel = {"n_wells": int(len(g)), "n_sites_in_footprint": int(len(s))}
    keep = np.ones(len(s), bool)
    for f in flags:
        v = s[f].to_numpy(bool)
        funnel[f"dropped_{f}"] = int((keep & ~v).sum())
        keep &= v
    funnel["n_eligible_pre_payload"] = int(keep.sum())
    log("funnel: " + json.dumps(funnel))
    s["build_id"] = build_id
    return s, funnel


def dem_range(s: pd.DataFrame) -> np.ndarray:
    x = s["x5070"].to_numpy("float64")
    y = s["y5070"].to_numpy("float64")
    pad = 2000.0
    with rasterio.open(DEM) as ds:
        w = (
            from_bounds(
                x.min() - pad, y.min() - pad, x.max() + pad, y.max() + pad, ds.transform
            )
            .round_offsets()
            .round_lengths()
        )
        arr = ds.read(1, window=w).astype("float64")
        tr = ds.window_transform(w)
        nod = ds.nodata
    if nod is not None:
        arr[arr == nod] = np.nan
    col = np.floor((x - tr.c) / tr.a).astype(int)
    row = np.floor((y - tr.f) / tr.e).astype(int)
    if (
        (row < 0).any()
        or (row >= arr.shape[0]).any()
        or (col < 0).any()
        or (col >= arr.shape[1]).any()
    ):
        raise SystemExit("sites fall outside the DEM window")
    cls = s["h_accuracy_class"].astype(str).to_numpy()
    out = np.full(len(s), np.nan)
    for c in np.unique(cls):
        n = RELIEF_WINDOW_CELLS.get(c)
        if n is None:
            raise SystemExit(f"no relief window for h_accuracy_class {c!r}")
        m = cls == c
        if n == 1:
            out[m] = 0.0
            continue
        rng = maximum_filter(arr, size=n, mode="nearest") - minimum_filter(
            arr, size=n, mode="nearest"
        )
        out[m] = rng[row[m], col[m]]
    if not np.isfinite(out).all():
        raise SystemExit(
            f"{int((~np.isfinite(out)).sum())} sites have a non-finite DEM range"
        )
    return out


def finalize(args: argparse.Namespace, out_dir: Path, pfx: str) -> None:
    s = pd.read_parquet(out_dir / f"{pfx}_sites_input.parquet")
    p = pd.read_parquet(args.pt_preds)
    need = ["x5070", "y5070", "z_surf_m", "r_wte_m", "pred_dtw_m"]
    p = p[need + [c for c in ("fac_rem_dtw_m",) if c in p.columns]].drop_duplicates(
        ["x5070", "y5070"]
    )
    m = s.merge(p, on=["x5070", "y5070"], how="left")
    if len(m) != len(s):
        raise SystemExit("point-prediction join is not 1:1 on x5070/y5070")
    m = m.rename(columns={"pred_dtw_m": f"{args.pred_name}_pt_dtw_m"})
    m["payload_ok"] = np.isfinite(m["z_surf_m"]) & np.isfinite(m["r_wte_m"])
    m["is_admission_eligible"] = m["is_admission_eligible_pre"] & m["payload_ok"]
    m["wte_residual_m"] = (m["z_surf_m"] - m["obs_dtw_m"]) - m["r_wte_m"]
    elig = np.where(m["is_admission_eligible"].to_numpy(bool))[0]
    n_adm = len(elig) // 2
    rng = np.random.default_rng(args.seed)
    pick = elig[rng.choice(len(elig), n_adm, replace=False)]
    admitted = np.zeros(len(m), bool)
    admitted[pick] = True
    m["is_admitted"] = admitted
    src = m[admitted].reset_index(drop=True)
    tgt = m[~admitted].reset_index(drop=True)
    src_cols = [
        "x5070",
        "y5070",
        "wte_residual_m",
        "z_surf_m",
        "sx",
        "sy",
        "obs_dtw_m",
        "r_wte_m",
        "por_end_yr",
        "pod_basin",
        "huc8",
        "n_wells",
        "dtw_std_m",
        "use_code",
        "well_class",
        "confinement_class",
        "h_accuracy_class",
        "dem_range_m",
        "source_well_ids",
        "build_id",
    ]
    src[src_cols].to_parquet(
        out_dir / f"{pfx}_admitted_sources_frozen.parquet", index=False
    )
    tgt_cols = [c for c in m.columns if c not in ("is_admitted",)]
    tgt[tgt_cols].to_parquet(out_dir / f"{pfx}_admission_targets.parquet", index=False)
    funnel = json.loads((out_dir / f"{pfx}_funnel_phase1.json").read_text())
    funnel.update(
        {
            "dropped_payload": int(
                (m["is_admission_eligible_pre"] & ~m["payload_ok"]).sum()
            ),
            "n_eligible": int(len(elig)),
            "n_admitted": int(n_adm),
            "n_held_out_eligible": int(len(elig) - n_adm),
            "n_targets_total": int(len(tgt)),
        }
    )
    man = {
        "source": args.source,
        "gwx": args.gwx,
        "build_id": str(m["build_id"].iloc[0]),
        "raw_pod": args.raw_pod,
        "bundle": args.bundle,
        "pt_preds": args.pt_preds,
        "pt_preds_sha1": hashlib.sha1(Path(args.pt_preds).read_bytes()).hexdigest(),
        "use_codes": args.use_codes,
        "classes": args.classes,
        "era_min_year": args.era_min_year,
        "exclude_pod_basins": args.exclude_pod_basins,
        "exclude_huc8": args.exclude_huc8,
        "relief_max_m": args.relief_max_m,
        "relief_window_cells": RELIEF_WINDOW_CELLS,
        "huc8_prefixes": args.huc8_prefixes,
        "split": {
            "seed": args.seed,
            "fraction": 0.5,
            "rule": "default_rng(seed).choice(n_eligible, n_eligible//2) over eligible sites in (sx, sy)-sorted panel order",
        },
        "payload": "wte_residual_m = (z_surf_m - obs_dtw_m) - r_wte_m, z_surf/r_wte from pt_preds",
        "funnel": funnel,
        "admitted_summary": {
            "median_obs_dtw_m": float(src["obs_dtw_m"].median()),
            "frac_lt_5m": float((src["obs_dtw_m"] < 5).mean()),
            "use_code": src["use_code"].value_counts().to_dict(),
            "h_accuracy_class": src["h_accuracy_class"].value_counts().to_dict(),
        },
    }
    (out_dir / f"{pfx}_admission_manifest.json").write_text(json.dumps(man, indent=2))
    log("funnel: " + json.dumps(funnel))
    log(
        f"admitted {n_adm:,} sources (median DTW {src['obs_dtw_m'].median():.1f} m, {(src['obs_dtw_m'] < 5).mean():.0%} < 5 m); targets {len(tgt):,} ({len(elig) - n_adm:,} eligible held out)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gwx", default=GWX)
    ap.add_argument("--raw-pod", default=RAW_POD)
    ap.add_argument("--source", default="nm_ose")
    ap.add_argument(
        "--bundle",
        default=BUNDLE,
        help="base bundle (training wells to exclude; must not carry --source)",
    )
    ap.add_argument("--use-codes", default="MON,DOM", help="raw POD use codes admitted")
    ap.add_argument(
        "--classes",
        default="unconfined,unconfined_marginal",
        help="gwx confinement classes admitted (every well at a site)",
    )
    ap.add_argument("--era-min-year", type=int, default=1980)
    ap.add_argument(
        "--exclude-pod-basins",
        default="RA",
        help="OSE basin codes excluded (Roswell Artesian)",
    )
    ap.add_argument("--exclude-huc8", default="13060008")
    ap.add_argument("--relief-max-m", type=float, default=10.0)
    ap.add_argument("--huc8-dir", default=HUC8_DIR)
    ap.add_argument(
        "--huc8-prefixes",
        default="13,1108,1408,1502",
        help="render-footprint HUC8 prefixes",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="ose")
    ap.add_argument(
        "--pt-preds",
        help="phase 2: predict_gnn_at_points.py output on the phase-1 site panel",
    )
    ap.add_argument(
        "--pred-name",
        default="m20_ord",
        help="phase 2: column prefix for the point prediction carried on targets",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.pt_preds:
        finalize(args, out_dir, args.prefix)
        return
    s, funnel = site_panel(args, out_dir)
    s.to_parquet(out_dir / f"{args.prefix}_sites_input.parquet", index=False)
    (out_dir / f"{args.prefix}_funnel_phase1.json").write_text(
        json.dumps(funnel, indent=2)
    )
    log(f"wrote {out_dir / (args.prefix + '_sites_input.parquet')}: {len(s):,} sites")


if __name__ == "__main__":
    main()
