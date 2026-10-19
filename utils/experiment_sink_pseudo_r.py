"""Sink pseudo-well experiment for the leave-fold-out crossfit R (NV fix round, item A).

Adds SOURCE-ONLY "sink" pseudo-wells (a cluster per HUC8 at the land-surface
minimum, WTE = DEM - 5 m) to the neighbor pool of the bundle's crossfit
relief-IDW R and measures R - wte_obs at the real bundle wells, before vs after.

Contract replicated from the bundle (graph_manifest ``appended_wells.r_contract``):
R = ``crossfit_idw`` on [all real rows] with pool = base real rows (the appended
``nv_ndwr`` rows are labelled targets but NOT R sources), folds = ``cv_fold``
(HUC12-unit blocked), k / power / relief vw from the manifest, z = z_surf_well_m.
Sink rows are added to the pool only; R is evaluated at real wells only.

Fold variants for the sink rows:
  * ``unit``: the fold of the sink's HUC12 unit via the bundle's cv_unit -> cv_fold
    map (novel units: fold of the nearest real well, whole unit), so a held-out
    unit loses its own sink.
  * ``nofold``: never held out (always in the pool).

Usage:
  uv run python utils/experiment_sink_pseudo_r.py --bundle <dir> --n-sink 5,10,20
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
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_conus_graph_inputs as B  # noqa: E402
from append_labelled_wells import novel_unit_folds  # noqa: E402

DEM = B.DEM
WBD_HU8 = B.WBD_HU8_PARQUET
HUC12_PATH = "/nas/hydrography/HUC_Boundaries/wbd_national/wbdhu12_5070.parquet"
STATES = "/nas/boundaries/us_states_tiger_wgs.shp"
HUC8_ROOT = Path("/data/ssd2/handily/huc8")
SHIPPED_ARM = "gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold"
PV_BBOX = (-117.24, 41.02, -117.09, 41.17)  # Paradise Valley patch, lon/lat
PV_RADIUS_M = 12000.0
PV_BLOCK = "-44_55"
PV_FOLD = 5
NEG_BASINS = [
    "16020301",
    "16020306",
    "16020308",
    "16040105",
    "16040202",
    "16050103",
    "16050302",
    "16060008",
    "16060014",
    "17050102",
    "17050107",
    "17120009",
    "18090201",
    "18090203",
]
DEPTH_BANDS = [0, 2, 5, 10, 30, np.inf]


def nv_basins(state: str = "NV") -> list[str]:
    """Same rule as the renderer's --state: WBD `states` contains NV, basin on disk."""
    wbd = pd.read_parquet(WBD_HU8, columns=["huc8", "states"])
    codes = set(
        wbd[wbd["states"].astype(str).str.contains(state, na=False)]["huc8"].astype(str)
    )
    have = {d.name for d in HUC8_ROOT.iterdir() if d.is_dir()}
    return sorted(codes & have)


def sink_huc8s(buffer_m: float) -> gpd.GeoDataFrame:
    """HUC8 polygons intersecting Nevada buffered by ``buffer_m``."""
    st = gpd.read_file(STATES)
    nv = st[st["STUSPS"] == "NV"].to_crs(5070)
    zone = nv.geometry.union_all().buffer(buffer_m)
    h8 = gpd.read_parquet(WBD_HU8)[["huc8", "geometry"]]
    if h8.crs is None or h8.crs.to_epsg() != 5070:
        raise SystemExit(f"HUC8 polys not EPSG:5070: {WBD_HU8}")
    return h8[h8.intersects(zone)].reset_index(drop=True)


def basin_sinks(
    src: rasterio.DatasetReader, geom, n_max: int, spacing_m: float, tol_m: float
) -> pd.DataFrame:
    """Greedy low-point cluster: lowest in-basin DEM cells, >= spacing apart, z <= min+tol."""
    win = (
        from_bounds(*geom.bounds, transform=src.transform)
        .round_offsets()
        .round_lengths()
    )
    win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
    arr = src.read(1, window=win)
    t = src.window_transform(win)
    inside = geometry_mask([geom], out_shape=arr.shape, transform=t, invert=True)
    valid = inside & (arr != src.nodata)
    if not valid.any():
        raise SystemExit(f"no valid DEM cells in basin window {geom.bounds}")
    z = arr[valid].astype("float64")
    zmin = z.min()
    rows, cols = np.nonzero(valid)
    cand = z <= zmin + tol_m
    z, rows, cols = z[cand], rows[cand], cols[cand]
    order = np.argsort(z, kind="stable")
    z, rows, cols = z[order], rows[order], cols[order]
    x = t.c + (cols + 0.5) * t.a
    y = t.f + (rows + 0.5) * t.e
    alive = np.ones(len(z), bool)
    pick: list[int] = []
    while len(pick) < n_max and alive.any():
        i = int(np.argmax(alive))
        pick.append(i)
        alive &= np.hypot(x - x[i], y - y[i]) >= spacing_m
    pick = np.array(pick)
    return pd.DataFrame(
        {"x5070": x[pick], "y5070": y[pick], "z": z[pick], "z_min_basin": zmin}
    )


def build_sinks(
    h8: gpd.GeoDataFrame, n_max: int, spacing_m: float, tol_m: float, depth_m: float
) -> pd.DataFrame:
    out = []
    with rasterio.open(DEM) as src:
        for huc, geom in zip(h8["huc8"].astype(str), h8.geometry):
            df = basin_sinks(src, geom, n_max, spacing_m, tol_m)
            df["huc8"] = huc
            df["rank"] = np.arange(len(df))
            out.append(df)
    s = pd.concat(out, ignore_index=True)
    s["z_surf_well_m"] = s.pop("z")
    s["wte_obs_m"] = s["z_surf_well_m"] - depth_m
    s["mean_dtw"] = depth_m
    s["well_class"] = "sink_pseudo"
    s["is_sink_pseudo"] = True
    s["canonical_id"] = [
        f"sink_{hashlib.sha1(f'{a:.1f}_{b:.1f}'.encode()).hexdigest()[:16]}"
        for a, b in zip(s["x5070"], s["y5070"])
    ]
    return s


def sink_folds(
    sinks: pd.DataFrame, real: pd.DataFrame, cv_block_km: float
) -> pd.Series:
    """cv_fold per sink row from the bundle's HUC12 unit -> fold map (whole-unit)."""
    sx = sinks["x5070"].to_numpy("float64")
    sy = sinks["y5070"].to_numpy("float64")
    unit = np.asarray(B.huc12_units(sx, sy, HUC12_PATH, cv_block_km))
    uf = real[["cv_unit", "cv_fold"]].drop_duplicates().set_index("cv_unit")["cv_fold"]
    if uf.index.has_duplicates:
        raise SystemExit("bundle cv_unit -> cv_fold map is not unique; investigate")
    fold = pd.Series(unit).map(uf).to_numpy()
    novel = pd.isna(fold)
    if novel.any():
        fold[novel] = novel_unit_folds(
            unit[novel],
            np.c_[sx[novel], sy[novel]],
            real[["x5070", "y5070"]].to_numpy("float64"),
            real["cv_fold"].to_numpy("int64"),
        )
    print(
        f"sink folds: {int((~novel).sum())} rows in bundle units, "
        f"{int(novel.sum())} rows in {pd.unique(unit[novel]).size} novel units"
    )
    return pd.Series(fold.astype("int64"), index=sinks.index), pd.Series(
        unit, index=sinks.index
    )


def md_table(df: pd.DataFrame, index: bool = False, fmt: str = ".2f") -> str:
    """Pipe-table markdown without the optional tabulate dependency."""
    if index:
        df = df.reset_index()
    cells = [
        [f"{v:{fmt}}" if isinstance(v, float) else str(v) for v in row]
        for row in df.itertuples(index=False)
    ]
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(c) + " |" for c in cells])


def panel(resid: np.ndarray) -> dict:
    r = resid[np.isfinite(resid)]
    a = np.abs(r)
    if not len(r):
        return {"n": 0}
    return {
        "n": int(len(r)),
        "median_abs_m": float(np.median(a)),
        "p90_abs_m": float(np.percentile(a, 90)),
        "mean_bias_m": float(np.mean(r)),
        "median_resid_m": float(np.median(r)),
        "rmse_m": float(np.sqrt(np.mean(r**2))),
    }


def crossfit_r(xy, z, wte, fold, pool, k, p, vw) -> np.ndarray:
    return B.crossfit_idw(xy, wte, fold, k, p, z=z, vw=vw, pool=pool)


def sink_weight_share(
    pool: pd.DataFrame,
    sinks: pd.DataFrame,
    sink_fold: np.ndarray,
    q: pd.DataFrame,
    k: int,
    p: float,
    vw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per query well: IDW weight share of sink rows, of out-of-basin sink rows, and
    the horizontal distance (m) to every sink neighbor -- same kNN as crossfit_idw."""
    cols = ["x5070", "y5070", "z_surf_well_m", "cv_fold", "huc8"]
    P = pd.concat(
        [
            pool[cols].assign(sink=False),
            sinks[cols].assign(cv_fold=sink_fold, sink=True),
        ],
        ignore_index=True,
    )
    sw = np.zeros(len(q))
    swo = np.zeros(len(q))
    dh = []
    qf = q["cv_fold"].to_numpy()
    for f in np.unique(qf):
        te = qf == f
        Pf = P[P["cv_fold"].to_numpy() != f]
        tree = cKDTree(
            B._relief_coords(
                Pf[["x5070", "y5070"]].to_numpy("float64"),
                Pf["z_surf_well_m"].to_numpy("float64"),
                vw,
            )
        )
        Q = q[te]
        dist, idx = tree.query(
            B._relief_coords(
                Q[["x5070", "y5070"]].to_numpy("float64"),
                Q["z_surf_well_m"].to_numpy("float64"),
                vw,
            ),
            k=k,
        )
        w = 1.0 / np.maximum(dist, 1.0) ** p
        w /= w.sum(1, keepdims=True)
        isk = Pf["sink"].to_numpy(bool)[idx]
        same = (
            Pf["huc8"].astype(str).to_numpy()[idx]
            == Q["huc8"].astype(str).to_numpy()[:, None]
        )
        sw[te] = (w * isk).sum(1)
        swo[te] = (w * isk * ~same).sum(1)
        hd = np.hypot(
            Pf["x5070"].to_numpy()[idx] - Q["x5070"].to_numpy()[:, None],
            Pf["y5070"].to_numpy()[idx] - Q["y5070"].to_numpy()[:, None],
        )
        dh.append(hd[isk])
    return sw, swo, np.concatenate(dh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--bundle",
        default="/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2_ndwr_ufold",
    )
    ap.add_argument("--n-sink", default="5,10,20")
    ap.add_argument(
        "--out-dir", default="/data/ssd2/handily/nv/regional/sink_pseudo_r/"
    )
    ap.add_argument("--spacing-m", type=float, default=2000.0)
    ap.add_argument("--tol-m", type=float, default=2.0)
    ap.add_argument("--sink-depth-m", type=float, default=5.0)
    ap.add_argument("--buffer-m", type=float, default=60000.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = Path(args.bundle)
    man = json.loads((bundle / "graph_manifest.json").read_text())
    k, p, vw = man["idw_k"], man["idw_power"], man["r_relief_vw"]
    print(f"manifest: k={k} power={p} vw={vw} folds={man['cv_scheme']}")
    prev = [man["appended_wells"]["source_name"]] if man.get("appended_wells") else []

    qn = pd.read_parquet(bundle / "query_nodes.parquet")
    for c in ("is_shore_pseudo", "is_swl_aux"):
        if c in qn.columns:
            raise SystemExit(f"bundle has {c}; extend the real-row rule before use")
    real = qn[~qn["is_water_pseudo"].to_numpy(bool)].reset_index(drop=True)
    pool_real = ~real["source"].astype(str).isin(prev).to_numpy(bool)
    print(
        f"real rows {len(real):,}; R pool (excl. {prev}) {int(pool_real.sum()):,}; "
        f"monitoring {int((real['well_class'] == 'monitoring').sum()):,}"
    )
    xy = real[["x5070", "y5070"]].to_numpy("float64")
    z = real["z_surf_well_m"].to_numpy("float64")
    wte = real["wte_obs_m"].to_numpy("float64")
    fold = real["cv_fold"].to_numpy("int64")
    arch = real["regional_wte_idw_oof_m"].to_numpy("float64")

    # --- groups -------------------------------------------------------------
    nv = nv_basins()
    is_nv = real["huc8"].astype(str).isin(nv).to_numpy()
    is_mon = (real["well_class"] == "monitoring").to_numpy()
    lon0, lat0 = (PV_BBOX[0] + PV_BBOX[2]) / 2, (PV_BBOX[1] + PV_BBOX[3]) / 2
    to5070 = Transformer.from_crs(4326, 5070, always_xy=True)
    cx, cy = to5070.transform(lon0, lat0)
    d_pv = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    is_pv = d_pv <= PV_RADIUS_M
    bx, by = to5070.transform(
        [PV_BBOX[0], PV_BBOX[2], PV_BBOX[2], PV_BBOX[0]],
        [PV_BBOX[1], PV_BBOX[1], PV_BBOX[3], PV_BBOX[3]],
    )
    from shapely.geometry import Polygon

    pv_poly = gpd.GeoSeries([Polygon(zip(bx, by))], crs=5070)
    d_poly = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=5070).distance(
        pv_poly.iloc[0]
    )
    n_pv_poly = int((d_poly.to_numpy() <= PV_RADIUS_M).sum())
    # negative-DTW patch of the shipped render inside the bbox (100 m cells < 0 m)
    with rasterio.open(
        HUC8_ROOT / "16040105" / "gnn" / SHIPPED_ARM / "gnn_dtw_100m.tif"
    ) as src:
        dtw_pv = src.read(1, masked=True)
        tpv = src.transform
    prow, pcol = np.nonzero((dtw_pv < 0).filled(False))
    px = tpv.c + (pcol + 0.5) * tpv.a
    py = tpv.f + (prow + 0.5) * tpv.e
    plon, plat = Transformer.from_crs(5070, 4326, always_xy=True).transform(px, py)
    inb = (
        (plon >= PV_BBOX[0])
        & (plon <= PV_BBOX[2])
        & (plat >= PV_BBOX[1])
        & (plat <= PV_BBOX[3])
    )
    d_patch, _ = cKDTree(np.c_[px[inb], py[inb]]).query(xy)
    is_pv_patch = d_patch <= PV_RADIUS_M
    print(
        f"PV negative patch: {int(inb.sum())} cells ({inb.sum() * 0.01:.1f} km2); "
        f"wells <= 12 km of patch {int(is_pv_patch.sum())}"
    )
    is_pvblk = (real["block_40km"].astype(str) == PV_BLOCK).to_numpy() & (
        fold == PV_FOLD
    )
    is_neg = real["huc8"].astype(str).isin(NEG_BASINS).to_numpy()
    print(
        f"NV basins {len(nv)}; NV real wells {int(is_nv.sum()):,}; PV wells "
        f"(<= 12 km of bbox centroid) {int(is_pv.sum())}, (<= 12 km of bbox edge) "
        f"{n_pv_poly}; PV folds {np.unique(fold[is_pv]).tolist()} blocks "
        f"{pd.unique(real.loc[is_pv, 'block_40km']).tolist()}; fold5/{PV_BLOCK} "
        f"{int(is_pvblk.sum())}"
    )

    # --- 1. replication -------------------------------------------------------
    r0 = crossfit_r(xy, z, wte, fold, pool_real, k, p, vw)
    d0 = np.abs(r0 - arch)
    rep = {
        "all_real_median_m": float(np.median(d0)),
        "all_real_p99_m": float(np.percentile(d0, 99)),
        "all_real_max_m": float(np.max(d0)),
        "nv_median_m": float(np.median(d0[is_nv])),
        "nv_p99_m": float(np.percentile(d0[is_nv], 99)),
        "nv_max_m": float(np.max(d0[is_nv])),
    }
    print("replication |R_rep - R_archived|:", json.dumps(rep))
    if rep["all_real_max_m"] > 1e-3:
        raise SystemExit("replication FAILED (max > 1e-3 m); experiment invalid")
    (out_dir / "replication.json").write_text(json.dumps(rep, indent=1))

    # --- 2. sinks + arms ------------------------------------------------------
    h8 = sink_huc8s(args.buffer_m)
    print(f"sink HUC8s (NV + {args.buffer_m / 1000:.0f} km): {len(h8)}")
    n_list = [int(v) for v in args.n_sink.split(",")]
    sinks_max = build_sinks(
        h8, max(n_list), args.spacing_m, args.tol_m, args.sink_depth_m
    )
    sfold, sunit = sink_folds(sinks_max, real, man["cv_block_km"])
    sinks_max["cv_fold"] = sfold
    sinks_max["cv_unit"] = sunit

    groups = {
        "pv_12km_of_neg_patch": is_pv_patch,
        "pv_12km_of_bbox_centroid": is_pv,
        "pv_12km_of_bbox_edge": d_poly.to_numpy() <= PV_RADIUS_M,
        "pv_fold5_block_-44_55": is_pvblk,
        "neg_dtw_14_basins": is_neg,
        "nv_all": is_nv,
        "nv_monitoring": is_nv & is_mon,
        "nv_ndwr": is_nv & ~is_mon,
        "conus_monitoring_34503": is_mon,
    }
    arms = {"baseline": arch}
    meta = []
    wdiag = []
    for n in n_list:
        s = sinks_max[sinks_max["rank"] < n].reset_index(drop=True)
        s.to_parquet(out_dir / f"sink_pseudo_rows_N{n}.parquet")
        per_basin = s.groupby("huc8").size()
        meta.append(
            {
                "N": n,
                "n_rows": len(s),
                "n_huc8": int(per_basin.size),
                "rows_per_huc8_median": float(per_basin.median()),
                "huc8_lt_N": int((per_basin < n).sum()),
            }
        )
        sxy = s[["x5070", "y5070"]].to_numpy("float64")
        sz = s["z_surf_well_m"].to_numpy("float64")
        swte = s["wte_obs_m"].to_numpy("float64")
        for var in ("unit", "nofold"):
            sf = (
                s["cv_fold"].to_numpy("int64") if var == "unit" else np.full(len(s), -1)
            )
            r = crossfit_r(
                np.r_[xy, sxy],
                np.r_[z, sz],
                np.r_[wte, swte],
                np.r_[fold, sf],
                np.r_[pool_real, np.ones(len(s), bool)],
                k,
                p,
                vw,
            )[: len(real)]
            arms[f"N{n}_{var}"] = r
            sw, swo, dh = sink_weight_share(
                real[pool_real], s, sf, real[is_nv], k, p, vw
            )
            pvq = is_pvblk[is_nv]
            wdiag.append(
                {
                    "arm": f"N{n}_{var}",
                    "nv_sink_wshare_mean": float(sw.mean()),
                    "nv_outbasin_sink_wshare_mean": float(swo.mean()),
                    "pv_block_sink_wshare_mean": float(sw[pvq].mean()),
                    "pv_block_outbasin_sink_wshare_mean": float(swo[pvq].mean()),
                    "sink_neighbor_hdist_km_median": float(np.median(dh) / 1e3),
                }
            )
    print(pd.DataFrame(meta).to_string(index=False))
    pd.DataFrame(meta).to_csv(out_dir / "sink_counts.csv", index=False)
    wd = pd.DataFrame(wdiag)
    wd.to_csv(out_dir / "sink_weight_diagnostic.csv", index=False)
    print(wd.round(3).to_string(index=False))

    # --- 3. panels ------------------------------------------------------------
    rows = []
    for arm, r in arms.items():
        res = r - wte
        for g, m in groups.items():
            rows.append({"arm": arm, "group": g, **panel(res[m])})
    pan = pd.DataFrame(rows)
    pan.to_csv(out_dir / "panel_groups.csv", index=False)

    dep = real["mean_dtw"].to_numpy("float64")
    band = pd.cut(
        dep, DEPTH_BANDS, right=False, labels=["0-2", "2-5", "5-10", "10-30", "30+"]
    )
    rows = []
    for arm, r in arms.items():
        res = r - wte
        for b in band.categories:
            m = is_nv & (band == b)
            rows.append({"arm": arm, "obs_dtw_band_m": b, **panel(res[m])})
    pd.DataFrame(rows).to_csv(out_dir / "panel_nv_depth.csv", index=False)

    rows = []
    hu = real["huc8"].astype(str).to_numpy()
    for arm, r in arms.items():
        res = np.abs(r - wte)
        for b in sorted(set(hu[is_nv])):
            m = is_nv & (hu == b)
            rows.append(
                {
                    "arm": arm,
                    "huc8": b,
                    "n": int(m.sum()),
                    "median_abs_m": float(np.median(res[m])),
                    "p90_abs_m": float(np.percentile(res[m], 90)),
                    "mean_bias_m": float(np.mean((r - wte)[m])),
                }
            )
    hb = pd.DataFrame(rows)
    hb.to_csv(out_dir / "panel_nv_huc8.csv", index=False)

    # per-well R for the PV + negative basins (QGIS-free inspection)
    keep = is_pv_patch | is_pvblk | is_neg
    pw = real.loc[
        keep,
        [
            "canonical_id",
            "source",
            "huc8",
            "x5070",
            "y5070",
            "cv_fold",
            "block_40km",
            "z_surf_well_m",
            "wte_obs_m",
        ],
    ].copy()
    for arm, r in arms.items():
        pw[f"R_{arm}"] = r[keep]
    pw.to_parquet(out_dir / "per_well_r_pv_neg.parquet")

    # --- 4. markdown summary --------------------------------------------------
    def fmt(df):
        cols = [
            "arm",
            "n",
            "median_abs_m",
            "p90_abs_m",
            "mean_bias_m",
            "median_resid_m",
            "rmse_m",
        ]
        return md_table(df[cols])

    md = [
        "# Sink pseudo-well R experiment",
        "",
        f"Bundle `{bundle}`; k {k}, power {p}, relief vw {vw}; resid = R - wte_obs (m).",
        "",
        "## Replication |R_rep - R_archived| (m)",
        "",
        json.dumps(rep, indent=1),
        "",
        "## Sink rows",
        "",
        md_table(pd.DataFrame(meta)),
        "",
        "## Sink IDW weight share at NV wells",
        "",
        md_table(wd, fmt=".3f"),
        "",
    ]
    for g in groups:
        md += [f"## {g}", "", fmt(pan[pan["group"] == g]), ""]
    neg = hb[hb["huc8"].isin(NEG_BASINS)].pivot(
        index="huc8", columns="arm", values="mean_bias_m"
    )
    md += [
        "## Mean bias (m) by negative-DTW basin",
        "",
        md_table(neg, index=True, fmt=".1f"),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
