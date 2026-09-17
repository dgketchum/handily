"""Append externally labelled wells to a CONUS GNN query-node bundle as REAL rows.

Writes a DERIVED bundle: heavy base files (reach_nodes, channel_edges,
permanent_water_blocks, reach_graph_manifest) are symlinked; query_nodes,
lateral_edges and graph_manifest are rewritten with the new rows appended after
the base rows (base rows byte-identical, query_node_idx = arange). Unlike the
E4 SWL path the appended rows are ordinary labelled query nodes: they enter
`real`, carry label weight 1.0, are source-eligible, and are scored by the
trainer's OOF panel under their own cv_fold.

Contract (mirrors build_swl_labels.py):
  * R and the deep datum are crossfit on [base real U new] with pool = base real,
    so every base row reproduces its archived R (parity assert <= 1e-3 m) and a
    new row's R comes from base wells outside its own fold. R is never refit.
  * cv_fold = the base bundle's HUC12 unit -> fold map; a novel HUC12 inherits, as
    a whole unit, the fold of the base real well nearest to any of its rows
    (assign_folds is never re-run; a unit is never split across folds).
  * 34 query features via the base samplers; any new row NaN in a base-finite
    feature column is dropped (feature-matrix width unchanged).
  * Wells within --dedupe-m of a base real well are dropped (already labelled).

Usage:
  uv run python utils/append_labelled_wells.py --bundle <base> --wells <parquet>
      --out-bundle <dir> --source-name nv_ndwr --well-class ndwr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_conus_graph_inputs as B  # noqa: E402

DEM = B.DEM
GEOM = "/data/ssd2/handily/conus/wte_gnn/fac_flowline_geom_conus.parquet"
DD_POINTS = "/data/ssd2/handily/conus/wte_gnn/drilled_depth_points.parquet"
HUC12_PATH = "/nas/hydrography/HUC_Boundaries/wbd_national/wbdhu12_5070.parquet"
HUC8_PATH = B.WBD_HU8_PARQUET
SYMLINK = [
    "reach_nodes.parquet",
    "channel_edges.parquet",
    "permanent_water_blocks.parquet",
    "reach_graph_manifest.json",
]
BLOCK_SIZE_M = 40000.0
DD_SELF_EXCLUDE_M = 100.0


def _ids(prefix: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array(
        [
            f"{prefix}_{hashlib.sha1(f'{xi:.3f}_{yi:.3f}'.encode()).hexdigest()[:16]}"
            for xi, yi in zip(x, y)
        ],
        dtype=object,
    )


def novel_unit_folds(
    units: np.ndarray,
    xy: np.ndarray,
    base_xy: np.ndarray,
    base_fold: np.ndarray,
) -> np.ndarray:
    """Fold per row for rows whose HUC12 unit has no base well.

    Assigned per UNIT: every row of a novel unit takes the fold of the base
    well nearest to any row of that unit, so the spatial block stays intact
    (a per-row nearest-well rule split 35 of 425 novel NV units across folds).
    """
    d, nn = cKDTree(base_xy).query(xy)
    pick = (
        pd.DataFrame({"unit": units, "d": d, "fold": base_fold[nn]})
        .sort_values(["unit", "d"], kind="stable")
        .drop_duplicates("unit")
        .set_index("unit")["fold"]
    )
    out = pd.Series(units).map(pick).to_numpy("int64")
    assert pd.DataFrame({"u": units, "f": out}).groupby("u")["f"].nunique().max() == 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", required=True, help="base bundle dir (v2)")
    ap.add_argument("--wells", required=True, help="parquet: x5070,y5070,obs_dtw_m")
    ap.add_argument("--out-bundle", required=True)
    ap.add_argument("--source-name", default="nv_ndwr")
    ap.add_argument("--well-class", default="ndwr")
    ap.add_argument("--dtw-col", default="obs_dtw_m")
    ap.add_argument("--dedupe-m", type=float, default=1.0)
    args = ap.parse_args()

    bundle = Path(args.bundle)
    out_dir = Path(args.out_bundle)
    out_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads((bundle / "graph_manifest.json").read_text())
    idw_k, idw_p, vw = man["idw_k"], man["idw_power"], man["r_relief_vw"]
    dd = man["deep_datum"]
    dh = man["dupuit_hang"]
    knn_lat, cond_p = man["knn_lateral"], man["conductance_p"]
    funnel: dict[str, int] = {}

    qn = pd.read_parquet(bundle / "query_nodes.parquet")
    base_n = len(qn)
    water = qn["is_water_pseudo"].to_numpy(bool)
    shore = (
        qn["is_shore_pseudo"].to_numpy(bool)
        if "is_shore_pseudo" in qn.columns
        else np.zeros(base_n, bool)
    )
    swl = (
        qn["is_swl_aux"].to_numpy(bool)
        if "is_swl_aux" in qn.columns
        else np.zeros(base_n, bool)
    )
    real = ~water & ~shore & ~swl
    mon = qn[real].reset_index(drop=True)
    mx = mon["x5070"].to_numpy("float64")
    my = mon["y5070"].to_numpy("float64")
    mz = mon["z_surf_well_m"].to_numpy("float64")
    mwte = mon["wte_obs_m"].to_numpy("float64")
    mfold = mon["cv_fold"].to_numpy("int64")
    n_mon = len(mon)
    print(f"base query nodes: {base_n:,} (real {n_mon:,})")

    wells = pd.read_parquet(args.wells)
    funnel["wells_in"] = len(wells)
    ax = wells["x5070"].to_numpy("float64")
    ay = wells["y5070"].to_numpy("float64")
    a_dtw = wells[args.dtw_col].to_numpy("float64")
    keep = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(a_dtw)
    funnel["dropped_nonfinite_input"] = int((~keep).sum())
    # collocated with a base real well -> already labelled, drop
    d_near, _ = cKDTree(np.c_[mx, my]).query(np.c_[ax, ay])
    dup = d_near <= args.dedupe_m
    funnel[f"dropped_collocated_le_{args.dedupe_m:g}m"] = int((dup & keep).sum())
    keep &= ~dup
    ax, ay, a_dtw = ax[keep], ay[keep], a_dtw[keep]

    az = B.sample_coarse(DEM, ax, ay)
    fz = np.isfinite(az)
    funnel["dropped_nonfinite_z_surf"] = int((~fz).sum())
    ax, ay, a_dtw, az = ax[fz], ay[fz], a_dtw[fz], az[fz]
    fac_dtw = B.sample_fac_rem(ax, ay)
    ff = np.isfinite(fac_dtw)
    funnel["dropped_nonfinite_fac_rem"] = int((~ff).sum())
    ax, ay, a_dtw, az, fac_dtw = ax[ff], ay[ff], a_dtw[ff], az[ff], fac_dtw[ff]

    # --- HUC8 / HUC12 unit / fold from the base unit->fold map -----------------
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(ax, ay), crs=5070)
    h8 = gpd.read_parquet(HUC8_PATH)[["huc8", "geometry"]]
    if h8.crs is None or h8.crs.to_epsg() != 5070:
        raise SystemExit(f"HUC8 polys not EPSG:5070: {HUC8_PATH}")
    j = gpd.sjoin(pts, h8, predicate="within", how="left")
    j = j[~j.index.duplicated(keep="first")].sort_index()
    huc8 = j["huc8"].to_numpy(object)
    fh = ~pd.isna(huc8)
    funnel["dropped_no_huc8"] = int((~fh).sum())
    ax, ay, a_dtw, az, fac_dtw, huc8 = (
        ax[fh],
        ay[fh],
        a_dtw[fh],
        az[fh],
        fac_dtw[fh],
        huc8[fh].astype(str),
    )
    n_aux = len(ax)
    cv_unit = B.huc12_units(ax, ay, HUC12_PATH, man["cv_block_km"])
    unit_fold = (
        mon[["cv_unit", "cv_fold"]].drop_duplicates().set_index("cv_unit")["cv_fold"]
    )
    if unit_fold.index.has_duplicates:
        raise SystemExit("base cv_unit -> cv_fold map is not unique; investigate")
    cv_unit = np.asarray(cv_unit)
    afold = pd.Series(cv_unit).map(unit_fold).to_numpy()
    novel = pd.isna(afold)
    funnel["rows_in_novel_huc12_units"] = int(novel.sum())
    funnel["novel_huc12_units"] = int(pd.unique(cv_unit[novel]).size)
    if novel.any():
        afold[novel] = novel_unit_folds(
            cv_unit[novel], np.c_[ax[novel], ay[novel]], np.c_[mx, my], mfold
        )
    afold = afold.astype("int64")
    a_wte = az - a_dtw

    # --- R + deep datum: crossfit on [base real U new], pool = base real --------
    cxy = np.c_[np.r_[mx, ax], np.r_[my, ay]]
    cz = np.r_[mz, az]
    cwte = np.r_[mwte, a_wte]
    cfold = np.r_[mfold, afold]
    pool = np.r_[np.ones(n_mon, bool), np.zeros(n_aux, bool)]
    r_all = B.crossfit_idw(cxy, cwte, cfold, idw_k, idw_p, z=cz, vw=vw, pool=pool)
    r_mon, r_aux = r_all[:n_mon], r_all[n_mon:]
    dr = np.abs(r_mon - mon["regional_wte_idw_oof_m"].to_numpy("float64"))
    print(f"R parity (base real): max|dR|={np.nanmax(dr):.6g} m")
    if np.nanmax(dr) > 1e-3:
        raise SystemExit(f"R parity FAILED: max|dR|={np.nanmax(dr):.4g} m")
    deep_mask = B.deep_well_mask(
        mon[["mean_dtw", "huc8"]].assign(huc8=mon["huc8"].astype(str)),
        dd["quantile"],
        dd["unit"],
        dd["min_per_unit"],
    )
    deep_all = B.crossfit_deep_idw(
        cxy,
        cxy[:n_mon][deep_mask],
        cwte[:n_mon][deep_mask],
        cfold,
        cfold[:n_mon][deep_mask],
        dd["idw_k_deep"],
        idw_p,
        z_all=cz,
        z_deep=cz[:n_mon][deep_mask],
        vw=vw,
    )
    deep_mon, deep_aux = deep_all[:n_mon], deep_all[n_mon:]
    dd_par = np.abs(deep_mon - mon["deep_regional_wte_idw_oof_m"].to_numpy("float64"))
    print(f"deep parity (base real): max|dD|={np.nanmax(dd_par):.6g} m")
    if np.nanmax(dd_par) > 1e-3:
        raise SystemExit(f"deep parity FAILED: max|dD|={np.nanmax(dd_par):.4g} m")
    # plain-DTW OOF IDW diagnostic (regional_idw_dtw_oof_m): same crossfit, no relief
    dtw_all = B.crossfit_idw(
        cxy,
        np.r_[mon["mean_dtw"].to_numpy("float64"), a_dtw],
        cfold,
        idw_k,
        idw_p,
        pool=pool,
    )

    axy = np.c_[ax, ay]
    out = pd.DataFrame(
        {
            "canonical_id": _ids(args.well_class, ax, ay),
            "source": np.full(n_aux, args.source_name, object),
            "is_nwis": np.zeros(n_aux, bool),
            "x5070": ax,
            "y5070": ay,
            "huc8": huc8,
            "huc4": np.array([h[:4] for h in huc8], object),
            "huc2": np.array([h[:2] for h in huc8], object),
            "mean_dtw": a_dtw,
            "z_surf_well_m": az,
            "wte_obs_m": a_wte,
            "regional_wte_idw_oof_m": r_aux,
            "deep_regional_wte_idw_oof_m": deep_aux,
            "regional_idw_dtw_oof_m": dtw_all[n_mon:],
            "wte_resid_base_m": az - r_aux,
            "wte_residual_m": a_wte - r_aux,
            "fac_rem_wte_anom_m": (az - fac_dtw) - r_aux,
            "deep_regional_wte_anom_m": deep_aux - r_aux,
            "fac_rem_dtw_m": fac_dtw,
            "cv_fold": afold,
            "cv_unit": cv_unit,
            "confinement_class": np.full(n_aux, "unknown", object),
            "well_class": np.full(n_aux, args.well_class, object),
            "is_water_pseudo": np.zeros(n_aux, bool),
        }
    )
    if out["canonical_id"].duplicated().any():
        raise SystemExit("duplicate canonical_id among new wells (collocated inputs)")
    bx = (ax // BLOCK_SIZE_M).astype("int64")
    by = (ay // BLOCK_SIZE_M).astype("int64")
    out["block_40km"] = np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))

    for col, vals in B.sample_relief_etrm(ax, ay, az).items():
        out[col] = vals
    for col, vals in B.sample_gridmet(ax, ay).items():
        out[col] = vals
    for col, vals in B.sample_terrain_multiscale(ax, ay).items():
        out[col] = vals
    from pyproj import Transformer

    lon_q, lat_q = Transformer.from_crs(5070, 4326, always_xy=True).transform(ax, ay)
    out["gsw_occ_pct"] = B._sample_gsw_occurrence(np.asarray(lon_q), np.asarray(lat_q))
    blocks = pd.read_parquet(bundle / "permanent_water_blocks.parquet")
    for col, vals in B.water_query_features(
        axy, az, blocks[["x5070", "y5070"]].to_numpy("float64"), DEM
    ).items():
        out[col] = vals
    for col, vals in B.sample_drilled_depth(
        DD_POINTS,
        axy,
        idw_k,
        idw_p,
        query_ids=out["canonical_id"].to_numpy(),
        self_exclude_m=DD_SELF_EXCLUDE_M,
    ).items():
        out[col] = vals
    bnd = B.build_boundaries(
        str(bundle / "reach_nodes.parquet"), GEOM, top_orders=dh["top_orders"]
    )
    hang_wte, d_bnd = B.hang_interp(
        bnd[["cx", "cy"]].to_numpy("float64"),
        bnd["reach_elev_m"].to_numpy("float64"),
        axy,
        k=dh["idw_k"],
        power=dh["idw_power"],
    )
    out["dupuit_hang_dtw_m"] = az - hang_wte
    out["log1p_dupuit_d_m"] = np.log1p(d_bnd)

    qfc = man["query_feature_cols"]
    base_nan = {c for c in qfc if qn[c].isna().any()}
    new_nan = sorted({c for c in qfc if out[c].isna().any()} - base_nan)
    bad = np.zeros(n_aux, bool)
    for c in new_nan:
        bad |= ~np.isfinite(out[c].to_numpy("float64"))
    funnel["dropped_new_nan_feature"] = int(bad.sum())
    if bad.any():
        print(f"dropping {int(bad.sum()):,} rows w/ NaN in base-finite cols {new_nan}")
        out = out[~bad].reset_index(drop=True)
    n_aux = len(out)
    axy = out[["x5070", "y5070"]].to_numpy("float64")
    az = out["z_surf_well_m"].to_numpy("float64")

    for c in qn.columns:
        if c not in out.columns:
            out[c] = np.nan
    out = out[list(qn.columns)]
    out["query_node_idx"] = np.arange(base_n, base_n + n_aux, dtype="int64")
    for c in qn.columns:
        try:
            out[c] = out[c].astype(qn[c].dtype)
        except (TypeError, ValueError):
            pass  # object/float promotion of all-NaN diagnostics is fine

    reach_nodes = pd.read_parquet(
        bundle / "reach_nodes.parquet",
        columns=[
            "reach_node_idx",
            "comid",
            "log1p_totda_km2",
            "streamorde",
            "reach_elev_m",
            "totdasqkm",
        ],
    )
    comid_to_idx = dict(
        zip(
            reach_nodes["comid"].to_numpy("int64"),
            reach_nodes["reach_node_idx"].to_numpy("int64"),
        )
    )
    geom = gpd.read_parquet(GEOM)
    geom["reach_node_idx"] = geom["comid"].map(comid_to_idx)
    geom = geom[geom["reach_node_idx"].notna()].copy()
    geom["reach_node_idx"] = geom["reach_node_idx"].astype("int64")
    lat = B.build_lateral_edges(axy, geom, comid_to_idx, knn_lat)
    lat = B.attach_lateral_attrs(lat, reach_nodes, az, cond_p)
    lat["query_node_idx"] = lat["query_node_idx"].to_numpy("int64") + base_n
    le = pd.read_parquet(bundle / "lateral_edges.parquet")
    le_all = pd.concat([le, lat[le.columns]], ignore_index=True)

    qn_all = pd.concat([qn, out], ignore_index=True)
    assert (qn_all["query_node_idx"].to_numpy() == np.arange(len(qn_all))).all()
    pd.testing.assert_frame_equal(qn_all.iloc[:base_n].reset_index(drop=True), qn)

    for f in SYMLINK:
        dst = out_dir / f
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(os.path.realpath(bundle / f), dst)
    qn_all.to_parquet(out_dir / "query_nodes.parquet", index=False)
    le_all.to_parquet(out_dir / "lateral_edges.parquet", index=False)
    man2 = json.loads(json.dumps(man))
    man2["counts"]["query_nodes"] = int(len(qn_all))
    man2["counts"]["query_nodes_non_nwis"] = int((~qn_all["is_nwis"]).sum())
    man2["counts"]["lateral_edges"] = int(len(le_all))
    man2["well_class_filter"] = list(man.get("well_class_filter", [])) + [
        args.well_class
    ]
    man2["appended_wells"] = {
        "base_bundle": str(bundle),
        "wells": args.wells,
        "source_name": args.source_name,
        "well_class": args.well_class,
        "label_weight": 1.0,
        "n_appended": int(n_aux),
        "base_n": int(base_n),
        "funnel": funnel,
        "fold_counts": {
            int(k): int(v)
            for k, v in out["cv_fold"].value_counts().sort_index().items()
        },
        "r_contract": "crossfit relief-IDW on [base real U new], pool=base real; "
        "base R parity <= 1e-3 m",
    }
    (out_dir / "graph_manifest.json").write_text(json.dumps(man2, indent=1))

    funnel["FINAL_appended"] = n_aux
    print("\n=== APPEND FUNNEL ===")
    for k, v in funnel.items():
        print(f"  {k}: {v:,}")
    print(f"total query nodes: {len(qn_all):,} | lateral edges: {len(le_all):,}")
    print("fold distribution:", man2["appended_wells"]["fold_counts"])
    print("new-row feature finite frac:")
    for c in qfc:
        print(f"  {c}: {np.isfinite(out[c].to_numpy('float64')).mean():.3f}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
