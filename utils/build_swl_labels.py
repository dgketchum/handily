"""E4: build SWL auxiliary training-label query nodes + lateral edges.

Additive to `graph_conus_monitoring_water_v2/` -- the base bundle is NEVER modified.
Reuses the exact builder machinery of `build_conus_graph_inputs.py` so the SWL aux
points land in the identical feature/target space as the monitoring anchors:

  * z_surf via sample_coarse(DEM); target space wte_residual = (z_surf - mean_dtw) - R.
  * R = relief-aware leave-one-fold-out well-IDW of MONITORING observed WTE
    (crossfit_idw, k=32, power=2, vw=100, pool=monitoring-only), the SWL point's own
    fold held out -- the IDENTICAL procedure that built the anchors' R. Recomputing R
    on [monitoring U aux] with pool=monitoring reproduces the anchors' archived R
    bit-for-bit (parity-checked here) and gives each aux point a leak-safe R.
  * deep_regional_wte_anom via crossfit_deep_idw from the SAME monitoring deep pool.
  * all 34 query features (relief/etrm/gridmet/fac_rem/terrain-multiscale/water/
    drilled-depth/dupuit) sampled by the same functions as the base bundle.
  * lateral edges via build_lateral_edges(knn=3) + attach_lateral_attrs.

Footprint contract: aux must carry a finite z_surf (hard: the target cannot be
reconstructed otherwise) AND a finite fac_rem_dtw_m -- MIRRORING the base bundle's own
--require-fac contract, so aux live in the identical footprint + feature schema as the
anchors (no new NaN-indicator columns -> the trainer's query matrix width is unchanged,
keeping the marginal-vs-E5 comparison architecturally clean). Both drops are reported.

Outputs (standalone, additive):
  e4_swl_labels.parquet         SWL query rows (is_swl_aux=True), all base qn columns.
  e4_swl_lateral_edges.parquet  aux lateral edges, query_node_idx local 0..n-1
                                (the trainer offsets by the base query count on append).
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_conus_graph_inputs as B  # noqa: E402

BUNDLE = Path("/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2")
POOL = "/data/ssd2/handily/conus/wte_gnn/e4_swl_pool_folded.parquet"
OUT_Q = "/data/ssd2/handily/conus/wte_gnn/e4_swl_labels.parquet"
OUT_L = "/data/ssd2/handily/conus/wte_gnn/e4_swl_lateral_edges.parquet"

DEM = B.DEM
GEOM = "/data/ssd2/handily/conus/wte_gnn/fac_flowline_geom_conus.parquet"
DD_POINTS = "/data/ssd2/handily/conus/wte_gnn/drilled_depth_points.parquet"

# params locked to the bundle manifest (graph_conus_monitoring_water_v2)
IDW_K, IDW_POWER, R_RELIEF_VW = 32, 2.0, 100.0
DEEP_Q, DEEP_UNIT, DEEP_MIN, IDW_K_DEEP = 0.75, "huc6", 30, 16
KNN_LATERAL, CONDUCTANCE_P = 3, 0.5
DD_SELF_EXCLUDE_M = 100.0
BLOCK_SIZE_M = 40000.0
DUPUIT_TOP_ORDERS, DUPUIT_K, DUPUIT_POWER = 2, 8, 2.0


def main() -> None:
    funnel: dict[str, int] = {}
    aux = pd.read_parquet(POOL)
    funnel["swl_pool_folded"] = len(aux)
    ax = aux["x5070"].to_numpy("float64")
    ay = aux["y5070"].to_numpy("float64")

    # --- monitoring anchors (bundle real query nodes) ---------------------------
    qn = pd.read_parquet(
        f"{BUNDLE}/query_nodes.parquet",
        columns=[
            "x5070",
            "y5070",
            "z_surf_well_m",
            "wte_obs_m",
            "mean_dtw",
            "huc8",
            "cv_fold",
            "is_water_pseudo",
            "regional_wte_idw_oof_m",
            "deep_regional_wte_idw_oof_m",
        ],
    )
    mon = qn[~qn["is_water_pseudo"]].reset_index(drop=True)
    mx = mon["x5070"].to_numpy("float64")
    my = mon["y5070"].to_numpy("float64")
    mz = mon["z_surf_well_m"].to_numpy("float64")
    mwte = mon["wte_obs_m"].to_numpy("float64")
    mfold = mon["cv_fold"].to_numpy()
    n_mon = len(mon)
    print(f"monitoring anchors: {n_mon:,}")

    # --- z_surf (DEM) for aux; drop non-finite (target cannot reconstruct) ------
    az = B.sample_coarse(DEM, ax, ay)
    finite_z = np.isfinite(az)
    funnel["dropped_nonfinite_z_surf"] = int((~finite_z).sum())
    aux, ax, ay, az = (
        aux[finite_z].reset_index(drop=True),
        ax[finite_z],
        ay[finite_z],
        az[finite_z],
    )

    # --- fac_rem_dtw (require finite -- mirrors the base --require-fac contract) -
    fac_dtw = B.sample_fac_rem(ax, ay)
    finite_fac = np.isfinite(fac_dtw)
    funnel["dropped_nonfinite_fac_rem"] = int((~finite_fac).sum())
    print(
        f"fac_rem finite frac (pre-drop): {finite_fac.mean():.4f} "
        f"({int(finite_fac.sum()):,}/{len(finite_fac):,})"
    )
    aux, ax, ay, az, fac_dtw = (
        aux[finite_fac].reset_index(drop=True),
        ax[finite_fac],
        ay[finite_fac],
        az[finite_fac],
        fac_dtw[finite_fac],
    )
    n_aux = len(aux)
    afold = aux["cv_fold"].to_numpy()
    a_dtw = aux["mean_dtw"].to_numpy("float64")
    a_wte = az - a_dtw  # observed head at the aux SWL point

    # --- R: relief-aware crossfit well-IDW on [monitoring U aux], pool=monitoring
    # For a monitoring well this reproduces the archived R exactly (aux excluded by
    # pool); for an aux point R comes from monitoring wells outside its own fold.
    cxy = np.c_[np.r_[mx, ax], np.r_[my, ay]]
    cz = np.r_[mz, az]
    cwte = np.r_[mwte, a_wte]
    cfold = np.r_[mfold, afold]
    pool_mask = np.r_[np.ones(n_mon, bool), np.zeros(n_aux, bool)]
    r_all = B.crossfit_idw(
        cxy, cwte, cfold, IDW_K, IDW_POWER, z=cz, vw=R_RELIEF_VW, pool=pool_mask
    )
    r_mon, r_aux = r_all[:n_mon], r_all[n_mon:]
    dr = np.abs(r_mon - mon["regional_wte_idw_oof_m"].to_numpy("float64"))
    print(
        f"R parity (monitoring): max|dR|={np.nanmax(dr):.6g} m  "
        f"median|dR|={np.nanmedian(dr):.6g} m"
    )
    if np.nanmax(dr) > 1e-3:
        raise SystemExit(
            f"R parity FAILED: max|dR|={np.nanmax(dr):.4g} m (>1e-3) -- the aux R "
            "crossfit does not reproduce the anchor R contract; investigate"
        )

    # --- deep regional WTE anomaly: same monitoring deep pool -------------------
    deep_mask = B.deep_well_mask(
        mon[["mean_dtw", "huc8"]].assign(huc8=mon["huc8"].astype(str)),
        DEEP_Q,
        DEEP_UNIT,
        DEEP_MIN,
    )
    deep_all = B.crossfit_deep_idw(
        cxy,
        cxy[:n_mon][deep_mask],
        cwte[:n_mon][deep_mask],
        cfold,
        cfold[:n_mon][deep_mask],
        IDW_K_DEEP,
        IDW_POWER,
        z_all=cz,
        z_deep=cz[:n_mon][deep_mask],
        vw=R_RELIEF_VW,
    )
    deep_mon, deep_aux = deep_all[:n_mon], deep_all[n_mon:]
    dd = np.abs(deep_mon - mon["deep_regional_wte_idw_oof_m"].to_numpy("float64"))
    print(
        f"deep parity (monitoring): max|dD|={np.nanmax(dd):.6g} m  "
        f"median|dD|={np.nanmedian(dd):.6g} m  (deep pool n={int(deep_mask.sum()):,})"
    )
    if np.nanmax(dd) > 1e-3:
        raise SystemExit(
            f"deep parity FAILED: max|dD|={np.nanmax(dd):.4g} m (>1e-3); investigate"
        )

    # --- target space (identical anchor definitions) ---------------------------
    axy = np.c_[ax, ay]
    fac_wte = az - fac_dtw
    out = pd.DataFrame(
        {
            "canonical_id": aux["canonical_id"].to_numpy(),
            "source": aux["source"].to_numpy(),
            "is_nwis": np.zeros(n_aux, bool),
            "x5070": ax,
            "y5070": ay,
            "huc8": aux["huc8"].astype(str).to_numpy(),
            "huc4": aux["huc4"].astype(str).to_numpy(),
            "huc2": aux["huc2"].astype(str).to_numpy(),
            "mean_dtw": a_dtw,
            "z_surf_well_m": az,
            "wte_obs_m": a_wte,
            "regional_wte_idw_oof_m": r_aux,
            "deep_regional_wte_idw_oof_m": deep_aux,
            "wte_resid_base_m": az - r_aux,
            "wte_residual_m": a_wte - r_aux,
            "fac_rem_wte_anom_m": fac_wte - r_aux,
            "deep_regional_wte_anom_m": deep_aux - r_aux,
            "fac_rem_dtw_m": fac_dtw,
            "cv_fold": afold.astype("int64"),
            "cv_unit": aux["cv_unit"].to_numpy(),
            "confinement_class": aux["confinement_class"].to_numpy(),
            "well_depth": aux["well_depth"].to_numpy("float64"),
            "well_class": np.full(n_aux, "swl_aux", object),
            "is_water_pseudo": np.zeros(n_aux, bool),
            "is_swl_aux": np.ones(n_aux, bool),
        }
    )
    bx = (ax // BLOCK_SIZE_M).astype("int64")
    by = (ay // BLOCK_SIZE_M).astype("int64")
    out["block_40km"] = np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))

    # --- 34 query features (same samplers as the base bundle) -------------------
    for col, vals in B.sample_relief_etrm(ax, ay, az).items():
        out[col] = vals
    for col, vals in B.sample_gridmet(ax, ay).items():
        out[col] = vals
    for col, vals in B.sample_terrain_multiscale(ax, ay).items():
        out[col] = vals
    # water context: gsw occ (4326) + dist/height above permanent-water blocks
    from pyproj import Transformer

    lon_q, lat_q = Transformer.from_crs(5070, 4326, always_xy=True).transform(ax, ay)
    out["gsw_occ_pct"] = B._sample_gsw_occurrence(np.asarray(lon_q), np.asarray(lat_q))
    blocks = pd.read_parquet(f"{BUNDLE}/permanent_water_blocks.parquet")
    for col, vals in B.water_query_features(
        axy, az, blocks[["x5070", "y5070"]].to_numpy("float64"), DEM
    ).items():
        out[col] = vals
    # drilled-depth field (self/nest excluded at 100 m -- construction record, not a
    # water level; public at any point, so no fold crossfit)
    for col, vals in B.sample_drilled_depth(
        DD_POINTS,
        axy,
        IDW_K,
        IDW_POWER,
        query_ids=aux["canonical_id"].to_numpy(),
        self_exclude_m=DD_SELF_EXCLUDE_M,
    ).items():
        out[col] = vals
    # dupuit hang (well-free boundary-conditioned surface)
    bnd = B.build_boundaries(
        str(BUNDLE / "reach_nodes.parquet"), GEOM, top_orders=DUPUIT_TOP_ORDERS
    )
    hang_wte, d_bnd = B.hang_interp(
        bnd[["cx", "cy"]].to_numpy("float64"),
        bnd["reach_elev_m"].to_numpy("float64"),
        axy,
        k=DUPUIT_K,
        power=DUPUIT_POWER,
    )
    out["dupuit_hang_dtw_m"] = az - hang_wte
    out["log1p_dupuit_d_m"] = np.log1p(d_bnd)

    # --- feature-schema guard: aux must not introduce NEW NaN columns vs base ---
    man = __import__("json").loads((BUNDLE / "graph_manifest.json").read_text())
    qfc = man["query_feature_cols"]
    base_feat = pd.read_parquet(f"{BUNDLE}/query_nodes.parquet", columns=qfc)
    base_nan = {c for c in qfc if base_feat[c].isna().any()}
    aux_nan = {c for c in qfc if out[c].isna().any()}
    new_nan = sorted(aux_nan - base_nan)
    if new_nan:
        # any aux row NaN in a base-all-finite feature -> drop it (report), so the
        # combined NaN-column set == base and the query matrix width is unchanged.
        bad = np.zeros(n_aux, bool)
        for c in new_nan:
            bad |= ~np.isfinite(out[c].to_numpy("float64"))
        funnel["dropped_new_nan_feature"] = int(bad.sum())
        print(
            f"dropping {int(bad.sum()):,} aux rows w/ NaN in base-finite cols {new_nan}"
        )
        out = out[~bad].reset_index(drop=True)
        axy = out[["x5070", "y5070"]].to_numpy("float64")
        az = out["z_surf_well_m"].to_numpy("float64")
        n_aux = len(out)
    else:
        funnel["dropped_new_nan_feature"] = 0
    out["query_node_idx"] = np.arange(n_aux, dtype="int64")

    # column order: exactly the base query_nodes schema + is_swl_aux (additive) ---
    base_cols = list(pd.read_parquet(f"{BUNDLE}/query_nodes.parquet").columns)
    for c in base_cols:
        if c not in out.columns:
            out[c] = np.nan  # diagnostics we do not compute for aux (never scored)
    out = out[base_cols + ["is_swl_aux"]]

    # --- lateral edges (query_node_idx local 0..n-1) ----------------------------
    reach_nodes = pd.read_parquet(
        f"{BUNDLE}/reach_nodes.parquet",
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
    lat = B.build_lateral_edges(axy, geom, comid_to_idx, KNN_LATERAL)
    lat = B.attach_lateral_attrs(lat, reach_nodes, az, CONDUCTANCE_P)

    out.to_parquet(OUT_Q, index=False)
    lat.to_parquet(OUT_L, index=False)

    funnel["FINAL_swl_labels"] = n_aux
    print("\n=== E4 SWL LABEL BUILD FUNNEL ===")
    for k, v in funnel.items():
        print(f"  {k}: {v:,}")
    nv = int((out["huc2"] == "16").sum())
    wa = int(out["huc2"].isin(["13", "15", "16"]).sum())
    print(
        f"\nFINAL: {n_aux:,} SWL labels across {out['huc8'].nunique():,} HUC8s / "
        f"{out['huc4'].nunique():,} HUC4s / {out['huc2'].nunique()} HUC2s"
    )
    print(f"  NV (HUC2 16): {nv:,} | wide-arid (13/15/16): {wa:,}")
    print(f"  lateral edges: {len(lat):,} ({len(lat) / max(n_aux, 1):.2f}/label)")
    print(
        "  fold distribution:",
        dict(pd.Series(out["cv_fold"]).value_counts().sort_index()),
    )
    print("  query-feature finite frac (aux):")
    for c in qfc:
        print(f"    {c}: {np.isfinite(out[c].to_numpy('float64')).mean():.4f}")
    print(f"\nwrote {OUT_Q}\nwrote {OUT_L}")


if __name__ == "__main__":
    main()
