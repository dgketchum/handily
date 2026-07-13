"""Track A of the w25_prod GNN vs Ma vs Janssen NM eval: leak-free OOF panel,
sliced to one state.

Reuses ``score_conus_gnn``'s metric functions verbatim (never MAD alone: core
metrics, depth bands, shallow P/R, per-HUC2) so the state sub-panel is identical
in construction to the CONUS headline -- only the well population changes. The
CONUS scorer + its committed headline are left untouched.

State label: the OOF parquet carries no ``state`` column, and its ``canonical_id``
is an older 8-hex-truncated hash that does NOT join to the current GWX index. So
wells are labeled by an EXACT 5070-coordinate join to the GWX national index
(round-to-1 m key; verified 34,503/34,503 OOF wells match). The current index's
``confinement_class`` is carried too, to re-apply the unconfined screen on the
present labels.

    uv run python utils/score_conus_gnn_state_slice.py \
        --gnn-dir /data/ssd2/handily/conus/wte_gnn/gnn_conus_monitoring_water_gate_mirror_sigma_w25_prod \
        --state NM --streams /path/nm_streams_merged.fgb \
        --out-dir <model>/nm_eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from score_conus_gnn import (
    discover_ma_specs,
    fit_hand_cal_oof,
    full_panel,
    log_panel,
    sample_ma,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("score_state_slice")

GWX_INDEX = "/data/ssd2/gwx/products/current/wells.geoparquet"
WT_CLASSES = ("unconfined", "unconfined_marginal")


def coord_key(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact 1 m 5070 key -- OOF wells were snapped from the same GWX coords."""
    xi = np.round(np.asarray(x, dtype="float64")).astype("int64")
    yi = np.round(np.asarray(y, dtype="float64")).astype("int64")
    return np.char.add(np.char.add(xi.astype(str), "_"), yi.astype(str))


def attach_state(df: pd.DataFrame, gwx_index: str) -> pd.DataFrame:
    """Bring GWX ``state`` + current ``confinement_class`` onto OOF wells by coord."""
    gwx = pd.read_parquet(
        gwx_index, columns=["longitude", "latitude", "state", "confinement_class"]
    )
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    gx, gy = tr.transform(gwx["longitude"].to_numpy(), gwx["latitude"].to_numpy())
    gwx["k"] = coord_key(gx, gy)
    gwx = gwx.drop_duplicates("k").set_index("k")
    k = coord_key(df["x5070"].to_numpy(), df["y5070"].to_numpy())
    df = df.copy()
    df["state"] = pd.Series(k, index=df.index).map(gwx["state"])
    df["confinement_now"] = pd.Series(k, index=df.index).map(gwx["confinement_class"])
    return df


def tag_valley(df: pd.DataFrame, streams_path: str, dist_m: float) -> np.ndarray:
    """valley (<=dist_m to nearest FAC stream) vs upland, for the setting split."""
    import geopandas as gpd

    streams = gpd.read_file(streams_path).to_crs(5070)
    pts = gpd.GeoDataFrame(
        df[["x5070", "y5070"]],
        geometry=gpd.points_from_xy(df["x5070"], df["y5070"]),
        crs=5070,
    )
    near = gpd.sjoin_nearest(
        pts[["geometry"]], streams[["geometry"]], distance_col="_d"
    )
    near = near[~near.index.duplicated(keep="first")]
    d = near["_d"].reindex(pts.index).to_numpy()
    return np.where(d <= dist_m, "valley", "upland")


def band_table(panel: dict, predcols: list[str]) -> None:
    """Log the full depth-band structure (MAD/medR/bias/RMSE/p95) for each predictor."""
    for c in predcols:
        bands = panel["predictors"][c]["by_depth_band"]
        log.info("  -- %s by obs-depth --", c)
        for b, v in bands.items():
            if not v.get("n"):
                continue
            log.info(
                "     %-8s n=%-5d MAD=%5.2f medR=%+6.2f bias=%+7.2f RMSE=%6.2f p95=%6.2f",
                b,
                v["n"],
                v["mad_m"],
                v["median_resid_m"],
                v["bias_mean_m"],
                v["rmse_m"],
                v["p95_abs_err_m"],
            )


def shallow_table(panel: dict, predcols: list[str]) -> None:
    for c in predcols:
        sk = panel["predictors"][c]["shallow_skill"]
        cells = " ".join(
            f"{thr}:P{v['precision']:.2f}/R{v['recall']:.2f}" for thr, v in sk.items()
        )
        log.info("  shallow P/R %-12s %s", c, cells)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gnn-dir", required=True)
    ap.add_argument("--state", default="NM")
    ap.add_argument("--gwx-index", default=GWX_INDEX)
    ap.add_argument("--streams", default=None, help="merged FAC streams for valley tag")
    ap.add_argument("--valley-dist-m", type=float, default=500.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ma-only-state", default=None, help="e.g. new_mexico (speed)")
    args = ap.parse_args()

    gdir = Path(args.gnn_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(gdir / "gnn_oof_predictions.parquet")
    df["huc2"] = df["huc2"].astype(str).str.zfill(2)
    if "is_water_pseudo" in df.columns:
        n0 = len(df)
        df = df[~df["is_water_pseudo"].astype(bool)].reset_index(drop=True)
        log.info("dropped %d water pseudo-rows", n0 - len(df))

    df = attach_state(df, args.gwx_index)
    n_state = int((df["state"] == args.state).sum())
    log.info("%s OOF wells (by GWX coord-join): %d", args.state, n_state)

    # predictor columns -- identical construction to score_conus_gnn.main()
    df["gnn"] = df["gnn_dtw_m"]
    df["regional"] = df["regional_idw_dtw_oof_m"]
    df["janssen"] = df["janssen_dtw_m"]
    df["hand_cal"] = fit_hand_cal_oof(df)
    predcols = ["gnn", "regional", "janssen", "hand_cal"]
    common = ["gnn", "regional", "janssen", "hand_cal"]
    if "regional_deep_idw_dtw_oof_m" in df.columns:
        df["regional_deep"] = df["regional_deep_idw_dtw_oof_m"]
        predcols.append("regional_deep")
    if {"fac_rem_wte_m", "z_surf_well_m"}.issubset(df.columns):
        df["fac_rem"] = df["z_surf_well_m"] - df["fac_rem_wte_m"]
        predcols.append("fac_rem")

    if args.ma_only_state:
        ma_specs = [
            f"Ma_{args.ma_only_state}=/nas/gwx/wtd_states/wtd_{args.ma_only_state}.tif"
        ]
    else:
        ma_specs = discover_ma_specs()
    df["ma"] = sample_ma(df, ma_specs)
    predcols.append("ma")

    # slice to state, re-apply the CURRENT-index unconfined screen (doctrine)
    st = df[df["state"] == args.state].reset_index(drop=True)
    keep_conf = st["confinement_now"].isin(WT_CLASSES)
    n_drop = int((~keep_conf).sum())
    if n_drop:
        log.info(
            "dropping %d %s wells now labeled non-unconfined (%s)",
            n_drop,
            args.state,
            dict(st.loc[~keep_conf, "confinement_now"].value_counts()),
        )
    st = st[keep_conf].reset_index(drop=True)
    ma_cov = float(np.isfinite(st["ma"].to_numpy("float64")).mean())
    log.info("%s Ma coverage: %.1f%%", args.state, 100 * ma_cov)

    non_nwis = st[~st["is_nwis"]].reset_index(drop=True)
    nwis = st[st["is_nwis"]].reset_index(drop=True)
    log.info(
        "%s unconfined: %d total (%d non-NWIS headline, %d NWIS diagnostic)",
        args.state,
        len(st),
        len(non_nwis),
        len(nwis),
    )

    panels = {}
    headline = full_panel(non_nwis, predcols, "obs_dtw_m", common)
    log_panel(f"{args.state} HEADLINE -- non-NWIS unconfined", headline, predcols)
    band_table(headline, predcols)
    shallow_table(headline, predcols)
    panels["headline_non_nwis"] = headline

    if len(nwis) >= 25:
        nwis_panel = full_panel(nwis, predcols, "obs_dtw_m", common)
        log_panel(f"{args.state} NWIS (diagnostic; Ma-leaked)", nwis_panel, predcols)
        panels["nwis_panel"] = nwis_panel

    # valley/upland setting split on the leak-free non-NWIS set
    if args.streams:
        setting = tag_valley(non_nwis, args.streams, args.valley_dist_m)
        for s in ("valley", "upland"):
            sub = non_nwis[setting == s].reset_index(drop=True)
            if len(sub) >= 25:
                p = full_panel(sub, predcols, "obs_dtw_m", common)
                log_panel(f"{args.state} {s} (non-NWIS)", p, predcols)
                band_table(p, predcols)
                panels[f"setting_{s}"] = p
            else:
                log.info("%s %s: n=%d < 25, skipped", args.state, s, len(sub))

    # HUC2 sub-panels (Rio Grande 13 = valley corridor; 12 = Pecos/Texas-Gulf; 15/14 = Colorado)
    by_huc2 = {}
    for h2, sub in non_nwis.groupby("huc2"):
        if len(sub) < 25:
            continue
        p = full_panel(sub.reset_index(drop=True), predcols, "obs_dtw_m", common)
        log_panel(f"{args.state} HUC2={h2} (non-NWIS)", p, predcols)
        by_huc2[str(h2)] = p
    panels["by_huc2"] = by_huc2

    summary = {
        "state": args.state,
        "gnn_dir": str(gdir),
        "predictors": predcols,
        "common_footprint_predictors": common,
        "n_state_unconfined": int(len(st)),
        "n_non_nwis": int(len(non_nwis)),
        "n_nwis": int(len(nwis)),
        "ma_coverage": ma_cov,
        "panels": panels,
    }
    out_json = out_dir / f"{args.state.lower()}_oof_panel.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote %s", out_json)


if __name__ == "__main__":
    main()
