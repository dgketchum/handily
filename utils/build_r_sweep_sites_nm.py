"""NM scoring sites for the pumping-well R density sweep (utils/sweep_pumping_r.py).

Two site sets, one schema (site_id, x5070, y5070, z_m, obs_dtw_m, block, r_ref_m,
r_ref_cap_m, plus context columns):

  ose         the OSE scoring half: ``is_admission_eligible`` rows of
              ose_admission_targets.parquet (10,035 sites). Reference R =
              ``r_wte_m``, the crossfit R interpolated to the site in the render
              lattice frame (what handily sees there at inference), uncapped;
              capped = min(R, z - 2 m), the inference ``--r-ground-cap-m 2``.
              Context: ``ma_dtw_m`` (Ma DTW at the site, m).
  monitoring  monitoring wells of the NM OSE parity bundle inside the 97 NM
              HUC8s (state appended sources excluded). Reference R = the base
              bundle's uncapped crossfit R; capped = the parity bundle column,
              verified equal to min(base R, z - 2 m).

z_m is the 100 m DEM surface (the bundle's shared datum) and obs WTE = z_m -
obs_dtw_m. block = HUC8, the bootstrap block of the NM panel scorer.

Also writes ``stale_ose_hand_ids.parquet``: HAND-table nm_ose donor records whose
id matches no record of the current GWX nm_ose build. The HAND table mixes GWX
builds; current-build records sit exactly on their GWX position (funnel wells
among them fall to the 10 m exclusion), but the stale ones cannot be tied to a
POD, and about 40% of those 10-100 m from a funnel site share its DTW within
0.5 m. Dropping them all keeps duplicate funnel wells out of the donor pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TARGETS = "/data/ssd2/handily/nm/regional/wells/ose_admission_targets.parquet"
BUNDLE = "/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2_ndwr_ufold_ose_parity"
BASE_BUNDLE = "/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2"
HUC8_LIST = "/data/ssd2/handily/nm/regional/ose_lbl/nm_huc8_97.txt"
OUT = "/data/ssd2/handily/nm/regional/r_density_sweep/inputs"
HAND = "/data/ssd2/handily/conus/wte_gnn/conus_wells_hand.parquet"
OSE_GWX = (
    "/data/ssd2/handily/nm/regional/wells/nm_ose_gwx_20260919T2220Z-78ad348.parquet"
)
CAP_M = 2.0


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--targets", default=TARGETS)
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--base-bundle", default=BASE_BUNDLE)
    ap.add_argument("--huc8-list", default=HUC8_LIST)
    ap.add_argument("--hand", default=HAND)
    ap.add_argument("--ose-gwx", default=OSE_GWX, help="current GWX nm_ose build")
    ap.add_argument("--out-dir", default=OUT)
    return ap.parse_args()


def ose_sites(path: str) -> pd.DataFrame:
    t = pd.read_parquet(path)
    t = t[t["is_admission_eligible"].astype(bool)].reset_index(drop=True)
    z = t["z_surf_m"].to_numpy("float64")
    r = t["r_wte_m"].to_numpy("float64")
    return pd.DataFrame(
        {
            "site_id": t.index.map(lambda i: f"ose_{i:05d}"),
            "x5070": t["x5070"].to_numpy("float64"),
            "y5070": t["y5070"].to_numpy("float64"),
            "z_m": z,
            "obs_dtw_m": t["obs_dtw_m"].to_numpy("float64"),
            "block": t["huc8"].astype(str).to_numpy(),
            "r_ref_m": r,
            "r_ref_cap_m": np.minimum(r, z - CAP_M),
            "ma_dtw_m": t["ma_dtw_m"].to_numpy("float64"),
            "pod_basin": t["pod_basin"].astype(str).to_numpy(),
            "source": "nm_ose",
        }
    )


def monitoring_sites(bundle: str, base_bundle: str, huc8_list: str) -> pd.DataFrame:
    qn = pd.read_parquet(f"{bundle}/query_nodes.parquet")
    hucs = {s.strip() for s in open(huc8_list) if s.strip()}
    man = json.load(open(f"{bundle}/graph_manifest.json"))
    appended = {"nv_ndwr", man["appended_wells"]["source_name"]}
    m = qn[
        qn["huc8"].astype(str).isin(hucs)
        & (qn["well_class"] == "monitoring")
        & ~qn["is_water_pseudo"].astype(bool)
        & ~qn["source"].isin(appended)
    ].reset_index(drop=True)
    base = pd.read_parquet(
        f"{base_bundle}/query_nodes.parquet",
        columns=["query_node_idx", "regional_wte_idw_oof_m"],
    ).set_index("query_node_idx")
    r = base["regional_wte_idw_oof_m"].reindex(m["query_node_idx"]).to_numpy("float64")
    z = m["z_surf_well_m"].to_numpy("float64")
    r_cap = m["regional_wte_idw_oof_m"].to_numpy("float64")
    gap = np.nanmax(np.abs(np.minimum(r, z - CAP_M) - r_cap))
    if not np.all(np.isfinite(r)) or gap > 1e-3:
        raise SystemExit(
            f"base R does not reproduce the parity capped R (max gap {gap})"
        )
    return pd.DataFrame(
        {
            "site_id": m["query_node_idx"].map(lambda i: f"qn_{i}"),
            "x5070": m["x5070"].to_numpy("float64"),
            "y5070": m["y5070"].to_numpy("float64"),
            "z_m": z,
            "obs_dtw_m": m["mean_dtw"].to_numpy("float64"),
            "block": m["huc8"].astype(str).to_numpy(),
            "r_ref_m": r,
            "r_ref_cap_m": r_cap,
            "ma_dtw_m": np.nan,
            "pod_basin": "",
            "source": m["source"].to_numpy(),
        }
    )


def stale_ose_ids(hand: str, ose_gwx: str) -> pd.DataFrame:
    """HAND nm_ose rows whose 12-character canonical id is no prefix of any
    current-build id (HAND ids are truncated GWX ids)."""
    cur = set(
        pq.read_table(ose_gwx, columns=["canonical_id"])["canonical_id"]
        .to_pandas()
        .str[:12]
    )
    h = pd.read_parquet(hand, columns=["canonical_id", "source"])
    h = h[h["source"] == "nm_ose"].drop_duplicates("canonical_id")
    stale = h[~h["canonical_id"].isin(cur)][["canonical_id"]].reset_index(drop=True)
    print(f"HAND nm_ose ids: {len(h)}; stale (no current-build match): {len(stale)}")
    return stale


def main() -> None:
    args = parse()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stale_ose_ids(args.hand, args.ose_gwx).to_parquet(
        out / "stale_ose_hand_ids.parquet", index=False
    )
    for name, df in (
        ("ose", ose_sites(args.targets)),
        ("monitoring", monitoring_sites(args.bundle, args.base_bundle, args.huc8_list)),
    ):
        bad = ~np.isfinite(df[["x5070", "y5070", "z_m", "obs_dtw_m", "r_ref_m"]]).all(1)
        if bad.any():
            raise SystemExit(
                f"{name}: {int(bad.sum())} sites with non-finite core values"
            )
        df.to_parquet(out / f"sites_{name}.parquet", index=False)
        print(
            f"{name}: {len(df)} sites, {df['block'].nunique()} HUC8 blocks, "
            f"obs DTW p50 {df['obs_dtw_m'].median():.1f} m, "
            f"<5 m {np.mean(df['obs_dtw_m'] < 5):.2f}, "
            f"ref cap binds {np.mean(df['r_ref_m'] > df['z_m'] - CAP_M):.3f}"
        )


if __name__ == "__main__":
    main()
