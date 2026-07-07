"""Drilled-depth point pool for the drilled-depth field prior.

Filters the GWX national well index to unconfined(+marginal) wells carrying a
sane positive drilled depth and writes a compact EPSG:5070 point parquet
consumed by ``build_conus_graph_inputs.py --drilled-depth-points`` (training
features, self-excluded) and later by the 10 m inference pipeline (all-well
field, no exclusion).

Drilled depth ("how deep do people have to drill here") is CONSTRUCTION
metadata, not an observed water level: it is a behavioral observation of the
deep regime that no raster covariate carries (the deep-bench finding -- every
independent geology/climate/flux covariate came out AUC 0.50-0.53). Because it
is deployment-available everywhere, it needs no fold cross-fit; only a query
well's OWN record (and co-located nest siblings) must be excluded downstream.

Depth sanity: per-source medians are meters everywhere (GWX converts at
ingest); the >1500 m tail is typo-class garbage (63 wells across 12 sources,
e.g. a 1,244,200 m ct_deep entry and a 30,480 m = 100,000 ft NWIS entry), so
rows outside (0, --max-depth-m] are dropped and counted in the provenance.

    uv run python utils/build_drilled_depth_points.py
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pyproj import Transformer

log = logging.getLogger(__name__)

GWX_WELLS = "/data/ssd2/gwx/products/current/wells.geoparquet"
OUT = "/data/ssd2/handily/conus/wte_gnn/drilled_depth_points.parquet"
UNCONFINED = ("unconfined", "unconfined_marginal")
COLS = [
    "canonical_id",
    "source",
    "longitude",
    "latitude",
    "well_depth",
    "confinement_class",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gwx-wells", default=GWX_WELLS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument(
        "--max-depth-m",
        type=float,
        default=1500.0,
        help="drop depths above this (typo-class entries; deepest plausible "
        "unconfined/marginal water wells sit well under this)",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.gwx_wells, columns=COLS)
    n0 = len(df)
    drops: dict[str, int] = {}

    m = df["confinement_class"].isin(UNCONFINED)
    drops["not_unconfined"] = int((~m).sum())
    df = df[m]

    m = df["longitude"].notna() & df["latitude"].notna()
    drops["missing_coords"] = int((~m).sum())
    df = df[m]

    # Out-of-range lon/lat: a known upstream GWX ingest bug leaves raw UTM 15N
    # coordinates in the longitude/latitude columns for a subset of mn_cwi rows
    # (eastings ~245-747 km, northings ~4.9-5.5 Mm); these project to inf in
    # EPSG:5070. Fix belongs in gwx ingest -- dropped and counted here, not
    # repaired (converting another product's coordinates would paper over it).
    m = df["longitude"].between(-180.0, 180.0) & df["latitude"].between(-90.0, 90.0)
    drops["invalid_lonlat"] = int((~m).sum())
    if drops["invalid_lonlat"]:
        log.warning(
            "dropping %d rows with out-of-range lon/lat (upstream gwx ingest bug); "
            "by source: %s",
            drops["invalid_lonlat"],
            df.loc[~m, "source"].value_counts().to_dict(),
        )
    df = df[m]

    m = df["well_depth"].notna() & (df["well_depth"] > 0)
    drops["missing_or_nonpositive_depth"] = int((~m).sum())
    df = df[m]

    m = df["well_depth"] <= args.max_depth_m
    drops[f"depth_gt_{args.max_depth_m:g}m"] = int((~m).sum())
    df = df[m]

    n_dup = int(df["canonical_id"].duplicated().sum())
    drops["duplicate_canonical_id"] = n_dup
    df = df.drop_duplicates("canonical_id")

    tf = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = tf.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    bad = ~(pd.Series(x).notna() & pd.Series(y).notna() & (abs(x) != float("inf")))
    if bad.any():
        raise SystemExit(
            f"{int(bad.sum())} rows still project non-finite after the lon/lat "
            "range screen -- investigate (do not patch)"
        )
    out = pd.DataFrame(
        {
            "canonical_id": df["canonical_id"].to_numpy(),
            "source": df["source"].to_numpy(),
            "x5070": x,
            "y5070": y,
            "drilled_depth_m": df["well_depth"].to_numpy("float64"),
        }
    )
    out_path = Path(args.out)
    out.to_parquet(out_path, index=False)

    prov = {
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gwx_wells": args.gwx_wells,
        "gwx_rows": n0,
        "pool_rows": len(out),
        "confinement_kept": list(UNCONFINED),
        "max_depth_m": args.max_depth_m,
        "drops": drops,
        "depth_m_quantiles": {
            q: float(out["drilled_depth_m"].quantile(q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        },
        "semantics": "well CONSTRUCTION depth (m), not an observed water level; "
        "deployment-available metadata -> no fold cross-fit needed, self/nest "
        "exclusion handled by the consumer",
    }
    out_path.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2))
    log.info("wrote %d points -> %s", len(out), out_path)
    log.info("drops: %s", drops)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
