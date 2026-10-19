"""Export the Nevada portion of the gwx well products for a Drive handoff.

Selection: wells whose WGS84 point falls inside the Nevada polygon of
``conus_states.parquet`` (strict state boundary, no buffer). Writes the subset
of ``wells.geoparquet`` (geoparquet metadata preserved), the inner-joined side
tables, every per-well time-series parquet the index points at, the product
provenance files, and a MANIFEST.json with counts per source and series files
found / missing. A missing file for a ``has_time_series`` row is reported, not
patched.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import shapely

log = logging.getLogger("export_gwx_nevada")

PRODUCTS = Path("/data/ssd2/gwx/products/current")
STATES = Path("/data/ssd2/gwx/nwis/reference/conus_states.parquet")
KEY = ["source", "source_well_id"]


def git_sha(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", type=Path, default=PRODUCTS)
    ap.add_argument("--states", type=Path, default=STATES)
    ap.add_argument("--state", default="NV")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gwx-repo", type=Path, default=Path.home() / "code" / "gwx")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    states = gpd.read_parquet(args.states)
    col = next(c for c in ("usps", "STUSPS", "STUSAB", "STATE_ABBR") if c in states)
    poly = states.loc[states[col] == args.state, "geometry"]
    if len(poly) != 1:
        raise SystemExit(f"{args.state}: {len(poly)} polygons in {args.states}")
    poly = poly.iloc[0]

    idx = args.products / "wells.geoparquet"
    ll = pq.read_table(idx, columns=["longitude", "latitude", "state"]).to_pandas()
    inside = shapely.contains_xy(poly, ll["longitude"].values, ll["latitude"].values)
    log.info(
        "%s: %d of %d wells inside the polygon (state column says %d)",
        args.state,
        int(inside.sum()),
        len(ll),
        int((ll["state"] == args.state).sum()),
    )
    wells = gpd.read_parquet(idx)
    wells = wells.loc[inside].reset_index(drop=True)
    out = args.out
    (out / "wells").mkdir(parents=True, exist_ok=True)
    wells.to_parquet(out / "wells.geoparquet")

    keys = wells[KEY]
    side = {}
    for name in ("wells_use.parquet", "confinement_predictions.parquet"):
        t = pd.read_parquet(args.products / name)
        sub = t.merge(keys, on=KEY, how="inner")
        sub.to_parquet(out / name, index=False)
        side[name] = int(len(sub))
        log.info("%s: %d rows", name, len(sub))

    found: dict[str, int] = {}
    missing: dict[str, int] = {}
    extra: dict[str, int] = {}
    missing_rows = []
    for row in wells[KEY + ["file_path", "has_time_series"]].itertuples(index=False):
        src = Path(row.file_path)
        has = bool(row.has_time_series)
        if src.exists():
            shutil.copy2(src, out / "wells" / src.name)
            (found if has else extra)[row.source] = (found if has else extra).get(
                row.source, 0
            ) + 1
        elif has:
            missing[row.source] = missing.get(row.source, 0) + 1
            missing_rows.append([row.source, row.source_well_id, str(src)])
    log.info(
        "series files: %d for has_time_series rows, %d for static-only rows, %d missing",
        sum(found.values()),
        sum(extra.values()),
        sum(missing.values()),
    )
    if missing_rows:
        pd.DataFrame(missing_rows, columns=KEY + ["file_path"]).to_csv(
            out / "missing_series_files.csv", index=False
        )

    for name in ("update_summary.json", "diff_report.html"):
        shutil.copy2(args.products / name, out / name)

    manifest = {
        "date": date.today().isoformat(),
        "product_dir": str(args.products),
        "gwx_repo_sha": git_sha(args.gwx_repo),
        "selection_rule": (
            f"well point (longitude, latitude; WGS84) inside the {args.state} "
            f"polygon of {args.states}; strict boundary, no buffer"
        ),
        "n_wells": int(len(wells)),
        "n_wells_state_column": int((ll["state"] == args.state).sum()),
        "n_per_source": wells["source"].value_counts().to_dict(),
        "n_has_time_series": int(wells["has_time_series"].sum()),
        "series_files_found_has_time_series": found,
        "series_files_found_static_only": extra,
        "series_files_missing_has_time_series": missing,
        "n_series_files": sum(found.values()) + sum(extra.values()),
        "side_tables": side,
        "bbox_wgs84": [float(v) for v in wells.total_bounds],
        "columns": list(wells.columns),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %s", out / "MANIFEST.json")


if __name__ == "__main__":
    main()
