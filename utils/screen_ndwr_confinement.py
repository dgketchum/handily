"""Confinement screen for the frozen NDWR label/source pool and admission targets.

Joins every NDWR site (rounded EPSG:5070 metres) to the GWX well product restricted
to ``source == 'nv_ndwr'`` and flags a site when any collocated well carries a
``confined`` / ``likely_confined`` / ``artesian`` confinement class. Writes the
per-site flag table and cleaned copies of the frozen source pool and the
admission targets (identical columns and dtypes; flagged sites dropped under the
``any_confined_flag`` policy). The ``all_confined_flag`` policy is counted only.

Run::

    uv run --directory /home/dgketchum/code/handily python /home/dgketchum/code/handily/utils/screen_ndwr_confinement.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FLAG_CLASSES = {"confined", "likely_confined", "artesian"}
GWX_COLS = [
    "longitude",
    "latitude",
    "confinement_class",
    "confinement_source",
    "well_use",
    "well_class",
    "source",
    "state",
    "canonical_id",
    "geometry",
]


def site_key(df: pd.DataFrame) -> list[tuple[int, int]]:
    return list(zip(df.x5070.round().astype("int64"), df.y5070.round().astype("int64")))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wells-dir", default="/data/ssd2/handily/nv/regional/wells")
    ap.add_argument("--sites", default="ndwr_sites_a4_input.parquet")
    ap.add_argument("--sources", default="ndwr_admitted_sources_frozen.parquet")
    ap.add_argument("--targets", default="ndwr_admission_targets.parquet")
    ap.add_argument("--gwx", default="/data/ssd2/gwx/products/current/wells.geoparquet")
    ap.add_argument(
        "--gwx-source", default="nv_ndwr", help="GWX source name to join on"
    )
    ap.add_argument(
        "--suffix", default="_unconf", help="suffix for the cleaned source/target files"
    )
    args = ap.parse_args()

    wd = Path(args.wells_dir)
    sites = pd.read_parquet(wd / args.sites)
    src = pd.read_parquet(wd / args.sources)
    tgt = pd.read_parquet(wd / args.targets)
    sites["k"] = site_key(sites)
    assert sites.k.is_unique, "site key not unique"

    tab = pq.read_table(
        args.gwx, columns=GWX_COLS, filters=[("source", "==", args.gwx_source)]
    )
    gw = gpd.GeoDataFrame.from_arrow(tab).to_crs(5070)
    gw["x5070"] = gw.geometry.x
    gw["y5070"] = gw.geometry.y
    gw["k"] = site_key(gw)
    gw["flag"] = gw.confinement_class.isin(FLAG_CLASSES)
    print(f"gwx {args.gwx_source} wells: {len(gw)}")

    agg = gw.groupby("k").agg(
        n_wells_matched=("flag", "size"),
        n_confined_flag=("flag", "sum"),
        confinement_sources=(
            "confinement_source",
            lambda s: "|".join(sorted(set(map(str, s)))),
        ),
        classes=("confinement_class", lambda s: "|".join(sorted(set(map(str, s))))),
    )
    f = sites[["x5070", "y5070", "k"]].merge(
        agg, left_on="k", right_index=True, how="left"
    )
    f["n_wells_matched"] = f.n_wells_matched.fillna(0).astype(int)
    f["n_confined_flag"] = f.n_confined_flag.fillna(0).astype(int)
    f["any_confined_flag"] = f.n_confined_flag > 0
    f["all_confined_flag"] = (f.n_wells_matched > 0) & (
        f.n_confined_flag == f.n_wells_matched
    )

    src["k"] = site_key(src)
    tgt["k"] = site_key(tgt)
    sk = set(src.k)
    pk = set(tgt.loc[tgt.is_admission_eligible, "k"])
    f["site_role"] = np.where(
        f.k.isin(sk), "admitted", np.where(f.k.isin(pk), "primary", "other")
    )
    print(
        f.groupby("site_role").agg(
            n=("k", "size"),
            matched=("n_wells_matched", lambda s: (s > 0).sum()),
            any_flag=("any_confined_flag", "sum"),
            all_flag=("all_confined_flag", "sum"),
        )
    )

    flags_path = wd / "ndwr_site_confinement_flags.parquet"
    f.drop(columns=["k"]).to_parquet(flags_path, index=False)

    flagged = set(f.loc[f.any_confined_flag, "k"])
    all_flagged = set(f.loc[f.all_confined_flag, "k"])
    src_u = src[~src.k.isin(flagged)].drop(columns="k")
    tgt_u = tgt[~tgt.k.isin(flagged)].drop(columns="k")
    for a, b in ((src_u, src), (tgt_u, tgt)):
        b = b.drop(columns="k")
        assert list(a.columns) == list(b.columns) and list(a.dtypes) == list(b.dtypes)

    def sets(t: pd.DataFrame) -> dict[str, int]:
        return dict(
            A_primary=int(t.is_admission_eligible.sum()),
            B_pumping=int(t.is_pumping.sum()),
            C_pre1980=int((t.por_end_yr < 1980).sum()),
            D_all=len(t),
        )

    src_out = wd / f"{Path(args.sources).stem}{args.suffix}.parquet"
    tgt_out = wd / f"{Path(args.targets).stem}{args.suffix}.parquet"
    src_u.to_parquet(src_out, index=False)
    tgt_u.to_parquet(tgt_out, index=False)
    print(f"sources {len(src)} -> {len(src_u)}; targets {sets(tgt)} -> {sets(tgt_u)}")
    print(
        "all_confined policy would remove: sources",
        int(src.k.isin(all_flagged).sum()),
        "targets",
        int(tgt.k.isin(all_flagged).sum()),
    )
    rem = src.loc[src.k.isin(flagged), "obs_dtw_m"]
    keep = src_u.obs_dtw_m
    print(
        "removed label obs_dtw_m median/p10/p90 %.2f/%.2f/%.2f; kept %.2f/%.2f/%.2f"
        % (
            rem.median(),
            rem.quantile(0.1),
            rem.quantile(0.9),
            keep.median(),
            keep.quantile(0.1),
            keep.quantile(0.9),
        )
    )
    print(f"wrote {flags_path}\n      {src_out}\n      {tgt_out}")


if __name__ == "__main__":
    main()
