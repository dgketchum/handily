"""Merge per-source mountain-well acquisitions into one inventory + observations table.

Reads  /nas/gwx/studies/mountain_wells/parsed/<source_id>/{inventory.csv, observations.parquet|csv,
observations_daily.parquet, manifest.json} for every source present, runs one consistent 100 m terrain
check (EPSG:5070 covariates), and writes parsed/_merged/{inventory.csv, inventory.gpkg,
observations.parquet, observations_daily.parquet, manifest.json, SUMMARY.md}.

Terrain check (documented method, same for every source):
  elev_dem_m   100 m USGS NED (elev48i0100a.tif)
  tpi_500m/2km WhiteboxTools DevFromMeanElev, dimensionless (sigma units of the window)
  geomorphon   WBT 10-class: 1 flat 2 peak 3 ridge 4 shoulder 5 spur 6 slope 7 hollow 8 footslope 9 valley 10 pit
  relief_2km_m max - min elevation in a 21x21 cell (2.1 km) window
  dem_class    ridgetop_divide  : geomorphon in {2,3} or (tpi_500m >= 1.0 and tpi_2km >= 0.5)
               mesa_plateau_top : geomorphon == 1 (flat) and tpi_2km >= 0.5 and relief_2km_m >= 100
               mountain_valley_bottom : geomorphon in {9,10} or tpi_500m <= -1.0
               mountain_slope   : otherwise, if relief_2km_m >= 150
               lowland          : otherwise
  'alpine' cannot be derived from the DEM; it is kept only when the source supports it.
  dem_agrees   yes when dem_class == landscape_class, or alpine vs DEM ridge/slope, or mesa top vs DEM ridge
               (a 5 km geomorphon search reads a mesa edge/top as 'ridge').
Run:  uv run python -I utils/mountain_wells/merge_mountain_wells.py [--root /nas/gwx/studies/mountain_wells]
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

TERRAIN = Path("/nas/handily/covariates/terrain")
SOURCES = [
    "handcart_co",
    "jemez_deep_nm",
    "jemez_piez_nm",
    "marshall_gulch_az",
    "rainier_mesa_nv",
    "sacramento_mtns_nm",
    "helena_bedrock_mt",
]
INV_COLS = [
    "source_id",
    "well_id",
    "interval_id",
    "well_name",
    "site_no",
    "lat",
    "lon",
    "horiz_datum",
    "coord_source",
    "coord_confidence",
    "elev_m",
    "elev_datum",
    "elev_source",
    "mp_height_m",
    "borehole_depth_m",
    "well_depth_m",
    "screen_top_m",
    "screen_bot_m",
    "geology",
    "aquifer_setting",
    "state",
    "landscape_class",
    "landscape_evidence",
    "landscape_confidence",
    "record_start",
    "record_end",
    "n_obs",
    "source_file",
    "notes",
]
OBS_COLS = [
    "source_id",
    "well_id",
    "interval_id",
    "timestamp",
    "orig_value",
    "orig_units",
    "orig_reference",
    "mp_offset_m",
    "dtw_bgs_m",
    "conversion",
    "flags",
    "censored",
    "source_file",
    "source_row",
]
CLASSES = [
    "alpine",
    "ridgetop_divide",
    "mountain_slope",
    "mountain_valley_bottom",
    "mesa_plateau_top",
    "uncertain",
]


def read_table(path_base: Path) -> pd.DataFrame | None:
    for ext in (".parquet", ".csv"):
        p = path_base.with_suffix(ext)
        if p.exists():
            return (
                pd.read_parquet(p)
                if ext == ".parquet"
                else pd.read_csv(p, low_memory=False)
            )
    return None


def conform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra]


def sample_terrain(lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
    tr = Transformer.from_crs(4326, 5070, always_xy=True)
    x, y = tr.transform(lon, lat)
    out = {"x_5070": x, "y_5070": y}
    layers = {
        "elev_dem_m": "elev48i0100a.tif",
        "tpi_500m": "tpi_500m.tif",
        "tpi_2km": "tpi_2km.tif",
        "geomorphon": "geomorphons.tif",
    }
    for name, fn in layers.items():
        with rasterio.open(TERRAIN / fn) as src:
            vals = np.array([v[0] for v in src.sample(zip(x, y))], dtype="float64")
            nd = src.nodata
            if nd is not None:
                vals[vals == nd] = np.nan
            out[name] = vals
    relief = np.full(len(x), np.nan)
    with rasterio.open(TERRAIN / "elev48i0100a.tif") as src:
        nd = src.nodata
        for i, (xi, yi) in enumerate(zip(x, y)):
            if not np.isfinite(xi):
                continue
            r, c = src.index(xi, yi)
            if r < 10 or c < 10 or r + 11 > src.height or c + 11 > src.width:
                continue
            a = src.read(1, window=Window(c - 10, r - 10, 21, 21)).astype("float64")
            a[a == nd] = np.nan
            if np.isfinite(a).any():
                relief[i] = np.nanmax(a) - np.nanmin(a)
    out["relief_2km_m"] = relief
    return pd.DataFrame(out)


def dem_class(row) -> str:
    g, t5, t2, rl = (
        row["geomorphon"],
        row["tpi_500m"],
        row["tpi_2km"],
        row["relief_2km_m"],
    )
    if not np.isfinite(g) or not np.isfinite(rl):
        return "no_dem"
    if g in (2, 3) or (t5 >= 1.0 and t2 >= 0.5):
        return "ridgetop_divide"
    if g == 1 and t2 >= 0.5 and rl >= 100:
        return "mesa_plateau_top"
    if g in (9, 10) or t5 <= -1.0:
        return "mountain_valley_bottom"
    if rl >= 150:
        return "mountain_slope"
    return "lowland"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/nas/gwx/studies/mountain_wells")
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "parsed" / "_merged"
    out.mkdir(parents=True, exist_ok=True)

    invs, obss, dailies, manifests, missing = [], [], [], [], []
    for sid in SOURCES:
        d = root / "parsed" / sid
        inv = read_table(d / "inventory")
        if inv is None:
            missing.append(sid)
            continue
        inv = conform(inv, INV_COLS)
        inv["source_id"] = sid
        invs.append(inv)
        ob = read_table(d / "observations")
        if ob is not None:
            ob = conform(ob, OBS_COLS)
            ob["source_id"] = sid
            obss.append(ob)
        dy = read_table(d / "observations_daily")
        if dy is not None:
            dy["source_id"] = sid
            dailies.append(dy)
        mp = d / "manifest.json"
        manifests.append(
            json.loads(mp.read_text())
            if mp.exists()
            else {"source_id": sid, "status": "no_manifest"}
        )

    inv = pd.concat(invs, ignore_index=True)
    inv["lat"] = pd.to_numeric(inv["lat"], errors="coerce")
    inv["lon"] = pd.to_numeric(inv["lon"], errors="coerce")
    bad_class = ~inv["landscape_class"].isin(CLASSES)
    inv.loc[bad_class, "landscape_class"] = "uncertain"

    ter = sample_terrain(inv["lat"].to_numpy(), inv["lon"].to_numpy())
    inv = pd.concat([inv.reset_index(drop=True), ter], axis=1)
    inv["dem_class"] = inv.apply(dem_class, axis=1)
    inv["elev_diff_dem_m"] = inv["elev_dem_m"] - pd.to_numeric(
        inv["elev_m"], errors="coerce"
    )
    agree = (
        (inv["dem_class"] == inv["landscape_class"])
        | (
            (inv["landscape_class"] == "alpine")
            & inv["dem_class"].isin(["ridgetop_divide", "mountain_slope"])
        )
        | (
            (inv["landscape_class"] == "mesa_plateau_top")
            & (inv["dem_class"] == "ridgetop_divide")
        )
    )
    inv["dem_agrees"] = np.where(
        inv["dem_class"] == "no_dem", "no_dem", np.where(agree, "yes", "no")
    )

    obs = pd.concat(obss, ignore_index=True) if obss else pd.DataFrame(columns=OBS_COLS)
    obs["dtw_bgs_m"] = pd.to_numeric(obs["dtw_bgs_m"], errors="coerce")
    obs["timestamp"] = obs["timestamp"].astype("string")
    obs.loc[obs["timestamp"].isin(["", "nan", "NaT", "None"]), "timestamp"] = pd.NA
    # recompute n_obs / record range from the merged observations where present
    if len(obs):
        g = obs.groupby("interval_id")
        rng = g["timestamp"].agg(["min", "max", "size"])
        inv = inv.merge(rng, left_on="interval_id", right_index=True, how="left")
        inv["n_obs"] = inv["size"].fillna(inv["n_obs"])
        inv["record_start"] = inv["min"].fillna(inv["record_start"])
        inv["record_end"] = inv["max"].fillna(inv["record_end"])
        inv = inv.drop(columns=["min", "max", "size"])
        ndtw = g["dtw_bgs_m"].agg(
            n_dtw="count", dtw_median_m="median", dtw_min_m="min", dtw_max_m="max"
        )
        inv = inv.merge(ndtw, left_on="interval_id", right_index=True, how="left")

    inv.to_csv(out / "inventory.csv", index=False)
    gdf = gpd.GeoDataFrame(
        inv[inv["lat"].notna()],
        geometry=gpd.points_from_xy(
            inv.loc[inv["lat"].notna(), "lon"], inv.loc[inv["lat"].notna(), "lat"]
        ),
        crs=4326,
    )
    (out / "inventory.gpkg").unlink(missing_ok=True)
    gdf.to_file(out / "inventory.gpkg", layer="wells", driver="GPKG")
    obs.to_parquet(out / "observations.parquet", index=False)
    if dailies:
        pd.concat(dailies, ignore_index=True).to_parquet(
            out / "observations_daily.parquet", index=False
        )

    man = {
        "merged_on": date.today().isoformat(),
        "sources": manifests,
        "sources_missing_parsed_output": missing,
        "terrain_method": __doc__.split("Terrain check")[1].split("Run:")[0].strip(),
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str))

    # summary
    lines = [f"# Mountain wells — merged summary ({date.today().isoformat()})", ""]
    lines.append("## Per source")
    lines.append(
        "| source | status | intervals | wells | obs | obs with dtw | first | last |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    for m in manifests:
        sid = m.get("source_id")
        i = inv[inv.source_id == sid]
        o = obs[obs.source_id == sid]
        lines.append(
            f"| {sid} | {m.get('status')} | {len(i)} | {i.well_id.nunique()} | {len(o)} | {int(o.dtw_bgs_m.notna().sum())} | "
            f"{str(o.timestamp.min())[:10] if len(o) else ''} | {str(o.timestamp.max())[:10] if len(o) else ''} |"
        )
    for sid in missing:
        lines.append(f"| {sid} | NO PARSED OUTPUT | | | | | | |")
    lines += ["", "## Landscape class (source) × DEM class", ""]
    ct = pd.crosstab(inv["landscape_class"], inv["dem_class"])
    lines.append(ct.to_markdown())
    lines += ["", "## Landscape class by source (intervals)", ""]
    lines.append(pd.crosstab(inv["source_id"], inv["landscape_class"]).to_markdown())
    cand = inv[
        inv.landscape_class.isin(["alpine", "ridgetop_divide", "mesa_plateau_top"])
    ].copy()
    cand = cand.sort_values(["landscape_class", "dem_agrees", "landscape_confidence"])
    lines += ["", "## Alpine / ridgetop / mesa-top candidates", ""]
    cols = [
        "source_id",
        "well_name",
        "interval_id",
        "landscape_class",
        "landscape_confidence",
        "dem_class",
        "dem_agrees",
        "elev_m",
        "elev_dem_m",
        "relief_2km_m",
        "tpi_500m",
        "well_depth_m",
        "n_obs",
        "n_dtw",
        "dtw_median_m",
        "record_start",
        "record_end",
    ]
    cols = [c for c in cols if c in cand.columns]
    lines.append(
        cand[cols].to_markdown(index=False, floatfmt=".1f") if len(cand) else "_none_"
    )
    lines += ["", "## Missing metadata (counts of intervals)", ""]
    for c in [
        "lat",
        "elev_m",
        "well_depth_m",
        "screen_top_m",
        "geology",
        "horiz_datum",
        "elev_datum",
    ]:
        lines.append(f"- {c}: {int(inv[c].isna().sum())} of {len(inv)} missing")
    lines += ["", "## Unresolved issues from source manifests", ""]
    for m in manifests:
        for u in m.get("unresolved", []) or []:
            lines.append(f"- {m.get('source_id')}: {u}")
    lines += ["", "## Flags in observations", ""]
    if len(obs):
        fl = obs["flags"].fillna("").astype(str).str.split("|").explode()
        fl = fl[fl != ""]
        lines.append(fl.value_counts().to_frame("n").to_markdown())
        lines.append("")
        lines.append(
            f"censored: {obs['censored'].fillna('').astype(str).value_counts().to_dict()}"
        )
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
