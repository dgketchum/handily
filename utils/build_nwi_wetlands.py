"""Consolidate raw NWI state wetland shapefiles into analysis-ready FlatGeobuf.

The raw FWS National Wetlands Inventory downloads live as per-state shapefiles in
``/nas/irrmapper/wetlands/raw_shp/`` (``<ST>_Wetlands*.shp``), in NAD83 / CONUS
Albers (= EPSG:5070) but tagged with the ESRI ``NAD_1983_Albers`` WKT and with no
parsed classification. This makes them usable: clean EPSG:5070 stamp, parsed
Cowardin code, and a regime-aware ``near_surface_wt`` flag for water-table anchoring.

Why the parse matters (arid west): the single most common NWI class in NM is
``R4SBC`` -- Riverine **Intermittent** Streambed, i.e. the dry arroyos. The riverine
subsystem digit encodes perenniality (R4 = intermittent; R1/2/3/5 = perennial), so a
naive "all riverine" rule re-imports the perched-channel contamination that sinks the
HAND/stream-bed water-table prior. We split perennial-riverine + vegetated-palustrine
(near-surface water table) from intermittent (R4), open-water ponds/lakes, and
artificial (diked/excavated) classes.

Cowardin code layout: System(A) Subsystem(d?) Class(AA) Subclass(d?) Regime(A?)
Special(a*), e.g. ``R4SBC`` = R/4/SB/-/C, ``PEM1Ch`` = P/-/EM/1/C/h.

Run::

    uv run python utils/build_nwi_wetlands.py --states NM,MT

Writes ``<out_dir>/<ST>_wetlands_5070.fgb`` (default out_dir
``/nas/hydrography/nwi``); multi-part states (e.g. MT East/West, big-state splits)
are concatenated into one file.
"""

import argparse
import glob
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

RAW_DIR = Path("/nas/irrmapper/wetlands/raw_shp")
OUT_DIR = Path("/nas/hydrography/nwi")
EPSG = 5070

VEG_PALUSTRINE = {"EM", "FO", "SS"}  # emergent / forested / scrub-shrub
PERENNIAL_RIVERINE_SUBSYS = {"1", "2", "3", "5"}  # NOT 4 (intermittent)
ARTIFICIAL_SPECIAL = set("hxrkd")  # diked, excavated, artificial substrate, ...
KEEP_RAW = ["ATTRIBUTE", "WETLAND_TY", "ACRES"]


def parse_cowardin(attr: str) -> dict:
    """Split an NWI ATTRIBUTE code into components + derived water-table flags."""
    a = (attr or "").strip()
    out = {
        "system": "",
        "subsystem": "",
        "cls": "",
        "regime": "",
        "special": "",
        "perennial": False,
        "near_surface_wt": False,
        "artificial": False,
        "lacustrine": False,
        "intermittent": False,
    }
    if not a:
        return out
    out["system"] = a[0].upper()
    i = 1
    if len(a) > i and a[i].isdigit():
        out["subsystem"] = a[i]
        i += 1
    out["cls"] = a[i : i + 2].upper()
    special = re.findall(r"[a-z]", a)  # trailing lowercase special modifiers
    out["special"] = "".join(special)
    regime = re.search(r"[A-Z]([A-Z])[a-z]*$", a)  # last uppercase before special run
    out["regime"] = regime.group(1) if regime else ""

    out["artificial"] = any(c in ARTIFICIAL_SPECIAL for c in out["special"])
    out["lacustrine"] = out["system"] == "L"
    out["intermittent"] = out["system"] == "R" and out["subsystem"] == "4"
    perennial_riv = (
        out["system"] == "R" and out["subsystem"] in PERENNIAL_RIVERINE_SUBSYS
    )
    veg_pal = out["system"] == "P" and out["cls"] in VEG_PALUSTRINE
    out["perennial"] = perennial_riv
    out["near_surface_wt"] = (perennial_riv or veg_pal) and not out["artificial"]
    return out


def process_state(state: str, out_dir: Path) -> Path:
    parts = sorted(glob.glob(str(RAW_DIR / f"{state}_Wetlands*.shp")))
    if not parts:
        raise SystemExit(f"no raw shp for {state} under {RAW_DIR}")
    print(f"\n=== {state}: {len(parts)} source file(s) ===")
    gdfs = []
    for p in parts:
        info = pyogrio.read_info(p)
        cols = [c for c in KEEP_RAW if c in info["fields"]]
        g = gpd.read_file(p, columns=cols)
        print(f"  {Path(p).name}: {len(g)} feats")
        gdfs.append(g)
    gdf = pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
    # data is already CONUS Albers; stamp clean EPSG:5070 (identity, no transform)
    gdf = gdf.set_crs(EPSG, allow_override=True)

    parsed = pd.DataFrame(
        [parse_cowardin(a) for a in gdf["ATTRIBUTE"].astype("string").fillna("")],
        index=gdf.index,
    )
    gdf = pd.concat([gdf, parsed], axis=1)
    gdf.columns = [c.lower() for c in gdf.columns]

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{state}_wetlands_5070.fgb"
    gdf.to_file(out, driver="FlatGeobuf")

    n = len(gdf)
    print(f"  -> {out}  ({n} feats)")
    print(
        f"     near_surface_wt: {gdf['near_surface_wt'].sum()} ({100 * gdf['near_surface_wt'].mean():.0f}%)"
        f"  perennial_riverine: {gdf['perennial'].sum()}"
        f"  intermittent(R4): {gdf['intermittent'].sum()}"
        f"  lacustrine: {gdf['lacustrine'].sum()}"
        f"  artificial: {gdf['artificial'].sum()}"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", required=True, help="comma-separated, e.g. NM,MT")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    for st in [s.strip().upper() for s in args.states.split(",") if s.strip()]:
        process_state(st, out_dir)


if __name__ == "__main__":
    main()
