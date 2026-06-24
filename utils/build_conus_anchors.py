"""New stage of the CONUS WTE/DTW GNN v2: extract water-table anchor nodes.

Springs, perennial open water, and persistent wetlands are *surface expressions of
the water table* (head ~ land surface => DTW ~ 0), so they are Dirichlet/soft
boundary conditions for the anchor-conditioned flow-network relaxation
(``notes/CONUS_GNN_V2_PLAN.md``). This stage extracts them, attaches each to its
controlling reach, and assigns a class/subtype-dependent head uncertainty
(springs = hard BC; lakes/wetlands = soft BC -- they can be managed/perched).

Anchors carry a DEM land-surface head + a fixed BC DTW=0; they NEVER carry an
observed well DTW, so they are not target leakage (same target-blind logic as
ConusFAC -- a *different measurement* than the GWX wells).

State-list-driven, zero-edit expansion: the only hardcoded tables are
``CONUS_STATES`` + ``STATE_NAME_TO_USPS``. Sources are discovered by glob (NHD-HR
State Shape, NWI raw_shp); 3DHP CONUS is the fallback where a state's NHD-HR/NWI is
absent. Per state, each class resolves NHD-HR/NWI as primary, 3DHP as fallback.
Every coverage gap is enumerated in the manifest + log (no silent caps).

Outputs under ``--out-dir`` (default ``.../wte_gnn/anchors/``):
  anchor_nodes.parquet         one row per retained primary anchor (EPSG:5070)
  anchor_edges.parquet         anchor->reach attachment (k-nearest controlling reach)
  anchor_dedup_audit.parquet   all candidates with dedup_group / is_primary
  anchor_manifest.json         effective config + per-state/-class counts + gaps
  anchor_coverage_qa.csv       per-state/-class candidate + retained counts
  state_cache/<ST>_<class>.parquet   per-state candidate cache (skip-if-exists)

    uv run python utils/build_conus_anchors.py --classes spring \\
        --out-dir /data/ssd2/handily/conus/wte_gnn/anchors
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conus_graph_inputs import build_lateral_edges  # noqa: E402
from build_stacker_features import sample_coarse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_conus_anchors")

# --- the ONLY hardcoded tables (everything else is glob-discovered) -----------
STATE_NAME_TO_USPS = {
    "Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}  # fmt: skip
CONUS_STATES = sorted(STATE_NAME_TO_USPS.values())

ANCHOR_CLASSES = ("spring", "open_water", "wetland")

# NHD (HR + V2 share FCodes). Springs are points; perennial LakePond polygons are
# the open-water class; reservoirs / playas / intermittent / swamp-marsh are NOT.
NHD_SPRING_FCODE = 45800
NHD_PERENNIAL_LAKE_FCODES = frozenset({39000, 39001, 39004, 39009, 39012})
NHD_AREA_STREAMRIVER_FTYPE = 460  # behind --include-nhdarea

# 3DHP feature labels (fallback only).
TDHP_GPKG = "/nas/hydrography/3dhp/3dhp_all_CONUS_20260112_GPKG.gpkg"
TDHP_SPRING_LAYER = "hydro_3dhp_all_landscape"
TDHP_WATER_LAYER = "hydro_3dhp_all_waterbody"

# NWI ATTRIBUTE persistent-freshwater system/class prefixes (after the leading
# system letter): Palustrine PEM/PFO/PSS/PAB/PUB; Lacustrine L1*/L2*; perennial
# Riverine R2*/R3*. R4*/R5* (intermittent/unknown) and farmed/excavated/drained/
# temporary water-regime modifiers are excluded.
NWI_KEEP_PREFIXES = ("PEM", "PFO", "PSS", "PAB", "PUB", "L1", "L2", "R2", "R3")
NWI_EXCLUDE_PREFIXES = ("R4", "R5")
# Special / water-regime modifier letters that mark a non-persistent or managed
# wetland (NWI Cowardin modifiers): b=beaver, d=partly drained, f=farmed,
# h=diked/impounded, x=excavated, A/J=temporary, K=artificially flooded.
NWI_EXCLUDE_MODIFIERS = ("d", "f", "x", "K")

# Head uncertainty (m) by class -- the soft-vs-hard BC encoding. Springs pin the
# table to the land surface (clean Dirichlet); lakes/wetlands are priors the gate
# can down-weight.
HEAD_UNCERTAINTY = {"spring": 0.5, "open_water": 1.5, "wetland": 2.5}
# Subtype bumps for anything managed/evaporative that may slip the filters.
HEAD_UNCERTAINTY_NHDAREA = 2.0  # river/area polygon, looser than a lake centroid
HEAD_UNCERTAINTY_MANAGED_BUMP = 1.5

SOURCE_RANK = {"nhd_hr": 0, "nwi": 1, "3dhp": 2}  # precedence for dedup primary

DEM = "/data/ssd1/streamflow-ml-data/conus-dem/data/elev48i0100a.tif"
ANCHOR_NODE_COLS = [
    "anchor_node_idx", "x5070", "y5070", "head_m", "head_uncertainty_m",
    "anchor_class", "source", "state", "src_fcode", "src_typelabel",
    "area_sqkm", "gnis_id",
]  # fmt: skip


# ---------------------------------------------------------------------------
# Source discovery (glob-driven; mirrors src/handily/naip_rf.py idioms)
# ---------------------------------------------------------------------------
def discover_nhd_hr_states(nhd_hr_dir: str) -> dict[str, str]:
    """{USPS: shape_dir} for every NHD_H_<State>_State_Shape with an NHDPoint.shp."""
    out: dict[str, str] = {}
    for d in sorted(glob.glob(os.path.join(nhd_hr_dir, "NHD_H_*_State_Shape"))):
        m = re.match(r"NHD_H_(.+)_State_Shape$", os.path.basename(d))
        if not m:
            continue
        name = m.group(1).replace("_", " ")
        usps = STATE_NAME_TO_USPS.get(name)
        if usps is None:
            continue
        if os.path.exists(os.path.join(d, "Shape", "NHDPoint.shp")):
            out[usps] = os.path.join(d, "Shape")
    return out


def discover_nwi_states(nwi_dir: str) -> dict[str, list[str]]:
    """{USPS: [shp,...]} from ^([A-Z]{2})_Wetlands.*\\.shp$ (groups multi-part CA/CO)."""
    pat = re.compile(r"^([A-Z]{2})_Wetlands.*\.shp$", re.IGNORECASE)
    out: dict[str, list[str]] = {}
    for fname in sorted(os.listdir(nwi_dir)):
        m = pat.match(fname)
        if not m:
            continue
        out.setdefault(m.group(1).upper(), []).append(os.path.join(nwi_dir, fname))
    return out


# ---------------------------------------------------------------------------
# Per-state, per-class candidate extraction (NHD-HR / NWI primary)
# ---------------------------------------------------------------------------
def _xy_5070(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Representative-point coords in EPSG:5070 (inside polygons; .x/.y for points)."""
    g = gdf.to_crs(5070)
    geom = g.geometry
    is_pt = geom.geom_type.isin(["Point", "MultiPoint"]).to_numpy()
    rep = geom.representative_point()
    x = np.where(is_pt, geom.x.to_numpy(), rep.x.to_numpy())
    y = np.where(is_pt, geom.y.to_numpy(), rep.y.to_numpy())
    return x.astype("float64"), y.astype("float64")


def _frame(
    x, y, anchor_class, source, state, src_fcode, src_typelabel, area_sqkm, gnis_id
) -> pd.DataFrame:
    n = len(x)

    def col(v):
        return v if hasattr(v, "__len__") and not isinstance(v, str) else [v] * n

    return pd.DataFrame(
        {
            "x5070": x,
            "y5070": y,
            "anchor_class": anchor_class,
            "source": source,
            "state": state,
            "src_fcode": np.asarray(col(src_fcode), dtype="int32"),
            "src_typelabel": col(src_typelabel),
            "area_sqkm": np.asarray(col(area_sqkm), dtype="float64"),
            "gnis_id": col(gnis_id),
        }
    )


def extract_nhd_springs(shape_dir: str, state: str) -> pd.DataFrame:
    pts = gpd.read_file(os.path.join(shape_dir, "NHDPoint.shp"))
    fcode = pts["fcode"].to_numpy()
    pts = pts[fcode == NHD_SPRING_FCODE].copy()
    if pts.empty:
        return pts.iloc[0:0]
    x, y = _xy_5070(pts)
    gnis = pts["gnis_id"].astype("string").to_numpy()
    return _frame(
        x, y, "spring", "nhd_hr", state, NHD_SPRING_FCODE, "spring", 0.0, gnis
    )


def extract_nhd_open_water(
    shape_dir: str, state: str, include_nhdarea: bool, min_lake_sqkm: float
) -> pd.DataFrame:
    parts = []
    wb_path = os.path.join(shape_dir, "NHDWaterbody.shp")
    if os.path.exists(wb_path):
        wb = gpd.read_file(wb_path)
        keep = wb["fcode"].isin(NHD_PERENNIAL_LAKE_FCODES) & (
            wb["areasqkm"].fillna(0.0) >= min_lake_sqkm
        )
        wb = wb[keep].copy()
        if not wb.empty:
            x, y = _xy_5070(wb)
            parts.append(
                _frame(
                    x, y, "open_water", "nhd_hr", state,
                    wb["fcode"].to_numpy(),
                    ["NHDWaterbody_" + str(c) for c in wb["fcode"].to_numpy()],
                    wb["areasqkm"].fillna(0.0).to_numpy(),
                    wb["gnis_id"].astype("string").to_numpy(),
                )
            )  # fmt: skip
    area_path = os.path.join(shape_dir, "NHDArea.shp")
    if include_nhdarea and os.path.exists(area_path):
        ar = gpd.read_file(area_path)
        keep = (ar["ftype"] == NHD_AREA_STREAMRIVER_FTYPE) & (
            ar["areasqkm"].fillna(0.0) >= min_lake_sqkm
        )
        ar = ar[keep].copy()
        if not ar.empty:
            x, y = _xy_5070(ar)
            parts.append(
                _frame(
                    x, y, "open_water", "nhd_hr", state,
                    ar["fcode"].to_numpy(),
                    ["NHDArea_" + str(t) for t in ar["ftype"].to_numpy()],
                    ar["areasqkm"].fillna(0.0).to_numpy(),
                    ar["gnis_id"].astype("string").to_numpy(),
                )
            )  # fmt: skip
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _nwi_persistent_mask(attr: pd.Series) -> np.ndarray:
    """Persistent-freshwater classes, excluding intermittent + managed modifiers."""
    a = attr.astype("string").fillna("")
    # base system/class code is the leading letters+digits before any '/'.
    base = a.str.split("/").str[0]
    keep = base.str.startswith(NWI_KEEP_PREFIXES) & ~base.str.startswith(
        NWI_EXCLUDE_PREFIXES
    )
    # trailing water-regime / special modifier letter (last alpha char of the full
    # attribute) flags farmed/excavated/drained/artificially-flooded -> drop.
    last = a.str.extract(r"([A-Za-z])\s*$", expand=False).fillna("")
    managed = last.isin(list(NWI_EXCLUDE_MODIFIERS))
    return (keep & ~managed).to_numpy()


def extract_nwi_wetland(
    shp_paths: list[str], state: str, min_wetland_sqkm: float
) -> pd.DataFrame:
    parts = []
    for p in shp_paths:
        w = gpd.read_file(p)
        attr = (
            w["ATTRIBUTE"] if "ATTRIBUTE" in w.columns else pd.Series("", index=w.index)
        )
        area_sqkm = (
            w["Shape_Area"].to_numpy("float64") / 1e6
            if "Shape_Area" in w.columns
            else w.geometry.to_crs(5070).area.to_numpy() / 1e6
        )
        keep = _nwi_persistent_mask(attr) & (area_sqkm >= min_wetland_sqkm)
        w = w[keep].copy()
        if w.empty:
            continue
        x, y = _xy_5070(w)
        base = attr[keep].astype("string").fillna("").str.split("/").str[0]
        parts.append(
            _frame(
                x, y, "wetland", "nwi", state, -1, base.to_numpy(),
                area_sqkm[keep], np.full(len(w), None, dtype=object),
            )
        )  # fmt: skip
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def extract_3dhp(
    anchor_class: str,
    states: list[str],
    state_polys: gpd.GeoDataFrame | None,
    min_lake_sqkm: float,
) -> pd.DataFrame:
    """3DHP CONUS fallback for states lacking NHD-HR. Needs ``state_polys`` (USPS-
    keyed) to assign/clip features to the missing states; without it the gap is
    logged and skipped (never fabricated)."""
    if not states:
        return pd.DataFrame()
    if state_polys is None:
        log.warning(
            "3DHP fallback needed for %s but no --state-polys given; "
            "logging coverage gap (no anchors emitted for these states)",
            ",".join(states),
        )
        return pd.DataFrame()
    if anchor_class == "spring":
        layer, where = TDHP_SPRING_LAYER, "featuretypelabel = 'Spring'"
    elif anchor_class == "open_water":
        layer, where = TDHP_WATER_LAYER, "featuretypelabel = 'Lake'"
    else:
        return pd.DataFrame()  # no 3DHP wetlands
    polys = state_polys[state_polys["usps"].isin(states)].to_crs(5070)
    g = gpd.read_file(TDHP_GPKG, layer=layer, where=where).to_crs(5070)
    g = gpd.sjoin(g, polys[["usps", "geometry"]], predicate="within", how="inner")
    if anchor_class == "open_water" and "areasqkm" in g.columns:
        g = g[g["areasqkm"].fillna(0.0) >= min_lake_sqkm].copy()
    if g.empty:
        return pd.DataFrame()
    x, y = _xy_5070(g)
    area = g["areasqkm"].fillna(0.0).to_numpy() if "areasqkm" in g.columns else 0.0
    return _frame(
        x, y, anchor_class, "3dhp", g["usps"].to_numpy(),
        g["featuretype"].fillna(-1).astype("int32").to_numpy(),
        g["featuretypelabel"].astype("string").to_numpy(),
        area, g["gnisid"].astype("string").to_numpy() if "gnisid" in g.columns else None,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Head + uncertainty + dedup
# ---------------------------------------------------------------------------
def assign_head_uncertainty(cand: pd.DataFrame) -> np.ndarray:
    u = cand["anchor_class"].map(HEAD_UNCERTAINTY).to_numpy("float64")
    lbl = cand["src_typelabel"].astype("string").fillna("")
    is_area = lbl.str.startswith("NHDArea").to_numpy()
    u = np.where(is_area, HEAD_UNCERTAINTY_NHDAREA, u)
    managed = lbl.str.contains(
        "reservoir|playa|managed", case=False, regex=True
    ).to_numpy()
    u = u + np.where(managed, HEAD_UNCERTAINTY_MANAGED_BUMP, 0.0)
    return u


def dedup_within_class(
    cand: pd.DataFrame, radius_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Union-find merge of anchors within ``radius_m``; returns (dedup_group,
    is_primary). Primary = best (lowest source_rank, then largest area)."""
    xy = cand[["x5070", "y5070"]].to_numpy("float64")
    n = len(xy)
    parent = np.arange(n)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    if n > 1:
        pairs = cKDTree(xy).query_pairs(radius_m, output_type="ndarray")
        for a, b in pairs:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
    group = np.array([find(i) for i in range(n)], dtype="int64")
    src_rank = cand["source"].map(SOURCE_RANK).fillna(9).to_numpy()
    area = cand["area_sqkm"].fillna(0.0).to_numpy()
    # priority key: smaller is better -> (source_rank, -area). argmin per group.
    order = np.lexsort((-area, src_rank))  # primary candidate appears first per key
    is_primary = np.zeros(n, dtype=bool)
    seen: set[int] = set()
    for i in order:
        g = group[i]
        if g not in seen:
            seen.add(g)
            is_primary[i] = True
    return group, is_primary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/data/ssd2/handily/conus/wte_gnn/anchors")
    ap.add_argument(
        "--reach-nodes",
        default="/data/ssd2/handily/conus/wte_gnn/graph/reach_nodes.parquet",
    )
    ap.add_argument(
        "--reach-geom",
        default="/data/ssd2/handily/conus/wte_gnn/nhd_flowline_geom.parquet",
    )
    ap.add_argument("--dem", default=DEM)
    ap.add_argument("--nhd-hr-dir", default="/nas/hydrography/nhd/hr_state")
    ap.add_argument("--nwi-dir", default="/nas/irrmapper/wetlands/raw_shp")
    ap.add_argument(
        "--state-polys",
        default=None,
        help="optional USPS-keyed states polygon (3DHP fallback)",
    )
    ap.add_argument(
        "--classes",
        default="spring,open_water,wetland",
        help="comma list subset of spring,open_water,wetland",
    )
    ap.add_argument(
        "--states", default=None, help="comma list of USPS to restrict (default CONUS)"
    )
    ap.add_argument("--include-nhdarea", action="store_true")
    ap.add_argument("--min-lake-sqkm", type=float, default=0.01)
    ap.add_argument("--min-wetland-sqkm", type=float, default=0.004)
    ap.add_argument(
        "--dedup-radius-m", type=float, default=100.0, help="spring dedup radius"
    )
    ap.add_argument(
        "--dedup-radius-poly-m",
        type=float,
        default=50.0,
        help="lake/wetland dedup radius",
    )
    ap.add_argument(
        "--knn-anchor", type=int, default=1, help="reaches per anchor (controlling)"
    )
    ap.add_argument(
        "--max-attach-dist-m",
        type=float,
        default=5000.0,
        help="flag (not drop) far attachments",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "state_cache"
    log_dir = out_dir / "logs"
    for d in (out_dir, cache_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    bad = [c for c in classes if c not in ANCHOR_CLASSES]
    if bad:
        raise SystemExit(
            f"unknown anchor class(es): {bad}; choose from {ANCHOR_CLASSES}"
        )
    target_states = (
        [s.strip().upper() for s in args.states.split(",")]
        if args.states
        else CONUS_STATES
    )

    nhd_hr = discover_nhd_hr_states(args.nhd_hr_dir)
    nwi = discover_nwi_states(args.nwi_dir)
    log.info("discovered NHD-HR states=%d, NWI states=%d", len(nhd_hr), len(nwi))
    state_polys = gpd.read_file(args.state_polys) if args.state_polys else None

    # --- per-state, per-class candidate extraction (cached) -------------------
    coverage_rows: list[dict] = []
    gaps: list[dict] = []
    cand_parts: list[pd.DataFrame] = []
    for cls in classes:
        for st in target_states:
            cache = cache_dir / f"{st}_{cls}.parquet"
            if cache.exists():
                df = pd.read_parquet(cache)
            else:
                if cls == "spring":
                    df = (
                        extract_nhd_springs(nhd_hr[st], st)
                        if st in nhd_hr
                        else pd.DataFrame()
                    )
                elif cls == "open_water":
                    df = (
                        extract_nhd_open_water(
                            nhd_hr[st], st, args.include_nhdarea, args.min_lake_sqkm
                        )
                        if st in nhd_hr
                        else pd.DataFrame()
                    )
                else:  # wetland
                    df = (
                        extract_nwi_wetland(nwi[st], st, args.min_wetland_sqkm)
                        if st in nwi
                        else pd.DataFrame()
                    )
                df.to_parquet(cache)
            src = "" if df.empty else df["source"].iloc[0]
            coverage_rows.append(
                {
                    "state": st,
                    "class": cls,
                    "n_candidates": int(len(df)),
                    "primary_source": src,
                }
            )
            if df.empty:
                # a real gap only if no primary source existed for this (state,class)
                primary_missing = (
                    cls in ("spring", "open_water") and st not in nhd_hr
                ) or (cls == "wetland" and st not in nwi)
                if primary_missing:
                    gaps.append(
                        {"state": st, "class": cls, "reason": "no primary source"}
                    )
            else:
                cand_parts.append(df)

    # --- 3DHP fallback for states whose primary source was absent -------------
    for cls in classes:
        if cls == "wetland":
            continue  # 3DHP has no wetlands
        missing = [st for st in target_states if st not in nhd_hr]
        fb = extract_3dhp(cls, missing, state_polys, args.min_lake_sqkm)
        if not fb.empty:
            cand_parts.append(fb)
            for st in missing:
                coverage_rows.append(
                    {
                        "state": st,
                        "class": cls,
                        "n_candidates": int((fb["state"] == st).sum()),
                        "primary_source": "3dhp",
                    }
                )

    if not cand_parts:
        raise SystemExit("no anchor candidates extracted for the requested classes")
    cand = pd.concat(cand_parts, ignore_index=True)
    log.info("total candidates (pre-dedup): %d", len(cand))

    # --- head + uncertainty ---------------------------------------------------
    cand["head_m"] = sample_coarse(
        args.dem, cand["x5070"].to_numpy(), cand["y5070"].to_numpy()
    )
    cand["head_uncertainty_m"] = assign_head_uncertainty(cand)
    head_finite = float(np.isfinite(cand["head_m"]).mean())
    log.info("head_m finite fraction: %.4f", head_finite)

    # --- dedup per class (union-find cKDTree) ---------------------------------
    cand["dedup_group"] = pd.Series(np.empty(len(cand), dtype=object), index=cand.index)
    cand["is_primary"] = False
    for cls in classes:
        sel = (cand["anchor_class"] == cls).to_numpy()
        if not sel.any():
            continue
        radius = args.dedup_radius_m if cls == "spring" else args.dedup_radius_poly_m
        sub = cand.loc[sel].reset_index()
        group, primary = dedup_within_class(sub, radius)
        # namespace group ids per class to keep them globally unique
        cand.loc[sub["index"].to_numpy(), "dedup_group"] = [f"{cls}_{g}" for g in group]
        cand.loc[sub["index"].to_numpy(), "is_primary"] = primary
        log.info(
            "%s: %d candidates -> %d primary (dedup radius %.0fm)",
            cls,
            int(sel.sum()),
            int(primary.sum()),
            radius,
        )

    audit = cand.copy()
    nodes = cand[cand["is_primary"]].reset_index(drop=True).copy()
    nodes["anchor_node_idx"] = np.arange(len(nodes), dtype="int64")

    # --- attach each anchor to its controlling reach --------------------------
    reach_nodes = pd.read_parquet(args.reach_nodes, columns=["comid", "reach_node_idx"])
    comid_to_idx = dict(
        zip(
            reach_nodes["comid"].to_numpy("int64"),
            reach_nodes["reach_node_idx"].to_numpy("int64"),
        )
    )
    geom = gpd.read_parquet(args.reach_geom)
    geom["reach_node_idx"] = geom["comid"].map(comid_to_idx)
    geom = geom[geom["reach_node_idx"].notna()].copy()
    geom["reach_node_idx"] = geom["reach_node_idx"].astype("int64")
    axy = nodes[["x5070", "y5070"]].to_numpy("float64")
    edges = build_lateral_edges(axy, geom, comid_to_idx, args.knn_anchor)
    edges = edges.rename(
        columns={
            "query_node_idx": "anchor_node_idx",
            "lateral_dist_m": "anchor_dist_m",
            "log1p_lateral_dist_m": "log1p_anchor_dist_m",
        }
    )
    edges = edges[
        [
            "anchor_node_idx",
            "reach_node_idx",
            "anchor_dist_m",
            "log1p_anchor_dist_m",
            "rank",
            "is_controlling",
        ]
    ].copy()
    edges["is_within_R"] = (edges["anchor_dist_m"] <= args.max_attach_dist_m).astype(
        "float64"
    )
    n_far = int((edges["anchor_dist_m"] > args.max_attach_dist_m).sum())
    log.info(
        "anchor->reach edges: %d (%d beyond %.0fm, flagged not dropped)",
        len(edges),
        n_far,
        args.max_attach_dist_m,
    )

    # --- verification asserts (plan) ------------------------------------------
    ctrl = edges[edges["is_controlling"] == 1.0]
    assert ctrl["anchor_node_idx"].is_unique, "an anchor has >1 controlling reach"
    assert set(nodes["anchor_node_idx"]) <= set(edges["anchor_node_idx"]), (
        "anchor with no edge"
    )
    bad_3dhp = nodes[
        (nodes["state"].isin(["MT", "NV", "NM"])) & (nodes["source"] == "3dhp")
    ]
    if len(bad_3dhp):
        log.warning(
            "%d MT/NV/NM nodes from 3dhp (expected 0 -- NHD-HR present)", len(bad_3dhp)
        )

    # --- write outputs --------------------------------------------------------
    nodes[ANCHOR_NODE_COLS].to_parquet(out_dir / "anchor_nodes.parquet")
    edges.to_parquet(out_dir / "anchor_edges.parquet")
    audit.to_parquet(out_dir / "anchor_dedup_audit.parquet")
    cov = pd.DataFrame(coverage_rows)
    primary_by = {cls: int((nodes["anchor_class"] == cls).sum()) for cls in classes}
    cov.to_csv(out_dir / "anchor_coverage_qa.csv", index=False)

    manifest = {
        "stage": "anchors_v2",
        "crs": "EPSG:5070",
        "classes": classes,
        "counts": {
            "candidates_total": int(len(cand)),
            "primary_anchors": int(len(nodes)),
            "by_class": primary_by,
            "by_source": {k: int(v) for k, v in nodes["source"].value_counts().items()},
            "anchor_edges": int(len(edges)),
            "head_finite_frac": head_finite,
            "edges_beyond_max_attach": n_far,
        },
        "discovered": {
            "nhd_hr_states": sorted(nhd_hr.keys()),
            "nwi_states": sorted(nwi.keys()),
            "n_nhd_hr": len(nhd_hr),
            "n_nwi": len(nwi),
        },
        "coverage_gaps": gaps,
        "params": {
            "include_nhdarea": args.include_nhdarea,
            "min_lake_sqkm": args.min_lake_sqkm,
            "min_wetland_sqkm": args.min_wetland_sqkm,
            "dedup_radius_m": args.dedup_radius_m,
            "dedup_radius_poly_m": args.dedup_radius_poly_m,
            "knn_anchor": args.knn_anchor,
            "max_attach_dist_m": args.max_attach_dist_m,
        },
        "filters": {
            "spring": f"NHD-HR fcode=={NHD_SPRING_FCODE}; 3DHP landscape Spring",
            "open_water": f"NHD-HR perennial LakePond fcode in {sorted(NHD_PERENNIAL_LAKE_FCODES)}; "
            f"exclude reservoir 43xxx/playa 36100/intermittent 39010-11/swampmarsh; "
            f"NHDArea ftype {NHD_AREA_STREAMRIVER_FTYPE} behind --include-nhdarea; 3DHP Lake",
            "wetland": f"NWI ATTRIBUTE base in {NWI_KEEP_PREFIXES}; "
            f"exclude {NWI_EXCLUDE_PREFIXES} + modifiers {NWI_EXCLUDE_MODIFIERS}",
        },
        "head_uncertainty_m": HEAD_UNCERTAINTY,
        "leakage_notes": [
            "Anchors carry DEM head + fixed BC DTW=0, never an observed well DTW.",
            "head_m is for audit + relative-elevation edge attrs only; NEVER a model feature.",
            "Anchors are a fixed BC in all CV folds (no label to hold out).",
        ],
        "sources": {"dem": args.dem, "reach_geom": args.reach_geom, "3dhp": TDHP_GPKG},
    }
    (out_dir / "anchor_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    log.info(
        "wrote %d anchor_nodes + %d anchor_edges -> %s (classes=%s, gaps=%d)",
        len(nodes),
        len(edges),
        out_dir,
        classes,
        len(gaps),
    )


if __name__ == "__main__":
    main()
