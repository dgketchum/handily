"""E6: land-side shoreline ring pseudo-label points at permanent natural lakes.

Mirror of the water-stage pseudo-label idea (build_conus_graph_inputs.py
--water-pseudo-labels), but on LAND: the existing is_water_pseudo nodes sit ON
permanent water (occ>=90); the gate has no supervised DTW=0 evidence on the
adjacent LAND cells where the shallow prior over-deepens. This builder emits one
land-side query row per 100 m lattice cell just outside a perennial natural lake,
labelled DTW=0 (the water table meets the ground at the shore), for the trainer to
supervise the saturated shallow extreme.

Target-blind losing-reach screen (Janssen V2-inspired, dose-disciplined):
  * NHD NHDWaterbody, FType 390 (LakePond) only, perennial FCode subset
    {39004,39009,39010,39011,39012} -- excludes Reservoir 436 (+tailings/sewage/
    evaporator), Playa 361, SwampMarsh 466, Ice 378, Estuary 493, and
    intermittent/unspecified LakePond 39000/39001/39005/39006 by construction.
  * NHDArea is NOT used (skipping StreamRiver/CanalDitch/Wash/InundationArea is the
    losing-reach screen in its simplest form).
  * polygon area >= 0.1 km^2 (EPSG:5070 equal-area).
  * PER SHORELINE SEGMENT GSW permanence: JRC GSW occurrence >= 90 at a probe ~75 m
    INSIDE the polygon (kills drawdown-zone shorelines + stale digitizations locally).
  * land cell own GSW occurrence < 50 (genuinely land).
  * NO observed-WTE screen (rejected: leakage).

Geometry / lattice:
  * densify each polygon boundary at ~250 m; for each vertex take the canonical
    100 m EPSG:5070 lattice cell (origin -2540000/3258000, same grid the bundle's
    covariate/embedding sampling uses) on the LAND side whose center is OUTSIDE the
    polygon and within 150 m of the boundary. One point per lattice cell (dedupe).
  * drop cells colliding (same lattice cell) with any existing v2 query node.
  * finiteness screen on the 22 all-finite v2 query-feature columns (fac_rem + DEM +
    relief/terrain rasters) so the injected rows cannot widen the trainer's
    NaN-indicator matrix (feat dim must stay 46).

Dose (same machinery as water pseudo):
  * within 100 km of a real monitoring well (v2 real query nodes);
  * per-HUC8 cap 50 (seeded random subsample), seed 0.

    uv run python utils/build_shoreline_points.py \
        --bundle /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2 \
        --out-dir /data/ssd2/handily/conus/wte_gnn/shoreline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry.polygon import orient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_conus_graph_inputs as bcg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_shoreline_points")

# --- screens / geometry constants ------------------------------------------------
FTYPE_LAKEPOND = 390
PERENNIAL_FCODES = {39004, 39009, 39010, 39011, 39012}
MIN_AREA_KM2 = 0.1
DENSIFY_M = 250.0
LAND_OFFSET_M = 100.0  # outward step from a boundary vertex toward the land cell
PROBE_INSIDE_M = 75.0  # GSW permanence probe, inside the polygon
MAX_BOUNDARY_DIST_M = 150.0  # land cell center must be within this of the boundary
GSW_WATER_OCC = 90.0  # segment permanence: occ>=90 at the inside probe
GSW_LAND_OCC = 50.0  # land cell must be genuinely land: occ<50
# canonical 100 m EPSG:5070 lattice (dist_to_stream_m.tif / covariate grid origin)
LAT_X0, LAT_Y0, LAT_RES = -2540000.0, 3258000.0, 100.0
# v2 query-feature columns that are all-finite over the base bundle; shore rows MUST
# keep these finite (else the trainer's per-NaN-col indicator matrix widens).
MUST_FINITE_RELIEF = [
    "slope_deg",
    "tri_100m",
    "dist_to_stream_m",
    "elev_above_coarse_m",
]
MUST_FINITE_TERRAIN = [
    "haf_500m",
    "haf_2km",
    "haf_5km",
    "haf_10km",
    "tpi_500m",
    "tpi_2km",
    "tpi_5km",
    "tpi_10km",
]
DOSE_MAX_WELL_KM = 100.0
PER_HUC8_CAP = 50
SEED = 0


def cell_ids(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Canonical 100 m lattice (col, row) integer cell ids for EPSG:5070 points."""
    col = np.floor((np.asarray(x, "float64") - LAT_X0) / LAT_RES).astype("int64")
    row = np.floor((LAT_Y0 - np.asarray(y, "float64")) / LAT_RES).astype("int64")
    return col * 100_000_000 + row


def cell_centers(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Snap EPSG:5070 points to their canonical 100 m lattice cell centers."""
    col = np.floor((np.asarray(x, "float64") - LAT_X0) / LAT_RES)
    row = np.floor((LAT_Y0 - np.asarray(y, "float64")) / LAT_RES)
    cx = LAT_X0 + (col + 0.5) * LAT_RES
    cy = LAT_Y0 - (row + 0.5) * LAT_RES
    return cx, cy


def gsw_occ_5070(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """JRC GSW occurrence (0-100) at EPSG:5070 points (transform to 4326 + sample)."""
    if len(x) == 0:
        return np.zeros(0)
    lon, lat = Transformer.from_crs(5070, 4326, always_xy=True).transform(x, y)
    return bcg._sample_gsw_occurrence(np.asarray(lon), np.asarray(lat))


def _pt_seg_dist(px, py, ax, ay, bx, by) -> np.ndarray:
    """Vectorized distance from points p to segments a->b (all equal-length arrays)."""
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    t = np.where(
        denom > 0.0, ((px - ax) * abx + (py - ay) * aby) / np.maximum(denom, 1e-12), 0.0
    )
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))


def polygon_ring_candidates(poly) -> tuple[np.ndarray, np.ndarray]:
    """Land-cell centers + inside-probe points for one polygon's densified boundary.

    Returns (land_xy, probe_xy) as (N,2) arrays. Land cell = the canonical 100 m
    lattice cell one LAND_OFFSET_M step along the outward normal; probe = PROBE_INSIDE_M
    inward. Vertices whose land cell is inside the polygon or > MAX_BOUNDARY_DIST_M from
    the boundary are dropped.
    """
    poly = orient(poly, 1.0)  # exterior CCW -> interior on the LEFT of travel
    ext = shapely.segmentize(poly.exterior, DENSIFY_M)
    c = np.asarray(ext.coords, "float64")[:, :2]  # drop any Z (NHD is 3D)
    if len(c) < 4:
        return np.zeros((0, 2)), np.zeros((0, 2))
    c = c[:-1]  # drop closing duplicate
    prev = np.roll(c, 1, axis=0)
    nxt = np.roll(c, -1, axis=0)
    t = nxt - prev
    tn = np.hypot(t[:, 0], t[:, 1])
    ok = tn > 1e-9
    if not ok.any():
        return np.zeros((0, 2)), np.zeros((0, 2))
    c, prev, nxt, t, tn = c[ok], prev[ok], nxt[ok], t[ok], tn[ok]
    # outward normal for a CCW ring = right-hand perpendicular (dy, -dx)
    n_out = np.column_stack([t[:, 1], -t[:, 0]]) / tn[:, None]
    land = c + n_out * LAND_OFFSET_M
    lcx, lcy = cell_centers(land[:, 0], land[:, 1])
    probe = c - n_out * PROBE_INSIDE_M
    lc = np.column_stack([lcx, lcy])
    # boundary distance = min distance to the two densified segments adjacent to the
    # source vertex (the nearest boundary point to a 100 m outward offset is on one of
    # them); exact-enough for the 150 m gate and ~200x faster than distance-to-exterior.
    dbnd = np.minimum(
        _pt_seg_dist(lcx, lcy, prev[:, 0], prev[:, 1], c[:, 0], c[:, 1]),
        _pt_seg_dist(lcx, lcy, c[:, 0], c[:, 1], nxt[:, 0], nxt[:, 1]),
    )
    # land cell center must be OUTSIDE the polygon and within MAX_BOUNDARY_DIST_M
    shapely.prepare(poly)
    outside = ~shapely.contains_xy(poly, lcx, lcy)
    keep = outside & (dbnd <= MAX_BOUNDARY_DIST_M)
    return lc[keep], probe[keep]


def state_candidates(shp: Path) -> tuple[pd.DataFrame, dict]:
    """Screened land-side ring cells for one state's NHDWaterbody (funnel in meta)."""
    # Push the FType/FCode filter down to OGR (reads ~10k perennial LakePond geoms
    # instead of all ~50k features); raw count comes free from the layer metadata.
    fcodes = ",".join(str(c) for c in sorted(PERENNIAL_FCODES))
    where = f"ftype = {FTYPE_LAKEPOND} AND fcode IN ({fcodes})"
    meta = {"raw": int(pyogrio.read_info(str(shp))["features"])}
    gdf = gpd.read_file(shp, engine="pyogrio", columns=["ftype", "fcode"], where=where)
    meta["perennial"] = int(len(gdf))
    gdf = gdf.to_crs(5070)
    gdf = gdf[gdf.geometry.area >= MIN_AREA_KM2 * 1e6]
    meta["area_ok"] = int(len(gdf))
    if len(gdf) == 0:
        meta.update(gsw_verified=0, ring_cells=0)
        return pd.DataFrame(columns=["x5070", "y5070"]), meta
    land_parts, probe_parts = [], []
    for geom in gdf.geometry.values:
        if geom is None or geom.is_empty:
            continue
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for p in polys:
            lc, pr = polygon_ring_candidates(p)
            if len(lc):
                land_parts.append(lc)
                probe_parts.append(pr)
    if not land_parts:
        meta.update(gsw_verified=0, ring_cells=0)
        return pd.DataFrame(columns=["x5070", "y5070"]), meta
    land = np.concatenate(land_parts)
    probe = np.concatenate(probe_parts)
    # per-segment GSW permanence (probe inside >= 90) + land cell genuinely land (< 50)
    n = len(land)
    both = np.vstack([probe, land])
    occ = gsw_occ_5070(both[:, 0], both[:, 1])
    occ_probe, occ_land = occ[:n], occ[n:]
    verified = (
        np.isfinite(occ_probe)
        & (occ_probe >= GSW_WATER_OCC)
        & np.isfinite(occ_land)
        & (occ_land < GSW_LAND_OCC)
    )
    meta["gsw_verified"] = int(verified.sum())
    land = land[verified]
    if len(land) == 0:
        meta["ring_cells"] = 0
        return pd.DataFrame(columns=["x5070", "y5070"]), meta
    # one point per lattice cell
    cid = cell_ids(land[:, 0], land[:, 1])
    _, first = np.unique(cid, return_index=True)
    land = land[np.sort(first)]
    meta["ring_cells"] = int(len(land))
    return pd.DataFrame({"x5070": land[:, 0], "y5070": land[:, 1]}), meta


def finiteness_screen(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows finite on fac_rem, DEM z_surf, and the all-finite relief/terrain cols.

    Reuses the EXACT builder samplers so the injected rows match the bundle's feature
    finiteness (feat dim stays 46). Adds z_surf (DEM) for the label + audit. Called
    ONCE globally (sample_fac_rem reopens all 997 registry basins per call).
    """
    x = df["x5070"].to_numpy("float64")
    y = df["y5070"].to_numpy("float64")
    fac = bcg.sample_fac_rem(x, y)
    zsurf = bcg.sample_coarse(bcg.DEM, x, y)
    keep = np.isfinite(fac) & np.isfinite(zsurf)
    x, y, zsurf = x[keep], y[keep], zsurf[keep]
    df = df.loc[keep].reset_index(drop=True)
    relief = bcg.sample_relief_etrm(x, y, zsurf)
    terrain = bcg.sample_terrain_multiscale(x, y)
    fin = np.ones(len(x), bool)
    for col in MUST_FINITE_RELIEF:
        fin &= np.isfinite(relief[col])
    for col in MUST_FINITE_TERRAIN:
        fin &= np.isfinite(terrain[col])
    df = df.loc[fin].reset_index(drop=True)
    df["z_surf"] = zsurf[fin]
    return df


def assign_huc8(df: pd.DataFrame) -> pd.DataFrame:
    """WBD point-in-polygon HUC8; rows matching no polygon are dropped (coastal/border)."""
    polys = gpd.read_parquet(bcg.WBD_HU8_PARQUET)[["huc8", "geometry"]]
    if polys.crs is None or polys.crs.to_epsg() != 5070:
        raise SystemExit(f"HUC8 polys not EPSG:5070: {bcg.WBD_HU8_PARQUET}")
    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["x5070"], df["y5070"]), crs=5070
    )
    joined = gpd.sjoin(pts, polys, predicate="within", how="left")
    joined = joined[~joined.index.duplicated(keep="first")].sort_index()
    df = df.copy()
    df["huc8"] = joined["huc8"].to_numpy(object)
    n0 = len(df)
    df = df[df["huc8"].notna()].reset_index(drop=True)
    df["huc8"] = df["huc8"].astype(str)
    if len(df) < n0:
        log.info(
            "HUC8 assign: %d/%d cells match no HUC8 polygon -- dropped",
            n0 - len(df),
            n0,
        )
    return df


def label_plausibility_audit(allc: pd.DataFrame, bundle: Path, out_path: Path) -> None:
    """Report-only (NO gating): for retained ring points with an unconfined well within
    2 km, tabulate (ring z_surf - well WTE) overall and by HUC2. A DTW=0 shore label is
    plausible where the shore ground elevation sits near the nearby water-table altitude;
    a large positive tail flags rings on high ground above the regional table.
    """
    qn = pd.read_parquet(
        bundle / "query_nodes.parquet",
        columns=["x5070", "y5070", "is_water_pseudo", "wte_obs_m", "confinement_class"],
    )
    well = qn[
        (~qn["is_water_pseudo"].astype(bool))
        & qn["confinement_class"].isin({"unconfined", "unconfined_marginal"})
        & np.isfinite(qn["wte_obs_m"].to_numpy("float64"))
    ]
    wxy = well[["x5070", "y5070"]].to_numpy("float64")
    wwte = well["wte_obs_m"].to_numpy("float64")
    d, idx = cKDTree(wxy).query(allc[["x5070", "y5070"]].to_numpy("float64"), k=1)
    within = d <= 2000.0
    diff = allc["z_surf"].to_numpy("float64")[within] - wwte[idx[within]]
    huc2 = allc["huc8"].astype(str).str[:2].to_numpy()[within]

    def _q(a):
        p = np.percentile(a, [5, 25, 50, 75, 95])
        return (
            f"n={len(a)} mean={a.mean():+.2f} median={np.median(a):+.2f} "
            f"sd={a.std():.2f} p5={p[0]:+.2f} p25={p[1]:+.2f} p50={p[2]:+.2f} "
            f"p75={p[3]:+.2f} p95={p[4]:+.2f}"
        )

    lines = [
        "E6 shoreline-ring label-plausibility audit (report-only, NOT a gate)",
        "metric: (ring land-surface elevation z_surf - nearest unconfined well WTE), meters",
        "positive = ring ground above the well water-table altitude within 2 km",
        f"retained ring points: {len(allc)}; with unconfined well <=2 km: {int(within.sum())}",
        "",
        f"OVERALL  {_q(diff)}"
        if within.any()
        else "OVERALL  n=0 (no ring within 2 km of a well)",
        "",
        "by HUC2:",
    ]
    for h in sorted(set(huc2)):
        a = diff[huc2 == h]
        if len(a):
            lines.append(f"  {h}  {_q(a)}")
    out_path.write_text("\n".join(lines) + "\n")
    log.info(
        "label-plausibility audit -> %s (%d/%d rings matched a well <=2 km)",
        out_path,
        int(within.sum()),
        len(allc),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bundle",
        default="/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2",
    )
    ap.add_argument("--out-dir", default="/data/ssd2/handily/conus/wte_gnn/shoreline")
    ap.add_argument("--nhd-dir", default="/nas/hydrography/nhd/hr_state")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "state_cache").mkdir(parents=True, exist_ok=True)

    # existing v2 query nodes: cell ids for collision, real wells for the 100 km dose
    qn = pd.read_parquet(
        Path(args.bundle) / "query_nodes.parquet",
        columns=["x5070", "y5070", "is_water_pseudo"],
    )
    existing_cids = set(
        cell_ids(qn["x5070"].to_numpy(), qn["y5070"].to_numpy()).tolist()
    )
    real = qn[~qn["is_water_pseudo"].astype(bool)]
    well_xy = real[["x5070", "y5070"]].to_numpy("float64")
    log.info(
        "v2 query nodes: %d (%d real wells for the dose); %d occupied lattice cells",
        len(qn),
        len(real),
        len(existing_cids),
    )

    # NHDWaterbody.shp, plus the _0/_1 splits large states (FL, TX) ship instead.
    shps = sorted(
        Path(args.nhd_dir).glob("NHD_H_*_State_Shape/Shape/NHDWaterbody*.shp")
    )
    log.info("NHDWaterbody shapefiles: %d (state files, incl. _N splits)", len(shps))
    all_states = sorted(Path(args.nhd_dir).glob("NHD_H_*_State_Shape"))
    missing = [
        d.name for d in all_states if not list((d / "Shape").glob("NHDWaterbody*.shp"))
    ]
    if missing:
        log.info("state dirs without any NHDWaterbody*.shp (skipped): %s", missing)

    # Per state: geometry + per-segment GSW screen + per-cell dedupe only. Collision,
    # HUC8, dose, and the (registry-expensive) finiteness screen run ONCE globally.
    frames = []
    for shp in shps:
        state = shp.parents[1].name.replace("NHD_H_", "").replace("_State_Shape", "")
        # unique cache/log tag per file so split states (Florida_0/_1) don't collide
        tag = f"{state}{shp.stem.replace('NHDWaterbody', '')}"
        cache = out_dir / "state_cache" / f"{tag}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            log.info("[%s] cached: %d screened land cells", tag, len(df))
            frames.append(df)
            continue
        try:
            cand, meta = state_candidates(shp)
        except Exception as e:  # noqa: BLE001 -- log + continue, resume-safe
            log.warning("[%s] FAILED: %s", tag, e)
            continue
        cand["state"] = state
        cand.to_parquet(cache)
        log.info(
            "[%s] funnel raw=%d perennial=%d area>=0.1km2=%d gsw_verified=%d "
            "ring_cells=%d",
            tag,
            meta["raw"],
            meta["perennial"],
            meta["area_ok"],
            meta["gsw_verified"],
            meta["ring_cells"],
        )
        frames.append(cand)

    allc = pd.concat([f for f in frames if len(f)], ignore_index=True)
    log.info("all states: %d screened land cells", len(allc))
    # global dedupe by lattice cell (border lakes appear in >1 state file)
    cid = cell_ids(allc["x5070"].to_numpy(), allc["y5070"].to_numpy())
    allc = allc.loc[~pd.Series(cid).duplicated().to_numpy()].reset_index(drop=True)
    log.info("after global cell dedupe: %d", len(allc))
    # collision with existing v2 query nodes (any class, same lattice cell)
    cid = cell_ids(allc["x5070"].to_numpy(), allc["y5070"].to_numpy())
    allc = allc.loc[~pd.Series(cid).isin(existing_cids).to_numpy()].reset_index(
        drop=True
    )
    log.info("after v2 query-node collision drop: %d", len(allc))
    # HUC8 assignment (national WBD point-in-polygon)
    allc = assign_huc8(allc)
    log.info("after HUC8 assignment: %d", len(allc))
    # dose 1: within 100 km of a real well
    d = cKDTree(well_xy).query(allc[["x5070", "y5070"]].to_numpy("float64"), k=1)[0]
    allc = allc[d <= DOSE_MAX_WELL_KM * 1000.0].reset_index(drop=True)
    log.info("after within-%gkm-of-well: %d", DOSE_MAX_WELL_KM, len(allc))
    # finiteness screen (fac_rem + DEM + relief/terrain; adds z_surf) -- ONCE
    allc = finiteness_screen(allc)
    log.info("after finiteness screen (feat dim stays 46): %d", len(allc))
    # dose 2: per-HUC8 cap (seeded)
    rng = np.random.RandomState(SEED)
    allc = allc.sort_values(["x5070", "y5070"]).reset_index(drop=True)
    keep_idx = []
    for _, grp in allc.groupby("huc8", sort=True):
        take = grp.index.to_numpy()
        if len(take) > PER_HUC8_CAP:
            take = rng.choice(take, size=PER_HUC8_CAP, replace=False)
        keep_idx.append(take)
    allc = allc.loc[np.sort(np.concatenate(keep_idx))].reset_index(drop=True)
    log.info(
        "after per-HUC8 cap %d: %d points in %d HUC8s",
        PER_HUC8_CAP,
        len(allc),
        allc["huc8"].nunique(),
    )

    allc["mean_dtw"] = 0.0
    allc["wte_obs"] = allc[
        "z_surf"
    ]  # DTW=0 -> wte_obs = z_surf (water-pseudo convention)
    out = out_dir / "shoreline_ring_points.parquet"
    allc.to_parquet(out)
    log.info("wrote %d shoreline ring points -> %s", len(allc), out)

    label_plausibility_audit(allc, Path(args.bundle), out_dir / "label_audit.txt")


if __name__ == "__main__":
    main()
