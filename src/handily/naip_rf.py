"""NAIP Random Forest land-use classifier.

Assembles training polygons from compiled IrrMapper labels and NWI wetlands,
generates stratified random sample points, extracts NAIP bands + SARL class
via Earth Engine, trains a Random Forest classifier, and exports predictions.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, mapping
from shapely.validation import make_valid

from handily.ee.common import initialize_ee

LOGGER = logging.getLogger("handily.naip_rf")

TRAINING_DIR = "/nas/irrmapper/compiled_training_data/aea"
WETLANDS_DIR = "/nas/irrmapper/wetlands/raw_shp"
ANALYSIS_CRS = "EPSG:5070"

CLASS_SCHEMA = {
    "uncultivated": 0,
    "irrigated": 1,
    "surface_water": 2,
}

# Binary schema used at training time
BINARY_SCHEMA = {
    "other": 0,
    "surface_water": 1,
}

# Glob patterns for each class in the compiled training directory
_CLASS_FILE_PATTERNS = {
    "uncultivated": ["uncultivated*.shp"],
    "irrigated": ["irrigated*.shp", "irrigatd*.shp", "irrigation*.shp"],
}

# Cross-class erase priority (higher priority erases from lower)
_ERASE_PRIORITY = ["surface_water", "irrigated", "uncultivated"]

# Interior buffer (meters) before sampling — reduces mixed-pixel edge effects
_INNER_BUFFER = {
    "surface_water": -3.0,
    "irrigated": -2.0,
    "uncultivated": -2.0,
}

MIN_POLYGON_AREA_M2 = 16.0

NAIP_COLLECTION = "USDA/NAIP/DOQQ"
SARL_IMAGE = "projects/sat-io/open-datasets/SARL"

POINT_PROPS = [
    "sample_id",
    "state",
    "class_name",
    "class_code",
    "source_group",
    "source_file",
    "source_date",
    "spatial_fold",
]

NAIP_BANDS = ["naip_r", "naip_g", "naip_b", "naip_n"]
EXPORT_SELECTORS = POINT_PROPS + NAIP_BANDS + ["sarl_class", "naip_year"]


# ---------------------------------------------------------------------------
# State boundaries
# ---------------------------------------------------------------------------


STATE_BOUNDARIES_DIR = "/nas/boundaries/states"


def _load_state_boundaries(states: list[str]) -> gpd.GeoDataFrame:
    """Load state boundaries from local shapefiles or geopackages."""
    parts = []
    for st in states:
        candidates = [
            os.path.join(STATE_BOUNDARIES_DIR, f"{st}.shp"),
            os.path.join(STATE_BOUNDARIES_DIR, f"{st}_WGS.shp"),
            os.path.join(STATE_BOUNDARIES_DIR, f"STUSPS_{st}.gpkg"),
        ]
        path = next(
            (p for p in candidates if os.path.exists(p) and os.path.getsize(p) > 0),
            None,
        )
        if path is None:
            raise FileNotFoundError(f"State boundary not found: tried {candidates}")
        gdf = gpd.read_file(path)
        gdf = gdf.to_crs(ANALYSIS_CRS)
        gdf["state"] = st.upper()
        parts.append(gdf[["state", "geometry"]])
    if not parts:
        raise ValueError(f"No state boundaries found for {states}")
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Phase 1: Assemble training polygons
# ---------------------------------------------------------------------------


def _parse_source_date(filename: str) -> str:
    """Extract date string from filename like 'irrigated_11JAN2021.shp'."""
    m = re.search(r"_(\d{1,2}[A-Z]{3}\d{4})\.", filename)
    return m.group(1) if m else ""


def discover_training_sources(
    states: list[str],
    training_dir: str = TRAINING_DIR,
    wetlands_dir: str = WETLANDS_DIR,
) -> pd.DataFrame:
    """Build a manifest of all source shapefiles for the requested states."""
    rows = []

    for class_name, patterns in _CLASS_FILE_PATTERNS.items():
        for pattern in patterns:
            for path in sorted(glob.glob(os.path.join(training_dir, pattern))):
                fname = os.path.basename(path)
                rows.append(
                    {
                        "state": "ALL",
                        "class_name": class_name,
                        "source_file": path,
                        "source_group": "compiled_training",
                        "source_date": _parse_source_date(fname),
                    }
                )

    for st in states:
        pattern = re.compile(rf"^{st}_Wetlands.*\.shp$", re.IGNORECASE)
        for fname in sorted(os.listdir(wetlands_dir)):
            if pattern.match(fname):
                rows.append(
                    {
                        "state": st,
                        "class_name": "surface_water",
                        "source_file": os.path.join(wetlands_dir, fname),
                        "source_group": "state_wetlands_surface_water",
                        "source_date": "",
                    }
                )

    return pd.DataFrame(rows)


def _load_and_clip_compiled(
    manifest: pd.DataFrame,
    class_name: str,
    state_bounds: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Load all compiled training shapefiles for one class, clip to states."""
    class_files = manifest[
        (manifest["class_name"] == class_name)
        & (manifest["source_group"] == "compiled_training")
    ]
    if class_files.empty:
        return gpd.GeoDataFrame(columns=["class_name", "geometry"], crs=ANALYSIS_CRS)

    parts = []
    clip_geom = state_bounds.union_all()

    for _, row in class_files.iterrows():
        LOGGER.info("Loading %s: %s", class_name, row["source_file"])
        gdf = gpd.read_file(row["source_file"])
        gdf = gdf.to_crs(ANALYSIS_CRS)
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf["geometry"] = gdf["geometry"].apply(make_valid)
        gdf = gdf[~gdf.is_empty].copy()

        # Clip to state union
        gdf = gpd.clip(gdf, clip_geom)
        gdf = gdf[~gdf.is_empty].copy()
        if gdf.empty:
            continue

        gdf["class_name"] = class_name
        gdf["source_file"] = row["source_file"]
        gdf["source_date"] = row["source_date"]
        gdf["source_group"] = "compiled_training"
        parts.append(
            gdf[
                ["class_name", "source_file", "source_date", "source_group", "geometry"]
            ]
        )

    if not parts:
        return gpd.GeoDataFrame(columns=["class_name", "geometry"], crs=ANALYSIS_CRS)

    return pd.concat(parts, ignore_index=True)


def _load_surface_water(
    manifest: pd.DataFrame,
    state_bounds: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Load NWI wetlands filtered to Riverine + Lake."""
    sw_files = manifest[manifest["class_name"] == "surface_water"]
    if sw_files.empty:
        return gpd.GeoDataFrame(columns=["class_name", "geometry"], crs=ANALYSIS_CRS)

    parts = []
    for _, row in sw_files.iterrows():
        LOGGER.info("Loading surface_water: %s", row["source_file"])
        gdf = gpd.read_file(row["source_file"])
        gdf = gdf[gdf["WETLAND_TY"].isin(["Riverine", "Lake"])].copy()
        if gdf.empty:
            continue
        gdf = gdf.to_crs(ANALYSIS_CRS)
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf["geometry"] = gdf["geometry"].apply(make_valid)
        gdf = gdf[~gdf.is_empty].copy()

        # Clip to the state this file belongs to
        st = row["state"]
        st_geom = state_bounds[state_bounds["state"] == st].union_all()
        gdf = gpd.clip(gdf, st_geom)
        gdf = gdf[~gdf.is_empty].copy()
        if gdf.empty:
            continue

        gdf["class_name"] = "surface_water"
        gdf["source_file"] = row["source_file"]
        gdf["source_date"] = ""
        gdf["source_group"] = "state_wetlands_surface_water"
        parts.append(
            gdf[
                ["class_name", "source_file", "source_date", "source_group", "geometry"]
            ]
        )

    if not parts:
        return gpd.GeoDataFrame(columns=["class_name", "geometry"], crs=ANALYSIS_CRS)

    return pd.concat(parts, ignore_index=True)


def load_training_polygons(
    states: list[str],
    training_dir: str = TRAINING_DIR,
    wetlands_dir: str = WETLANDS_DIR,
) -> dict[str, gpd.GeoDataFrame]:
    """Load and clip training polygons for all classes.

    Returns a dict keyed by class_name with polygon GeoDataFrames.
    Cross-class overlap is handled at the point level after sampling.
    """
    states = [s.upper() for s in states]
    if not states:
        raise ValueError("states list cannot be empty")

    state_bounds = _load_state_boundaries(states)
    manifest = discover_training_sources(states, training_dir, wetlands_dir)
    LOGGER.info("Source manifest: %d entries", len(manifest))

    raw: dict[str, gpd.GeoDataFrame] = {}

    # Compiled training classes
    for class_name in _CLASS_FILE_PATTERNS:
        gdf = _load_and_clip_compiled(manifest, class_name, state_bounds)
        if not gdf.empty:
            raw[class_name] = gdf
            LOGGER.info("  %s: %d polygons clipped", class_name, len(gdf))

    # Surface water
    sw = _load_surface_water(manifest, state_bounds)
    if not sw.empty:
        raw["surface_water"] = sw
        LOGGER.info("  surface_water: %d polygons clipped", len(sw))

    # Min-area filter (no dissolve or cross-class erase — handled at point level)
    filtered: dict[str, gpd.GeoDataFrame] = {}
    for class_name, gdf in raw.items():
        gdf = gdf[gdf.area >= MIN_POLYGON_AREA_M2].reset_index(drop=True)
        filtered[class_name] = gdf
        LOGGER.info("  %s: %d polygons after min-area filter", class_name, len(gdf))

    # Assign state via spatial join
    for class_name in filtered:
        gdf = filtered[class_name]
        gdf = gdf.drop(columns=["state"], errors="ignore")
        gdf_repr = gdf.copy()
        gdf_repr["geometry"] = gdf_repr.representative_point()
        joined = gpd.sjoin(gdf_repr, state_bounds, how="left", predicate="within")
        gdf["state"] = joined["state"].fillna("UNK").values
        # Keep only requested states
        gdf = gdf[gdf["state"].isin(states)].reset_index(drop=True)
        filtered[class_name] = gdf

    return filtered


# ---------------------------------------------------------------------------
# Phase 2: Generate training points
# ---------------------------------------------------------------------------


def sample_training_points(
    polygons: dict[str, gpd.GeoDataFrame],
    n_water: int = 2000,
    m_other: int = 8000,
    seed: int = 42,
    fold_grid_m: float = 10_000.0,
) -> gpd.GeoDataFrame:
    """Generate stratified random sample points within training polygons.

    Parameters
    ----------
    polygons : dict of class_name -> GeoDataFrame
        Training polygons (may overlap across classes).
    n_water : int
        Total surface_water points.
    m_other : int
        Total points split across the four non-water classes.
    seed : int
        Random seed for reproducibility.
    fold_grid_m : float
        Grid cell size (meters) for spatial_fold assignment.

    Returns a GeoDataFrame with POINT_PROPS columns.
    """
    rng = np.random.default_rng(seed)

    other_classes = [c for c in _ERASE_PRIORITY if c != "surface_water"]
    base = m_other // len(other_classes)
    remainder = m_other % len(other_classes)
    class_counts = {}
    for i, c in enumerate(other_classes):
        class_counts[c] = base + (1 if i < remainder else 0)
    class_counts["surface_water"] = n_water

    all_points = []

    for class_name, target_n in class_counts.items():
        if class_name not in polygons or polygons[class_name].empty:
            LOGGER.warning("No polygons for %s, skipping", class_name)
            continue

        gdf = polygons[class_name].copy()

        # Interior buffer
        buf = _INNER_BUFFER.get(class_name, -2.0)
        gdf["buffered"] = gdf.geometry.buffer(buf)
        # Fall back to original if buffer collapses
        collapsed = gdf["buffered"].is_empty | gdf["buffered"].isna()
        gdf.loc[collapsed, "buffered"] = gdf.loc[collapsed, "geometry"]
        gdf = gdf[gdf["buffered"].area >= MIN_POLYGON_AREA_M2].reset_index(drop=True)

        if gdf.empty:
            LOGGER.warning("All polygons collapsed for %s after buffer", class_name)
            continue

        # Area-weighted polygon selection
        areas = gdf["buffered"].area.values
        weights = areas / areas.sum()

        points = []
        attempts = 0
        max_attempts = target_n * 20

        while len(points) < target_n and attempts < max_attempts:
            idx = rng.choice(len(gdf), p=weights)
            poly = gdf.iloc[idx]["buffered"]
            minx, miny, maxx, maxy = poly.bounds
            px = rng.uniform(minx, maxx)
            py = rng.uniform(miny, maxy)
            pt = Point(px, py)
            if poly.contains(pt):
                row = gdf.iloc[idx]
                points.append(
                    {
                        "class_name": class_name,
                        "class_code": CLASS_SCHEMA[class_name],
                        "state": row.get("state", "UNK"),
                        "source_group": row.get("source_group", ""),
                        "source_file": os.path.basename(row.get("source_file", "")),
                        "source_date": row.get("source_date", ""),
                        "geometry": pt,
                    }
                )
            attempts += 1

        LOGGER.info(
            "%s: requested %d, sampled %d (%d attempts)",
            class_name,
            target_n,
            len(points),
            attempts,
        )
        all_points.extend(points)

    result = gpd.GeoDataFrame(all_points, crs=ANALYSIS_CRS)

    # Post-sampling priority filter: drop points that fall in higher-priority polygons.
    # Build STRtree on the higher-priority polygons and query with points.
    from shapely import STRtree

    priority_rank = {c: i for i, c in enumerate(_ERASE_PRIORITY)}
    drop_idx = set()
    for class_name in _ERASE_PRIORITY[1:]:
        mask = result["class_name"] == class_name
        if not mask.any():
            continue
        rank = priority_rank[class_name]
        higher_classes = _ERASE_PRIORITY[:rank]
        higher_geoms = []
        for hc in higher_classes:
            if hc in polygons and not polygons[hc].empty:
                higher_geoms.extend(polygons[hc].geometry.tolist())
        if not higher_geoms:
            continue
        class_idx = result.index[mask]
        class_geoms = result.loc[mask, "geometry"].tolist()
        poly_tree = STRtree(higher_geoms)
        # hits[0] = input (point) indices, hits[1] = tree (polygon) indices
        hits = poly_tree.query(class_geoms, predicate="within")
        if hits[0].size > 0:
            conflict_positions = np.unique(hits[0])
            conflicts = class_idx[conflict_positions]
            drop_idx.update(conflicts)
            LOGGER.info(
                "  %s: dropping %d points inside higher-priority polygons",
                class_name,
                len(conflicts),
            )
    if drop_idx:
        result = result.drop(index=drop_idx).reset_index(drop=True)
        LOGGER.info(
            "Points after priority filter: %d (dropped %d)", len(result), len(drop_idx)
        )

    result["sample_id"] = range(len(result))

    # Spatial fold via coarse grid
    coords = np.column_stack([result.geometry.x, result.geometry.y])
    grid_x = (coords[:, 0] // fold_grid_m).astype(int)
    grid_y = (coords[:, 1] // fold_grid_m).astype(int)
    # Encode grid cell as unique integer
    result["spatial_fold"] = grid_x * 100_000 + grid_y

    LOGGER.info("Total sampled points: %d", len(result))
    for cn in CLASS_SCHEMA:
        n = (result["class_name"] == cn).sum()
        if n > 0:
            LOGGER.info("  %s: %d", cn, n)

    return result[POINT_PROPS + ["geometry"]]


# ---------------------------------------------------------------------------
# Phases 3-5: EE predictor stack and sampling
# ---------------------------------------------------------------------------


def _gdf_to_ee_fc(gdf: gpd.GeoDataFrame, keep_props: list[str]) -> ee.FeatureCollection:
    """Convert GeoDataFrame to EE FeatureCollection."""
    gdf_4326 = gdf.to_crs(epsg=4326)
    feats = []
    for _, row in gdf_4326.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = {}
        for k in keep_props:
            if k in row.index:
                val = row[k]
                if hasattr(val, "item"):
                    val = val.item()
                props[k] = val
        feats.append(ee.Feature(ee.Geometry(mapping(geom)), props))
    return ee.FeatureCollection(feats)


def _best_naip_year(geometry: ee.Geometry) -> int:
    """Find the NAIP year with best coverage (most images) for a geometry."""
    coll = ee.ImageCollection(NAIP_COLLECTION).filterBounds(geometry)
    years = coll.aggregate_array("year").getInfo()
    if not years:
        raise ValueError("No NAIP imagery found for the given geometry")
    from collections import Counter

    counts = Counter(years)
    # Pick the most recent among years tied for max image count
    max_count = max(counts.values())
    best = max(y for y, c in counts.items() if c == max_count)
    LOGGER.info(
        "NAIP year coverage: %s — selected %d (%d images)",
        dict(counts),
        best,
        max_count,
    )
    return best


def _build_naip_sarl_stack(year: int, geometry: ee.Geometry) -> ee.Image:
    """Build NAIP 4-band + SARL predictor image for a year and geometry."""
    naip = (
        ee.ImageCollection(NAIP_COLLECTION)
        .filterBounds(geometry)
        .filter(ee.Filter.eq("year", year))
        .mosaic()
    )
    naip_4 = naip.select(["R", "G", "B", "N"]).rename(NAIP_BANDS)
    sarl_year = min(year, 2021)  # SARL bands span Y1984–Y2021
    sarl = ee.Image(SARL_IMAGE).select([f"Y{sarl_year}"]).rename(["sarl_class"])
    return naip_4.addBands(sarl).set("naip_year", year)


def export_naip_training_table(
    states: list[str],
    n_water: int = 2000,
    m_other: int = 8000,
    seed: int = 42,
    ee_project: str | None = None,
    dest: str = "drive",
    bucket: str | None = None,
    drive_folder: str = "handily_naip_rf",
    points_path: str | None = None,
) -> dict:
    """End-to-end: assemble polygons, sample points, extract NAIP+SARL via EE.

    Parameters
    ----------
    states : list of str
        State abbreviations (e.g. ['NM']).
    n_water, m_other : int
        Point counts for surface_water and other classes.
    seed : int
        Random seed.
    ee_project : str, optional
        EE project for initialization.
    dest : str
        'drive' or 'bucket'.
    bucket : str, optional
        GCS bucket name (required if dest='bucket').
    drive_folder : str
        Google Drive folder for export.
    points_path : str, optional
        If provided, load pre-generated points instead of sampling new ones.

    Returns
    -------
    dict with keys: task, points_local, naip_years
    """
    states = [s.upper() for s in states]
    if not states:
        raise ValueError("states list cannot be empty")

    initialize_ee(ee_project)

    # Phase 1-2: polygons + points
    if points_path and os.path.exists(points_path):
        LOGGER.info("Loading pre-generated points from %s", points_path)
        if points_path.endswith(".parquet"):
            points_gdf = gpd.read_parquet(points_path)
        else:
            points_gdf = gpd.read_file(points_path)
    else:
        LOGGER.info("Phase 1: Assembling training polygons for %s", states)
        polygons = load_training_polygons(states)
        LOGGER.info("Phase 2: Generating sample points")
        points_gdf = sample_training_points(
            polygons, n_water=n_water, m_other=m_other, seed=seed
        )
        # Save local artifacts
        points_gdf.to_parquet("/tmp/naip_rf_training_points.parquet")
        points_gdf.to_file("/tmp/naip_rf_training_points.fgb", driver="FlatGeobuf")
        LOGGER.info("Saved points to /tmp/naip_rf_training_points.{parquet,fgb}")

    # Phase 4-5: per-state NAIP stack + sampleRegions
    # Load state boundaries for EE filterBounds (simple polygon, not 8k-point union)
    state_bounds = _load_state_boundaries(states)
    unique_states = sorted(points_gdf["state"].unique())
    naip_years = {}
    sampled_collections = []

    for st in unique_states:
        st_mask = points_gdf["state"] == st
        st_points = points_gdf[st_mask]
        st_fc = _gdf_to_ee_fc(st_points, keep_props=POINT_PROPS)

        # Use state boundary polygon for filterBounds — much cheaper than point union
        st_bound = state_bounds[state_bounds["state"] == st].to_crs(epsg=4326)
        st_geom_ee = ee.Geometry(mapping(st_bound.union_all()))

        year = _best_naip_year(st_geom_ee)
        naip_years[st] = year
        LOGGER.info(
            "State %s: latest NAIP year = %d (%d points)", st, year, len(st_points)
        )

        stack = _build_naip_sarl_stack(year, st_geom_ee)

        sampled = stack.sampleRegions(
            collection=st_fc,
            properties=POINT_PROPS,
            scale=1,
            geometries=False,
            tileScale=8,
        )
        # Attach naip_year as a property on every feature
        sampled = sampled.map(lambda f, y=year: f.set("naip_year", y))
        sampled_collections.append(sampled)

    # Merge all state collections
    if not sampled_collections:
        raise ValueError("No points survived sampling — nothing to export")
    merged = sampled_collections[0]
    for sc in sampled_collections[1:]:
        merged = merged.merge(sc)

    # Phase 6: export
    states_str = "_".join(sorted(states)).lower()
    desc = f"naip_rf_training_{states_str}"

    if dest == "bucket":
        if not bucket:
            raise ValueError("bucket required for bucket export")
        prefix = f"handily/naip_rf/{desc}"
        task = ee.batch.Export.table.toCloudStorage(
            collection=merged,
            description=desc,
            bucket=bucket,
            fileNamePrefix=prefix,
            fileFormat="CSV",
            selectors=EXPORT_SELECTORS,
        )
    else:
        task = ee.batch.Export.table.toDrive(
            collection=merged,
            description=desc,
            folder=drive_folder,
            fileNamePrefix=desc,
            fileFormat="CSV",
            selectors=EXPORT_SELECTORS,
        )

    task.start()
    LOGGER.info("Started EE export: %s", desc)

    return {
        "task": task,
        "description": desc,
        "points_local": "/tmp/naip_rf_training_points.parquet",
        "naip_years": naip_years,
        "n_points": len(points_gdf),
    }


# ---------------------------------------------------------------------------
# Phase 7: Random Forest training and evaluation
# ---------------------------------------------------------------------------

FEATURE_COLS = NAIP_BANDS + ["sarl_class"]
TARGET_COL = "class_code"
CLASS_NAMES = list(BINARY_SCHEMA.keys())


def train_rf_classifier(
    training_csv: str,
    n_estimators: int = 200,
    test_fraction: float = 0.2,
    seed: int = 42,
    out_dir: str | None = None,
) -> dict:
    """Train a Random Forest classifier on EE-extracted NAIP+SARL features.

    Uses spatial_fold for group-aware train/test splitting so nearby points
    don't leak across the split boundary.

    Parameters
    ----------
    training_csv : str
        Path to the CSV exported by ``export_naip_training_table``.
    n_estimators : int
        Number of trees.
    test_fraction : float
        Approximate fraction of spatial folds held out for testing.
    seed : int
        Random seed.
    out_dir : str, optional
        Directory for model + report artifacts. Defaults to same dir as CSV.

    Returns
    -------
    dict with keys: model_path, accuracy, report, confusion_matrix, feature_importance
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    import joblib

    df = pd.read_csv(training_csv)
    LOGGER.info("Loaded %d rows from %s", len(df), training_csv)

    # Drop rows with any missing feature values
    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    if len(df) < n_before:
        LOGGER.warning("Dropped %d rows with NaN features", n_before - len(df))

    # Binary remap: surface_water=1, everything else=0 ("other")
    df["class_name"] = df["class_name"].apply(
        lambda x: x if x == "surface_water" else "other"
    )
    df[TARGET_COL] = df["class_name"].map(BINARY_SCHEMA)

    # Filter surface_water to only points where SARL confirms non-zero class
    sw_mask = df[TARGET_COL] == BINARY_SCHEMA["surface_water"]
    sw_sarl_drop = sw_mask & (df["sarl_class"] == 0)
    if sw_sarl_drop.any():
        LOGGER.info(
            "Dropping %d surface_water points with sarl_class=0", sw_sarl_drop.sum()
        )
        df = df[~sw_sarl_drop].reset_index(drop=True)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Spatial-fold group split: assign each unique fold to train or test
    rng = np.random.default_rng(seed)
    folds = df["spatial_fold"].values
    unique_folds = np.unique(folds)
    rng.shuffle(unique_folds)
    n_test_folds = max(1, int(len(unique_folds) * test_fraction))
    test_folds = set(unique_folds[:n_test_folds])

    test_mask = np.isin(folds, list(test_folds))
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    LOGGER.info(
        "Split: %d train (%d folds), %d test (%d folds)",
        train_mask.sum(),
        len(unique_folds) - n_test_folds,
        test_mask.sum(),
        n_test_folds,
    )

    # Train
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    importances = dict(zip(FEATURE_COLS, clf.feature_importances_))

    LOGGER.info("Accuracy: %.4f", acc)
    LOGGER.info("Classification report:\n%s", report)
    LOGGER.info("Confusion matrix:\n%s", cm)
    LOGGER.info("Feature importances: %s", importances)

    # Save artifacts
    if out_dir is None:
        out_dir = os.path.dirname(training_csv)
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "naip_rf_model.joblib")
    joblib.dump(clf, model_path)
    LOGGER.info("Saved model to %s", model_path)

    report_path = os.path.join(out_dir, "naip_rf_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Training CSV: {training_csv}\n")
        f.write(f"Train: {train_mask.sum()}  Test: {test_mask.sum()}\n")
        f.write(f"Spatial folds: {len(unique_folds)} total, {n_test_folds} held out\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(f"Classes: {CLASS_NAMES}\n")
        f.write(np.array2string(cm))
        f.write("\n\nFeature importances:\n")
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
            f.write(f"  {feat}: {imp:.4f}\n")
    LOGGER.info("Saved report to %s", report_path)

    return {
        "model_path": model_path,
        "report_path": report_path,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "feature_importance": importances,
    }
