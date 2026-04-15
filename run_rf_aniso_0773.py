"""RF water mask + anisotropic frame REM for NV 0773."""
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

import glob
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import geopandas as gpd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from handily import compute, rem_frame
from handily.config import HandilyConfig

aoi_dir = "/data/ssd2/handily/nv/aoi_0773"
rf_dir = "/data/ssd2/handily/naip_rf"

# ---- Step 1: Train RF (NAIP-only, no SARL) ----
print("=== Step 1: Train RF (NAIP bands only) ===")
df = pd.read_csv(f"{rf_dir}/naip_rf_training_combined.csv")
df["class_name"] = df["class_name"].apply(
    lambda x: x if x == "surface_water" else "other"
)
binary_schema = {"other": 0, "surface_water": 1}
df["class_code"] = df["class_name"].map(binary_schema)
feature_cols = ["naip_r", "naip_g", "naip_b", "naip_n"]
df = df.dropna(subset=feature_cols + ["class_code"]).reset_index(drop=True)
X, y = df[feature_cols].values, df["class_code"].values

rng = np.random.default_rng(42)
folds = df["spatial_fold"].values
uf = np.unique(folds)
rng.shuffle(uf)
n_test = max(1, int(len(uf) * 0.2))
test_mask = np.isin(folds, list(uf[:n_test]))
MAX_RF_WORKERS = 12
clf = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=MAX_RF_WORKERS, class_weight="balanced"
)
clf.fit(X[~test_mask], y[~test_mask])
acc = accuracy_score(y[test_mask], clf.predict(X[test_mask]))
print(f"  Accuracy: {acc:.4f}")
model_path = f"{rf_dir}/naip_rf_model_nosrl.joblib"
joblib.dump(clf, model_path)

# ---- Step 2: Predict on DEM-aligned NAIP (chunked) ----
print("=== Step 2: Predict RF water mask (DEM grid, chunked) ===")
dem_da = rioxarray.open_rasterio(f"{aoi_dir}/dem_bounds_1m.tif").squeeze(
    "band", drop=True
)
print(f"  DEM: {dem_da.shape} crs={dem_da.rio.crs}")

naip_path = glob.glob(f"{aoi_dir}/naip/*.tif")[0]
naip_raw = rioxarray.open_rasterio(naip_path)
print(f"  NAIP raw: {naip_raw.shape} crs={naip_raw.rio.crs}")

naip_reproj = naip_raw.sel(band=[1, 2, 3, 4]).rio.reproject_match(dem_da)
print(f"  NAIP reprojected: {naip_reproj.shape}")

r = naip_reproj.sel(band=1).values.astype(np.float32)
g = naip_reproj.sel(band=2).values.astype(np.float32)
b = naip_reproj.sel(band=3).values.astype(np.float32)
n = naip_reproj.sel(band=4).values.astype(np.float32)
del naip_raw, naip_reproj

h, w = r.shape
pred = np.zeros((h, w), dtype=np.uint8)
chunk_rows = 2000
for row_start in range(0, h, chunk_rows):
    row_end = min(row_start + chunk_rows, h)
    pixels = np.column_stack([
        r[row_start:row_end].ravel(),
        g[row_start:row_end].ravel(),
        b[row_start:row_end].ravel(),
        n[row_start:row_end].ravel(),
    ])
    pred[row_start:row_end] = clf.predict(pixels).reshape(row_end - row_start, w)
    if row_start % 4000 == 0:
        print(f"  Rows {row_start}-{row_end} / {h}")

n_water = int(pred.sum())
print(f"  Water pixels: {n_water} ({n_water / pred.size * 100:.2f}%)")

rf_da = xr.DataArray(pred, dims=dem_da.dims, coords=dem_da.coords, name="water_rf")
rf_da = rf_da.rio.write_crs(dem_da.rio.crs, inplace=False)
rf_da.rio.to_raster(f"{aoi_dir}/water_mask_rf.tif")
print(f"  Written: {aoi_dir}/water_mask_rf.tif")
del r, g, b, n, pred

# ---- Step 3: Propagate + anisotropic REM ----
print("=== Step 3: Propagate RF mask + anisotropic frame REM ===")
flowlines = gpd.read_file(f"{aoi_dir}/flowlines_bounds.fgb")
flowlines_dem = flowlines.to_crs(dem_da.rio.crs)
flowlines_for_mask = compute.filter_disconnected_flowlines(flowlines_dem)

annotated = compute.propagate_flowline_confirmation(
    flowlines_for_mask, dem_da, ndwi_da=rf_da,
    ndwi_threshold=0.5, flowlines_buffer_m=10.0, max_hops=None,
)
seeded = annotated[annotated["water_seeded"]].copy()
reachable = annotated[annotated["reachable_from_seed"]].copy()
print(f"  Seeded: {len(seeded)} / {len(annotated)} flowlines")
print(f"  Reachable: {len(reachable)} / {len(annotated)} flowlines")

streams = compute.rasterize_confirmed_flowlines(reachable, dem_da)
streams.rio.to_raster(f"{aoi_dir}/streams_rf_propagated.tif")

config = HandilyConfig(
    out_dir=aoi_dir, flowlines_local_dir="", ndwi_dir="",
    stac_dir="", fields_path="",
    rem_method="anisotropic_frame",
    rem_zero_mode="thalweg",
    rem_frame_station_spacing_m=7.5,
    rem_snap_max_offset_m=20.0,
    rem_snap_search_spacing_m=5.0,
    rem_frame_smoothing_m=200.0,
    rem_cross_max_dist_m=1200.0,
    rem_cross_step_m=5.0,
    rem_cross_ridge_prominence_m=5.0,
    rem_cross_descend_stop_m=3.0,
    rem_min_support_width_m=20.0,
    rem_debug_write_intermediates=True,
    # New explicit weights — water-dominant
    rem_snap_w_elev=0.3,
    rem_snap_w_water=0.4,
    rem_snap_w_prior=0.3,
    rem_snap_w_transition=1.0,
    # Corridor support
    rem_support_corridor_half_width_m=15.0,
    rem_support_corridor_half_length_m=10.0,
    rem_water_support_mode="binary_mask",
    # Reach acceptance thresholds
    rem_min_station_water_hit_fraction=0.10,
    rem_max_consecutive_no_water_m=2000.0,
    rem_max_mean_snap_offset_m=100.0,
    rem_min_seeded_fraction=0.0,
)

print("Running anisotropic frame REM...")
result = rem_frame.compute_rem_anisotropic_frame(dem_da, rf_da, reachable, config)
print(f"  Metrics: {result.metrics}")

print("Writing artifacts...")
result.confirmed_flowlines.to_file(
    f"{aoi_dir}/flowlines_confirmed.fgb", driver="FlatGeobuf"
)
if not result.snapped_flowlines.empty:
    result.snapped_flowlines.to_file(
        f"{aoi_dir}/flowlines_snapped.fgb", driver="FlatGeobuf"
    )
if not result.frame_flowlines.empty:
    result.frame_flowlines.to_file(
        f"{aoi_dir}/flowlines_frame.fgb", driver="FlatGeobuf"
    )
if not result.cross_sections.empty:
    result.cross_sections.to_file(
        f"{aoi_dir}/cross_sections.fgb", driver="FlatGeobuf"
    )
result.water_surface_da.rio.to_raster(f"{aoi_dir}/water_surface_aniso.tif")
result.rem_da.rio.to_raster(f"{aoi_dir}/rem_anisotropic.tif")

print(f"Done: {aoi_dir}/rem_anisotropic.tif")
