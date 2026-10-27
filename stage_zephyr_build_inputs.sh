#!/usr/bin/env bash
# Stage the small, non-public inputs that let zephyr build FAC-REM, train and
# render without any raster transfer from zoran. Idempotent: objects already
# at the destination are skipped, so re-run until it prints STAGE COMPLETE.
# Rasters derived from public data (3DEP, GSW via MPC, Hydrography90m) are NOT
# staged; zephyr fetches them. Plan: notes/ZEPHYR_BUILD_MIGRATION_PLAN.md.
set -uo pipefail
DST="gs://wudr/zephyr/build_inputs"
W=/data/ssd2/handily/conus/wte_gnn
BASE=$W/graph_conus_monitoring_water_v2
AZP=/data/ssd2/handily/az/regional/adwr_lbl/parity
AZW=/data/ssd2/handily/az/regional/wells
PILOT=15060102
n_up=0; n_skip=0; n_miss=0; fail=0
put() {  # put <absolute local path> [dest override] -> ${DST}<same path>
  local src=$1 dst="${DST}${2:-$1}"
  if [ ! -e "$src" ]; then echo "MISSING: $src"; n_miss=$((n_miss+1)); return 0; fi
  if gsutil -q stat "$dst"; then n_skip=$((n_skip+1)); return 0; fi
  if gsutil -q cp "$src" "$dst"; then n_up=$((n_up+1)); else echo "UPLOAD FAILED: $src"; fail=1; fi
}
putdir() {  # putdir <absolute local dir> -> rsync, no deletes
  [ -d "$1" ] || { echo "MISSING DIR: $1"; n_miss=$((n_miss+1)); return 0; }
  gsutil -m -q rsync -r "$1" "${DST}$1" || { echo "RSYNC FAILED: $1"; fail=1; }
}
echo "=== A. builder reference inputs ==="
for f in $W/huc8_polys.parquet \
         /nas/hydrography/HUC_Boundaries/wbd_national/wbdhu8_5070.parquet \
         /nas/hydrography/HUC_Boundaries/wbd_national/wbdhu12_5070.parquet \
         /data/ssd2/handily/conus/covariates/modis_ndvi_jja_mean.tif \
         /nas/hydrography/nhd/canals_mt_nv_nm_5070.fgb \
         /nas/handily/covariates/anthropogenic/irrigation_2017.tif; do put $f; done
putdir /nas/hydrography/gsw/occurrence
echo "=== B. admission / append inputs ==="
for f in /data/ssd2/gwx/products/current/wells.geoparquet \
         /data/ssd2/gwx/nm_ose/pod.parquet \
         /data/ssd2/gwx/mt_gwic/sites.parquet \
         /data/ssd2/gwx/az_adwr/registry.parquet \
         /data/ssd2/handily/conus/covariates/elev48i0100a.tif \
         $W/fac_flowline_geom_conus.parquet \
         $W/drilled_depth_points.parquet; do put $f; done
echo "=== C. train / infer inputs ==="
for f in $W/confidence_by_canonical.parquet $W/v02/wp2/assignment_priors.parquet \
         /data/ssd2/handily/conus/hydrography90m/coarse_surface_mw2km_100m.tif \
         /nas/handily/covariates/water_mask_v4/water_mask_v4_100m_5070.tif; do put $f; done
putdir $W/gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold
for f in $(find $BASE -maxdepth 1 -type f); do put $f; done
: > $AZP/symlinks_$(basename $BASE).txt
for l in $(find $BASE -maxdepth 1 -type l); do t=$(readlink -f $l); put $t; echo "$l -> $t" >> $AZP/symlinks_$(basename $BASE).txt; done
put $AZP/symlinks_$(basename $BASE).txt
echo "=== D. AZ parity tables (so the AZ arm can be trained from zephyr-built rasters) ==="
for f in $AZP/lane_0.txt $AZP/lane_1.txt $AZP/lane_2.txt $AZP/lane_3.txt $AZP/nm_nv_mt_az_fac_basins.txt $AZP/frozen_r_az.parquet \
         $AZW/az_huc8_84.txt $AZW/adwr_admitted_sources_frozen.parquet $AZW/adwr_admission_targets.parquet $AZW/adwr_admission_manifest.json; do put $f; done
AZB=$W/graph_conus_monitoring_water_v2_adwr_parity
for f in $(find $AZB -maxdepth 1 -type f); do put $f; done
: > $AZP/symlinks_$(basename $AZB).txt
for l in $(find $AZB -maxdepth 1 -type l); do t=$(readlink -f $l); put $t; echo "$l -> $t" >> $AZP/symlinks_$(basename $AZB).txt; done
put $AZP/symlinks_$(basename $AZB).txt
echo "=== E. pilot reference: zoran's scalable_v2 build of $PILOT (comparison only, outside /data) ==="
d=/data/ssd2/handily/huc8/$PILOT
for f in $d/rem/${PILOT}_scalable_v2/fac_head_depth_rem_10m.tif $d/rem/${PILOT}_scalable_v2/fac_rem_run.json $d/.fac_rem_done_scalable_v2 $d/basin_boundary.fgb $d/${PILOT}_scalable_v2.toml; do
  put $f "/pilot/$PILOT/$(basename $f)"
done
echo "=== F. AZ completion on zephyr: eval + 10 m post-processing inputs ==="
# Ma per-state tiles covering the 84 AZ basins only (sample_ma_tiles globs wtd_*.tif, first finite
# in sorted order; the five covering tiles keep zoran's order among themselves)
for st in arizona california nevada new_mexico utah; do put /nas/gwx/wtd_states/wtd_$st.tif; done
A0=gnn_conus_monitoring_water_gate_mirror_sigma_w25_prod
n0=0
for b in $(cat $AZW/az_huc8_84.txt); do
  put /data/ssd2/handily/huc8/$b/streams_regional_scalable_v2.fgb   # straight-run smoothing input
  if [ -d /data/ssd2/handily/huc8/$b/gnn/$A0 ]; then                 # reference arm, 100 m files only
    n0=$((n0+1)); for f in $(find /data/ssd2/handily/huc8/$b/gnn/$A0 -maxdepth 1 -type f ! -name "*10m*"); do put $f; done
  fi
done
echo "F: reference arm $A0 present in $n0 AZ basins (expect 19)"
for a in gnn_conus_monitoring_water_src_r1e_m20_ord_gwiclbl_parity gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold_oselbl_parity gnn_conus_monitoring_water_src_r1e_m20_ord; do putdir $W/$a; done
put $W/graph_conus_monitoring_water_v2_gwic_parity/query_nodes.parquet
put $W/graph_conus_monitoring_water_v2_ndwr_ufold_ose_parity/query_nodes.parquet
echo "uploaded=$n_up skipped=$n_skip missing=$n_miss"
if [ $fail -eq 0 ] && [ $n_miss -eq 0 ]; then echo "STAGE COMPLETE"; gsutil du -sh $DST; else echo "STAGE INCOMPLETE (re-run)"; exit 1; fi
