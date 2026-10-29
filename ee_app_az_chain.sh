#!/usr/bin/env bash
# EE app AZ unit: add the AZ ADWR parity arm (84 HUC8) beside the shipped NV + NM + MT units. Runs on zephyr
# (10 m renders live there): 10 m DTW/WTE tiles -> 30 m COGs -> gs://ucrc-wtd/handily/az/ -> EE ingest conus/{dtw,wte}/az.
# Wells: all AZ water-table wells (current GWX build) + wells_train from the parity bundle (monitoring + ADWR admitted half).
# Same steps as ee_app_mt_chain.sh (zoran, 2026-10-27); only paths and the unit differ. Runbook: notes/DATA_SHARE_EE_APP.md
# Prerequisites on the node: gdalbuildvrt/gdalwarp on PATH (module load gdal, or export the user-space GDAL env before
# calling), `earthengine authenticate` done once for this account, gsutil write access to gs://ucrc-wtd.
# EE picks the earthengine CLI: default `earthengine` on PATH; on zephyr run as
#   EE="uv run --directory /project/handily/code/handily-nv earthengine" bash ee_app_az_chain.sh
# Runs only after the straight-run smoothing (84 straight_smooth_run.json) so the ingested mosaic is the smoothed product.
set -euo pipefail
H=/data/ssd2/handily; HC=/project/handily/code/handily-nv; S=$H/share/ee_app/az; B=gs://ucrc-wtd/handily/az
ROOT=projects/ee-dgketchum/assets/handily; TAG=parity; st=az
ARM=gnn_conus_monitoring_water_src_r1e_m20_ord_adwrlbl_parity
BAS=$H/az/regional/wells/az_huc8_84.txt
BUN=$H/conus/wte_gnn/graph_conus_monitoring_water_v2_adwr_parity
GRP=gwx-users@googlegroups.com; USR=dgketchum@gmail.com
EE=${EE:-earthengine}
command -v gdalwarp > /dev/null || { echo "gdalwarp not on PATH (module load gdal?)"; exit 1; }
$EE ls $ROOT/conus > /dev/null 2>&1 || { echo "earthengine CLI ($EE) not authenticated or no access to $ROOT"; exit 1; }
echo "CHAIN START $(date) on $(hostname); $(gdalinfo --version); df: $(df -h $H | tail -1 | awk '{print $4" free"}')"
for c in dtw wte wells wells_train; do $EE asset info $ROOT/conus/$c/$st > /dev/null 2>&1 && { echo "conus/$c/$st already exists -- refusing to overwrite"; exit 1; }; done
n=$(ls $H/huc8/*/gnn/$ARM/straight_smooth_run.json 2>/dev/null | wc -l); [ "$n" = "$(wc -l < $BAS)" ] || { echo "only $n smoothing records for $(wc -l < $BAS) basins; run after the straight-run smoothing"; exit 1; }

# 1. 10 m VRTs + 30 m COGs (nodata-safe average resampling; see runbook)
mkdir -p $S
for v in dtw wte; do
  lst=$S/${v}_10m_${TAG}_list.txt; : > $lst
  for b in $(cat $BAS); do f=$H/huc8/$b/gnn/$ARM/gnn_${v}_10m.tif; [ -f $f ] || { echo "MISSING $f"; exit 1; }; echo $f >> $lst; done
  vrt=$S/handily_${v}_${st}_${TAG}_10m.vrt; cog=$S/handily_${v}_${st}_${TAG}_30m.tif
  gdalbuildvrt -q -overwrite -input_file_list $lst $vrt
  if [ ! -f $cog ]; then
    echo "== warp $st $v ($(wc -l < $lst) tiles) $(date)"
    gdalwarp -q -overwrite -tr 30 30 -r average -of COG -co COMPRESS=ZSTD -co PREDICTOR=3 -co BLOCKSIZE=512 -co OVERVIEW_RESAMPLING=AVERAGE -co OVERVIEWS=AUTO -co NUM_THREADS=ALL_CPUS -wo NUM_THREADS=ALL_CPUS $vrt $cog
  fi
  gdalinfo $cog | grep -E "Size is|NoData" | tr '\n' ' '; echo
done
echo "COGs done $(date); df: $(df -h $H | tail -1 | awk '{print $4" free"}')"

# 2. wells tables: all AZ water-table wells (GWX current build) + training labels of the parity bundle inside the 84 basins
uv run --directory $HC python utils/build_ee_wells.py --state AZ --out $S/handily_wells_${st}_${TAG}.csv
uv run --directory $HC python utils/build_ee_wells.py --bundle-dir $BUN --bundle-basins $BAS --label $st --out $S/handily_wells_train_${st}_${TAG}.csv

# 3. bucket
gsutil -m cp $S/handily_{dtw,wte}_${st}_${TAG}_30m.tif $S/handily_wells_${st}_${TAG}.csv $S/handily_wells_train_${st}_${TAG}.csv $B/
gsutil ls -l $B/

# 4. ingest
TASKS=""
for v in dtw wte; do
  t=$($EE upload image --asset_id=$ROOT/conus/$v/$st --nodata_value=-9999 --pyramiding_policy=mean $B/handily_${v}_${st}_${TAG}_30m.tif 2>&1 | grep -v Warn | tee /dev/stderr | grep -oE '[A-Z0-9]{20,}' | tail -1); TASKS="$TASKS $t"
done
t=$($EE upload table --asset_id=$ROOT/conus/wells/$st --x_column longitude --y_column latitude --crs EPSG:4326 $B/handily_wells_${st}_${TAG}.csv 2>&1 | grep -v Warn | tee /dev/stderr | grep -oE '[A-Z0-9]{20,}' | tail -1); TASKS="$TASKS $t"
t=$($EE upload table --asset_id=$ROOT/conus/wells_train/$st --x_column longitude --y_column latitude --crs EPSG:4326 $B/handily_wells_train_${st}_${TAG}.csv 2>&1 | grep -v Warn | tee /dev/stderr | grep -oE '[A-Z0-9]{20,}' | tail -1); TASKS="$TASKS $t"
echo "tasks:$TASKS"
$EE task wait --timeout 21600 $TASKS
for t in $TASKS; do $EE task info $t | grep -v Warn | grep -E "State|Description" | tr '\n' ' '; echo; done

# 5. table ACLs (tables under a folder do not inherit; app is PUBLIC); collections inherit their grants
for c in wells wells_train; do $EE acl ch -u $USR:R -g $GRP:R -u allUsers:R $ROOT/conus/$c/$st 2>&1 | grep -v -E "Warn|warnings.warn" || true; $EE acl get $ROOT/conus/$c/$st | grep -v Warn | tr '\n' ' '; echo; done

# 6. verify the collections now hold nv + nm + mt + az
for c in dtw wte wells wells_train; do echo "== $c"; $EE ls $ROOT/conus/$c | grep -v Warn; done
echo "CHAIN DONE $(date); df: $(df -h $H | tail -1 | awk '{print $4" free"}')"
