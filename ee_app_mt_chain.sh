#!/usr/bin/env bash
# EE app MT unit (2026-10-27): add the MT GWIC parity arm (115 HUC8) beside the shipped NV + NM units.
# Rasters: 10 m DTW/WTE tiles -> 30 m COGs -> gs://ucrc-wtd/handily/mt/ -> EE ingest conus/{dtw,wte}/mt.
# Wells: all MT water-table wells (current GWX build) + wells_train from the parity bundle (monitoring + GWIC admitted half).
# July MT objects (handily_*_mt_30m.tif, no _parity tag) and the archived July assets are untouched. Runbook: notes/DATA_SHARE_EE_APP.md
set -euo pipefail
H=/data/ssd2/handily; HC=/home/dgketchum/code/handily; S=$H/share/ee_app/mt; B=gs://ucrc-wtd/handily/mt
ROOT=projects/ee-dgketchum/assets/handily; TAG=parity; st=mt
ARM=gnn_conus_monitoring_water_src_r1e_m20_ord_gwiclbl_parity
BAS=$H/mt/regional/wells/mt_huc8_115.txt
BUN=$H/conus/wte_gnn/graph_conus_monitoring_water_v2_gwic_parity
GRP=gwx-users@googlegroups.com; USR=dgketchum@gmail.com
echo "CHAIN START $(date); df: $(df -h $H | tail -1 | awk '{print $4" free"}')"
for c in dtw wte wells wells_train; do earthengine asset info $ROOT/conus/$c/$st > /dev/null 2>&1 && { echo "conus/$c/$st already exists -- refusing to overwrite"; exit 1; }; done

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

# 2. wells tables: all MT water-table wells (GWX current build) + training labels of the parity bundle inside the 115 basins
uv run --directory $HC python utils/build_ee_wells.py --state MT --out $S/handily_wells_${st}_${TAG}.csv
uv run --directory $HC python utils/build_ee_wells.py --bundle-dir $BUN --bundle-basins $BAS --label $st --out $S/handily_wells_train_${st}_${TAG}.csv

# 3. bucket (new _parity object names; July objects untouched)
gsutil -m cp $S/handily_{dtw,wte}_${st}_${TAG}_30m.tif $S/handily_wells_${st}_${TAG}.csv $S/handily_wells_train_${st}_${TAG}.csv $B/
gsutil ls -l $B/

# 4. ingest
TASKS=""
for v in dtw wte; do
  t=$(earthengine upload image --asset_id=$ROOT/conus/$v/$st --nodata_value=-9999 --pyramiding_policy=mean $B/handily_${v}_${st}_${TAG}_30m.tif 2>&1 | grep -v Warn | tee /dev/stderr | grep -oE '[A-Z0-9]{20,}' | tail -1); TASKS="$TASKS $t"
done
t=$(earthengine upload table --asset_id=$ROOT/conus/wells/$st --x_column longitude --y_column latitude --crs EPSG:4326 $B/handily_wells_${st}_${TAG}.csv 2>&1 | grep -v Warn | tee /dev/stderr | grep -oE '[A-Z0-9]{20,}' | tail -1); TASKS="$TASKS $t"
t=$(earthengine upload table --asset_id=$ROOT/conus/wells_train/$st --x_column longitude --y_column latitude --crs EPSG:4326 $B/handily_wells_train_${st}_${TAG}.csv 2>&1 | grep -v Warn | tee /dev/stderr | grep -oE '[A-Z0-9]{20,}' | tail -1); TASKS="$TASKS $t"
echo "tasks:$TASKS"
earthengine task wait --timeout 21600 $TASKS
for t in $TASKS; do earthengine task info $t | grep -v Warn | grep -E "State|Description" | tr '\n' ' '; echo; done

# 5. table ACLs (tables under a folder do not inherit; app is PUBLIC since 2026-10-21 so allUsers too); collections inherit their grants
for c in wells wells_train; do earthengine acl ch -u $USR:R -g $GRP:R -u allUsers:R $ROOT/conus/$c/$st 2>&1 | grep -v -E "Warn|warnings.warn" || true; earthengine acl get $ROOT/conus/$c/$st | grep -v Warn | tr '\n' ' '; echo; done

# 6. verify the collections now hold nv + nm + mt
for c in dtw wte wells wells_train; do echo "== $c"; earthengine ls $ROOT/conus/$c | grep -v Warn; done
echo "CHAIN DONE $(date); df: $(df -h $H | tail -1 | awk '{print $4" free"}')"
