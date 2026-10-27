#!/usr/bin/env bash
# EE app refresh (2026-10-21): replace the July MT/NM/UCRB w25_prod units with the shipped NV (72 HUC8) and
# NM (97 HUC8) parity arms. Rasters: 10 m DTW/WTE tiles -> 30 m COGs -> gs://ucrc-wtd -> EE ingest.
# Wells: all water-table wells per state (current GWX build) + wells_train from each arm's training bundle.
# July assets are MOVED to handily/archive_w25_prod (reversible), never deleted. Runbook: notes/DATA_SHARE_EE_APP.md
set -euo pipefail
H=/data/ssd2/handily; HC=/home/dgketchum/code/handily; S=$H/share/ee_app; B=gs://ucrc-wtd/handily
ROOT=projects/ssebop-montana/assets/handily; ARCH=$ROOT/archive_w25_prod; TAG=parity
ARM_nv=gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold_parity
ARM_nm=gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold_oselbl_parity
BAS_nv=$H/nv/regional/fix_round/nv_huc8_72.txt
BAS_nm=$H/nm/regional/ose_lbl/nm_huc8_97.txt
BUN_nv=$H/conus/wte_gnn/graph_conus_monitoring_water_v2_ndwr_ufold_parity
BUN_nm=$H/conus/wte_gnn/graph_conus_monitoring_water_v2_ndwr_ufold_ose_parity
GRP=gwx-users@googlegroups.com; USR=dgketchum@gmail.com
echo "CHAIN START $(date); df: $(df -h $H | tail -1 | awk '{print $4" free"}')"

# 1. 10 m VRTs + 30 m COGs (nodata-safe average resampling; see runbook)
for st in nv nm; do
  arm=ARM_$st; bas=BAS_$st; mkdir -p $S/$st
  for v in dtw wte; do
    lst=$S/$st/${v}_10m_${TAG}_list.txt; : > $lst
    for b in $(cat ${!bas}); do f=$H/huc8/$b/gnn/${!arm}/gnn_${v}_10m.tif; [ -f $f ] || { echo "MISSING $f"; exit 1; }; echo $f >> $lst; done
    vrt=$S/$st/handily_${v}_${st}_${TAG}_10m.vrt; cog=$S/$st/handily_${v}_${st}_${TAG}_30m.tif
    gdalbuildvrt -q -overwrite -input_file_list $lst $vrt
    if [ ! -f $cog ]; then
      echo "== warp $st $v ($(wc -l < $lst) tiles) $(date)"
      gdalwarp -q -overwrite -tr 30 30 -r average -of COG -co COMPRESS=ZSTD -co PREDICTOR=3 -co BLOCKSIZE=512 -co OVERVIEW_RESAMPLING=AVERAGE -co OVERVIEWS=AUTO -co NUM_THREADS=ALL_CPUS -wo NUM_THREADS=ALL_CPUS $vrt $cog
    fi
    gdalinfo $cog | grep -E "Size is|NoData" | tr '\n' ' '; echo
  done
done
echo "COGs done $(date); df: $(df -h $H | tail -1 | awk '{print $4" free"}')"

# 2. wells tables: all water-table wells (GWX current build) + training labels of each arm's bundle
for st in nv nm; do
  bas=BAS_$st; bun=BUN_$st; ST=${st^^}
  uv run --directory $HC python utils/build_ee_wells.py --state $ST --out $S/$st/handily_wells_${st}_${TAG}.csv
  uv run --directory $HC python utils/build_ee_wells.py --bundle-dir ${!bun} --bundle-basins ${!bas} --label $st --out $S/$st/handily_wells_train_${st}_${TAG}.csv
done

# 3. bucket (new object names; July objects untouched)
for st in nv nm; do gsutil -m cp $S/$st/handily_{dtw,wte}_${st}_${TAG}_30m.tif $S/$st/handily_wells_${st}_${TAG}.csv $S/$st/handily_wells_train_${st}_${TAG}.csv $B/$st/; done
gsutil ls -l $B/nv/ $B/nm/

# 4. archive the July units (mv, reversible) so the collections hold only NV + NM
earthengine asset info $ARCH > /dev/null 2>&1 || earthengine create folder $ARCH
for c in dtw wte; do earthengine asset info $ARCH/$c > /dev/null 2>&1 || earthengine create collection $ARCH/$c; done
for c in wells wells_train; do earthengine asset info $ARCH/$c > /dev/null 2>&1 || earthengine create folder $ARCH/$c; done
for c in dtw wte wells wells_train; do for u in mt nm ucrb; do
  if earthengine asset info $ROOT/conus/$c/$u > /dev/null 2>&1; then echo "mv $c/$u -> archive"; earthengine mv $ROOT/conus/$c/$u $ARCH/$c/$u; fi
done; done

# 5. ingest
TASKS=""
for st in nv nm; do
  for v in dtw wte; do
    t=$(earthengine upload image --asset_id=$ROOT/conus/$v/$st --nodata_value=-9999 --pyramiding_policy=mean $B/$st/handily_${v}_${st}_${TAG}_30m.tif 2>&1 | grep -v Warn | tee /dev/stderr | grep -oE '[A-Z0-9]{20,}' | tail -1); TASKS="$TASKS $t"
  done
  t=$(earthengine upload table --asset_id=$ROOT/conus/wells/$st --x_column longitude --y_column latitude --crs EPSG:4326 $B/$st/handily_wells_${st}_${TAG}.csv 2>&1 | grep -v Warn | tee /dev/stderr | grep -oE '[A-Z0-9]{20,}' | tail -1); TASKS="$TASKS $t"
  t=$(earthengine upload table --asset_id=$ROOT/conus/wells_train/$st --x_column longitude --y_column latitude --crs EPSG:4326 $B/$st/handily_wells_train_${st}_${TAG}.csv 2>&1 | grep -v Warn | tee /dev/stderr | grep -oE '[A-Z0-9]{20,}' | tail -1); TASKS="$TASKS $t"
done
echo "tasks:$TASKS"
earthengine task wait --timeout 21600 $TASKS
for t in $TASKS; do earthengine task info $t | grep -v Warn | grep -E "State|Description" | tr '\n' ' '; echo; done

# 6. table ACLs (tables under a folder do not inherit); collections already carry the group/user grants
for st in nv nm; do for c in wells wells_train; do earthengine acl ch -u $USR:R -g $GRP:R $ROOT/conus/$c/$st; earthengine acl get $ROOT/conus/$c/$st | grep -v Warn | tr '\n' ' '; echo; done; done

# 7. verify the collections hold exactly nv + nm
for c in dtw wte wells wells_train; do echo "== $c"; earthengine ls $ROOT/conus/$c | grep -v Warn; done
echo "CHAIN DONE $(date); df: $(df -h $H | tail -1 | awk '{print $4" free"}')"
