#!/usr/bin/env bash
# Stage the AZ parity render inputs for zephyr. Idempotent: objects already at
# the destination are skipped, so re-run until it prints STAGE COMPLETE.
set -uo pipefail
DST="gs://wudr/zephyr/az_parity_stage"
H=/home/dgketchum/code/handily
P=/data/ssd2/handily/az/regional/adwr_lbl/parity
W=/data/ssd2/handily/conus/wte_gnn
HUC=/data/ssd2/handily/huc8
AZ84=/data/ssd2/handily/az/regional/wells/az_huc8_84.txt
BUNDLE=$W/graph_conus_monitoring_water_v2_adwr_parity
ARM=$W/gnn_conus_monitoring_water_src_r1e_m20_ord_adwrlbl_parity
SUB=$P/fac_rem_manifest_az84.json
RECIPE=scalable_v2
n_up=0; n_skip=0; n_miss=0; fail=0
put() {  # put <absolute local path>  -> ${DST}<same path>
  local src=$1 dst="${DST}${1}"
  if [ ! -e "$src" ]; then n_miss=$((n_miss+1)); return 0; fi
  if gsutil -q stat "$dst"; then n_skip=$((n_skip+1)); return 0; fi
  if gsutil -q cp "$src" "$dst"; then n_up=$((n_up+1)); else echo "UPLOAD FAILED: $src"; fail=1; fi
}
echo "=== A. AZ-84 ${RECIPE} rasters present locally ==="
n_v2=0
for b in $(cat $AZ84); do
  d=$HUC/$b/rem/${b}_${RECIPE}
  if [ -f $d/fac_head_depth_rem_10m.tif ]; then
    n_v2=$((n_v2+1)); put $d/fac_head_depth_rem_10m.tif; put $d/fac_rem_run.json; put $HUC/$b/basin_boundary.fgb
  fi
done
echo "A: ${n_v2}/84 ${RECIPE} rasters exist locally"
echo "=== B. render subset of the bundle's FAC-REM manifest ==="
SRCMAN=$ARM/fac_rem_manifest.json; [ -f $SRCMAN ] || SRCMAN=$BUNDLE/fac_rem_manifest.json
if [ -f $SRCMAN ]; then
  if [ ! -f $SUB ] || [ $SRCMAN -nt $SUB ]; then
    uv run --directory $H python utils/write_fac_manifest.py --from $SRCMAN --basins $AZ84 --out $SUB --note "AZ-84 render subset of $SRCMAN" || { echo "subset write FAILED"; fail=1; }
  fi
fi
if [ -f $SUB ]; then
  off=$(jq -r --arg r $RECIPE '.entries[] | select(.recipe != $r) | .basin' $SUB | grep -xf $AZ84 || true)
  if [ -n "$off" ]; then echo "TARGETS OFF-RECIPE in $SUB: $off"; fail=1
  else
    for path in $(jq -r '.entries[].path' $SUB); do
      put "$path"; put "$(dirname "$path")/fac_rem_run.json"; put "$(dirname "$(dirname "$(dirname "$path")")")/basin_boundary.fgb"
    done
    echo "B: subset staged ($(jq '.n' $SUB) rasters, $(jq -c '.n_by_recipe' $SUB))"
  fi
else
  echo "B: no manifest yet (bundle not built: $BUNDLE)"
fi
echo "=== C. small inputs ==="
for f in $SUB $P/lane_0.txt $P/lane_1.txt $P/lane_2.txt $P/lane_3.txt $P/nm_nv_mt_az_fac_basins.txt $P/frozen_r_az.parquet $AZ84 \
         /data/ssd2/handily/az/regional/wells/adwr_admitted_sources_frozen.parquet \
         /data/ssd2/handily/az/regional/wells/adwr_admission_targets.parquet \
         /data/ssd2/handily/az/regional/wells/adwr_admission_manifest.json \
         /nas/handily/covariates/water_mask_v4/water_mask_v4_100m_5070.tif \
         /data/ssd2/handily/conus/hydrography90m/coarse_surface_mw2km_100m.tif; do put $f; done
if [ -d $BUNDLE ]; then
  for f in $(find $BUNDLE -maxdepth 1 -type f); do put $f; done
  # symlinks: record the fully resolved target and stage that file too, so the
  # bundle resolves on a node that never held the intermediate bundles
  : > $P/symlinks_$(basename $BUNDLE).txt
  for l in $(find $BUNDLE -maxdepth 1 -type l); do t=$(readlink -f $l); echo "$l -> $t" >> $P/symlinks_$(basename $BUNDLE).txt; put "$t"; done
  gsutil -q cp $P/symlinks_$(basename $BUNDLE).txt "$DST/" && echo "bundle staged (+ resolved symlink list, $(wc -l < $P/symlinks_$(basename $BUNDLE).txt) links)"
else echo "bundle not built yet: $BUNDLE"; fi
if [ -d $ARM ]; then for f in $(find $ARM -maxdepth 1 -type f); do put $f; done; echo "arm staged: $(basename $ARM)"; else echo "arm not trained yet (expected: zephyr trains it)"; fi
gsutil -q cp $H/notes/ZEPHYR_AZ_PARITY_HANDOFF.md $H/zephyr_pull_az_parity.sh "$DST/" || { echo "handoff/pull copy FAILED"; fail=1; }
echo "uploaded $n_up, already staged $n_skip, missing locally $n_miss"
if [ $fail -eq 0 ] && [ $n_miss -eq 0 ] && [ -f $SUB ] && [ -d $BUNDLE ]; then echo "STAGE COMPLETE"; else echo "STAGE INCOMPLETE (fail=$fail missing=$n_miss)"; fi
