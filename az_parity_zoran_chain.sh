#!/usr/bin/env bash
# AZ parity (state-specific: base CONUS bundle + ADWR Wells 55 only), zoran side: wait for the AZ FAC-REM
# builds, write the point-path FAC manifest, phase-1 site panel, point-path payload (one GPU), finalize the
# admission split, append the admitted wells to the base CONUS bundle, source edges, parity bundle (dry run
# gated, then real), stage everything to GCS.
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
H=/home/dgketchum/code/handily; U=$H/utils
W=/data/ssd2/handily/conus/wte_gnn
HUC=/data/ssd2/handily/huc8
WL=/data/ssd2/handily/az/regional/wells
P=/data/ssd2/handily/az/regional/adwr_lbl/parity
AZ84=$WL/az_huc8_84.txt
BASE=$W/graph_conus_monitoring_water_v2
BUNDLE_APP=$W/graph_conus_monitoring_water_v2_adwr
BUNDLE=$W/graph_conus_monitoring_water_v2_adwr_parity
PTMODEL=$W/gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold
PTMAN=$P/fac_rem_manifest_pt_conus.json
PT=$WL/adwr_sites_ndwrlbl_ufold_nosrc_ramp_pt_preds.parquet
FACB=$P/nm_nv_mt_az_fac_basins.txt
echo "CHAIN START $(date)"
echo "== (0) wait for the FAC build shards"
while kill -0 3135943 2>/dev/null || kill -0 3135944 2>/dev/null; do sleep 60; done
echo "builds exited $(date)"
grep -l Traceback $P/build_fac_*.log $P/shard_*_driver.log 2>/dev/null && { echo "build traceback(s) above; stopping"; exit 1; }
miss=0; nv2=0
for b in $(cat $AZ84); do [ -d $HUC/$b ] || { echo "no dir: $b"; miss=$((miss+1)); }; [ -f $HUC/$b/rem/${b}_scalable_v2/fac_head_depth_rem_10m.tif ] && nv2=$((nv2+1)); done
echo "AZ-84: $nv2 scalable_v2 rasters, $miss missing dirs"
[ $miss -eq 0 ] || exit 1
echo "== (a) point-path FAC manifest (scalable_v2, fallback scalable, CONUS discovery) + fac-basins list"
uv run --directory $H python $U/write_fac_manifest.py --recipe scalable_v2 --fallback-recipe scalable --out $PTMAN --note "AZ parity point path, discovered $(date +%F)" > $P/write_pt_manifest.log 2>&1 || { echo "manifest FAILED"; tail -5 $P/write_pt_manifest.log; exit 1; }
python3 -c "import json;d=json.load(open('$PTMAN'));print('pt manifest',d['n'],d['n_by_recipe'])"
{ cat /data/ssd2/handily/mt/regional/gwic_lbl/parity/nm_nv_mt_fac_basins_284.txt; for b in $(cat $AZ84); do [ -f $HUC/$b/rem/${b}_scalable_v2/fac_head_depth_rem_10m.tif ] && echo $b; done; } | sort -u > $FACB
echo "fac-basins list: $(wc -l < $FACB) basins"
echo "== (b) phase-1 site panel"
uv run --directory $H python $U/build_admission_split.py --source az_adwr --source-kind adwr --bundle $BASE --use-codes domestic,monitoring,livestock --exclude-pod-basins "" --exclude-huc8 "" --huc8-list $AZ84 --out-dir $WL --prefix adwr > $P/phase1_split.log 2>&1 || { echo "phase 1 FAILED"; tail -5 $P/phase1_split.log; exit 1; }
grep -E "recovered|funnel|wrote" $P/phase1_split.log
echo "== (c) point-path payload (ndwrlbl_ufold, nv_ndwr excluded, ramp)"
until gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1) && [ ${gpu#*, } -lt 20000 ]; do echo "no GPU under 20 GB used ($(date)); waiting"; sleep 300; done
echo "occupancy before predict: load $(cut -d' ' -f1-3 /proc/loadavg); gpus: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';') -> using GPU ${gpu%%,*}"
CUDA_VISIBLE_DEVICES=${gpu%%,*} uv run --directory $H python $U/predict_gnn_at_points.py --model-dir $PTMODEL --points $WL/adwr_sites_input.parquet --out $PT --exclude-source-name nv_ndwr --water-flatten-inference --water-flatten-ramp --fac-manifest $PTMAN > $WL/adwr_sites_pt_preds.log 2>&1 || { echo "predict FAILED"; tail -5 $WL/adwr_sites_pt_preds.log; exit 1; }
[ -f $PT ] || { echo "point payload missing: $PT"; exit 1; }
grep -q Traceback $WL/adwr_sites_pt_preds.log && { echo "predict traceback"; exit 1; }
echo "payload ready $(date)"
echo "== (d) finalize split"
uv run --directory $H python $U/build_admission_split.py --source az_adwr --source-kind adwr --bundle $BASE --use-codes domestic,monitoring,livestock --exclude-pod-basins "" --exclude-huc8 "" --huc8-list $AZ84 --out-dir $WL --prefix adwr --pt-preds $PT --pred-name ndwrlbl_ufold > $P/finalize_split.log 2>&1 || { echo "finalize FAILED"; tail -5 $P/finalize_split.log; exit 1; }
grep "admitted" $P/finalize_split.log | tail -1
echo "== (e) frozen R table (AZ ADWR only)"
uv run --directory $H python - <<PY || exit 1
import pandas as pd
g=pd.read_parquet("$WL/adwr_admitted_sources_frozen.parquet",columns=["x5070","y5070","r_wte_m"]); g["src"]="az_adwr_adm"
g.to_parquet("$P/frozen_r_az.parquet",index=False)
print("frozen R rows",len(g),g.src.value_counts().to_dict())
PY
echo "== (f) append ADWR admitted wells -> $(basename $BUNDLE_APP)"
[ -d $BUNDLE_APP ] && { echo "exists: $BUNDLE_APP (not overwriting)"; exit 1; }
uv run --directory $H python $U/append_labelled_wells.py --bundle $BASE --wells $WL/adwr_admitted_sources_frozen.parquet --out-bundle $BUNDLE_APP --source-name az_adwr_adm --well-class adwr > $P/append_adwr.log 2>&1 || { echo "append FAILED"; tail -5 $P/append_adwr.log; exit 1; }
grep -i "appended\|funnel\|parity\|dropped" $P/append_adwr.log | tail -4
echo "== (g) source edges k16 on the appended bundle"
uv run --directory $H python $U/build_source_edges.py --bundle $BUNDLE_APP --k 16 > $P/build_source_edges_adwr.log 2>&1 || { echo "source edges FAILED"; tail -5 $P/build_source_edges_adwr.log; exit 1; }
python3 -c "import json;d=json.load(open('$BUNDLE_APP/source_edges.json'));print({k:d[k] for k in ['n_query_nodes','n_eligible_sources','n_edges','mean_degree','median_geo_dist_km_rank0','frac_same_basin']})"
PB="--base-bundle $BUNDLE_APP --out-bundle $BUNDLE --appended-source az_adwr_adm --frozen-r $P/frozen_r_az.parquet --r-ground-cap-m 2 --fac-recipe scalable_v2 --fac-fallback-recipe scalable --fac-basins $FACB --dd-dist-floor-m 1000 --drop-feature elev_above_coarse_m"
echo "== (h) parity bundle dry run"
uv run --directory $H python $U/build_parity_bundle.py $PB --dry-run > $P/build_parity_bundle_dryrun.log 2>&1 || { echo "dry run FAILED"; tail -5 $P/build_parity_bundle_dryrun.log; exit 1; }
grep -E "frozen|cap|FAC|rasters|drilled|features|manifest" $P/build_parity_bundle_dryrun.log | head -12
grep -q "frozen r_wte_m.*max|.|=0.000 m" $P/build_parity_bundle_dryrun.log || { echo "GATE FAIL: frozen-R parity not 0.000 m; stopping before the real build"; exit 1; }
grep -q Traceback $P/build_parity_bundle_dryrun.log && { echo "dry run traceback"; exit 1; }
echo "== (i) parity bundle build"
[ -d $BUNDLE ] && { echo "exists: $BUNDLE (not overwriting)"; exit 1; }
uv run --directory $H python $U/build_parity_bundle.py $PB > $P/build_parity_bundle.log 2>&1 || { echo "bundle build FAILED"; tail -5 $P/build_parity_bundle.log; exit 1; }
for f in source_edges.parquet source_edges.json; do [ -e $BUNDLE/$f ] || ln -s $BUNDLE_APP/$f $BUNDLE/$f; done
ls -la $BUNDLE/fac_rem_manifest.json $BUNDLE/source_edges.parquet $BUNDLE/query_nodes.parquet
python3 -c "import json;d=json.load(open('$BUNDLE/fac_rem_manifest.json'));print('manifest',d['n'],d['n_by_recipe'])"
echo "== (j) stage to GCS"
bash $H/stage_zephyr_az_parity.sh > $P/stage_zephyr_pass1.log 2>&1
tail -3 $P/stage_zephyr_pass1.log
echo "CHAIN DONE $(date)"
