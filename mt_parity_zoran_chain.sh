#!/usr/bin/env bash
# zoran post-return chain for the MT parity arm (state-specific: base CONUS + GWIC):
#   pull-back from GCS -> OOF compare vs the base m20_ord arm and the NM parity arm (background, CPU)
#   || 10 m DEM restore (8 wide) -> 10 m render (6 wide) -> straight-run smoothing (16 wide, max-slope 1 %,
#   in place) -> statewide 100 m raster eval vs the shipped w25_prod MT render and Ma, HUC8 blocks.
# Every stage is gated on its predecessor's file count; re-runnable (each stage skips what exists).
set -uo pipefail
export PYTHONUNBUFFERED=1
H=/home/dgketchum/code/handily
U=$H/utils
W=/data/ssd2/handily/conus/wte_gnn
HUC=/data/ssd2/handily/huc8
O=/data/ssd2/handily/mt/regional/gwic_lbl
P=$O/parity
MT115=/data/ssd2/handily/mt/regional/wells/mt_huc8_115.txt
ARM0=gnn_conus_monitoring_water_gate_mirror_sigma_w25_prod
ARM=gnn_conus_monitoring_water_src_r1e_m20_ord_gwiclbl_parity
BASEARM=gnn_conus_monitoring_water_src_r1e_m20_ord
NMARM=gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold_oselbl_parity
MA=/nas/gwx/wtd_states
mkdir -p $P/chain_logs/dem $P/chain_logs/render $P/chain_logs/smooth
count() { for b in $(cat $MT115); do ls $HUC/$b/$1 2>/dev/null; done | wc -l; }
echo "CHAIN START $(date) arm $ARM"

# 0. pull-back (idempotent rsync), then guard: arm + 115 basin returns on disk
bash $P/pull_mt_parity_return.sh > $P/pull_mt_parity_return.log 2>&1; tail -2 $P/pull_mt_parity_return.log
n=$(count gnn/$ARM/infer_run.json)
[ -f $W/$ARM/models/inference_manifest.json ] || { echo "arm incomplete: $W/$ARM/models/inference_manifest.json missing"; exit 1; }
[ -f $W/$ARM/fac_rem_manifest.json ] || { echo "arm missing fac_rem_manifest.json"; exit 1; }
[ "$n" -eq 115 ] || { echo "basin returns incomplete: $n/115 infer_run.json"; exit 1; }
echo "guard ok: arm + 115/115 infer_run.json"

# 1. OOF compare (background, CPU): vs the base-bundle m20_ord arm (same base rows) and vs the NM parity arm
if [ ! -f $O/oof_parity_vs_m20ord/deltas.csv ]; then
  nohup uv run --directory $H python $U/compare_conus_oof_arms.py --ref m20_ord=$W/$BASEARM --arm mtparity=$W/$ARM --out-dir $O/oof_parity_vs_m20ord > $P/compare_oof_vs_m20ord.log 2>&1 &
  echo "OOF compare vs m20_ord launched (PID $!)"
fi
if [ ! -f $O/oof_parity_vs_nmparity/deltas.csv ]; then
  nohup uv run --directory $H python $U/compare_conus_oof_arms.py --ref nmparity=$W/$NMARM --arm mtparity=$W/$ARM --out-dir $O/oof_parity_vs_nmparity > $P/compare_oof_vs_nmparity.log 2>&1 &
  echo "OOF compare vs NM parity launched (PID $!)"
fi

# 2. restore 10 m DEMs, 8 wide (idempotent)
echo "DEM restore start $(date): $(count dem_10m.tif)/115 present"
cat $MT115 | xargs -P 8 -I{} sh -c "uv run --directory $H python $U/restore_huc8_dems.py --basins {} > $P/chain_logs/dem/{}.log 2>&1 || echo 'DEM FAILED {}'"
nd=$(count dem_10m.tif); echo "DEM restore done $(date): $nd/115; df: $(df -h /data/ssd2 | tail -1 | awk '{print $4" free"}')"
[ "$nd" -eq 115 ] || { echo "DEM restore incomplete ($nd/115); stopping before render"; exit 1; }

# 3. 10 m render, 6 wide (CPU-only exact recomposition; skips basins with existing 10 m)
echo "render start $(date): $(count gnn/$ARM/gnn_dtw_10m.tif)/115 present"
cat $MT115 | xargs -P 6 -I{} sh -c "uv run --directory $H python $U/render_gnn_10m.py --model-dir $W/$ARM --basins {} --out-name $ARM > $P/chain_logs/render/{}.log 2>&1 || echo 'RENDER FAILED {}'"
nr=$(count gnn/$ARM/gnn_dtw_10m.tif); echo "render done $(date): $nr/115 gnn_dtw_10m.tif, $(count gnn/$ARM/render_run.json)/115 render_run.json; tracebacks: $(grep -l Traceback $P/chain_logs/render/*.log 2>/dev/null | wc -l); df: $(df -h /data/ssd2 | tail -1 | awk '{print $4" free"}')"
[ "$nr" -eq 115 ] || { echo "render incomplete ($nr/115); stopping before smoothing"; exit 1; }

# 4. straight-run smoothing, 16 wide, max-slope 1 %, in place (originals kept as *_presmooth.tif)
echo "smoothing start $(date)"
cat $MT115 | xargs -P 16 -I{} sh -c "uv run --directory $H python $U/postprocess_straight_fac.py --model-name $ARM --basins {} --huc8-root $HUC --max-slope-pct 1.0 --inplace > $P/chain_logs/smooth/{}.log 2>&1 || echo 'SMOOTH FAILED {}'"
ns=$(count gnn/$ARM/straight_smooth_run.json); echo "smoothing done $(date): $ns/115 straight_smooth_run.json; tracebacks: $(grep -l Traceback $P/chain_logs/smooth/*.log 2>/dev/null | wc -l)"
[ "$ns" -eq 115 ] || { echo "smoothing incomplete ($ns/115); stopping before eval"; exit 1; }

# 5. statewide 100 m raster eval: parity (primary) vs shipped w25_prod (reference) and Ma; HUC8 blocks
echo "eval start $(date)"
uv run --directory $H python $U/eval_nv_statewide_raster.py --arms shipped=$ARM0 parity=$ARM --primary-arm parity --reference-arm shipped --targets /data/ssd2/handily/mt/regional/wells/gwic_admission_targets.parquet --sources /data/ssd2/handily/mt/regional/wells/gwic_admitted_sources_frozen.parquet --ma-dir $MA --block-col huc8 --label MT --out-dir $O/statewide_eval_parity_sm4 > $P/statewide_eval_parity_sm4.log 2>&1 || echo "EVAL FAILED"
echo "eval done $(date): $(ls $O/statewide_eval_parity_sm4/report.md 2>/dev/null || echo 'no report.md')"
echo "OOF compare: $(ls $O/oof_parity_vs_m20ord/ $O/oof_parity_vs_nmparity/ 2>/dev/null | tr '\n' ' ')"
echo "CHAIN DONE $(date)"
