#!/usr/bin/env bash
# Run on zephyr-login: pull the build-input set, recreate bundle symlinks,
# prime WhiteboxTools. After this zephyr builds FAC-REM from public sources.
set -euo pipefail
SRC="gs://wudr/zephyr/build_inputs"
H=/project/handily/code/handily-nv
gsutil -m rsync -r "${SRC}/data/" /data/
gsutil -m rsync -r "${SRC}/nas/" /nas/
mkdir -p /data/ssd2/handily/pilot_ref && gsutil -m rsync -r "${SRC}/pilot/" /data/ssd2/handily/pilot_ref/
P=/data/ssd2/handily/az/regional/adwr_lbl/parity
for s in $P/symlinks_*.txt; do
  while read -r link arrow target; do [ -e "$link" ] || ln -s "$target" "$link"; [ -e "$link" ] || echo "BROKEN SYMLINK TARGET MISSING: $link -> $target"; done < "$s"
done
uv run --directory $H python -c "import whitebox; w=whitebox.WhiteboxTools(); print('whitebox', w.version().splitlines()[0])"
ls /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2 | wc -l
ls /nas/hydrography/gsw/occurrence | wc -l
ls /data/ssd2/handily/pilot_ref/15060102
ls /nas/gwx/wtd_states/
ls /data/ssd2/handily/huc8/*/streams_regional_scalable_v2.fgb | wc -l
ls -d /data/ssd2/handily/huc8/*/gnn/gnn_conus_monitoring_water_gate_mirror_sigma_w25_prod | wc -l
echo "pull complete; next: compute-node egress check, then the 15060102 pilot build (ZEPHYR_BUILD_HANDOFF.md §2-3)"
