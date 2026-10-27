#!/usr/bin/env bash
# Run on zephyr-login: mirror the MT parity stage into place and verify it.
set -euo pipefail
SRC="gs://wudr/zephyr/mt_parity_stage"
H=/project/handily/code/handily-nv
gsutil -m rsync -r "${SRC}/data/" /data/
gsutil -m rsync -r "${SRC}/nas/" /nas/
P=/data/ssd2/handily/mt/regional/gwic_lbl/parity
B=/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2_gwic_parity
ARM=/data/ssd2/handily/conus/wte_gnn/gnn_conus_monitoring_water_src_r1e_m20_ord_gwiclbl_parity
gsutil cp "${SRC}/symlinks_$(basename $B).txt" $P/
while read -r link arrow target; do [ -e "$link" ] || ln -s "$target" "$link"; [ -e "$link" ] || echo "BROKEN SYMLINK TARGET MISSING: $link -> $target"; done < $P/symlinks_$(basename $B).txt
ls -la $B
ls -la $B/fac_rem_manifest.json
uv run --directory $H python utils/write_fac_manifest.py --from $P/fac_rem_manifest_mt115.json --verify
[ -d $ARM ] && echo "WARNING: $ARM already exists on this node; training must use a fresh --out-dir" || true
echo "pull complete; train_conus_gnn.py copies $B/fac_rem_manifest.json into the arm and infer_conus_gnn.py reads it there (no env vars, no FAC flags)"
