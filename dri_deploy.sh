#!/usr/bin/env bash
# Run ON ZEPHYR-LOGIN (DRI HPC), inside tmux. Fetch then run (two steps -- step 4
# needs an interactive terminal, so do NOT pipe this script into bash):
#   curl -sLO https://raw.githubusercontent.com/dgketchum/handily/main/dri_deploy.sh
#   tmux new -s deploy
#   bash dri_deploy.sh
# One-time deploy: probe storage/QOS, install uv + gcloud userspace, clone handily,
# build env, pull the v2 MAE raster mirror from GCS (~147 GB, resumable), write the
# DRI manifest, submit a patch-build + 2-epoch-train smoke chain (general -> L40S).
# Idempotent: safe to re-run; each step skips what already exists.
# gcloud auth (step 4) is interactive ONCE: copy the URL from the terminal into the
# remote desktop's browser (desktop-internal clipboard works), paste the code back.
set -euo pipefail

P=/project/handily                 # keepers: code, checkpoints (1 TB quota)
S=/scratch/dketchum                # bulk regenerable: mirror, patches
ACCT=dketchum
REPO=$P/code/handily

# ---- 0. probe: storage + slurm associations ------------------------------------
df -h /project /scratch /data 2>/dev/null || true
if ! mkdir -p "$S" 2>/dev/null; then
  echo "WARN: $S not writable; falling back to $P/scratch"
  S=$P/scratch; mkdir -p "$S"
fi
MIRROR=$S/handily/mae_v2_mirror
sacctmgr -nP show assoc where user="$USER" format=account,partition,qos 2>/dev/null || true
QOS_GPU=$(sacctmgr -nP show assoc where user="$USER" format=qos 2>/dev/null | tr ',' '\n' | grep -i gpu | head -1 || true)
[ -n "$QOS_GPU" ] || QOS_GPU=qos-general
echo "using: scratch=$S gpu_qos=$QOS_GPU"

# ---- 1. uv (userspace) ----------------------------------------------------------
UV=$HOME/.local/bin/uv
command -v uv >/dev/null 2>&1 && UV=$(command -v uv)
[ -x "$UV" ] || curl -LsSf https://astral.sh/uv/install.sh | sh

# ---- 2. code (public repo, https read-only) --------------------------------------
mkdir -p "$P/code"
if [ -d "$REPO/.git" ]; then git -C "$REPO" pull --ff-only; else
  git clone https://github.com/dgketchum/handily.git "$REPO"; fi

# ---- 3. env ----------------------------------------------------------------------
(cd "$REPO" && "$UV" sync --all-extras)

# ---- 4. gcloud CLI userspace (interactive auth ONCE; see header) -----------------
if ! command -v gcloud >/dev/null && [ ! -x "$HOME/google-cloud-sdk/bin/gcloud" ]; then
  curl -sSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz | tar -xz -C "$HOME"
fi
export PATH=$HOME/google-cloud-sdk/bin:$PATH
gcloud auth print-access-token >/dev/null 2>&1 || gcloud auth login --no-launch-browser

# ---- 5. mirror pull (flat; ~147 GB; just re-run on interruption) -----------------
mkdir -p "$MIRROR"
gcloud storage rsync gs://wudr/handily/mae_v2_mirror/ "$MIRROR/"

# ---- 6. verify by count + apparent-size sum --------------------------------------
EXPECTED_COUNT=54
EXPECTED_BYTES=147355846493
count=$(ls -1 "$MIRROR" | wc -l)
bytes=$(ls -l "$MIRROR" | awk 'NR>1{s+=$5} END{print s}')
echo "mirror: $count files, $bytes bytes (expect $EXPECTED_COUNT / $EXPECTED_BYTES)"
[ "$count" = "$EXPECTED_COUNT" ] && [ "$bytes" = "$EXPECTED_BYTES" ] || { echo "MIRROR VERIFY FAILED"; exit 1; }

# ---- 7. DRI manifest: zoran manifest + cache_dir -> flat mirror -------------------
"$UV" run --project "$REPO" python - "$REPO" "$MIRROR" << 'PY'
import json, sys
repo, mirror = sys.argv[1], sys.argv[2]
m = json.load(open(f"{repo}/configs/mae/manifest_v2.json"))
m["cache_dir"] = mirror
out = f"{repo}/configs/mae/manifest_v2_dri.json"
json.dump(m, open(out, "w"), indent=2)
print("wrote", out)
PY

# ---- 8. smoke: 2k-patch build (general) -> 2-epoch pyr_wide train (L40S) ----------
mkdir -p "$S/handily/mae/patches/smoke" "$P/handily/checkpoints" "$S/handily/logs"

cat > "$S/handily/smoke_build.sbatch" << SB
#!/bin/bash
#SBATCH --job-name=mae-smoke-build
#SBATCH --partition=general
#SBATCH --account=$ACCT
#SBATCH --qos=qos-general
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=$S/handily/logs/smoke_build_%j.log
$UV run --project $REPO python $REPO/utils/build_mae_patches.py \
  --manifest $REPO/configs/mae/manifest_v2_dri.json \
  --n-patches 2000 --out-dir $S/handily/mae/patches/smoke
SB

cat > "$S/handily/smoke_train.sbatch" << SB
#!/bin/bash
#SBATCH --job-name=mae-smoke-train
#SBATCH --partition=single-gpu
#SBATCH --account=$ACCT
#SBATCH --qos=$QOS_GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=$S/handily/logs/smoke_train_%j.log
$UV run --project $REPO python $REPO/utils/train_neighborhood_mae.py \
  --patch-dir $S/handily/mae/patches/smoke --arm pyr_wide \
  --out $P/handily/checkpoints/smoke.pt --epochs 2 --batch 64 --dim 64 --resume
SB

b=$(sbatch --parsable "$S/handily/smoke_build.sbatch")
t=$(sbatch --parsable --dependency=afterok:"$b" "$S/handily/smoke_train.sbatch")
echo "submitted: build=$b train=$t (train runs after build succeeds)"
echo "watch: squeue -u $USER ; logs in $S/handily/logs/"
