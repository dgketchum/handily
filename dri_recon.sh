#!/usr/bin/env bash
# DRI login-node recon: read-only. Fetch and run with one typed line:
#   curl -sL raw.githubusercontent.com/dgketchum/handily/main/dri_recon.sh | bash
# Reports OS, scheduler, module system, available tools, egress, storage, GPUs.

echo "=== os ==="; uname -a; head -2 /etc/os-release 2>/dev/null
echo "=== scheduler ==="; for t in sinfo squeue sbatch qstat qsub; do command -v "$t"; done 2>/dev/null
sinfo -s 2>/dev/null | head -10
echo "=== modules ==="; command -v module >/dev/null 2>&1 && module avail 2>&1 | head -40 || echo "no module command"
echo "=== tools ==="; for t in git curl wget rsync tmux screen python3 gcc cmake; do printf '%-8s %s\n' "$t" "$(command -v "$t" 2>/dev/null || echo MISSING)"; done
echo "=== egress (needs outbound 443) ==="
curl -sI --max-time 10 https://storage.googleapis.com 2>/dev/null | head -1 || echo "GCS: BLOCKED or curl missing"
curl -sI --max-time 10 https://github.com 2>/dev/null | head -1 || echo "GitHub: BLOCKED or curl missing"
echo "=== storage ==="; df -h "$HOME" 2>/dev/null; quota -s 2>/dev/null | head -5
ls -d /scratch* /project* /projects* /work* /data* 2>/dev/null
echo "=== gpus ==="; sinfo -o '%P %G %D %m %c' 2>/dev/null | head -15
echo "=== recon done ==="
