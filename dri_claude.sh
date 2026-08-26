#!/usr/bin/env bash
# Run ON ZEPHYR-LOGIN. Installs Claude Code userspace + tmux/settings config.
#   curl -sLO https://raw.githubusercontent.com/dgketchum/handily/main/dri_claude.sh
#   bash dri_claude.sh
# Then log in (interactive ONCE, same dance as gcloud auth):
#   exec bash            # pick up PATH
#   claude auth login    # open the printed URL in the remote desktop's browser,
#                        # paste the code back into the terminal
# Run claude itself inside tmux (tmux new -s claude) so sessions survive the
# laptop -> Citrix -> desktop -> ssh chain dropping.
set -euo pipefail

# ---- 1. native binary (userspace, ~/.local; needs only glibc >= 2.17) ------------
command -v claude >/dev/null 2>&1 || curl -fsSL https://claude.ai/install.sh | bash
grep -q '\.local/bin' "$HOME/.bashrc" 2>/dev/null || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
export PATH="$HOME/.local/bin:$PATH"

# ---- 2. tmux passthrough/extended keys (Shift+Enter, notifications) ---------------
touch "$HOME/.tmux.conf"
grep -q allow-passthrough "$HOME/.tmux.conf" || cat >> "$HOME/.tmux.conf" << 'TX'
set -g allow-passthrough on
set -s extended-keys on
set -as terminal-features 'xterm*:extkeys'
TX

# ---- 3. settings: no background auto-updates on the shared node -------------------
mkdir -p "$HOME/.claude"
[ -f "$HOME/.claude/settings.json" ] || cat > "$HOME/.claude/settings.json" << 'JS'
{
  "env": { "DISABLE_AUTOUPDATER": "1" }
}
JS

# ---- 4. credentials/transcripts live in $HOME on shared NFS: lock it down ---------
chmod 700 "$HOME" 2>/dev/null || true

# ---- 5. cluster CLAUDE.md for the repo clone (kept untracked via info/exclude) ----
REPO=/project/handily/code/handily
if [ -d "$REPO/.git" ] && [ ! -f "$REPO/CLAUDE.md" ]; then
  cat > "$REPO/CLAUDE.md" << 'MD'
# Handily on DRI zephyr (cluster clone)

- This is the DRI HPC (Rocky Linux 8, Slurm). You run on the login node
  `zephyr-login`: NEVER run heavy compute (patch builds, training, big
  raster work) here — submit it via sbatch.
- sbatch headers: `--account=dketchum --qos=qos-general`. Partitions:
  `general` (CPU, 14 nodes), `single-gpu`/`all-gpu` (gpu1: 4x NVIDIA L40S
  48 GB). GPU QOS: check `sacctmgr -nP show assoc where user=$USER format=qos`.
- Environment: uv exclusively (`uv sync --all-extras`, `uv run python ...`).
  No conda, no pip, no modules needed for this project.
- Paths: keepers in /project/handily (code, checkpoints, 1 TB quota);
  regenerable bulk in /scratch/dketchum (mae_v2_mirror raster mirror, patch
  sets); job logs under /scratch/dketchum/handily/logs.
- MAE manifest for this machine: configs/mae/manifest_v2_dri.json
  (written by dri_deploy.sh; cache_dir points at the scratch mirror).
- This clone is read-only https: commit locally if useful, but you cannot
  push. Relay files/results via `gcloud storage cp ... gs://wudr/zephyr/`.
- Long foreground work goes in tmux; batch work goes in sbatch.
MD
  grep -qx 'CLAUDE.md' "$REPO/.git/info/exclude" 2>/dev/null || \
    echo 'CLAUDE.md' >> "$REPO/.git/info/exclude"
  echo "wrote $REPO/CLAUDE.md"
fi

"$HOME/.local/bin/claude" --version || true
echo "done. next: exec bash, then: claude auth login"
