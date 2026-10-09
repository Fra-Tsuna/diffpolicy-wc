#!/usr/bin/env bash
# Run closed-loop PushT episodes; MP4 videos go to outputs/<name>/rollout/<date>/<time>/videos.
# Without a checkpoint, uses the pretrained policy in ckpt/diffusion_pusht (see download_pretrained.sh).
# Usage:
#   scripts/rollout.sh
#   scripts/rollout.sh pretrained_path=ckpt/<other_policy>
#   scripts/rollout.sh outputs/diffusion_pusht/<date>/<time>/checkpoints/best.pth n_episodes=10
set -euo pipefail
ARGS=()
if [ $# -ge 1 ] && [[ "$1" != *=* ]]; then
    ARGS+=("checkpoint_path=$(realpath "$1")")
    shift
fi
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
if [ ${#ARGS[@]} -eq 0 ] && [[ " $* " != *" pretrained_path="* ]] && [ ! -d ckpt/diffusion_pusht ]; then
    echo "ckpt/diffusion_pusht not found. Run scripts/download_pretrained.sh first." >&2
    exit 1
fi
python rollout.py "${ARGS[@]}" "$@"
