#!/usr/bin/env bash
# Run closed-loop episodes in the experiment's simulator; MP4 videos and metrics.json go to
# outputs/<name>/rollout/<date>/<time>/videos.
# PushT without a checkpoint uses the pretrained policy in ckpt/diffusion_pusht (see download_pretrained.sh).
# ALOHA has no pretrained policy and needs a checkpoint.
# Usage:
#   scripts/rollout.sh
#   scripts/rollout.sh pretrained_path=ckpt/<other_policy>
#   scripts/rollout.sh outputs/diffusion_pusht/<date>/<time>/checkpoints/best.pth n_episodes=10
#   scripts/rollout.sh outputs/diffusion_aloha_transfer_cube/<date>/<time>/checkpoints/best.pth experiment=aloha_transfer_cube
set -euo pipefail
ARGS=()
if [ $# -ge 1 ] && [[ "$1" != *=* ]]; then
    ARGS+=("checkpoint_path=$(realpath "$1")")
    shift
fi
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python rollout.py "${ARGS[@]}" "$@"
