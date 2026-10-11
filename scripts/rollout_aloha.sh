#!/usr/bin/env bash
# Run closed-loop ALOHA Transfer Cube episodes; MP4 videos and metrics.json (success rate, rewards, lengths)
# go to outputs/diffusion_aloha_transfer_cube/rollout/<date>/<time>/videos.
# The overrides below restate the most common settings; edit them here or override them on the command
# line, which takes precedence. ENV_NAME works as in rollout.sh.
# Usage:
#   scripts/rollout_aloha.sh
#   scripts/rollout_aloha.sh checkpoint_path=outputs/diffusion_aloha_transfer_cube/<date>/<time>/checkpoints/best.pth
#   scripts/rollout_aloha.sh n_episodes=50 apply_ema=true
set -euo pipefail

OVERRIDES=(
    # Policy: a best.pth trained here (copy a run's checkpoints/best.pth to this path, or override it)
    checkpoint_path=ckpt/diffusion_aloha_transfer_cube/best.pth
    # EMA weights; needs a checkpoint trained with trainer.use_ema=true
    apply_ema=false

    # Episodes: episode i uses seed + i for the initial cube pose and the diffusion noise
    n_episodes=10
    seed=42
    # At most the env's time limit (400 steps = 8 s at 50 Hz)
    max_steps=400
)

exec "$(dirname "${BASH_SOURCE[0]}")/rollout.sh" experiment=aloha_transfer_cube "${OVERRIDES[@]}" "$@"
