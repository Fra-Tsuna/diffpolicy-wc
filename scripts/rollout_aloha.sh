#!/usr/bin/env bash
# Run closed-loop ALOHA Transfer Cube episodes; MP4 videos and metrics.json (success rate, rewards, lengths)
# go to outputs/diffusion_aloha_transfer_cube/rollout/<date>/<time>/videos.
# The overrides below restate the most common settings; edit them here or override them on the command
# line, which takes precedence. ENV_NAME works as in rollout.sh.
# Usage:
#   scripts/rollout_aloha.sh
#   scripts/rollout_aloha.sh checkpoint_path=outputs/diffusion_aloha_transfer_cube/<date>/<time>/checkpoints/best.pth
#   scripts/rollout_aloha.sh n_episodes=50 checkpoint_path=ckpt/diffusion_aloha_transfer_cube/iter_50000.pth apply_ema=false
set -euo pipefail

OVERRIDES=(
    # Policy: a best.pth trained here (copy a run's checkpoints/best.pth to this path, or override it)
    checkpoint_path=ckpt/diffusion_aloha_transfer_cube/best.pth
    # EMA weights, as used by training's rollouts to select best.pth; set false for a checkpoint trained without
    # EMA (trainer.use_ema=false), which has none
    apply_ema=true

    # Episodes: episode i uses seed + i for the initial cube pose and the diffusion noise
    n_episodes=10
    seed=42
    # At most the env's time limit (400 steps = 8 s at 50 Hz)
    max_steps=400

    # Camera of the videos (the policy always sees "top"): top, angle, front_close, left_pillar, right_pillar,
    # left_wrist, right_wrist
    video_camera=angle
    # Playback rate: the simulator runs at 50 Hz, so 10 fps plays the videos 5x slower than real time
    video_fps=10
)

exec "$(dirname "${BASH_SOURCE[0]}")/rollout.sh" experiment=aloha_transfer_cube "${OVERRIDES[@]}" "$@"
