#!/usr/bin/env bash
# Train Diffusion Policy on ALOHA Transfer Cube (config/experiment/aloha_transfer_cube.yaml).
# The overrides below restate the most common settings; edit them here or override them on the command
# line, which takes precedence. ENV_NAME works as in train.sh. NUM_GPUS defaults to 2 here, and
# GLOBAL_BATCH_SIZE (default 64) is split evenly across them (train_dataloader.batch_size is per GPU).
# Usage:
#   scripts/train_aloha.sh
#   scripts/train_aloha.sh trainer.iterations=200000 trainer.rollout.n_episodes=50
#   scripts/train_aloha.sh resume_from=outputs/diffusion_aloha_transfer_cube/<date>/<time>/checkpoints/last
#   NUM_GPUS=1 GLOBAL_BATCH_SIZE=64 scripts/train_aloha.sh
set -euo pipefail

export NUM_GPUS="${NUM_GPUS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
if (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE is not divisible by NUM_GPUS=$NUM_GPUS" >&2
    exit 1
fi

OVERRIDES=(
    # Run
    name=diffusion_aloha_transfer_cube
    seed=42
    wandb.mode=online

    # Training length and schedule
    trainer.iterations=100000
    lr_scheduler=cosine
    lr_scheduler.num_warmup_steps=500

    # Validation and checkpoints
    trainer.validate_every=5000
    trainer.checkpoint_every=5000
    trainer.patience=null

    # Rollouts at every validation (split across GPUs, EMA weights); best.pth maximizes their success rate.
    # Videos and metrics.json go to <run_dir>/rollouts/iter_<step>.
    trainer.rollout.n_episodes=20
    trainer.rollout.seed=1000
    trainer.rollout.save_videos=true
    trainer.best_metric=success_rate

    # EMA
    trainer.use_ema=true
    trainer.ema.decay=0.999

    # Batches and data loading (the val loader uses the same number of workers). The brightness, contrast and
    # saturation jitter of the training frames is configured in the experiment file.
    train_dataloader.batch_size=$(( GLOBAL_BATCH_SIZE / NUM_GPUS ))
    val_dataloader.batch_size=64
    train_dataloader.num_workers=2

    # Optimizer and precision
    optimizer.lr=1e-4
    optimizer.weight_decay=1e-2
    accelerator.mixed_precision=no

    # Model (horizon must be a multiple of 2^len(down_dims))
    model.n_obs_steps=2
    model.horizon=16
    model.n_action_steps=8
    model.down_dims=[128,256,512]
    model.num_inference_steps=20
    model.resize_shape=[192,256]
    # Vision backbone: torchvision weights (e.g. ResNet18_Weights.IMAGENET1K_V1) or null for random init.
    # use_group_norm=true swaps BatchNorm for GroupNorm and requires pretrained_backbone_weights=null.
    model.pretrained_backbone_weights=null
    model.use_group_norm=true
)

exec "$(dirname "${BASH_SOURCE[0]}")/train.sh" experiment=aloha_transfer_cube "${OVERRIDES[@]}" "$@"
