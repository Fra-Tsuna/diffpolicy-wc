#!/usr/bin/env bash
# Train Diffusion Policy on ALOHA Transfer Cube (config/experiment/aloha_transfer_cube.yaml).
# The overrides below restate the most common settings; edit them here or override them on the command
# line, which takes precedence. ENV_NAME works as in train.sh. NUM_GPUS defaults to 2 here
# (batch_size is per GPU, so the effective batch is NUM_GPUS x train_dataloader.batch_size).
# Usage:
#   scripts/train_aloha.sh
#   scripts/train_aloha.sh trainer.iterations=100000 trainer.use_ema=true
#   scripts/train_aloha.sh resume_from=outputs/diffusion_aloha_transfer_cube/<date>/<time>/checkpoints/last
#   NUM_GPUS=1 scripts/train_aloha.sh
set -euo pipefail

export NUM_GPUS="${NUM_GPUS:-2}"

OVERRIDES=(
    # Run
    name=diffusion_aloha_transfer_cube
    seed=42
    wandb.mode=online

    # Training length and schedule
    trainer.iterations=100000
    lr_scheduler=cosine
    lr_scheduler.num_warmup_steps=50

    # Validation and checkpoints
    trainer.validate_every=5000
    trainer.checkpoint_every=5000
    trainer.patience=null

    # EMA
    trainer.use_ema=false
    trainer.ema.decay=0.9999

    # Batches and data loading (the val loader uses the same number of workers)
    train_dataloader.batch_size=128
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
    model.resize_shape=[96,128]
    # Vision backbone: torchvision weights (e.g. ResNet18_Weights.IMAGENET1K_V1) or null for random init.
    # use_group_norm=true swaps BatchNorm for GroupNorm and requires pretrained_backbone_weights=null.
    model.pretrained_backbone_weights=null
    model.use_group_norm=false
)

exec "$(dirname "${BASH_SOURCE[0]}")/train.sh" experiment=aloha_transfer_cube "${OVERRIDES[@]}" "$@"
