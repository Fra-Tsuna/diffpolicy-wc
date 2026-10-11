#!/usr/bin/env bash
# Train Diffusion Policy (PushT by default). Extra arguments are Hydra overrides.
# Usage:
#   scripts/train.sh
#   scripts/train.sh experiment=aloha_transfer_cube
#   scripts/train.sh trainer.iterations=50000 wandb.mode=offline
#   scripts/train.sh resume_from=outputs/diffusion_pusht/<date>/<time>/checkpoints/last
#   NUM_GPUS=2 scripts/train.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
NUM_GPUS="${NUM_GPUS:-1}"
# The timestamp is fixed here so all processes of a multi-GPU run share one output directory.
RUN_DIR='outputs/${name}/'"$(date +%Y-%m-%d/%H-%M-%S)"
if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch --multi_gpu --num_processes "$NUM_GPUS" main.py "hydra.run.dir=$RUN_DIR" "$@"
else
    python main.py "hydra.run.dir=$RUN_DIR" "$@"
fi
