#!/usr/bin/env bash
# Evaluate a checkpoint on the held-out test split (denoising loss).
# Usage:
#   scripts/eval.sh outputs/diffusion_pusht/<date>/<time>/checkpoints/best.pth
#   scripts/eval.sh <checkpoint> split=val apply_ema=true
#   scripts/eval.sh <aloha checkpoint> experiment=aloha_transfer_cube
set -euo pipefail
if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint.pth> [hydra overrides...]" >&2
    exit 1
fi
CHECKPOINT="$(realpath "$1")"
shift
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python eval.py "checkpoint_path=$CHECKPOINT" "$@"
