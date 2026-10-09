#!/usr/bin/env bash
# Create the conda env (if missing) and install all dependencies.
# Usage: scripts/setup.sh
set -euo pipefail
ENV_NAME="${ENV_NAME:-diffpolicy}"
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.12
fi
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
pip install -r requirements-lerobot.txt -r requirements-dev.txt
python -c "import torch, lerobot; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
