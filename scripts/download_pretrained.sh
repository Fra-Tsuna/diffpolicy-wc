#!/usr/bin/env bash
# Download a pretrained LeRobot policy from the Hub and convert it for lerobot 0.6.
# Usage:
#   scripts/download_pretrained.sh                      # lerobot/diffusion_pusht -> ckpt/diffusion_pusht
#   scripts/download_pretrained.sh --repo-id <user/repo> --output-dir ckpt/<name>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python scripts/download_pretrained.py "$@"
