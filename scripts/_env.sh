# Sourced by the other scripts: move to the repo root and activate the conda env.
# Override the env with ENV_NAME=myenv scripts/train.sh
# Without conda (e.g. in the Docker image) the current Python environment is used as is.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if command -v conda >/dev/null 2>&1; then
    ENV_NAME="${ENV_NAME:-diffpolicy}"
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
fi
