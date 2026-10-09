#!/usr/bin/env bash
# Run the test suite. Extra arguments go to pytest.
# Usage: scripts/test.sh [-k pattern]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python -m pytest -q "$@"
