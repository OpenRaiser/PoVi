#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$CONDA_SH"
conda activate "$DIFF_ENV"

PYTHON_BIN="${PYTHON_BIN:-$CONDA_ROOT/envs/$DIFF_ENV/bin/python}"
"$PYTHON_BIN" "$SCRIPT_DIR/check_python_deps.py" --group qwen35_27b
