#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$CONDA_SH"
conda activate "$DIFF_ENV"

PYTHON_BIN="${PYTHON_BIN:-$CONDA_ROOT/envs/$DIFF_ENV/bin/python}"
export PYTHONPATH="$DIFFSYNTH_ROOT:$VGEN_ROOT:${PYTHONPATH:-}"

if [[ ! -f "$DIFFSYNTH_ROOT/pyproject.toml" ]]; then
  echo "[FAIL] Embedded DiffSynth-Studio not found: $DIFFSYNTH_ROOT" >&2
  exit 1
fi

"$PYTHON_BIN" -m pip install -e "$DIFFSYNTH_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/check_python_deps.py" --group diffsynth
