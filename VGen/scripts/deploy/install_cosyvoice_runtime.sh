#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
source "$CONDA_SH"
conda activate "$DIFF_ENV"

PYTHON_BIN="${PYTHON_BIN:-$CONDA_ROOT/envs/$DIFF_ENV/bin/python}"

if ! "$PYTHON_BIN" "$SCRIPT_DIR/check_python_deps.py" --group cosyvoice; then
  echo "[WARN] CosyVoice runtime is incomplete in env '$DIFF_ENV'. Install CosyVoice runtime before enabling TTS inference."
  exit 0
fi
