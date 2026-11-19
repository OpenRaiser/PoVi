#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

mkdir -p "$LOG_DIR" "$RUNTIME_DIR"

python3 "$SCRIPT_DIR/check_models.py" \
  --qwen-image "$QWEN_IMAGE_MODEL" \
  --cosyvoice "$COSYVOICE_MODEL" \
  --wan-i2v "$WAN_I2V_MODEL" \
  --qwen35-27b "$QWEN35_27B_MODEL"

"$SCRIPT_DIR/install_diff_runtime.sh"
"$SCRIPT_DIR/install_qwen35_27b_runtime.sh"
"$SCRIPT_DIR/install_cosyvoice_runtime.sh"

cat > "$RUNTIME_DIR/deployment.env" <<EOF
VGEN_ROOT=$VGEN_ROOT
DIFFSYNTH_ROOT=$DIFFSYNTH_ROOT
QWEN_IMAGE_MODEL=$QWEN_IMAGE_MODEL
COSYVOICE_MODEL=$COSYVOICE_MODEL
WAN_I2V_MODEL=$WAN_I2V_MODEL
QWEN35_27B_MODEL=$QWEN35_27B_MODEL
CONDA_SH=$CONDA_SH
DIFF_ENV=$DIFF_ENV
EOF

echo "[ OK ] VGen deployment finished. Runtime config: $RUNTIME_DIR/deployment.env"
