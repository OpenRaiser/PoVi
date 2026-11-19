#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
exec "$VGEN_ROOT/scripts/services/serve_qwen35_27b_vllm.sh"
