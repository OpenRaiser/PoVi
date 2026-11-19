#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../deploy/env.sh"
source "$CONDA_SH"
conda activate "$DIFF_ENV"
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
hash -r

PYTHON_BIN="${PYTHON_BIN:-$CONDA_ROOT/envs/$DIFF_ENV/bin/python}"
MODEL_PATH="${MODEL_PATH:-$QWEN35_27B_MODEL}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-27b-local}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION_USER_SET="${GPU_MEMORY_UTILIZATION+x}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-4}"
API_KEY="${API_KEY:-qwen3.5-27b-local}"
SERVICE_LOG_DIR="${SERVICE_LOG_DIR:-$LOG_DIR/vllm}"

if [[ -z "${LIMIT_MM_PER_PROMPT:-}" ]]; then
  LIMIT_MM_PER_PROMPT=$(printf '{"image":%s}' "$MAX_IMAGES_PER_PROMPT")
fi

mkdir -p "$SERVICE_LOG_DIR"
export CUDA_VISIBLE_DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_USE_V1 VLLM_MAX_SIZE_MB

if [[ -z "$GPU_MEMORY_UTILIZATION" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_MEMORY_UTILIZATION=$(nvidia-smi \
      --query-gpu=memory.free,memory.total \
      --format=csv,noheader,nounits \
      -i "$CUDA_VISIBLE_DEVICES" | "$PYTHON_BIN" -c '
import sys
ratios = []
for line in sys.stdin:
    if not line.strip():
        continue
    free_mb, total_mb = [float(x.strip()) for x in line.split(",")[:2]]
    ratios.append(free_mb / total_mb)
if not ratios:
    raise SystemExit(1)
# Keep a small safety margin so vLLM startup memory check does not fail on busy GPUs.
value = max(0.10, min(0.85, min(ratios) - 0.03))
print(f"{value:.2f}")
')
  else
    GPU_MEMORY_UTILIZATION="0.40"
  fi
  echo "=== Auto GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION based on current free memory ==="
elif [[ -n "$GPU_MEMORY_UTILIZATION_USER_SET" ]]; then
  echo "=== Use user GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION ==="
fi

"$PYTHON_BIN" - <<'PY'
import sys
import transformers
import vllm
print("=== Python runtime ===", flush=True)
print("sys.executable=", sys.executable, flush=True)
print("vllm=", vllm.__version__, vllm.__file__, flush=True)
print("transformers=", transformers.__version__, transformers.__file__, flush=True)
PY

echo "=== Start qwen3.5-27b model: GPU $CUDA_VISIBLE_DEVICES, port $PORT ==="
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
  --allowed-local-media-path / \
  --api-key "$API_KEY" \
  2>&1 | tee "$SERVICE_LOG_DIR/qwen35_27b_vllm_${PORT}.log"
