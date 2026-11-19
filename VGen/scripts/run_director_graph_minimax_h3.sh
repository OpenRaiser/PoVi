#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VGEN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_PY="${ENV_PY:?Set ENV_PY to your python}"

if [[ ! -x "$ENV_PY" ]]; then
  echo "Python runtime not found or not executable: $ENV_PY" >&2
  exit 1
fi

cd "$VGEN_ROOT"
export PYTHONPATH="$VGEN_ROOT:$VGEN_ROOT/third_party/CosyVoice:$VGEN_ROOT/third_party/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH:-}"

export LLM_BASE_URL="${LLM_BASE_URL:?Set LLM_BASE_URL}"
export LLM_MODEL="${LLM_MODEL:-doubao/doubao-seed-2-0-lite}"
export LLM_API_KEY="${LLM_API_KEY:-${OPENAI_API_KEY:-}}"

export MINIMAX_HOST="${MINIMAX_HOST:?Set MINIMAX_HOST}"
export MINIMAX_MODEL="${MINIMAX_MODEL:-MiniMax-H3}"
export MINIMAX_DURATION_SECONDS="${MINIMAX_DURATION_SECONDS:-4}"
export MINIMAX_RESOLUTION="${MINIMAX_RESOLUTION:-768P}"
export MINIMAX_RATIO="${MINIMAX_RATIO:-adaptive}"
export MINIMAX_POLL_SECONDS="${MINIMAX_POLL_SECONDS:-10}"
export MINIMAX_TIMEOUT_SECONDS="${MINIMAX_TIMEOUT_SECONDS:-1800}"
export MINIMAX_REFERENCE_MODE="${MINIMAX_REFERENCE_MODE:-data_uri}"
export MINIMAX_API_KEY="${MINIMAX_API_KEY:-${MINIMAX_APP_ID:-}:${MINIMAX_APP_KEY:-}}"

export IMAGE_BACKEND="${IMAGE_BACKEND:-openai_image}"
export IMAGE_MODEL_MARKER="${IMAGE_MODEL_MARKER:-azure_openai/gpt-image-2}"
export OPENAI_IMAGE_MODEL="${OPENAI_IMAGE_MODEL:-azure_openai/gpt-image-2}"
export OPENAI_IMAGE_BASE_URL="${OPENAI_IMAGE_BASE_URL:-${LLM_BASE_URL:?Set LLM_BASE_URL}}"
export IMAGE_API_KEY="${IMAGE_API_KEY:-${MINIMAX_API_KEY:-}}"
export IMAGE_DEVICE="${IMAGE_DEVICE:-cuda:0}"
export IMAGE_STEPS="${IMAGE_STEPS:-30}"
export IMAGE_WORKERS="${IMAGE_WORKERS:-1}"
export VIDEO_WORKERS="${VIDEO_WORKERS:-1}"

export COSYVOICE_MODEL="${COSYVOICE_MODEL:-$VGEN_ROOT/../../Models/CosyVoice3-0.5B-2512}"
export COSYVOICE_REPO="${COSYVOICE_REPO:-$VGEN_ROOT/third_party/CosyVoice}"
export COSYVOICE_PYTHON="${COSYVOICE_PYTHON:-$ENV_PY}"
export AUDIO_BACKEND="${AUDIO_BACKEND:-dashscope_cosyvoice}"
export COSYVOICE_API_MODEL="${COSYVOICE_API_MODEL:-cosyvoice-v3.5-flash}"
export DASHSCOPE_HTTP_URL="${DASHSCOPE_HTTP_URL:-https://dashscope.aliyuncs.com/api/v1}"
export DASHSCOPE_WS_URL="${DASHSCOPE_WS_URL:-wss://dashscope.aliyuncs.com/api-ws/v1/inference}"

RUN_NAME="${RUN_NAME:-changxiangsi_q1_minimax_h3_$(date +%Y%m%d_%H%M%S)}"
DIRECTOR_GRAPH="${DIRECTOR_GRAPH:-$VGEN_ROOT/schema/examples/qingyu_an_yuanxi.director_graph.json}"
REFERENCE_IMAGES="${REFERENCE_IMAGES:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$VGEN_ROOT/outputs/director_graph}"

if [[ ! -f "$DIRECTOR_GRAPH" ]]; then
  echo "Director graph not found: $DIRECTOR_GRAPH" >&2
  exit 1
fi
if [[ "$REFERENCE_IMAGES" != "0" && -z "$IMAGE_API_KEY" ]]; then
  echo "Set IMAGE_API_KEY (or MINIMAX_API_KEY) for character reference-image generation." >&2
  exit 1
fi

REFERENCE_ARGS=()
if [[ "$REFERENCE_IMAGES" == "0" ]]; then
  REFERENCE_ARGS+=(--no-reference-images)
fi

if [[ "${1:-}" != "--dry-run" && "${MINIMAX_API_KEY}" == ":" ]]; then
  echo "Set MINIMAX_API_KEY=app_id:app_key or MINIMAX_APP_ID/MINIMAX_APP_KEY." >&2
  exit 1
fi
if [[ "${AUDIO_BACKEND}" == "dashscope_cosyvoice" && -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "Set DASHSCOPE_API_KEY for official CosyVoice enrollment TTS." >&2
  exit 1
fi

echo "=== director graph -> MiniMax-H3: $RUN_NAME ==="
echo "input=$DIRECTOR_GRAPH"
echo "llm=$LLM_MODEL"
echo "video_backend=minimax_h3"
echo "duration=max_${MINIMAX_DURATION_SECONDS}s"
echo "reference_images=$([[ "$REFERENCE_IMAGES" == "0" ]] && echo disabled || echo generated_character_assets)"
echo "python=$ENV_PY"

"$ENV_PY" -m vgen.director_graph_pipeline \
  --director-graph "$DIRECTOR_GRAPH" \
  --run-name "$RUN_NAME" \
  --output-root "$OUTPUT_ROOT" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-api-key "$LLM_API_KEY" \
  --llm-model "$LLM_MODEL" \
  --video-backend minimax_h3 \
  --minimax-api-key "$MINIMAX_API_KEY" \
  --minimax-host "$MINIMAX_HOST" \
  --minimax-model "$MINIMAX_MODEL" \
  --minimax-duration-seconds "$MINIMAX_DURATION_SECONDS" \
  --minimax-resolution "$MINIMAX_RESOLUTION" \
  --minimax-ratio "$MINIMAX_RATIO" \
  --minimax-poll-seconds "$MINIMAX_POLL_SECONDS" \
  --minimax-timeout-seconds "$MINIMAX_TIMEOUT_SECONDS" \
  --image-backend "$IMAGE_BACKEND" \
  --image-device "$IMAGE_DEVICE" \
  --image-steps "$IMAGE_STEPS" \
  --cosyvoice-model "$COSYVOICE_MODEL" \
  --cosyvoice-repo "$COSYVOICE_REPO" \
  --cosyvoice-python "$COSYVOICE_PYTHON" \
  --audio-backend "$AUDIO_BACKEND" \
  --video-workers "$VIDEO_WORKERS" \
  "${REFERENCE_ARGS[@]}" \
  "$@"

echo "=== done: $OUTPUT_ROOT/$RUN_NAME ==="
