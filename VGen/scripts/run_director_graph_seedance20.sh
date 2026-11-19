#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VGEN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_SH="${CONDA_SH:?Set CONDA_SH to your conda.sh}"
DIFF_ENV="${DIFF_ENV:-diff}"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda activation script not found: $CONDA_SH" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$DIFF_ENV"

export PYTHONPATH="$VGEN_ROOT:${PYTHONPATH:-}"
export LLM_BASE_URL="${LLM_BASE_URL:?Set LLM_BASE_URL}"
export LLM_MODEL="${LLM_MODEL:-doubao-seed-2.0-pro}"
export LLM_API_KEY="${LLM_API_KEY:-${OPENAI_API_KEY:-}}"
export VIDEO_BACKEND="seedance20"
export SEEDANCE_HOST="${SEEDANCE_HOST:?Set SEEDANCE_HOST}"
export SEEDANCE_MODEL_MARKER="${SEEDANCE_MODEL_MARKER:-api_doubao_doubao-seedance-2-0-260128}"
export SEEDANCE_DURATION_SECONDS="${SEEDANCE_DURATION_SECONDS:-8}"
export SEEDANCE_RESOLUTION="${SEEDANCE_RESOLUTION:-720p}"
export SEEDANCE_GENERATE_AUDIO="${SEEDANCE_GENERATE_AUDIO:-false}"
export SEEDANCE_REFERENCE_MODE="${SEEDANCE_REFERENCE_MODE:-data_uri}"
export VIDEO_WORKERS="${VIDEO_WORKERS:-1}"
export QWEN_IMAGE_MODEL="${QWEN_IMAGE_MODEL:?Set QWEN_IMAGE_MODEL}"
export IMAGE_BACKEND="${IMAGE_BACKEND:-gemini_image}"
export IMAGE_MODEL_MARKER="${IMAGE_MODEL_MARKER:-api_google_gemini-3-pro-image-preview}"
export IMAGE_API_KEY="${IMAGE_API_KEY:-${SEEDANCE_API_KEY:-}}"
export GEMINI_IMAGE_PROMPT_PREFIX="${GEMINI_IMAGE_PROMPT_PREFIX:-中国传统工笔重彩与现代二维动画融合的角色设定插画，明确的虚构动画角色，清晰手绘轮廓线，平涂与轻微纸本纹理，非照片，非真人肖像，非写实皮肤，非三维渲染。}"
export IMAGE_DEVICE="${IMAGE_DEVICE:-cuda:0}"
export IMAGE_STEPS="${IMAGE_STEPS:-30}"
export COSYVOICE_MODEL="${COSYVOICE_MODEL:?Set COSYVOICE_MODEL}"
export COSYVOICE_PYTHON="${COSYVOICE_PYTHON:?Set COSYVOICE_PYTHON}"
export REFERENCE_IMAGES="${REFERENCE_IMAGES:-1}"

if [[ -z "${SEEDANCE_API_KEY:-}" && -n "${DOUBAO_TTS_USER:-}" && -n "${DOUBAO_TTS_KEY:-}" ]]; then
  export SEEDANCE_API_KEY="${DOUBAO_TTS_USER}:${DOUBAO_TTS_KEY}"
fi
if [[ -z "${IMAGE_API_KEY:-}" && -n "${SEEDANCE_API_KEY:-}" ]]; then
  export IMAGE_API_KEY="$SEEDANCE_API_KEY"
fi

REFERENCE_ARGS=()
if [[ "$REFERENCE_IMAGES" == "0" ]]; then
  REFERENCE_ARGS+=(--no-reference-images)
  REFERENCE_DESCRIPTION="disabled_text_to_video"
else
  REFERENCE_DESCRIPTION="generated_character_assets"
fi

RUN_NAME="${RUN_NAME:-qingyuan_causal_seedance20_$(date +%Y%m%d_%H%M%S)}"
DIRECTOR_GRAPH="${DIRECTOR_GRAPH:-$VGEN_ROOT/schema/examples/qingyu_an_yuanxi.director_graph.json}"

RESUME_MEDIA=0
DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--resume-media" ]] && RESUME_MEDIA=1
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done
if [[ "$RESUME_MEDIA" == "0" ]]; then
  : "${LLM_API_KEY:?Set LLM_API_KEY or OPENAI_API_KEY for text planning}"
fi
if [[ "$DRY_RUN" == "0" ]]; then
  : "${SEEDANCE_API_KEY:?Set SEEDANCE_API_KEY or DOUBAO_TTS_USER/DOUBAO_TTS_KEY}"
  if [[ "$REFERENCE_IMAGES" != "0" ]]; then
    : "${IMAGE_API_KEY:?Set IMAGE_API_KEY for Gemini character assets}"
  fi
fi

echo "=== director graph -> Seedance 2.0: $RUN_NAME ==="
echo "input=director_graph:$DIRECTOR_GRAPH"
echo "llm=$LLM_MODEL"
echo "planning=context_story_narration_close_reading_card_visual_agents_without_critic"
echo "timeline=edit_groups_one_to_many_many_to_one_visual_only"
echo "seedance_duration=dynamic_4_6_8,max_${SEEDANCE_DURATION_SECONDS}s"
echo "reference_images=$REFERENCE_DESCRIPTION"
echo "poem_card=close_reading_agent_generated"
echo "subtitles=enabled"

python -m vgen.director_graph_pipeline \
  --director-graph "$DIRECTOR_GRAPH" \
  --run-name "$RUN_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-api-key "$LLM_API_KEY" \
  --llm-model "$LLM_MODEL" \
  --video-backend seedance20 \
  --video-workers "$VIDEO_WORKERS" \
  --cosyvoice-python "$COSYVOICE_PYTHON" \
  "${REFERENCE_ARGS[@]}" \
  "$@"

echo "=== done: $VGEN_ROOT/outputs/director_graph/$RUN_NAME ==="
