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
export LLM_MODEL="${LLM_MODEL:-doubao/doubao-seed-2-0-lite}"
export LLM_API_KEY="${LLM_API_KEY:-${OPENAI_API_KEY:-}}"
export VEO_API_KEY="${VEO_API_KEY:-${OPENAI_API_KEY:-}}"
export VEO_HOST="${VEO_HOST:?Set VEO_HOST}"
export VEO_API_VERSION="${VEO_API_VERSION:-v2.03}"
export VEO_MODEL_MARKER="${VEO_MODEL_MARKER:-api_google_veo-3.1-generate-001}"
export VEO_DURATION_SECONDS="${VEO_DURATION_SECONDS:-8}"
export VEO_GENERATE_AUDIO="${VEO_GENERATE_AUDIO:-false}"
export VIDEO_WORKERS="${VIDEO_WORKERS:-1}"
export QWEN_IMAGE_MODEL="${QWEN_IMAGE_MODEL:?Set QWEN_IMAGE_MODEL}"
export IMAGE_BACKEND="${IMAGE_BACKEND:-gemini_image}"
export IMAGE_MODEL_MARKER="${IMAGE_MODEL_MARKER:-api_google_gemini-3-pro-image-preview}"
export IMAGE_DEVICE="${IMAGE_DEVICE:-cuda:0}"
export IMAGE_STEPS="${IMAGE_STEPS:-30}"
export COSYVOICE_MODEL="${COSYVOICE_MODEL:?Set COSYVOICE_MODEL}"
export COSYVOICE_PYTHON="${COSYVOICE_PYTHON:?Set COSYVOICE_PYTHON}"
export REFERENCE_IMAGES="${REFERENCE_IMAGES:-1}"

REFERENCE_ARGS=()
if [[ "$REFERENCE_IMAGES" == "0" ]]; then
  REFERENCE_ARGS+=(--no-reference-images)
  REFERENCE_DESCRIPTION="disabled_text_to_video"
else
  REFERENCE_DESCRIPTION="generated_character_assets"
fi

RUN_NAME="${RUN_NAME:-qingyuan_causal_veo31_$(date +%Y%m%d_%H%M%S)}"
DIRECTOR_GRAPH="${DIRECTOR_GRAPH:-$VGEN_ROOT/schema/examples/qingyu_an_yuanxi.director_graph.json}"

RESUME_MEDIA=0
for arg in "$@"; do
  if [[ "$arg" == "--resume-media" ]]; then
    RESUME_MEDIA=1
    break
  fi
done
if [[ "$RESUME_MEDIA" == "0" ]]; then
  : "${LLM_API_KEY:?Set LLM_API_KEY or OPENAI_API_KEY for text planning}"
fi

INPUT_DESCRIPTION="director_graph:$DIRECTOR_GRAPH"

echo "=== director graph -> remote Veo 3.1: $RUN_NAME ==="
echo "input=$INPUT_DESCRIPTION"
echo "llm=$LLM_MODEL"
echo "planning=context_story_narration_close_reading_card_visual_agents_without_critic"
echo "timeline=edit_groups_one_to_many_many_to_one_visual_only"
echo "veo_duration=dynamic_4_6_8_after_tts,max_${VEO_DURATION_SECONDS}s"
echo "reference_images=$REFERENCE_DESCRIPTION"
echo "poem_card=close_reading_agent_generated"
echo "subtitles=enabled"

python -m vgen.director_graph_pipeline \
  --director-graph "$DIRECTOR_GRAPH" \
  --run-name "$RUN_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-api-key "$LLM_API_KEY" \
  --llm-model "$LLM_MODEL" \
  --video-workers "$VIDEO_WORKERS" \
  --cosyvoice-python "$COSYVOICE_PYTHON" \
  "${REFERENCE_ARGS[@]}" \
  "$@"

echo "=== done: $VGEN_ROOT/outputs/director_graph/$RUN_NAME ==="
