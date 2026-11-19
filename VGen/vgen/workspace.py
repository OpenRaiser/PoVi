from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = DEFAULT_ROOT.parent
DEV_ROOT = WORKSPACE_ROOT.parent
SHARED_DEV_ROOT = Path(os.environ.get("SHARED_DEV_ROOT", str(DEV_ROOT))).expanduser()
LOCAL_MODELS_ROOT = DEV_ROOT / "Models"
WORKSPACE_MODELS_ROOT = WORKSPACE_ROOT / "models"
SHARED_MODELS_ROOT = SHARED_DEV_ROOT / "Models"

DEFAULT_QWEN35_API_BASE = "http://127.0.0.1:8002/v1"
DEFAULT_QWEN35_MODEL = "qwen3.5-27b-local"
DEFAULT_TARGET_TOTAL_SECONDS = 120
DEFAULT_SEGMENTS = 8
DEFAULT_CLIP_SECONDS = 14
DEFAULT_MIN_SEGMENT_SECONDS = 12
DEFAULT_MAX_SEGMENT_SECONDS = 16


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def first_existing_path(*candidates: Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def resolve_workspace_path(env_name: str, default_path: Path, *fallbacks: Path) -> Path:
    return first_existing_path(_path_from_env(env_name), default_path, *fallbacks) or default_path


def resolve_model_dir(default_name: str, env_name: str | None = None) -> Path:
    explicit = _path_from_env(env_name) if env_name else None
    local_path = LOCAL_MODELS_ROOT / default_name
    workspace_path = WORKSPACE_MODELS_ROOT / default_name
    shared_path = SHARED_MODELS_ROOT / default_name
    return first_existing_path(explicit, local_path, workspace_path, shared_path) or local_path


def load_speaker_reference_text(path: Path, speaker: str) -> str:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        for separator in ("：", ":"):
            prefix = f"{speaker}{separator}"
            if line.startswith(prefix):
                text = line[len(prefix) :].strip()
                if text:
                    return text
    raise ValueError(f"Speaker {speaker!r} not found in reference transcript: {path}")


DEFAULT_MODELS_ROOT = first_existing_path(
    _path_from_env("MODELS_ROOT"),
    LOCAL_MODELS_ROOT,
    WORKSPACE_MODELS_ROOT,
    SHARED_MODELS_ROOT,
) or LOCAL_MODELS_ROOT
DEFAULT_QWEN_IMAGE_MODEL = resolve_model_dir("Qwen-Image", "QWEN_IMAGE_MODEL")
DEFAULT_WAN_VACE_MODEL = resolve_model_dir("Wan2.2-VACE-Fun-A14B", "WAN_VACE_MODEL")
DEFAULT_WAN_I2V_MODEL = resolve_model_dir("Wan2.2-I2V-A14B", "WAN_I2V_MODEL")
DEFAULT_COSYVOICE_MODEL = resolve_model_dir("CosyVoice3-0.5B-2512", "COSYVOICE_MODEL")
DEFAULT_QWEN35_27B_MODEL_DIR = resolve_model_dir("qwen3.5-27b", "QWEN35_27B_MODEL")
DEFAULT_BERNINI_CONFIG = resolve_model_dir("Bernini-Diffusers", "BERNINI_CONFIG")
DEFAULT_BERNINI_R_CONFIG = resolve_model_dir("Bernini-R-Diffusers", "BERNINI_R_CONFIG")
DEFAULT_HUNYUANVIDEO_MODEL = resolve_model_dir("HunyuanVideo-1.5", "HUNYUANVIDEO_MODEL")

DEFAULT_COSYVOICE_REPO = resolve_workspace_path("COSYVOICE_REPO", DEFAULT_ROOT / "third_party" / "CosyVoice")
DEFAULT_COSYVOICE_ASSET_DIR = DEFAULT_ROOT / "third_party" / "CosyVoice" / "asset"
DEFAULT_COSYVOICE_PROMPT_WAV = resolve_workspace_path(
    "COSYVOICE_PROMPT_WAV",
    DEFAULT_COSYVOICE_ASSET_DIR / "龙老姨.mp3",
)
DEFAULT_COSYVOICE_PROMPT_TEXT_PATH = resolve_workspace_path(
    "COSYVOICE_PROMPT_TEXT_PATH",
    DEFAULT_COSYVOICE_ASSET_DIR / "text.txt",
)
DEFAULT_COSYVOICE_PROMPT_SPEAKER = "龙老姨"
DEFAULT_COSYVOICE_PROMPT_TEXT = load_speaker_reference_text(
    DEFAULT_COSYVOICE_PROMPT_TEXT_PATH,
    DEFAULT_COSYVOICE_PROMPT_SPEAKER,
)
DEFAULT_TANGSHI300_PATH = resolve_workspace_path(
    "TANGSHI300_PATH",
    DEFAULT_ROOT / "data" / "tangshi300" / "tangshi300.jsonl",
)
DEFAULT_COMFYUI_ROOT = first_existing_path(_path_from_env("COMFYUI_ROOT"), WORKSPACE_ROOT / "ComfyUI", WORKSPACE_ROOT / "ComfyUI-master") or (WORKSPACE_ROOT / "ComfyUI")
DEFAULT_BERNINI_ROOT = first_existing_path(_path_from_env("BERNINI_ROOT"), WORKSPACE_ROOT / "Bernini", WORKSPACE_ROOT / "Bernini-main") or (WORKSPACE_ROOT / "Bernini")
DEFAULT_BERNINI_PYTHON = first_existing_path(
    _path_from_env("BERNINI_PYTHON"),
    SHARED_DEV_ROOT / "miniconda3" / "envs" / "bernini" / "bin" / "python",
    WORKSPACE_ROOT / "venv" / "bernini" / "bin" / "python",
) or (SHARED_DEV_ROOT / "miniconda3" / "envs" / "bernini" / "bin" / "python")
DEFAULT_HUNYUANVIDEO_ROOT = first_existing_path(
    _path_from_env("HUNYUANVIDEO_ROOT"),
    WORKSPACE_ROOT / "HunyuanVideo-1.5-local",
    SHARED_DEV_ROOT / "HunyuanVideo-1.5",
) or (WORKSPACE_ROOT / "HunyuanVideo-1.5-local")
DEFAULT_HUNYUANVIDEO_PYTHON = first_existing_path(
    _path_from_env("HUNYUANVIDEO_PYTHON"),
    SHARED_DEV_ROOT / "miniconda3" / "envs" / "diff" / "bin" / "python",
) or (SHARED_DEV_ROOT / "miniconda3" / "envs" / "diff" / "bin" / "python")
