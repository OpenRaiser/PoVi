from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_HTTP_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
DEFAULT_MODEL = "cosyvoice-v3.5-flash"
DEFAULT_SAMPLE_URL = (
    "https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/cosyvoice/cosyvoice-zeroshot-sample.wav"
)
DEFAULT_VOICE_CACHE = Path(
    os.environ.get("COSYVOICE_VOICE_REGISTRY", "")
)


def configure_dashscope() -> None:
    import dashscope

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set DASHSCOPE_API_KEY for official CosyVoice TTS.")
    dashscope.api_key = api_key
    dashscope.base_http_api_url = os.environ.get("DASHSCOPE_HTTP_URL", DEFAULT_HTTP_URL)
    dashscope.base_websocket_api_url = os.environ.get("DASHSCOPE_WS_URL", DEFAULT_WS_URL)


def _read_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_cache(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_enrolled_voice(
    *,
    model: str = DEFAULT_MODEL,
    prefix: str = "vgen",
    sample_url: str = DEFAULT_SAMPLE_URL,
    cache_path: Path | None = None,
) -> str:
    """Reuse COSYVOICE_VOICE / cached enrolled voice, otherwise clone from sample_url."""
    configure_dashscope()
    from dashscope.audio.tts_v2 import VoiceEnrollmentService

    explicit = os.environ.get("COSYVOICE_VOICE", "").strip()
    if explicit:
        return explicit

    cache = cache_path or Path(os.environ.get("COSYVOICE_VOICE_CACHE", str(DEFAULT_VOICE_CACHE)))
    cached = _read_cache(cache)
    if cached.get("voice_id") and cached.get("model") == model:
        return str(cached["voice_id"])

    service = VoiceEnrollmentService()
    existing = service.list_voices() or []
    for item in existing:
        if not isinstance(item, dict):
            continue
        voice_id = str(item.get("voice_id") or "")
        status = str(item.get("status") or "")
        if voice_id.startswith(f"{model}-{prefix}-") and status == "OK":
            _write_cache(cache, {"voice_id": voice_id, "model": model, "source_url": sample_url})
            return voice_id

    voice_id = service.create_voice(target_model=model, prefix=prefix, url=sample_url)
    for _ in range(30):
        info = service.query_voice(voice_id=voice_id)
        status = info.get("status") if isinstance(info, dict) else None
        if status == "OK":
            _write_cache(
                cache,
                {
                    "voice_id": voice_id,
                    "model": model,
                    "source_url": sample_url,
                    "request_id": service.get_last_request_id(),
                },
            )
            return voice_id
        if status == "UNDEPLOYED":
            raise RuntimeError(f"CosyVoice voice enrollment failed: {voice_id}")
        time.sleep(8)
    raise TimeoutError(f"CosyVoice voice enrollment timed out: {voice_id}")


def synthesize_mp3(text: str, *, voice_id: str, model: str = DEFAULT_MODEL) -> bytes:
    configure_dashscope()
    from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

    if not text.strip():
        raise ValueError("Empty TTS text")
    last_error: Exception | None = None
    retries = max(1, int(os.environ.get("COSYVOICE_RETRIES", "6")))
    for attempt in range(retries):
        synthesizer = SpeechSynthesizer(
            model=model,
            voice=voice_id,
            format=AudioFormat.MP3_22050HZ_MONO_256KBPS,
        )
        try:
            audio = synthesizer.call(text.strip())
        except Exception as exc:
            last_error = exc
            message = str(exc)
            throttle = any(token in message for token in ("Throttling", "RateQuota", "429", "rate limit"))
            if not throttle or attempt >= retries - 1:
                raise
            time.sleep(min(120.0, 10.0 * (2 ** attempt)))
            continue
        if not audio:
            raise RuntimeError(
                f"CosyVoice synthesis returned no audio (request_id={synthesizer.get_last_request_id()})"
            )
        return audio
    raise RuntimeError(f"CosyVoice synthesis failed: {last_error}")


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    from shutil import which

    path = which("ffmpeg")
    if not path:
        raise FileNotFoundError("ffmpeg is required to convert CosyVoice MP3 to WAV")
    return path


def write_wav(mp3_bytes: bytes, output_wav: Path) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    mp3_path = output_wav.with_suffix(".mp3")
    mp3_path.write_bytes(mp3_bytes)
    command = [
        _ffmpeg(),
        "-y",
        "-i",
        str(mp3_path),
        "-ac",
        "1",
        "-ar",
        "22050",
        str(output_wav),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not output_wav.exists():
        raise RuntimeError(f"ffmpeg failed to convert CosyVoice audio: {completed.stderr[-400:]}")
    return output_wav


def synthesize_to_wav(text: str, output_wav: Path) -> Path:
    model = os.environ.get("COSYVOICE_API_MODEL", DEFAULT_MODEL)
    voice_id = ensure_enrolled_voice(model=model)
    return write_wav(synthesize_mp3(text, voice_id=voice_id, model=model), output_wav)
