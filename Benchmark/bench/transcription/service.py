from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from ..media.audio import extract_audio_to_wav
from ..providers import OpenAIProvider, provider_config
from ..progress import emit_progress


def update_submission(submission_path: Path, transcript_path: Path) -> None:
    payload = json.loads(submission_path.read_text(encoding="utf-8"))
    artifact_dir = submission_path.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / "asr_transcript.txt"
    shutil.copy2(transcript_path, target)
    relative = target.relative_to(submission_path.parent)
    payload.setdefault("artifacts", {})["asr_transcript"] = relative.as_posix()
    submission_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_speech_transcription(
    video_path: Path,
    output_dir: Path,
    *,
    provider: OpenAIProvider | None = None,
    submission_path: Path | None = None,
) -> dict[str, Any]:
    """Extract mono audio and send it to the configured speech-transcription API."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "final_audio_16k_mono.wav"
    emit_progress("asr", "progress", "正在提取 16kHz 单声道音频", video=video_path)
    extraction = extract_audio_to_wav(video_path.resolve(), wav_path)
    emit_progress(
        "asr",
        "progress",
        "音频提取完成",
        duration_seconds=extraction.get("duration_seconds"),
        sample_rate=extraction.get("sample_rate"),
        wav=wav_path,
    )
    if wav_path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError(
            "Extracted WAV exceeds the speech-transcription API 50 MB request limit; "
            "split the video first"
        )
    provider = provider or OpenAIProvider(provider_config("speech_transcription"))
    emit_progress(
        "asr",
        "progress",
        "正在请求语音转写服务",
        provider=provider.config.name,
        model=provider.config.model,
    )
    text, attempts = provider.transcribe(wav_path)
    emit_progress(
        "asr",
        "progress",
        "语音转写服务已返回",
        characters=len(text),
        retries=len(attempts),
    )
    transcript_path = output_dir / "asr_transcript.txt"
    transcript_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
    report = {
        "schema_version": "0.1",
        "provider": provider.config.name,
        "base_url": provider.config.base_url,
        "model": provider.config.model,
        "extraction": extraction,
        "result": {
            "text": text,
            "words": [],
            "timestamp_policy": "provider_returns_text_only; no timestamps fabricated",
        },
        "attempts": attempts,
    }
    result_path = output_dir / "asr_result.json"
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if submission_path is not None:
        update_submission(submission_path.resolve(), transcript_path)
    return {
        "ok": bool(text),
        "transcript_path": str(transcript_path),
        "result_path": str(result_path),
        "text_characters": len(text),
        "words_available": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark ASR on a video")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    report = run_speech_transcription(
        args.video,
        args.output_dir,
        submission_path=args.submission,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
