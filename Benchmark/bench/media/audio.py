"""Low-level audio extraction and segmentation.

This module has no knowledge of Providers, Track routing, Submission updates,
or ASR model selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def extract_audio_to_wav(
    media_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    import av
    import numpy as np
    import soundfile as sf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    with av.open(str(media_path)) as container:
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            raise ValueError(f"No audio stream in {media_path}")
        stream = audio_streams[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            converted = resampler.resample(frame)
            frames = converted if isinstance(converted, list) else [converted] if converted is not None else []
            for converted_frame in frames:
                chunks.append(converted_frame.to_ndarray().reshape(-1))
        flushed = resampler.resample(None)
        frames = flushed if isinstance(flushed, list) else [flushed] if flushed is not None else []
        for converted_frame in frames:
            chunks.append(converted_frame.to_ndarray().reshape(-1))
    if not chunks:
        raise RuntimeError(f"Decoded no audio samples from {media_path}")
    samples = np.concatenate(chunks).astype(np.int16, copy=False)
    sf.write(str(output_path), samples, sample_rate, subtype="PCM_16")
    return {
        "source_media": str(media_path),
        "wav_path": str(output_path),
        "sample_rate": sample_rate,
        "sample_count": int(samples.size),
        "duration_seconds": float(samples.size) / sample_rate,
    }


def split_wav_by_timeline(
    wav_path: Path,
    timeline_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    import soundfile as sf

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    wave, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    for index, item in enumerate(timeline.get("segments", []), start=1):
        start = max(0.0, float(item.get("start_seconds") or 0.0))
        end = max(start, float(item.get("end_seconds") or start))
        start_frame = min(len(wave), round(start * sample_rate))
        end_frame = min(len(wave), round(end * sample_rate))
        if end_frame <= start_frame:
            continue
        output_path = output_dir / f"{index:02d}_{item.get('segment_id') or 'segment'}.wav"
        sf.write(str(output_path), wave[start_frame:end_frame], sample_rate, subtype="PCM_16")
        requests.append(
            {
                "segment_id": str(item.get("segment_id") or f"segment_{index:02d}"),
                "wav_path": str(output_path),
                "offset_seconds": start,
                "end_seconds": end,
                "reference_text": str(item.get("text") or ""),
            }
        )
    return requests
