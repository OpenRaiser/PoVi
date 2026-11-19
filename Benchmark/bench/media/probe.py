"""Media readability probes shared by Gate and reports."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any


def probe_video(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
    }
    if not path.exists():
        return result
    try:
        import av

        with av.open(str(path)) as container:
            result["container_duration_seconds"] = (
                float(container.duration) / 1_000_000
                if container.duration is not None
                else 0.0
            )
            result["streams"] = [
                {
                    "index": stream.index,
                    "type": stream.type,
                    "codec": stream.codec_context.name,
                    "duration_seconds": (
                        float(stream.duration * stream.time_base)
                        if stream.duration is not None and stream.time_base is not None
                        else None
                    ),
                    "sample_rate": getattr(stream, "sample_rate", None),
                    "average_rate": (
                        float(stream.average_rate)
                        if getattr(stream, "average_rate", None) is not None
                        else None
                    ),
                }
                for stream in container.streams
            ]
            result["has_video_stream"] = any(
                stream["type"] == "video" for stream in result["streams"]
            )
            result["has_audio_stream"] = any(
                stream["type"] == "audio" for stream in result["streams"]
            )
    except Exception as exc:
        result["stream_probe_error"] = str(exc)

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        result["error"] = f"OpenCV unavailable: {exc}"
        return result

    capture = cv2.VideoCapture(str(path))
    try:
        opened = bool(capture.isOpened())
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
        positions = [0]
        if frame_count > 2:
            positions.extend([frame_count // 2, max(0, frame_count - 1)])
        decoded = 0
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                decoded += 1
        result.update(
            {
                "opened": opened,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_seconds": duration,
                "probe_frames_requested": len(positions),
                "probe_frames_decoded": decoded,
                "readable": (
                    opened
                    and fps > 0
                    and frame_count > 0
                    and width > 0
                    and height > 0
                    and decoded == len(positions)
                ),
            }
        )
    finally:
        capture.release()
    return result


def probe_wav(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
    }
    if not path.exists():
        return result
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            result.update(
                {
                    "channels": audio.getnchannels(),
                    "sample_width": audio.getsampwidth(),
                    "sample_rate": rate,
                    "frame_count": frames,
                    "duration_seconds": frames / rate if rate else 0.0,
                    "readable": frames > 0 and rate > 0,
                }
            )
    except Exception as exc:
        result["error"] = str(exc)
    return result


def probe_audio(path: Path) -> dict[str, Any]:
    """Probe a standalone audio file or the audio track of a media container."""
    if path.suffix.lower() == ".wav":
        return probe_wav(path)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
        "has_audio_stream": False,
    }
    if not path.exists():
        return result
    try:
        import av

        with av.open(str(path)) as container:
            streams = list(container.streams.audio)
            result["has_audio_stream"] = bool(streams)
            if not streams:
                return result
            stream = streams[0]
            decoded_frames = 0
            decoded_samples = 0
            for frame in container.decode(stream):
                decoded_frames += 1
                decoded_samples += int(getattr(frame, "samples", 0) or 0)
                if decoded_frames >= 3:
                    break
            result.update(
                {
                    "codec": stream.codec_context.name,
                    "sample_rate": getattr(stream, "sample_rate", None),
                    "decoded_frames": decoded_frames,
                    "decoded_samples": decoded_samples,
                    "readable": decoded_frames > 0 and decoded_samples > 0,
                }
            )
    except Exception as exc:
        result["error"] = str(exc)
    return result
