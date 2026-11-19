from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..progress import emit_progress


def extract_timestamped_frames(
    video_path: Path,
    output_dir: Path,
    *,
    interval_seconds: float = 2.0,
    segment_seconds: float = 30.0,
    max_frames_per_segment: int = 16,
    max_edge: int = 1280,
) -> dict[str, Any]:
    if interval_seconds <= 0 or segment_seconds <= 0 or max_frames_per_segment <= 0:
        raise ValueError("Frame sampling values must be positive")
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Missing dependency: av") from exc

    video_path = video_path.resolve()
    output_dir = output_dir.resolve()
    emit_progress(
        "frames",
        "progress",
        "开始解码视频帧",
        interval_seconds=interval_seconds,
        segment_seconds=segment_seconds,
        max_frames_per_segment=max_frames_per_segment,
    )
    frame_root = output_dir / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    per_segment: dict[int, int] = defaultdict(int)
    next_timestamp = 0.0
    duration_seconds: float | None = None
    with av.open(str(video_path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"No video stream in {video_path}")
        stream = streams[0]
        if container.duration is not None:
            duration_seconds = float(container.duration / av.time_base)
            emit_progress(
                "frames",
                "progress",
                "已读取视频时长",
                duration_seconds=duration_seconds,
                estimated_segments=(int(duration_seconds // segment_seconds) + 1),
            )
        for frame in container.decode(stream):
            timestamp = frame.time
            if timestamp is None and frame.pts is not None and stream.time_base is not None:
                timestamp = float(frame.pts * stream.time_base)
            if timestamp is None:
                continue
            timestamp = max(0.0, float(timestamp))
            if timestamp + 1e-6 < next_timestamp:
                continue
            segment_index = int(timestamp // segment_seconds)
            if per_segment[segment_index] >= max_frames_per_segment:
                next_timestamp = (segment_index + 1) * segment_seconds
                continue
            segment_dir = frame_root / f"segment_{segment_index:04d}"
            segment_dir.mkdir(parents=True, exist_ok=True)
            frame_index = len(frames)
            output_path = segment_dir / f"frame_{frame_index:06d}_{timestamp:010.3f}s.jpg"
            image = frame.to_image()
            image.thumbnail((max_edge, max_edge))
            image.convert("RGB").save(output_path, format="JPEG", quality=88, optimize=True)
            frames.append(
                {
                    "frame_id": f"frame_{frame_index:06d}",
                    "timestamp_seconds": round(timestamp, 6),
                    "segment_index": segment_index,
                    "path": output_path.relative_to(output_dir).as_posix(),
                }
            )
            per_segment[segment_index] += 1
            next_timestamp = timestamp + interval_seconds
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    segments = []
    for segment_index in sorted(per_segment):
        selected = [item for item in frames if item["segment_index"] == segment_index]
        segments.append(
            {
                "segment_index": segment_index,
                "start_seconds": segment_index * segment_seconds,
                "end_seconds": (segment_index + 1) * segment_seconds,
                "frame_ids": [item["frame_id"] for item in selected],
            }
        )
    manifest = {
        "schema_version": "0.1",
        "source_video": str(video_path),
        "duration_seconds": duration_seconds,
        "sampling": {
            "interval_seconds": interval_seconds,
            "segment_seconds": segment_seconds,
            "max_frames_per_segment": max_frames_per_segment,
            "max_edge": max_edge,
        },
        "frames": frames,
        "segments": segments,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frame_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    emit_progress(
        "frames",
        "progress",
        "帧文件和清单已写入",
        frames=len(frames),
        segments=len(segments),
        output=output_dir / "frame_manifest.json",
    )
    return manifest
