from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioArtifactResolution:
    """The single selected audio source for all evaluation stages."""

    source_type: str
    path: Path | None
    sha256: str | None
    fallback_reason: str | None = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "fallback_reason": self.fallback_reason,
            "warnings": list(self.warnings),
        }


def resolve_artifact_path(root: Path, value: str) -> Path:
    """Resolve a submission-relative artifact path.

    Submission manifests may point to symlinks created by local adapters, but
    the declared path itself must stay inside the submission tree.
    """
    relative = Path(value)
    if not value.strip() or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Artifact path must be submission-relative: {value!r}")
    return (root / relative).resolve()


def _existing_artifact_path(root: Path, value: Any) -> tuple[Path | None, str]:
    if not isinstance(value, str) or not value.strip():
        return None, "not_declared"
    try:
        path = resolve_artifact_path(root, value)
    except ValueError:
        return None, "invalid_path"
    if not path.is_file():
        return None, "not_found"
    return path, "available"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_submitted_content_text(
    submission: dict[str, Any], submission_dir: Path
) -> bool:
    """True when the submission itself declares usable C-domain evaluation text.

    Submitted text outranks evaluator ASR transcription for content evaluation;
    evaluator ASR is only needed when no submitted text exists.  A declared
    artifact only counts when it resolves to an existing file with non-empty
    content, matching what text consumers will actually read.
    """
    artifacts = submission.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    for name in ("narration", "planning_output"):
        value = artifacts.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            path = resolve_artifact_path(submission_dir, value)
        except ValueError:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if content.strip():
            return True
    return False


def resolve_audio_artifact(
    submission: dict[str, Any], submission_dir: Path
) -> AudioArtifactResolution:
    """Select the authoritative audio input with a safe video fallback.

    A declared independent audio file wins only when it exists as a regular
    file.  Otherwise an existing ``final_video`` is selected so Gate checks,
    ASR, deterministic reports, and blinded human packages share one source.
    """
    artifacts = submission.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    audio_path, audio_state = _existing_artifact_path(
        submission_dir, artifacts.get("audio")
    )
    if audio_path is not None:
        return AudioArtifactResolution(
            source_type="audio",
            path=audio_path,
            sha256=_sha256_file(audio_path),
        )

    video_path, video_state = _existing_artifact_path(
        submission_dir, artifacts.get("final_video")
    )
    if video_path is not None:
        fallback_reason = (
            "audio_not_declared"
            if audio_state == "not_declared"
            else "declared_audio_unavailable"
        )
        warnings = ()
        if audio_state != "not_declared":
            warnings = (
                "Declared audio artifact is unavailable; using final_video fallback.",
            )
        return AudioArtifactResolution(
            source_type="final_video",
            path=video_path,
            sha256=_sha256_file(video_path),
            fallback_reason=fallback_reason,
            warnings=warnings,
        )

    warnings = ()
    if audio_state != "not_declared":
        warnings = ("Declared audio artifact is unavailable and no final_video fallback exists.",)
    return AudioArtifactResolution(
        source_type="missing",
        path=None,
        sha256=None,
        fallback_reason=(
            "no_usable_audio_or_final_video"
            if video_state == "not_declared"
            else "no_usable_audio_or_final_video"
        ),
        warnings=warnings,
    )
