from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import AudioArtifactResolution, resolve_artifact_path, resolve_audio_artifact
from ..tracks import Route


@dataclass(frozen=True)
class MetricContext:
    task_input: dict[str, Any]
    submission: dict[str, Any]
    submission_dir: Path
    route: Route

    @property
    def artifacts(self) -> dict[str, Any]:
        value = self.submission.get("artifacts")
        return value if isinstance(value, dict) else {}

    def artifact_path(self, name: str) -> Path | None:
        value = str(self.artifacts.get(name) or "").strip()
        if not value:
            return None
        try:
            return resolve_artifact_path(self.submission_dir, value)
        except ValueError:
            return None

    def resolve_audio_artifact(self) -> AudioArtifactResolution:
        return resolve_audio_artifact(self.submission, self.submission_dir)

    def read_text(self, name: str) -> str:
        path = self.artifact_path(name)
        if path is None or not path.exists():
            return ""
        return path.read_text(encoding="utf-8")
