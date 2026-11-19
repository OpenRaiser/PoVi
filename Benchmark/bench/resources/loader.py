"""Runtime resource resolution with an explicit external override."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path


def resource_root(override: Path | str | None = None) -> Path:
    if override is not None:
        return Path(override).resolve()
    environment = os.environ.get("BENCH_RESOURCE_ROOT", "").strip()
    if environment:
        return Path(environment).resolve()
    return Path(str(files("bench.resources")))


def resource_path(category: str, name: str | None = None) -> Path:
    root = resource_root() / category
    return root / name if name is not None else root
