from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .atomic import atomic_write_text


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
