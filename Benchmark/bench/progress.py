from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Any


_DISABLED_VALUES = {"0", "false", "no", "off"}


def progress_enabled() -> bool:
    value = os.environ.get("BENCH_PROGRESS", "1").strip().lower()
    return value not in _DISABLED_VALUES


def elapsed_seconds(started_at: float) -> float:
    return round(max(0.0, time.monotonic() - started_at), 2)


def emit_progress(
    stage: str,
    status: str,
    message: str,
    **details: Any,
) -> None:
    """Write a compact, machine-safe progress line to stderr.

    The evaluate command reserves stdout for its final JSON manifest. Keeping
    progress on stderr lets humans monitor a run without breaking callers that
    parse stdout.
    """
    if not progress_enabled():
        return
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    suffix = " ".join(
        f"{key}={_format_value(value)}"
        for key, value in details.items()
        if value is not None
    )
    line = f"[{timestamp}] [bench] [{stage}] {status.upper()} {message}"
    if suffix:
        line += f" | {suffix}"
    print(line, file=sys.stderr, flush=True)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text if " " not in text else repr(text)
