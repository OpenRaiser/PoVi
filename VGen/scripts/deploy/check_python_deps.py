#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys


DEP_GROUPS = {
    "diffsynth": [
        "torch",
        "transformers",
        "diffsynth",
        "safetensors",
        "imageio",
        "PIL",
        "requests",
        "vgen",
    ],
    "qwen35_27b": [
        "torch",
        "transformers",
        "vllm",
    ],
    "cosyvoice": [
        "torch",
        "torchaudio",
        "cosyvoice",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check required Python imports.")
    parser.add_argument("--group", required=True, choices=sorted(DEP_GROUPS))
    args = parser.parse_args()

    missing = [name for name in DEP_GROUPS[args.group] if importlib.util.find_spec(name) is None]
    if missing:
        print(f"[FAIL] {args.group}: missing {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[ OK ] {args.group}: all imports available")


if __name__ == "__main__":
    main()
