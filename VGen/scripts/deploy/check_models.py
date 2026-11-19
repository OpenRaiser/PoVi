#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_REQUIREMENTS = {
    "qwen_image": [
        "model_index.json",
        "transformer",
        "text_encoder",
        "tokenizer",
        "vae",
    ],
    "cosyvoice": [
        "cosyvoice3.yaml",
        "llm.pt",
        "flow.pt",
        "hift.pt",
    ],
    "wan_i2v": [
        "configuration.json",
        "Wan2.1_VAE.pth",
        "high_noise_model",
        "low_noise_model",
    ],
    "qwen35_27b": [
        "config.json",
        "tokenizer.json",
    ],
}


def missing_index_shards(root: Path) -> list[str]:
    missing: list[str] = []
    for index_file in root.rglob("*.safetensors.index.json"):
        data = json.loads(index_file.read_text())
        weight_map = data.get("weight_map", {})
        for shard in sorted(set(weight_map.values())):
            shard_path = index_file.parent / shard
            if not shard_path.exists():
                missing.append(str(shard_path))
    return missing


def check_model(name: str, root: Path) -> list[str]:
    missing: list[str] = []
    for item in MODEL_REQUIREMENTS[name]:
        if not (root / item).exists():
            missing.append(str(root / item))
    if root.exists():
        missing.extend(missing_index_shards(root))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VGen model directories.")
    parser.add_argument("--qwen-image", required=True)
    parser.add_argument("--cosyvoice", required=True)
    parser.add_argument("--wan-i2v", required=True)
    parser.add_argument("--qwen35-27b", required=True)
    args = parser.parse_args()

    paths = {
        "qwen_image": Path(args.qwen_image),
        "cosyvoice": Path(args.cosyvoice),
        "wan_i2v": Path(args.wan_i2v),
        "qwen35_27b": Path(args.qwen35_27b),
    }

    failed = False
    for name, path in paths.items():
        missing = check_model(name, path)
        if missing:
            failed = True
            print(f"[FAIL] {name}: {path}")
            for item in missing:
                print(f"  missing: {item}")
        else:
            print(f"[ OK ] {name}: {path}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
