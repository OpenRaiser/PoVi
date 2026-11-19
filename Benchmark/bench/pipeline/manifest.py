"""Versioned, hash-addressed run manifest construction."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from ..resources.loader import resource_root
from ..storage.atomic import canonical_json_hash, sha256_file


MANIFEST_VERSION = "1.1"
SUPPORTED_MANIFEST_VERSIONS = {"1.0", MANIFEST_VERSION}


def resource_bundle_hash() -> str:
    entries: list[tuple[str, str]] = []
    root = resource_root()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append((path.relative_to(root).as_posix(), sha256_file(path)))
    return canonical_json_hash(entries)


def stage_output_hashes(stages: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict):
            continue
        paths: dict[str, str] = {}
        for key in (
            "output",
            "manifest",
            "package",
            "transcript_path",
            "result_path",
        ):
            value = stage.get(key)
            if not isinstance(value, str):
                continue
            path = Path(value)
            if path.exists() and path.is_file():
                paths[key] = sha256_file(path)
        if paths:
            result[stage_name] = paths
    return result


def build_manifest(
    *,
    input_hashes: dict[str, str],
    stages: dict[str, Any],
    errors: list[str],
    registry_version: str,
    evaluation_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    plan = evaluation_plan or {}
    return {
        "manifest_version": MANIFEST_VERSION,
        "benchmark_version": __version__,
        "created_at": now,
        "updated_at": now,
        "ok": not errors,
        "hash_algorithm": "sha256",
        "input_hashes": dict(sorted(input_hashes.items())),
        "runtime_contract": {
            "registry_version": registry_version,
            "resource_bundle_sha256": resource_bundle_hash(),
            "evaluation_plan_sha256": canonical_json_hash(plan),
        },
        "stage_output_hashes": stage_output_hashes(stages),
        "stages": stages,
        "errors": errors,
    }


def manifest_matches(
    manifest: dict[str, Any],
    *,
    input_hashes: dict[str, str],
    registry_version: str,
    evaluation_plan: dict[str, Any],
) -> bool:
    contract = manifest.get("runtime_contract") or {}
    return (
        manifest.get("manifest_version") in SUPPORTED_MANIFEST_VERSIONS
        and manifest.get("input_hashes") == input_hashes
        and contract.get("registry_version") == registry_version
        and contract.get("resource_bundle_sha256") == resource_bundle_hash()
        and contract.get("evaluation_plan_sha256") == canonical_json_hash(evaluation_plan)
    )
