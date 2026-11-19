from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import resolve_artifact_path
from .results.models import Judgment
from .tracks import resolve_route


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _optional_jsonschema_validate(payload: dict[str, Any], schema_path: Path) -> list[str]:
    try:
        import jsonschema
    except Exception:
        return []
    schema = load_json(schema_path)
    validator = jsonschema.Draft202012Validator(schema)
    validation_errors = sorted(
        validator.iter_errors(payload),
        key=lambda item: list(item.absolute_path),
    )
    return [
        f"{'/'.join(str(value) for value in error.absolute_path)}: {error.message}"
        for error in validation_errors
    ]


def validate_task_input(payload: dict[str, Any], schema_path: Path | None = None) -> list[str]:
    errors: list[str] = []
    # These fields were public "evidence packet"/condition aliases in V3.  V4
    # uses track + setting as the only public condition source; hidden Gold is
    # intentionally validated separately.
    for key in (
        "provided_sources",
        "input_condition",
        "oracle_artifacts",
        "metadata",
    ):
        if key in payload:
            errors.append(f"Removed task input field: {key}")
    for key in (
        "schema_version",
        "sample_id",
        "track",
        "setting",
        "poem",
        "task",
    ):
        if key not in payload:
            errors.append(f"Missing task input field: {key}")
    poem = payload.get("poem")
    if isinstance(poem, dict):
        for key in ("title", "author", "dynasty", "text", "lines"):
            if key not in poem:
                errors.append(f"Missing poem field: {key}")
    else:
        errors.append("poem must be an object")
    track = str(payload.get("track") or "")
    setting = str(payload.get("setting") or "")
    try:
        resolve_route(track, setting)
    except ValueError as exc:
        errors.append(str(exc))
    required_condition_fields = {
        "visual_generation": {"visual_prompt"},
        "audio_generation": {"narration_text"},
        "audio_visual_generation": {"visual_prompt", "narration_text"},
    }
    condition_inputs = payload.get("condition_inputs")
    expected_fields = required_condition_fields.get(track)
    if expected_fields is None:
        if "condition_inputs" in payload:
            errors.append(f"condition_inputs is not allowed for track: {track}")
    elif not isinstance(condition_inputs, dict):
        errors.append(f"Missing condition_inputs for track: {track}")
    else:
        actual_fields = set(condition_inputs)
        if actual_fields != expected_fields:
            expected = ", ".join(sorted(expected_fields))
            errors.append(
                f"condition_inputs for {track} must contain exactly: {expected}"
            )
        for field in expected_fields:
            if not isinstance(condition_inputs.get(field), str) or not condition_inputs[field].strip():
                errors.append(f"condition_inputs/{field} must be a non-empty string")
    if schema_path is not None:
        errors.extend(_optional_jsonschema_validate(payload, schema_path))
    return sorted(set(errors))


def validate_gold_annotation(
    payload: dict[str, Any],
    schema_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    for key in (
        "schema_version",
        "sample_id",
        "gold_status",
        "evidence_claims",
        "required_information",
        "metaphor_maps",
        "visual_requirements",
        "key_events",
        "pronunciation_lexicon",
        "learning_items",
    ):
        if key not in payload:
            errors.append(f"Missing gold field: {key}")
    if schema_path is not None:
        errors.extend(_optional_jsonschema_validate(payload, schema_path))
    id_fields = {
        "evidence_claims": "claim_id",
        "required_information": "unit_id",
        "metaphor_maps": "metaphor_id",
        "visual_requirements": "requirement_id",
        "key_events": "event_id",
        "learning_items": "item_id",
        "consistency_groups": "group_id",
        "event_relations": "relation_id",
        "generation_requirements": "requirement_id",
        "alignment_relations": "relation_id",
    }
    for collection, id_field in id_fields.items():
        seen: set[str] = set()
        for index, item in enumerate(payload.get(collection) or []):
            if not isinstance(item, dict):
                continue
            value = str(item.get(id_field) or "")
            if value and value in seen:
                errors.append(f"Duplicate {collection}/{id_field}: {value}")
            seen.add(value)
    claim_ids = {
        str(item.get("claim_id"))
        for item in payload.get("evidence_claims") or []
        if isinstance(item, dict) and item.get("claim_id")
    }
    metaphor_ids = {
        str(item.get("metaphor_id"))
        for item in payload.get("metaphor_maps") or []
        if isinstance(item, dict) and item.get("metaphor_id")
    }
    for item in payload.get("required_information") or []:
        if not isinstance(item, dict):
            continue
        for claim_id in item.get("source_claim_ids") or []:
            if str(claim_id) not in claim_ids:
                errors.append(f"Unknown source_claim_id: {claim_id}")
    for item in payload.get("visual_requirements") or []:
        if not isinstance(item, dict):
            continue
        for metaphor_id in item.get("metaphor_ids") or []:
            if str(metaphor_id) not in metaphor_ids:
                errors.append(f"Unknown metaphor_id: {metaphor_id}")
    return sorted(set(errors))


def validate_submission(
    payload: dict[str, Any],
    submission_dir: Path,
    schema_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    for key in ("schema_version", "system_id", "sample_id", "track", "setting", "artifacts"):
        if key not in payload:
            errors.append(f"Missing submission field: {key}")
    track = str(payload.get("track") or "")
    setting = str(payload.get("setting") or "")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
        artifacts = {}
    route = None
    try:
        route = resolve_route(track, setting)
    except ValueError as exc:
        errors.append(str(exc))
    for name, value in artifacts.items():
        if isinstance(value, str) and value.strip():
            try:
                path = resolve_artifact_path(submission_dir, value)
            except ValueError as exc:
                errors.append(f"Invalid artifact {name}: {exc}")
                continue
        elif isinstance(value, list):
            for item in value:
                if name == "prompt_media_pairs" and isinstance(item, dict):
                    for field in ("prompt", "media"):
                        path_value = item.get(field)
                        if not isinstance(path_value, str) or not path_value.strip():
                            continue
                        try:
                            resolve_artifact_path(submission_dir, path_value)
                        except ValueError as exc:
                            errors.append(
                                f"Invalid artifact {name}/{field}: {exc}"
                            )
                    continue
                if not isinstance(item, str):
                    continue
                try:
                    path = resolve_artifact_path(submission_dir, str(item))
                except ValueError as exc:
                    errors.append(f"Invalid artifact {name}: {exc}")
    if schema_path is not None:
        errors.extend(_optional_jsonschema_validate(payload, schema_path))
    return sorted(set(errors))


def validate_pair(
    task_input: dict[str, Any],
    submission: dict[str, Any],
    gold: dict[str, Any] | None = None,
) -> list[str]:
    errors = []
    if task_input.get("sample_id") != submission.get("sample_id"):
        errors.append(
            "sample_id mismatch: "
            f"task={task_input.get('sample_id')} submission={submission.get('sample_id')}"
        )
    for key in ("track", "setting"):
        if task_input.get(key) != submission.get(key):
            errors.append(
                f"{key} mismatch: task={task_input.get(key)} "
                f"submission={submission.get(key)}"
            )
    if gold is not None and gold.get("sample_id") != task_input.get("sample_id"):
        errors.append(
            "sample_id mismatch: "
            f"task={task_input.get('sample_id')} gold={gold.get('sample_id')}"
        )
    return errors


def _validate_experimental_judgment_source(
    item: dict[str, Any],
    spec: Any,
    *,
    prefix: str,
) -> list[str]:
    """Validate provenance required by an experimental, disabled metric."""
    accepted_source = spec.accepted_source
    if not accepted_source:
        return []
    errors: list[str] = []
    if item.get("source") != accepted_source:
        errors.append(
            f"{prefix}: Metric {spec.metric_id} only accepts source "
            f"{accepted_source!r}"
        )
    judge = item.get("judge") or {}
    if judge.get("type") != accepted_source:
        errors.append(
            f"{prefix}: Metric {spec.metric_id} requires judge.type="
            f"{accepted_source!r}, not a normal human score"
        )
    raw = item.get("raw") or {}
    experiment = raw.get(accepted_source) if isinstance(raw, dict) else None
    if not isinstance(experiment, dict):
        errors.append(
            f"{prefix}: Metric {spec.metric_id} requires raw."
            f"{accepted_source} experiment metadata"
        )
        return errors
    for field in ("experiment_id", "protocol_version", "data_hash"):
        if not isinstance(experiment.get(field), str) or not experiment[field].strip():
            errors.append(
                f"{prefix}: Metric {spec.metric_id} raw.{accepted_source}."
                f"{field} must be a non-empty string"
            )
    return errors


def validate_judgments(
    payload: dict[str, Any],
    schema_path: Path | None = None,
    registry: Any | None = None,
) -> list[str]:
    errors: list[str] = []
    judgments = payload.get("judgments")
    if not isinstance(judgments, list):
        errors.append("judgments must be an array")
        judgments = []
    for index, item in enumerate(judgments):
        if not isinstance(item, dict):
            errors.append(f"judgments/{index}: must be an object")
            continue
        try:
            Judgment.from_dict(item)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(
                f"judgments/{index}: invalid canonical Judgment: {exc}"
            )
        track = str(item.get("track") or "")
        setting = str(item.get("setting") or "")
        try:
            resolve_route(track, setting)
        except ValueError as exc:
            errors.append(f"judgments/{index}: {exc}")
            continue
        if registry is not None:
            metric_id = str(item.get("metric_id") or "")
            if metric_id not in registry.metrics:
                try:
                    registry.get(metric_id)
                except KeyError as exc:
                    errors.append(f"judgments/{index}: {exc.args[0]}")
                continue
            spec = registry.get(metric_id)
            score = item.get("score")
            if score is not None:
                try:
                    spec.validate_score(score)
                except ValueError as exc:
                    errors.append(f"judgments/{index}: {exc}")
            errors.extend(
                _validate_experimental_judgment_source(
                    item,
                    spec,
                    prefix=f"judgments/{index}",
                )
            )
            allowed = set(
                registry.applicable_metric_ids(
                    track,
                    setting,
                    include_diagnostics=True,
                    include_disabled=bool(spec.accepted_source),
                )
            )
            if metric_id not in allowed:
                errors.append(
                    f"judgments/{index}: Metric {metric_id} is not applicable to "
                    f"{track}/{setting}"
                )
    if schema_path is not None:
        errors.extend(_optional_jsonschema_validate(payload, schema_path))
    return sorted(set(errors))
