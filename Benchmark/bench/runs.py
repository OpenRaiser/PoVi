"""Explicit lifecycle state and path management for benchmark runs."""

from __future__ import annotations

import os
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator

from .storage.atomic import sha256_file
from .storage.json_files import write_json
from .validation import load_json


RUN_RECORD_VERSION = "1.1"


class RunState(str, Enum):
    CREATED = "CREATED"
    PREPARED = "PREPARED"
    AUTO_COMPLETED = "AUTO_COMPLETED"
    HUMAN_EXPORTED = "HUMAN_EXPORTED"
    HUMAN_PARTIAL = "HUMAN_PARTIAL"
    HUMAN_COMPLETED = "HUMAN_COMPLETED"
    FINALIZED = "FINALIZED"


class PreparationStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class MachineStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STALE = "STALE"


class HumanStatus(str, Enum):
    PENDING = "PENDING"
    EXPORTED = "EXPORTED"
    PARTIAL = "PARTIAL"
    COMPLETED = "COMPLETED"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """The only authority for locating runs and reading lifecycle state."""

    def __init__(self, root: Path | str = "runs"):
        self.root = Path(root).resolve()

    def run_dir(self, run_id: str) -> Path:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if not run_id or any(char not in allowed for char in run_id):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        return self.root / run_id

    def record_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def create(
        self,
        *,
        task_input: Path,
        gold: Path,
        submission: Path,
        options: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        inputs = {
            "task_input": task_input.resolve(),
            "gold": gold.resolve(),
            "submission": submission.resolve(),
        }
        for name, path in inputs.items():
            if not path.is_file():
                raise ValueError(f"{name} does not exist: {path}")
        if run_id is not None:
            run_dir = self.run_dir(run_id)
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise ValueError(f"run_id already exists: {run_id}") from exc
        else:
            while True:
                run_id = f"run_{now.strftime('%Y%m%dT%H%M%SZ')}_{secrets.token_hex(6)}"
                run_dir = self.run_dir(run_id)
                try:
                    run_dir.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    continue
        timestamp = utc_now()
        record: dict[str, Any] = {
            "record_version": RUN_RECORD_VERSION,
            "run_id": run_id,
            "state": RunState.CREATED.value,
            "created_at": timestamp,
            "updated_at": timestamp,
            "inputs": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in inputs.items()
            },
            "options": dict(options or {}),
            "blinding_salt": secrets.token_hex(32),
            "preparation": {
                "status": PreparationStatus.PENDING.value,
                "human_ready": False,
                "machine_ready": False,
                "manifest": None,
                "prepared_at": None,
                "last_error": None,
            },
            "machine": {
                "status": MachineStatus.PENDING.value,
                "manifest": None,
                "completed_at": None,
                "last_error": None,
            },
            "human": {
                "status": HumanStatus.PENDING.value,
                "rubric_version": None,
                "item_count": 0,
                "required_annotation_count": 0,
                "validated_annotation_count": 0,
                "pending_item_count": 0,
                "pending_adjudication_count": 0,
                "imports": [],
                "items": {},
                "assignments": {},
            },
            "final": None,
            "last_error": None,
            "history": [
                {
                    "at": timestamp,
                    "from": None,
                    "to": RunState.CREATED.value,
                    "operation": "create",
                }
            ],
        }
        write_json(self.record_path(str(run_id)), record)
        return record

    def load(self, run_id: str) -> dict[str, Any]:
        path = self.record_path(run_id)
        if not path.is_file():
            raise ValueError(f"Unknown run_id: {run_id}")
        record = load_json(path)
        if record.get("run_id") != run_id:
            raise ValueError(f"Run record identity mismatch for {run_id}")
        RunState(str(record.get("state")))
        return self._normalize_record(record)

    def save(self, record: dict[str, Any]) -> None:
        run_id = str(record.get("run_id") or "")
        record["record_version"] = RUN_RECORD_VERSION
        record["updated_at"] = utc_now()
        write_json(self.record_path(run_id), record)

    @staticmethod
    def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        """Add 1.1 phase state in memory without eagerly rewriting old runs."""
        state = RunState(str(record.get("state")))
        legacy = str(record.get("record_version") or "1.0") == "1.0"
        if legacy:
            prepared = state != RunState.CREATED
            machine_complete = state not in {RunState.CREATED, RunState.PREPARED}
            human_status = {
                RunState.HUMAN_EXPORTED: HumanStatus.EXPORTED,
                RunState.HUMAN_PARTIAL: HumanStatus.PARTIAL,
                RunState.HUMAN_COMPLETED: HumanStatus.COMPLETED,
                RunState.FINALIZED: HumanStatus.COMPLETED,
            }.get(state, HumanStatus.PENDING)
            record.setdefault(
                "preparation",
                {
                    "status": (
                        PreparationStatus.COMPLETED.value
                        if prepared
                        else PreparationStatus.PENDING.value
                    ),
                    "human_ready": prepared,
                    "machine_ready": prepared,
                    "manifest": None,
                    "prepared_at": record.get("updated_at") if prepared else None,
                    "last_error": None,
                },
            )
            record.setdefault(
                "machine",
                {
                    "status": (
                        MachineStatus.COMPLETED.value
                        if machine_complete
                        else MachineStatus.PENDING.value
                    ),
                    "manifest": None,
                    "completed_at": (
                        record.get("updated_at") if machine_complete else None
                    ),
                    "last_error": None,
                },
            )
            record.setdefault("human", {})["status"] = human_status.value
        else:
            record.setdefault("preparation", {})
            record["preparation"].setdefault(
                "status", PreparationStatus.PENDING.value
            )
            record["preparation"].setdefault("human_ready", False)
            record["preparation"].setdefault("machine_ready", False)
            record["preparation"].setdefault("last_error", None)
            record.setdefault("machine", {})
            record["machine"].setdefault("status", MachineStatus.PENDING.value)
            record["machine"].setdefault("last_error", None)
            record.setdefault("human", {}).setdefault(
                "status", HumanStatus.PENDING.value
            )
        PreparationStatus(str(record["preparation"]["status"]))
        MachineStatus(str(record["machine"]["status"]))
        HumanStatus(str(record["human"]["status"]))
        return record

    @staticmethod
    def derive_state(record: dict[str, Any]) -> RunState:
        if (
            record.get("final") is not None
            or record.get("state") == RunState.FINALIZED.value
        ):
            return RunState.FINALIZED
        human = HumanStatus(
            str(
                (record.get("human") or {}).get(
                    "status", HumanStatus.PENDING.value
                )
            )
        )
        if human == HumanStatus.COMPLETED:
            return RunState.HUMAN_COMPLETED
        if human == HumanStatus.PARTIAL:
            return RunState.HUMAN_PARTIAL
        if human == HumanStatus.EXPORTED:
            return RunState.HUMAN_EXPORTED
        machine = MachineStatus(
            str(
                (record.get("machine") or {}).get(
                    "status", MachineStatus.PENDING.value
                )
            )
        )
        if machine == MachineStatus.COMPLETED:
            return RunState.AUTO_COMPLETED
        preparation = record.get("preparation") or {}
        preparation_status = PreparationStatus(
            str(preparation.get("status", PreparationStatus.PENDING.value))
        )
        if bool(preparation.get("human_ready")) or preparation_status in {
            PreparationStatus.COMPLETED,
            PreparationStatus.PARTIAL,
        }:
            return RunState.PREPARED
        return RunState.CREATED

    def set_phase_status(
        self,
        record: dict[str, Any],
        phase: str,
        status: str,
        *,
        operation: str,
        **fields: Any,
    ) -> None:
        if phase not in {"preparation", "machine", "human"}:
            raise ValueError(f"Unknown run phase: {phase}")
        current = RunState(str(record["state"]))
        payload = record.setdefault(phase, {})
        payload["status"] = status
        payload.update(fields)
        target = self.derive_state(record)
        record["state"] = target.value
        record["last_error"] = None
        record.setdefault("history", []).append(
            {
                "at": utc_now(),
                "from": current.value,
                "to": target.value,
                "operation": operation,
                "phase": phase,
                "phase_status": status,
            }
        )
        self.save(record)

    def mark_machine_stale(self, record: dict[str, Any], *, reason: str) -> None:
        machine = record.get("machine") or {}
        if machine.get("status") != MachineStatus.COMPLETED.value:
            return
        self.set_phase_status(
            record,
            "machine",
            MachineStatus.STALE.value,
            operation="invalidate-machine",
            last_error={"at": utc_now(), "message": reason},
        )

    def transition(
        self,
        record: dict[str, Any],
        target: RunState,
        *,
        allowed_from: Iterable[RunState],
        operation: str,
    ) -> None:
        current = RunState(str(record["state"]))
        allowed = set(allowed_from)
        if current not in allowed:
            expected = ", ".join(sorted(state.value for state in allowed))
            raise ValueError(
                f"Cannot {operation} run in state {current.value}; expected {expected}"
            )
        timestamp = utc_now()
        record["state"] = target.value
        if target == RunState.AUTO_COMPLETED:
            record.setdefault("machine", {})["status"] = (
                MachineStatus.COMPLETED.value
            )
        elif target == RunState.HUMAN_EXPORTED:
            record.setdefault("human", {})["status"] = HumanStatus.EXPORTED.value
        elif target == RunState.HUMAN_PARTIAL:
            record.setdefault("human", {})["status"] = HumanStatus.PARTIAL.value
        elif target == RunState.HUMAN_COMPLETED:
            record.setdefault("human", {})["status"] = HumanStatus.COMPLETED.value
        record["last_error"] = None
        record.setdefault("history", []).append(
            {
                "at": timestamp,
                "from": current.value,
                "to": target.value,
                "operation": operation,
            }
        )
        self.save(record)

    def record_error(
        self, record: dict[str, Any], *, operation: str, error: str
    ) -> None:
        record["last_error"] = {
            "at": utc_now(),
            "operation": operation,
            "message": error,
        }
        self.save(record)

    def record_phase_error(
        self, record: dict[str, Any], *, phase: str, operation: str, error: str
    ) -> None:
        payload = record.setdefault(phase, {})
        payload["last_error"] = {"at": utc_now(), "message": error}
        record["last_error"] = {
            "at": utc_now(),
            "operation": operation,
            "message": error,
        }
        self.save(record)

    @contextmanager
    def operation_lock(self, run_id: str, operation: str) -> Iterator[None]:
        """Reject concurrent mutating commands for one run."""
        lock_path = self.run_dir(run_id) / ".run.lock"
        try:
            descriptor = os.open(
                lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError as exc:
            owner = ""
            try:
                owner = lock_path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
            suffix = f" ({owner})" if owner else ""
            raise ValueError(
                f"Run {run_id} is locked by another operation{suffix}"
            ) from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(
                    f"pid={os.getpid()} operation={operation} at={utc_now()}\n"
                )
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def input_path(record: dict[str, Any], name: str) -> Path:
        entry = (record.get("inputs") or {}).get(name) or {}
        path = Path(str(entry.get("path") or ""))
        if not path.is_file():
            raise ValueError(f"Recorded input is missing: {name}={path}")
        expected = str(entry.get("sha256") or "")
        actual = sha256_file(path)
        if expected != actual:
            raise ValueError(
                f"Recorded input changed after run creation: {name}"
            )
        return path
