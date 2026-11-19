from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .progress import emit_progress


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def sanitize(self, value: object) -> str:
        text = str(value)
        return text.replace(self.api_key, "***") if self.api_key else text


@dataclass
class RemoteResult:
    payload: dict[str, Any]
    raw_text: str
    provider: str
    model: str
    attempts: list[dict[str, Any]] = field(default_factory=list)


def provider_config(name: str) -> ProviderConfig:
    normalized = name.lower()
    if normalized == "text_judge":
        return ProviderConfig(
            name="text_judge",
            base_url=os.environ.get("TEXT_JUDGE_BASE_URL", "https://api.deepseek.com"),
            api_key=os.environ.get("TEXT_JUDGE_API_KEY", ""),
            model=os.environ.get("TEXT_JUDGE_MODEL", "deepseek-v4-pro"),
        )
    if normalized == "text_judge_fallback":
        return ProviderConfig(
            name="text_judge_fallback",
            base_url=os.environ.get("TEXT_JUDGE_FALLBACK_BASE_URL")
            or os.environ.get("LLM_BASE_URL", "http://llm-api.model-eval.woa.com/v1"),
            api_key=os.environ.get("TEXT_JUDGE_FALLBACK_API_KEY")
            or os.environ.get("OPENAI_API_KEY", ""),
            model=os.environ.get("TEXT_JUDGE_FALLBACK_MODEL")
            or os.environ.get("LLM_MODEL", "doubao/doubao-seed-2-0-lite"),
        )
    if normalized == "speech_transcription":
        return ProviderConfig(
            name="speech_transcription",
            base_url=os.environ.get("SPEECH_TRANSCRIPTION_BASE_URL", "https://api.siliconflow.cn/v1"),
            api_key=os.environ.get("SPEECH_TRANSCRIPTION_API_KEY", ""),
            model=os.environ.get("SPEECH_TRANSCRIPTION_MODEL", "TeleAI/TeleSpeechASR"),
        )
    if normalized == "multimodal_judge":
        return ProviderConfig(
            name="multimodal_judge",
            base_url=os.environ.get("MULTIMODAL_JUDGE_BASE_URL", "https://api.siliconflow.cn/v1"),
            api_key=os.environ.get("MULTIMODAL_JUDGE_API_KEY", ""),
            model=os.environ.get(
                "MULTIMODAL_JUDGE_MODEL", "Qwen/Qwen3.5-397B-A17B"
            ),
        )
    raise ValueError(f"Unknown provider: {name}")


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Remote JSON response must be an object")
    return payload


def _status_code(error: BaseException) -> int | None:
    value = getattr(error, "status_code", None)
    if value is None:
        response = getattr(error, "response", None)
        value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_transient(error: BaseException) -> bool:
    status = _status_code(error)
    if status is not None:
        return status in {408, 409, 429} or status >= 500
    name = type(error).__name__.lower()
    return any(token in name for token in ("timeout", "connection", "ratelimit"))


class OpenAIProvider:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        max_attempts: int = 3,
        client_factory: Callable[..., Any] | None = None,
    ):
        if not config.configured:
            raise ValueError(f"Provider {config.name} is not fully configured")
        self.config = config
        self.max_attempts = max(1, max_attempts)
        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Missing dependency: openai") from exc
            client_factory = OpenAI
        self.client = client_factory(
            base_url=config.base_url.rstrip("/"),
            api_key=config.api_key,
            timeout=300.0,
        )

    def _call(self, operation: Callable[[], Any]) -> tuple[Any, list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation(), attempts
            except Exception as exc:
                transient = _is_transient(exc)
                attempts.append(
                    {
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "status_code": _status_code(exc),
                        "transient": transient,
                        "message": self.config.sanitize(exc),
                    }
                )
                if not transient or attempt >= self.max_attempts:
                    emit_progress(
                        "remote",
                        "fail",
                        "远程请求失败",
                        provider=self.config.name,
                        model=self.config.model,
                        attempt=f"{attempt}/{self.max_attempts}",
                        transient=transient,
                        error_type=type(exc).__name__,
                        status_code=_status_code(exc),
                    )
                    raise RuntimeError(
                        f"{self.config.name} request failed: {self.config.sanitize(exc)}"
                    ) from exc
                delay = min(2 ** (attempt - 1), 4)
                emit_progress(
                    "remote",
                    "retry",
                    "远程请求暂时失败，准备重试",
                    provider=self.config.name,
                    model=self.config.model,
                    attempt=f"{attempt}/{self.max_attempts}",
                    status_code=_status_code(exc),
                    wait_seconds=delay,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 24000,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> RemoteResult:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            request["response_format"] = response_format
        if reasoning_effort is not None:
            request["reasoning_effort"] = reasoning_effort
        response, attempts = self._call(
            lambda: self.client.chat.completions.create(**request)
        )
        raw = response.choices[0].message.content or ""
        return RemoteResult(
            payload=extract_json(raw),
            raw_text=raw,
            provider=self.config.name,
            model=self.config.model,
            attempts=attempts,
        )

    def transcribe(self, audio_path: Path) -> tuple[str, list[dict[str, Any]]]:
        def operation() -> Any:
            with audio_path.open("rb") as handle:
                return self.client.audio.transcriptions.create(
                    model=self.config.model,
                    file=handle,
                )

        response, attempts = self._call(operation)
        if isinstance(response, str):
            text = response
        elif isinstance(response, dict):
            text = str(response.get("text") or "")
        else:
            text = str(getattr(response, "text", "") or "")
        return text.strip(), attempts

    def probe(self) -> dict[str, Any]:
        response, attempts = self._call(lambda: self.client.models.list())
        models = getattr(response, "data", None) or []
        model_ids = [str(getattr(item, "id", "")) for item in models]
        return {
            "ok": True,
            "provider": self.config.name,
            "base_url": self.config.base_url,
            "model": self.config.model,
            "model_visible": self.config.model in model_ids if model_ids else None,
            "attempts": attempts,
        }


class TextProviderRouter:
    def __init__(
        self,
        primary: OpenAIProvider | None = None,
    ):
        if primary is not None:
            self.primary = primary
        elif provider_config("text_judge").configured:
            self.primary = OpenAIProvider(provider_config("text_judge"))
        else:
            raise ValueError("Text-judging API is not configured")

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> RemoteResult:
        # 推理模型（如 deepseek-v4-flash）不限制 reasoning_effort 时会耗尽 max_tokens 返回空内容
        kwargs.setdefault(
            "reasoning_effort",
            os.environ.get("TEXT_JUDGE_REASONING_EFFORT", "low"),
        )
        return self.primary.complete_json(messages, **kwargs)


def doctor_report(*, probe_remote: bool = False) -> dict[str, Any]:
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("openai", "jsonschema", "av", "numpy", "soundfile", "PIL")
    }
    providers: dict[str, Any] = {}
    for name in ("text_judge", "text_judge_fallback", "speech_transcription", "multimodal_judge"):
        config = provider_config(name)
        entry: dict[str, Any] = {
            "configured": config.configured,
            "base_url": config.base_url,
            "model": config.model,
            "api_key_present": bool(config.api_key),
        }
        if probe_remote and config.configured and dependencies["openai"]:
            try:
                entry["probe"] = OpenAIProvider(config).probe()
            except Exception as exc:
                entry["probe"] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "message": config.sanitize(exc),
                }
        providers[name] = entry
    return {
        "ok": all(dependencies.values()) and all(item["configured"] for item in providers.values()),
        "python": os.sys.version,
        "dependencies": dependencies,
        "providers": providers,
    }
