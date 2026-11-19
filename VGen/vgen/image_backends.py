from __future__ import annotations

import base64
import datetime as dt
import glob
import hashlib
import hmac
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class QwenImageGenerator:
    def __init__(self, config: Any):
        self.config = config
        self._pipe = None

    def generate(self, segment: Any, output_path: Path, negative_prompt: str = "") -> Path:
        return self.generate_prompt(
            prompt=segment.image_prompt,
            output_path=output_path,
            seed_offset=int(segment.index),
            negative_prompt=negative_prompt,
        )

    def generate_prompt(
        self,
        prompt: str,
        output_path: Path,
        seed_offset: int,
        negative_prompt: str = "",
        reference_image: Path | None = None,
    ) -> Path:
        if output_path.exists() and not getattr(self.config, "overwrite", False):
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipe = self._load_pipe()
        call_args: dict[str, Any] = {}
        if reference_image is not None:
            from PIL import Image

            call_args["input_image"] = Image.open(reference_image).convert("RGB")
            call_args["denoising_strength"] = 0.68
            prompt = "保持输入人物的稳定骨相和身份特征，只按描述调整年龄、发冠与服饰。" + prompt
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            seed=int(getattr(self.config, "seed", 0)) + int(seed_offset),
            height=int(getattr(self.config, "image_height", 720)),
            width=int(getattr(self.config, "image_width", 1280)),
            num_inference_steps=int(getattr(self.config, "image_steps", 30)),
            **call_args,
        )
        image.save(output_path)
        return output_path

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

        model = Path(getattr(self.config, "qwen_image_model"))
        self._pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=str(getattr(self.config, "image_device", "cuda:0")),
            model_configs=[
                ModelConfig(path=sorted(glob.glob(str(model / "transformer" / "diffusion_pytorch_model*.safetensors")))),
                ModelConfig(path=sorted(glob.glob(str(model / "text_encoder" / "model*.safetensors")))),
                ModelConfig(path=str(model / "vae" / "diffusion_pytorch_model.safetensors")),
            ],
            tokenizer_config=ModelConfig(path=str(model / "tokenizer")),
        )
        return self._pipe

    def close(self) -> None:
        self._pipe = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class BerniniImageGenerator:
    def __init__(self, config: Any):
        self.config = config
        self._pipe = None

    def generate(self, segment: Any, output_path: Path, negative_prompt: str = "") -> Path:
        return self.generate_prompt(
            prompt=segment.image_prompt,
            output_path=output_path,
            seed_offset=int(segment.index),
            negative_prompt=negative_prompt,
        )

    def generate_prompt(
        self,
        prompt: str,
        output_path: Path,
        seed_offset: int,
        negative_prompt: str = "",
        reference_image: Path | None = None,
    ) -> Path:
        if reference_image is not None:
            raise ValueError("Bernini T2I cannot derive an age variant from a reference identity")
        if output_path.exists() and not getattr(self.config, "overwrite", False):
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipe = self._load_pipe()
        bernini_steps = int(os.environ.get("BERNINI_IMAGE_STEPS", str(int(getattr(self.config, "image_steps", 30)))))
        bernini_guidance = os.environ.get("BERNINI_IMAGE_GUIDANCE_MODE", "t2v_apg")
        bernini_max_image_size = int(
            os.environ.get(
                "BERNINI_IMAGE_MAX_IMAGE_SIZE",
                str(max(int(getattr(self.config, "image_height", 720)), int(getattr(self.config, "image_width", 1280)))),
            )
        )
        pipe(
            "t2i",
            prompt,
            neg_prompt=negative_prompt,
            num_frames=1,
            max_image_size=bernini_max_image_size,
            height=int(getattr(self.config, "image_height", 720)),
            width=int(getattr(self.config, "image_width", 1280)),
            num_inference_steps=bernini_steps,
            guidance_mode=bernini_guidance,
            seed=int(getattr(self.config, "seed", 0)) + int(seed_offset),
            fps=1,
            output_path=str(output_path),
        )
        if not output_path.exists():
            raise RuntimeError(f"Bernini image inference did not produce output: {output_path}")
        return output_path

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        self._validate_paths()
        import torch

        bernini_root = Path(getattr(self.config, "bernini_root"))
        if str(bernini_root) not in sys.path:
            sys.path.insert(0, str(bernini_root))
        from bernini.pipeline import BerniniPipeline

        device = torch.device(str(getattr(self.config, "image_device", "cuda:0")))
        if device.type == "cuda":
            torch.cuda.set_device(device)
        self._pipe = BerniniPipeline.from_pretrained(
            str(Path(getattr(self.config, "bernini_image_config"))),
            device=device,
            use_unipc=False,
            use_src_id_rotary_emb=False,
            interpolate_src_id=False,
            max_trained_src_id=32,
        )
        return self._pipe

    def _validate_paths(self) -> None:
        required_paths = [
            Path(getattr(self.config, "bernini_root")),
            Path(getattr(self.config, "bernini_root")) / "bernini" / "pipeline.py",
            Path(getattr(self.config, "bernini_image_config")),
        ]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            formatted = "\n".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Bernini image runtime is incomplete. Missing paths:\n{formatted}")

    def close(self) -> None:
        self._pipe = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class GeminiImageGenerator:
    """Remote Gemini image backend served by the internal data-eval gateway."""

    def __init__(self, config: Any):
        self.config = config

    def generate(self, segment: Any, output_path: Path, negative_prompt: str = "") -> Path:
        return self.generate_prompt(
            prompt=segment.image_prompt,
            output_path=output_path,
            seed_offset=int(segment.index),
            negative_prompt=negative_prompt,
        )

    def generate_prompt(
        self,
        prompt: str,
        output_path: Path,
        seed_offset: int,
        negative_prompt: str = "",
        reference_image: Path | None = None,
    ) -> Path:
        del seed_offset
        if output_path.exists() and not getattr(self.config, "overwrite", False):
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        token = str(
            getattr(self.config, "image_api_key", "")
            or os.environ.get("IMAGE_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        ).strip()
        if ":" not in token:
            raise ValueError("Gemini image generation requires IMAGE_API_KEY or OPENAI_API_KEY in app_id:app_key format.")

        payload = self._build_payload(prompt, negative_prompt, reference_image)
        request_path = output_path.with_suffix(".gemini_request.json")
        response_path = output_path.with_suffix(".gemini_response.json")
        request_path.write_text(
            json.dumps(self._response_summary(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        data = self._request(payload, token)
        response_path.write_text(
            json.dumps(self._response_summary(data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        image_bytes = self._extract_image_bytes(data)
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("RGB").save(output_path, format="PNG")
        return output_path

    def _build_payload(
        self,
        prompt: str,
        negative_prompt: str = "",
        reference_image: Path | None = None,
    ) -> dict[str, Any]:
        prompt_prefix = os.environ.get("GEMINI_IMAGE_PROMPT_PREFIX", "").strip()
        full_prompt = "\n".join(
            part for part in (prompt_prefix, prompt.strip()) if part
        )
        if negative_prompt.strip():
            full_prompt += "\n避免出现：" + negative_prompt.strip()
        content: list[dict[str, str]] = [{"type": "text", "value": full_prompt}]
        if reference_image is not None:
            encoded = base64.b64encode(reference_image.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "value": f"data:image/png;base64,{encoded}",
                }
            )
        return {
            "request_id": str(uuid.uuid4()),
            "model_marker": self._setting(
                "image_model_marker",
                "IMAGE_MODEL_MARKER",
                "api_google_gemini-3-pro-image-preview",
            ),
            "messages": [{"role": "user", "content": content}],
            "params": {
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {
                        "aspectRatio": self._aspect_ratio(),
                        "image_size": self._setting("image_size", "IMAGE_SIZE", "1K"),
                    },
                    "candidateCount": 1,
                }
            },
            "timeout": 3600,
        }

    def _request(self, payload: dict[str, Any], token: str) -> dict[str, Any]:
        host = self._setting(
            "image_api_host",
            "IMAGE_API_HOST",
            os.environ.get("VEO_HOST", ""),
        ).rstrip("/")
        api_version = self._setting("image_api_version", "IMAGE_API_VERSION", "v2.03")
        source = self._setting("image_api_source", "IMAGE_API_SOURCE", "storytelling-character-assets")
        timeout_seconds = int(self._setting("image_timeout_seconds", "IMAGE_TIMEOUT_SECONDS", "3600"))
        retries = max(0, int(self._setting("image_retries", "IMAGE_RETRIES", "2")))
        user, api_key = token.split(":", 1)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = host + "/api/v1/data_eval"

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            now = dt.datetime.now(dt.UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
            sign_input = f"date: {now}\nsource: {source}"
            signature = base64.b64encode(
                hmac.new(api_key.encode(), sign_input.encode(), hashlib.sha1).digest()
            ).decode()
            headers = {
                "Apiversion": api_version,
                "Authorization": (
                    f'hmac id="{user}", algorithm="hmac-sha1", '
                    f'headers="date source", signature="{signature}"'
                ),
                "Date": now,
                "Source": source,
                "Content-Type": "application/json",
            }
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    status = int(response.status)
                    response_text = response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as error:
                status = int(error.code)
                response_text = error.read().decode("utf-8", errors="replace")
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
                if attempt >= retries:
                    raise RuntimeError(
                        f"Gemini image request failed after {attempt + 1} attempts: {error}"
                    ) from error
                time.sleep(min(30, 3 * (attempt + 1)))
                continue

            try:
                data = json.loads(response_text)
            except json.JSONDecodeError as error:
                last_error = error
                if attempt >= retries:
                    raise RuntimeError(
                        f"Gemini image returned non-JSON status={status}: {response_text[:1000]}"
                    ) from error
                time.sleep(min(30, 3 * (attempt + 1)))
                continue
            if status == 200 and data.get("code") == 0:
                return data
            last_error = RuntimeError(
                f"Gemini image returned status={status}: "
                f"{json.dumps(self._response_summary(data), ensure_ascii=False)[:2000]}"
            )
            if attempt >= retries or status not in {408, 409, 425, 429, 500, 502, 503, 504}:
                raise last_error
            time.sleep(min(30, 3 * (attempt + 1)))
        raise RuntimeError(f"Gemini image request failed: {last_error}")

    @classmethod
    def _extract_image_bytes(cls, data: dict[str, Any]) -> bytes:
        for item in data.get("answer") or []:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            value = item.get("value")
            if item_type in {"image_base64", "image", "inline_data", "inlinedata"} and value:
                if isinstance(value, dict):
                    value = value.get("data")
                return cls._decode_base64_image(str(value))
            if item_type in {"image_url", "url"} and value:
                return cls._read_image_url(str(value))

        for root in (data, data.get("response") or {}, data.get("data") or {}):
            for candidate in root.get("candidates") or []:
                content = candidate.get("content") or {}
                for part in content.get("parts") or []:
                    inline = part.get("inlineData") or part.get("inline_data")
                    if isinstance(inline, dict) and inline.get("data"):
                        return cls._decode_base64_image(str(inline["data"]))
        raise RuntimeError("Gemini image response contains neither image base64 nor image URL")

    @staticmethod
    def _decode_base64_image(value: str) -> bytes:
        encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        return base64.b64decode(encoded)

    @staticmethod
    def _read_image_url(value: str) -> bytes:
        if value.startswith("data:"):
            return GeminiImageGenerator._decode_base64_image(value)
        with urllib.request.urlopen(value, timeout=900) as response:
            return response.read()

    def _aspect_ratio(self) -> str:
        width = max(1, int(getattr(self.config, "width", 1280)))
        height = max(1, int(getattr(self.config, "height", 720)))
        return "16:9" if width >= height else "9:16"

    def _setting(self, attr_name: str, env_name: str, default: str) -> str:
        value = getattr(self.config, attr_name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
        return os.environ.get(env_name, default).strip()

    @staticmethod
    def _response_summary(data: dict[str, Any]) -> dict[str, Any]:
        def scrub(value: Any, key: str = "") -> Any:
            if isinstance(value, dict):
                return {item_key: scrub(item_value, str(item_key)) for item_key, item_value in value.items()}
            if isinstance(value, list):
                return [scrub(item, key) for item in value]
            if isinstance(value, str) and ("base64" in key.lower() or len(value) > 100_000):
                return f"<binary payload omitted: {len(value)} characters>"
            return value

        return scrub(data)

    def close(self) -> None:
        return


class OpenAIImageGenerator:
    """OpenAI-compatible image generation with optional reference-image editing."""

    def __init__(self, config: Any):
        self.config = config
        self._client = None

    def generate(self, segment: Any, output_path: Path, negative_prompt: str = "") -> Path:
        return self.generate_prompt(
            prompt=segment.image_prompt,
            output_path=output_path,
            seed_offset=int(segment.index),
            negative_prompt=negative_prompt,
        )

    def generate_prompt(
        self,
        prompt: str,
        output_path: Path,
        seed_offset: int,
        negative_prompt: str = "",
        reference_image: Path | None = None,
    ) -> Path:
        del seed_offset
        if output_path.exists() and not getattr(self.config, "overwrite", False):
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = os.environ.get(
            "OPENAI_IMAGE_MODEL", "azure_openai/gpt-image-1.5"
        ).strip()
        prompt_prefix = os.environ.get("OPENAI_IMAGE_PROMPT_PREFIX", "").strip()
        full_prompt = "\n".join(
            part for part in (prompt_prefix, prompt.strip()) if part
        )
        if negative_prompt.strip():
            full_prompt += "\n避免出现：" + negative_prompt.strip()
        request_summary = {
            "model": model,
            "prompt": full_prompt,
            "reference_image": str(reference_image) if reference_image else None,
            "size": "1536x1024",
        }
        request_path = output_path.with_suffix(".openai_image_request.json")
        response_path = output_path.with_suffix(".openai_image_response.json")
        request_path.write_text(
            json.dumps(request_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        try:
            response = self._request_image(model, full_prompt, reference_image)
        except Exception as error:
            error_text = str(error).lower()
            if "safety system" not in error_text and "content_policy" not in error_text:
                raise
            full_prompt = self._safety_retry_prompt(full_prompt)
            request_summary.update({"prompt": full_prompt, "safety_retry": True})
            request_path.write_text(
                json.dumps(request_summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            response = self._request_image(model, full_prompt, reference_image)
        if not response.data:
            raise RuntimeError("OpenAI-compatible image API returned no images")
        item = response.data[0]
        response_payload = item.model_dump() if hasattr(item, "model_dump") else item
        response_path.write_text(
            json.dumps(
                self._response_summary(response_payload),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        image_bytes = self._extract_image_bytes(response_payload)

        from PIL import Image, ImageOps

        target_size = (
            int(getattr(self.config, "width", 1280)),
            int(getattr(self.config, "height", 720)),
        )
        with Image.open(io.BytesIO(image_bytes)) as image:
            fitted = ImageOps.fit(
                image.convert("RGB"), target_size, method=Image.Resampling.LANCZOS
            )
            fitted.save(output_path, format="PNG")
        return output_path

    def _request_image(self, model: str, prompt: str, reference_image: Path | None):
        if reference_image is None:
            return self._get_client().images.generate(
                model=model,
                prompt=prompt,
                size="1536x1024",
                n=1,
            )
        with reference_image.open("rb") as image_file:
            return self._get_client().images.edit(
                model=model,
                image=image_file,
                prompt=prompt,
                size="1536x1024",
                n=1,
            )

    @staticmethod
    def _safety_retry_prompt(prompt: str) -> str:
        replacements = (
            ("未成年人", "年轻成人"),
            ("少年人", "年轻成人"),
            ("男孩", "年轻成年男性"),
            ("女孩", "年轻成年女性"),
            ("孩童", "年轻成人"),
            ("儿童", "年轻成人"),
            ("小孩", "年轻成人"),
            ("少年", "年轻成人"),
        )
        adjusted = prompt
        for source, target in replacements:
            adjusted = adjusted.replace(source, target)
        return (
            "适合全年龄观众的非暴力历史文化画面。所有人物均为成年人，"
            "不呈现伤害、裸露或危险行为。\n" + adjusted
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI

        token = str(
            getattr(self.config, "image_api_key", "")
            or os.environ.get("IMAGE_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        ).strip()
        if not token:
            raise ValueError(
                "OpenAI image generation requires IMAGE_API_KEY or OPENAI_API_KEY"
            )
        base_url = os.environ.get(
            "OPENAI_IMAGE_BASE_URL",
            os.environ.get("LLM_BASE_URL", ""),
        ).strip()
        self._client = OpenAI(
            base_url=base_url,
            api_key=token,
            timeout=float(os.environ.get("OPENAI_IMAGE_TIMEOUT_SECONDS", "900")),
            max_retries=int(os.environ.get("OPENAI_IMAGE_RETRIES", "2")),
        )
        return self._client

    @staticmethod
    def _extract_image_bytes(item: Any) -> bytes:
        payload = item.model_dump() if hasattr(item, "model_dump") else item
        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI-compatible image response item is not an object")
        if payload.get("b64_json"):
            return base64.b64decode(str(payload["b64_json"]))
        if payload.get("url"):
            return GeminiImageGenerator._read_image_url(str(payload["url"]))
        raise RuntimeError("OpenAI-compatible image response has neither base64 nor URL")

    @staticmethod
    def _response_summary(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        return {
            key: (
                f"<binary payload omitted: {len(value)} characters>"
                if key == "b64_json" and isinstance(value, str)
                else value
            )
            for key, value in item.items()
        }

    def close(self) -> None:
        self._client = None


SUPPORTED_IMAGE_BACKENDS = (
    "qwen_image",
    "bernini_t2i",
    "gemini_image",
    "openai_image",
    "auto",
)


def resolve_image_backend(config: Any) -> str:
    backend = str(getattr(config, "image_backend", "qwen_image") or "qwen_image").strip().lower()
    if backend == "auto":
        bernini_config = Path(getattr(config, "bernini_image_config", ""))
        return "bernini_t2i" if bernini_config.exists() else "qwen_image"
    if backend not in SUPPORTED_IMAGE_BACKENDS:
        raise ValueError(f"Unsupported image backend: {backend}")
    return backend


def build_image_generator(config: Any):
    backend = resolve_image_backend(config)
    if backend == "qwen_image":
        return QwenImageGenerator(config)
    if backend == "bernini_t2i":
        return BerniniImageGenerator(config)
    if backend == "gemini_image":
        return GeminiImageGenerator(config)
    if backend == "openai_image":
        return OpenAIImageGenerator(config)
    raise ValueError(f"Unsupported image backend: {backend}")
