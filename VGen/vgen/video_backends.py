from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


DEFAULT_NEGATIVE_VIDEO_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，"
    "现代服装，现代建筑，现代交通工具，汽车，电线，路灯，玻璃幕墙，塑料制品，现代文字标识，"
    "清晰正脸肖像，现代写真感"
)
VEO_SUPPORTED_DURATION_SECONDS = (4, 6, 8)
MINIMAX_H3_SUPPORTED_DURATION_SECONDS = (4, 6, 8)


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    raise RuntimeError("ffmpeg is not available; install ffmpeg or imageio-ffmpeg.")


def _transcode_video(input_path: Path, output_path: Path, *, width: int, height: int, fps: int) -> None:
    ffmpeg = _ffmpeg_executable()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p",
        "-r", str(int(fps)),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _adjust_num_frames(target_frames: int) -> int:
    target_frames = max(1, int(target_frames))
    return max(1, round((target_frames - 1) / 4) * 4 + 1)


def _native_num_frames(duration: int, fps: int, max_frames: int | None = None) -> int:
    target_frames = max(1, int(duration) * int(fps))
    if max_frames is not None and int(max_frames) > 0:
        target_frames = min(target_frames, int(max_frames))
    return _adjust_num_frames(target_frames)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _ensure_python_executable(path_value: Any | None) -> Path:
    if path_value is None:
        return Path(sys.executable)
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Python runtime does not exist: {path}")
    return path


class BerniniVideoGenerator:
    def __init__(self, config: Any):
        self.config = config

    def generate(self, segment: Any, image_path: Path, output_path: Path, reference_images: list[Path] | None = None) -> Path:
        if output_path.exists() and not self.config.overwrite:
            return output_path
        self._validate_paths()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        key = f"segment_{segment.index:02d}"
        work_dir = output_path.parent.parent / "bernini" / key
        work_dir.mkdir(parents=True, exist_ok=True)
        raw_output_path = work_dir / f"{key}_raw.mp4"
        case_path = work_dir / "case.json"
        request_path = output_path.with_suffix(".bernini_request.json")
        references = self._reference_images(image_path, reference_images)

        case_payload = {
            "task_type": "r2v",
            "guidance_mode": "vae_txt_vit_wapg",
            "prompt": str(segment.video_prompt),
            "images": [str(path.resolve()) for path in references],
            "output": str(raw_output_path),
        }
        case_path.write_text(json.dumps(case_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        nproc_per_node, ulysses = self._distributed_shape()
        bernini_height = int(os.environ.get("BERNINI_HEIGHT", str(int(self.config.height))))
        bernini_width = int(os.environ.get("BERNINI_WIDTH", str(int(self.config.width))))
        bernini_steps = int(os.environ.get("BERNINI_VIDEO_STEPS", str(int(self.config.video_steps))))
        bernini_max_image_size = int(
            os.environ.get(
                "BERNINI_MAX_IMAGE_SIZE",
                str(max(842, bernini_height, bernini_width)),
            )
        )

        request_payload = {
            "backend": "bernini",
            "segment_index": int(segment.index),
            "prompt": str(segment.video_prompt),
            "case_path": str(case_path),
            "images": case_payload["images"],
            "output": str(output_path),
            "raw_output": str(raw_output_path),
            "distributed": {
                "nproc_per_node": nproc_per_node,
                "ulysses": ulysses,
            },
            "runtime": {
                "height": bernini_height,
                "width": bernini_width,
                "fps": int(self.config.fps),
                "num_frames": self._num_frames_for_duration(segment.duration),
                "num_inference_steps": bernini_steps,
                "max_image_size": bernini_max_image_size,
            },
        }
        request_path.write_text(json.dumps(request_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        command = [
            str(self._python_executable()),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(nproc_per_node),
            "infer_multi_gpu.py",
            "--config",
            str(self.config.bernini_config),
            "--ulysses",
            str(ulysses),
            "--case",
            str(case_path),
            "--task_type",
            "r2v",
            "--guidance_mode",
            "vae_txt_vit_wapg",
            "--max_image_size",
            str(bernini_max_image_size),
            "--height",
            str(bernini_height),
            "--width",
            str(bernini_width),
            "--num_frames",
            str(self._num_frames_for_duration(segment.duration)),
            "--num_inference_steps",
            str(bernini_steps),
            "--flow_shift",
            "5.0",
            "--seed",
            str(int(self.config.seed) + int(segment.index)),
            "--fps",
            str(int(self.config.fps)),
            "--omega_txt",
            "4",
            "--omega_tgt",
            "1.5",
            "--omega_img",
            "4.5",
            "--omega_vid",
            "1.25",
            "--omega_scale",
            "0.8",
            "--vit_denoising_step",
            "5",
            "--vit_txt_cfg",
            "1.2",
            "--vit_img_cfg",
            "1.0",
        ]
        if bool(getattr(self.config, "bernini_use_prompt_enhancer", False)):
            command.append("--use_pe")
        self._run(command)

        if not raw_output_path.exists():
            raise RuntimeError(f"Bernini inference did not produce video: {raw_output_path}")
        _transcode_video(
            raw_output_path,
            output_path,
            width=int(self.config.width),
            height=int(self.config.height),
            fps=int(self.config.fps),
        )
        return output_path

    def _reference_images(self, image_path: Path, reference_images: list[Path] | None) -> list[Path]:
        paths = [image_path]
        for ref_path in reference_images or []:
            if ref_path.exists() and ref_path != image_path:
                paths.append(ref_path)
        return _dedupe_paths(paths)

    def _run(self, command: list[str]) -> None:
        env = os.environ.copy()
        bernini_root = self.config.bernini_root
        env["PYTHONPATH"] = os.pathsep.join([str(bernini_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env.setdefault("NCCL_NET_PLUGIN", "none")
        env.setdefault("NCCL_DEBUG", "WARN")
        env.setdefault("FSDP", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        if bool(getattr(self.config, "bernini_use_prompt_enhancer", False)):
            env.setdefault("BERNINI_PE_BASE_URL", str(self.config.qwen35_api_base))
            env.setdefault("BERNINI_PE_API_KEY", str(self.config.qwen35_api_key))
        subprocess.run(command, cwd=str(bernini_root), env=env, check=True)

    def _validate_paths(self) -> None:
        required_paths = [
            self.config.bernini_root,
            self.config.bernini_root / "infer_multi_gpu.py",
            self.config.bernini_config,
            self._python_executable(),
        ]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            formatted = "\n".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Bernini runtime is incomplete. Missing paths:\n{formatted}")

    def _python_executable(self) -> Path:
        return _ensure_python_executable(self.config.bernini_python)

    def _distributed_shape(self) -> tuple[int, int]:
        visible_devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        default_nproc = len(visible_devices) if visible_devices else 8
        nproc_per_node = int(os.environ.get("BERNINI_NPROC_PER_NODE", str(default_nproc)))
        if nproc_per_node < 1:
            raise ValueError("BERNINI_NPROC_PER_NODE must be >= 1")
        if visible_devices and nproc_per_node > len(visible_devices):
            raise ValueError(
                f"BERNINI_NPROC_PER_NODE={nproc_per_node} exceeds visible CUDA devices ({len(visible_devices)}): {visible_devices}"
            )
        ulysses = int(os.environ.get("BERNINI_ULYSSES", str(nproc_per_node)))
        if ulysses < 1 or ulysses > nproc_per_node or nproc_per_node % ulysses != 0:
            raise ValueError(
                f"BERNINI_ULYSSES must be in [1, {nproc_per_node}] and divide BERNINI_NPROC_PER_NODE; got {ulysses}."
            )
        return nproc_per_node, ulysses

    def _num_frames_for_duration(self, duration: int) -> int:
        max_frames = int(getattr(self.config, "bernini_max_frames", 81))
        return _native_num_frames(duration, int(self.config.fps), max_frames=max_frames)


class BerniniRVideoGenerator:
    def __init__(self, config: Any):
        self.config = config

    def generate(self, segment: Any, image_path: Path, output_path: Path, reference_images: list[Path] | None = None) -> Path:
        if output_path.exists() and not self.config.overwrite:
            return output_path
        self._validate_paths()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        key = f"segment_{segment.index:02d}"
        work_dir = output_path.parent.parent / "bernini_r" / key
        work_dir.mkdir(parents=True, exist_ok=True)
        raw_output_path = work_dir / f"{key}_raw.mp4"
        case_path = work_dir / "case.json"
        case_payload = {
            "task_type": "r2v",
            "guidance_mode": "r2v_apg",
            "prompt": str(segment.video_prompt),
            "images": [str(path.resolve()) for path in self._reference_images(image_path, reference_images)],
            "output": str(raw_output_path),
        }
        case_path.write_text(json.dumps(case_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        bernini_height = int(os.environ.get("BERNINI_HEIGHT", str(int(self.config.height))))
        bernini_width = int(os.environ.get("BERNINI_WIDTH", str(int(self.config.width))))
        bernini_steps = int(os.environ.get("BERNINI_VIDEO_STEPS", str(int(self.config.video_steps))))
        command = [
            str(self._python_executable()),
            "infer_single_gpu.py",
            "--config", str(self.config.bernini_r_config),
            "--case", str(case_path),
            "--task_type", "r2v",
            "--guidance_mode", "r2v_apg",
            "--height", str(bernini_height),
            "--width", str(bernini_width),
            "--num_frames", str(self._num_frames_for_duration(segment.duration)),
            "--num_inference_steps", str(bernini_steps),
            "--seed", str(int(self.config.seed) + int(segment.index)),
            "--fps", str(int(self.config.fps)),
        ]
        if bool(getattr(self.config, "bernini_use_prompt_enhancer", False)):
            command.append("--use_pe")
        self._run(command)

        if not raw_output_path.exists():
            raise RuntimeError(f"Bernini-R inference did not produce video: {raw_output_path}")
        _transcode_video(
            raw_output_path,
            output_path,
            width=int(self.config.width),
            height=int(self.config.height),
            fps=int(self.config.fps),
        )
        return output_path

    def _reference_images(self, image_path: Path, reference_images: list[Path] | None) -> list[Path]:
        paths = [image_path]
        for ref_path in reference_images or []:
            if ref_path.exists() and ref_path != image_path:
                paths.append(ref_path)
        return _dedupe_paths(paths)

    def _run(self, command: list[str]) -> None:
        env = os.environ.copy()
        bernini_root = self.config.bernini_root
        env["PYTHONPATH"] = os.pathsep.join([str(bernini_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        if bool(getattr(self.config, "bernini_use_prompt_enhancer", False)):
            env.setdefault("BERNINI_PE_BASE_URL", str(self.config.qwen35_api_base))
            env.setdefault("BERNINI_PE_API_KEY", str(self.config.qwen35_api_key))
        subprocess.run(command, cwd=str(bernini_root), env=env, check=True)

    def _validate_paths(self) -> None:
        required_paths = [
            self.config.bernini_root,
            self.config.bernini_root / "infer_single_gpu.py",
            self.config.bernini_r_config,
            self._python_executable(),
        ]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            formatted = "\n".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Bernini-R runtime is incomplete. Missing paths:\n{formatted}")

    def _python_executable(self) -> Path:
        return _ensure_python_executable(self.config.bernini_python)

    def _num_frames_for_duration(self, duration: int) -> int:
        max_frames = int(getattr(self.config, "bernini_max_frames", 81))
        return _native_num_frames(duration, int(self.config.fps), max_frames=max_frames)


class HunyuanVideoGenerator:
    def __init__(self, config: Any):
        self.config = config

    def generate(self, segment: Any, image_path: Path, output_path: Path, reference_images: list[Path] | None = None) -> Path:
        del reference_images
        if output_path.exists() and not self.config.overwrite:
            return output_path
        self._validate_paths()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        key = f"segment_{segment.index:02d}"
        work_dir = output_path.parent.parent / "hunyuanvideo" / key
        work_dir.mkdir(parents=True, exist_ok=True)
        raw_output_path = work_dir / f"{key}_raw.mp4"

        command = [
            str(self._python_executable()),
            str(self.config.hunyuanvideo_root / "generate.py"),
            "--prompt", str(segment.video_prompt),
            "--negative_prompt", DEFAULT_NEGATIVE_VIDEO_PROMPT,
            "--resolution", self._resolution(),
            "--aspect_ratio", self._aspect_ratio(),
            "--model_path", str(self.config.hunyuanvideo_model),
            "--num_inference_steps", str(int(self.config.video_steps)),
            "--video_length", str(self._num_frames_for_duration(segment.duration)),
            "--sr", "false",
            "--rewrite", "false",
            "--offloading", "true",
            "--group_offloading", "true",
            "--overlap_group_offloading", "false",
            "--dtype", str(getattr(self.config, "hunyuanvideo_dtype", "bf16")),
            "--seed", str(int(self.config.seed) + int(segment.index)),
            "--output_path", str(raw_output_path),
            "--save_generation_config", "false",
            "--image_path", str(image_path) if image_path.exists() else "none",
        ]
        self._run(command)

        if not raw_output_path.exists():
            raise RuntimeError(f"HunyuanVideo inference did not produce video: {raw_output_path}")
        _transcode_video(
            raw_output_path,
            output_path,
            width=int(self.config.width),
            height=int(self.config.height),
            fps=int(self.config.fps),
        )
        return output_path

    def _run(self, command: list[str]) -> None:
        env = os.environ.copy()
        hunyuan_root = self.config.hunyuanvideo_root
        env["PYTHONPATH"] = os.pathsep.join([str(hunyuan_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        subprocess.run(command, cwd=str(hunyuan_root), env=env, check=True)

    def _validate_paths(self) -> None:
        required_paths = [
            self.config.hunyuanvideo_root,
            self.config.hunyuanvideo_root / "generate.py",
            self.config.hunyuanvideo_model,
            self._python_executable(),
        ]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            formatted = "\n".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"HunyuanVideo runtime is incomplete. Missing paths:\n{formatted}")

    def _python_executable(self) -> Path:
        return _ensure_python_executable(self.config.hunyuanvideo_python)

    def _num_frames_for_duration(self, duration: int) -> int:
        max_frames = int(getattr(self.config, "hunyuanvideo_max_frames", 121))
        return _native_num_frames(duration, int(self.config.fps), max_frames=max_frames)

    def _resolution(self) -> str:
        return "720p" if max(int(self.config.height), int(self.config.width)) > 640 else "480p"

    def _aspect_ratio(self) -> str:
        width = max(1, int(self.config.width))
        height = max(1, int(self.config.height))
        ratio = width / height
        if abs(ratio - 16 / 9) <= 0.2:
            return "16:9"
        if abs(ratio - 9 / 16) <= 0.2:
            return "9:16"
        if abs(ratio - 4 / 3) <= 0.15:
            return "4:3"
        if abs(ratio - 3 / 4) <= 0.15:
            return "3:4"
        if abs(ratio - 1.0) <= 0.05:
            return "1:1"
        return "16:9" if ratio >= 1.0 else "9:16"


class Veo31VideoGenerator:
    """Remote Veo 3.1 backend with optional character reference images."""

    def __init__(self, config: Any):
        self.config = config

    def generate(
        self,
        segment: Any,
        image_path: Path | None,
        output_path: Path,
        reference_images: list[Path] | None = None,
    ) -> Path:
        if output_path.exists() and not bool(getattr(self.config, "overwrite", False)):
            return output_path
        del image_path

        token = str(
            getattr(self.config, "veo_api_key", "")
            or os.environ.get("VEO_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        ).strip()
        if ":" not in token:
            raise ValueError("Veo 3.1 requires VEO_API_KEY or OPENAI_API_KEY in app_id:app_key format.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        request_path = output_path.with_suffix(".veo31_request.json")
        response_path = output_path.with_suffix(".veo31_response.json")
        raw_path = output_path.with_suffix(".veo31_raw.mp4")
        prompt = self._build_prompt(segment)
        payload = self._build_payload(segment, prompt, reference_images)
        request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        data = self._request(payload, token)
        response_path.write_text(
            json.dumps(self._response_summary(data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_video(data, raw_path)
        _transcode_video(
            raw_path,
            output_path,
            width=int(getattr(self.config, "width", 1280)),
            height=int(getattr(self.config, "height", 720)),
            fps=int(getattr(self.config, "fps", 24)),
        )
        return output_path

    def _build_payload(
        self,
        segment: Any,
        prompt: str | None = None,
        reference_images: list[Path] | None = None,
    ) -> dict[str, Any]:
        urls = [str(value).strip() for value in getattr(segment, "reference_image_urls", []) if str(value).strip()]
        local_references = [path for path in reference_images or [] if path.exists()]
        if local_references and len(urls) != len(local_references):
            raise ValueError(
                "Every local Veo reference image needs a published URL; configure the reference-image server"
            )
        content = [{"type": "text", "value": prompt or self._build_prompt(segment)}]
        content.extend(
            {
                "type": "reference_image_url",
                "value": url,
                "reference_type": "asset",
            }
            for url in urls
        )
        return {
            "request_id": str(uuid.uuid4()),
            "model_marker": self._setting(
                "veo_model_marker",
                "VEO_MODEL_MARKER",
                "api_google_veo-3.1-generate-001",
            ),
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "params": {
                "durationSeconds": self._request_duration_seconds(segment),
                "generateAudio": self._env_bool("VEO_GENERATE_AUDIO", False),
                "personGeneration": self._setting(
                    "veo_person_generation",
                    "VEO_PERSON_GENERATION",
                    "allow_adult",
                ),
            },
            "timeout": 3600,
        }

    def _request_duration_seconds(self, segment: Any) -> int:
        maximum = int(self._setting("veo_duration_seconds", "VEO_DURATION_SECONDS", "8"))
        allowed = [value for value in VEO_SUPPORTED_DURATION_SECONDS if value <= maximum]
        if not allowed:
            raise ValueError(f"Veo maximum duration must allow one of {VEO_SUPPORTED_DURATION_SECONDS}")
        if getattr(segment, "reference_image_urls", None):
            if 8 not in allowed:
                raise ValueError("Veo reference-to-video requires an 8-second request")
            return 8
        requested = max(1, int(math.ceil(float(getattr(segment, "duration", maximum) or maximum))))
        return next((value for value in allowed if value >= requested), allowed[-1])

    def _request(self, payload: dict[str, Any], token: str) -> dict[str, Any]:
        host = self._setting(
            "veo_host",
            "VEO_HOST",
            "",
        ).rstrip("/")
        api_version = self._setting("veo_api_version", "VEO_API_VERSION", "v2.03")
        source = self._setting("veo_source", "VEO_SOURCE", "storytelling-director-graph")
        timeout_seconds = int(self._setting("veo_timeout_seconds", "VEO_TIMEOUT_SECONDS", "900"))
        retries = max(0, int(self._setting("veo_retries", "VEO_RETRIES", "2")))
        user, api_key = token.split(":", 1)
        url = host + "/api/v1/data_eval"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            now = dt.datetime.now(dt.UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
            sign_input = f"date: {now}\nsource: {source}"
            signature = base64.b64encode(
                hmac.new(api_key.encode(), sign_input.encode(), hashlib.sha1).digest()
            ).decode()
            authorization = (
                f'hmac id="{user}", algorithm="hmac-sha1", '
                f'headers="date source", signature="{signature}"'
            )
            headers = {
                "Apiversion": api_version,
                "Authorization": authorization,
                "Date": now,
                "Source": source,
                "Content-Type": "application/json",
            }
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    status = int(response.status)
                    text = response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                text = exc.read().decode("utf-8", errors="replace")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt >= retries:
                    raise RuntimeError(f"Veo request failed after {attempt + 1} attempts: {exc}") from exc
                time.sleep(min(30, 3 * (attempt + 1)))
                continue

            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                last_error = exc
                if attempt >= retries:
                    raise RuntimeError(f"Veo returned non-JSON status={status}: {text[:1000]}") from exc
                time.sleep(min(30, 3 * (attempt + 1)))
                continue

            if status == 200 and data.get("code") == 0:
                return data
            message = json.dumps(self._response_summary(data), ensure_ascii=False)[:2000]
            last_error = RuntimeError(f"Veo returned status={status}: {message}")
            if attempt >= retries or status not in {408, 409, 425, 429, 500, 502, 503, 504}:
                raise last_error
            time.sleep(min(30, 3 * (attempt + 1)))

        raise RuntimeError(f"Veo request failed: {last_error}")

    def _write_video(self, data: dict[str, Any], raw_path: Path) -> None:
        answer = data.get("answer") or []
        video_item = next((item for item in answer if item.get("type") == "video_base64"), None)
        if video_item and video_item.get("value"):
            raw_path.write_bytes(base64.b64decode(video_item["value"]))
            return
        url_item = next((item for item in answer if item.get("type") in {"video_url", "url"}), None)
        if url_item and url_item.get("value"):
            with urllib.request.urlopen(str(url_item["value"]), timeout=900) as response:
                raw_path.write_bytes(response.read())
            return
        raise RuntimeError("Veo response contains neither video_base64 nor video_url.")

    def _build_prompt(self, segment: Any) -> str:
        suffix = os.environ.get(
            "VEO_PROMPT_SUFFIX",
            (
                "16:9 cinematic historical realism. No subtitles, captions, logos, watermarks, modern objects, "
                "modern architecture, modern clothing, or readable text. Keep character identity and period details stable."
            ),
        ).strip()
        has_references = bool(getattr(segment, "reference_image_urls", None))
        labels = [
            str(value).strip()
            for value in getattr(segment, "reference_asset_labels", [])
            if has_references and str(value).strip()
        ]
        reference_instruction = "\n".join(
            f"参考图{i}只对应角色【{label}】；保持其脸型、发冠、服饰轮廓和年龄特征一致。"
            for i, label in enumerate(labels, start=1)
        )
        if len(labels) > 1:
            reference_instruction += (
                "\n不同参考角色的脸型、服装、站位和动作身份不得互换；"
                "严格按正文中点名的角色执行动作。"
            )
        return "\n\n".join(
            part
            for part in (str(segment.video_prompt).strip(), reference_instruction, suffix)
            if part
        )

    def _setting(self, attr_name: str, env_name: str, default: str) -> str:
        value = getattr(self.config, attr_name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
        return os.environ.get(env_name, default).strip()

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

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


class Seedance20VideoGenerator:
    """Remote Seedance 2.0 backend using the custom HMAC data_eval protocol."""

    def __init__(self, config: Any):
        self.config = config

    def generate(
        self,
        segment: Any,
        image_path: Path | None,
        output_path: Path,
        reference_images: list[Path] | None = None,
    ) -> Path:
        if output_path.exists() and not bool(getattr(self.config, "overwrite", False)):
            return output_path
        del image_path

        token = self._api_token()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request_path = output_path.with_suffix(".seedance20_request.json")
        response_path = output_path.with_suffix(".seedance20_response.json")
        raw_path = output_path.with_suffix(".seedance20_raw.mp4")
        payload = self._build_payload(segment, reference_images=reference_images)
        request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        data = self._request(payload, token, response_path=response_path)
        response_path.write_text(
            json.dumps(self._response_summary(data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_usage(payload, data, output_path)
        self._write_video(data, raw_path)
        _transcode_video(
            raw_path,
            output_path,
            width=int(getattr(self.config, "width", 1280)),
            height=int(getattr(self.config, "height", 720)),
            fps=int(getattr(self.config, "fps", 24)),
        )
        return output_path

    @staticmethod
    def _write_usage(payload: dict[str, Any], data: dict[str, Any], output_path: Path) -> None:
        text = "\n".join(
            str(item.get("value") or "")
            for message in payload.get("messages", [])
            for item in message.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        )
        try:
            import tiktoken

            estimated_tokens = len(tiktoken.get_encoding("cl100k_base").encode(text))
            estimation_method = "tiktoken:cl100k_base"
        except Exception:
            estimated_tokens = len(text)
            estimation_method = "unicode_character_fallback"

        api_usage = Seedance20VideoGenerator._find_usage(data)
        reference_bytes = sum(
            max(0, (len(str(item.get("value") or "")) - 22) * 3 // 4)
            for message in payload.get("messages", [])
            for item in message.get("content", [])
            if isinstance(item, dict)
            and item.get("type") == "image_url"
            and str(item.get("value") or "").startswith("data:image/")
        )
        report = {
            "backend": "seedance20",
            "usage_source": "api" if api_usage else "estimated_text_only",
            "api_usage": api_usage,
            "estimated_prompt_tokens": estimated_tokens,
            "estimation_method": estimation_method,
            "prompt_text_characters": len(text),
            "reference_image_count": sum(
                1
                for message in payload.get("messages", [])
                for item in message.get("content", [])
                if isinstance(item, dict) and item.get("type") == "image_url"
            ),
            "reference_image_bytes_estimated": reference_bytes,
            "model_marker": payload.get("model_marker", ""),
            "duration_seconds": payload.get("params", {}).get("duration"),
        }
        output_path.with_suffix(".seedance20_usage.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _find_usage(data: Any) -> dict[str, Any]:
        if isinstance(data, dict):
            for key in ("usage", "token_usage", "cost_info", "billing"):
                value = data.get(key)
                if isinstance(value, dict):
                    return value
            for value in data.values():
                found = Seedance20VideoGenerator._find_usage(value)
                if found:
                    return found
        elif isinstance(data, list):
            for value in data:
                found = Seedance20VideoGenerator._find_usage(value)
                if found:
                    return found
        return {}

    def _build_payload(
        self,
        segment: Any,
        prompt: str | None = None,
        reference_images: list[Path] | None = None,
    ) -> dict[str, Any]:
        urls = [
            str(value).strip()
            for value in getattr(segment, "reference_image_urls", [])
            if str(value).strip()
        ]
        local_references = [path for path in reference_images or [] if path.exists()]
        reference_mode = os.environ.get("SEEDANCE_REFERENCE_MODE", "url").strip().lower()
        if reference_mode not in {"url", "data_uri"}:
            raise ValueError("SEEDANCE_REFERENCE_MODE must be url or data_uri")
        if reference_mode == "data_uri" and local_references:
            reference_values = [
                "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
                for path in local_references
            ]
        else:
            reference_values = urls
        if reference_mode == "url" and local_references and len(urls) != len(local_references):
            raise ValueError(
                "Every local Seedance reference image needs a published URL; "
                "configure the reference-image server"
            )
        content = [{"type": "image_url", "value": value} for value in reference_values]
        content.append({"type": "text", "value": prompt or self._build_prompt(segment)})
        timeout_seconds = int(
            self._setting("seedance_timeout_seconds", "SEEDANCE_TIMEOUT_SECONDS", "900")
        )
        params: dict[str, Any] = {
            "duration": str(self._request_duration_seconds(segment)),
            "ratio": self._aspect_ratio(),
            "resolution": self._setting(
                "seedance_resolution", "SEEDANCE_RESOLUTION", "720p"
            ),
            "framespersecond": str(int(getattr(self.config, "fps", 24))),
            "camerafixed": self._env_bool_text("SEEDANCE_CAMERA_FIXED", False),
            "watermark": self._env_bool_text("SEEDANCE_WATERMARK", False),
            "generate_audio": self._env_bool("SEEDANCE_GENERATE_AUDIO", False),
        }
        seed = os.environ.get("SEEDANCE_SEED", "").strip()
        if seed:
            params["seed"] = seed
        return {
            "request_id": str(uuid.uuid4()),
            "model_marker": self._setting(
                "seedance_model_marker",
                "SEEDANCE_MODEL_MARKER",
                "api_doubao_doubao-seedance-2-0-260128",
            ),
            "messages": [{"role": "user", "content": content}],
            "params": params,
            "timeout": timeout_seconds,
        }

    def _request_duration_seconds(self, segment: Any) -> int:
        maximum = int(
            self._setting("seedance_duration_seconds", "SEEDANCE_DURATION_SECONDS", "8")
        )
        requested = max(1, int(math.ceil(float(getattr(segment, "duration", maximum) or maximum))))
        return min(requested, maximum)

    def _request(
        self,
        payload: dict[str, Any],
        token: str,
        *,
        response_path: Path | None = None,
    ) -> dict[str, Any]:
        host = self._setting(
            "seedance_host",
            "SEEDANCE_HOST",
            "",
        ).rstrip("/")
        source = self._setting(
            "seedance_source", "SEEDANCE_SOURCE", "storytelling-director-graph-seedance20"
        )
        timeout_seconds = int(
            self._setting("seedance_timeout_seconds", "SEEDANCE_TIMEOUT_SECONDS", "900")
        )
        retries = max(
            0, int(self._setting("seedance_retries", "SEEDANCE_RETRIES", "2"))
        )
        url = host + "/api/v1/data_eval"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            now = dt.datetime.now(dt.UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
            headers = self._auth_headers(token, source, now)
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds + 120) as response:
                    status = int(response.status)
                    text = response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                text = exc.read().decode("utf-8", errors="replace")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt >= retries:
                    raise RuntimeError(
                        f"Seedance request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                time.sleep(min(30, 3 * (attempt + 1)))
                continue

            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                last_error = exc
                if response_path is not None:
                    response_path.write_text(
                        json.dumps(
                            {
                                "http_status": status,
                                "error": "non_json_response",
                                "body_preview": text[:4000],
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                if attempt >= retries:
                    raise RuntimeError(
                        f"Seedance returned non-JSON status={status}: {text[:1000]}"
                    ) from exc
                time.sleep(min(30, 3 * (attempt + 1)))
                continue

            if response_path is not None:
                response_path.write_text(
                    json.dumps(self._response_summary(data), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            if status == 200 and data.get("code") == 0:
                return data
            message = json.dumps(self._response_summary(data), ensure_ascii=False)[:2000]
            last_error = RuntimeError(f"Seedance returned status={status}: {message}")
            bad_request_detail = str(data.get("msg") or "")
            upstream_bad_request = data.get("code") == 1001 and any(
                marker in bad_request_detail
                for marker in (
                    "BadRequest",
                    "InvalidParameter",
                    "SensitiveContent",
                    "PrivacyInformation",
                )
            )
            if (
                attempt >= retries
                or upstream_bad_request
                or status not in {408, 409, 425, 429, 500, 502, 503, 504}
            ):
                raise last_error
            time.sleep(min(30, 3 * (attempt + 1)))

        raise RuntimeError(f"Seedance request failed: {last_error}")

    def _write_video(self, data: dict[str, Any], raw_path: Path) -> None:
        answer = data.get("answer") or []
        url_item = next(
            (item for item in answer if item.get("type") in {"video_url", "url"}), None
        )
        if not url_item or not url_item.get("value"):
            raise RuntimeError("Seedance response contains no video_url.")
        url = str(url_item["value"])
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=900) as response:
                    raw_path.write_bytes(response.read())
                return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"Failed to download Seedance video_url: {last_error}")

    def _build_prompt(self, segment: Any) -> str:
        suffix = os.environ.get(
            "SEEDANCE_PROMPT_SUFFIX",
            (
                "16:9 cinematic historical realism. No subtitles, captions, logos, watermarks, modern objects, "
                "modern architecture, modern clothing, or readable text. Keep character identity and period details stable."
            ),
        ).strip()
        has_references = bool(getattr(segment, "reference_image_urls", None))
        labels = [
            str(value).strip()
            for value in getattr(segment, "reference_asset_labels", [])
            if has_references and str(value).strip()
        ]
        reference_instruction = "\n".join(
            f"{label}的人物形象如图，请保持其脸型、发冠、服饰轮廓和年龄特征一致。"
            for label in labels
        )
        return "\n\n".join(
            part
            for part in (str(segment.video_prompt).strip(), reference_instruction, suffix)
            if part
        )

    def _api_token(self) -> str:
        token = str(
            getattr(self.config, "seedance_api_key", "")
            or os.environ.get("SEEDANCE_API_KEY", "")
        ).strip()
        if not token:
            app_id = os.environ.get("DOUBAO_TTS_USER", "").strip()
            app_key = os.environ.get("DOUBAO_TTS_KEY", "").strip()
            if app_id and app_key:
                token = f"{app_id}:{app_key}"
        if ":" not in token:
            raise ValueError(
                "Seedance 2.0 requires SEEDANCE_API_KEY in app_id:app_key format "
                "or both DOUBAO_TTS_USER and DOUBAO_TTS_KEY."
            )
        return token

    def _aspect_ratio(self) -> str:
        width = int(getattr(self.config, "width", 1280))
        height = int(getattr(self.config, "height", 720))
        ratio = width / max(1, height)
        return "16:9" if ratio >= 1.0 else "9:16"

    def _setting(self, attr_name: str, env_name: str, default: str) -> str:
        value = getattr(self.config, attr_name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
        return os.environ.get(env_name, default).strip()

    @staticmethod
    def _auth_headers(token: str, source: str, date_string: str) -> dict[str, str]:
        app_id, app_key = token.split(":", 1)
        sign_input = f"date: {date_string}\nsource: {source}"
        signature = base64.b64encode(
            hmac.new(app_key.encode(), sign_input.encode(), hashlib.sha1).digest()
        ).decode()
        return {
            "Authorization": (
                f'hmac id="{app_id}", algorithm="hmac-sha1", '
                f'headers="date source", signature="{signature}"'
            ),
            "Date": date_string,
            "Source": source,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _env_bool_text(cls, name: str, default: bool) -> str:
        return "true" if cls._env_bool(name, default) else "false"

    @staticmethod
    def _response_summary(data: dict[str, Any]) -> dict[str, Any]:
        def scrub(value: Any, key: str = "") -> Any:
            if isinstance(value, dict):
                return {
                    item_key: scrub(item_value, str(item_key))
                    for item_key, item_value in value.items()
                }
            if isinstance(value, list):
                return [scrub(item, key) for item in value]
            if isinstance(value, str) and ("base64" in key.lower() or len(value) > 100_000):
                return f"<binary payload omitted: {len(value)} characters>"
            return value

        return scrub(data)


class MiniMaxH3VideoGenerator:
    """Remote MiniMax-H3 image-to-video backend.

    MiniMax-H3 uses a different asynchronous API from the existing data-eval
    backends: creation returns a task id, and the video URL is obtained by
    polling ``/v2/query/video_generation/{task_id}``.
    """

    def __init__(self, config: Any):
        self.config = config

    def generate(
        self,
        segment: Any,
        image_path: Path | None,
        output_path: Path,
        reference_images: list[Path] | None = None,
    ) -> Path:
        if output_path.exists() and not bool(getattr(self.config, "overwrite", False)):
            return output_path
        del image_path

        token = self._api_token()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request_path = output_path.with_suffix(".minimax_h3_request.json")
        response_path = output_path.with_suffix(".minimax_h3_response.json")
        raw_path = output_path.with_suffix(".minimax_h3_raw.mp4")
        payload = self._build_payload(segment, reference_images=reference_images)
        request_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        try:
            data = self._request(payload, token)
        except RuntimeError as exc:
            if self._is_sensitive_rejection(exc):
                # Content-moderation rejected this shot's prompt. Honor the
                # "let it pass" rule: render a static clip from the reference
                # image(s) so the assembled director graph stays complete.
                print(
                    f"[minimax-h3] shot {getattr(segment, 'title', '?')} rejected"
                    f" for sensitive text ({exc}); emitting static reference clip.",
                    flush=True,
                )
                refs = [p for p in (reference_images or []) if p and p.exists()]
                if not refs:
                    first = (payload.get("content") or [{}])
                    for item in first:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            refs.append(Path("/dev/null"))
                            break
                if not (reference_images or []):
                    reference_images = refs
                self._make_static_clip(reference_images, raw_path)
                response_path.write_text(
                    json.dumps(
                        {
                            "passthrough": True,
                            "reason": "minimax_1026_sensitive_text",
                            "segment": str(getattr(segment, "title", "")),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                _transcode_video(
                    raw_path,
                    output_path,
                    width=int(getattr(self.config, "width", 1280)),
                    height=int(getattr(self.config, "height", 720)),
                    fps=int(getattr(self.config, "fps", 24)),
                )
                return output_path
            raise

        response_path.write_text(
            json.dumps(self._response_summary(data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_video(data, raw_path)
        _transcode_video(
            raw_path,
            output_path,
            width=int(getattr(self.config, "width", 1280)),
            height=int(getattr(self.config, "height", 720)),
            fps=int(getattr(self.config, "fps", 24)),
        )
        return output_path

    def _build_payload(
        self,
        segment: Any,
        *,
        reference_images: list[Path] | None = None,
    ) -> dict[str, Any]:
        urls = [
            str(value).strip()
            for value in getattr(segment, "reference_image_urls", [])
            if str(value).strip()
        ]
        local_references = [path for path in reference_images or [] if path.exists()]
        reference_mode = os.environ.get("MINIMAX_REFERENCE_MODE", "url").strip().lower()
        if reference_mode not in {"url", "data_uri"}:
            raise ValueError("MINIMAX_REFERENCE_MODE must be url or data_uri")

        if reference_mode == "data_uri" and local_references:
            reference_values = [
                "data:image/png;base64,"
                + base64.b64encode(path.read_bytes()).decode("ascii")
                for path in local_references
            ]
        else:
            reference_values = urls
        if reference_mode == "url" and local_references and len(urls) != len(local_references):
            raise ValueError(
                "Every local MiniMax-H3 reference image needs a published URL; "
                "configure the reference-image server or use MINIMAX_REFERENCE_MODE=data_uri"
            )

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": self._build_prompt(segment, has_references=bool(reference_values)),
            }
        ]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": value},
                "role": "reference_image",
            }
            for value in reference_values
        )
        return {
            "model": self._setting("minimax_model", "MINIMAX_MODEL", "MiniMax-H3"),
            "content": content,
            "resolution": self._setting(
                "minimax_resolution", "MINIMAX_RESOLUTION", "768P"
            ),
            "duration": self._request_duration_seconds(segment),
            "ratio": self._setting("minimax_ratio", "MINIMAX_RATIO", "adaptive"),
        }

    def _request_duration_seconds(self, segment: Any) -> int:
        maximum = int(
            self._setting("minimax_duration_seconds", "MINIMAX_DURATION_SECONDS", "4")
        )
        allowed = [
            value
            for value in MINIMAX_H3_SUPPORTED_DURATION_SECONDS
            if value <= maximum
        ]
        if not allowed:
            raise ValueError(
                "MiniMax-H3 maximum duration must allow one of "
                f"{MINIMAX_H3_SUPPORTED_DURATION_SECONDS}"
            )
        requested = max(
            1,
            int(math.ceil(float(getattr(segment, "duration", maximum) or maximum))),
        )
        return next((value for value in allowed if value >= requested), allowed[-1])

    def _request(self, payload: dict[str, Any], token: str) -> dict[str, Any]:
        host = self._setting(
            "minimax_host",
            "MINIMAX_HOST",
            "",
        ).rstrip("/")
        model = str(payload.get("model") or "MiniMax-H3")
        timeout_seconds = int(
            self._setting("minimax_timeout_seconds", "MINIMAX_TIMEOUT_SECONDS", "1800")
        )
        poll_seconds = max(
            1.0,
            float(
                self._setting(
                    "minimax_poll_seconds", "MINIMAX_POLL_SECONDS", "10"
                )
            ),
        )
        retries = max(
            0,
            int(self._setting("minimax_retries", "MINIMAX_RETRIES", "8")),
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        cache_task_id = hashlib.md5(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        query = urllib.parse.urlencode(
            {
                "provider": "minimax",
                "model": model,
                "timeout": 180,
                "cache_task_id": cache_task_id,
            }
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}?{query}",
        }
        create_url = host + "/v2/video_generation"
        create_data = self._request_json(
            create_url,
            body=body,
            headers=headers,
            timeout=max(190, timeout_seconds),
            retries=retries,
            label="MiniMax-H3 create",
        )
        task_id = str(
            create_data.get("task_id")
            or (create_data.get("task") or {}).get("task_id")
            or ""
        ).strip()
        if not task_id:
            raise RuntimeError(
                "MiniMax-H3 creation response contains no task_id: "
                + json.dumps(self._response_summary(create_data), ensure_ascii=False)[:2000]
            )

        query_url = (
            host
            + "/v2/query/video_generation/"
            + urllib.parse.quote(task_id, safe="")
        )
        deadline = time.monotonic() + timeout_seconds
        last_task: dict[str, Any] = {}
        while time.monotonic() < deadline:
            time.sleep(poll_seconds)
            query_data = self._request_json(
                query_url,
                body=None,
                headers=headers,
                timeout=max(190, timeout_seconds),
                retries=retries,
                label="MiniMax-H3 query",
            )
            task = query_data.get("task")
            if not isinstance(task, dict):
                task = query_data
            last_task = task
            status = str(task.get("status") or "").strip().lower()
            if status in {"succeeded", "success", "completed", "complete"}:
                video_url = self._extract_video_url(task)
                if not video_url:
                    raise RuntimeError(
                        "MiniMax-H3 task succeeded but contains no video URL: "
                        + json.dumps(
                            self._response_summary(task), ensure_ascii=False
                        )[:2000]
                    )
                return {
                    "task_id": task_id,
                    "task": task,
                    "video_url": video_url,
                }
            if status in {"failed", "failure", "cancelled", "canceled"}:
                raise RuntimeError(
                    "MiniMax-H3 task failed: "
                    + json.dumps(self._response_summary(task), ensure_ascii=False)[:4000]
                )

        raise TimeoutError(
            f"MiniMax-H3 task {task_id} did not finish within {timeout_seconds}s: "
            + json.dumps(self._response_summary(last_task), ensure_ascii=False)[:2000]
        )

    @staticmethod
    def _request_json(
        url: str,
        *,
        body: bytes | None,
        headers: dict[str, str],
        timeout: int,
        retries: int,
        label: str,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            request = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method="POST" if body is not None else "GET",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    status = int(response.status)
                    text = response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                text = exc.read().decode("utf-8", errors="replace")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt >= retries:
                    raise RuntimeError(
                        f"{label} request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                time.sleep(min(30, 3 * (attempt + 1)))
                continue

            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                last_error = exc
                if attempt >= retries:
                    raise RuntimeError(
                        f"{label} returned non-JSON status={status}: {text[:1000]}"
                    ) from exc
                time.sleep(min(30, 3 * (attempt + 1)))
                continue

            if 200 <= status < 300:
                if isinstance(data, dict):
                    return data
                raise RuntimeError(f"{label} returned a non-object JSON response")

            last_error = RuntimeError(
                f"{label} returned status={status}: "
                f"{json.dumps(MiniMaxH3VideoGenerator._response_summary(data), ensure_ascii=False)[:2000]}"
            )
            if attempt >= retries or status not in {408, 409, 425, 429, 500, 502, 503, 504}:
                raise last_error
            if status == 429:
                time.sleep(min(120.0, 20.0 * (2 ** attempt)))
            else:
                time.sleep(min(30, 3 * (attempt + 1)))
        raise RuntimeError(f"{label} request failed: {last_error}")

    @classmethod
    def _extract_video_url(cls, task: dict[str, Any]) -> str:
        content = task.get("content")
        if isinstance(content, dict):
            for key in ("url", "video_url", "videoUrl"):
                value = content.get(key)
                if value:
                    return str(value)
        if isinstance(content, str) and content.startswith(("http://", "https://")):
            return content
        for key in ("url", "video_url", "videoUrl"):
            value = task.get(key)
            if value:
                return str(value)
        return ""

    def _write_video(self, data: dict[str, Any], raw_path: Path) -> None:
        url = str(data.get("video_url") or "").strip()
        if not url:
            raise RuntimeError("MiniMax-H3 response contains no video_url.")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=900) as response:
                    raw_path.write_bytes(response.read())
                return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"Failed to download MiniMax-H3 video_url: {last_error}")

    @staticmethod
    def _is_sensitive_rejection(exc: Exception) -> bool:
        msg = str(exc)
        if "input new_sensitive" in msg or "input text sensitive" in msg:
            return True
        return '"code": "1026"' in msg or 'code": 1026' in msg or '"code":1026' in msg

    def _make_static_clip(
        self, reference_images: list[Path] | None, raw_path: Path
    ) -> None:
        """Render a short static mp4 from the first reference image using ffmpeg.

        Used only when MiniMax rejects a shot for sensitive text so the overall
        director graph still completes.
        """
        ffmpeg = _ffmpeg_executable()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        src = None
        for p in (reference_images or []):
            if p and p.exists() and os.path.getsize(p) > 0:
                src = p
                break
        width = int(getattr(self.config, "width", 1280))
        height = int(getattr(self.config, "height", 720))
        fps = int(getattr(self.config, "fps", 24))
        seconds = max(4, int(getattr(self.config, "minimax_duration_seconds", 8) or 8))
        if src is None:
            # No reference image: synthesize a blank slate clip.
            command = [
                ffmpeg, "-y", "-f", "lavfi", "-i",
                f"color=c=black:s={width}x{height}:r={fps}:d={seconds}",
                "-pix_fmt", "yuv420p", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "18", str(raw_path),
            ]
        else:
            command = [
                ffmpeg, "-y", "-loop", "1", "-i", str(src),
                "-f", "lavfi", "-i", f"anullsrc=r={fps}:cl=stereo",
                "-t", str(seconds), "-r", str(fps),
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                       f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p",
                "-pix_fmt", "yuv420p", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "18", str(raw_path),
            ]
        subprocess.run(command, check=True)

    def _build_prompt(self, segment: Any, *, has_references: bool = False) -> str:
        suffix = os.environ.get(
            "MINIMAX_PROMPT_SUFFIX",
            (
                "16:9 cinematic historical realism. No subtitles, captions, logos, watermarks, "
                "modern objects, modern architecture, modern clothing, or readable text. "
                "Keep the referenced character identity, face, hairstyle, clothing silhouette, "
                "age, and period details stable."
            ),
        ).strip()
        labels = [
            str(value).strip()
            for value in getattr(segment, "reference_asset_labels", [])
            if has_references and str(value).strip()
        ]
        reference_instruction = "\n".join(
            f"{label}的人物形象如参考图，请保持其脸型、发冠、服饰轮廓和年龄特征一致。"
            for label in labels
        )
        if len(labels) > 1:
            reference_instruction += (
                "\n不同参考角色的脸型、服装、站位和动作身份不得互换；"
                "严格按正文中点名的角色执行动作。"
            )
        return "\n\n".join(
            part
            for part in (
                str(getattr(segment, "video_prompt", "")).strip(),
                reference_instruction,
                suffix,
            )
            if part
        )

    def _api_token(self) -> str:
        token = str(
            getattr(self.config, "minimax_api_key", "")
            or os.environ.get("MINIMAX_API_KEY", "")
        ).strip()
        if not token:
            app_id = os.environ.get("MINIMAX_APP_ID", "").strip()
            app_key = os.environ.get("MINIMAX_APP_KEY", "").strip()
            if app_id and app_key:
                token = f"{app_id}:{app_key}"
        if ":" not in token:
            raise ValueError(
                "MiniMax-H3 requires MINIMAX_API_KEY in app_id:app_key format "
                "or both MINIMAX_APP_ID and MINIMAX_APP_KEY."
            )
        return token

    def _setting(self, attr_name: str, env_name: str, default: str) -> str:
        value = getattr(self.config, attr_name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
        return os.environ.get(env_name, default).strip()

    @staticmethod
    def _response_summary(data: Any) -> Any:
        def scrub(value: Any, key: str = "") -> Any:
            if isinstance(value, dict):
                return {
                    item_key: scrub(item_value, str(item_key))
                    for item_key, item_value in value.items()
                }
            if isinstance(value, list):
                return [scrub(item, key) for item in value]
            if isinstance(value, str) and (
                "base64" in key.lower() or len(value) > 100_000
            ):
                return f"<binary payload omitted: {len(value)} characters>"
            return value

        return scrub(data)


# Backward-compatible import for existing scripts and tests.
Veo31TextVideoGenerator = Veo31VideoGenerator
