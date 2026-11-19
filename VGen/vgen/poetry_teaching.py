from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from collections.abc import Iterable
from pathlib import Path
from typing import Any


from vgen.workspace import (
    DEFAULT_BERNINI_CONFIG,
    DEFAULT_BERNINI_R_CONFIG,
    DEFAULT_BERNINI_ROOT,
    DEFAULT_BERNINI_PYTHON,
    DEFAULT_HUNYUANVIDEO_MODEL,
    DEFAULT_HUNYUANVIDEO_ROOT,
    DEFAULT_HUNYUANVIDEO_PYTHON,
    DEFAULT_CLIP_SECONDS,
    DEFAULT_COSYVOICE_MODEL,
    DEFAULT_COSYVOICE_PROMPT_TEXT,
    DEFAULT_COSYVOICE_PROMPT_WAV,
    DEFAULT_COSYVOICE_REPO,
    DEFAULT_MAX_SEGMENT_SECONDS,
    DEFAULT_MIN_SEGMENT_SECONDS,
    DEFAULT_MODELS_ROOT,
    DEFAULT_QWEN35_API_BASE,
    DEFAULT_QWEN35_MODEL,
    DEFAULT_QWEN_IMAGE_MODEL,
    DEFAULT_ROOT,
    DEFAULT_SEGMENTS,
    DEFAULT_TARGET_TOTAL_SECONDS,
    DEFAULT_TANGSHI300_PATH,
    DEFAULT_WAN_I2V_MODEL,
    DEFAULT_WAN_VACE_MODEL,
)
from vgen.image_backends import build_image_generator
from vgen.script_sync import apply_script_first_sync, derive_full_script, upsert_sync_suffix

ANCIENT_CHINESE_VISUAL_STYLE = (
    "中国古代写实历史教育短片风格；服饰、发式、建筑、器物、船只、灯具、道路、环境都必须符合中国古代语境；"
    "禁止现代服装、现代建筑、现代交通工具、电线、路灯、玻璃幕墙、塑料制品、现代文字标识"
)
VACE_REFERENCE_ASSET_VISUAL_RULE = (
    "VACE参考资产图，不是最终视频截图或PPT插图；必须先选择一个资产类型：人物资产、环境资产、物象/器物资产、人物+环境资产或物象+环境资产；"
    "资产选择优先级：旁白核心是人物处境、动作或心理时优先人物资产或人物+环境资产；核心是地点、季节、天气、空间和氛围时优先环境资产；"
    "核心是诗中物象、器物或声音来源时优先物象/器物资产或物象+环境资产；每段只锁定一个最重要的视觉锚点，"
    "构图稳定、轮廓明确、景深自然、留有少量环境边界，便于后续视频模型保持参考一致性；"
    "禁止文字、标题、诗句、字幕、分镜编号、拼贴、多宫格、海报排版、镜头运动文字说明"
)
SILHOUETTE_CHARACTER_VISUAL_RULE = (
    "如画面出现人物，人物必须为中国古代人物剪影、背影剪影或侧影剪影；不露正脸，无清晰五官，"
    "不得出现现代写真感或清晰肖像；没有人物需求时不要强行加人"
)
TEXT_CARD_NEGATIVE_IMAGE_PROMPT = (
    "任何额外文字，任何额外汉字，英文，拼音，数字，水印，logo，印章，红章，落款，边注，小字，页眉，标题栏，"
    "说明文字，教学视频，古诗教学，装饰性文字，错别字，缺字，多字，改字，改标点，乱码，模糊文字，变形文字，"
    "繁体字，现代字体，现代书本，打开的书，书册，书脊，封面，人物，现代物品，低质量，模糊，噪声"
)
NEGATIVE_IMAGE_PROMPT = (
    "模糊，低清晰度，低质量，主体不明确，噪声，文字，字幕，水印，变形，畸形，过曝，欠曝，"
    "现代服装，现代建筑，现代交通工具，汽车，电线，路灯，玻璃幕墙，塑料制品，现代文字标识，"
    "清晰正脸肖像，现代写真感"
)
NEGATIVE_VIDEO_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，"
    "现代服装，现代建筑，现代交通工具，汽车，电线，路灯，玻璃幕墙，塑料制品，现代文字标识，"
    "清晰正脸肖像，现代写真感"
)
INTRO_TITLE_SECONDS = 3
INTRO_POEM_SECONDS = 6

@dataclass(slots=True)
class PipelineConfig:
    vgen_root: Path = DEFAULT_ROOT
    models_root: Path = DEFAULT_MODELS_ROOT
    output_root: Path = DEFAULT_ROOT / "outputs" / "poetry_teaching"
    qwen_image_model: Path = DEFAULT_QWEN_IMAGE_MODEL
    image_backend: str = "auto"
    bernini_image_config: Path = DEFAULT_BERNINI_CONFIG
    video_backend: str = "wan"
    veo_api_key: str = ""
    veo_host: str = ""
    veo_api_version: str = "v2.03"
    veo_model_marker: str = "api_google_veo-3.1-generate-001"
    veo_duration_seconds: int = 8
    veo_timeout_seconds: int = 900
    veo_retries: int = 2
    seedance_api_key: str = ""
    seedance_host: str = ""
    seedance_model_marker: str = "api_doubao_doubao-seedance-2-0-260128"
    seedance_duration_seconds: int = 8
    seedance_resolution: str = "720p"
    seedance_timeout_seconds: int = 900
    seedance_retries: int = 2
    minimax_api_key: str = ""
    minimax_host: str = ""
    minimax_model: str = "MiniMax-H3"
    minimax_duration_seconds: int = 4
    minimax_resolution: str = "768P"
    minimax_ratio: str = "adaptive"
    minimax_timeout_seconds: int = 1800
    minimax_poll_seconds: float = 10.0
    minimax_retries: int = 2
    bernini_root: Path = DEFAULT_BERNINI_ROOT
    bernini_python: Path | None = DEFAULT_BERNINI_PYTHON
    bernini_config: Path = DEFAULT_BERNINI_CONFIG
    bernini_r_config: Path = DEFAULT_BERNINI_R_CONFIG
    bernini_max_frames: int = 81
    bernini_use_prompt_enhancer: bool = False
    hunyuanvideo_root: Path = DEFAULT_HUNYUANVIDEO_ROOT
    hunyuanvideo_python: Path | None = DEFAULT_HUNYUANVIDEO_PYTHON
    hunyuanvideo_model: Path = DEFAULT_HUNYUANVIDEO_MODEL
    hunyuanvideo_max_frames: int = 121
    hunyuanvideo_dtype: str = "bf16"
    wan_i2v_model: Path = DEFAULT_WAN_VACE_MODEL
    cosyvoice_model: Path = DEFAULT_COSYVOICE_MODEL
    cosyvoice_repo: Path = DEFAULT_COSYVOICE_REPO
    cosyvoice_python: Path | None = None
    tangshi300_path: Path | None = DEFAULT_TANGSHI300_PATH
    llm_backend: str = "qwen35"
    qwen35_api_base: str = DEFAULT_QWEN35_API_BASE
    qwen35_model: str = DEFAULT_QWEN35_MODEL
    openai_api_base: str = ""
    openai_api_key: str = ""
    openai_model: str = "doubao-seed-2.0-pro"
    openai_temperature: float = 0.4
    openai_top_p: float = 0.9
    openai_max_tokens: int = 8192
    qwen35_api_key: str = "qwen3.5-27b-local"
    target_total_seconds: int = DEFAULT_TARGET_TOTAL_SECONDS
    min_segment_seconds: int = DEFAULT_MIN_SEGMENT_SECONDS
    max_segment_seconds: int = DEFAULT_MAX_SEGMENT_SECONDS
    segments: int = DEFAULT_SEGMENTS
    clip_seconds: int = DEFAULT_CLIP_SECONDS
    fps: int = 15
    height: int = 480
    width: int = 832
    image_height: int = 720
    image_width: int = 1280
    image_steps: int = 30
    video_steps: int = 35
    image_device: str = "cuda:1"
    video_device: str = "cuda:2"
    video_usp_devices: str = ""
    video_usp_master_port: int = 29511
    seed: int = 0
    image_workers: int = 1
    video_workers: int = 1
    audio_workers: int = 1
    skip_image: bool = False
    skip_video: bool = False
    skip_audio: bool = False
    overwrite: bool = False
    prompt_wav: Path | None = DEFAULT_COSYVOICE_PROMPT_WAV
    prompt_text: str = DEFAULT_COSYVOICE_PROMPT_TEXT
    audio_backend: str = "local_cosyvoice"

    @property
    def num_frames(self) -> int:
        target_frames = max(1, self.clip_seconds * self.fps)
        return max(1, round((target_frames - 1) / 4) * 4 + 1)


@dataclass(slots=True)
class SegmentPlan:
    index: int
    title: str
    story_beat: str
    teaching_goal: str
    image_prompt: str
    video_prompt: str
    audio_text: str
    duration: int = 15
    image_prompts: list[str] = field(default_factory=list)
    script_beat: str = ""
    av_sync_plan: list[dict[str, Any]] = field(default_factory=list)
    asset_refs: list[str] = field(default_factory=list)
    reference_asset_labels: list[str] = field(default_factory=list)
    reference_image_urls: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PoetryPlan:
    poem_title: str
    author: str
    narrative_mode: str
    storyline: str
    recurring_character_style: str
    appreciation: str
    plan4teach: str
    plan4img: str
    plan4video: str
    plan4audio: str
    segments: list[SegmentPlan] = field(default_factory=list)
    raw_response: str = ""
    full_script: str = ""


@dataclass(slots=True)
class SegmentArtifacts:
    index: int
    image: Path | None = None
    images: list[Path] = field(default_factory=list)
    video: Path | None = None
    audio: Path | None = None
    muxed_video: Path | None = None

@dataclass(slots=True)
class IntroArtifacts:
    title_author_image: Path | None = None
    full_poem_image: Path | None = None
    title_author_video: Path | None = None
    full_poem_video: Path | None = None
    silent_video: Path | None = None
    audio: Path | None = None
    muxed_video: Path | None = None

@dataclass(slots=True)
class PipelineResult:
    output_dir: Path
    plan_path: Path
    artifacts: list[SegmentArtifacts]
    final_video: Path | None
    intro: IntroArtifacts | None = None


class Tangshi300ReferenceProvider:
    def __init__(self, data_path: Path | None):
        self.data_path = data_path
        self._items: list[dict[str, Any]] | None = None

    def build_reference(self, poem: str) -> str:
        item = self.find_poem(poem)
        if item is None:
            return "未检索到唐诗三百首参考资料，请仅依据用户输入的古诗生成，并避免编造具体背景。"
        return self.format_reference(item)

    def find_poem(self, poem: str) -> dict[str, Any] | None:
        items = self._load_items()
        if not items:
            return None
        title, author, content_lines = self._parse_poem_input(poem)
        if title:
            candidates = [item for item in items if title in {str(item.get("title", "")), str(item.get("detail_title", ""))}]
            if author:
                exact = [item for item in candidates if author in {str(item.get("author", "")), str(item.get("list_author", ""))}]
                if exact:
                    return exact[0]
            if candidates:
                return candidates[0]
        if content_lines:
            first_line = content_lines[0]
            for item in items:
                if first_line and first_line in str(item.get("content", "")):
                    return item
        normalized_poem = self._normalize_text(poem)
        for item in items:
            content = self._normalize_text(str(item.get("content", "")))
            if content and (content in normalized_poem or normalized_poem in content):
                return item
        return None

    def format_reference(self, item: dict[str, Any]) -> str:
        raw_sections = item.get("sections")
        sections = raw_sections if isinstance(raw_sections, list) else []
        section_map = {str(section.get("title", "")): str(section.get("text", "")) for section in sections if isinstance(section, dict)}
        translation = str(item.get("translation_annotation") or section_map.get("译文及注释") or "")
        appreciation = str(item.get("appreciation") or section_map.get("简析") or section_map.get("赏析") or "")
        background = str(item.get("background") or section_map.get("创作背景") or "")
        author_life = self._extract_author_life(item, section_map)
        content = str(item.get("content", ""))
        visual_reference = self._build_visual_reference(content, translation, appreciation)
        raw_tags = item.get("tags")
        tags = raw_tags if isinstance(raw_tags, list) else []
        return f"""
【唐诗三百首参考资料】
诗题：{item.get("detail_title") or item.get("title") or "未知"}
作者：{item.get("author") or item.get("list_author") or "未知"}
朝代：{item.get("dynasty") or "唐代"}
体裁/标签：{item.get("category") or ""}{"；" if tags else ""}{"；".join(map(str, tags))}
原文：
{content}

译文与注释要点：
{self._shorten(translation, 520)}

创作背景：
{self._shorten(background, 320)}

赏析要点：
{self._shorten(appreciation, 620)}

作者生平参考：
{self._shorten(author_life, 260)}

可视化参考：
{visual_reference}
""".strip()

    def _load_items(self) -> list[dict[str, Any]]:
        if self._items is not None:
            return self._items
        if self.data_path is None or not self.data_path.exists():
            self._items = []
            return self._items
        items: list[dict[str, Any]] = []
        for line in self.data_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                items.append(data)
        self._items = items
        return self._items

    def _parse_poem_input(self, poem: str) -> tuple[str, str, list[str]]:
        lines = [line.strip() for line in poem.splitlines() if line.strip()]
        title = lines[0] if lines else ""
        author = lines[1] if len(lines) > 1 and not re.search(r"[，。！？；、]", lines[1]) else ""
        content_lines = [line for line in lines[2 if author else 1:] if re.search(r"[，。！？；、]", line)]
        if not content_lines:
            content_lines = [line for line in lines if re.search(r"[，。！？；、]", line)]
        return title, author, content_lines

    def _extract_author_life(self, item: dict[str, Any], section_map: dict[str, str]) -> str:
        for key in ["author_intro", "author_life", "author_biography", "author_desc", "writer_intro"]:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for title, text in section_map.items():
            if any(marker in title for marker in ["作者简介", "作者生平", "生平", "作者"]):
                return text.strip()
        return "暂无明确作者生平资料；如需在旁白最后介绍作者，只能使用可靠常识做一句简短概括，不能编造具体年月、官职或轶事。"

    def _build_visual_reference(self, content: str, translation: str, appreciation: str) -> str:
        poem_phrases = self._extract_poem_phrases(content)
        translation_scenes = self._extract_labeled_sentences(translation, ["译文"], max_items=4)
        annotation_details = self._extract_annotation_details(translation, max_items=6)
        appreciation_scenes = self._extract_sentences_with_scene_markers(appreciation, max_items=4)
        lines = []
        if poem_phrases:
            lines.append("- 原诗画面短语：" + "；".join(poem_phrases))
        if translation_scenes:
            lines.append("- 译文画面短句：" + "；".join(translation_scenes))
        if annotation_details:
            lines.append("- 注释名物地点：" + "；".join(annotation_details))
        if appreciation_scenes:
            lines.append("- 赏析画面线索：" + "；".join(appreciation_scenes))
        return "\n".join(lines) if lines else "- 请从原诗、译文和赏析中提取每首诗自身可画出的地点、景物、人物、器物、光线与动作。"

    def _extract_poem_phrases(self, content: str) -> list[str]:
        phrases = []
        for phrase in re.split(r"[，。！？；、\n\r]+", content):
            phrase = phrase.strip()
            if 2 <= len(phrase) <= 14 and phrase not in phrases:
                phrases.append(phrase)
        return phrases[:10]

    def _extract_labeled_sentences(self, text: str, labels: list[str], max_items: int) -> list[str]:
        cleaned = self._remove_expand_marker(text)
        for label in labels:
            cleaned = cleaned.replace(label, "")
        return self._split_scene_sentences(cleaned, max_items=max_items)

    def _extract_annotation_details(self, text: str, max_items: int) -> list[str]:
        details = []
        for line in self._remove_expand_marker(text).splitlines():
            line = line.strip()
            if "：" not in line:
                continue
            key, value = line.split("：", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value or key in {"译文", "注释"}:
                continue
            detail = f"{key}：{self._shorten_inline(value, 42)}"
            if detail not in details:
                details.append(detail)
        return details[:max_items]

    def _extract_sentences_with_scene_markers(self, text: str, max_items: int) -> list[str]:
        sentences = self._split_scene_sentences(text, max_items=max_items * 2)
        scene_markers = ("景", "象", "画", "声", "色", "动", "静", "明", "暗", "空间", "意境", "情境", "场面")
        selected = [sentence for sentence in sentences if any(marker in sentence for marker in scene_markers)]
        return (selected or sentences)[:max_items]

    def _split_scene_sentences(self, text: str, max_items: int) -> list[str]:
        sentences = []
        for sentence in re.split(r"[。！？；\n\r]+", self._remove_expand_marker(text)):
            sentence = sentence.strip(" ，、\t")
            if 6 <= len(sentence) <= 90 and sentence not in sentences:
                sentences.append(self._shorten_inline(sentence, 64))
        return sentences[:max_items]

    @staticmethod
    def _remove_expand_marker(text: str) -> str:
        return text.replace("展开阅读全文", "").strip()

    @staticmethod
    def _shorten_inline(text: str, limit: int) -> str:
        text = re.sub(r"\s+", "", text.strip())
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "……"

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", "", text)

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        if len(text) <= limit:
            return text or "无"
        return text[:limit].rstrip() + "……"


class Qwen35Planner:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.reference_provider = Tangshi300ReferenceProvider(config.tangshi300_path)

    def build_prompt(self, poem: str) -> str:
        reference = self.reference_provider.build_reference(poem)
        return f"""
你是古诗教学短片总导演、语文教研员和多模态生成规划师。请把古诗创作背景和赏析要点转化为有故事性和教育意义的可执行的图像、视频和音频旁白规划。

用户给出的古诗如下：

{poem}

下面是从唐诗三百首资料库中检索到的参考资料。旁白必须重点结合“创作背景”和“赏析要点”来讲，图像和视频必须配套旁白，并借鉴“译文与注释要点”“原文”和“可视化参考”里的原始意象；不要逐字照抄；如果参考资料与用户输入不一致，以用户输入为准。

{reference}

请生成一个古诗教学推理规划，必须满足：
1. 只输出以下 5 个标签，且每个标签都必须显式闭合：<analysis>...</analysis>、<plan4teach>...</plan4teach>、<plan4img>...</plan4img>、<plan4video>...</plan4video>、<plan4audio>...</plan4audio>。
2. 不要输出任何额外标签，不要输出 Markdown 代码块，不要输出标签外的解释文字。
3. 规划必须体现依赖链：先基于参考资料和原诗生成 <analysis>；再基于创作背景和赏析要点生成 <plan4teach>；再基于 <plan4teach> 生成 <plan4img> 的 VACE 参考视觉资产；再基于这些参考资产生成 <plan4video> 的 Wan VACE 视频运动规划；最后基于 <plan4video> 生成每段配套的 <plan4audio>。
4. 视频分成 {self.config.segments} 段，含片头后全片目标时长约 {self.config.target_total_seconds} 秒；每段时长由你根据旁白内容自行规划，duration 必须是 {self.config.min_segment_seconds} 到 {self.config.max_segment_seconds} 秒之间的整数；每段只讲一个重点，段与段之间要有递进，但整条旁白必须像一个连续故事，而不是几页 PPT。
5. 整体视觉是强约束：必须符合中国古代语境和写实历史教育短片质感；服饰、发式、建筑、器物、船只、灯具、道路、自然环境都要中国古代化；禁止现代服装、现代建筑、现代交通工具、电线、路灯、玻璃幕墙、塑料制品、现代文字标识。<plan4img> 负责生成稳定、清晰、可复用的 VACE 参考资产图；<plan4video> 才负责镜头推进、主体动态、环境变化和段落衔接。
6. 不要出现老师、教室、黑板、考点提示；不要动漫、卡通、二次元、插画风、水墨动画、儿童绘本风。
7. <analysis> 内写适合初中及以上学生理解的赏析文本，不要写成考点堆砌；必须把创作背景和赏析要点结合起来，点出这首诗的类型、适合采用的讲述主线、核心情绪、关键画面、声音/空间变化，以及一句话主旨。
8. 如果输出中需要提到作者本人，必须使用作者姓名，不要写“诗人”；如果参考资料出现了“诗人”，也要在输出时替换成作者姓名。
9. <plan4teach>、<plan4img>、<plan4video>、<plan4audio> 内都必须是合法 JSON 对象，不能有 Markdown 代码块。
10. <plan4teach> JSON 格式必须为：{{"poem_title":"...","author":"...","narrative_mode":"按诗歌实际内容选择：抒情/羁旅/山水写景/咏物/叙事/送别/边塞/怀古/其他之一","storyline":"全片讲述主线。抒情或羁旅诗可沿作者/抒情主体的处境、观察和心理变化；咏物诗应沿物象的形态、环境、遭遇和象征含义，不要硬转成作者故事；山水写景诗应沿时间、空间、视线、声色变化和人的感受推进；叙事诗应沿事件因果、人物关系和情节转折推进；怀古诗应沿眼前遗迹到历史想象再到现实感慨推进","recurring_character_style":"如果全片存在同一人物或同一类人物，给出统一剪影描述；没有人物则写无固定人物","segments":[{{"index":1,"title":"...","story_beat":"这一段在整条主线里的作用，必须是处境、物象登场、空间推进、事件推进、发现、反应、转折、回望或点题之一，不要写成诗句翻译或意象标题","teaching_goal":"具体教学目标，必须同时照应创作背景和赏析要点","duration":{self.config.min_segment_seconds}到{self.config.max_segment_seconds}之间的整数}}]}}
11. <plan4img> JSON 格式必须为：{{"segments":[{{"index":1,"image_prompts":["第1张 Wan VACE 参考视觉资产 prompt","第2张 Wan VACE 参考视觉资产 prompt，可选","第3张 Wan VACE 参考视觉资产 prompt，可选"]}}]}}。每段 image_prompts 必须有 1 到 3 条；每条开头必须写明资产类型，且只能从人物资产、环境资产、物象/器物资产、人物+环境资产、物象+环境资产中选择一种；它们不是最终视频截图，不写镜头运动；每条必须描述一个需要锁定的视觉锚点，包含可复用主体、地点环境、时间天气、光线色彩、材质细节和情绪氛围，清晰、写实、构图稳定、轮廓明确、无文字。
12. <plan4video> JSON 格式必须为：{{"segments":[{{"index":1,"video_prompt":"用于 Wan VACE 的视频 prompt，必须以该段 image_prompts 生成的一组参考资产为视觉锚点，配套该段旁白，包含镜头运动、主体动态、环境变化和情绪推进，不要只重复 image_prompts"}}]}}
13. <plan4audio> JSON 格式必须为：{{"segments":[{{"index":1,"audio_text":"自然口语化旁白，像老师带学生进入诗里的现场，必须承接前后段主线，并把创作背景与赏析要点讲成具体、有因果的讲述"}}]}}
14. 先在 <analysis> 中判断这首诗更适合哪种主线，不能一律写作者行动：
    - 抒情、羁旅、送别诗：可沿作者或抒情主体的处境、观察顺序、听觉远近、心理转折推进。
    - 咏物诗：主线应落在物象本身，例如物的形态、处境、季节环境、遭遇、品格联想和象征含义；只在需要点出寄托时再提作者，不要把每段都转成作者故事。
    - 山水写景诗：主线应落在时间、空间、视线移动、声色变化和人在景中的感受；可以无人，也不要强行加人物。
    - 叙事诗：主线必须落在事件因果、人物关系和情节转折，不能拆成景物意象。
    - 怀古诗：主线可从眼前遗迹或景物，转入历史想象，再回到现实感慨。
15. <plan4teach> 的 narrative_mode、storyline、recurring_character_style 与每段 title、story_beat 必须彼此一致；story_beat 沿主线推进，不能按原诗句子或意象清单拆段。
16. 严禁把 audio_text 写成逐句翻译或逐句赏析，严禁每次复述原诗然后翻译：不要让第 2 段只对应第 1 句、第 3 段只对应第 2 句、第 4 段只对应第 3 句；可以借用原诗意象，但每段必须讲清“此刻发生了什么/物象有什么变化/事件怎样推进/感受怎样变化”，而不是“这一句写了什么”。
17. 每段 audio_text 要按该段 duration 自然规划长度：至少适合 {self.config.min_segment_seconds} 秒，最多适合 {self.config.max_segment_seconds} 秒；每段一到两句话，语气要温和、讲故事、有因果或转折，不要为了卡字数牺牲内容。
18. 每段 audio_text 必须包含一个清楚的推进点：可以是人物动作或心理变化，也可以是物象变化、空间移动、声音进入、事件转折、历史联想；再自然带出一个教学点。不能只有景物描写加情绪标签。
19. audio_text 必须重点结合“创作背景”和“赏析要点”：先让学生进入一个具体瞬间或具体物象，再讲这个瞬间为什么能说明当时处境、情绪、物象寄托、事件因果或写法妙处。
20. audio_text 必须前后连贯：第 1 段带学生进入这首诗自己的时代背景、题材类型、时间地点或核心物象/事件；中间段沿 narrative_mode 选择的主线推进，而不是沿着诗句逐字推进；最后一段先把讲述自然收住，再用口语化方式点出这首诗最该抓住的 1 到 2 个重点，例如核心情绪、关键意象、写法妙处、声音/空间变化、物象寄托、事件转折，或创作背景与情感之间的关系；如果出现作者名，再自然补一句作者生平简介。
21. 最后一段点重点时，不能写成“本诗重点是……”“考点是……”“表达了……”这样的总结句，也不能罗列术语；要像老师讲完故事后顺手提醒学生“读到这里，要记住……”或“这首诗最动人的地方，在于……”。
22. 作者生平简介只放在最后一段：如果参考资料有“作者生平参考”，优先使用；如果没有，只能用可靠常识做一句简短概括，不能编造具体年月、官职或轶事。
23. audio_text 禁止写成考点陈述、意象清单、原诗译文或 PPT 标题串联；不要使用“关键意象是”“动静结合”“冷暖对比”“主旨是”“表达了”“本段视频”“画面中”“这一句写了”“视线拉远”“空间感逐渐开阔”“历史的回响”“诗人”等生硬讲解句或镜头说明。
24. audio_text 必须避免“判断句模板化”：不要反复使用“这是……”“这正是……”“这不仅是……更是……”“……是……也是……”“……是……的写照/回响/象征/慰藉”等句式；整组旁白里“是”字判断句最多出现 1 次，且不能用于解释意象。作者生平简介也优先写成“张继，唐代诗人……”这样的同位语，不写“张继是……”。
25. audio_text 还要避免“这 + 抽象评述”的讲解模板：少用或不用“这清冷的秋夜……”“这静谧中的对比……”“这钟声……”“这艘孤舟……”开头；避免“承载着、象征着、凸显了、引发了、映照着、成为……回响/写照”等抽象套话。把含义改写成人物动作、环境变化、听觉/视觉推进、因果转折和心理反应。
26. audio_text 禁止把 4 个以上景物名词用逗号连续排列，例如“明月、山风、落花、流水、孤灯……”；如果需要多个景物，必须让它们服务同一个故事动作或心理变化。
27. 下面两句只用于理解风格，严禁照抄到输出中：“钟声越过江面，落进船舱；他听了一会儿，刚被压下去的离愁又涌上来。”、“他在暮色里停住脚步，忽然想起远方的人；这个小停顿，把整首诗的牵挂带出来了。” 坏例子风格：“这钟声不仅是慰藉，更是家国之痛的回响。”、“明月、山风、落花、流水，表现作者愁绪。”
28. image_prompt 必须从参考资料或原诗中抽取可画出来的地点、景物、人物、器物、光线、材质和空间关系，生成该段给 Wan VACE 使用的参考资产图；禁止只写抽象情绪，也不要写推拉摇移、转场、由远及近等镜头运动。若 audio_text 讲创作背景，image_prompt 要用参考资料中的时代环境、人物处境和原诗可视意象来暗示，不要直接画与原诗无关的宏大战争、宫廷或灾难场面。
29. image_prompt 必须先判断资产类型并写在开头：人物资产用于锁定作者、旅人、友人、女子、将士等跨段人物身份和剪影外观；环境资产用于锁定山水、江面、雨夜、庭院、寺院、边塞、驿路、城楼等空间和时代氛围；物象/器物资产用于锁定客船、寺灯、窗烛、钟、酒杯、书信、花木、月亮等诗中核心意象或声音来源；人物+环境资产用于人物处境离不开地点时；物象+环境资产用于物象必须依托空间才有诗意时。
30. image_prompt 的资产化要求是硬约束：每段只锁定一个最重要的视觉锚点，优先选择能在后续视频里保持一致、能承载该段教学含义的资产；允许主体加少量环境，但禁止多主体拼贴、分屏、多宫格、海报标题页、文字题字、诗句书法、字幕说明、课堂板书。画面要像可被 VACE 继续驱动的参考图，而不是已经完成运动的镜头截图。
31. 人物出现规则是硬约束：如果对应 audio_text 中出现作者姓名、人物身份、人物动作、人物心理或叙事角色，例如某人行走、凝望、倾听、停驻、独坐、远望、对话、出场等，image_prompt 必须出现与旁白一致的人物；人物必须以中国古代人物剪影、背影剪影或侧影剪影呈现，可以被雾、窗棂、舟舱、树影、门影遮挡；禁止清晰正脸、现代写真感、现代服饰。
32. 同一人物或同一类人物跨段出现时，必须复用 <plan4teach> 的 recurring_character_style 中完全相同的剪影描述，包括时代服饰、发式、姿态、相对年龄、携带物和遮挡方式；不能一段像青年、一段像老人，不能一段正脸、一段背影，不能改变服饰身份。若 narrative_mode 不需要人物，不要为了画面好看而强行加人物。
33. video_prompt 必须以 image_prompt 的参考资产为视觉锚点，在其基础上增加时间流动、镜头调度或物体动态，例如云雾流动、水面波纹、灯火摇曳、人物行走、鸟群掠过、声音由远及近带出的空间感。视频运动必须服务旁白里的故事推进，并尽量保持参考资产的主体身份、场景结构和古代写实风格一致。
34. plan4audio 的 {self.config.segments} 段合起来要像一位老师顺着一条明确线索慢慢讲完这首诗，这条线索必须服从 narrative_mode：可以是作者处境、物象变化、空间视线、声音推进、事件因果、历史联想或情绪变化；不要每段重新开头，不要像切换幻灯片，不要把一首诗拆成几句白话翻译。
""".strip()

    def generate(self, poem: str) -> PoetryPlan:
        messages = [
            {"role": "system", "content": "你只输出用户要求的 5 个闭合标签；标签外不要输出任何文字；JSON 标签内容必须是严格合法 JSON。"},
            {"role": "user", "content": self.build_prompt(poem)},
        ]
        payload = {
            "model": self.config.qwen35_model,
            "messages": messages,
            "temperature": 0.4,
            "top_p": 0.9,
            "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        content = self._complete(payload)
        try:
            return parse_poetry_plan(content)
        except Exception as first_error:
            repair_messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "上一次输出无法被程序解析，错误如下："
                        f"{first_error}\n"
                        "请重新输出完整结果，只保留 5 个闭合标签；"
                        "<plan4teach>、<plan4img>、<plan4video>、<plan4audio> 内必须是严格合法 JSON，"
                        "<plan4img> 每段必须使用 image_prompts 数组字段，数组内放 1 到 3 条 VACE 参考资产图 prompt；"
                        "所有字符串里的英文双引号必须转义，不要使用 Markdown，不要输出标签外文字。"
                    ),
                },
            ]
            repair_payload = dict(payload)
            repair_payload["messages"] = repair_messages
            repair_payload["temperature"] = 0.2
            repaired_content = self._complete(repair_payload)
            try:
                return parse_poetry_plan(repaired_content)
            except Exception as repair_error:
                raise RuntimeError(
                    "Qwen35 returned invalid poetry planning output after one repair attempt."
                ) from repair_error

    def _complete(self, payload: dict[str, Any]) -> str:
        response = self._post_chat_completion(payload)
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        if not content.strip():
            raise RuntimeError(f"Qwen35 returned an empty message content: {message}")
        return content

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        import requests

        url = self.config.qwen35_api_base.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.qwen35_api_key}"}
        result = requests.post(url, headers=headers, json=payload, timeout=600)
        result.raise_for_status()
        return result.json()


class OpenAICompatPlanner(Qwen35Planner):
    """Planner that talks to an OpenAI-compatible endpoint (e.g. litellm doubao).

    Reuses all prompt building / repair / parsing logic from ``Qwen35Planner``
    and only overrides how the chat completion request is issued so that the
    OpenAI SDK is used and vLLM-only fields (``chat_template_kwargs``) are
    dropped before the request is sent.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenAICompatPlanner requires the 'openai' package. Install it via 'pip install openai>=1.0'."
            ) from exc
        self._client = OpenAI(
            base_url=config.openai_api_base,
            api_key=config.openai_api_key,
        )

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self.config.openai_model,
            "messages": payload["messages"],
            "temperature": float(payload.get("temperature", self.config.openai_temperature)),
            "top_p": float(payload.get("top_p", self.config.openai_top_p)),
            "max_tokens": int(payload.get("max_tokens", self.config.openai_max_tokens)),
        }
        response = self._client.chat.completions.create(**request_kwargs)
        return response.model_dump()


def build_planner(config: PipelineConfig):
    backend = str(getattr(config, "llm_backend", "qwen35") or "qwen35").strip().lower()
    if backend in ("qwen35", "qwen3.5", "qwen"):
        return Qwen35Planner(config)
    if backend in ("openai", "openai_compat", "litellm", "doubao"):
        return OpenAICompatPlanner(config)
    raise ValueError(f"Unsupported llm backend: {backend}. Choose from ['qwen35', 'openai'].")


class QwenImageGenerator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._pipe = None

    def generate(self, segment: SegmentPlan, output_path: Path, negative_prompt: str = NEGATIVE_IMAGE_PROMPT) -> Path:
        return self.generate_prompt(
            prompt=segment.image_prompt,
            output_path=output_path,
            seed_offset=segment.index,
            negative_prompt=negative_prompt,
        )

    def generate_prompt(self, prompt: str, output_path: Path, seed_offset: int, negative_prompt: str = NEGATIVE_IMAGE_PROMPT) -> Path:
        if output_path.exists() and not self.config.overwrite:
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pipe = self._load_pipe()
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            seed=self.config.seed + seed_offset,
            height=self.config.image_height,
            width=self.config.image_width,
            num_inference_steps=self.config.image_steps,
        )
        image.save(output_path)
        return output_path

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

        model = self.config.qwen_image_model
        self._pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.config.image_device,
            model_configs=[
                ModelConfig(path=sorted(glob.glob(str(model / "transformer" / "diffusion_pytorch_model*.safetensors")))),
                ModelConfig(path=sorted(glob.glob(str(model / "text_encoder" / "model*.safetensors")))),
                ModelConfig(path=str(model / "vae" / "diffusion_pytorch_model.safetensors")),
            ],
            tokenizer_config=ModelConfig(path=str(model / "tokenizer")),
        )
        return self._pipe


class WanI2VGenerator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._pipe = None

    def generate(self, segment: SegmentPlan, image_path: Path, output_path: Path, reference_images: Optional[list[Path]] = None) -> Path:
        if output_path.exists() and not self.config.overwrite:
            return output_path
        if self.config.video_usp_devices.strip():
            return self._generate_with_usp(segment, image_path, output_path, reference_images)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        from diffsynth.utils.data import save_video

        pipe = self._load_pipe()
        # 准备参考图列表：当前段的主参考图 + 可选的其他参考图
        reference_image_list = []
        
        # 主参考图（当前段的关键帧）
        main_reference = Image.open(image_path).convert("RGB").resize((self.config.width, self.config.height))
        reference_image_list.append(main_reference)
        
        # 添加其他参考图（用于跨片段一致性）
        if reference_images:
            for ref_path in reference_images:
                if ref_path.exists() and ref_path != image_path:
                    ref_img = Image.open(ref_path).convert("RGB").resize((self.config.width, self.config.height))
                    reference_image_list.append(ref_img)
        
        video = pipe(
            prompt=segment.video_prompt,
            negative_prompt=NEGATIVE_VIDEO_PROMPT,
            seed=self.config.seed + segment.index,
            tiled=True,
            vace_reference_image=reference_image_list,
            height=self.config.height,
            width=self.config.width,
            num_frames=self._num_frames_for_duration(segment.duration),
            num_inference_steps=self.config.video_steps,
            switch_DiT_boundary=0.875,
        )
        save_video(video, str(output_path), fps=self.config.fps, quality=5)
        return output_path

    def _generate_with_usp(self, segment: SegmentPlan, image_path: Path, output_path: Path, reference_images: Optional[list[Path]] = None) -> Path:
        devices = [item.strip() for item in self.config.video_usp_devices.split(",") if item.strip()]
        if len(devices) < 2:
            raise ValueError("--video-usp-devices must contain at least two GPU ids, for example: 1,2,3,4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request_path = output_path.with_suffix(".wan_vace_usp_request.json")
        request = {
            "segment": asdict(segment),
            "image_path": str(image_path),
            "output_path": str(output_path),
            "reference_images": [str(ref_path) for ref_path in reference_images] if reference_images else [],
            "config": self._worker_config_payload(),
        }
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        env.setdefault("PYTHONPATH", "")
        python_paths = [str(self.config.vgen_root / "third_party" / "DiffSynth-Studio"), str(self.config.vgen_root)]
        env["PYTHONPATH"] = os.pathsep.join(python_paths + ([env["PYTHONPATH"]] if env["PYTHONPATH"] else []))
        master_port = str(self.config.video_usp_master_port + segment.index)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(len(devices)),
            "--master_port",
            master_port,
            "-m",
            "vgen.poetry_teaching",
            "--wan-usp-worker-request",
            str(request_path),
        ]
        subprocess.run(command, env=env, check=True)
        if not output_path.exists():
            raise RuntimeError(f"Wan VACE USP worker finished without writing video: {output_path}")
        return output_path

    def _num_frames_for_duration(self, duration: int) -> int:
        target_frames = max(1, int(duration) * self.config.fps)
        return max(1, round((target_frames - 1) / 4) * 4 + 1)

    def _worker_config_payload(self) -> dict[str, Any]:
        return {
            "vgen_root": str(self.config.vgen_root),
            "wan_i2v_model": str(self.config.wan_i2v_model),
            "clip_seconds": self.config.clip_seconds,
            "fps": self.config.fps,
            "height": self.config.height,
            "width": self.config.width,
            "video_steps": self.config.video_steps,
            "seed": self.config.seed,
        }

    @staticmethod
    def _vace_vram_config(torch_module: Any, device: str) -> dict[str, Any]:
        offload_dtype: Any = torch_module.bfloat16
        offload_device: Any = "cpu"
        if os.environ.get("WAN_VACE_DISK_OFFLOAD", "").lower() in {"1", "true", "yes"}:
            offload_dtype = "disk"
            offload_device = "disk"
        return {
            "offload_dtype": offload_dtype,
            "offload_device": offload_device,
            "onload_dtype": torch_module.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch_module.bfloat16,
            "preparing_device": device,
            "computation_dtype": torch_module.bfloat16,
            "computation_device": device,
        }

    @staticmethod
    def _vram_limit(torch_module: Any, device: str) -> float | None:
        if not str(device).startswith("cuda") or not torch_module.cuda.is_available():
            return None
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
        del free_bytes
        return max(1.0, total_bytes / (1024 ** 3) - 2.0)

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

        model = self.config.wan_i2v_model
        vram_config = self._vace_vram_config(torch, self.config.video_device)
        self._pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.config.video_device,
            model_configs=[
                ModelConfig(path=sorted(glob.glob(str(model / "high_noise_model" / "diffusion_pytorch_model*.safetensors"))), **vram_config),
                ModelConfig(path=sorted(glob.glob(str(model / "low_noise_model" / "diffusion_pytorch_model*.safetensors"))), **vram_config),
                ModelConfig(path=str(model / "models_t5_umt5-xxl-enc-bf16.pth"), **vram_config),
                ModelConfig(path=str(model / "Wan2.1_VAE.pth"), **vram_config),
            ],
            tokenizer_config=ModelConfig(path=str(model / "google" / "umt5-xxl")),
            redirect_common_files=False,
            vram_limit=self._vram_limit(torch, self.config.video_device),
        )
        return self._pipe




def run_wan_usp_worker(request_path: Path) -> None:
    import torch
    import torch.distributed as dist
    from PIL import Image
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import save_video

    request = json.loads(request_path.read_text(encoding="utf-8"))
    payload = request["config"]
    segment = SegmentPlan(**request["segment"])
    image_path = Path(request["image_path"])
    output_path = Path(request["output_path"])
    model = Path(payload["wan_i2v_model"])
    reference_images = [Path(ref_path) for ref_path in request.get("reference_images", [])]

    vram_config = WanI2VGenerator._vace_vram_config(torch, "cuda")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=sorted(glob.glob(str(model / "high_noise_model" / "diffusion_pytorch_model*.safetensors"))), **vram_config),
            ModelConfig(path=sorted(glob.glob(str(model / "low_noise_model" / "diffusion_pytorch_model*.safetensors"))), **vram_config),
            ModelConfig(path=str(model / "models_t5_umt5-xxl-enc-bf16.pth"), **vram_config),
            ModelConfig(path=str(model / "Wan2.1_VAE.pth"), **vram_config),
        ],
        tokenizer_config=ModelConfig(path=str(model / "google" / "umt5-xxl")),
        redirect_common_files=False,
        use_usp=True,
        vram_limit=WanI2VGenerator._vram_limit(torch, "cuda"),
    )
    
    # 准备参考图列表：当前段的主参考图 + 可选的其他参考图
    reference_image_list = []
    
    # 主参考图（当前段的关键帧）
    main_reference = Image.open(image_path).convert("RGB").resize((int(payload["width"]), int(payload["height"])))
    reference_image_list.append(main_reference)
    
    # 添加其他参考图（用于跨片段一致性）
    for ref_path in reference_images:
        if ref_path.exists() and ref_path != image_path:
            ref_img = Image.open(ref_path).convert("RGB").resize((int(payload["width"]), int(payload["height"])))
            reference_image_list.append(ref_img)
    
    target_frames = max(1, int(segment.duration) * int(payload["fps"]))
    num_frames = max(1, round((target_frames - 1) / 4) * 4 + 1)
    video = pipe(
        prompt=segment.video_prompt,
        negative_prompt=NEGATIVE_VIDEO_PROMPT,
        seed=int(payload["seed"]) + segment.index,
        tiled=True,
        vace_reference_image=reference_image_list,
        height=int(payload["height"]),
        width=int(payload["width"]),
        num_frames=num_frames,
        num_inference_steps=int(payload["video_steps"]),
        switch_DiT_boundary=0.875,
    )
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_video(video, str(output_path), fps=int(payload["fps"]), quality=5)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def run_cosyvoice_worker(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    payload = request["config"]
    prompt_wav = payload.get("prompt_wav") or ""
    config = PipelineConfig(
        vgen_root=Path(payload["vgen_root"]),
        cosyvoice_model=Path(payload["cosyvoice_model"]),
        cosyvoice_repo=Path(payload["cosyvoice_repo"]),
        prompt_wav=Path(prompt_wav) if prompt_wav else None,
        prompt_text=payload["prompt_text"],
        overwrite=bool(payload.get("overwrite", True)),
    )
    segment = SegmentPlan(**request["segment"])
    output_path = Path(request["output_path"])
    CosyVoiceGenerator(config).generate(segment, output_path)


class CosyVoiceGenerator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None

    @staticmethod
    def _cosyvoice3_prompt_text(prompt_text: str) -> str:
        prompt_text = prompt_text.strip()
        if "<|endofprompt|>" in prompt_text:
            return prompt_text
        return f"You are a helpful assistant.<|endofprompt|>{prompt_text}"

    def generate(self, segment: SegmentPlan, output_path: Path) -> Path:
        backend = str(getattr(self.config, "audio_backend", "") or os.environ.get("AUDIO_BACKEND", "local_cosyvoice"))
        if backend == "dashscope_cosyvoice":
            return self._generate_with_dashscope(segment, output_path)
        if self.config.cosyvoice_python is not None:
            return self._generate_with_worker(segment, output_path)
        if output_path.exists() and not self.config.overwrite:
            return output_path
        prompt_wav = self.config.prompt_wav
        if prompt_wav is None:
            raise ValueError("CosyVoice zero-shot inference requires --prompt-wav.")
        if not prompt_wav.exists():
            raise FileNotFoundError(f"CosyVoice prompt wav does not exist: {prompt_wav}")
        if not segment.audio_text.strip():
            raise ValueError(f"Segment {segment.index} has empty audio_text.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model = self._load_model()
        import soundfile as sf

        prompt_text = self._cosyvoice3_prompt_text(self.config.prompt_text)
        chunks = list(model.inference_zero_shot(segment.audio_text, prompt_text, str(prompt_wav), stream=False))
        if not chunks:
            raise RuntimeError(f"CosyVoice produced no audio for segment {segment.index}.")
        speech = chunks[0]["tts_speech"]
        for chunk in chunks[1:]:
            import torch
            speech = torch.cat([speech, chunk["tts_speech"]], dim=-1)
        if speech.numel() == 0:
            raise RuntimeError(f"CosyVoice produced empty audio for segment {segment.index}.")
        sf.write(str(output_path), speech.squeeze(0).detach().cpu().numpy(), model.sample_rate)
        return output_path

    def _generate_with_dashscope(self, segment: SegmentPlan, output_path: Path) -> Path:
        if output_path.exists() and not self.config.overwrite:
            return output_path
        if not segment.audio_text.strip():
            raise ValueError(f"Segment {segment.index} has empty audio_text.")
        from vgen.dashscope_cosyvoice import synthesize_to_wav

        return synthesize_to_wav(segment.audio_text, output_path)

    def _generate_with_worker(self, segment: SegmentPlan, output_path: Path) -> Path:
        if output_path.exists() and not self.config.overwrite:
            return output_path
        cosyvoice_python = self.config.cosyvoice_python
        if cosyvoice_python is None:
            raise ValueError("CosyVoice worker requires --cosyvoice-python.")
        if not cosyvoice_python.exists():
            raise FileNotFoundError(f"CosyVoice python does not exist: {cosyvoice_python}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request_path = output_path.with_suffix(".cosyvoice_request.json")
        request = {
            "segment": asdict(segment),
            "output_path": str(output_path),
            "config": {
                "vgen_root": str(self.config.vgen_root),
                "cosyvoice_model": str(self.config.cosyvoice_model),
                "cosyvoice_repo": str(self.config.cosyvoice_repo),
                "prompt_wav": str(self.config.prompt_wav) if self.config.prompt_wav is not None else "",
                "prompt_text": self.config.prompt_text,
                "overwrite": self.config.overwrite,
            },
        }
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")

        env = os.environ.copy()
        repo = self.config.cosyvoice_repo
        matcha = repo / "third_party" / "Matcha-TTS"
        python_paths = [str(self.config.vgen_root), str(repo), str(matcha)]
        env["PYTHONPATH"] = os.pathsep.join(python_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        command = [
            str(cosyvoice_python),
            "-m",
            "vgen.poetry_teaching",
            "--cosyvoice-worker-request",
            str(request_path),
        ]
        subprocess.run(command, env=env, check=True)
        if not output_path.exists():
            raise RuntimeError(f"CosyVoice worker finished without writing audio: {output_path}")
        return output_path

    def _load_model(self):
        if self._model is not None:
            return self._model
        self._prepare_import_path()
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError as exc:
            raise RuntimeError(f"CosyVoice package is not importable from {self.config.cosyvoice_repo}.") from exc
        self._model = AutoModel(model_dir=str(self.config.cosyvoice_model))
        return self._model

    def _prepare_import_path(self) -> None:
        repo = self.config.cosyvoice_repo
        if not repo.exists():
            raise FileNotFoundError(f"CosyVoice repo does not exist: {repo}")
        matcha = repo / "third_party" / "Matcha-TTS"
        for path in [repo, matcha]:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


class MediaComposer:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def mux(self, video_path: Path, audio_path: Path, output_path: Path, shortest: bool = True) -> Path:
        if output_path.exists() and not self.config.overwrite:
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self._ffmpeg_executable(), "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
        ]
        if shortest:
            command.append("-shortest")
        command.append(str(output_path))
        self._run_ffmpeg(command)
        return output_path

    def mux_extend_last_frame(self, video_path: Path, audio_path: Path, output_path: Path) -> Path:
        audio_seconds = self.audio_duration(audio_path)
        if output_path.exists() and not self.config.overwrite:
            try:
                muxed_audio_seconds = self.audio_stream_duration(output_path)
                muxed_media_seconds = self.media_duration(output_path)
                if muxed_audio_seconds + 0.05 >= audio_seconds and abs(muxed_media_seconds - audio_seconds) <= 0.5:
                    return output_path
            except RuntimeError:
                pass
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_ffmpeg([
            self._ffmpeg_executable(), "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration=3600,trim=duration={audio_seconds:.3f},setpts=PTS-STARTPTS,setsar=1,format=yuv420p[v]",
            "-map", "[v]",
            "-map", "1:a:0",
            "-t", f"{audio_seconds:.3f}",
            "-r", str(self.config.fps),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-c:a", "aac",
            str(output_path),
        ])
        return output_path

    def concat(self, videos: Iterable[Path], output_path: Path, reencode: bool = False) -> Path:
        videos = list(videos)
        if not videos:
            raise ValueError("No segment videos to concatenate.")
        if output_path.exists() and not self.config.overwrite:
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        list_path = output_path.parent / "concat_list.txt"
        list_path.write_text("".join(f"file '{path.resolve()}'\n" for path in videos), encoding="utf-8")
        if reencode:
            command = [self._ffmpeg_executable(), "-y"]
            for path in videos:
                command.extend(["-i", str(path)])
            filter_parts: list[str] = []
            concat_inputs: list[str] = []
            for index in range(len(videos)):
                filter_parts.append(f"[{index}:v:0]setpts=PTS-STARTPTS,setsar=1,format=yuv420p[v{index}]")
                filter_parts.append(f"[{index}:a:0]asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0[a{index}]")
                concat_inputs.append(f"[v{index}][a{index}]")
            filter_parts.append("".join(concat_inputs) + f"concat=n={len(videos)}:v=1:a=1[v][a]")
            command.extend([
                "-filter_complex", ";".join(filter_parts),
                "-map", "[v]",
                "-map", "[a]",
                "-r", str(self.config.fps),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "18",
                "-c:a", "aac",
                "-ar", "24000",
                "-ac", "1",
                str(output_path),
            ])
        else:
            command = [
                self._ffmpeg_executable(), "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_path),
                "-c", "copy",
                str(output_path),
            ]
        self._run_ffmpeg(command)
        return output_path

    def image_to_video(self, image_path: Path, duration: float, output_path: Path) -> Path:
        if output_path.exists() and not self.config.overwrite:
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_ffmpeg([
            self._ffmpeg_executable(), "-y",
            "-loop", "1",
            "-t", f"{duration:.3f}",
            "-i", str(image_path),
            "-vf", f"scale={self.config.width}:{self.config.height}:force_original_aspect_ratio=decrease,pad={self.config.width}:{self.config.height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p",
            "-r", str(self.config.fps),
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            str(output_path),
        ])
        return output_path

    def media_duration(self, path: Path) -> float:
        try:
            ffprobe = self._ffprobe_executable()
        except RuntimeError:
            ffprobe = ""
        if ffprobe:
            command = [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                return float(result.stdout.strip())
            except (subprocess.CalledProcessError, ValueError):
                pass
        command = [
            self._ffmpeg_executable(),
            "-i", str(path),
            "-f", "null",
            "-",
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr or "")
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        raise RuntimeError(f"Unable to determine media duration: {path}")

    def audio_stream_duration(self, path: Path) -> float:
        try:
            ffprobe = self._ffprobe_executable()
        except RuntimeError:
            return self.media_duration(path)
        command = [
            ffprobe,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            value = result.stdout.strip()
            if value and value != "N/A":
                return float(value)
        except (subprocess.CalledProcessError, ValueError):
            pass
        return self.audio_duration(path)

    def audio_duration(self, audio_path: Path) -> float:
        try:
            return self.media_duration(audio_path)
        except RuntimeError:
            pass

        if audio_path.suffix.lower() == ".wav":
            try:
                import wave

                with wave.open(str(audio_path), "rb") as audio_file:
                    frame_rate = audio_file.getframerate()
                    if frame_rate > 0:
                        return audio_file.getnframes() / float(frame_rate)
            except (EOFError, OSError, wave.Error):
                pass

        command = [
            self._ffmpeg_executable(),
            "-i", str(audio_path),
            "-f", "null",
            "-",
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr or "")
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        raise RuntimeError(f"Unable to determine audio duration: {audio_path}")

    @staticmethod
    def _ffmpeg_executable() -> str:
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
        import shutil

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        raise RuntimeError("ffmpeg is not available; install ffmpeg or imageio-ffmpeg.")

    @staticmethod
    def _ffprobe_executable() -> str:
        import shutil

        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            return ffprobe
        ffmpeg = MediaComposer._ffmpeg_executable()
        candidate = Path(ffmpeg).with_name("ffprobe")
        if candidate.exists():
            return str(candidate)
        raise RuntimeError("ffprobe is not available; install ffmpeg with ffprobe.")

    @staticmethod
    def _run_ffmpeg(command: list[str]) -> None:
        subprocess.run(command, check=True)


SUPPORTED_VIDEO_BACKENDS = ("wan", "bernini", "bernini_r", "hunyuan")


def build_video_generator(config: PipelineConfig):
    backend = str(getattr(config, "video_backend", "wan") or "wan").strip().lower()
    if backend not in SUPPORTED_VIDEO_BACKENDS:
        raise ValueError(f"Unsupported video backend: {backend}. Choose from {SUPPORTED_VIDEO_BACKENDS}.")
    if backend == "wan":
        return WanI2VGenerator(config)
    if backend == "bernini":
        from vgen.video_backends import BerniniVideoGenerator
        return BerniniVideoGenerator(config)
    if backend == "bernini_r":
        from vgen.video_backends import BerniniRVideoGenerator
        return BerniniRVideoGenerator(config)
    if backend == "hunyuan":
        from vgen.video_backends import HunyuanVideoGenerator
        return HunyuanVideoGenerator(config)
    raise ValueError(f"Unsupported video backend: {backend}")


class PoetryTeachingPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.planner = build_planner(config)
        self.image_generator = build_image_generator(config)
        self.video_generator = build_video_generator(config)
        self.audio_generator = CosyVoiceGenerator(config)
        self.composer = MediaComposer(config)

    def run(self, poem: str, run_name: str | None = None) -> PipelineResult:
        self._validate_config()
        output_dir = self._make_output_dir(run_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        plan_path = output_dir / "plan.json"
        if plan_path.exists() and not self.config.overwrite:
            plan = load_plan(plan_path)
        else:
            plan = self.planner.generate(poem)
            save_plan(plan, plan_path)
            (output_dir / "plan_tagged.txt").write_text(plan.raw_response, encoding="utf-8")

        apply_script_first_sync(plan)
        for segment in plan.segments:
            segment.video_prompt = upsert_sync_suffix(segment.video_prompt, segment.av_sync_plan, limit=760)

        normalize_segment_durations(
            plan.segments,
            total_seconds=self.config.target_total_seconds,
            min_seconds=self.config.min_segment_seconds,
            max_seconds=self.config.max_segment_seconds,
        )
        save_plan(plan, plan_path)

        intro_artifacts = self._generate_intro(poem, plan, output_dir)
        artifacts = [SegmentArtifacts(index=segment.index) for segment in plan.segments]
        by_index = {artifact.index: artifact for artifact in artifacts}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.image_workers) as image_pool, \
             concurrent.futures.ThreadPoolExecutor(max_workers=self.config.video_workers) as video_pool, \
             concurrent.futures.ThreadPoolExecutor(max_workers=self.config.audio_workers) as audio_pool, \
             concurrent.futures.ThreadPoolExecutor(max_workers=2) as mux_pool:
            # 第一阶段：生成所有 VACE 参考资产图
            image_futures: dict[int, concurrent.futures.Future[list[Path]]] = {}
            audio_futures: dict[int, concurrent.futures.Future[Path]] = {}

            for segment in plan.segments:
                paths = self._segment_paths(output_dir, segment.index)
                image_futures[segment.index] = image_pool.submit(self._generate_images, segment, paths["images"])
                audio_futures[segment.index] = audio_pool.submit(self._generate_audio, segment, paths["audio"])

            # 等待所有资产图生成完成
            image_paths_by_segment: dict[int, list[Path]] = {}
            for segment_index, future in image_futures.items():
                image_paths_by_segment[segment_index] = future.result()

            # 第二阶段：生成视频（优先使用同段多资产图作为 VACE 参考）
            video_futures: list[concurrent.futures.Future[Path]] = []
            mux_futures: list[concurrent.futures.Future[Path]] = []

            for segment in plan.segments:
                paths = self._segment_paths(output_dir, segment.index)
                current_images = image_paths_by_segment[segment.index]
                if not current_images:
                    raise RuntimeError(f"No reference images were generated for segment {segment.index}.")
                main_image = current_images[0]
                reference_images = current_images[1:]

                # 创建一个已完成的Future，包含当前段的主参考图路径
                completed_future = concurrent.futures.Future()
                completed_future.set_result(main_image)

                video_futures.append(video_pool.submit(
                    self._generate_video_after_image,
                    segment,
                    completed_future,
                    paths["video"],
                    reference_images,
                ))

            for future in concurrent.futures.as_completed(video_futures):
                video_path = future.result()
                segment_index = parse_segment_index(video_path)
                by_index[segment_index].video = video_path
                audio_path = audio_futures[segment_index].result()
                by_index[segment_index].audio = audio_path
                paths = self._segment_paths(output_dir, segment_index)
                mux_futures.append(mux_pool.submit(self.composer.mux_extend_last_frame, video_path, audio_path, paths["muxed"]))

            for future in concurrent.futures.as_completed(mux_futures):
                muxed_path = future.result()
                by_index[parse_segment_index(muxed_path)].muxed_video = muxed_path

        for segment in plan.segments:
            paths = self._segment_paths(output_dir, segment.index)
            artifact = by_index[segment.index]
            artifact.images = [image_path for image_path in paths["images"] if image_path.exists()]
            artifact.image = artifact.images[0] if artifact.images else None
            artifact.video = paths["video"] if paths["video"].exists() else None
            artifact.audio = paths["audio"] if paths["audio"].exists() else None
            artifact.muxed_video = paths["muxed"] if paths["muxed"].exists() else None

        muxed_videos = [artifact.muxed_video for artifact in artifacts if artifact.muxed_video is not None]
        if intro_artifacts.muxed_video is not None:
            muxed_videos = [intro_artifacts.muxed_video] + muxed_videos
        final_video = self.composer.concat(muxed_videos, output_dir / "final_poetry_teaching.mp4", reencode=True) if muxed_videos else None
        return PipelineResult(output_dir=output_dir, plan_path=plan_path, artifacts=artifacts, final_video=final_video, intro=intro_artifacts)

    def _validate_config(self) -> None:
        if self.config.skip_audio:
            return
        if not self.config.cosyvoice_repo.exists():
            raise FileNotFoundError(f"CosyVoice repo does not exist: {self.config.cosyvoice_repo}")
        if self.config.prompt_wav is None:
            raise ValueError("CosyVoice zero-shot inference requires --prompt-wav.")
        if not self.config.prompt_wav.exists():
            raise FileNotFoundError(f"CosyVoice prompt wav does not exist: {self.config.prompt_wav}")

    def _generate_image(self, segment: SegmentPlan, path: Path) -> Path:
        if self.config.skip_image:
            if path.exists():
                return path
            raise RuntimeError("--skip-image requires existing image files; this path does not exist: " + str(path))
        return self.image_generator.generate(segment, path)

    def _generate_images(self, segment: SegmentPlan, paths: list[Path]) -> list[Path]:
        prompts = segment.image_prompts or [segment.image_prompt]
        if len(paths) < len(prompts):
            raise RuntimeError(f"Not enough image output paths for segment {segment.index}.")
        generated_paths: list[Path] = []
        for asset_index, prompt in enumerate(prompts):
            path = paths[asset_index]
            if self.config.skip_image:
                if path.exists():
                    generated_paths.append(path)
                    continue
                raise RuntimeError("--skip-image requires existing image files; this path does not exist: " + str(path))
            generated_paths.append(
                self.image_generator.generate_prompt(
                    prompt=prompt,
                    output_path=path,
                    seed_offset=segment.index * 10 + asset_index,
                )
            )
        return generated_paths

    def _generate_text_card_image(self, segment: SegmentPlan, path: Path) -> Path:
        if self.config.skip_image:
            if path.exists():
                return path
            raise RuntimeError("--skip-image requires existing intro image file: " + str(path))
        return self.image_generator.generate(segment, path, negative_prompt=TEXT_CARD_NEGATIVE_IMAGE_PROMPT)

    def _generate_audio(self, segment: SegmentPlan, path: Path) -> Path:
        if self.config.skip_audio:
            if path.exists():
                return path
            raise RuntimeError("--skip-audio requires existing audio files; this path does not exist: " + str(path))
        return self.audio_generator.generate(segment, path)

    def _generate_video_after_image(self, segment: SegmentPlan, image_future: concurrent.futures.Future[Path], path: Path, reference_images: Optional[list[Path]] = None) -> Path:
        if self.config.skip_video:
            if path.exists():
                return path
            raise RuntimeError("--skip-video requires existing video files; this path does not exist: " + str(path))
        
        # 获取当前段的图像路径
        if image_future.done():
            image_path = image_future.result()
        else:
            # 如果传入的是占位符Future，则直接从路径中获取
            output_dir = path.parent.parent
            paths = self._segment_paths(output_dir, segment.index)
            image_path = paths["image"]
            if not image_path.exists():
                raise RuntimeError(f"Image file does not exist for segment {segment.index}: {image_path}")
        
        return self.video_generator.generate(segment, image_path, path, reference_images)

    def _generate_intro(self, poem: str, plan: PoetryPlan, output_dir: Path) -> IntroArtifacts:
        paths = self._intro_paths(output_dir)
        poem_item = self.planner.reference_provider.find_poem(poem)
        title = plan.poem_title.strip()
        author = plan.author.strip()
        dynasty = "唐"
        poem_text = extract_poem_content_from_input(poem)
        background = ""
        if poem_item:
            title = str(poem_item.get("detail_title") or poem_item.get("title") or title).strip()
            author = str(poem_item.get("author") or poem_item.get("list_author") or author).strip()
            dynasty = normalize_dynasty(poem_item.get("dynasty") or dynasty)
            poem_text = str(poem_item.get("content") or poem_text).strip()
            background = extract_background_from_reference(poem_item)
        intro_audio_text = build_intro_audio_text(title, author, background)
        (output_dir / "intro").mkdir(parents=True, exist_ok=True)

        title_segment = build_intro_title_segment(title, author, dynasty)
        poem_segment = build_intro_poem_segment(poem_text)
        self._generate_text_card_image(title_segment, paths["title_image"])
        self._generate_text_card_image(poem_segment, paths["poem_image"])

        audio_path: Path | None = None
        intro_audio_duration = 0.0
        if intro_audio_text.strip():
            intro_audio_segment = SegmentPlan(
                index=0,
                title="intro_background",
                story_beat="intro",
                teaching_goal="introduce poem and original creation background",
                image_prompt="",
                video_prompt="",
                audio_text=intro_audio_text,
                duration=INTRO_TITLE_SECONDS + INTRO_POEM_SECONDS,
            )
            audio_path = self._generate_audio(intro_audio_segment, paths["background_audio"])
            intro_audio_duration = self.composer.audio_duration(audio_path)

        poem_seconds = max(float(INTRO_POEM_SECONDS), math.ceil(max(0.0, intro_audio_duration - INTRO_TITLE_SECONDS)))
        if self.config.skip_video:
            for key in ["title_video", "poem_video", "silent_video"]:
                if not paths[key].exists():
                    raise RuntimeError("--skip-video requires existing intro video file: " + str(paths[key]))
        else:
            self.composer.image_to_video(paths["title_image"], INTRO_TITLE_SECONDS, paths["title_video"])
            self.composer.image_to_video(paths["poem_image"], poem_seconds, paths["poem_video"])
            self.composer.concat([paths["title_video"], paths["poem_video"]], paths["silent_video"])

        muxed_path: Path | None = None
        if audio_path is not None:
            muxed_path = self.composer.mux(paths["silent_video"], audio_path, paths["muxed"], shortest=False)
        elif paths["silent_video"].exists():
            muxed_path = paths["silent_video"]
        paths["metadata"].write_text(json.dumps({
            "title": title,
            "author": author,
            "dynasty": dynasty,
            "poem_text": poem_text,
            "background": background,
            "audio_text": intro_audio_text,
            "title_seconds": INTRO_TITLE_SECONDS,
            "poem_seconds": poem_seconds,
            "audio_seconds": intro_audio_duration,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return IntroArtifacts(
            title_author_image=paths["title_image"] if paths["title_image"].exists() else None,
            full_poem_image=paths["poem_image"] if paths["poem_image"].exists() else None,
            title_author_video=paths["title_video"] if paths["title_video"].exists() else None,
            full_poem_video=paths["poem_video"] if paths["poem_video"].exists() else None,
            silent_video=paths["silent_video"] if paths["silent_video"].exists() else None,
            audio=audio_path,
            muxed_video=muxed_path,
        )

    def _segment_paths(self, output_dir: Path, index: int) -> dict[str, Any]:
        stem = f"segment_{index:02d}"
        image_paths = [
            output_dir / "images" / f"{stem}.png",
            output_dir / "images" / f"{stem}_asset_02.png",
            output_dir / "images" / f"{stem}_asset_03.png",
        ]
        return {
            "image": image_paths[0],
            "images": image_paths,
            "video": output_dir / "videos" / f"{stem}.mp4",
            "audio": output_dir / "audio" / f"{stem}.wav",
            "muxed": output_dir / "muxed" / f"{stem}.mp4",
        }

    def _intro_paths(self, output_dir: Path) -> dict[str, Path]:
        return {
            "metadata": output_dir / "intro" / "intro_metadata.json",
            "title_image": output_dir / "images" / "intro_title_author.png",
            "poem_image": output_dir / "images" / "intro_poem.png",
            "title_video": output_dir / "videos" / "intro_title_author.mp4",
            "poem_video": output_dir / "videos" / "intro_poem.mp4",
            "silent_video": output_dir / "videos" / "intro_silent.mp4",
            "background_audio": output_dir / "audio" / "intro_background.wav",
            "muxed": output_dir / "muxed" / "intro.mp4",
        }

    def _make_output_dir(self, run_name: str | None) -> Path:
        if run_name:
            return self.config.output_root / sanitize_name(run_name)
        return self.config.output_root / time.strftime("%Y%m%d_%H%M%S")


class PlanEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def parse_poetry_plan(raw_response: str) -> PoetryPlan:
    analysis = extract_required_tag(raw_response, "analysis")
    plan4teach = extract_required_tag(raw_response, "plan4teach")
    plan4img = extract_required_tag(raw_response, "plan4img")
    plan4video = extract_required_tag(raw_response, "plan4video")
    plan4audio = extract_required_tag(raw_response, "plan4audio")

    teach_data = load_json_object(plan4teach, "plan4teach")
    img_data = load_json_object(plan4img, "plan4img")
    video_data = load_json_object(plan4video, "plan4video")
    audio_data = load_json_object(plan4audio, "plan4audio")

    require_keys(teach_data, ["poem_title", "author", "segments"], "plan4teach")
    teach_segments = require_segments(teach_data, "plan4teach")
    img_segments = require_segments(img_data, "plan4img")
    video_segments = require_segments(video_data, "plan4video")
    audio_segments = require_segments(audio_data, "plan4audio")
    if not teach_segments:
        raise ValueError("<plan4teach> must contain at least one segment.")

    img_by_index = index_segments(img_segments, "plan4img")
    video_by_index = index_segments(video_segments, "plan4video")
    audio_by_index = index_segments(audio_segments, "plan4audio")
    require_keys(teach_data, ["narrative_mode", "storyline", "recurring_character_style"], "plan4teach")
    author = str(teach_data["author"])
    recurring_character_style = replace_poet_placeholder(str(teach_data["recurring_character_style"]), author)

    segments = []
    for item in teach_segments:
        require_keys(item, ["index", "title", "story_beat", "teaching_goal", "duration"], "plan4teach segment")
        index = int(item["index"])
        if index not in img_by_index or index not in video_by_index or index not in audio_by_index:
            raise ValueError(f"Segment {index} is not present in every plan tag.")
        img_item = img_by_index[index]
        video_item = video_by_index[index]
        audio_item = audio_by_index[index]
        require_keys(video_item, ["video_prompt"], f"plan4video segment {index}")
        require_keys(audio_item, ["audio_text"], f"plan4audio segment {index}")
        duration = clamp_segment_duration(int(item["duration"]), DEFAULT_MIN_SEGMENT_SECONDS, DEFAULT_MAX_SEGMENT_SECONDS)
        audio_text = replace_poet_placeholder(str(audio_item["audio_text"]), author)
        image_prompts = extract_image_prompts(img_item, author, f"plan4img segment {index}")
        video_prompt = replace_poet_placeholder(str(video_item["video_prompt"]), author)
        image_prompts, video_prompt = enforce_visual_prompt_rules(
            image_prompts=image_prompts,
            video_prompt=video_prompt,
            audio_text=audio_text,
            recurring_character_style=recurring_character_style,
            author=author,
        )
        image_prompt = image_prompts[0]
        sync_plan = audio_item.get("av_sync_plan") if isinstance(audio_item.get("av_sync_plan"), list) else []
        script_beat = replace_poet_placeholder(str(audio_item.get("script_beat") or audio_text), author)
        video_prompt = upsert_sync_suffix(video_prompt, sync_plan or [{"audio_cue": script_beat}], limit=760)
        segments.append(
            SegmentPlan(
                index=index,
                title=replace_poet_placeholder(str(item["title"]), author),
                story_beat=replace_poet_placeholder(str(item["story_beat"]), author),
                teaching_goal=replace_poet_placeholder(str(item["teaching_goal"]), author),
                image_prompt=image_prompt,
                video_prompt=video_prompt,
                audio_text=audio_text,
                duration=duration,
                image_prompts=image_prompts,
                script_beat=script_beat,
                av_sync_plan=sync_plan,
            )
        )
    plan = PoetryPlan(
        poem_title=str(teach_data["poem_title"]),
        author=author,
        narrative_mode=replace_poet_placeholder(str(teach_data["narrative_mode"]), author),
        storyline=replace_poet_placeholder(str(teach_data["storyline"]), author),
        recurring_character_style=recurring_character_style,
        appreciation=replace_poet_placeholder(analysis, author),
        plan4teach=plan4teach,
        plan4img=plan4img,
        plan4video=plan4video,
        plan4audio=plan4audio,
        segments=segments,
        raw_response=raw_response,
        full_script=replace_poet_placeholder(str(audio_data.get("full_script") or teach_data.get("full_script") or derive_full_script(segments)), author),
    )
    apply_script_first_sync(plan)
    for segment in plan.segments:
        segment.video_prompt = upsert_sync_suffix(segment.video_prompt, segment.av_sync_plan, limit=760)
    return plan


def load_json_object(text: str, tag: str) -> dict[str, Any]:
    data = json.loads(text.strip())
    if not isinstance(data, dict):
        raise ValueError(f"<{tag}> must contain a JSON object.")
    return data


def extract_image_prompts(img_item: dict[str, Any], author: str, context: str) -> list[str]:
    if "image_prompts" in img_item:
        raw_prompts = img_item["image_prompts"]
        if not isinstance(raw_prompts, list):
            raise ValueError(f"{context} image_prompts must be a list.")
        prompts = [replace_poet_placeholder(str(prompt), author).strip() for prompt in raw_prompts if str(prompt).strip()]
    elif "image_prompt" in img_item:
        prompts = [replace_poet_placeholder(str(img_item["image_prompt"]), author).strip()]
    else:
        raise ValueError(f"Missing required keys in {context}: ['image_prompts']")
    if not 1 <= len(prompts) <= 3:
        raise ValueError(f"{context} image_prompts must contain 1 to 3 prompts.")
    return prompts


def enforce_visual_prompt_rules(
    image_prompts: list[str],
    video_prompt: str,
    audio_text: str,
    recurring_character_style: str,
    author: str,
) -> tuple[list[str], str]:
    image_suffixes = [VACE_REFERENCE_ASSET_VISUAL_RULE, ANCIENT_CHINESE_VISUAL_STYLE]
    video_suffixes = [ANCIENT_CHINESE_VISUAL_STYLE, "以参考资产图为视觉锚点，保持主体身份、场景结构、服饰器物和整体风格一致"]
    character_style = recurring_character_style.strip()
    has_recurring_character = character_style and character_style != "无固定人物"
    mentions_person_in_images = any(text_mentions_person(prompt, author) for prompt in image_prompts)
    if has_recurring_character and text_mentions_person(audio_text, author):
        character_suffix = f"同一人物形象必须保持一致：{character_style}"
        image_suffixes.append(character_suffix)
        video_suffixes.append(character_suffix)
        image_suffixes.append(SILHOUETTE_CHARACTER_VISUAL_RULE)
        video_suffixes.append(SILHOUETTE_CHARACTER_VISUAL_RULE)
    elif mentions_person_in_images or text_mentions_person(video_prompt, author):
        image_suffixes.append(SILHOUETTE_CHARACTER_VISUAL_RULE)
        video_suffixes.append(SILHOUETTE_CHARACTER_VISUAL_RULE)
    image_prompts = [append_prompt_suffix(prompt, image_suffixes) for prompt in image_prompts]
    video_prompt = append_prompt_suffix(video_prompt, video_suffixes)
    return image_prompts, video_prompt


def append_prompt_suffix(prompt: str, suffixes: list[str]) -> str:
    prompt = prompt.strip()
    for suffix in suffixes:
        if suffix and suffix not in prompt:
            prompt = f"{prompt}；{suffix}" if prompt else suffix
    return prompt


def text_mentions_person(text: str, author: str = "") -> bool:
    person_pattern = (
        r"人物|作者|男子|女子|老人|孩童|旅人|游子|行人|客子|将士|士兵|征夫|渔夫|樵夫|僧人|道士|君王|"
        r"宫女|友人|主人|客人|故人|亲人|背影|侧影|剪影|身影|独坐|凝望|远望|回望|倾听|行走|停驻|"
        r"登临|倚窗|倚栏|船头|舟中|立于|坐在|走向|他|她|我|你|张继|李白|杜甫|王维|白居易|孟浩然|"
        r"王昌龄|岑参|杜牧|李商隐|刘禹锡|柳宗元|韩愈|王之涣|王勃|陈子昂|高适|韦应物|孟郊|贾岛|"
        r"元稹|张九龄|王翰|贺知章|崔颢|常建|祖咏|李贺"
    )
    if author.strip() and author.strip() != "未知" and author.strip() in text:
        return True
    return bool(re.search(person_pattern, text))

def build_intro_title_segment(title: str, author: str, dynasty: str) -> SegmentPlan:
    text_line_1 = f"《{title}》"
    text_line_2 = f"【{dynasty}】{author}"
    return SegmentPlan(
        index=9001,
        title="intro_title_author",
        story_beat="intro title author card",
        teaching_goal="show poem title and author before teaching video",
        image_prompt=(
            "生成一张中国古代诗词教学短片头图，中式风格。"
            "画面中央必须只有两行中文文字，文字必须准确、清晰、可读、居中："
            f"第一行一字不差为{text_line_1}；第二行一字不差为{text_line_2}。"
            "除这两行和背景外不要出现任何其他字、标点、装饰性文字以及其他元素。"
        ),
        video_prompt="",
        audio_text="",
        duration=INTRO_TITLE_SECONDS,
    )

def build_intro_poem_segment(poem_text: str) -> SegmentPlan:
    poem_text = poem_text.strip()
    return SegmentPlan(
        index=9002,
        title="intro_full_poem",
        story_beat="intro full poem card",
        teaching_goal="show original poem text before teaching video",
        image_prompt=(
            "生成文本图，宣纸风格，美观排版，传统中国书法黑色。"
            "必须只包含下面这段中文，逐字逐句、标点必须保持一致，不要出现任何其他元素\n"
            f"{poem_text}\n"
        ),
        video_prompt="",
        audio_text="",
        duration=INTRO_POEM_SECONDS,
    )

def build_intro_audio_text(title: str, author: str, background: str) -> str:
    title = title.strip()
    author = author.strip()
    background = clean_reference_text(background)
    if not title or not author or not background:
        return ""
    return f"同学你好，今天我们要走近的是{author}写的《{title}》。{background}"

def extract_background_from_reference(item: dict[str, Any] | None) -> str:
    if not item:
        return ""
    background = str(item.get("background") or "").strip()
    if background:
        return background
    raw_sections = item.get("sections")
    sections = raw_sections if isinstance(raw_sections, list) else []
    for section in sections:
        if isinstance(section, dict) and str(section.get("title", "")).strip() == "创作背景":
            return str(section.get("text") or "").strip()
    return ""

def extract_poem_content_from_input(poem: str) -> str:
    lines = [line.strip() for line in poem.splitlines() if line.strip()]
    content_lines = [line for line in lines if re.search(r"[，。！？；、]", line)]
    if content_lines:
        return "\n".join(content_lines)
    parts = re.split(r"(?<=[。！？])\s*", poem.strip())
    return "\n".join(part.strip() for part in parts if re.search(r"[，。！？；、]", part))

def normalize_dynasty(value: Any) -> str:
    dynasty = str(value or "唐代").strip() or "唐代"
    return dynasty[:-1] if dynasty.endswith("代") and len(dynasty) > 1 else dynasty

def clean_reference_text(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("展开阅读全文", "").strip())

def replace_poet_placeholder(text: str, author: str) -> str:
    author = author.strip()
    if not author or author == "未知":
        return text
    text = text.replace("这位诗人", author).replace("那位诗人", author)
    text = re.sub(r"(?<!代)诗人(?=在|把|将|用|以|从|向|对|为|被|听|看|望|想|写|说|立|坐|行|停|泊|登|回|独|正|也|又|仍|还|已|曾|会|要|心|身|眼|耳|笔|脚|船|舟|门|窗|江|山|月|风|雨|雪|花|水|路|夜|晨|暮|，|。|；|：|、|\s|$)", author, text)
    return text


def require_keys(data: dict[str, Any], keys: list[str], context: str) -> None:
    missing_keys = [key for key in keys if key not in data or data[key] in (None, "", [])]
    if missing_keys:
        raise ValueError(f"Missing required keys in {context}: {missing_keys}")


def require_segments(data: dict[str, Any], tag: str) -> list[dict[str, Any]]:
    require_keys(data, ["segments"], tag)
    segments = data["segments"]
    if not isinstance(segments, list):
        raise ValueError(f"<{tag}> segments must be a list.")
    for i, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"<{tag}> segment {i + 1} must be a JSON object.")
    return segments


def index_segments(segments: list[dict[str, Any]], tag: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for segment in segments:
        require_keys(segment, ["index"], f"{tag} segment")
        index = int(segment["index"])
        if index in indexed:
            raise ValueError(f"Duplicate segment index {index} in <{tag}>.")
        indexed[index] = segment
    return indexed


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_required_tag(text: str, tag: str) -> str:
    value = extract_tag(text, tag)
    if not value:
        raise ValueError(f"Missing required <{tag}> tag in Qwen35 response.")
    return value


def parse_segment_index(path: Path) -> int:
    match = re.search(r"segment_(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse segment index from path: {path}")
    return int(match.group(1))


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_.-]+", "_", value.strip())
    return value.strip("_") or "poetry_teaching"


def save_plan(plan: PoetryPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(plan)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, cls=PlanEncoder), encoding="utf-8")


def load_plan(path: Path) -> PoetryPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = [SegmentPlan(**item) for item in payload.get("segments", [])]
    payload = {key: value for key, value in payload.items() if key != "segments"}
    return PoetryPlan(segments=segments, **payload)


def clamp_segment_duration(duration: int, min_seconds: int, max_seconds: int) -> int:
    min_seconds = int(min_seconds)
    max_seconds = max(min_seconds, int(max_seconds))
    return min(max_seconds, max(min_seconds, int(duration)))


def target_segment_total_seconds(
    total_seconds: int,
    segment_count: int,
    min_seconds: int,
    max_seconds: int,
    intro_seconds: int = INTRO_TITLE_SECONDS + INTRO_POEM_SECONDS,
) -> int:
    segment_count = max(1, int(segment_count))
    min_total = segment_count * int(min_seconds)
    max_total = segment_count * max(int(min_seconds), int(max_seconds))
    desired_total = max(1, int(total_seconds)) - int(intro_seconds)
    return max(min_total, min(max_total, desired_total))


def normalize_segment_durations(
    segments: list[SegmentPlan],
    total_seconds: int,
    min_seconds: int,
    max_seconds: int,
    intro_seconds: int = INTRO_TITLE_SECONDS + INTRO_POEM_SECONDS,
) -> None:
    if not segments:
        return

    min_seconds = int(min_seconds)
    max_seconds = max(min_seconds, int(max_seconds))
    durations = [clamp_segment_duration(segment.duration, min_seconds, max_seconds) for segment in segments]
    target_total = target_segment_total_seconds(total_seconds, len(segments), min_seconds, max_seconds, intro_seconds)
    current_total = sum(durations)

    while current_total < target_total:
        changed = False
        for index, duration in enumerate(durations):
            if duration < max_seconds:
                durations[index] += 1
                current_total += 1
                changed = True
                if current_total >= target_total:
                    break
        if not changed:
            break

    while current_total > target_total:
        changed = False
        for index in range(len(durations) - 1, -1, -1):
            if durations[index] > min_seconds:
                durations[index] -= 1
                current_total -= 1
                changed = True
                if current_total <= target_total:
                    break
        if not changed:
            break

    for segment, duration in zip(segments, durations):
        segment.duration = duration


def build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(
        output_root=Path(args.output_root),
        qwen35_api_base=args.qwen35_api_base,
        qwen35_model=args.qwen35_model,
        llm_backend=args.llm_backend,
        openai_api_base=args.openai_api_base,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model,
        qwen35_api_key=args.qwen35_api_key,
        target_total_seconds=args.target_total_seconds,
        min_segment_seconds=args.min_segment_seconds,
        max_segment_seconds=args.max_segment_seconds,
        segments=args.segments,
        clip_seconds=args.clip_seconds,
        fps=args.fps,
        height=args.height,
        width=args.width,
        image_height=args.image_height,
        image_width=args.image_width,
        image_steps=args.image_steps,
        video_steps=args.video_steps,
        image_device=args.image_device,
        video_device=args.video_device,
        video_usp_devices=args.video_usp_devices,
        video_usp_master_port=args.video_usp_master_port,
        seed=args.seed,
        image_workers=args.image_workers,
        video_workers=args.video_workers,
        audio_workers=args.audio_workers,
        skip_image=args.skip_image,
        skip_video=args.skip_video,
        skip_audio=args.skip_audio,
        overwrite=args.overwrite,
        prompt_wav=Path(args.prompt_wav),
        prompt_text=args.prompt_text,
        cosyvoice_python=Path(args.cosyvoice_python) if args.cosyvoice_python else None,
        tangshi300_path=Path(args.tangshi300_path) if args.tangshi300_path else None,
    )
    config.qwen_image_model = Path(args.qwen_image_model)
    config.image_backend = args.image_backend
    config.bernini_image_config = Path(args.bernini_image_config)
    config.video_backend = args.video_backend
    config.bernini_root = Path(args.bernini_root)
    config.bernini_python = Path(args.bernini_python) if args.bernini_python else None
    config.bernini_config = Path(args.bernini_config)
    config.bernini_r_config = Path(args.bernini_r_config)
    config.bernini_max_frames = int(args.bernini_max_frames)
    config.bernini_use_prompt_enhancer = bool(args.bernini_use_prompt_enhancer)
    config.hunyuanvideo_root = Path(args.hunyuanvideo_root)
    config.hunyuanvideo_python = Path(args.hunyuanvideo_python) if args.hunyuanvideo_python else None
    config.hunyuanvideo_model = Path(args.hunyuanvideo_model)
    config.hunyuanvideo_max_frames = int(args.hunyuanvideo_max_frames)
    config.hunyuanvideo_dtype = str(args.hunyuanvideo_dtype)
    config.wan_i2v_model = Path(args.wan_i2v_model)
    config.cosyvoice_model = Path(args.cosyvoice_model)
    config.cosyvoice_repo = Path(args.cosyvoice_repo)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ancient-poetry teaching multi-agent inference pipeline.")
    parser.add_argument("--poem", default=None, help="Poem title, text, author, or a full poem block to explain.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT / "outputs" / "poetry_teaching"))
    parser.add_argument("--qwen35-api-base", default=os.environ.get("QWEN35_API_BASE", DEFAULT_QWEN35_API_BASE))
    parser.add_argument("--qwen35-model", default=os.environ.get("QWEN35_MODEL", DEFAULT_QWEN35_MODEL))
    parser.add_argument("--llm-backend", default=os.environ.get("LLM_BACKEND", "qwen35"), choices=["qwen35", "openai"], help="Which planner LLM to use. 'qwen35' keeps the self-hosted vLLM Qwen3.5 endpoint; 'openai' talks to an OpenAI-compatible endpoint (e.g. litellm doubao-seed-2.0-pro).")
    parser.add_argument("--openai-api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--openai-model", default=os.environ.get("OPENAI_MODEL", "doubao-seed-2.0-pro"))
    parser.add_argument("--qwen35-api-key", default=os.environ.get("QWEN35_API_KEY", "qwen3.5-27b-local"))
    parser.add_argument("--tangshi300-path", default=os.environ.get("TANGSHI300_PATH", str(DEFAULT_TANGSHI300_PATH)), help="Tangshi300 jsonl path used as Qwen3.5 planning reference. Empty disables it.")
    parser.add_argument("--qwen-image-model", default=os.environ.get("QWEN_IMAGE_MODEL", str(DEFAULT_QWEN_IMAGE_MODEL)))
    parser.add_argument("--image-backend", default=os.environ.get("IMAGE_BACKEND", "auto"), choices=["auto", "qwen_image", "bernini_t2i"])
    parser.add_argument("--bernini-image-config", default=os.environ.get("BERNINI_IMAGE_CONFIG", str(DEFAULT_BERNINI_CONFIG)))
    parser.add_argument("--video-backend", default=os.environ.get("VIDEO_BACKEND", "wan"), choices=["wan", "bernini", "bernini_r", "hunyuan"], help="Which video generator to use. 'wan' keeps the default diffsynth WanI2V pipeline; 'bernini' / 'bernini_r' route to the Bernini backends in vgen.video_backends; 'hunyuan' routes to HunyuanVideoGenerator.")
    parser.add_argument("--bernini-root", default=os.environ.get("BERNINI_ROOT", str(DEFAULT_BERNINI_ROOT)))
    parser.add_argument("--bernini-python", default=os.environ.get("BERNINI_PYTHON", str(DEFAULT_BERNINI_PYTHON)), help="Python executable of the standalone Bernini conda env (needed when --video-backend is bernini or bernini_r). Empty falls back to sys.executable.")
    parser.add_argument("--bernini-config", default=os.environ.get("BERNINI_CONFIG", str(DEFAULT_BERNINI_CONFIG)), help="Bernini-Diffusers weights directory (used by BerniniVideoGenerator).")
    parser.add_argument("--bernini-r-config", default=os.environ.get("BERNINI_R_CONFIG", str(DEFAULT_BERNINI_R_CONFIG)), help="Bernini-R-Diffusers weights directory (used by BerniniRVideoGenerator).")
    parser.add_argument("--bernini-max-frames", type=int, default=int(os.environ.get("BERNINI_MAX_FRAMES", "81")))
    parser.add_argument("--bernini-use-prompt-enhancer", action="store_true", help="Enable Bernini's --use_pe prompt enhancer.")
    parser.add_argument("--hunyuanvideo-root", default=os.environ.get("HUNYUANVIDEO_ROOT", str(DEFAULT_HUNYUANVIDEO_ROOT)))
    parser.add_argument("--hunyuanvideo-python", default=os.environ.get("HUNYUANVIDEO_PYTHON", str(DEFAULT_HUNYUANVIDEO_PYTHON)))
    parser.add_argument("--hunyuanvideo-model", default=os.environ.get("HUNYUANVIDEO_MODEL", str(DEFAULT_HUNYUANVIDEO_MODEL)))
    parser.add_argument("--hunyuanvideo-max-frames", type=int, default=int(os.environ.get("HUNYUANVIDEO_MAX_FRAMES", "121")))
    parser.add_argument("--hunyuanvideo-dtype", default=os.environ.get("HUNYUANVIDEO_DTYPE", "bf16"))
    parser.add_argument("--wan-i2v-model", default=os.environ.get("WAN_VACE_MODEL", str(DEFAULT_WAN_VACE_MODEL)))
    parser.add_argument("--cosyvoice-model", default=os.environ.get("COSYVOICE_MODEL", str(DEFAULT_COSYVOICE_MODEL)))
    parser.add_argument("--cosyvoice-repo", default=str(DEFAULT_COSYVOICE_REPO))
    parser.add_argument("--cosyvoice-python", default=os.environ.get("COSYVOICE_PYTHON_BIN", ""), help="Python executable used for CosyVoice audio generation. Empty keeps in-process generation.")
    parser.add_argument("--prompt-wav", default=os.environ.get("PROMPT_WAV", str(DEFAULT_COSYVOICE_PROMPT_WAV)), help="Reference voice wav required by CosyVoice zero-shot inference.")
    parser.add_argument("--prompt-text", default=os.environ.get("PROMPT_TEXT", DEFAULT_COSYVOICE_PROMPT_TEXT))
    parser.add_argument("--target-total-seconds", type=int, default=DEFAULT_TARGET_TOTAL_SECONDS)
    parser.add_argument("--min-segment-seconds", type=int, default=DEFAULT_MIN_SEGMENT_SECONDS)
    parser.add_argument("--max-segment-seconds", type=int, default=DEFAULT_MAX_SEGMENT_SECONDS)
    parser.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS)
    parser.add_argument("--clip-seconds", type=int, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-steps", type=int, default=30)
    parser.add_argument("--video-steps", type=int, default=35)
    parser.add_argument("--image-device", default=os.environ.get("IMAGE_DEVICE", "cuda:1"))
    parser.add_argument("--video-device", default=os.environ.get("VIDEO_DEVICE", "cuda:2"))
    parser.add_argument("--video-usp-devices", default=os.environ.get("VIDEO_USP_DEVICES", ""), help="Comma-separated GPU ids for Wan USP multi-GPU inference, for example: 1,2,3,4. Empty keeps single-GPU video inference.")
    parser.add_argument("--video-usp-master-port", type=int, default=int(os.environ.get("VIDEO_USP_MASTER_PORT", "29511")))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-workers", type=int, default=1)
    parser.add_argument("--video-workers", type=int, default=1)
    parser.add_argument("--audio-workers", type=int, default=1)
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wan-usp-worker-request", default=None)
    parser.add_argument("--cosyvoice-worker-request", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wan_usp_worker_request:
        run_wan_usp_worker(Path(args.wan_usp_worker_request))
        return
    if args.cosyvoice_worker_request:
        run_cosyvoice_worker(Path(args.cosyvoice_worker_request))
        return
    if not args.poem:
        raise ValueError("--poem is required unless a worker request is used.")
    result = PoetryTeachingPipeline(build_config(args)).run(args.poem, run_name=args.run_name)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, cls=PlanEncoder))


if __name__ == "__main__":
    main()
