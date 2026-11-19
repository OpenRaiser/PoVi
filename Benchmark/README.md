# PoVi-Bench

古诗词教学视频评测：论文指标 **S1–S6 / D1–D4 / R1–R7**（共 17 项）。对照表见 [`METRIC_ID_MAP.md`](METRIC_ID_MAP.md)。1–5 分归一化为 **score / 5**。

## Install

```bash
conda env create -f environment.yml
conda activate poetry-bench
pip install -e .
cp .env.example .env   # fill API keys as needed
python run.py doctor
```

## Data

```text
data/task_inputs/   # 120 samples (60 唐诗 + 60 宋词)
data/gold/          # 60 adjudicated gold annotations (subset of the 120)
```

有 Gold 的样本才能做 grounded 评测；其余 60 条仅提供 `task_input`。跑评测时自行准备 `submission.json`（及媒体）；`runs/` 等产物请放本地，勿提交。

## Quick start

```bash
# 例：有 Gold 的样本
SAMPLE=data/task_inputs/tangshi_300/038.json
GOLD=data/gold/tangshi_300/038.json

python run.py validate \
  --task-input "$SAMPLE" \
  --gold "$GOLD" \
  --submission path/to/submission.json

python run.py evaluate \
  --task-input "$SAMPLE" \
  --gold "$GOLD" \
  --submission path/to/submission.json \
  --runs-dir runs
```

`evaluate` = `prepare` → `machine-evaluate`。人评：`human-export` / `human-import` / `finalize`。

## Tracks

| Track | Metrics |
|---|---|
| `text_planning` | S1–S6, D1–D2 |
| `visual_generation` | R1–R5 |
| `audio_generation` | gates only |
| `audio_visual_generation` | R1–R5, D3–D4, R6–R7 |
| `full_task_teaching` | all 17 |

## Layout

```text
run.py  bench/  data/  METRIC_ID_MAP.md
```

Local only (gitignored): `input/`, `runs/`, `output(s)/`, `.env`.
