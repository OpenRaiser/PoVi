# PoVi VGen

方法：Director Graph（知识规划）→ 五 Agent 叙事/分镜 → 图 / TTS / 视频合成。

## Setup

```bash
cp .env.example .env   # LLM_*、模型路径、可选 COSYVOICE_*
```

## Pipeline

### 1) 生成 Director Graph（调 LLM）

```bash
export LLM_BASE_URL='https://your-openai-compatible-endpoint/v1'
export LLM_MODEL='your-model'
export LLM_API_KEY='your-key'

python scripts/build_poetry_director_graphs.py \
  --corpus data/examples/qingyu_an_yuanxi.json \
  --index 1 \
  --overwrite
```

输出默认在 `outputs/director_graphs/`（gitignore，勿提交）。这一步会顺带写 `source_packet.json` / `locked_graph_fields.json`，不必先跑 prepare。

可选：只检查本地证据包、不调 API 时加 `--prepare-only`（调试用）。

### 2) 媒体实现

同一套 `director_graph_pipeline`；按后端选脚本（Veo / Seedance / MiniMax）：

```bash
DIRECTOR_GRAPH=outputs/director_graphs/.../director_graph.json \
RUN_NAME=demo \
bash scripts/run_director_graph_veo31.sh --dry-run
```

## 新诗词

写一条 JSON（或 `{"records":[...]}`），字段至少：`index`, `poem_id`, `title`, `author`, `dynasty`, `form`, `content`。示例：`data/examples/qingyu_an_yuanxi.json`。

## Layout

```text
vgen/            论文方法代码（graph builder + 五 agent + media）
scripts/         入口（build + director_graph_{veo,seedance,minimax}）
schema/          schema + rhetoric maps + 一条示例 director_graph
data/examples/   示例输入诗词
assets/ tests/
```

`outputs/director_graphs/` 是**跑出来的产物**（prepare / LLM 生成），不是方法代码，默认 gitignore，不必进仓库。  
`schema/examples/*.director_graph.json` 只是一条可 dry-run 的示例图，和批量输出目录不是一回事。

`vgen/poetry_teaching.py` 保留是因为 media 实现（TTS / 合成等）被 `director_graph_pipeline` 复用；旧 teaching / baseline / planloop / Wan-VACE 入口已删。
