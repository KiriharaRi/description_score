# Description fMRI Brain-Region Pipeline

这个仓库当前的 `brain_region_pipeline` 是一个面向 movie-fMRI encoding 的脑区特异性特征构造流程。主线已经收敛为：

```text
atlas labels + target region
  -> Meta Prompt 生成 region-level module prompt 和 dimensions
  -> 外部 dense description 逐段打分
  -> TR-aligned feature rows
  -> 暂保留的 ROI-level encoding
```

当前不再维护旧的 `make-module-pool` / `build-features` 视频切片、clip annotation、description embedding 流程。默认输入是外部已经生成好的 dense description。

## 当前状态

核心流程：

1. `make-module-prompt`
   - 根据 Schaefer atlas label space 和目标脑区生成一个 region-level module prompt。
   - prompt 内包含 functional hypothesis、simulation prompt、selection rules 和可打分 dimensions。
   - 当前 MVP 默认目标脑区是 `vmPFC`。
2. `score-descriptions`
   - 读取外部 timestamped dense description。
   - 用上一步的 module prompt 对每个 segment 打分。
   - 输出 segment-level scores 和 TR-aligned feature rows。
3. `encode`
   - 暂时保留为下游接口。
   - 从 train/test feature 目录读取 `tr_features.jsonl`。
   - 使用同一个 `module_prompt.json` 里的 `selection_rules` 做 ROI mapping。
   - 具体 encoding 实验设计等 scoring 流程稳定后再完善。

## 环境准备

本项目使用 `uv` 管理 Python 环境。

```bash
uv sync
```

如果要运行 `encode`，需要额外安装 encoding 依赖：

```bash
uv sync --extra encoding
```

Python 版本要求见 `pyproject.toml`：

```text
requires-python = ">=3.14"
```

需要配置 Gemini API key：

```bash
export GEMINI_API_KEY="..."
# 或
export GOOGLE_API_KEY="..."
```

如果使用自定义 Gemini 网关：

```bash
export GEMINI_BASE_URL="..."
```

## Quickstart

### 1. 生成 module prompt

```bash
uv run python -m brain_region_pipeline make-module-prompt \
  --atlas-labels test_data/Schaefer2018_1000Parcels_17Networks_order\ \(1\).txt \
  --target-region vmPFC \
  --output-file vmpfc_module_prompt.json
```

### 2. 对已有 dense description 打分

```bash
uv run python -m brain_region_pipeline score-descriptions \
  --descriptions segment_004.md \
  --module-prompt vmpfc_module_prompt.json \
  --output-dir vmPFC_demo \
  --tr-s 1.49 \
  --alignment overlap_weighted
```

典型输出：

```text
vmPFC_demo/
  segment_module_scores.jsonl
  tr_features.jsonl
  tr_descriptions_readable.jsonl
  scoring_metadata.json
```

### 3. 暂保留的 encoding

```bash
uv run python -m brain_region_pipeline encode \
  --train-dir <train_root> \
  --test-dir <test_root> \
  --fmri-h5 <bold.h5> \
  --module-prompt vmpfc_module_prompt.json \
  --atlas-labels <labels.txt> \
  --output-dir <encoding_output_dir>
```

注意：`encode` 当前只是和新 workflow 对齐了输入契约。后续应在 scoring schema、feature 分组和实验设计稳定后继续完善。

## CLI 参数

### `make-module-prompt`

```bash
uv run python -m brain_region_pipeline make-module-prompt \
  --atlas-labels <labels.txt> \
  --target-region <region> \
  --output-file <module_prompt.json> \
  --model gemini-3.1-pro-preview
```

参数：

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--atlas-labels` | 是 | 无 | Schaefer atlas label file |
| `--target-region` | 否 | `vmPFC` | 目标脑区名称 |
| `--output-file` | 是 | 无 | 输出的 `module_prompt.json` 路径 |
| `--model` | 否 | `gemini-3.1-pro-preview` | Gemini generation model |

设计约束：

- 单个 target region 只生成一个 region-level module。
- 不把 vmPFC 这类脑区拆成多个 modules。
- 功能差异应该体现在 `dimensions` 中。
- Meta Prompt 不提前写死 vmPFC 维度，让 LLM 根据 atlas label space 和目标脑区自行推理。

### `score-descriptions`

```bash
uv run python -m brain_region_pipeline score-descriptions \
  --descriptions <description.txt> \
  --module-prompt <module_prompt.json> \
  --output-dir <run_output_dir> \
  --model gemini-3.1-pro-preview \
  --tr-s 1.49 \
  --alignment overlap_weighted
```

参数：

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--descriptions` | 是 | 无 | Timestamped dense description 文本 |
| `--module-prompt` | 是 | 无 | `make-module-prompt` 生成的 JSON |
| `--output-dir` | 是 | 无 | scoring 输出目录 |
| `--model` | 否 | `gemini-3.1-pro-preview` | Gemini generation model |
| `--tr-s` | 否 | `1.49` | TR 秒数 |
| `--total-trs` | 否 | 自动从 segment end time 推断 | 显式覆盖输出 TR 数 |
| `--alignment` | 否 | `overlap_weighted` | 可选 `overlap_weighted` 或 `repeat` |

### `encode`

```bash
uv run python -m brain_region_pipeline encode \
  --train-dir <train_root> \
  --test-dir <test_root> \
  --fmri-h5 <bold.h5> \
  --module-prompt <module_prompt.json> \
  --atlas-labels <labels.txt> \
  --output-dir <encoding_output_dir> \
  --lag-trs 3 \
  --gap-trs 10
```

参数：

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--train-dir` | 是 | 无 | train run 输出根目录 |
| `--test-dir` | 是 | 无 | test run 输出根目录 |
| `--fmri-h5` | 是 | 无 | HDF5 BOLD 文件 |
| `--module-prompt` | 是 | 无 | 用于 ROI mapping 的 `module_prompt.json` |
| `--atlas-labels` | 是 | 无 | Schaefer atlas label file |
| `--output-dir` | 是 | 无 | encoding 输出目录 |
| `--lag-trs` | 否 | `3` | HRF lag，以 TR 为单位 |
| `--gap-trs` | 否 | `10` | `TimeSeriesSplit` 内部 CV gap |

## Dense Description 输入格式

当前 parser 支持空行分隔的 timestamped blocks：

```text
00:00 - 00:01  In a kitchen area with shelves of food, Ross stands holding a white cordless phone.

00:01 - 00:09  Ross raises the phone to chest level and says, "right..."
He promises to show Chandler how well he can flirt.
```

规则：

- 每个 block 的第一行必须以 `MM:SS - MM:SS` 或 `HH:MM:SS - HH:MM:SS` 开头。
- 后续行会拼接到同一个 segment 的 `description` 中。
- 空行表示新 segment。
- 时间戳会转成 `start_s` 和 `end_s`。

也支持带 Markdown segment header 的输入：

```md
# Segment 4
**Time Range:** 00:09:11-00:11:04

00:00 - 00:01  In a kitchen area with shelves of food, Ross stands holding a white cordless phone.
```

注意：当前 `**Time Range:**` 只作为元信息跳过，不会自动加到 block 内局部时间戳上。如果外部 description 的每个大段都从 `00:00` 重新计时，需要先把时间转换为全片绝对时间，或后续扩展 `description_io.py`。

## 关键数据契约

### `module_prompt.json`

顶层字段：

```json
{
  "version": "module_prompt_v1",
  "source_model": "gemini-3.1-pro-preview",
  "modules": []
}
```

当前单脑区 demo 下，`modules` 应只有一个元素。每个 module 包含：

| 字段 | 说明 |
| --- | --- |
| `module_id` | 短 ASCII `snake_case` id |
| `target_region` | 目标脑区名称 |
| `display_name` | 人类可读名称 |
| `functional_hypothesis` | 该脑区在 movie viewing 中追踪什么 |
| `simulation_prompt` | scoring 阶段使用的模块提示 |
| `dimensions` | 可从 description 评分的维度 |
| `selection_rules` | atlas parcel selection rules |

每个 dimension 包含：

| 字段 | 说明 |
| --- | --- |
| `dimension_id` | 短 ASCII `snake_case` id |
| `name` | 人类可读名称 |
| `definition` | 维度定义 |
| `score_min` | 最低分 |
| `score_max` | 最高分 |
| `low_anchor` | 低分锚点 |
| `high_anchor` | 高分锚点 |

`selection_rules` 语义：

- 单条 rule 内，非空字段按交集匹配。
- 多条 rule 之间，按并集扩张 parcel。
- 当前支持字段：`networks`、`sub_regions`、`hemispheres`。

### `segment_module_scores.jsonl`

一行一个 description segment：

```json
{
  "start_s": 0.0,
  "end_s": 1.0,
  "description": "...",
  "module_scores": {
    "vmpfc": {
      "dimension_scores": {
        "affective_valence": 0.3
      },
      "rationale": "..."
    }
  }
}
```

### `tr_features.jsonl`

一行一个 TR：

```json
{
  "tr_index": 0,
  "tr_start_s": 0.0,
  "tr_end_s": 1.49,
  "module_descriptions": {
    "vmpfc": "..."
  },
  "feature_vector": [0.3, 0.8, 0.4],
  "weights": {
    "seg_0": 1.0
  }
}
```

说明：

- `feature_vector` 按 `ModulePromptPool.ordered_feature_keys()` 的顺序拼接。
- `module_descriptions` 保存 rationale，主要用于人工检查，不是 feature 来源。
- `weights` 记录当前 TR 和 segment 的对应关系。

### `scoring_metadata.json`

记录本次 scoring 的关键上下文：

```json
{
  "n_segments": 39,
  "n_trs": 76,
  "tr_s": 1.49,
  "alignment": "overlap_weighted",
  "ordered_feature_keys": [
    {
      "module_id": "vmpfc",
      "dimension_id": "affective_valence"
    }
  ]
}
```

## TR Alignment 策略

### `overlap_weighted`

默认策略。

- 计算每个 TR 与所有 scored segments 的时间重叠比例。
- 如果有重叠，对 score vectors 做加权平均。
- 如果没有任何重叠，回退到最近 segment。
- `module_descriptions` 取 best-overlap segment 的 rationale，只用于人工检查。

### `repeat`

- 每个 TR 直接取重叠最大的 segment。
- 如果没有重叠，回退到最近 segment。

## Encoding 输入目录

`encode` 期望 train/test 目录下每个 run 一个子目录：

```text
<train_root>/
  <episode_id>/
    tr_features.jsonl

<test_root>/
  <episode_id>/
    tr_features.jsonl
```

约束：

- `<episode_id>` 必须能唯一匹配 HDF5 中以 `task-{episode_id}` 结尾的 dataset key。
- train 和 test 的 `run_key` 不能重叠。
- 所有 run 的 feature dimension 必须一致。
- train/test 的 BOLD parcel dimension 必须一致。

输出：

```text
<encoding_output_dir>/
  encoding_results.json
  training_metadata.json
```

## 包结构

```text
brain_region_pipeline/
  __main__.py          # python -m brain_region_pipeline
  cli.py               # CLI 参数解析和 stage 分发
  runner.py            # stage orchestration、日志、依赖注入、落盘
  config.py            # ModulePromptConfig / ScoreDescriptionsConfig / EncodeConfig
  models.py            # 稳定数据契约
  atlas.py             # Schaefer labels 解析和 selection rules 展开
  module_prompt.py     # Meta Prompt 生成 module_prompt.json
  description_io.py    # 外部 dense description 解析
  module_scorer.py     # LLM segment scoring
  score_aligner.py     # segment scores -> TR features
  tr_output.py         # TR 可读检查文件输出
  encoding_eval.py     # 暂保留的 encoding 下游
  genai.py             # Google GenAI structured JSON helper
  io_utils.py          # JSON / JSONL helpers
```

## 开发约定

- 默认只维护 `make-module-prompt` 和 `score-descriptions` 主线。
- 不要在没有明确需求时重新加入旧的视频切片、clip annotation、embedding cache 或本地 LLM embedding backend。
- `cli.py` 只放参数解析和 stage dispatch，不放业务逻辑。
- `runner.py` 只做 orchestration；复杂逻辑继续下沉到职责模块。
- 不要静默修改输出 JSONL 字段名。
- 不要静默修改 module 顺序或 `ordered_feature_keys` 拼接顺序。
- 如果改输出契约，必须同步更新测试和 README。
- 新增 description 输入格式时，优先扩展 `description_io.py`。
- 新增脑区时，优先复用 `make-module-prompt --target-region <region>`，不要为每个脑区复制 scorer。

## 验证

推荐运行：

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
uv run python -m compileall brain_region_pipeline tests
```

检查 CLI：

```bash
uv run python -m brain_region_pipeline
uv run python -m brain_region_pipeline make-module-prompt --help
uv run python -m brain_region_pipeline score-descriptions --help
uv run python -m brain_region_pipeline encode --help
```

当前旧命令应不可用：

```bash
uv run python -m brain_region_pipeline make-module-pool --help
uv run python -m brain_region_pipeline build-features --help
```

如果旧命令重新出现，说明主线边界被意外放宽，需要重新审查。
