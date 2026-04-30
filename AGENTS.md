# AGENTS.md

## 1. 仓库协作约定

- 始终使用中文沟通。
- Python 环境统一使用 `uv` 管理，不要直接依赖系统 `pip`。
- 开始改动前，先确认真实入口、真实输出文件、真实数据契约，再下结论。
- 默认做最小必要修改；不要顺手做无关重构。
- 设计上保持高内聚、低耦合、职责单一，不要把大量逻辑塞进一个“万能类”或一个超长文件。
- 注释要解释设计意图、边界条件、兼容性约束和容易误解的逻辑；不要给显而易见的简单代码写废话注释。
- 如果需求只涉及 `brain_region_pipeline`，不要默认连带重构 `test_pipeline`；先确认共享边界再动。

## 2. `brain_region_pipeline` 的当前主线

`brain_region_pipeline` 当前只维护外部 dense description 驱动的 brain-region prompt workflow：

1. `make-module-prompt`
   根据 atlas label space 和目标脑区，用 Meta Prompt 生成一个 region-level module prompt，并在 prompt 内推理 simulation prompt、selection rules 和可打分维度。
2. `score-descriptions`
   读取外部已有 dense description，对每个时间段应用 module prompt，输出每个维度的推理分数，并对齐成 `tr_features.jsonl`。
3. `encode`
   暂时保留为下游接口，读取 scored `tr_features.jsonl` 和同一个 `module_prompt.json` 的 selection rules 做 ROI-level encoding。具体 encoding 细节等前面的打分流程稳定后再继续完善。

旧的 `make-module-pool` / `build-features` 视频切片、clip annotation、description embedding 路径已经退出主线，不应在没有明确需求时重新加入。

核心思想：

- 默认复用外部 dense description，不在本 pipeline 里重新看视频生成 description。
- pipeline 负责把脑区功能假设变成可审计的 module prompt、维度定义、逐段评分和 TR 特征。
- 单个 `target_region` 默认只生成一个 region-level module prompt；不要把 vmPFC 这类目标脑区再拆成多个 module，功能差异应体现在 `dimensions` 里。
- module prompt、segment score、TR features、encoding 结果是分阶段落盘的中间产物，而不是临时内存变量。
- 新 `score-descriptions` 路径当前不做 hash 校验；科研 demo 阶段优先保证结构清晰、能检查、能迭代。

## 3. 真实入口与运行方式

真正入口是：

- `python -m brain_region_pipeline`
- 对应文件：`brain_region_pipeline/__main__.py -> brain_region_pipeline/cli.py:main`

推荐环境命令：

```bash
uv sync
uv sync --extra encoding
```

当前 workflow：

```bash
uv run python -m brain_region_pipeline make-module-prompt \
  --atlas-labels <labels.txt> \
  --target-region vmPFC \
  --output-file <module_prompt.json>

uv run python -m brain_region_pipeline score-descriptions \
  --descriptions <description.txt> \
  --module-prompt <module_prompt.json> \
  --output-dir <run_output_dir>
```

暂保留的 encoding 入口：

```bash
uv run python -m brain_region_pipeline encode \
  --train-dir <train_root> \
  --test-dir <test_root> \
  --fmri-h5 <bold.h5> \
  --module-prompt <module_prompt.json> \
  --atlas-labels <labels.txt> \
  --output-dir <encoding_output_dir>
```

当前关键默认值：

- `target_region = vmPFC`
- `tr_s = 1.49`
- `alignment = overlap_weighted`
- `lag_trs = 3`
- `gap_trs = 10`

环境变量约定：

- Meta Prompt 生成与 description scoring 依赖 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`
- 如需自定义 Gemini 网关，可设置 `GEMINI_BASE_URL`

## 4. 包结构与职责分层

### 4.1 顶层编排

- `brain_region_pipeline/cli.py`
  只负责 CLI 参数解析与 stage 分发，不放业务实现。
- `brain_region_pipeline/runner.py`
  负责 stage orchestration、日志、依赖注入和落盘；复杂逻辑应继续下沉到专门模块。
- `brain_region_pipeline/config.py`
  定义分阶段 frozen config：`ModulePromptConfig`、`ScoreDescriptionsConfig`、`EncodeConfig`。
- `brain_region_pipeline/models.py`
  定义稳定数据契约：`SelectionRule`、`DimensionSpec`、`RegionModulePrompt`、`ModulePromptPool`、`DescriptionSegment`、`ModuleScoreResult`、`SegmentModuleScore`、`TRFeatureRow`。

### 4.2 module prompt + description scoring

- `brain_region_pipeline/atlas.py`
  负责解析 Schaefer label、汇总 label space、把 `ModulePromptPool` 的 selection rules 展开为 parcel index。
- `brain_region_pipeline/module_prompt.py`
  负责基于目标脑区和 atlas label space 生成单个 region-level module prompt、维度 schema、保存与加载。
- `brain_region_pipeline/description_io.py`
  负责读取外部已有 dense description。当前 MVP 支持空行分段的纯文本格式：`MM:SS - MM:SS  description`，也支持 `HH:MM:SS`。
- `brain_region_pipeline/module_scorer.py`
  负责把 `DescriptionSegment + ModulePromptPool` 交给 LLM，输出每个 module 的维度分数和 rationale。
- `brain_region_pipeline/score_aligner.py`
  负责把 segment-level module scores 对齐成 `tr_features.jsonl`。
- `brain_region_pipeline/tr_output.py`
  负责保存 TR 级可读检查文件。

### 4.3 暂保留 encoding

- `brain_region_pipeline/encoding_eval.py`
  负责从 train/test 目录发现 run、读取 `tr_features.jsonl`、拼接数据并按 `module_prompt.json` 的 selection rules 做 ROI encoding。

当前 encoding 仍复用：

- `test_pipeline.fmri_loader`
- `test_pipeline.encoding_model`

注意：

- `encode` 阶段只负责外层 cross-run train/test 组织。
- 内层 alpha 选择仍使用 `test_pipeline.encoding_model.fit_ridge_encoding()` 中的 `TimeSeriesSplit(gap=gap_trs)`。
- 不要把“外层按 run 切分”和“内层 RidgeCV 的时间序列 CV”混成一件事。

## 5. 关键数据契约与目录规范

### 5.1 `module_prompt.json`

单个 `target_region` 对应一个 region-level module prompt。关键字段：

- `module_id`
- `target_region`
- `display_name`
- `functional_hypothesis`
- `simulation_prompt`
- `dimensions`
- `selection_rules`

`dimensions` 的每一项应包含：

- `dimension_id`
- `name`
- `definition`
- `score_min`
- `score_max`
- `low_anchor`
- `high_anchor`

约定：

- `module_id` 和 `dimension_id` 使用短的 ASCII `snake_case`
- module 顺序有语义，后续 `feature_vector` 会按 module 顺序和 dimension 顺序拼接；当前单脑区 demo 下 `modules` 应只有一个元素
- intensity 类维度优先用 `0.0-1.0`
- 只有明确有方向性的 valence 类维度才用 `-1.0-1.0`
- `selection_rules` 语义：单条 rule 内按交集匹配，多条 rule 之间按并集扩张 parcel
- Meta Prompt 不要把 vmPFC 维度答案提前写死进 prompt；让 LLM 根据目标脑区和 atlas 信息自行推理 dimensions

### 5.2 外部 dense description 输入

当前 MVP 支持纯文本格式：

```text
00:00 - 00:01  In a kitchen area with shelves of food, Ross stands holding a white cordless phone.

00:01 - 00:09  Ross raises the phone to chest level and says, "right..."
```

也支持带 segment 元信息的 Markdown：

```md
# Segment 4
**Time Range:** 00:09:11-00:11:04

00:00 - 00:01  In a kitchen area with shelves of food, Ross stands holding a white cordless phone.
```

解析规则：

- 空行分隔 segment
- 每个 segment 第一行必须以 `MM:SS - MM:SS` 或 `HH:MM:SS - HH:MM:SS` 开头
- Markdown 标题行和 `**Time Range:** ...` 会被当作元信息跳过
- 时间戳会转成 `start_s` / `end_s`
- 当前默认使用文件内局部时间，不会把 `**Time Range:**` 的起始时间自动加到每条 segment 上
- 同一个 block 的后续行会拼接进同一个 `description`

### 5.3 `score-descriptions` 的核心输出

典型 run 目录结构：

```text
<run_output_dir>/
  segment_module_scores.jsonl
  tr_features.jsonl
  tr_descriptions_readable.jsonl
  scoring_metadata.json
```

字段约定：

- `segment_module_scores.jsonl`
  - 一行一个 description segment
  - 包含 `start_s`、`end_s`、`description`、`module_scores`
  - `module_scores` 下每个 module 包含 `dimension_scores` 和 `rationale`
- `tr_features.jsonl`
  - 一行一个 TR
  - `feature_vector` 按 `ModulePromptPool.ordered_feature_keys()` 拼接维度分数
  - `module_descriptions` 保存 rationale，主要用于人工检查
- `tr_descriptions_readable.jsonl`
  - 面向人工检查的 TR 可读版本
- `scoring_metadata.json`
  - 记录 `n_segments`、`n_trs`、`tr_s`、`alignment`、`ordered_feature_keys`

### 5.4 `encode` 的输入目录规范

`encode` 期望如下目录结构：

```text
<train_root>/
  <episode_id>/
    tr_features.jsonl

<test_root>/
  <episode_id>/
    tr_features.jsonl
```

关键要求：

- `episode_id` 必须能唯一匹配 HDF5 中以 `task-{episode_id}` 结尾的 dataset key
- train 和 test 的 `run_key` 不能重叠
- 所有 run 的 feature 维度必须一致
- train/test 的 parcel 维度必须一致

输出文件：

```text
<encoding_output_dir>/
  encoding_results.json
  training_metadata.json
```

说明：

- `encoding_results.json` 是按 `module_id` 存的 ROI-level 结果，不输出 `all_parcels` 聚合结果
- ROI mapping 来自 `module_prompt.json` 里的 `selection_rules`
- `training_metadata.json` 记录 train/test run 的解析结果、TR 数量、特征维度和参数

## 6. 修改 `brain_region_pipeline` 时的 guardrails

### 6.1 结构 guardrails

- 不要把多个 stage 揉回一个大函数或一个大类。
- 新功能优先放进对应职责模块，不要把业务细节继续堆进 `runner.py`。
- description scoring workflow 要保持 `description_io.py`、`module_prompt.py`、`module_scorer.py`、`score_aligner.py` 分工清楚。
- `make-module-prompt` 对单个目标脑区只生成一个 region-level module prompt；不要新增 `--max-modules` 这类会鼓励脑区内拆 module 的参数。
- 如果只是改 vmPFC demo，不要顺手扩成多脑区大框架；先让单脑区结果可检查。
- 不要在没有明确需求时重新加入视频切片、clip annotation、embedding cache 或本地 LLM embedding backend。

### 6.2 契约 guardrails

- 不要静默修改输出 JSONL 的字段名。
- 不要静默修改 module 顺序。
- 不要静默修改 `ordered_feature_keys` 的拼接顺序，否则特征列语义会变。
- 如果要变更输出契约，必须同步更新测试和文档。

### 6.3 扩展 guardrails

- 新增 description 输入格式时，优先扩展 `description_io.py`，不要把解析逻辑塞进 CLI 或 runner。
- 新增脑区时，优先复用 `make-module-prompt --target-region <region>`，不要为每个脑区复制一套 scorer。
- 新增 alignment 策略时，保持 `align_scores_to_trs()` 的 dispatch 方式清晰。
- 新增外部副作用或第三方调用时，优先通过 `PipelineDependencies` 注入，保持单元测试可替换。

## 7. 推荐验证方式

优先跑现有单测：

```bash
uv run python -m unittest tests.test_brain_region_description_workflow tests.test_brain_region_runner tests.test_brain_region_atlas
```

改 CLI 或 stage dispatch 后，至少检查：

```bash
uv run python -m brain_region_pipeline
uv run python -m brain_region_pipeline make-module-prompt --help
uv run python -m brain_region_pipeline score-descriptions --help
uv run python -m brain_region_pipeline encode --help
```

如果修改点涉及：

- module prompt / description scoring：优先补 `tests/test_brain_region_description_workflow.py`
- atlas 展开：优先补 `tests/test_brain_region_atlas.py`
- stage orchestration / CLI / encoding 参数：优先补 `tests/test_brain_region_runner.py`

## 8. 给后续 agent 的工作方式建议

- 先读真实入口，再读真实输出，再决定改哪里。
- 先确认改动属于 module prompt、description scoring、TR alignment，还是暂保留的 encoding。
- 当前 vmPFC demo 的核心入口是 `make-module-prompt` 和 `score-descriptions`；不要误以为必须先跑 `build-features`。
- 如果只是为了实验一次性验证而写脚本，验证结束后要考虑是否清理残留文件。
