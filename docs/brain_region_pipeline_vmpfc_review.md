# brain_region_pipeline vmPFC 设计审查材料

更新日期：2026-04-29

本文档整理当前 `brain_region_pipeline` 的主线设计。旧的 `make-module-pool` / `build-features` 视频切片、clip annotation、description embedding 流程已经退出主包；当前代码只保留外部 dense description 驱动的 brain-region prompt workflow，以及一个待后续完善的 encoding 下游接口。

## 1. 当前目标

`brain_region_pipeline` 的目标是把目标脑区的功能假设转成可审计、可落盘、可对齐到 TR 的数值特征。当前 vmPFC demo 不重新看视频生成描述，而是复用外部 dense description：

1. `make-module-prompt`：根据 atlas label space 和目标脑区生成一个 region-level module prompt。
2. `score-descriptions`：用该 prompt 对外部 dense description 逐段打分，并输出 TR-aligned features。
3. `encode`：暂时保留为下游接口，读取 scored features 和同一个 `module_prompt.json` 的 selection rules 做 ROI-level encoding。具体 encoding 方案等 scoring 流程稳定后再细化。

## 2. 环境与真实入口

项目路径：

```text
/Users/kiriharari/codex_paper/Description预测高级fMRI
```

Python 环境：

```bash
uv sync
uv sync --extra encoding
```

真实入口：

```bash
uv run python -m brain_region_pipeline
```

入口链路：

```text
brain_region_pipeline/__main__.py
  -> brain_region_pipeline/cli.py:main
  -> brain_region_pipeline/runner.py
```

当前 CLI commands：

```text
make-module-prompt    atlas labels + target region -> module_prompt.json
score-descriptions    external dense descriptions + module_prompt.json -> scored TR features
encode                provisional downstream encoding from scored feature dirs
```

## 3. 当前 workflow

### 3.1 make-module-prompt

输入：

```text
Schaefer atlas label file
target_region, currently vmPFC
```

输出：

```text
module_prompt.json
```

核心约束：

- 单个 `target_region` 只生成一个 region-level module。
- 不把 vmPFC 再拆成多个 module。
- vmPFC 内部功能差异通过 `dimensions` 表达。
- module prompt 内包含 `module_id`、`target_region`、`display_name`、`functional_hypothesis`、`simulation_prompt`、`dimensions`、`selection_rules`。
- `module_prompt.py` 的 response schema 把 `modules` 限制为 `maxItems=1`。
- prompt 明确要求 LLM 自行推理 dimensions，不把 vmPFC 维度答案提前写死。

### 3.2 score-descriptions

输入：

```text
timestamped dense description text
module_prompt.json
```

输出目录：

```text
<run_output_dir>/
  segment_module_scores.jsonl
  tr_features.jsonl
  tr_descriptions_readable.jsonl
  scoring_metadata.json
```

处理流程：

1. `description_io.py` 读取外部 dense description。
2. `module_scorer.py` 根据 module prompt 动态构建 JSON schema。
3. LLM 对每个 description segment 输出每个 dimension 的 numeric score 和 rationale。
4. `score_aligner.py` 把 segment-level scores 对齐到 TR。
5. `runner.py` 写出 segment scores、TR features、可读检查文件和 metadata。

当前支持的 description 格式：

```text
00:00 - 00:01  In a kitchen area with shelves of food, Ross stands holding a white cordless phone.

00:01 - 00:09  Ross raises the phone to chest level and says, "right..."
```

也支持带 Markdown segment header 的文本，但当前 `**Time Range:**` 只作为元信息跳过，不会自动加到每条局部时间戳上。

### 3.3 TR alignment

当前 `score-descriptions` 支持两种 alignment：

```text
overlap_weighted
repeat
```

默认是 `overlap_weighted`：

- 对每个 TR，计算它与所有 scored segments 的时间重叠比例。
- 如果有重叠，按重叠比例对 score vectors 加权平均。
- 如果没有任何重叠，回退到最近 segment。
- `feature_vector` 是数值特征。
- `module_descriptions` 保存 best segment 的 rationale，主要用于人工检查，不是 feature 拼接来源。

## 4. 暂保留的 encoding

`encode` 从 train/test feature 目录读取每个 run 的 `tr_features.jsonl`，并和 HDF5 BOLD 数据对齐。

期望目录结构：

```text
<train_root>/
  <episode_id>/
    tr_features.jsonl

<test_root>/
  <episode_id>/
    tr_features.jsonl
```

当前命令：

```bash
uv run python -m brain_region_pipeline encode \
  --train-dir <train_root> \
  --test-dir <test_root> \
  --fmri-h5 <bold.h5> \
  --module-prompt <module_prompt.json> \
  --atlas-labels <labels.txt> \
  --output-dir <encoding_output_dir>
```

当前 encoding 逻辑：

- 每个 run 读取 `feature_vector` 形成 feature matrix。
- 读取对应 BOLD。
- 应用 HRF lag，默认 `lag_trs=3`。
- train runs 拼接为 pooled train，test runs 拼接为 pooled test。
- train/test run keys 不允许重叠。
- ROI mapping 来自 `module_prompt.json` 的 `selection_rules`。
- ROI-level encoding 仍由 `test_pipeline.encoding_model.fit_ridge_encoding()` 完成。

注意：这里目前只是把旧 `--module-pool` 接口改成与新 workflow 一致的 `--module-prompt`。更细的 encoding 实验设计需要等 scoring schema 和输出稳定后再完善。

## 5. 模块职责分层

当前主线模块：

| 模块 | 职责 |
| --- | --- |
| `cli.py` | CLI 参数解析和 stage 分发 |
| `runner.py` | stage orchestration、日志、依赖注入、落盘 |
| `config.py` | 分阶段 config dataclass |
| `models.py` | 稳定数据契约 |
| `atlas.py` | 解析 Schaefer labels、展开 module prompt selection rules |
| `module_prompt.py` | 生成、保存、加载 region module prompt |
| `description_io.py` | 读取外部 dense description |
| `module_scorer.py` | 根据 description 和 module prompt 生成 segment-level scores |
| `score_aligner.py` | 把 segment scores 对齐到 TR features |
| `tr_output.py` | 保存 TR 级可读检查文件 |
| `encoding_eval.py` | 暂保留的目录驱动 cross-run encoding |

## 6. 当前 vmPFC demo 设计

当前 vmPFC prompt 文件：

```text
vmpfc_module_prompt.json
```

当前 module：

```text
module_id: vmPFC_affective_evaluation
target_region: vmPFC
display_name: Ventromedial Prefrontal Cortex
```

当前 demo 输出：

```text
vmPFC_demo/
  segment_module_scores.jsonl
  tr_features.jsonl
  tr_descriptions_readable.jsonl
  scoring_metadata.json
```

`scoring_metadata.json` 中记录：

```text
n_segments = 39
n_trs = 76
tr_s = 1.49
alignment = overlap_weighted
ordered_feature_keys =
  vmPFC_affective_evaluation / affective_valence
  vmPFC_affective_evaluation / social_moral_evaluation
  vmPFC_affective_evaluation / subjective_significance
```

## 7. 主要风险与注意事项

### 7.1 vmPFC 维度是否过宽

当前三个维度都和 vmPFC 功能相关，但可能存在重叠：

- `affective_valence` 与 `subjective_significance` 在强情绪片段中可能高度相关。
- `social_moral_evaluation` 与 `subjective_significance` 在社交冲突或亲密关系场景中也可能相关。

需要审查这些维度是否足够可分，还是会产生强共线性。

### 7.2 LLM scoring 的一致性

当前 score 是 LLM 从文本描述中推断的数值分数。风险包括：

- 不同 segment 的 score calibration 可能不稳定。
- 同一分数范围内，LLM 未必严格使用线性尺度。
- rationale 可读，但不保证数值分数跨 run 完全一致。
- 当前 `score-descriptions` 路径没有 hash 校验，科研 demo 阶段可接受，但正式实验时需要记录 prompt、model、输入文件版本和生成参数。

### 7.3 dense description 输入的时间语义

当前 parser 支持 Markdown `**Time Range:**` 元信息，但不会把该起始时间加到 block 内局部时间戳上。如果外部 description 是按大 segment 分块、块内时间从 00:00 重新开始，则需要确认时间戳是否已经转换为全片绝对时间。

### 7.4 TR 对齐解释

`overlap_weighted` 下，`feature_vector` 可能是多个 segment scores 的加权平均，但 `module_descriptions` 只取 best segment 的 rationale。这对人工检查有帮助，但不能把 `module_descriptions` 当作加权 feature 来源。

## 8. 建议审查问题清单

1. 当前 vmPFC 的三维表示是否是一个可 defensible 的 brain-region-specific feature schema？
2. 这三个维度是否能从普通 movie dense description 中稳定判断，还是需要更明确的 annotation instruction？
3. 当前分数范围是否应统一标准化，避免 Ridge encoding 中尺度差异影响解释？
4. 具体情绪类别是否应该作为辅助 feature group 或 ablation，而不是替代当前 functional dimensions？
5. 当前 selection rules 是否过宽或过窄？
6. 当前 external dense description -> numeric dimensions 路径是否足以支撑初步 fMRI encoding demo？
7. 进入正式 encoding 前，是否需要先固定 prompt 版本、输入 description 版本和 scoring metadata？

## 9. 可供审查者参考的关键文件

```text
brain_region_pipeline/cli.py
brain_region_pipeline/runner.py
brain_region_pipeline/config.py
brain_region_pipeline/models.py
brain_region_pipeline/atlas.py
brain_region_pipeline/module_prompt.py
brain_region_pipeline/description_io.py
brain_region_pipeline/module_scorer.py
brain_region_pipeline/score_aligner.py
brain_region_pipeline/tr_output.py
brain_region_pipeline/encoding_eval.py
vmpfc_module_prompt.json
vmPFC_demo/scoring_metadata.json
vmPFC_demo/segment_module_scores.jsonl
vmPFC_demo/tr_features.jsonl
```
