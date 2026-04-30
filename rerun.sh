#!/usr/bin/env bash
set -euo pipefail

# 先确认当前 shell 里已经有 GEMINI_API_KEY 或 GOOGLE_API_KEY。
# 这个脚本只跑当前主线：module prompt -> description scoring。

ATLAS_LABELS="${ATLAS_LABELS:-$PWD/test_data/Schaefer2018_1000Parcels_17Networks_order (1).txt}"
DESCRIPTIONS="${DESCRIPTIONS:-$PWD/segment_004.md}"
MODULE_PROMPT="${MODULE_PROMPT:-$PWD/vmpfc_module_prompt.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/vmPFC_demo}"
TARGET_REGION="${TARGET_REGION:-vmPFC}"
MODEL="${MODEL:-gemini-3.1-pro-preview}"

if [[ "${REGENERATE_PROMPT:-0}" == "1" ]]; then
  uv run python -m brain_region_pipeline make-module-prompt \
    --atlas-labels "$ATLAS_LABELS" \
    --target-region "$TARGET_REGION" \
    --output-file "$MODULE_PROMPT" \
    --model "$MODEL"
fi

uv run python -m brain_region_pipeline score-descriptions \
  --descriptions "$DESCRIPTIONS" \
  --module-prompt "$MODULE_PROMPT" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --tr-s 1.49 \
  --alignment overlap_weighted
