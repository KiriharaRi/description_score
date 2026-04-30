"""LLM scoring of existing dense descriptions using region module prompts."""

from __future__ import annotations

from typing import Any

from .config import ScoreDescriptionsConfig
from .genai import generate_structured_json
from .models import (
    DescriptionSegment,
    ModulePromptPool,
    ModuleScoreResult,
    SegmentModuleScore,
)

SCORE_SYSTEM_INSTRUCTION = """\
You infer brain-region-relevant dimensions from existing dense movie descriptions.

Use only the provided text description. Do not invent visual events beyond the
description. For each module, apply its simulation prompt and score every
dimension according to the anchors. Return compact rationales that explain the
main evidence behind the scores.
"""


def build_score_schema(pool: ModulePromptPool) -> dict[str, Any]:
    """Build the dynamic JSON schema for module-dimension scoring."""

    module_properties: dict[str, Any] = {}
    for module in pool.modules:
        dimension_properties = {
            dimension.dimension_id: {"type": "number"}
            for dimension in module.dimensions
        }
        module_properties[module.module_id] = {
            "type": "object",
            "required": ["dimension_scores", "rationale"],
            "properties": {
                "dimension_scores": {
                    "type": "object",
                    "required": [dimension.dimension_id for dimension in module.dimensions],
                    "additionalProperties": False,
                    "properties": dimension_properties,
                },
                "rationale": {"type": "string"},
            },
        }
    return {
        "type": "object",
        "required": ["module_scores"],
        "properties": {
            "module_scores": {
                "type": "object",
                "required": pool.ordered_module_ids(),
                "additionalProperties": False,
                "properties": module_properties,
            },
        },
    }


def _module_prompt_block(pool: ModulePromptPool) -> str:
    """Render module prompts and dimension anchors for the scorer."""

    lines: list[str] = []
    for module in pool.modules:
        lines.extend(
            [
                f"- {module.module_id} | {module.display_name} | target={module.target_region}",
                f"  hypothesis: {module.functional_hypothesis}",
                f"  simulation_prompt: {module.simulation_prompt}",
                "  dimensions:",
            ],
        )
        for dimension in module.dimensions:
            lines.extend(
                [
                    f"    - {dimension.dimension_id} ({dimension.score_min:g} to {dimension.score_max:g})",
                    f"      definition: {dimension.definition}",
                    f"      low_anchor: {dimension.low_anchor}",
                    f"      high_anchor: {dimension.high_anchor}",
                ],
            )
    return "\n".join(lines)


def _score_prompt(segment: DescriptionSegment, pool: ModulePromptPool) -> str:
    """Build the user prompt for one description segment."""

    return "\n".join(
        [
            "Score this dense description segment for the ordered modules.",
            f"Time range: [{segment.start_s:.2f}s, {segment.end_s:.2f}s)",
            "",
            "Ordered modules and dimensions:",
            _module_prompt_block(pool),
            "",
            "Dense description:",
            segment.description,
        ],
    )


def _parse_module_scores(raw_scores: dict[str, Any]) -> dict[str, ModuleScoreResult]:
    """Parse module score payloads into model objects."""

    return {
        module_id: ModuleScoreResult.from_dict(score)
        for module_id, score in raw_scores.items()
    }


def score_description_segments(
    segments: list[DescriptionSegment],
    pool: ModulePromptPool,
    cfg: ScoreDescriptionsConfig,
) -> list[SegmentModuleScore]:
    """Score dense description segments with Gemini."""

    rows: list[SegmentModuleScore] = []
    schema = build_score_schema(pool)
    for segment in segments:
        payload = generate_structured_json(
            model=cfg.generation_model,
            system_instruction=SCORE_SYSTEM_INSTRUCTION,
            contents=[_score_prompt(segment, pool)],
            response_schema=schema,
            cfg=cfg,
        )
        rows.append(
            SegmentModuleScore(
                start_s=segment.start_s,
                end_s=segment.end_s,
                description=segment.description,
                module_scores=_parse_module_scores(payload["module_scores"]),
            ),
        )
    return rows
