"""Meta-prompt generation and persistence for region module prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .atlas import render_label_space_summary
from .config import ModulePromptConfig
from .genai import generate_structured_json
from .io_utils import read_json, write_json
from .models import ModulePromptPool

MODULE_PROMPT_SYSTEM_INSTRUCTION = """\
You are designing interpretable brain-region activity simulators for movie-fMRI.

Given an atlas label space and a target region, infer one region-level module
prompt for that region. Do not split the target region into multiple modules;
represent functional diversity as scored dimensions inside the single module.
The module must specify:
- the atlas selection rules that identify the region parcels,
- the functional hypothesis for what the region tracks during movie viewing,
- a simulation prompt that another LLM can use with an existing dense
  description,
- a compact set of scored dimensions that turn the description into numeric
  features.

The output is for a scientific demo. Keep the dimensions interpretable,
non-overlapping, and directly scoreable from text descriptions.
"""


def _dimension_schema() -> dict[str, Any]:
    """Build the JSON schema for one scoring dimension."""

    return {
        "type": "object",
        "required": [
            "dimension_id",
            "name",
            "definition",
            "score_min",
            "score_max",
            "low_anchor",
            "high_anchor",
        ],
        "properties": {
            "dimension_id": {"type": "string"},
            "name": {"type": "string"},
            "definition": {"type": "string"},
            "score_min": {"type": "number"},
            "score_max": {"type": "number"},
            "low_anchor": {"type": "string"},
            "high_anchor": {"type": "string"},
        },
    }


def _selection_rule_schema() -> dict[str, Any]:
    """Build the JSON schema for one atlas selection rule."""

    return {
        "type": "object",
        "required": ["networks", "sub_regions", "hemispheres"],
        "properties": {
            "networks": {"type": "array", "items": {"type": "string"}},
            "sub_regions": {"type": "array", "items": {"type": "string"}},
            "hemispheres": {"type": "array", "items": {"type": "string"}},
        },
    }


def _module_prompt_schema() -> dict[str, Any]:
    """Build the response schema for module prompt generation."""

    module_schema = {
        "type": "object",
        "required": [
            "module_id",
            "target_region",
            "display_name",
            "functional_hypothesis",
            "simulation_prompt",
            "dimensions",
            "selection_rules",
        ],
        "properties": {
            "module_id": {"type": "string"},
            "target_region": {"type": "string"},
            "display_name": {"type": "string"},
            "functional_hypothesis": {"type": "string"},
            "simulation_prompt": {"type": "string"},
            "dimensions": {
                "type": "array",
                "minItems": 1,
                "items": _dimension_schema(),
            },
            "selection_rules": {
                "type": "array",
                "minItems": 1,
                "items": _selection_rule_schema(),
            },
        },
    }
    return {
        "type": "object",
        "required": ["modules"],
        "properties": {
            "modules": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": module_schema,
            },
        },
    }


def _build_prompt(
    parcels: list[dict[str, str | int]],
    cfg: ModulePromptConfig,
) -> str:
    """Build the meta prompt for target-region module prompt generation."""

    return "\n".join(
        [
            f"Target region: {cfg.target_region}",
            "Create exactly one region-level module prompt for the target region.",
            "Do not split the target region into multiple modules.",
            "",
            "Infer the scored dimensions from the target region, the atlas label",
            "space, and relevant neuroscience knowledge. Do not use example",
            "dimension names supplied by this prompt; decide the dimensions yourself.",
            "",
            "Each dimension must be possible to score from an existing movie dense",
            "description. Keep dimensions compact, interpretable, and minimally",
            "overlapping. For each dimension, choose a numeric score range and",
            "clear low/high anchors.",
            "",
            "Selection rule semantics:",
            "- Within one rule, a parcel must satisfy every non-empty field.",
            "- Across multiple rules, the final module selects the union.",
            "",
            "Atlas label space summary:",
            render_label_space_summary(parcels),
        ],
    )


def build_module_prompt_pool(
    parcels: list[dict[str, str | int]],
    cfg: ModulePromptConfig,
) -> ModulePromptPool:
    """Generate a region-specific module prompt pool with Gemini."""

    payload = generate_structured_json(
        model=cfg.generation_model,
        system_instruction=MODULE_PROMPT_SYSTEM_INSTRUCTION,
        contents=[_build_prompt(parcels, cfg)],
        response_schema=_module_prompt_schema(),
        cfg=cfg,
    )
    return ModulePromptPool.from_dict(
        {
            "version": "module_prompt_v1",
            "source_model": cfg.generation_model,
            "modules": payload["modules"],
        },
    )


def save_module_prompt_pool(pool: ModulePromptPool, path: str | Path) -> None:
    """Persist a region module prompt pool to disk."""

    write_json(path, pool.to_dict())


def load_module_prompt_pool(path: str | Path) -> ModulePromptPool:
    """Load a region module prompt pool from disk."""

    return ModulePromptPool.from_dict(read_json(path))
