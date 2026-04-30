"""Stage-scoped configuration objects for the brain-region prompt pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    """Shared Gemini generation settings used by prompt generation and scoring."""

    generation_model: str = "gemini-3.1-pro-preview"
    temperature: float = 0.2
    max_retries: int = 3
    retry_delay_s: float = 2.0
    generation_timeout_s: float = 300.0


@dataclass(frozen=True)
class ModulePromptConfig(GenerationConfig):
    """Configuration for region-specific module prompt generation."""

    target_region: str = "vmPFC"


@dataclass(frozen=True)
class EncodeConfig:
    """Configuration for directory-driven encoding."""

    lag_trs: int = 3
    gap_trs: int = 10


@dataclass(frozen=True)
class ScoreDescriptionsConfig(GenerationConfig):
    """Configuration for scoring existing dense descriptions."""

    tr_s: float = 1.49
    alignment_strategy: str = "overlap_weighted"
