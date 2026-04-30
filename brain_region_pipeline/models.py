"""Core data models for the brain-region prompt pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _normalize_strs(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize a list of labels into a deduplicated tuple."""

    if not values:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


@dataclass(frozen=True)
class SelectionRule:
    """One whitelist rule used to expand a region module onto Schaefer parcels."""

    networks: tuple[str, ...] = ()
    sub_regions: tuple[str, ...] = ()
    hemispheres: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelectionRule":
        """Build a rule from serialized JSON data."""

        return cls(
            networks=_normalize_strs(data.get("networks")),
            sub_regions=_normalize_strs(data.get("sub_regions")),
            hemispheres=_normalize_strs(data.get("hemispheres")),
        )

    def to_dict(self) -> dict[str, list[str]]:
        """Serialize to JSON."""

        return {
            "networks": list(self.networks),
            "sub_regions": list(self.sub_regions),
            "hemispheres": list(self.hemispheres),
        }


@dataclass(frozen=True)
class DimensionSpec:
    """One interpretable scoring dimension for a region-specific prompt."""

    dimension_id: str
    name: str
    definition: str
    score_min: float = 0.0
    score_max: float = 1.0
    low_anchor: str = ""
    high_anchor: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DimensionSpec":
        """Build a dimension spec from serialized JSON data."""

        return cls(
            dimension_id=str(data["dimension_id"]).strip(),
            name=str(data["name"]).strip(),
            definition=str(data["definition"]).strip(),
            score_min=float(data.get("score_min", 0.0)),
            score_max=float(data.get("score_max", 1.0)),
            low_anchor=str(data.get("low_anchor", "")).strip(),
            high_anchor=str(data.get("high_anchor", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON."""

        return {
            "dimension_id": self.dimension_id,
            "name": self.name,
            "definition": self.definition,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "low_anchor": self.low_anchor,
            "high_anchor": self.high_anchor,
        }


@dataclass(frozen=True)
class RegionModulePrompt:
    """One region-specific module prompt and its scoring dimensions."""

    module_id: str
    target_region: str
    display_name: str
    functional_hypothesis: str
    simulation_prompt: str
    dimensions: tuple[DimensionSpec, ...]
    selection_rules: tuple[SelectionRule, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegionModulePrompt":
        """Build a region module prompt from serialized JSON data."""

        return cls(
            module_id=str(data["module_id"]).strip(),
            target_region=str(data["target_region"]).strip(),
            display_name=str(data["display_name"]).strip(),
            functional_hypothesis=str(data["functional_hypothesis"]).strip(),
            simulation_prompt=str(data["simulation_prompt"]).strip(),
            dimensions=tuple(
                DimensionSpec.from_dict(item)
                for item in data.get("dimensions", [])
            ),
            selection_rules=tuple(
                SelectionRule.from_dict(rule)
                for rule in data.get("selection_rules", [])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON."""

        return {
            "module_id": self.module_id,
            "target_region": self.target_region,
            "display_name": self.display_name,
            "functional_hypothesis": self.functional_hypothesis,
            "simulation_prompt": self.simulation_prompt,
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
            "selection_rules": [rule.to_dict() for rule in self.selection_rules],
        }


@dataclass(frozen=True)
class ModulePromptPool:
    """Ordered region module prompts used to score existing descriptions."""

    modules: tuple[RegionModulePrompt, ...]
    version: str = "module_prompt_v1"
    source_model: str = ""

    def __post_init__(self) -> None:
        module_ids = [module.module_id for module in self.modules]
        if not self.modules:
            raise ValueError("Module prompt pool cannot be empty.")
        if len(module_ids) != len(set(module_ids)):
            raise ValueError(f"Duplicate module_id in prompt pool: {module_ids}")
        for module in self.modules:
            dimension_ids = [dimension.dimension_id for dimension in module.dimensions]
            if not module.dimensions:
                raise ValueError(f"Module {module.module_id!r} has no dimensions.")
            if len(dimension_ids) != len(set(dimension_ids)):
                raise ValueError(
                    f"Duplicate dimension_id in module {module.module_id!r}: {dimension_ids}",
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModulePromptPool":
        """Build a prompt pool from serialized JSON data."""

        return cls(
            modules=tuple(
                RegionModulePrompt.from_dict(item)
                for item in data.get("modules", [])
            ),
            version=str(data.get("version", "module_prompt_v1")),
            source_model=str(data.get("source_model", "")).strip(),
        )

    def ordered_module_ids(self) -> list[str]:
        """Return the fixed module order used for scoring and feature output."""

        return [module.module_id for module in self.modules]

    def ordered_feature_keys(self) -> list[tuple[str, str]]:
        """Return the ordered module/dimension keys used in feature vectors."""

        keys: list[tuple[str, str]] = []
        for module in self.modules:
            keys.extend(
                (module.module_id, dimension.dimension_id)
                for dimension in module.dimensions
            )
        return keys

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON."""

        return {
            "version": self.version,
            "source_model": self.source_model,
            "modules": [module.to_dict() for module in self.modules],
        }


@dataclass(frozen=True)
class DescriptionSegment:
    """One timestamped dense description segment from an external source."""

    start_s: float
    end_s: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON."""

        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "description": self.description,
        }


@dataclass(frozen=True)
class ModuleScoreResult:
    """Dimension scores and short rationale for one module."""

    dimension_scores: dict[str, float]
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModuleScoreResult":
        """Build a module score result from serialized JSON data."""

        return cls(
            dimension_scores={
                str(key): float(value)
                for key, value in data.get("dimension_scores", {}).items()
            },
            rationale=str(data.get("rationale", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON."""

        return {
            "dimension_scores": dict(self.dimension_scores),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class SegmentModuleScore:
    """All region-module scores inferred for one description segment."""

    start_s: float
    end_s: float
    description: str
    module_scores: dict[str, ModuleScoreResult]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentModuleScore":
        """Build segment-level module scores from serialized JSON data."""

        return cls(
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            description=str(data["description"]).strip(),
            module_scores={
                str(module_id): ModuleScoreResult.from_dict(score)
                for module_id, score in data.get("module_scores", {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON."""

        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "description": self.description,
            "module_scores": {
                module_id: score.to_dict()
                for module_id, score in self.module_scores.items()
            },
        }


@dataclass(frozen=True)
class TRFeatureRow:
    """One TR-aligned feature row with readable module texts."""

    tr_index: int
    tr_start_s: float
    tr_end_s: float
    module_descriptions: dict[str, str]
    feature_vector: list[float] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON."""

        return {
            "tr_index": self.tr_index,
            "tr_start_s": self.tr_start_s,
            "tr_end_s": self.tr_end_s,
            "module_descriptions": dict(self.module_descriptions),
            "feature_vector": self.feature_vector,
            "weights": self.weights,
        }
