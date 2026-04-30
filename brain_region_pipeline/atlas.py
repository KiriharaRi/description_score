"""Atlas helpers for brain-region module selection and Schaefer summaries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .models import ModulePromptPool, RegionModulePrompt, SelectionRule


def parse_schaefer_labels(
    label_path: str | Path,
) -> list[dict[str, str | int]]:
    """Parse Schaefer 17-network parcel labels without importing numpy."""

    parcels: list[dict[str, str | int]] = []
    with Path(label_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            idx_1based = int(parts[0])
            label = parts[1]
            comps = label.split("_")
            sub_parts = comps[3:]
            if sub_parts and sub_parts[-1].isdigit():
                sub_region = "_".join(sub_parts[:-1]) if len(sub_parts) > 1 else ""
            else:
                sub_region = "_".join(sub_parts)
            parcels.append(
                {
                    "idx_0based": idx_1based - 1,
                    "label": label,
                    "hemisphere": comps[1],
                    "network": comps[2],
                    "sub_region": sub_region,
                },
            )
    return parcels


def summarize_label_space(
    parcels: list[dict[str, str | int]],
) -> list[dict[str, int | str]]:
    """Summarize Schaefer labels by network and sub-region."""

    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"LH": 0, "RH": 0, "total": 0},
    )
    for parcel in parcels:
        key = (str(parcel["network"]), str(parcel["sub_region"]))
        hemi = str(parcel["hemisphere"])
        counts[key][hemi] += 1
        counts[key]["total"] += 1

    rows: list[dict[str, int | str]] = []
    for (network, sub_region), stat in sorted(counts.items()):
        rows.append(
            {
                "network": network,
                "sub_region": sub_region,
                "total": stat["total"],
                "LH": stat["LH"],
                "RH": stat["RH"],
            }
        )
    return rows


def render_label_space_summary(parcels: list[dict[str, str | int]]) -> str:
    """Render a compact atlas summary for the meta prompt."""

    lines = []
    for row in summarize_label_space(parcels):
        lines.append(
            "- {network}/{sub_region}: total={total}, LH={LH}, RH={RH}".format(
                **row,
            )
        )
    return "\n".join(lines)


def _matches_rule(parcel: dict[str, str | int], rule: SelectionRule) -> bool:
    """Check whether one parcel matches a selection rule."""

    if rule.networks and str(parcel["network"]) not in rule.networks:
        return False
    if rule.sub_regions and str(parcel["sub_region"]) not in rule.sub_regions:
        return False
    if rule.hemispheres and str(parcel["hemisphere"]) not in rule.hemispheres:
        return False
    return True


def expand_selection_rule(
    rule: SelectionRule,
    parcels: list[dict[str, str | int]],
) -> list[int]:
    """Expand one selection rule into 0-based parcel indices."""

    return [
        int(parcel["idx_0based"])
        for parcel in parcels
        if _matches_rule(parcel, rule)
    ]


def expand_module_indices(
    module: RegionModulePrompt,
    parcels: list[dict[str, str | int]],
) -> list[int]:
    """Expand all selection rules of a module into unique parcel indices."""

    all_indices: set[int] = set()
    for rule in module.selection_rules:
        all_indices.update(expand_selection_rule(rule, parcels))
    return sorted(all_indices)


def build_module_index_map(
    pool: ModulePromptPool,
    parcels: list[dict[str, str | int]],
) -> dict[str, list[int]]:
    """Build module_id -> parcel index mapping for prompt-based encoding."""

    mapping: dict[str, list[int]] = {}
    for module in pool.modules:
        indices = expand_module_indices(module, parcels)
        if len(indices) == 0:
            raise ValueError(
                f"Module {module.module_id!r} selects no parcels from atlas.",
            )
        mapping[module.module_id] = indices
    return mapping
