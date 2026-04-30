from __future__ import annotations

import unittest

from brain_region_pipeline.atlas import build_module_index_map
from brain_region_pipeline.models import (
    DimensionSpec,
    ModulePromptPool,
    RegionModulePrompt,
    SelectionRule,
)


def _dimension() -> DimensionSpec:
    return DimensionSpec(
        dimension_id="intensity",
        name="Intensity",
        definition="How strongly this module is expressed.",
        score_min=0.0,
        score_max=1.0,
        low_anchor="absent",
        high_anchor="strong",
    )


class AtlasPromptMappingTests(unittest.TestCase):
    """Validate atlas expansion for the current module-prompt workflow."""

    def test_build_module_index_map_expands_prompt_selection_rules(self) -> None:
        pool = ModulePromptPool(
            modules=(
                RegionModulePrompt(
                    module_id="vmpfc",
                    target_region="vmPFC",
                    display_name="vmPFC",
                    functional_hypothesis="Tracks affective value.",
                    simulation_prompt="Score affective value.",
                    dimensions=(_dimension(),),
                    selection_rules=(
                        SelectionRule(
                            networks=("DefaultA",),
                            sub_regions=("PFCm",),
                            hemispheres=("LH",),
                        ),
                        SelectionRule(
                            networks=("DefaultB",),
                            sub_regions=("PFCv",),
                            hemispheres=("RH",),
                        ),
                    ),
                ),
            ),
        )
        parcels = [
            {"idx_0based": 0, "network": "DefaultA", "sub_region": "PFCm", "hemisphere": "LH"},
            {"idx_0based": 1, "network": "DefaultA", "sub_region": "PFCm", "hemisphere": "RH"},
            {"idx_0based": 2, "network": "DefaultB", "sub_region": "PFCv", "hemisphere": "RH"},
        ]

        mapping = build_module_index_map(pool, parcels)

        self.assertEqual(mapping["vmpfc"], [0, 2])

    def test_build_module_index_map_rejects_empty_prompt_selection(self) -> None:
        pool = ModulePromptPool(
            modules=(
                RegionModulePrompt(
                    module_id="vmpfc",
                    target_region="vmPFC",
                    display_name="vmPFC",
                    functional_hypothesis="Tracks affective value.",
                    simulation_prompt="Score affective value.",
                    dimensions=(_dimension(),),
                    selection_rules=(
                        SelectionRule(networks=("LimbicB",), sub_regions=("OFC",)),
                    ),
                ),
            ),
        )
        parcels = [
            {"idx_0based": 0, "network": "DefaultA", "sub_region": "PFCm", "hemisphere": "LH"},
        ]

        with self.assertRaises(ValueError):
            build_module_index_map(pool, parcels)


if __name__ == "__main__":
    unittest.main()
