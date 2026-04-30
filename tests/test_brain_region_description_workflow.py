from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from brain_region_pipeline.cli import main
from brain_region_pipeline.description_io import parse_description_text
from brain_region_pipeline.config import ModulePromptConfig
from brain_region_pipeline.module_prompt import _build_prompt, _module_prompt_schema
from brain_region_pipeline.models import (
    DescriptionSegment,
    DimensionSpec,
    ModulePromptPool,
    ModuleScoreResult,
    RegionModulePrompt,
    SegmentModuleScore,
    SelectionRule,
)
from brain_region_pipeline.runner import PipelineDependencies


def _fake_module_prompt_pool(parcels, cfg):
    return _sample_module_prompt_pool(source_model=cfg.generation_model)


def _fake_score_description_segments(segments, pool, cfg):
    rows = []
    for idx, segment in enumerate(segments):
        rows.append(
            SegmentModuleScore(
                start_s=segment.start_s,
                end_s=segment.end_s,
                description=segment.description,
                module_scores={
                    "vmpfc": ModuleScoreResult(
                        dimension_scores={
                            "affective_valence": 0.2 + idx,
                            "social_reward": 0.7 + idx,
                        },
                        rationale=f"Segment {idx} has vmPFC-relevant value cues.",
                    ),
                },
            ),
        )
    return rows


def _unexpected_call(*args, **kwargs):
    raise AssertionError("Unexpected dependency call")


def _sample_module_prompt_pool(source_model: str = "fake-model") -> ModulePromptPool:
    return ModulePromptPool(
        modules=(
            RegionModulePrompt(
                module_id="vmpfc",
                target_region="vmPFC",
                display_name="vmPFC affective value",
                functional_hypothesis="Tracks affective value during movie viewing.",
                simulation_prompt="Infer affective value and social reward from the description.",
                dimensions=(
                    DimensionSpec(
                        dimension_id="affective_valence",
                        name="Affective valence",
                        definition="How positive or negative the moment feels.",
                        score_min=-1.0,
                        score_max=1.0,
                        low_anchor="strongly negative",
                        high_anchor="strongly positive",
                    ),
                    DimensionSpec(
                        dimension_id="social_reward",
                        name="Social reward",
                        definition="How socially rewarding or affiliative the moment is.",
                        score_min=0.0,
                        score_max=1.0,
                        low_anchor="no social reward",
                        high_anchor="strong social reward",
                    ),
                ),
                selection_rules=(
                    SelectionRule(
                        networks=("DefaultA",),
                        sub_regions=("PFCm",),
                        hemispheres=("LH", "RH"),
                    ),
                ),
            ),
        ),
        source_model=source_model,
    )


class DescriptionWorkflowTests(unittest.TestCase):
    def test_module_prompt_meta_prompt_requests_one_unbiased_region_prompt(self) -> None:
        parcels = [
            {"network": "DefaultA", "sub_region": "PFCm", "hemisphere": "LH", "idx_0based": 0},
            {"network": "DefaultA", "sub_region": "PFCm", "hemisphere": "RH", "idx_0based": 1},
        ]
        cfg = ModulePromptConfig(target_region="vmPFC")

        prompt = _build_prompt(parcels, cfg)
        schema = _module_prompt_schema()

        self.assertEqual(schema["properties"]["modules"]["minItems"], 1)
        self.assertEqual(schema["properties"]["modules"]["maxItems"], 1)
        self.assertIn("Create exactly one region-level module prompt", prompt)
        self.assertIn("Do not split the target region into multiple modules", prompt)
        for forbidden in [
            "For a vmPFC demo",
            "affective",
            "reward/avoidance",
            "social value",
            "self/other relevance",
            "emotion intensity",
            "0.0-1.0",
            "-1.0-1.0",
            "at most",
        ]:
            self.assertNotIn(forbidden, prompt)

    def test_parse_description_text_reads_timestamped_blocks(self) -> None:
        text = """
00:00 - 00:01  In a kitchen area, Ross stands holding a white phone.

00:01 - 00:09  Ross raises the phone and says he will order another pizza.
He promises to show Chandler how well he can flirt.
"""

        segments = parse_description_text(text)

        self.assertEqual(
            segments,
            [
                DescriptionSegment(
                    start_s=0.0,
                    end_s=1.0,
                    description="In a kitchen area, Ross stands holding a white phone.",
                ),
                DescriptionSegment(
                    start_s=1.0,
                    end_s=9.0,
                    description=(
                        "Ross raises the phone and says he will order another pizza. "
                        "He promises to show Chandler how well he can flirt."
                    ),
                ),
            ],
        )

    def test_parse_description_text_skips_markdown_segment_header(self) -> None:
        text = """
# Segment 4
**Time Range:** 00:09:11-00:11:04

00:00 - 00:01  In a kitchen area with shelves of food, Ross stands holding a white cordless phone.

00:01 - 00:09  Ross raises the phone to chest level and says, "right."
"""

        segments = parse_description_text(text)

        self.assertEqual(len(segments), 2)
        self.assertEqual(
            segments[0],
            DescriptionSegment(
                start_s=0.0,
                end_s=1.0,
                description="In a kitchen area with shelves of food, Ross stands holding a white cordless phone.",
            ),
        )
        self.assertEqual(segments[1].start_s, 1.0)
        self.assertEqual(segments[1].end_s, 9.0)

    def test_make_module_prompt_writes_prompt_pool(self) -> None:
        deps = self._deps()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            atlas = self._write_atlas(root)
            output_file = root / "vmpfc_module_prompt.json"
            stdout = self._run_main(
                [
                    "make-module-prompt",
                    "--atlas-labels",
                    str(atlas),
                    "--target-region",
                    "vmPFC",
                    "--output-file",
                    str(output_file),
                ],
                deps,
            )
            payload = json.loads(output_file.read_text(encoding="utf-8"))

        self.assertIn("Build module prompt pool", stdout)
        self.assertEqual(payload["modules"][0]["module_id"], "vmpfc")
        self.assertEqual(payload["modules"][0]["dimensions"][0]["dimension_id"], "affective_valence")

    def test_score_descriptions_writes_segment_scores_and_tr_features(self) -> None:
        deps = self._deps()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            descriptions = root / "description.txt"
            descriptions.write_text(
                "\n".join(
                    [
                        "00:00 - 00:01  Ross holds a phone in the kitchen.",
                        "",
                        "00:01 - 00:03  Ross says he will get the pizza woman's phone number.",
                    ],
                ),
                encoding="utf-8",
            )
            prompt_file = root / "vmpfc_module_prompt.json"
            prompt_file.write_text(
                json.dumps(_sample_module_prompt_pool().to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            output_dir = root / "scored"

            stdout = self._run_main(
                [
                    "score-descriptions",
                    "--descriptions",
                    str(descriptions),
                    "--module-prompt",
                    str(prompt_file),
                    "--output-dir",
                    str(output_dir),
                    "--tr-s",
                    "1.0",
                    "--total-trs",
                    "3",
                ],
                deps,
            )

            score_rows = self._read_jsonl(output_dir / "segment_module_scores.jsonl")
            tr_rows = self._read_jsonl(output_dir / "tr_features.jsonl")

        self.assertIn("Score 2 description segments", stdout)
        self.assertEqual(score_rows[1]["module_scores"]["vmpfc"]["dimension_scores"]["social_reward"], 1.7)
        self.assertEqual(tr_rows[0]["feature_vector"], [0.2, 0.7])
        self.assertEqual(tr_rows[2]["feature_vector"], [1.2, 1.7])

    def _deps(self) -> PipelineDependencies:
        return PipelineDependencies(
            build_module_prompt_pool=_fake_module_prompt_pool,
            score_description_segments=_fake_score_description_segments,
            load_bold=_unexpected_call,
            fit_ridge_encoding=_unexpected_call,
            save_encoding_results=_unexpected_call,
        )

    def _run_main(self, argv: list[str], deps: PipelineDependencies) -> str:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            main(argv, deps=deps)
        return stdout.getvalue()

    def _write_atlas(self, root: Path) -> Path:
        atlas = root / "atlas.txt"
        atlas.write_text(
            "\n".join(
                [
                    "1\t17Networks_LH_DefaultA_PFCm_1\t0\t0\t0\t0",
                    "2\t17Networks_RH_DefaultA_PFCm_2\t0\t0\t0\t0",
                ],
            ),
            encoding="utf-8",
        )
        return atlas

    def _read_jsonl(self, path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    unittest.main()
