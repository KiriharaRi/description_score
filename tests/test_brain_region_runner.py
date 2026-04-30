from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np

from brain_region_pipeline.cli import main
from brain_region_pipeline.models import (
    DimensionSpec,
    ModulePromptPool,
    RegionModulePrompt,
    SelectionRule,
)
from brain_region_pipeline.runner import PipelineDependencies
from test_pipeline.encoding_model import save_encoding_results


def _sample_prompt_pool() -> ModulePromptPool:
    """Return a deterministic two-module prompt pool for smoke tests."""

    dimension = DimensionSpec(
        dimension_id="intensity",
        name="Intensity",
        definition="How strongly this module is expressed.",
        score_min=0.0,
        score_max=1.0,
        low_anchor="absent",
        high_anchor="strong",
    )
    return ModulePromptPool(
        modules=(
            RegionModulePrompt(
                module_id="scene_shift",
                target_region="PFCm",
                display_name="Scene Shift",
                functional_hypothesis="Track event boundaries.",
                simulation_prompt="Score scene changes and transitions.",
                dimensions=(dimension,),
                selection_rules=(
                    SelectionRule(networks=("DefaultA",), sub_regions=("PFCm",)),
                ),
            ),
            RegionModulePrompt(
                module_id="social_intent",
                target_region="Temp",
                display_name="Social Intent",
                functional_hypothesis="Track social inference.",
                simulation_prompt="Score intentions and relationships.",
                dimensions=(dimension,),
                selection_rules=(
                    SelectionRule(networks=("DefaultB",), sub_regions=("Temp",)),
                ),
            ),
        ),
    )


def _fake_load_bold(h5_path, run_key=None):
    """Return a stable fake BOLD matrix and run key."""

    bold = np.arange(33, dtype=np.float32).reshape(11, 3)
    return bold, run_key or "fake_run"


def _fake_fit_ridge_encoding(X_train, Y_train, X_test, Y_test, gap_trs=10):
    """Return a stable cross-run Ridge result."""

    n_parcels = Y_train.shape[1]
    return {
        "pearson_r": np.full(n_parcels, 0.5, dtype=np.float32),
        "mean_r": 0.5,
        "best_alpha": 2.0,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }


def _unexpected_call(*args, **kwargs):
    """Fail the test when a dry-run unexpectedly executes work."""

    raise AssertionError("Unexpected dependency call during dry-run")


class RunnerSmokeTests(unittest.TestCase):
    """Run the staged pipeline end to end with injected fake dependencies."""

    def test_main_without_subcommand_prints_help(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            main([])
        help_text = stdout.getvalue()
        self.assertIn("make-module-prompt", help_text)
        self.assertIn("score-descriptions", help_text)
        self.assertIn("encode", help_text)
        self.assertNotIn("make-module-pool", help_text)
        self.assertNotIn("build-features", help_text)

    def test_encode_discovers_dirs_and_writes_module_results(self) -> None:
        deps = self._make_encode_deps()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            atlas = self._write_atlas(root)
            prompt_path = self._write_module_prompt(root / "module_prompt.json")
            train_root = root / "train"
            test_root = root / "test"
            self._write_tr_features(train_root / "s01e01a")
            self._write_tr_features(test_root / "s01e01b")
            h5_path = self._write_h5(
                root / "fake_bold.h5",
                ["ses-001_task-s01e01a", "ses-001_task-s01e01b"],
            )
            output_dir = root / "encoding"
            stdout = self._run_main(
                [
                    "encode",
                    "--train-dir",
                    str(train_root),
                    "--test-dir",
                    str(test_root),
                    "--fmri-h5",
                    str(h5_path),
                    "--module-prompt",
                    str(prompt_path),
                    "--atlas-labels",
                    str(atlas),
                    "--output-dir",
                    str(output_dir),
                ],
                deps,
            )
            with (output_dir / "encoding_results.json").open("r", encoding="utf-8") as handle:
                results = json.load(handle)
            metadata = json.loads((output_dir / "training_metadata.json").read_text(encoding="utf-8"))
        self.assertIn("Discovered 1 train run(s)", stdout)
        self.assertIn("Discovered 1 test run(s)", stdout)
        self.assertIn("Encoding scene_shift (Scene Shift, 1 parcels)...", stdout)
        self.assertIn("Encoding social_intent (Social Intent, 2 parcels)...", stdout)
        self.assertNotIn("all_parcels", results)
        self.assertEqual(sorted(results.keys()), ["scene_shift", "social_intent"])
        self.assertEqual(metadata["train_episodes"][0]["run_key"], "ses-001_task-s01e01a")
        self.assertEqual(metadata["test_episodes"][0]["run_key"], "ses-001_task-s01e01b")

    def test_encode_rejects_overlapping_run_keys(self) -> None:
        deps = self._make_encode_deps()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            atlas = self._write_atlas(root)
            prompt_path = self._write_module_prompt(root / "module_prompt.json")
            train_root = root / "train"
            test_root = root / "test"
            self._write_tr_features(train_root / "s01e01a")
            self._write_tr_features(test_root / "s01e01a")
            h5_path = self._write_h5(root / "fake_bold.h5", ["ses-001_task-s01e01a"])
            with self.assertRaises(ValueError):
                self._run_main(
                    [
                        "encode",
                        "--train-dir",
                        str(train_root),
                        "--test-dir",
                        str(test_root),
                        "--fmri-h5",
                        str(h5_path),
                        "--module-prompt",
                        str(prompt_path),
                        "--atlas-labels",
                        str(atlas),
                        "--output-dir",
                        str(root / "encoding"),
                    ],
                    deps,
                )

    def _make_encode_deps(self) -> PipelineDependencies:
        return PipelineDependencies(
            build_module_prompt_pool=_unexpected_call,
            score_description_segments=_unexpected_call,
            load_bold=_fake_load_bold,
            fit_ridge_encoding=_fake_fit_ridge_encoding,
            save_encoding_results=save_encoding_results,
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
                    "2\t17Networks_RH_DefaultB_Temp_1\t0\t0\t0\t0",
                    "3\t17Networks_LH_DefaultB_Temp_2\t0\t0\t0\t0",
                ],
            ),
            encoding="utf-8",
        )
        return atlas

    def _write_module_prompt(self, path: Path) -> Path:
        path.write_text(
            json.dumps(_sample_prompt_pool().to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def _write_tr_features(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx in range(8):
            rows.append(
                {
                    "tr_index": idx,
                    "tr_start_s": float(idx),
                    "tr_end_s": float(idx + 1),
                    "module_descriptions": {
                        "scene_shift": f"scene {idx}",
                        "social_intent": f"social {idx}",
                    },
                    "feature_vector": [float(idx), float(idx + 1), float(idx + 2), float(idx + 3)],
                    "weights": {f"seg_{idx}": 1.0},
                }
            )
        with (run_dir / "tr_features.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def _write_h5(self, path: Path, keys: list[str]) -> Path:
        with h5py.File(path, "w") as handle:
            for key in keys:
                handle.create_dataset(key, data=np.zeros((11, 3), dtype=np.float32))
        return path


if __name__ == "__main__":
    unittest.main()
