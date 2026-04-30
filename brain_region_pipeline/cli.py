"""CLI for the brain-region prompt scoring pipeline."""

from __future__ import annotations

import argparse

from .config import (
    EncodeConfig,
    ModulePromptConfig,
    ScoreDescriptionsConfig,
)
from .runner import (
    PipelineDependencies,
    encode_from_feature_dirs,
    make_module_prompt,
    score_descriptions_from_file,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(description="Brain-region prompt scoring pipeline")
    subparsers = parser.add_subparsers(dest="command")

    prompt_parser = subparsers.add_parser(
        "make-module-prompt",
        help="Generate region module prompts from atlas labels and a target region.",
    )
    prompt_parser.add_argument("--atlas-labels", required=True, help="Schaefer label file.")
    prompt_parser.add_argument("--target-region", default="vmPFC", help="Target region name.")
    prompt_parser.add_argument("--output-file", required=True, help="Output module prompt JSON path.")
    prompt_parser.add_argument("--model", default="gemini-3.1-pro-preview", help="Gemini generation model.")

    score_parser = subparsers.add_parser(
        "score-descriptions",
        help="Score existing dense descriptions with region module prompts.",
    )
    score_parser.add_argument("--descriptions", required=True, help="Timestamped dense description text file.")
    score_parser.add_argument("--module-prompt", required=True, help="Module prompt JSON path.")
    score_parser.add_argument("--output-dir", required=True, help="Output directory for scored features.")
    score_parser.add_argument("--model", default="gemini-3.1-pro-preview", help="Gemini generation model.")
    score_parser.add_argument("--tr-s", type=float, default=1.49, help="TR in seconds.")
    score_parser.add_argument("--total-trs", type=int, default=None, help="Override total TR count.")
    score_parser.add_argument(
        "--alignment",
        default="overlap_weighted",
        choices=["overlap_weighted", "repeat"],
        help="TR alignment strategy.",
    )

    encode_parser = subparsers.add_parser(
        "encode",
        help="Train on train-dir features and evaluate on test-dir features.",
    )
    encode_parser.add_argument("--train-dir", required=True, help="Root directory of train run outputs.")
    encode_parser.add_argument("--test-dir", required=True, help="Root directory of test run outputs.")
    encode_parser.add_argument("--fmri-h5", required=True, help="HDF5 file containing BOLD runs.")
    encode_parser.add_argument("--module-prompt", required=True, help="module_prompt.json used for ROI mapping.")
    encode_parser.add_argument("--atlas-labels", required=True, help="Schaefer label file.")
    encode_parser.add_argument("--output-dir", required=True, help="Output directory for encoding results.")
    encode_parser.add_argument("--lag-trs", type=int, default=3, help="HRF lag in TRs.")
    encode_parser.add_argument("--gap-trs", type=int, default=10, help="Gap for TimeSeriesSplit CV.")

    return parser


def _build_module_prompt_config(args: argparse.Namespace) -> ModulePromptConfig:
    """Build module prompt generation configuration from CLI args."""

    return ModulePromptConfig(
        generation_model=args.model,
        target_region=args.target_region,
    )


def _build_score_config(args: argparse.Namespace) -> ScoreDescriptionsConfig:
    """Build description scoring configuration from CLI args."""

    return ScoreDescriptionsConfig(
        generation_model=args.model,
        tr_s=args.tr_s,
        alignment_strategy=args.alignment,
    )


def _build_encode_config(args: argparse.Namespace) -> EncodeConfig:
    """Build stage3 configuration from CLI args."""

    return EncodeConfig(
        lag_trs=args.lag_trs,
        gap_trs=args.gap_trs,
    )


def main(
    argv: list[str] | None = None,
    deps: PipelineDependencies | None = None,
) -> None:
    """CLI entrypoint for the brain-region prompt pipeline."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    if args.command == "make-module-prompt":
        make_module_prompt(args, _build_module_prompt_config(args), deps=deps)
        return
    if args.command == "score-descriptions":
        score_descriptions_from_file(args, _build_score_config(args), deps=deps)
        return
    if args.command == "encode":
        encode_from_feature_dirs(args, _build_encode_config(args), deps=deps)
        return
    raise ValueError(f"Unknown command: {args.command!r}")
