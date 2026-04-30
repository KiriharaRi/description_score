"""Stage runners for the brain-region prompt scoring pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .atlas import build_module_index_map, parse_schaefer_labels
from .config import EncodeConfig, ModulePromptConfig, ScoreDescriptionsConfig
from .description_io import load_description_segments
from .encoding_eval import run_encoding_from_dirs
from .io_utils import write_json, write_jsonl
from .models import DescriptionSegment, ModulePromptPool, SegmentModuleScore
from .module_prompt import (
    build_module_prompt_pool,
    load_module_prompt_pool,
    save_module_prompt_pool,
)
from .module_scorer import score_description_segments
from .score_aligner import align_scores_to_trs
from .tr_output import save_readable_tr_rows


@dataclass(frozen=True)
class PipelineDependencies:
    """Dependency injection surface for external calls used by the pipeline."""

    build_module_prompt_pool: Callable[
        [list[dict[str, str | int]], ModulePromptConfig],
        ModulePromptPool,
    ]
    score_description_segments: Callable[
        [list[DescriptionSegment], ModulePromptPool, ScoreDescriptionsConfig],
        list[SegmentModuleScore],
    ]
    load_bold: Callable[..., tuple[object, str]]
    fit_ridge_encoding: Callable[..., dict]
    save_encoding_results: Callable[[dict[str, dict], str | Path], None]


def default_dependencies() -> PipelineDependencies:
    """Return the default dependency set."""

    def _load_bold(*args, **kwargs):
        from test_pipeline.fmri_loader import load_bold

        return load_bold(*args, **kwargs)

    def _fit_ridge_encoding(*args, **kwargs):
        from test_pipeline.encoding_model import fit_ridge_encoding

        return fit_ridge_encoding(*args, **kwargs)

    def _save_encoding_results(*args, **kwargs):
        from test_pipeline.encoding_model import save_encoding_results

        return save_encoding_results(*args, **kwargs)

    return PipelineDependencies(
        build_module_prompt_pool=build_module_prompt_pool,
        score_description_segments=score_description_segments,
        load_bold=_load_bold,
        fit_ridge_encoding=_fit_ridge_encoding,
        save_encoding_results=_save_encoding_results,
    )


def _log(message: str) -> None:
    print(f"[brain_region_pipeline] {message}", flush=True)


def make_module_prompt(
    args,
    cfg: ModulePromptConfig,
    deps: PipelineDependencies | None = None,
) -> None:
    """Run stage: atlas + target region -> module prompt pool."""

    deps = deps or default_dependencies()
    output_path = Path(args.output_file)
    parcels = parse_schaefer_labels(args.atlas_labels)
    _log("Step 1/1: Build module prompt pool")
    pool = deps.build_module_prompt_pool(parcels, cfg)
    # Validate selection rules immediately so a bad prompt fails before scoring.
    build_module_index_map(pool, parcels)
    save_module_prompt_pool(pool, output_path)
    _log(
        f"  Saved module prompt pool with {len(pool.modules)} module(s) to {output_path}",
    )
    _log("Module-prompt stage complete.")


def _infer_score_total_trs(
    scores: list[SegmentModuleScore],
    cfg: ScoreDescriptionsConfig,
    total_trs: int | None,
) -> int:
    """Infer TR count from scored segment end times when no override is given."""

    if total_trs is not None:
        return total_trs
    if not scores:
        return 0
    return int(math.ceil(max(score.end_s for score in scores) / cfg.tr_s))


def score_descriptions_from_file(
    args,
    cfg: ScoreDescriptionsConfig,
    deps: PipelineDependencies | None = None,
) -> None:
    """Run stage: existing dense descriptions -> module-dimension scores."""

    deps = deps or default_dependencies()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log("Step 1/4: Load module prompt pool")
    pool = load_module_prompt_pool(args.module_prompt)
    _log(f"  Module prompt pool ready: {len(pool.modules)} module(s) from {args.module_prompt}")
    _log("Step 2/4: Load dense descriptions")
    segments = load_description_segments(args.descriptions)
    _log(f"  Loaded {len(segments)} description segments from {args.descriptions}")
    _log(f"Step 3/4: Score {len(segments)} description segments")
    scores = deps.score_description_segments(segments, pool, cfg)
    write_jsonl(output_dir / "segment_module_scores.jsonl", [score.to_dict() for score in scores])
    _log(f"  Wrote segment scores to {output_dir / 'segment_module_scores.jsonl'}")
    _log("Step 4/4: Align scores to TR features")
    total_trs = _infer_score_total_trs(scores, cfg, args.total_trs)
    tr_rows = align_scores_to_trs(
        scores=scores,
        pool=pool,
        total_trs=total_trs,
        cfg=cfg,
    )
    write_jsonl(output_dir / "tr_features.jsonl", [row.to_dict() for row in tr_rows])
    save_readable_tr_rows(output_dir, tr_rows)
    write_json(
        output_dir / "scoring_metadata.json",
        {
            "n_segments": len(segments),
            "n_trs": total_trs,
            "tr_s": cfg.tr_s,
            "alignment": cfg.alignment_strategy,
            "ordered_feature_keys": [
                {"module_id": module_id, "dimension_id": dimension_id}
                for module_id, dimension_id in pool.ordered_feature_keys()
            ],
        },
    )
    _log(f"  Wrote {len(tr_rows)} TR rows to {output_dir / 'tr_features.jsonl'}")
    _log(f"  Wrote readable TR descriptions to {output_dir / 'tr_descriptions_readable.jsonl'}")
    _log(f"  Wrote scoring metadata to {output_dir / 'scoring_metadata.json'}")
    _log(f"Description-scoring stage complete. Outputs in {output_dir}")


def encode_from_feature_dirs(
    args,
    cfg: EncodeConfig,
    deps: PipelineDependencies | None = None,
) -> None:
    """Run stage: train/test scored feature dirs -> ROI encoding results."""

    deps = deps or default_dependencies()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log("Step 1/2: Load module prompt and atlas")
    pool = load_module_prompt_pool(args.module_prompt)
    parcels = parse_schaefer_labels(args.atlas_labels)
    build_module_index_map(pool, parcels)
    _log(f"  Module prompt ready: {len(pool.modules)} module(s) from {args.module_prompt}")
    _log("Step 2/2: Encode train/test feature directories")
    run_encoding_from_dirs(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        fmri_h5=Path(args.fmri_h5),
        pool=pool,
        parcels=parcels,
        cfg=cfg,
        load_bold_fn=deps.load_bold,
        fit_ridge_encoding_fn=deps.fit_ridge_encoding,
        save_encoding_results_fn=deps.save_encoding_results,
        output_dir=output_dir,
        log_fn=_log,
    )
    _log(f"Encode stage complete. Outputs in {output_dir}")
