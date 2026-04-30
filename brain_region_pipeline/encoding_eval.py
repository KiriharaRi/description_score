"""Directory-driven encoding helpers for scored brain-region features."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .atlas import build_module_index_map
from .config import EncodeConfig
from .io_utils import read_jsonl, write_json
from .models import ModulePromptPool


def _emit_log(log_fn: Callable[[str], None] | None, message: str) -> None:
    """Emit one optional progress log."""

    if log_fn:
        log_fn(message)


def _log_encoding_result(
    log_fn: Callable[[str], None] | None,
    label: str,
    result: dict,
) -> None:
    """Log one encoding result summary."""

    _emit_log(
        log_fn,
        f"    {label}: mean r = {result['mean_r']:.4f} (alpha={result['best_alpha']:.1f})",
    )


def _load_tr_features(path: Path) -> np.ndarray:
    """Return feature matrix from one run directory."""

    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty tr_features.jsonl at {path}")
    return np.array([row["feature_vector"] for row in rows], dtype=np.float32)


def _discover_run_dirs(root: Path) -> list[tuple[str, Path]]:
    """Find per-run subdirectories that contain tr_features.jsonl."""

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    found: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "tr_features.jsonl").exists():
            found.append((child.name, child))
    if not found:
        raise ValueError(
            f"No subdirectories with tr_features.jsonl found under {root}. "
            "Each run must live in its own subdirectory (e.g. train/s01e01a/tr_features.jsonl)."
        )
    return found


def _resolve_run_key(h5_path: Path, episode_id: str) -> str:
    """Find the unique HDF5 dataset key ending in task-{episode_id}."""

    import h5py

    with h5py.File(h5_path, "r") as handle:
        all_keys = list(handle.keys())
    suffix = f"task-{episode_id}"
    matches = [key for key in all_keys if key.endswith(suffix)]
    if not matches:
        raise ValueError(
            f"No HDF5 dataset key ends with '{suffix}' in {h5_path}. "
            f"Check that subdirectory name '{episode_id}' matches the HDF5 episode suffix.",
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple HDF5 dataset keys end with '{suffix}': {matches}. "
            "Include the part letter in the directory name.",
        )
    return matches[0]


def _load_run(
    ep_dir: Path,
    run_key: str,
    h5_path: Path,
    lag_trs: int,
    load_bold_fn: Callable[..., tuple[np.ndarray, str]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one run's features and BOLD, then apply HRF lag."""

    from test_pipeline.encoding_model import apply_hrf_lag

    features = _load_tr_features(ep_dir / "tr_features.jsonl")
    bold, _ = load_bold_fn(h5_path, run_key=run_key)
    n_feat_raw = features.shape[0]
    n_bold_raw = bold.shape[0]
    x_lag, y_lag = apply_hrf_lag(features, bold, lag_trs=lag_trs)
    meta = {
        "episode": ep_dir.name,
        "run_key": run_key,
        "dir": str(ep_dir),
        "n_trs_features_raw": int(n_feat_raw),
        "n_trs_bold_raw": int(n_bold_raw),
        "n_trs_after_lag": int(x_lag.shape[0]),
        "feature_dim": int(x_lag.shape[1]),
        "n_parcels": int(y_lag.shape[1]),
    }
    return x_lag, y_lag, meta


def _concat_runs(
    run_dirs: list[tuple[str, Path]],
    h5_path: Path,
    cfg: EncodeConfig,
    label: str,
    load_bold_fn: Callable[..., tuple[np.ndarray, str]],
    log_fn: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load runs one by one and concatenate lagged train/test matrices."""

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    metas: list[dict] = []
    for episode_id, ep_dir in run_dirs:
        run_key = _resolve_run_key(h5_path, episode_id)
        x_run, y_run, meta = _load_run(ep_dir, run_key, h5_path, cfg.lag_trs, load_bold_fn)
        _emit_log(
            log_fn,
            f"  [{label}] {episode_id} -> {run_key} | "
            f"{x_run.shape[0]} TRs after lag, feat_dim={x_run.shape[1]}, parcels={y_run.shape[1]}",
        )
        x_parts.append(x_run)
        y_parts.append(y_run)
        metas.append(meta)
    feat_dims = {part.shape[1] for part in x_parts}
    if len(feat_dims) > 1:
        raise ValueError(f"[{label}] Feature dimension mismatch across runs: {sorted(feat_dims)}")
    parcel_dims = {part.shape[1] for part in y_parts}
    if len(parcel_dims) > 1:
        raise ValueError(f"[{label}] BOLD parcel count mismatch across runs: {sorted(parcel_dims)}")
    return np.vstack(x_parts), np.vstack(y_parts), metas


def run_encoding_from_dirs(
    *,
    train_dir: Path,
    test_dir: Path,
    fmri_h5: Path,
    pool: ModulePromptPool,
    parcels: list[dict[str, str | int]],
    cfg: EncodeConfig,
    load_bold_fn: Callable[..., tuple[np.ndarray, str]],
    fit_ridge_encoding_fn: Callable[..., dict],
    save_encoding_results_fn: Callable[[dict[str, dict], str | Path], None],
    output_dir: Path,
    log_fn: Callable[[str], None] | None = None,
) -> None:
    """Encode train/test feature directories at the module ROI level."""

    train_runs = _discover_run_dirs(train_dir)
    test_runs = _discover_run_dirs(test_dir)
    _emit_log(log_fn, f"  Discovered {len(train_runs)} train run(s): {[name for name, _ in train_runs]}")
    _emit_log(log_fn, f"  Discovered {len(test_runs)} test run(s): {[name for name, _ in test_runs]}")

    x_train, y_train, train_meta = _concat_runs(
        train_runs,
        fmri_h5,
        cfg,
        "train",
        load_bold_fn,
        log_fn=log_fn,
    )
    x_test, y_test, test_meta = _concat_runs(
        test_runs,
        fmri_h5,
        cfg,
        "test",
        load_bold_fn,
        log_fn=log_fn,
    )

    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch train vs test: train={x_train.shape[1]}, test={x_test.shape[1]}"
        )
    if y_train.shape[1] != y_test.shape[1]:
        raise ValueError(
            f"BOLD parcel count mismatch train vs test: train={y_train.shape[1]}, test={y_test.shape[1]}"
        )

    train_run_keys = {meta["run_key"] for meta in train_meta}
    test_run_keys = {meta["run_key"] for meta in test_meta}
    overlap = train_run_keys & test_run_keys
    if overlap:
        raise ValueError(f"Train and test run_keys overlap: {sorted(overlap)}")

    _emit_log(
        log_fn,
        f"  Pooled train: {x_train.shape[0]} TRs x {x_train.shape[1]}d | "
        f"test: {x_test.shape[0]} TRs x {x_test.shape[1]}d",
    )

    module_index_map = build_module_index_map(pool, parcels)
    results: dict[str, dict] = {}
    train_run_str = ",".join(sorted(train_run_keys))
    test_run_str = ",".join(sorted(test_run_keys))

    for module_id, parcel_idx in module_index_map.items():
        display_name = _module_display_name(pool, module_id)
        _emit_log(
            log_fn,
            f"  Encoding {module_id} ({display_name}, {len(parcel_idx)} parcels)...",
        )
        result = fit_ridge_encoding_fn(
            x_train,
            y_train[:, parcel_idx],
            x_test,
            y_test[:, parcel_idx],
            gap_trs=cfg.gap_trs,
        )
        result["display_name"] = display_name
        result["validation_type"] = "multi_episode_cross_run"
        result["train_run"] = train_run_str
        result["test_run"] = test_run_str
        results[module_id] = result
        _log_encoding_result(log_fn, module_id, result)

    output_path = output_dir / "encoding_results.json"
    save_encoding_results_fn(results, output_path)
    _emit_log(log_fn, f"  Saved encoding results to {output_path}")

    metadata = {
        "train_episodes": train_meta,
        "test_episodes": test_meta,
        "total_train_trs_after_concat": int(x_train.shape[0]),
        "total_test_trs_after_concat": int(x_test.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "n_parcels": int(y_train.shape[1]),
        "lag_trs": cfg.lag_trs,
        "gap_trs": cfg.gap_trs,
    }
    meta_path = output_dir / "training_metadata.json"
    write_json(meta_path, metadata)
    _emit_log(log_fn, f"  Saved training metadata to {meta_path}")


def _module_display_name(pool: ModulePromptPool, module_id: str) -> str:
    """Look up one module's display name."""

    return next(module.display_name for module in pool.modules if module.module_id == module_id)
