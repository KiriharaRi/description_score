# ============================================================================
# TEST PIPELINE — Encoding Model (Ridge Baseline)
# ============================================================================
"""Fit ridge regression encoding models: text features → BOLD prediction.

Applies HRF lag, fits per-ROI RidgeCV, evaluates with Pearson r.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# Default gap (in TRs) to leave between train/test and between CV folds to
# avoid BOLD autocorrelation leaking across splits.  ~10 TRs ≈ 15 s at
# TR=1.49 s, which covers the bulk of the hemodynamic autocorrelation.
DEFAULT_SPLIT_GAP_TRS = 10


def apply_hrf_lag(
    features: np.ndarray,
    bold: np.ndarray,
    lag_trs: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift features backward by ``lag_trs`` to account for HRF delay.

    After shifting, both arrays are trimmed to the overlapping range.

    Parameters
    ----------
    features : np.ndarray, shape (n_trs_feat, n_features)
    bold : np.ndarray, shape (n_trs_bold, n_parcels)
    lag_trs : int
        Number of TRs to shift (default 3 ≈ 4.5 s at TR=1.49 s).

    Returns
    -------
    X : np.ndarray, shape (n_valid, n_features)
    Y : np.ndarray, shape (n_valid, n_parcels)
    """

    n_feat = features.shape[0]
    n_bold = bold.shape[0]
    n_expected = n_bold - lag_trs
    n_valid = min(n_feat, n_expected)
    if n_valid <= 0:
        raise ValueError(
            f"Not enough TRs: features={n_feat}, bold={n_bold}, lag={lag_trs}"
        )

    # Direction-sensitive handling:
    #   - BOLD has more usable TRs than features (common in segments-only mode
    #     where total_trs is inferred from max(segment.end_s) and slightly
    #     underestimates): silently trim BOLD tail, only warn.
    #   - features has more TRs than BOLD: something is wrong upstream
    #     (wrong run, wrong TR, wrong episode), so raise loudly.
    if n_feat > n_expected:
        diff = n_feat - n_expected
        if diff > 5:
            raise ValueError(
                f"Features exceed usable BOLD by {diff} TRs "
                f"(features={n_feat}, bold-lag={n_expected}). "
                f"Upstream feature count is wrong — check video duration, TR, "
                f"and BOLD run selection."
            )
        if diff >= 1:
            logger.warning(
                "Features slightly longer than BOLD: features=%d, bold-lag=%d "
                "(diff=%d). Truncating features to %d TRs.",
                n_feat, n_expected, diff, n_valid,
            )
    elif n_expected > n_feat:
        diff = n_expected - n_feat
        logger.warning(
            "BOLD has %d usable TRs but features cover only %d; "
            "trimming BOLD tail by %d TRs.",
            n_expected, n_feat, diff,
        )

    X = features[:n_valid]
    Y = bold[lag_trs:lag_trs + n_valid]
    return X, Y


def fit_ridge_encoding(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    alphas: np.ndarray | None = None,
    gap_trs: int = DEFAULT_SPLIT_GAP_TRS,
) -> dict:
    """Fit RidgeCV and evaluate on held-out data.

    Uses ``TimeSeriesSplit(gap=gap_trs)`` for internal alpha selection so
    adjacent TRs correlated by BOLD autocorrelation do not leak across CV
    folds.

    Parameters
    ----------
    X_train, Y_train : np.ndarray
        Training features and BOLD targets.
    X_test, Y_test : np.ndarray
        Test features and BOLD targets.
    alphas : np.ndarray, optional
        Regularization values to cross-validate over.
    gap_trs : int
        TR gap between adjacent CV folds (default 10, ≈15 s).

    Returns
    -------
    dict with keys:
        pearson_r : np.ndarray, shape (n_parcels,)
        mean_r : float
        best_alpha : float
        n_train, n_test : int
    """

    if alphas is None:
        alphas = np.logspace(-1, 6, 20)

    n_train = X_train.shape[0]
    # TimeSeriesSplit feasibility: (n_splits + 1) * test_size + gap <= n_train.
    # Pick a compact test_size so the gap fits comfortably, then find the
    # largest cv_folds ∈ [2, 5] that satisfies the constraint.
    cv_folds = 0
    test_size_used = 0
    if n_train > gap_trs:
        # Aim for ~8 evaluation windows per fold, clamped to [2, 10].
        test_size = max(2, min(10, (n_train - gap_trs) // 8))
        for candidate in range(5, 1, -1):
            if (candidate + 1) * test_size + gap_trs <= n_train:
                cv_folds = candidate
                test_size_used = test_size
                break

    if cv_folds == 0:
        median_alpha = float(np.median(alphas))
        logger.warning(
            "Only %d training sample(s) with gap_trs=%d; skipping CV, "
            "using fixed alpha=%.2f", n_train, gap_trs, median_alpha,
        )
        model = Ridge(alpha=median_alpha)
    else:
        if cv_folds < 5:
            logger.warning(
                "Training set has %d samples with gap_trs=%d; reducing CV "
                "folds to %d (test_size=%d)",
                n_train, gap_trs, cv_folds, test_size_used,
            )
        cv = TimeSeriesSplit(
            n_splits=cv_folds, gap=gap_trs, test_size=test_size_used,
        )
        model = RidgeCV(alphas=alphas, cv=cv)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    # Per-parcel Pearson r
    n_parcels = Y_test.shape[1]
    r_values = np.full(n_parcels, np.nan)
    for i in range(n_parcels):
        if Y_test[:, i].std() > 0 and Y_pred[:, i].std() > 0:
            r_values[i] = np.corrcoef(Y_test[:, i], Y_pred[:, i])[0, 1]

    best_alpha = float(getattr(model, "alpha_", getattr(model, "alpha", 0.0)))

    return {
        "pearson_r": r_values,
        "mean_r": float(np.nanmean(r_values)),
        "best_alpha": best_alpha,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
    }


def run_encoding(
    features: np.ndarray,
    bold: np.ndarray,
    parcel_indices: np.ndarray | None = None,
    lag_trs: int = 3,
    train_ratio: float = 0.75,
    gap_trs: int = DEFAULT_SPLIT_GAP_TRS,
) -> dict:
    """End-to-end encoding: lag → split → fit → evaluate.

    A temporal gap of ``gap_trs`` is left between the training and test
    portions to prevent BOLD autocorrelation from leaking across the split.

    Parameters
    ----------
    features : np.ndarray, shape (n_trs, n_features)
        TR-aligned feature vectors.
    bold : np.ndarray, shape (n_trs, n_parcels)
        BOLD time series (all parcels or subset).
    parcel_indices : np.ndarray, optional
        If given, select only these columns from *bold*.
    lag_trs : int
        HRF lag in TRs.
    train_ratio : float
        Fraction of TRs for training.
    gap_trs : int
        TR gap between train and test (default 10).

    Returns
    -------
    dict with encoding results.
    """

    if parcel_indices is not None:
        bold = bold[:, parcel_indices]

    X, Y = apply_hrf_lag(features, bold, lag_trs=lag_trs)
    n = X.shape[0]

    if n < 2:
        raise ValueError(
            f"Not enough valid TRs for train/test split: n={n} "
            f"(after lag={lag_trs}). Need at least 2."
        )

    split = max(1, int(n * train_ratio))
    test_start = split + gap_trs
    if test_start >= n:
        # Gap would leave no test samples — fall back to shrinking the gap
        # rather than the training set, since loss of train data hurts fit
        # more than loss of gap hurts leakage.
        test_start = n - 1
        effective_gap = test_start - split
        if effective_gap < gap_trs:
            logger.warning(
                "Requested gap_trs=%d leaves no test samples (n=%d, split=%d); "
                "shrinking gap to %d TRs.", gap_trs, n, split, max(effective_gap, 0),
            )

    n_test = n - test_start
    logger.info(
        "Encoding: %d TRs (train=%d, gap=%d, test=%d), %d features → %d parcels",
        n, split, max(test_start - split, 0), n_test, X.shape[1], Y.shape[1],
    )

    return fit_ridge_encoding(
        X_train=X[:split],
        Y_train=Y[:split],
        X_test=X[test_start:],
        Y_test=Y[test_start:],
        gap_trs=gap_trs,
    )


def cross_run_encoding(
    train_features: np.ndarray,
    train_bold: np.ndarray,
    test_features: np.ndarray,
    test_bold: np.ndarray,
    parcel_indices: np.ndarray | None = None,
    lag_trs: int = 3,
    gap_trs: int = DEFAULT_SPLIT_GAP_TRS,
) -> dict:
    """Cross-run encoding: train on run A, test on run B.

    Each run's features and BOLD are lag-aligned independently,
    so there is no temporal leakage between train and test.  The ``gap_trs``
    parameter still gates CV folds *within* the training run.

    Parameters
    ----------
    train_features, train_bold : np.ndarray
        Training run data.
    test_features, test_bold : np.ndarray
        Test run data (independent run).
    parcel_indices : np.ndarray, optional
        If given, select only these columns from both BOLD arrays.
    lag_trs : int
        HRF lag in TRs.
    gap_trs : int
        Gap used by internal TimeSeriesSplit for alpha selection.

    Returns
    -------
    dict with encoding results (same format as ``fit_ridge_encoding``).
    """

    if parcel_indices is not None:
        train_bold = train_bold[:, parcel_indices]
        test_bold = test_bold[:, parcel_indices]

    X_train, Y_train = apply_hrf_lag(train_features, train_bold, lag_trs)
    X_test, Y_test = apply_hrf_lag(test_features, test_bold, lag_trs)

    logger.info(
        "Cross-run encoding: train=%d TRs, test=%d TRs, %d features → %d parcels",
        X_train.shape[0], X_test.shape[0], X_train.shape[1], Y_train.shape[1],
    )

    return fit_ridge_encoding(
        X_train, Y_train, X_test, Y_test, gap_trs=gap_trs,
    )


def save_encoding_results(
    results: dict[str, dict],
    output_path: str | Path,
) -> None:
    """Save per-ROI encoding results to JSON.

    Parameters
    ----------
    results : dict
        Mapping from ROI name → encoding result dict.
    output_path : str or Path
        Output JSON file path.
    """

    serializable = {}
    for roi_name, res in results.items():
        entry = {
            "mean_pearson_r": res["mean_r"],
            "best_alpha": res["best_alpha"],
            "n_train": res["n_train"],
            "n_test": res["n_test"],
            "n_parcels": len(res["pearson_r"]),
            "per_parcel_r": res["pearson_r"].tolist(),
        }
        if "validation_type" in res:
            entry["validation_type"] = res["validation_type"]
            entry["train_run"] = res.get("train_run", "")
            entry["test_run"] = res.get("test_run", "")
        serializable[roi_name] = entry

    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    logger.info("Saved encoding results to %s", output_path)
