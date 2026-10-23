# Vendored from github.com/Prithv122/fraud-calibrated @ ebf6770, src/fraud_calibrated/calibration.py
# (MIT, same author). Imports rewritten to this package; logic unchanged.
"""Calibration measurement and correction.

A model can rank customers well (good AUC) while its predicted probabilities are
useless as probabilities -- gradient boosting in particular tends to push scores
toward 0 and 1 more aggressively than the true rate warrants. The cost model in
:mod:`mlops_pipeline.model.costs` divides a fixed cost by ``prob_default``, so a badly
calibrated score does not just look wrong on a reliability diagram, it produces the
wrong threshold and therefore the wrong decision. This module measures that gap and
fixes it with isotonic regression fitted on a fold the model never trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class CalibrationReport:
    brier: float
    ece: float
    bin_true_rate: np.ndarray
    bin_pred_mean: np.ndarray
    bin_count: np.ndarray


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and the 0/1 outcome."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    return float(np.mean((prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, prob: np.ndarray, *, n_bins: int = 10
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Equal-width-bin ECE: the count-weighted gap between predicted and true rate.

    Returns ``(ece, bin_true_rate, bin_pred_mean, bin_count)`` so the reliability
    diagram and the scalar summary come from one pass over the data.
    """
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, n_bins - 1)
    true_rate = np.zeros(n_bins)
    pred_mean = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=int)
    ece = 0.0
    n = len(prob)
    for b in range(n_bins):
        mask = bin_idx == b
        count[b] = int(mask.sum())
        if count[b] == 0:
            continue
        true_rate[b] = y_true[mask].mean()
        pred_mean[b] = prob[mask].mean()
        ece += (count[b] / n) * abs(true_rate[b] - pred_mean[b])
    return ece, true_rate, pred_mean, count


def report(y_true: np.ndarray, prob: np.ndarray, *, n_bins: int = 10) -> CalibrationReport:
    """Bundle Brier score and ECE for one score vector."""
    ece, true_rate, pred_mean, count = expected_calibration_error(y_true, prob, n_bins=n_bins)
    return CalibrationReport(
        brier=brier_score(y_true, prob),
        ece=ece,
        bin_true_rate=true_rate,
        bin_pred_mean=pred_mean,
        bin_count=count,
    )


def sklearn_reliability_curve(
    y_true: np.ndarray, prob: np.ndarray, *, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Thin wrapper on scikit-learn's quantile-binned curve, for cross-checking the
    hand-rolled equal-width version above against a library implementation."""
    true_rate, pred_mean = calibration_curve(y_true, prob, n_bins=n_bins, strategy="quantile")
    return true_rate, pred_mean


def fit_isotonic(y_true: np.ndarray, prob: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic (monotone, non-parametric) calibrator.

    Isotonic over Platt/sigmoid scaling because LightGBM's raw-score miscalibration is
    not obviously sigmoid-shaped, and with ~4,800 positives in the validation fold
    there is enough data that isotonic's extra flexibility does not just overfit noise
    (checked empirically in NOTES.md rather than assumed).
    """
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(prob, y_true)
    return calibrator


def apply_isotonic(calibrator: IsotonicRegression, prob: np.ndarray) -> np.ndarray:
    return calibrator.predict(prob)
