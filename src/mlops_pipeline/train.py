"""Fit and calibrate the credit-default model, and score it on a held-out frame.

``CalibratedModel`` is the single object the registry stores and the API serves: a LightGBM
booster plus the isotonic map fitted on the validation split. Keeping the two together
matters because the cost rule in :mod:`mlops_pipeline.model.costs` reads the output as a
probability; a raw booster score would be served by mistake the first time someone loaded
only half of the pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from mlops_pipeline.model.calibration import apply_isotonic, fit_isotonic, report
from mlops_pipeline.model.data import CATEGORICAL_COLS, FEATURE_COLS, TARGET, split
from mlops_pipeline.model.model import fit_lightgbm, make_dataset, predict_proba


@dataclass
class CalibratedModel:
    """A booster and the calibrator that turns its scores into probabilities."""

    booster: LGBMClassifier
    calibrator: IsotonicRegression

    def _design_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Reorder to training order and rebuild the categorical dtypes the booster saw.

        LightGBM maps a ``category`` column by position in its stored category list, so a
        frame whose categories were inferred from its own (possibly one-row) contents
        would silently score the wrong level. Pinning the categories to the training ones
        makes unseen codes and nulls both land in "missing" instead.
        """
        missing = [c for c in FEATURE_COLS if c not in frame.columns]
        if missing:
            raise KeyError(f"frame is missing feature columns: {missing}")
        x = frame[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").astype("float64")
        stored = self.booster.booster_.pandas_categorical or []
        for col, categories in zip(CATEGORICAL_COLS, stored, strict=True):
            known = x[col].where(x[col].isin(categories))
            x[col] = pd.Categorical(known, categories=categories)
        return x

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated P(default) for each row; FEATURE_COLS in any order, NaN allowed."""
        raw = self.booster.predict_proba(self._design_matrix(frame))[:, 1]
        return np.clip(apply_isotonic(self.calibrator, raw), 0.0, 1.0)


@dataclass(frozen=True)
class TrainResult:
    model: CalibratedModel
    metrics: dict[str, float]
    n_train: int
    n_valid: int
    n_test: int


def score_model(model: CalibratedModel, frame: pd.DataFrame) -> dict[str, float]:
    """AUC, Brier and ECE of the calibrated probabilities on a labelled frame."""
    y = frame[TARGET].to_numpy()
    prob = model.predict(frame)
    rep = report(y, prob)
    return {
        "test_auc": float(roc_auc_score(y, prob)),
        "test_brier": float(rep.brier),
        "test_ece": float(rep.ece),
    }


def train(df: pd.DataFrame, *, seed: int = 0) -> TrainResult:
    """Split 60/20/20, fit the booster on train, calibrate on valid, report on test.

    ``valid_auc`` is the booster's raw-score AUC on the validation split. It is not taken
    from the calibrated scores because the isotonic map was fitted on those same rows, so
    it would flatter the model.
    """
    parts = split(df, seed=seed)
    train_set = make_dataset(parts.train)
    valid_set = make_dataset(parts.valid)

    booster = fit_lightgbm(train_set, seed=seed)
    raw_valid = predict_proba(booster, valid_set)
    calibrator = fit_isotonic(valid_set.y, raw_valid)
    model = CalibratedModel(booster=booster, calibrator=calibrator)

    metrics = score_model(model, parts.test)
    metrics["valid_auc"] = float(roc_auc_score(valid_set.y, raw_valid))
    sizes = parts.sizes
    return TrainResult(
        model=model,
        metrics={k: metrics[k] for k in ("test_auc", "test_brier", "test_ece", "valid_auc")},
        n_train=sizes["train"],
        n_valid=sizes["valid"],
        n_test=sizes["test"],
    )
