# Vendored from github.com/Prithv122/fraud-calibrated @ ebf6770, src/fraud_calibrated/model.py
# (MIT, same author). Imports rewritten to this package; logic unchanged.
"""Model training: a LightGBM classifier plus a plain logistic-regression baseline.

Class imbalance (22.1% positive, 3.5:1) is handled with LightGBM's built-in
``scale_pos_weight``, not resampling. SMOTE-style oversampling is the more commonly
blogged answer, but it synthesises fake customers in a feature space with genuinely
discrete columns (PAY_1..PAY_6 are ordinal repayment-status codes, not continuous) --
interpolating between two real customers' repayment histories does not obviously
produce a customer that could exist. Reweighting the loss changes nothing about what
the rows *mean*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_pipeline.model.data import CATEGORICAL_COLS, FEATURE_COLS, TARGET


@dataclass(frozen=True)
class Dataset:
    """Feature matrix / label vector pair, keeping the frame's column names."""

    x: pd.DataFrame
    y: np.ndarray


def make_dataset(df: pd.DataFrame) -> Dataset:
    """Slice a cleaned frame down to model features and target."""
    x = df[FEATURE_COLS].copy()
    for col in CATEGORICAL_COLS:
        x[col] = x[col].astype("category")
    y = df[TARGET].to_numpy()
    return Dataset(x=x, y=y)


def scale_pos_weight(y: np.ndarray) -> float:
    """Ratio of negatives to positives -- LightGBM's recommended imbalance weight."""
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0:
        raise ValueError("no positive examples in y")
    return negatives / positives


def fit_lightgbm(train: Dataset, *, seed: int = 0, **overrides: object) -> LGBMClassifier:
    """Fit the primary model. Categorical columns are passed as pandas ``category``
    dtype so LightGBM handles them natively -- no one-hot expansion, no ordinal
    encoding that would invent a false ordering between e.g. marital-status codes.
    """
    params: dict[str, object] = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight(train.y),
        "random_state": seed,
        "verbosity": -1,
    }
    params.update(overrides)
    model = LGBMClassifier(**params)
    model.fit(train.x, train.y, categorical_feature=CATEGORICAL_COLS)
    return model


def fit_baseline(train: Dataset, *, seed: int = 0) -> Pipeline:
    """Plain logistic regression on one-hot categoricals, class-balanced.

    Exists purely as a reference point: if LightGBM cannot clear this by a healthy
    margin on AUC/PR-AUC, the extra complexity in the README is not earning its
    keep.
    """
    x = pd.get_dummies(train.x, columns=CATEGORICAL_COLS, drop_first=True)
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model = Pipeline(steps=[("scale", StandardScaler()), ("logreg", logreg)])
    model.fit(x, train.y)
    # Stash the training columns so predict_baseline can re-align a differently
    # ordered / differently-categoried frame at inference time.
    model.feature_names_in_baseline_ = list(x.columns)
    return model


def predict_baseline(model: Pipeline, data: Dataset) -> np.ndarray:
    """Score a frame with the baseline, re-aligning one-hot columns to training time."""
    x = pd.get_dummies(data.x, columns=CATEGORICAL_COLS, drop_first=True)
    x = x.reindex(columns=model.feature_names_in_baseline_, fill_value=0)
    return model.predict_proba(x)[:, 1]


def predict_proba(model: LGBMClassifier, data: Dataset) -> np.ndarray:
    """Score a frame with the primary model. Returns P(default)."""
    return model.predict_proba(data.x)[:, 1]
