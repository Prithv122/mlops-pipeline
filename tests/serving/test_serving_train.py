from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from mlops_pipeline.model.data import FEATURE_COLS
from mlops_pipeline.train import CalibratedModel, TrainResult


def test_train_reports_sensible_metrics(trained: TrainResult, sample_df: pd.DataFrame) -> None:
    m = trained.metrics
    assert set(m) == {"test_auc", "test_brier", "test_ece", "valid_auc"}
    assert 0.65 < m["test_auc"] < 1.0
    assert 0.65 < m["valid_auc"] < 1.0
    assert 0.0 < m["test_brier"] < 0.25  # 0.25 is what a constant 0.5 would score
    assert 0.0 <= m["test_ece"] < 0.2
    assert trained.n_train + trained.n_valid + trained.n_test == len(sample_df)
    assert (trained.n_train, trained.n_valid, trained.n_test) == (1800, 600, 600)


def test_predictions_are_probabilities(trained: TrainResult, sample_df: pd.DataFrame) -> None:
    p = trained.model.predict(sample_df)
    assert p.shape == (len(sample_df),)
    assert np.isfinite(p).all()
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert p.std() > 0.01  # a constant output would hide a broken calibrator


def test_column_order_does_not_matter(trained: TrainResult, sample_df: pd.DataFrame) -> None:
    rows = sample_df.head(50)
    shuffled = rows[list(reversed(rows.columns))]
    np.testing.assert_array_equal(trained.model.predict(rows), trained.model.predict(shuffled))


def test_extra_columns_are_ignored(trained: TrainResult, sample_df: pd.DataFrame) -> None:
    rows = sample_df.head(20).assign(exposure=1.0, note="x")
    np.testing.assert_array_equal(
        trained.model.predict(rows), trained.model.predict(sample_df.head(20)[FEATURE_COLS])
    )


def test_nulls_in_any_column_do_not_crash(trained: TrainResult, sample_df: pd.DataFrame) -> None:
    rows = sample_df.head(30)[FEATURE_COLS].astype(float)
    holed = rows.copy()
    holed.loc[holed.index[0], ["SEX", "EDUCATION", "MARRIAGE", "PAY_1", "BILL_AMT3"]] = np.nan
    holed["AGE"] = np.nan
    p_clean, p_holed = trained.model.predict(rows), trained.model.predict(holed)
    assert np.isfinite(p_holed).all()
    assert ((p_holed >= 0) & (p_holed <= 1)).all()
    assert not np.array_equal(p_clean, p_holed)  # the nulls were actually used


def test_object_frame_with_none_is_accepted(trained: TrainResult, sample_df: pd.DataFrame) -> None:
    row = sample_df.iloc[0][FEATURE_COLS].to_dict()
    row["PAY_2"] = None
    frame = pd.DataFrame([row], columns=FEATURE_COLS, dtype="object")
    assert 0.0 <= trained.model.predict(frame)[0] <= 1.0


def test_unseen_category_code_scores_as_missing(
    trained: TrainResult, sample_df: pd.DataFrame
) -> None:
    rows = sample_df.head(10)[FEATURE_COLS].astype(float)
    unseen = rows.assign(EDUCATION=99.0)
    missing = rows.assign(EDUCATION=np.nan)
    np.testing.assert_array_equal(trained.model.predict(unseen), trained.model.predict(missing))


def test_single_row_matches_batch(trained: TrainResult, sample_df: pd.DataFrame) -> None:
    """A one-row frame must not infer its own categories and shift the levels."""
    batch = trained.model.predict(sample_df.head(10))
    singles = [trained.model.predict(sample_df.iloc[[i]])[0] for i in range(10)]
    np.testing.assert_allclose(singles, batch, atol=1e-12)


def test_missing_feature_column_is_an_error(trained: TrainResult, sample_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="PAY_3"):
        trained.model.predict(sample_df.head(5).drop(columns=["PAY_3"]))


def test_model_pickles_exactly(trained: TrainResult, sample_df: pd.DataFrame) -> None:
    clone: CalibratedModel = pickle.loads(pickle.dumps(trained.model))
    np.testing.assert_array_equal(clone.predict(sample_df), trained.model.predict(sample_df))
