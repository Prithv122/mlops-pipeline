"""Shared fixtures for the training, registry, log and API tests."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd
import pytest

from mlops_pipeline import train as train_module
from mlops_pipeline.registry import LoadedModel
from mlops_pipeline.train import TrainResult, train

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "credit_sample.csv.gz"


@pytest.fixture(scope="session", autouse=True)
def _single_threaded_booster():
    """Fit with one thread. The model is identical; on a busy machine (or a CI runner with
    fewer cores than OpenMP threads) LightGBM's spin-waiting threads make a 2-second fit
    take minutes. The vendored hyperparameters are untouched."""
    patch = pytest.MonkeyPatch()
    patch.setattr(train_module, "fit_lightgbm", partial(train_module.fit_lightgbm, n_jobs=1))
    yield
    patch.undo()


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


@pytest.fixture(scope="session")
def trained(sample_df: pd.DataFrame) -> TrainResult:
    """One real fit with the vendored default parameters, shared by every test."""
    return train(sample_df, seed=10_000)


@pytest.fixture
def loaded(trained: TrainResult) -> LoadedModel:
    return LoadedModel(
        model=trained.model, version="7", run_id="run-abc", metrics=dict(trained.metrics)
    )
