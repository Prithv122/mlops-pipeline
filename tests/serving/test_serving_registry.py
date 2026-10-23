from __future__ import annotations

import itertools
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from mlops_pipeline import registry
from mlops_pipeline.model.data import FEATURE_COLS
from mlops_pipeline.registry import (
    CHAMPION,
    LoadedModel,
    load_model,
    log_and_register,
    retrain,
)
from mlops_pipeline.train import TrainResult

_names = itertools.count(1)


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One SQLite store for the module: creating and migrating one costs seconds, so tests
    share it and keep apart by registering under their own model name."""
    return tmp_path_factory.mktemp("registry")


@pytest.fixture(scope="module")
def uri(root: Path) -> str:
    return f"sqlite:///{(root / 'mlflow.db').as_posix()}"


@pytest.fixture
def name() -> str:
    return f"model-{next(_names)}"


@pytest.fixture
def champion_v1(uri: str, name: str, trained: TrainResult) -> str:
    """A model name with one registered version carrying the champion alias."""
    log_and_register(trained, tracking_uri=uri, model_name=name, data_path="no-such-file.csv")
    return name


def test_registers_a_version_and_sets_the_champion_alias(
    uri: str, name: str, trained: TrainResult
) -> None:
    version = log_and_register(trained, tracking_uri=uri, model_name=name, data_path="x")
    assert version == "1"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    assert str(client.get_model_version_by_alias(name, CHAMPION).version) == "1"


def test_artifacts_land_beside_the_database_not_in_the_working_directory(
    root: Path, uri: str, champion_v1: str
) -> None:
    assert any((root / "mlartifacts").rglob("calibrated_model.pkl"))
    assert not Path("mlruns").exists()
    assert not Path("mlartifacts").exists()


def test_load_model_round_trips_predictions_exactly(
    uri: str, champion_v1: str, trained: TrainResult, sample_df: pd.DataFrame
) -> None:
    got = load_model(tracking_uri=uri, model_name=champion_v1)
    assert isinstance(got, LoadedModel)
    assert got.version == "1"
    assert got.run_id
    assert got.metrics == pytest.approx(trained.metrics)
    np.testing.assert_array_equal(got.model.predict(sample_df), trained.model.predict(sample_df))


def test_registered_pyfunc_predicts_like_the_model(
    uri: str, champion_v1: str, trained: TrainResult, sample_df: pd.DataFrame
) -> None:
    mlflow.set_registry_uri(uri)
    pyfunc = mlflow.pyfunc.load_model(f"models:/{champion_v1}@{CHAMPION}")
    rows = sample_df.head(25)[FEATURE_COLS]
    np.testing.assert_array_equal(pyfunc.predict(rows), trained.model.predict(rows))


def test_missing_alias_raises_lookup_error(uri: str, champion_v1: str) -> None:
    with pytest.raises(LookupError, match="no-such-alias"):
        load_model(tracking_uri=uri, model_name=champion_v1, alias="no-such-alias")


def test_registering_without_an_alias_leaves_the_champion_alone(
    uri: str, champion_v1: str, trained: TrainResult
) -> None:
    second = log_and_register(
        trained, tracking_uri=uri, model_name=champion_v1, data_path="x", alias=None
    )
    assert second == "2"
    assert load_model(tracking_uri=uri, model_name=champion_v1).version == "1"


def test_data_hash_and_params_are_logged(
    tmp_path: Path, uri: str, name: str, trained: TrainResult
) -> None:
    data = tmp_path / "d.csv"
    data.write_text("one fixed line of data")
    log_and_register(trained, tracking_uri=uri, model_name=name, data_path=str(data))
    loaded = load_model(tracking_uri=uri, model_name=name)
    run = MlflowClient(tracking_uri=uri, registry_uri=uri).get_run(loaded.run_id)
    assert len(run.data.tags["data_sha256"]) == 64
    assert run.data.params["n_train"] == str(trained.n_train)
    assert run.data.params["lgbm_n_estimators"] == "400"


def test_first_retrain_promotes_unconditionally(
    uri: str, name: str, sample_df: pd.DataFrame
) -> None:
    decision = retrain(sample_df, tracking_uri=uri, model_name=name, seed=10_001)
    assert decision.promoted and decision.champion_version is None
    assert decision.challenger_version == "1"
    assert load_model(tracking_uri=uri, model_name=name).version == "1"


def test_retrain_promotes_a_challenger_inside_the_margins(
    uri: str, champion_v1: str, sample_df: pd.DataFrame
) -> None:
    decision = retrain(
        sample_df, tracking_uri=uri, model_name=champion_v1, seed=10_000
    )  # same seed: same split
    assert decision.promoted
    assert (decision.champion_version, decision.challenger_version) == ("1", "2")
    assert "promoted" in decision.reason
    assert load_model(tracking_uri=uri, model_name=champion_v1).version == "2"


@pytest.mark.parametrize(
    ("champion_scores", "expect_promoted"),
    [
        # champion scores on the shared split; the challenger is the fixture model
        ({"test_auc": 0.99, "test_brier": 0.18}, False),
        ({"test_auc": 0.60, "test_brier": 0.0}, False),
        ({"test_auc": 0.60, "test_brier": 0.99}, True),
    ],
    ids=["auc-too-low", "brier-too-high", "both-within-margin"],
)
def test_retrain_applies_the_promotion_rule(
    uri: str,
    champion_v1: str,
    trained: TrainResult,
    sample_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    champion_scores: dict[str, float],
    expect_promoted: bool,
) -> None:
    monkeypatch.setattr(registry, "train", lambda df, seed: trained)
    monkeypatch.setattr(registry, "_score_champion", lambda champion, test: champion_scores)
    decision = retrain(sample_df, tracking_uri=uri, model_name=champion_v1, seed=10_000)
    assert decision.promoted is expect_promoted
    assert decision.challenger_version == "2" and decision.champion_version == "1"
    live = load_model(tracking_uri=uri, model_name=champion_v1).version
    assert live == ("2" if expect_promoted else "1")
    if not expect_promoted:
        assert "kept champion" in decision.reason


def test_promotion_margins_are_exact(
    uri: str,
    champion_v1: str,
    trained: TrainResult,
    sample_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Challenger just inside each tolerance promotes; just outside does not."""
    real = trained.metrics
    monkeypatch.setattr(registry, "train", lambda df, seed: trained)

    def champion(auc_gap: float, brier_gap: float):
        return lambda c, t: {
            "test_auc": real["test_auc"] + auc_gap,
            "test_brier": real["test_brier"] - brier_gap,
        }

    def decide() -> bool:
        return retrain(sample_df, tracking_uri=uri, model_name=champion_v1, seed=10_000).promoted

    monkeypatch.setattr(registry, "_score_champion", champion(0.004, 0.0015))
    assert decide()
    monkeypatch.setattr(registry, "_score_champion", champion(0.006, 0.0))
    assert not decide()
    monkeypatch.setattr(registry, "_score_champion", champion(0.0, 0.0025))
    assert not decide()
