from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from httpx import ASGITransport

from mlops_pipeline.model.costs import CostModel, decide
from mlops_pipeline.model.data import FEATURE_COLS
from mlops_pipeline.predlog import PredictionLog
from mlops_pipeline.registry import LoadedModel, log_and_register
from mlops_pipeline.serve.app import create_app
from mlops_pipeline.serve.settings import Settings, get_settings
from mlops_pipeline.train import TrainResult

pytestmark = pytest.mark.anyio

ALL_TIME = (datetime(2000, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC))


def _payload(df: pd.DataFrame, i: int = 0, exposure: float = 5_000.0) -> dict:
    row = df.iloc[i]
    return {**{c: float(row[c]) for c in FEATURE_COLS}, "exposure": exposure}


def _app(tmp_path: Path, loaded: LoadedModel, **kw):
    settings = Settings(predlog_path=str(tmp_path / "pred.sqlite"))
    return create_app(settings, model=loaded, **kw), PredictionLog(settings.predlog_path)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health(tmp_path: Path, loaded: LoadedModel) -> None:
    app, _ = _app(tmp_path, loaded)
    async with _client(app) as c:
        r = await c.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


async def test_model_endpoint(tmp_path: Path, loaded: LoadedModel) -> None:
    app, _ = _app(tmp_path, loaded)
    async with _client(app) as c:
        body = (await c.get("/model")).json()
    assert body["name"] == "credit-default" and body["alias"] == "champion"
    assert body["version"] == "7" and body["run_id"] == "run-abc"
    assert body["metrics"] == pytest.approx(loaded.metrics)


async def test_predict_returns_probability_flag_and_version(
    tmp_path: Path, loaded: LoadedModel, sample_df: pd.DataFrame
) -> None:
    app, _ = _app(tmp_path, loaded)
    payload = _payload(sample_df, 3, exposure=250_000.0)
    async with _client(app) as c:
        r = await c.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"p_default", "flag", "model_version"}
    assert 0.0 <= body["p_default"] <= 1.0
    assert body["model_version"] == "7"
    expected_p = loaded.model.predict(sample_df.iloc[[3]])[0]
    assert body["p_default"] == pytest.approx(expected_p, abs=1e-12)
    expected_flag = decide(np.array([expected_p]), np.array([250_000.0]), CostModel())[0]
    assert body["flag"] is bool(expected_flag)


async def test_flag_follows_the_cost_rule(
    tmp_path: Path, loaded: LoadedModel, sample_df: pd.DataFrame
) -> None:
    """Zero exposure can never justify an intervention; a huge one flags any real risk."""
    app, _ = _app(tmp_path, loaded)
    async with _client(app) as c:
        zero = (await c.post("/predict", json=_payload(sample_df, 0, exposure=0.0))).json()
        # find a row the model scores clearly above zero
        probs = loaded.model.predict(sample_df.head(200))
        risky = int(np.argmax(probs))
        big = (await c.post("/predict", json=_payload(sample_df, risky, exposure=1e9))).json()
    assert zero["flag"] is False or zero["p_default"] >= 1.0
    assert big["flag"] is True


async def test_every_call_appends_one_log_row(
    tmp_path: Path, loaded: LoadedModel, sample_df: pd.DataFrame
) -> None:
    app, log = _app(tmp_path, loaded)
    before = datetime.now(UTC) - timedelta(seconds=1)
    async with _client(app) as c:
        for i in range(3):
            assert (await c.post("/predict", json=_payload(sample_df, i))).status_code == 200
    rows = log.window(before, datetime.now(UTC) + timedelta(seconds=1))
    assert len(rows) == 3 == log.count()
    assert (rows["model_version"] == "7").all()
    assert rows["exposure"].tolist() == [5_000.0] * 3
    assert rows["AGE"].tolist() == [float(sample_df.loc[i, "AGE"]) for i in range(3)]
    assert rows["p_default"].between(0, 1).all()


async def test_null_features_are_accepted_and_logged_as_nan(
    tmp_path: Path, loaded: LoadedModel, sample_df: pd.DataFrame
) -> None:
    app, log = _app(tmp_path, loaded)
    payload = _payload(sample_df, 1) | {"PAY_2": None, "EDUCATION": None, "BILL_AMT4": None}
    async with _client(app) as c:
        r = await c.post("/predict", json=payload)
    assert r.status_code == 200 and 0.0 <= r.json()["p_default"] <= 1.0
    row = log.window(*ALL_TIME).iloc[0]
    assert np.isnan(row["PAY_2"]) and np.isnan(row["EDUCATION"]) and np.isnan(row["BILL_AMT4"])
    assert row["AGE"] == float(sample_df.loc[1, "AGE"])


async def test_negative_exposure_is_rejected_and_not_logged(
    tmp_path: Path, loaded: LoadedModel, sample_df: pd.DataFrame
) -> None:
    app, log = _app(tmp_path, loaded)
    async with _client(app) as c:
        r = await c.post("/predict", json=_payload(sample_df, 0, exposure=-1.0))
    assert r.status_code == 422
    assert log.count() == 0


@pytest.mark.parametrize("drop", ["exposure", "AGE"])
async def test_missing_fields_are_rejected(
    tmp_path: Path, loaded: LoadedModel, sample_df: pd.DataFrame, drop: str
) -> None:
    app, log = _app(tmp_path, loaded)
    payload = _payload(sample_df)
    del payload[drop]
    async with _client(app) as c:
        r = await c.post("/predict", json=payload)
    assert r.status_code == 422 and log.count() == 0


async def test_non_numeric_and_non_finite_values_are_rejected(
    tmp_path: Path, loaded: LoadedModel, sample_df: pd.DataFrame
) -> None:
    app, log = _app(tmp_path, loaded)
    async with _client(app) as c:
        bad_text = await c.post("/predict", json=_payload(sample_df) | {"AGE": "old"})
        nan = await c.post(
            "/predict",
            content=json.dumps(_payload(sample_df) | {"exposure": float("nan")}),
            headers={"content-type": "application/json"},
        )
    assert bad_text.status_code == 422 and nan.status_code == 422
    assert log.count() == 0


async def test_extra_routers_and_responder_are_wired(tmp_path: Path, loaded: LoadedModel) -> None:
    extra = APIRouter()

    @extra.get("/ping")
    def ping() -> dict[str, str]:
        return {"pong": "yes"}

    marker = object()
    app, _ = _app(tmp_path, loaded, responder=marker, extra_routers=[extra])
    assert app.state.responder is marker
    assert app.state.settings.model_alias == "champion"
    async with _client(app) as c:
        assert (await c.get("/ping")).json() == {"pong": "yes"}


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=None)  # a developer's own .env must not change the defaults
    assert (s.tracking_uri, s.model_name, s.model_alias) == (
        "sqlite:///mlstore/mlflow.db",
        "credit-default",
        "champion",
    )
    assert s.slack_webhook_url is None
    monkeypatch.setenv("MLOPS_MODEL_ALIAS", "challenger")
    monkeypatch.setenv("MLOPS_TRACKING_URI", "sqlite:////app/mlstore/mlflow.db")
    monkeypatch.setenv("MLOPS_SLACK_WEBHOOK_URL", "https://hooks.example.test/x")
    s = Settings(_env_file=None)
    assert s.model_alias == "challenger"
    assert s.tracking_uri == "sqlite:////app/mlstore/mlflow.db"
    assert s.slack_webhook_url == "https://hooks.example.test/x"


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()


def test_lifespan_loads_the_registered_champion(
    tmp_path: Path, trained: TrainResult, sample_df: pd.DataFrame
) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    version = log_and_register(trained, tracking_uri=uri, data_path="x")
    settings = Settings(tracking_uri=uri, predlog_path=str(tmp_path / "pred.sqlite"))
    app = create_app(settings)
    assert app.state.model is None  # nothing loaded until startup
    with TestClient(app) as client:
        info = client.get("/model").json()
        assert info["version"] == version and info["run_id"]
        r = client.post("/predict", json=_payload(sample_df, 2))
        assert r.status_code == 200 and r.json()["model_version"] == version
    assert PredictionLog(settings.predlog_path).count() == 1


def test_startup_fails_without_a_champion(tmp_path: Path) -> None:
    uri = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
    app = create_app(Settings(tracking_uri=uri, predlog_path=str(tmp_path / "pred.sqlite")))
    with pytest.raises(LookupError), TestClient(app):
        pass


def test_blank_values_in_a_copied_env_file_keep_the_defaults(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("MLOPS_TRACKING_URI=\nMLOPS_MODEL_ALIAS=challenger\n", encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.tracking_uri == "sqlite:///mlstore/mlflow.db"
    assert s.model_alias == "challenger"
