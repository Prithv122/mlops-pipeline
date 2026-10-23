"""The parts working together: train, register, serve, log, respond, write, notify.

One real model on the 3,000-row fixture, a real MLflow registry in a temporary SQLite file, the
real responder calibrated on the fixture, and a stream of requests in which one column breaks.
Nothing is faked except the notifier's output stream.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from mlops_pipeline import train as train_module
from mlops_pipeline.cli import main
from mlops_pipeline.evaluate import fit_responder
from mlops_pipeline.incident.analyser import analyse
from mlops_pipeline.incident.notifier import JsonNotifier
from mlops_pipeline.model.data import FEATURE_COLS, split
from mlops_pipeline.registry import load_model, log_and_register
from mlops_pipeline.serve.app import create_app
from mlops_pipeline.serve.settings import Settings
from mlops_pipeline.train import train

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "credit_sample.csv.gz"
WINDOW = 200


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_booster():
    """Same model, one thread: LightGBM's spinning threads crawl on a busy or small machine."""
    patch = pytest.MonkeyPatch()
    patch.setattr(train_module, "fit_lightgbm", partial(train_module.fit_lightgbm, n_jobs=1))
    yield
    patch.undo()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


@pytest.fixture(scope="module")
def responder(df: pd.DataFrame):
    ref, null = fit_responder(df, window_size=WINDOW)

    class Responder:
        def respond(self, window: pd.DataFrame):
            return analyse(ref, null, window)

    return Responder()


def _requests(rows: pd.DataFrame) -> list[dict]:
    out = []
    for _, row in rows.iterrows():
        body = {c: (None if pd.isna(row[c]) else float(row[c])) for c in FEATURE_COLS}
        body["exposure"] = 1000.0
        out.append(body)
    return out


@pytest.mark.anyio
async def test_a_broken_column_flows_from_requests_to_a_postmortem(
    tmp_path: Path, df: pd.DataFrame, responder
) -> None:
    tracking = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    version = log_and_register(
        train(df, seed=10_000), tracking_uri=tracking, data_path=str(FIXTURE)
    )
    settings = Settings(
        tracking_uri=tracking,
        predlog_path=str(tmp_path / "predlog.sqlite"),
        postmortem_dir=str(tmp_path / "postmortems"),
    )
    app = create_app(settings, model=load_model(tracking_uri=tracking), responder=responder)
    stream = io.StringIO()
    app.state.notifier = JsonNotifier(stream)

    # An upstream fault: half of the rows arrive with PAY_3 missing.
    rows = split(df, seed=0).test.sample(WINDOW, random_state=10_000).reset_index(drop=True)
    broken = rows.index[rows.index % 2 == 0]
    rows.loc[broken, "PAY_3"] = np.nan

    start = datetime.now(UTC) - timedelta(seconds=1)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for body in _requests(rows):
            r = await client.post("/predict", json=body)
            assert r.status_code == 200
            assert r.json()["model_version"] == version
        end = datetime.now(UTC) + timedelta(seconds=1)

        alert = {
            "alert_id": "drift-it-1",
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
        }
        r = await client.post("/incident/webhook", json=alert)
        assert r.status_code == 200
        body = r.json()
        again = await client.post("/incident/webhook", json=alert)
        assert again.json() == body

    assert body["rows"] == WINDOW
    assert body["incident"] is True
    assert body["top"][0] == {**body["top"][0], "cause": "null_spike", "column": "PAY_3"}
    postmortem = Path(body["postmortem_path"]).read_text(encoding="utf-8")
    assert "null_spike" in postmortem and "PAY_3" in postmortem
    notifications = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(notifications) == 1  # the repeated alert did not notify twice
    assert notifications[0]["alert_id"] == "drift-it-1"


def test_cli_prepare_writes_the_split(tmp_path: Path) -> None:
    assert main(["prepare", "--data", str(FIXTURE), "--out", str(tmp_path)]) == 0
    sizes = {p.name: len(pd.read_csv(p)) for p in tmp_path.glob("*.csv.gz")}
    assert sizes == {"train.csv.gz": 1800, "valid.csv.gz": 600, "test.csv.gz": 600}


def test_cli_inject_refuses_evaluation_seeds(tmp_path: Path) -> None:
    out = tmp_path / "w.csv"
    code = main(
        [
            "inject",
            "--fault",
            "null_spike",
            "--seed",
            "3",
            "--data",
            str(FIXTURE),
            "--out",
            str(out),
        ]
    )
    assert code == 2
    assert not out.exists()


def test_cli_inject_writes_a_development_window(tmp_path: Path) -> None:
    out = tmp_path / "w.csv"
    args = ["inject", "--fault", "unit_break", "--severity", "2", "--seed", "10001"]
    args += ["--data", str(FIXTURE), "--window-size", "100", "--out", str(out)]
    assert main(args) == 0
    window = pd.read_csv(out)
    assert len(window) == 100
    assert window["BILL_AMT1"].abs().max() > 1e7  # x1000 on every row at severity 2


def test_cli_smoke_evaluation_writes_its_report(tmp_path: Path) -> None:
    args = ["eval-responder", "--smoke", "--data", str(FIXTURE), "--out", str(tmp_path)]
    args += ["--window-size", str(WINDOW), "--dev-seeds", "1", "--dev-controls", "2"]
    assert main(args) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["provenance"]["mode"] == "smoke"
    assert (tmp_path / "REPORT.md").exists()
    assert len(pd.read_csv(tmp_path / "records.csv")) == 4 * 3 + 2


def test_cli_full_evaluation_refuses_a_dirty_tree(tmp_path: Path, monkeypatch) -> None:
    from mlops_pipeline import cli

    monkeypatch.setattr(cli, "_git_state", lambda: ("abc", False))
    args = ["eval-responder", "--full", "--data", str(FIXTURE), "--out", str(tmp_path)]
    assert main(args) == 2
    assert not (tmp_path / "records.csv").exists()
