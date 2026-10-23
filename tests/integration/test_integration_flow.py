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


# ----------------------------------------------------------------- run guard and provenance


def _no_window_drawn(monkeypatch) -> None:
    """Make any attempt to evaluate a window fail the test."""
    from mlops_pipeline import evaluate

    def boom(*args, **kwargs):
        raise AssertionError("a guard let the evaluation start")

    monkeypatch.setattr(evaluate, "run_evaluation", boom)


def _full_args(out: Path, *extra: str) -> list[str]:
    return ["eval-responder", "--full", "--data", str(FIXTURE), "--out", str(out), *extra]


def test_full_evaluation_refuses_another_window_size(tmp_path: Path, monkeypatch) -> None:
    from mlops_pipeline import cli

    monkeypatch.setattr(cli, "_git_state", lambda: ("abc", True))
    monkeypatch.setattr(cli, "PROTOCOL_DATA_SHA256", cli._file_sha256(FIXTURE))
    _no_window_drawn(monkeypatch)
    assert main(_full_args(tmp_path, "--window-size", "500")) == 2
    assert not (tmp_path / "records.csv").exists()


def test_full_evaluation_refuses_data_that_is_not_the_vendored_cache(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from mlops_pipeline import cli

    monkeypatch.setattr(cli, "_git_state", lambda: ("abc", True))
    _no_window_drawn(monkeypatch)
    # the fixture is not the 30,000-row cache the protocol was frozen on
    assert main(_full_args(tmp_path, "--window-size", "1000")) == 2
    assert "hash differs" in capsys.readouterr().out
    assert not (tmp_path / "records.csv").exists()


def test_the_pinned_hash_is_the_vendored_caches() -> None:
    from mlops_pipeline import cli

    cache = Path(__file__).resolve().parents[2] / "data" / "credit_default.csv.gz"
    assert cli._file_sha256(cache) == cli.PROTOCOL_DATA_SHA256


def test_full_evaluation_never_overwrites_an_earlier_result(tmp_path: Path, monkeypatch) -> None:
    from mlops_pipeline import cli

    monkeypatch.setattr(cli, "_git_state", lambda: ("abc", True))
    monkeypatch.setattr(cli, "PROTOCOL_DATA_SHA256", cli._file_sha256(FIXTURE))
    _no_window_drawn(monkeypatch)
    earlier = tmp_path / "records.csv"
    earlier.write_text("the one-time result", encoding="utf-8")
    assert main(_full_args(tmp_path, "--window-size", "1000")) == 2
    assert earlier.read_text(encoding="utf-8") == "the one-time result"


def test_full_evaluation_starts_when_every_guard_passes(tmp_path: Path, monkeypatch) -> None:
    from mlops_pipeline import cli, evaluate

    class Reached(Exception):
        pass

    def stop(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(cli, "_git_state", lambda: ("abc", True))
    monkeypatch.setattr(cli, "PROTOCOL_DATA_SHA256", cli._file_sha256(FIXTURE))
    monkeypatch.setattr(evaluate, "run_evaluation", stop)
    with pytest.raises(Reached):
        main(_full_args(tmp_path, "--window-size", "1000"))


def test_full_evaluation_with_the_dirty_override_records_it_in_the_report(
    tmp_path: Path, monkeypatch
) -> None:
    from mlops_pipeline import cli, evaluate
    from mlops_pipeline.incident.faults import dev_grid

    cached = evaluate.run_evaluation(pd.read_csv(FIXTURE), dev_grid(1, 2), window_size=WINDOW)
    monkeypatch.setattr(cli, "_git_state", lambda: ("abc", False))
    monkeypatch.setattr(cli, "PROTOCOL_DATA_SHA256", cli._file_sha256(FIXTURE))
    monkeypatch.setattr(evaluate, "run_evaluation", lambda *a, **k: cached)
    assert main(_full_args(tmp_path, "--window-size", "1000", "--allow-dirty")) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["provenance"]["mode"] == "full"
    assert summary["provenance"]["tree_clean"] is False
    assert summary["provenance"]["allow_dirty"] is True
    assert "tree was not clean" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")


def test_smoke_provenance_carries_data_hash_and_versions(tmp_path: Path) -> None:
    import hashlib
    import platform

    args = ["eval-responder", "--smoke", "--data", str(FIXTURE), "--out", str(tmp_path)]
    args += ["--window-size", str(WINDOW), "--dev-seeds", "1", "--dev-controls", "2"]
    assert main(args) == 0
    prov = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["provenance"]
    assert prov["data_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert prov["data"] == str(FIXTURE)
    assert prov["window_size"] == WINDOW
    assert prov["allow_dirty"] is False
    assert prov["python"] == platform.python_version()
    assert set(prov["versions"]) == {"numpy", "pandas", "scipy", "mlops-pipeline"}
    assert all(v and v != "unknown" for v in prov["versions"].values())
    assert {"commit", "tree_clean", "started_utc", "finished_utc"} <= set(prov)
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "## Provenance" in report and prov["data_sha256"] in report
    assert "Exploratory sensitivity (declared before the run, not a hypothesis test)" in report
    assert "This arm does not change the H1/H2 verdicts above." in report


def test_git_state_runs_git_in_the_repository_root_not_the_current_folder(
    tmp_path: Path, monkeypatch
) -> None:
    import subprocess

    from mlops_pipeline import cli

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out = "deadbeef\n" if "rev-parse" in cmd else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=out)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    root = cli._repo_root()
    assert root is not None and (root / "pyproject.toml").is_file()
    assert cli._git_state() == ("deadbeef", True)
    assert len(calls) == 2
    for cmd in calls:
        assert cmd[:3] == ["git", "-C", str(root)]


# ------------------------------------------------------------- CLI: retrain, monitor, responder


def test_retrain_passes_the_data_path_for_provenance(monkeypatch) -> None:
    from mlops_pipeline import registry
    from mlops_pipeline.registry import RetrainDecision

    seen: dict = {}

    def fake_retrain(df, **kwargs):
        seen.update(kwargs)
        return RetrainDecision(True, "2", "1", "ok")

    monkeypatch.setattr(registry, "retrain", fake_retrain)
    assert main(["retrain", "--data", str(FIXTURE), "--tracking-uri", "sqlite:///x.db"]) == 0
    assert seen["data_path"] == str(FIXTURE)


def _null_stub(window_size: int):
    from mlops_pipeline.incident.ranker import SIGNALS, NullModel

    keys = [(s, c) for s in SIGNALS for c in FEATURE_COLS]
    return NullModel({k: 0.0 for k in keys}, {k: 1.0 for k in keys}, 1.0, window_size=window_size)


def test_cli_responder_refits_when_the_saved_model_is_for_another_file_or_window(
    tmp_path: Path, monkeypatch
) -> None:
    import dataclasses
    from types import SimpleNamespace

    from mlops_pipeline import cli, evaluate
    from mlops_pipeline.incident.ranker import NullModel
    from mlops_pipeline.monitor.stats import profile

    fits: list[int] = []

    def fake_fit(df, *, window_size=1000, progress=None):
        fits.append(window_size)
        return profile(split(df, seed=0).train), _null_stub(window_size)

    monkeypatch.setattr(evaluate, "fit_responder", fake_fit)
    settings = SimpleNamespace(reference_path=str(FIXTURE))
    saved = tmp_path / "null.json"
    digest = cli._file_sha256(FIXTURE)

    *_, responder = cli._responder(settings, str(saved), window_size=200)
    assert fits == [200] and responder.min_rows == 200
    stored = NullModel.from_json(saved)
    assert (stored.reference_sha256, stored.window_size) == (digest, 200)

    cli._responder(settings, str(saved), window_size=200)
    assert fits == [200]  # same file and size: the saved model is trusted

    *_, responder = cli._responder(settings, str(saved), window_size=100)
    assert fits == [200, 100] and responder.min_rows == 100  # another window size: refitted
    assert NullModel.from_json(saved).window_size == 100  # and overwritten

    current = NullModel.from_json(saved)
    dataclasses.replace(current, reference_sha256="0" * 64).to_json(saved)
    cli._responder(settings, str(saved), window_size=100)
    assert fits == [200, 100, 100]  # another reference file: refitted
    assert NullModel.from_json(saved).reference_sha256 == digest

    # a calibration written before the hash existed has none, so it is not trusted either
    dataclasses.replace(current, reference_sha256=None).to_json(saved)
    cli._responder(settings, str(saved), window_size=100)
    assert len(fits) == 4


def test_cli_monitor_reports_insufficient_rows_with_the_minimum(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from types import SimpleNamespace

    from mlops_pipeline import cli
    from mlops_pipeline.predlog import PredictionLog
    from mlops_pipeline.serve import settings as settings_module

    path = tmp_path / "predlog.sqlite"
    log = PredictionLog(path)
    for minute in range(3):
        log.append(
            ts=datetime(2026, 1, 1, 0, minute, tzinfo=UTC),
            model_version="1",
            features={c: 1.0 for c in FEATURE_COLS},
            exposure=1.0,
            p_default=0.2,
            flag=False,
        )
    monkeypatch.setattr(
        settings_module, "get_settings", lambda: SimpleNamespace(predlog_path=str(path))
    )
    monkeypatch.setattr(cli, "_responder", lambda *a, **k: (None, object(), _null_stub(10), None))
    args = ["monitor", "--start", "2026-01-01T00:00:00", "--end", "2026-01-01T01:00:00"]
    assert main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "insufficient_rows"
    assert (out["rows"], out["min_rows"]) == (3, 10)
    assert out["incident"] is False and out["alerts"] == [] and out["top"] is None
