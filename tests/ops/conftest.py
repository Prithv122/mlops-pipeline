"""Hand-made fakes for the incident ops layer: no model, no registry, no network."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlops_pipeline.contracts import ColumnStats, Hypothesis, Ranking
from mlops_pipeline.model.data import FEATURE_COLS

T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)


def make_stats(column: str, **overrides: float | bool) -> ColumnStats:
    base = {
        "psi": 0.1,
        "ks": 0.05,
        "null_delta": 0.0,
        "out_of_range": 0.0,
        "unseen_share": 0.0,
        "median_log_ratio": 0.0,
        "code_set_changed": False,
        "became_constant": False,
    }
    base.update(overrides)
    return ColumnStats(column=column, **base)  # type: ignore[arg-type]


def make_ranking(incident: bool = True) -> Ranking:
    hyps = (
        Hypothesis("unit_break", "BILL_AMT1", 412.5),
        Hypothesis("covariate_shift", "BILL_AMT1", 80.25),
        Hypothesis("unit_break", "BILL_AMT2", 70.0),
        Hypothesis("null_spike", "PAY_1", 9.5),
        Hypothesis("recode", "EDUCATION", 3.0),
        Hypothesis("covariate_shift", "AGE", 1.0),
    )
    return Ranking(method="ranker", hypotheses=hyps, incident=incident, threshold=12.5)


def make_all_stats() -> dict[str, ColumnStats]:
    out = {c: make_stats(c) for c in FEATURE_COLS}
    out["BILL_AMT1"] = make_stats("BILL_AMT1", psi=3.2, out_of_range=0.9, median_log_ratio=2.9)
    out["BILL_AMT2"] = make_stats("BILL_AMT2", psi=2.1)
    out["PAY_1"] = make_stats("PAY_1", null_delta=0.25)
    return out


def make_log_frame(n: int = 5, *, extra: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(10_000)
    frame = pd.DataFrame(
        {c: rng.normal(size=n).astype(float) for c in reversed(FEATURE_COLS)}
    )  # reversed on purpose: the collector must restore FEATURE_COLS order
    frame.loc[0, "PAY_1"] = np.nan
    if extra:
        frame["exposure"] = 1.0
        frame["p_default"] = 0.2
        frame["flag"] = False
        frame["ts"] = "2026-01-01T00:10:00+00:00"
        frame["model_version"] = "1"
    return frame


class FakeLog:
    """Stands in for PredictionLog: returns a fixed frame and records the window asked for."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[tuple[datetime, datetime]] = []

    def window(self, start: datetime, end: datetime) -> pd.DataFrame:
        self.calls.append((start, end))
        return self.frame.copy()


class FakeResponder:
    def __init__(self, ranking: Ranking | None = None) -> None:
        self.ranking = ranking or make_ranking()
        self.calls = 0

    def respond(self, window: pd.DataFrame):
        self.calls += 1
        return make_all_stats(), self.ranking


class FakeNotifier:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def notify(self, payload: dict) -> None:
        self.payloads.append(payload)


@pytest.fixture
def stats():
    return make_all_stats()


@pytest.fixture
def ranking():
    return make_ranking()


@pytest.fixture
def make_state(tmp_path):
    """Build the app.state attributes the webhook reads."""

    def build(frame: pd.DataFrame | None = None, responder: FakeResponder | None = None):
        return SimpleNamespace(
            settings=SimpleNamespace(postmortem_dir=str(tmp_path / "postmortems")),
            responder=responder or FakeResponder(),
            predlog=FakeLog(make_log_frame() if frame is None else frame),
            notifier=FakeNotifier(),
        )

    return build
