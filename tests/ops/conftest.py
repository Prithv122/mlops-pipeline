"""Fixtures for the incident ops tests; the fakes themselves live in ops_fakes.py."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from ops_fakes import (
    FakeLog,
    FakeNotifier,
    FakeResponder,
    make_all_stats,
    make_log_frame,
    make_ranking,
)


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
