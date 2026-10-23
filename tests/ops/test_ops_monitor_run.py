from __future__ import annotations

import sys
import types
from datetime import UTC, datetime

import pandas as pd
import pytest
from ops_fakes import T0, T1, FakeLog, make_all_stats, make_log_frame, make_ranking

from mlops_pipeline.incident.webhook import IncidentAlert
from mlops_pipeline.model.data import FEATURE_COLS
from mlops_pipeline.monitor import run as monitor_run


@pytest.fixture
def fake_analyser(monkeypatch):
    """Replace the analyser module so these tests do not depend on the real one."""
    calls: list[pd.DataFrame] = []
    state = {"incident": True}

    def analyse(ref, null, window):
        calls.append(window)
        return make_all_stats(), make_ranking(incident=state["incident"])

    module = types.ModuleType("mlops_pipeline.incident.analyser")
    module.analyse = analyse
    monkeypatch.setitem(sys.modules, "mlops_pipeline.incident.analyser", module)
    return types.SimpleNamespace(calls=calls, state=state)


def run(log, alert, **kwargs):
    return monitor_run.monitor_once(
        log, object(), object(), start=T0, end=T1, alert=alert, **kwargs
    )


def test_alert_raised_only_on_incident(fake_analyser):
    alerts: list[IncidentAlert] = []
    ranking = run(FakeLog(make_log_frame()), alerts.append)
    assert ranking is not None and ranking.incident
    assert len(alerts) == 1
    assert alerts[0].alert_id == "drift-20260101T000000-20260101T010000"
    assert alerts[0].window_start == T0 and alerts[0].window_end == T1
    assert alerts[0].source == "monitor"
    assert list(fake_analyser.calls[0].columns) == list(FEATURE_COLS)


def test_no_alert_when_no_incident(fake_analyser):
    fake_analyser.state["incident"] = False
    alerts: list[IncidentAlert] = []
    ranking = run(FakeLog(make_log_frame()), alerts.append)
    assert ranking is not None and not ranking.incident
    assert alerts == []


def test_empty_window_returns_none_and_never_alerts(fake_analyser):
    empty = pd.DataFrame(columns=[*FEATURE_COLS, "exposure", "p_default", "flag", "ts"])
    alerts: list[IncidentAlert] = []
    assert run(FakeLog(empty), alerts.append) is None
    assert alerts == []
    assert fake_analyser.calls == []


def test_alert_id_is_deterministic_for_naive_and_aware_times(fake_analyser):
    ids: list[str] = []
    for start, end in (
        (datetime(2026, 3, 4, 5, 6, 7), datetime(2026, 3, 4, 6, 6, 7)),
        (datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC), datetime(2026, 3, 4, 6, 6, 7, tzinfo=UTC)),
    ):
        monitor_run.monitor_once(
            FakeLog(make_log_frame()),
            object(),
            object(),
            start=start,
            end=end,
            alert=lambda a: ids.append(a.alert_id),
        )
    assert ids == ["drift-20260304T050607-20260304T060607"] * 2


def test_report_written_only_when_report_dir_given(fake_analyser, monkeypatch, tmp_path):
    written: list[tuple] = []
    module = types.ModuleType("mlops_pipeline.monitor.evidently_report")
    module.write_drift_report = lambda ref, cur, out: written.append((ref, cur, out)) or out
    monkeypatch.setitem(sys.modules, "mlops_pipeline.monitor.evidently_report", module)

    run(FakeLog(make_log_frame()), lambda a: None)
    assert written == []

    reference = pd.DataFrame({"AGE": [1.0]})
    run(FakeLog(make_log_frame()), lambda a: None, report_dir=tmp_path, reference_frame=reference)
    assert len(written) == 1
    assert written[0][0] is reference
    assert written[0][2] == tmp_path
    assert list(written[0][1].columns) == list(FEATURE_COLS)


def test_post_alert_posts_alert_json(monkeypatch):
    seen = {}

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = request.data
        seen["timeout"] = timeout
        seen["method"] = request.get_method()
        return Reply()

    monkeypatch.setattr(monitor_run.urllib.request, "urlopen", fake_urlopen)
    alert = IncidentAlert(alert_id="a-1", window_start=T0, window_end=T1)
    monitor_run.post_alert("http://localhost:8000/incident/webhook")(alert)
    assert seen["url"] == "http://localhost:8000/incident/webhook"
    assert seen["method"] == "POST"
    assert seen["timeout"] == 10
    assert b'"alert_id":"a-1"' in seen["body"].replace(b" ", b"")
