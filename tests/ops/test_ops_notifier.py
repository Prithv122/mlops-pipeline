from __future__ import annotations

import io
import json

from ops_fakes import T0, T1, make_ranking

from mlops_pipeline.contracts import Hypothesis, Ranking
from mlops_pipeline.incident.notifier import JsonNotifier, SlackNotifier, alert_payload
from mlops_pipeline.incident.webhook import IncidentAlert

ALERT = IncidentAlert(alert_id="drift-test-1", window_start=T0, window_end=T1)
URL = "https://hooks.example.test/services/SECRET"


def test_json_notifier_writes_one_parseable_line():
    stream = io.StringIO()
    payload = alert_payload(ALERT, make_ranking(), "postmortems/drift-test-1.md")
    JsonNotifier(stream).notify(payload)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["alert_id"] == "drift-test-1"
    assert parsed["incident"] is True
    assert [t["column"] for t in parsed["top"]] == [
        "BILL_AMT1",
        "BILL_AMT2",
        "PAY_1",
    ]  # one cause per column
    assert [t["column_score"] for t in parsed["top"]] == [412.5, 70.0, 9.5]


def test_payload_carries_the_column_score_apart_from_the_diagnosed_evidence():
    ranking = Ranking(
        method="ranker",
        hypotheses=(
            Hypothesis("null_spike", "PAY_1", 15.0),
            Hypothesis("covariate_shift", "PAY_1", 1000.0),
        ),
        incident=True,
        threshold=12.5,
    )
    top = alert_payload(ALERT, ranking, None)["top"]
    assert top == [
        {"cause": "null_spike", "column": "PAY_1", "evidence": 15.0, "column_score": 1000.0}
    ]
    section = SlackNotifier(URL).build_payload(ALERT, ranking, None)["blocks"][1]["text"]["text"]
    assert "cause evidence 15.00, column score 1000.00" in section


def test_json_notifier_defaults_to_stdout(capsys):
    JsonNotifier().notify({"a": 1})
    assert json.loads(capsys.readouterr().out) == {"a": 1}


def test_slack_payload_shape():
    payload = SlackNotifier(URL, transport=lambda u, p: None).build_payload(
        ALERT, make_ranking(), "postmortems/drift-test-1.md"
    )
    kinds = [b["type"] for b in payload["blocks"]]
    assert kinds == ["header", "section", "context"]
    section = payload["blocks"][1]["text"]["text"]
    assert section.count("\n") == 3  # heading line + three causes
    assert "unit_break" in section and "BILL_AMT1" in section and "recode" not in section
    context = payload["blocks"][2]["elements"][0]["text"]
    assert T0.isoformat() in context and T1.isoformat() in context
    assert "postmortems/drift-test-1.md" in context
    assert payload["text"].startswith("Drift alert drift-test-1")


def test_slack_notify_calls_the_transport_once():
    calls: list[tuple[str, dict]] = []
    notifier = SlackNotifier(URL, transport=lambda u, p: calls.append((u, p)))
    payload = notifier.build_payload(ALERT, make_ranking(), None)
    notifier.notify(payload)
    assert calls == [(URL, payload)]


def test_slack_notifier_never_shows_the_webhook_url():
    assert "SECRET" not in repr(SlackNotifier(URL))
