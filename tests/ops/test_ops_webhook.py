from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pandas as pd
import pytest
from conftest import FakeNotifier
from fastapi import FastAPI

from mlops_pipeline.incident.notifier import SlackNotifier
from mlops_pipeline.incident.postmortem import BLIND_SPOT
from mlops_pipeline.incident.webhook import router
from mlops_pipeline.model.data import FEATURE_COLS

BODY = {
    "alert_id": "drift-20260101T000000-20260101T010000",
    "window_start": "2026-01-01T00:00:00Z",
    "window_end": "2026-01-01T01:00:00Z",
    "source": "monitor",
}


def build_app(state) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    for name, value in vars(state).items():
        setattr(app.state, name, value)
    return app


def post(app: FastAPI, body: dict) -> httpx.Response:
    async def go() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/incident/webhook", json=body)

    return asyncio.run(go())


def test_happy_path_writes_postmortem_and_notifies_once(make_state):
    state = make_state()
    response = post(build_app(state), BODY)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"alert_id", "incident", "rows", "top", "postmortem_path"}
    assert data["alert_id"] == BODY["alert_id"]
    assert data["incident"] is True
    assert data["rows"] == 5
    assert [t["column"] for t in data["top"]] == ["BILL_AMT1", "BILL_AMT1", "BILL_AMT2"]
    assert set(data["top"][0]) == {"cause", "column", "evidence"}

    path = Path(data["postmortem_path"])
    assert path.name == f"{BODY['alert_id']}.md"
    assert path.parent == Path(state.settings.postmortem_dir).resolve()
    text = path.read_text(encoding="utf-8")
    assert "unit_break" in text and BLIND_SPOT in text

    assert len(state.notifier.payloads) == 1
    assert state.notifier.payloads[0]["alert_id"] == BODY["alert_id"]
    assert state.responder.calls == 1
    # the responder was given feature columns only
    start, end = state.predlog.calls[0]
    assert start.isoformat().startswith("2026-01-01T00:00:00")
    assert end.isoformat().startswith("2026-01-01T01:00:00")


def test_repeated_alert_id_returns_same_response_without_notifying_again(make_state):
    state = make_state()
    app = build_app(state)
    first = post(app, BODY)
    second = post(app, BODY)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert len(state.notifier.payloads) == 1
    assert state.responder.calls == 1


def test_a_different_alert_id_is_processed_separately(make_state):
    state = make_state()
    app = build_app(state)
    post(app, BODY)
    post(app, {**BODY, "alert_id": "drift-other"})
    assert len(state.notifier.payloads) == 2


def test_empty_window_is_not_an_incident_and_writes_nothing(make_state, tmp_path):
    empty = pd.DataFrame(columns=[*FEATURE_COLS, "exposure", "p_default", "flag", "ts"])
    state = make_state(frame=empty)
    response = post(build_app(state), BODY)
    assert response.status_code == 200
    assert response.json() == {
        "alert_id": BODY["alert_id"],
        "incident": False,
        "rows": 0,
        "top": [],
        "postmortem_path": None,
    }
    assert not (tmp_path / "postmortems").exists()
    assert state.notifier.payloads == []
    assert state.responder.calls == 0


def test_window_end_before_start_is_422(make_state):
    state = make_state()
    body = {**BODY, "window_start": BODY["window_end"], "window_end": BODY["window_start"]}
    assert post(build_app(state), body).status_code == 422
    assert state.notifier.payloads == []


def test_equal_window_bounds_are_422(make_state):
    body = {**BODY, "window_end": BODY["window_start"]}
    assert post(build_app(make_state()), body).status_code == 422


@pytest.mark.parametrize(
    "bad_id", ["../evil", "a/b", "a\\b", "..", "x..y", "/abs", ".hidden", "", "trailing."]
)
def test_unsafe_alert_id_is_422_and_writes_nothing(make_state, tmp_path, bad_id):
    state = make_state()
    response = post(build_app(state), {**BODY, "alert_id": bad_id})
    assert response.status_code == 422
    assert state.notifier.payloads == []
    assert not list(tmp_path.rglob("*.md"))


def test_unknown_source_is_422(make_state):
    assert post(build_app(make_state()), {**BODY, "source": "manual"}).status_code == 422


def test_missing_wiring_is_503():
    app = FastAPI()
    app.include_router(router)
    assert post(app, BODY).status_code == 503


def test_slack_notifier_receives_block_kit_payload(make_state):
    state = make_state()
    sent: list[tuple[str, dict]] = []
    state.notifier = SlackNotifier("https://hooks.example.test/x", lambda u, p: sent.append((u, p)))
    assert post(build_app(state), BODY).status_code == 200
    assert len(sent) == 1
    assert [b["type"] for b in sent[0][1]["blocks"]] == ["header", "section", "context"]


def test_notification_failure_does_not_fail_the_request_and_allows_retry(make_state):
    state = make_state()

    class Flaky(FakeNotifier):
        def notify(self, payload):
            if not self.payloads:
                self.payloads.append(payload)
                raise RuntimeError("down")
            super().notify(payload)

    state.notifier = Flaky()
    app = build_app(state)
    assert post(app, BODY).status_code == 200
    assert post(app, BODY).status_code == 200
    assert len(state.notifier.payloads) == 2  # the retry notified again
    assert post(app, BODY).status_code == 200
    assert len(state.notifier.payloads) == 2  # now recorded as handled


def test_default_notifier_prints_a_json_line(make_state, capsys):
    state = make_state()
    del state.notifier
    assert post(build_app(state), BODY).status_code == 200
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and BODY["alert_id"] in out[0]
