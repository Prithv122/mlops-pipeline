"""``POST /incident/webhook``: the monitor reports a window, the responder explains it.

The route reads everything it needs from ``request.app.state`` so the serving app decides what
is wired in and tests can drop in fakes:

``settings``           object with ``postmortem_dir``
``responder``          :class:`Responder`; optional ``min_rows`` attribute: windows with fewer
                       rows are not judged (the scores are calibrated for a full-size window)
``predlog``            object with ``.window(start, end)`` (the prediction log)
``notifier``           optional :class:`Notifier`; defaults to a JSON line on stdout
``postmortem_writer``  optional :class:`PostmortemWriter`; defaults to the template writer
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator, model_validator

from mlops_pipeline.contracts import ColumnStats, Ranking
from mlops_pipeline.incident.collector import collect
from mlops_pipeline.incident.display import column_suspects
from mlops_pipeline.incident.notifier import JsonNotifier, alert_payload
from mlops_pipeline.incident.postmortem import TemplateWriter

log = logging.getLogger(__name__)

TOP_CAUSES = 3

# The alert id becomes a file name, so it is restricted to a safe alphabet. The monitor's own
# ids ("drift-20261023T000000000000Z-20261023T010000000000Z") fit; anything with a separator
# or "..", does not.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class IncidentAlert(BaseModel):
    alert_id: str
    window_start: datetime
    window_end: datetime
    source: Literal["monitor"] = "monitor"

    @field_validator("alert_id")
    @classmethod
    def _safe_file_name(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value) or ".." in value or value.endswith("."):
            raise ValueError("alert_id must be 1-128 characters of letters, digits, '.', '_', '-'")
        return value

    @field_validator("window_start", "window_end")
    @classmethod
    def _assume_utc(cls, value: datetime) -> datetime:
        # The prediction log stores UTC. A naive time is read as UTC so naive and aware inputs
        # can be compared and ordered.
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value

    @model_validator(mode="after")
    def _window_is_forward(self) -> IncidentAlert:
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        return self


class Responder(Protocol):
    def respond(self, window: pd.DataFrame) -> tuple[dict[str, ColumnStats], Ranking]: ...


class TopCause(BaseModel):
    cause: str
    column: str
    evidence: float
    column_score: float


class IncidentResponse(BaseModel):
    """``status`` says why there may be no verdict: only ``analysed`` carries a real one.

    ``empty`` and ``insufficient_rows`` are not remembered for the alert id, because rows
    for the window may still arrive and a retry has to look again.
    """

    alert_id: str
    status: Literal["analysed", "empty", "insufficient_rows"]
    incident: bool
    rows: int
    top: list[TopCause]
    postmortem_path: str | None


router = APIRouter()

# Alerts already analysed by this process, keyed by alert_id. The monitor may retry a delivery;
# a repeat returns the stored response and does not notify again. This memory is per process:
# a restart, or a second worker, forgets it, so a duplicate can be notified once more. Empty
# and too-small windows are never stored: late rows may still arrive.
_LOCK = threading.Lock()


def _handled(state: Any) -> dict[str, IncidentResponse]:
    store = getattr(state, "handled_alerts", None)
    if store is None:
        store = {}
        state.handled_alerts = store
    return store


def _postmortem_file(directory: Path, alert_id: str) -> Path:
    target = (directory / f"{alert_id}.md").resolve()
    if target.parent != directory.resolve():
        raise HTTPException(status_code=422, detail="alert_id is not a valid file name")
    return target


@router.post("/incident/webhook", response_model=IncidentResponse)
def incident_webhook(alert: IncidentAlert, request: Request) -> IncidentResponse:
    state = request.app.state
    responder = getattr(state, "responder", None)
    predlog = getattr(state, "predlog", None)
    settings = getattr(state, "settings", None)
    if responder is None or predlog is None or settings is None:
        raise HTTPException(status_code=503, detail="incident responder is not configured")
    notifier = getattr(state, "notifier", None) or JsonNotifier()
    writer = getattr(state, "postmortem_writer", None) or TemplateWriter()

    with _LOCK:
        handled = _handled(state)
        if alert.alert_id in handled:
            return handled[alert.alert_id]

        window = collect(predlog, alert.window_start, alert.window_end)
        min_rows = getattr(responder, "min_rows", None)
        if len(window) == 0 or (min_rows is not None and len(window) < min_rows):
            return IncidentResponse(
                alert_id=alert.alert_id,
                status="empty" if len(window) == 0 else "insufficient_rows",
                incident=False,
                rows=len(window),
                top=[],
                postmortem_path=None,
            )

        stats, ranking = responder.respond(window)
        path = _postmortem_file(Path(settings.postmortem_dir), alert.alert_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(writer.write(alert, stats, ranking, len(window)), encoding="utf-8")

        response = IncidentResponse(
            alert_id=alert.alert_id,
            status="analysed",
            incident=ranking.incident,
            rows=len(window),
            top=[
                TopCause(
                    cause=s.hypothesis.cause,
                    column=s.hypothesis.column,
                    evidence=s.hypothesis.evidence,
                    column_score=s.column_score,
                )
                for s in column_suspects(ranking, TOP_CAUSES)
            ],
            postmortem_path=str(path),
        )

        build = getattr(notifier, "build_payload", None)
        payload = (
            build(alert, ranking, str(path)) if build else alert_payload(alert, ranking, str(path))
        )
        try:
            notifier.notify(payload)
        except Exception as exc:
            # The postmortem is on disk. A failed notification is logged by type only (a
            # message could echo a webhook URL) and the alert stays unrecorded so a retry
            # tries the notification again.
            log.warning("incident notification failed: %s", type(exc).__name__)
            return response
        handled[alert.alert_id] = response
        return response
