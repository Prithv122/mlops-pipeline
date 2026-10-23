"""One pass of the monitor: read a window, analyse it, alert if it looks like an incident."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from mlops_pipeline.contracts import Ranking
from mlops_pipeline.incident.collector import collect
from mlops_pipeline.incident.webhook import IncidentAlert

TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class MonitorResult:
    """What one monitor pass found.

    ``status`` is ``analysed`` (``ranking`` is set), ``empty`` (no rows) or
    ``insufficient_rows`` (fewer than ``min_rows``, the window length the scores were
    calibrated for). Only ``analysed`` carries a verdict.
    """

    status: Literal["analysed", "empty", "insufficient_rows"]
    rows: int
    min_rows: int
    ranking: Ranking | None = None


def window_alert_id(start: datetime, end: datetime) -> str:
    """Alert id from the window in UTC with microseconds, so two windows never share one.

    A naive time is read as UTC, as the webhook does. Only letters and digits, so it passes the
    webhook's file-name check.
    """

    def stamp(value: datetime) -> str:
        utc = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return f"{utc:%Y%m%dT%H%M%S%f}Z"

    return f"drift-{stamp(start)}-{stamp(end)}"


def monitor_window(
    log: Any,
    ref: Any,
    null: Any,
    *,
    start: Any,
    end: Any,
    alert: Callable[[IncidentAlert], None],
    report_dir: Path | None = None,
    reference_frame: pd.DataFrame | None = None,
) -> MonitorResult:
    """Analyse ``log`` rows with ``start <= ts < end`` against the reference.

    An empty window must not look like a quiet one, and neither must a short one: the scores
    are calibrated for windows of ``null.window_size`` rows, and a much smaller window reads as
    drift almost every time. In both cases nothing is analysed and ``alert`` is not called.
    ``alert`` is called once, with an id derived from the window, only when the ranking says
    incident, so re-running the same window repeats the same id and the webhook can ignore the
    duplicate.

    ``report_dir`` also writes the Evidently drift report for a human to read. That report needs
    the reference as a frame, so pass ``reference_frame`` when ``ref`` is a profile that does not
    carry one. The report feeds nothing else; the ranking is what decides.
    """
    # Imported here so a test can replace the analyser, and so a run without a report never
    # loads Evidently.
    from mlops_pipeline.incident.analyser import analyse

    min_rows = int(null.window_size)
    window = collect(log, start, end)
    rows = len(window)
    if rows == 0:
        return MonitorResult("empty", 0, min_rows)
    if rows < min_rows:
        return MonitorResult("insufficient_rows", rows, min_rows)

    _, ranking = analyse(ref, null, window)

    if ranking.incident:
        alert(
            IncidentAlert(alert_id=window_alert_id(start, end), window_start=start, window_end=end)
        )

    if report_dir is not None:
        from mlops_pipeline.monitor.evidently_report import write_drift_report

        frame = reference_frame if reference_frame is not None else _reference_frame(ref)
        write_drift_report(frame, window, report_dir)

    return MonitorResult("analysed", rows, min_rows, ranking)


def monitor_once(
    log: Any,
    ref: Any,
    null: Any,
    *,
    start: Any,
    end: Any,
    alert: Callable[[IncidentAlert], None],
    report_dir: Path | None = None,
    reference_frame: pd.DataFrame | None = None,
) -> Ranking | None:
    """:func:`monitor_window`, returning only the ranking.

    ``None`` means nothing was judged: the window was empty or shorter than the null model's
    ``window_size``. Use :func:`monitor_window` to tell those apart.
    """
    return monitor_window(
        log,
        ref,
        null,
        start=start,
        end=end,
        alert=alert,
        report_dir=report_dir,
        reference_frame=reference_frame,
    ).ranking


def _reference_frame(ref: Any) -> pd.DataFrame:
    if isinstance(ref, pd.DataFrame):
        return ref
    to_frame = getattr(ref, "to_frame", None)
    if callable(to_frame):
        return to_frame()
    raise ValueError("report_dir needs a reference frame: pass reference_frame=")


def post_alert(url: str) -> Callable[[IncidentAlert], None]:
    """Alert callable that POSTs the alert as JSON to the webhook at ``url``.

    A non-2xx reply or a network failure raises, so the caller can retry. Retrying is safe: the
    webhook ignores a repeated alert id.
    """

    def send(alert: IncidentAlert) -> None:
        request = urllib.request.Request(
            url,
            data=json.dumps(alert.model_dump(mode="json")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS):
                pass
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"incident webhook returned HTTP {exc.code}") from None

    return send
