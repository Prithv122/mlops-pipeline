"""One pass of the monitor: read a window, analyse it, alert if it looks like an incident."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_pipeline.contracts import Ranking
from mlops_pipeline.incident.collector import collect
from mlops_pipeline.incident.webhook import IncidentAlert

TIMEOUT_SECONDS = 10


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
    """Analyse ``log`` rows with ``start <= ts < end`` against the reference.

    Returns the ranking, or ``None`` when the window holds no rows (nothing to compare, and an
    empty window must not look like a quiet one). ``alert`` is called once, with an id derived from
    the window, only when the ranking says incident, so re-running the same window repeats the
    same id and the webhook can ignore the duplicate.

    ``report_dir`` also writes the Evidently drift report for a human to read. That report needs
    the reference as a frame, so pass ``reference_frame`` when ``ref`` is a profile that does not
    carry one. The report feeds nothing else; the ranking above is what decides.
    """
    # Imported here so a test can replace the analyser, and so a run without a report never
    # loads Evidently.
    from mlops_pipeline.incident.analyser import analyse

    window = collect(log, start, end)
    if len(window) == 0:
        return None

    _, ranking = analyse(ref, null, window)

    if ranking.incident:
        alert(
            IncidentAlert(
                alert_id=f"drift-{start:%Y%m%dT%H%M%S}-{end:%Y%m%dT%H%M%S}",
                window_start=start,
                window_end=end,
            )
        )

    if report_dir is not None:
        from mlops_pipeline.monitor.evidently_report import write_drift_report

        frame = reference_frame if reference_frame is not None else _reference_frame(ref)
        write_drift_report(frame, window, report_dir)

    return ranking


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
