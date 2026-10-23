"""Where an incident summary goes: a JSON line on a stream, or a Slack webhook."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TextIO

from mlops_pipeline.contracts import Ranking

if TYPE_CHECKING:
    from mlops_pipeline.incident.webhook import IncidentAlert

TIMEOUT_SECONDS = 10
TOP_CAUSES = 3


class Notifier(Protocol):
    def notify(self, payload: dict[str, Any]) -> None: ...


def alert_payload(
    alert: IncidentAlert, ranking: Ranking, postmortem_path: str | None
) -> dict[str, Any]:
    """Plain structured payload used by notifiers that have no format of their own."""
    return {
        "alert_id": alert.alert_id,
        "incident": ranking.incident,
        "window_start": alert.window_start.isoformat(),
        "window_end": alert.window_end.isoformat(),
        "top": [
            {"cause": h.cause, "column": h.column, "evidence": h.evidence}
            for h in ranking.top_causes(TOP_CAUSES)
        ],
        "postmortem_path": postmortem_path,
    }


class JsonNotifier:
    """One JSON object per line. The stream is resolved at call time so output capture works."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream

    def notify(self, payload: dict[str, Any]) -> None:
        stream = self._stream if self._stream is not None else sys.stdout
        stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        stream.flush()


def _post_json(url: str, payload: dict[str, Any]) -> None:
    """POST with the standard library. Never exercised in tests.

    Failures are re-raised without the original message because a URL error can echo the
    webhook address, which is a secret.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS):
            pass
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Slack webhook post failed ({type(exc).__name__})") from None


class SlackNotifier:
    """Slack Block Kit message. The transport is injectable; the default posts over HTTPS."""

    def __init__(
        self,
        webhook_url: str,
        transport: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._webhook_url = webhook_url
        self._transport = transport if transport is not None else _post_json

    def __repr__(self) -> str:
        return "SlackNotifier(webhook_url=<redacted>)"

    def build_payload(
        self, alert: IncidentAlert, ranking: Ranking, postmortem_path: str | None
    ) -> dict[str, Any]:
        verdict = "incident confirmed" if ranking.incident else "no incident confirmed"
        causes = [
            f"{i}. *{h.cause}* on `{h.column}` (evidence {h.evidence:.2f})"
            for i, h in enumerate(ranking.top_causes(TOP_CAUSES), start=1)
        ]
        body = "\n".join(causes) if causes else "No hypotheses."
        context = f"Window {alert.window_start.isoformat()} to {alert.window_end.isoformat()}"
        if postmortem_path:
            context += f" | Postmortem: {postmortem_path}"
        return {
            "text": f"Drift alert {alert.alert_id}: {verdict}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Drift alert {alert.alert_id}: {verdict}",
                    },
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Most likely causes*\n{body}"},
                },
                {"type": "context", "elements": [{"type": "mrkdwn", "text": context}]},
            ],
        }

    def notify(self, payload: dict[str, Any]) -> None:
        self._transport(self._webhook_url, payload)
