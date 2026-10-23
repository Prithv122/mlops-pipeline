"""Postmortem text for an incident alert.

Every number in a postmortem comes from the measured ranking and column statistics. The
template writer formats them; the model-backed writer may add prose in front, but the measured
section is appended verbatim afterwards so a figure can never be paraphrased by the model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from mlops_pipeline.contracts import ColumnStats, Ranking

if TYPE_CHECKING:
    from mlops_pipeline.incident.webhook import IncidentAlert

log = logging.getLogger(__name__)

TOP_HYPOTHESES = 5
TOP_COLUMNS = 3

BLIND_SPOT = (
    "What this does not cover: the responder reads feature values only. A fault that shows up "
    "only in the labels (a change in the default rate, concept drift) is invisible to it, "
    "because labels arrive months after scoring. A quiet ranking is not proof the model is fine."
)

_SIGNALS = (
    "psi",
    "ks",
    "null_delta",
    "out_of_range",
    "unseen_share",
    "median_log_ratio",
    "code_set_changed",
    "became_constant",
)


class PostmortemWriter(Protocol):
    def write(
        self,
        alert: IncidentAlert,
        stats: dict[str, ColumnStats],
        ranking: Ranking,
        rows: int,
    ) -> str:
        """Return the postmortem as markdown."""
        ...


def _num(value: float) -> str:
    return f"{value:.4f}"


def _header(alert: IncidentAlert, rows: int) -> str:
    return "\n".join(
        [
            f"# Incident postmortem: {alert.alert_id}",
            "",
            f"- Alert id: `{alert.alert_id}`",
            f"- Window: {alert.window_start.isoformat()} to {alert.window_end.isoformat()}",
            f"- Rows analysed: {rows}",
        ]
    )


def measured_section(stats: dict[str, ColumnStats], ranking: Ranking) -> str:
    """Verdict, ranked hypotheses and raw signals, formatted deterministically."""
    verdict = "incident" if ranking.incident else "no incident"
    lines = [
        "## Verdict",
        "",
        f"- Verdict: {verdict}",
        f"- Threshold: {_num(ranking.threshold)}",
        f"- Method: {ranking.method}",
        "",
        f"## Top {TOP_HYPOTHESES} hypotheses",
        "",
        "| # | Cause | Column | Evidence |",
        "|---|---|---|---|",
    ]
    for i, h in enumerate(ranking.hypotheses[:TOP_HYPOTHESES], start=1):
        lines.append(f"| {i} | {h.cause} | {h.column} | {_num(h.evidence)} |")
    if not ranking.hypotheses:
        lines.append("| - | none | none | - |")

    lines += ["", f"## Raw signals for the top {TOP_COLUMNS} columns", ""]
    shown = 0
    for column in ranking.column_order()[:TOP_COLUMNS]:
        col = stats.get(column)
        if col is None:
            continue
        shown += 1
        lines.append(f"### {column}")
        lines.append("")
        lines.append("| Signal | Value |")
        lines.append("|---|---|")
        for name in _SIGNALS:
            value = getattr(col, name)
            text = str(value) if isinstance(value, bool) else _num(value)
            lines.append(f"| {name} | {text} |")
        lines.append("")
    if shown == 0:
        lines += ["No column statistics available.", ""]

    lines += ["## Limits", "", BLIND_SPOT]
    return "\n".join(lines) + "\n"


class TemplateWriter:
    """Deterministic markdown from the measured ranking. Same input, same bytes."""

    def write(
        self,
        alert: IncidentAlert,
        stats: dict[str, ColumnStats],
        ranking: Ranking,
        rows: int,
    ) -> str:
        return _header(alert, rows) + "\n\n" + measured_section(stats, ranking)


def build_prompt(
    alert: IncidentAlert, stats: dict[str, ColumnStats], ranking: Ranking, rows: int
) -> str:
    """Prompt text for the model. Carries the measured table only, never raw data rows."""
    return (
        "You are writing two or three sentences for an on-call engineer about a drift alert on a "
        "credit-default scoring service.\n"
        "Use only the measured results below. Do not name any cause that is not in the ranking, "
        "do not guess at upstream systems, and do not repeat or change any number; the table "
        "is attached to your text afterwards.\n"
        "Say what the top-ranked hypothesis is, how it compares with the next one, and that "
        "label-dependent faults cannot be seen by this tool.\n\n"
        f"Alert: {alert.alert_id}\n"
        f"Window: {alert.window_start.isoformat()} to {alert.window_end.isoformat()}\n"
        f"Rows analysed: {rows}\n\n" + measured_section(stats, ranking)
    )


class LLMWriter:
    """Model-written summary in front of the measured section.

    ``complete`` is any callable from prompt to text, so tests and CI never call a model. On an
    exception or an empty reply the fallback writer's text is returned unchanged.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        fallback: PostmortemWriter | None = None,
    ) -> None:
        self._complete = complete
        self._fallback = fallback if fallback is not None else TemplateWriter()

    def write(
        self,
        alert: IncidentAlert,
        stats: dict[str, ColumnStats],
        ranking: Ranking,
        rows: int,
    ) -> str:
        try:
            prose = self._complete(build_prompt(alert, stats, ranking, rows))
        except Exception:
            log.warning("postmortem model call failed; using the template writer", exc_info=True)
            return self._fallback.write(alert, stats, ranking, rows)
        if not isinstance(prose, str) or not prose.strip():
            return self._fallback.write(alert, stats, ranking, rows)
        return (
            _header(alert, rows)
            + "\n\n## Summary\n\n"
            + prose.strip()
            + "\n\n"
            + measured_section(stats, ranking)
        )
