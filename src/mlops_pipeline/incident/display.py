"""What the postmortem, the notifier and the webhook show for each suspect column.

Columns are ordered by their strongest hypothesis, but each row names the *diagnosed* cause
(Amendment 1), whose own evidence can be lower than that strongest one: a capped shift score
can outrank a specific cause that crossed tau. Shown alone, the evidence column then looks
out of order. Each row therefore also carries the column's score, the maximum evidence over
all of that column's hypotheses, which is the number the columns are actually sorted by.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mlops_pipeline.contracts import Hypothesis, Ranking


@dataclass(frozen=True)
class ColumnSuspect:
    """One suspect column: its diagnosed cause and the score that placed it in the list."""

    hypothesis: Hypothesis
    column_score: float


def column_suspects(ranking: Ranking, k: int) -> list[ColumnSuspect]:
    """The first ``k`` columns of the ranking, each with its diagnosed cause and column score."""
    best: dict[str, float] = {}
    for h in ranking.hypotheses:
        if not math.isnan(h.evidence):
            best[h.column] = max(best.get(h.column, h.evidence), h.evidence)
    return [ColumnSuspect(h, best.get(h.column, h.evidence)) for h in ranking.top_causes(k)]
