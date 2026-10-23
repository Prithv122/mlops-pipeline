"""Analyser: one window in, column statistics and a cause ranking out.

The monitor, the webhook and the evaluation all call this one function, so the ranking that
is scored on injected faults is the ranking that is served.
"""

from __future__ import annotations

import pandas as pd

from mlops_pipeline.contracts import ColumnStats, Ranking
from mlops_pipeline.incident.ranker import NullModel, rank
from mlops_pipeline.monitor.stats import ReferenceProfile, column_stats


def analyse(
    ref: ReferenceProfile, null: NullModel, window: pd.DataFrame
) -> tuple[dict[str, ColumnStats], Ranking]:
    stats = column_stats(ref, window)
    return stats, rank(null, stats)
