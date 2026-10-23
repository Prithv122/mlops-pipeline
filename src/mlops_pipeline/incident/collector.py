"""Turn a slice of the prediction log into the frame the analyser reads."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

from mlops_pipeline.model.data import FEATURE_COLS


class _WindowSource(Protocol):
    def window(self, start: datetime, end: datetime) -> pd.DataFrame: ...


def collect(log: _WindowSource, start: datetime, end: datetime) -> pd.DataFrame:
    """Feature columns of the log rows with ``start <= ts < end``.

    Only ``FEATURE_COLS`` are returned, in that order, as float with NaN for null. The log also
    holds the prediction, the cost flag and the model version; those are outputs of the model, not
    inputs, and the responder must compare inputs only. An empty window keeps the columns.
    """
    frame = log.window(start, end)
    return frame.reindex(columns=list(FEATURE_COLS)).astype(float).reset_index(drop=True)
