from __future__ import annotations

import numpy as np
import pandas as pd
from ops_fakes import T0, T1, FakeLog, make_log_frame

from mlops_pipeline.incident.collector import collect
from mlops_pipeline.model.data import FEATURE_COLS


def test_collect_returns_only_feature_columns_in_order():
    frame = make_log_frame()
    out = collect(FakeLog(frame), T0, T1)
    assert list(out.columns) == list(FEATURE_COLS)
    assert len(out) == len(frame)


def test_collect_is_float_with_nan_for_null():
    frame = make_log_frame()
    frame["AGE"] = frame["AGE"].astype(object)
    frame.loc[1, "AGE"] = None
    out = collect(FakeLog(frame), T0, T1)
    assert all(dtype == np.float64 for dtype in out.dtypes)
    assert np.isnan(out.loc[0, "PAY_1"])
    assert np.isnan(out.loc[1, "AGE"])


def test_collect_passes_the_window_to_the_log():
    log = FakeLog(make_log_frame())
    collect(log, T0, T1)
    assert log.calls == [(T0, T1)]


def test_collect_empty_window_keeps_columns():
    empty = pd.DataFrame(columns=[*FEATURE_COLS, "exposure", "p_default", "flag", "ts"])
    out = collect(FakeLog(empty), T0, T1)
    assert len(out) == 0
    assert list(out.columns) == list(FEATURE_COLS)
