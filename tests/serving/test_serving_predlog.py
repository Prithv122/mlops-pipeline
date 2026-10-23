from __future__ import annotations

import math
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_pipeline.model.data import FEATURE_COLS
from mlops_pipeline.predlog import PredictionLog

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
LOG_COLUMNS = [*FEATURE_COLS, "exposure", "p_default", "flag", "ts", "model_version"]


def _features(value: float = 1.0) -> dict[str, float | None]:
    return {c: value for c in FEATURE_COLS}


def _add(log: PredictionLog, ts: datetime, **kw) -> None:
    log.append(
        ts=ts,
        model_version=kw.get("version", "3"),
        features=kw.get("features", _features()),
        exposure=kw.get("exposure", 100.0),
        p_default=kw.get("p", 0.25),
        flag=kw.get("flag", False),
    )


def test_window_start_is_inclusive_and_end_is_exclusive(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "p.sqlite")
    for i in range(4):
        _add(log, T0 + timedelta(minutes=i), p=i / 10)
    got = log.window(T0 + timedelta(minutes=1), T0 + timedelta(minutes=3))
    assert got["p_default"].tolist() == [0.1, 0.2]
    assert log.count() == 4


def test_microsecond_boundary(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "p.sqlite")
    _add(log, T0 + timedelta(microseconds=1))
    assert len(log.window(T0, T0 + timedelta(microseconds=1))) == 0
    assert len(log.window(T0 + timedelta(microseconds=1), T0 + timedelta(seconds=1))) == 1


def test_columns_and_round_trip(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "p.sqlite")
    feats = {c: float(i) for i, c in enumerate(FEATURE_COLS)}
    _add(log, T0, features=feats, exposure=12.5, p=0.4, flag=True, version="9")
    got = log.window(T0, T0 + timedelta(seconds=1))
    assert list(got.columns) == LOG_COLUMNS
    row = got.iloc[0]
    assert [row[c] for c in FEATURE_COLS] == [float(i) for i in range(len(FEATURE_COLS))]
    assert (row["exposure"], row["p_default"], row["model_version"]) == (12.5, 0.4, "9")
    assert bool(row["flag"]) is True
    assert pd.Timestamp(row["ts"]) == pd.Timestamp(T0)
    assert all(got[c].dtype == np.float64 for c in FEATURE_COLS)


def test_nulls_round_trip_as_nan(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "p.sqlite")
    _add(log, T0, features=_features() | {"PAY_3": None, "SEX": None})
    _add(log, T0 + timedelta(seconds=1), features=_features() | {"PAY_3": None})
    got = log.window(T0, T0 + timedelta(minutes=1))
    assert math.isnan(got.loc[0, "PAY_3"]) and math.isnan(got.loc[0, "SEX"])
    assert math.isnan(got.loc[1, "PAY_3"]) and got.loc[1, "SEX"] == 1.0
    assert got["PAY_3"].dtype == np.float64


def test_all_null_column_is_still_float(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "p.sqlite")
    for i in range(3):
        _add(log, T0 + timedelta(seconds=i), features=_features() | {"AGE": None})
    got = log.window(T0, T0 + timedelta(minutes=1))
    assert got["AGE"].dtype == np.float64 and got["AGE"].isna().all()


def test_empty_window_keeps_columns(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "p.sqlite")
    empty = log.window(T0, T0 + timedelta(hours=1))
    assert len(empty) == 0
    assert list(empty.columns) == LOG_COLUMNS
    assert all(empty[c].dtype == np.float64 for c in FEATURE_COLS)
    _add(log, T0 + timedelta(days=1))
    outside = log.window(T0, T0 + timedelta(hours=1))
    assert len(outside) == 0 and list(outside.columns) == LOG_COLUMNS


def test_rows_come_back_oldest_first(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "p.sqlite")
    for minutes in (5, 1, 3):
        _add(log, T0 + timedelta(minutes=minutes), p=minutes / 100)
    got = log.window(T0, T0 + timedelta(hours=1))
    assert got["p_default"].tolist() == [0.01, 0.03, 0.05]


def test_timezones_are_normalised(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "p.sqlite")
    ist = timezone(timedelta(hours=5, minutes=30))
    _add(log, T0.astimezone(ist))  # same instant as T0
    _add(log, datetime(2026, 1, 1, 12, 0, 30))  # naive means UTC
    assert len(log.window(T0, T0 + timedelta(minutes=1))) == 2


def test_persists_across_instances_and_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "p.sqlite"
    _add(PredictionLog(path), T0)
    assert PredictionLog(path).count() == 1


def test_concurrent_appends_from_threads(tmp_path: Path) -> None:
    log = PredictionLog(tmp_path / "p.sqlite")
    errors: list[Exception] = []

    def work(k: int) -> None:
        try:
            for i in range(10):
                _add(log, T0 + timedelta(seconds=k * 10 + i))
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert log.count() == 80
