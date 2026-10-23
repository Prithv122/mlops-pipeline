"""SQLite log of every served prediction; the monitor reads windows back out of it.

A connection is opened per call instead of shared: the API handlers run on a thread pool,
and a SQLite connection may not cross threads by default. Opening one costs microseconds
next to a model call, and WAL mode lets the monitor read while the API writes.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mlops_pipeline.model.data import FEATURE_COLS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    model_version TEXT NOT NULL,
    features TEXT NOT NULL,
    exposure REAL NOT NULL,
    p_default REAL NOT NULL,
    flag INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_ts ON predictions (ts);
"""

_COLUMNS = [*FEATURE_COLS, "exposure", "p_default", "flag", "ts", "model_version"]


def _iso(ts: datetime) -> str:
    """Fixed-width UTC text, so string order is time order and range queries are exact."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).isoformat(timespec="microseconds")


class PredictionLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def append(
        self,
        *,
        ts: datetime,
        model_version: str,
        features: dict[str, float | None],
        exposure: float,
        p_default: float,
        flag: bool,
    ) -> None:
        row = (
            _iso(ts),
            str(model_version),
            json.dumps({c: features.get(c) for c in FEATURE_COLS}),
            float(exposure),
            float(p_default),
            int(bool(flag)),
        )
        with closing(self._connect()) as con, con:
            con.execute(
                "INSERT INTO predictions (ts, model_version, features, exposure, p_default, flag)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                row,
            )

    def count(self) -> int:
        with closing(self._connect()) as con:
            return int(con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def window(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Rows with ``start <= ts < end``, oldest first, features as floats (NaN for null)."""
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT ts, model_version, features, exposure, p_default, flag FROM predictions"
                " WHERE ts >= ? AND ts < ? ORDER BY ts, id",
                (_iso(start), _iso(end)),
            ).fetchall()
        features = pd.DataFrame(
            [json.loads(r[2]) for r in rows], columns=FEATURE_COLS, dtype="object"
        ).astype("float64")
        out = features.assign(
            exposure=pd.Series([r[3] for r in rows], dtype="float64"),
            p_default=pd.Series([r[4] for r in rows], dtype="float64"),
            flag=pd.Series([bool(r[5]) for r in rows], dtype="bool"),
            ts=pd.to_datetime([r[0] for r in rows], utc=True, format="ISO8601"),
            model_version=pd.Series([r[1] for r in rows], dtype="object"),
        )
        return out[_COLUMNS]
