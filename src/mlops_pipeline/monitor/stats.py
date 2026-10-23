"""Per-column drift statistics of a scoring window against a fixed reference.

The responder looks at a window of 1,000 rows and has to decide, column by column, whether it
looks like the training data. This module turns that question into eight raw numbers per column
(see :class:`mlops_pipeline.contracts.ColumnStats`); turning them into scores is the ranker's
job. Nothing here is thresholded.

The evaluation calls :func:`column_stats` about 840 times against an 18,000-row reference, so
everything that depends only on the reference (sorted values, PSI bin edges and fractions, the
code set) is computed once in :func:`profile`. Scipy's ``ks_2samp`` re-sorts
the reference on every call and costs about 20 ms per column, which alone would be 0.4 s per
window; the KS statistic below is the same quantity computed from the precomputed pieces.

Empty-window conventions: a column with no non-null value in the window reports 0.0 for every
share and for the KS statistic (there is nothing to compare, and NaN would poison the ranker's
robust scores), and False for ``became_constant``. ``code_set_changed`` is True in that case
for a discrete or categorical column because the set of codes seen (empty) differs from the
reference's. PSI is the one signal that can still be NaN, and only when the *reference* column
is degenerate (fewer than three distinct quantile edges), which is the vendored function's
documented behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlops_pipeline.contracts import ColumnStats, column_kind
from mlops_pipeline.model.data import FEATURE_COLS

#: Same constants as ``model.drift.population_stability_index`` so the two agree exactly.
_PSI_BINS = 10
_PSI_CLIP = 1e-6


@dataclass(frozen=True)
class ColumnProfile:
    """Everything about one reference column that does not depend on the window."""

    column: str
    kind: str
    n_rows: int  # all rows, nulls included: the PSI denominator
    null_rate: float
    values: np.ndarray  # non-null values, sorted
    minimum: float
    maximum: float
    median: float
    codes: np.ndarray  # sorted distinct non-null values for discrete/categorical, else empty
    n_distinct: int
    psi_edges: np.ndarray | None  # quantile bin edges (outer ones infinite); None if degenerate
    psi_ref_frac: np.ndarray | None  # clipped reference mass per PSI bin


@dataclass(frozen=True)
class ReferenceProfile:
    """The reference frame, reduced to what :func:`column_stats` needs."""

    columns: tuple[str, ...]
    n_rows: int
    per_column: Mapping[str, ColumnProfile]


def _as_float(series: pd.Series) -> np.ndarray:
    """Series -> float64 array with NaN for nulls, whatever the pandas dtype."""
    return series.to_numpy(dtype=float, na_value=np.nan)


def _profile_column(column: str, series: pd.Series) -> ColumnProfile:
    kind = column_kind(column)
    raw = _as_float(series)
    n_rows = len(raw)
    values = np.sort(raw[~np.isnan(raw)])
    n = len(values)

    codes = np.unique(values) if kind != "continuous" else np.empty(0)
    n_distinct = len(codes) if kind != "continuous" else len(np.unique(values))

    psi_edges: np.ndarray | None = None
    psi_ref_frac: np.ndarray | None = None
    if kind == "categorical":
        # One bin per reference code, plus a final bin for any other value (mass 0 here).
        counts = np.array([(values == c).sum() for c in codes] + [0], dtype=float)
        psi_ref_frac = np.clip(counts / max(n_rows, 1), _PSI_CLIP, None)
    elif n > 0:
        # Edges from the non-null reference values. Identical to the vendored function's own
        # edges whenever the reference has no nulls, which is the case for this dataset.
        edges = np.unique(np.quantile(values, np.linspace(0, 1, _PSI_BINS + 1)))
        if len(edges) >= 3:
            edges[0], edges[-1] = -np.inf, np.inf
            ref_counts, _ = np.histogram(values, bins=edges)
            psi_edges = edges
            psi_ref_frac = np.clip(ref_counts / max(n_rows, 1), _PSI_CLIP, None)

    return ColumnProfile(
        column=column,
        kind=kind,
        n_rows=n_rows,
        null_rate=(n_rows - n) / n_rows if n_rows else 0.0,
        values=values,
        minimum=float(values[0]) if n else float("nan"),
        maximum=float(values[-1]) if n else float("nan"),
        median=float(np.median(values)) if n else float("nan"),
        codes=codes,
        n_distinct=n_distinct,
        psi_edges=psi_edges,
        psi_ref_frac=psi_ref_frac,
    )


def profile(reference: pd.DataFrame, columns: Sequence[str] = FEATURE_COLS) -> ReferenceProfile:
    """Precompute the reference side of every statistic. ``columns`` keep the given order."""
    cols = tuple(columns)
    return ReferenceProfile(
        columns=cols,
        n_rows=len(reference),
        per_column={c: _profile_column(c, reference[c]) for c in cols},
    )


def _psi(p: ColumnProfile, sorted_values: np.ndarray, n: int) -> float:
    """PSI of the window against the reference; see the module docstring for the NaN case.

    ``sorted_values`` are the window's non-null values in ascending order and ``n`` is the
    window's row count, nulls included.
    """
    if p.kind == "categorical":
        # One bin per reference code plus a final bin for any other value. Fractions are over
        # all rows of the window, so a null lands in no bin and simply shows up as missing mass.
        in_code = _code_counts(p, sorted_values)
        cur_counts = np.append(in_code, len(sorted_values) - in_code.sum())
    else:
        if p.psi_edges is None or p.psi_ref_frac is None:
            return float("nan")
        # Same bins as np.histogram(current, bins=edges) in the vendored function: bin k is
        # [edge_k, edge_k+1) and the last one is closed at +inf. NaN is the case to check: that
        # function passes raw values, and np.histogram with explicit edges sorts the data and
        # locates the edges with searchsorted, where NaN sorts after +inf and so falls outside
        # every bin. A null is therefore counted in no bin but still in the denominator
        # (n includes nulls), i.e. the null share is simply missing mass. Here the nulls are
        # already stripped, which gives the same counts; the tests assert exact equality with
        # the vendored function, with and without NaN.
        inner = np.searchsorted(sorted_values, p.psi_edges[1:-1], side="left")
        cur_counts = np.diff(np.concatenate(([0], inner, [len(sorted_values)])))
    cur_frac = np.clip(cur_counts / max(n, 1), _PSI_CLIP, None)
    ref_frac = p.psi_ref_frac
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def _code_counts(p: ColumnProfile, sorted_values: np.ndarray) -> np.ndarray:
    """Window rows per reference code (aligned with ``p.codes``)."""
    return (
        np.searchsorted(sorted_values, p.codes, side="right")
        - np.searchsorted(sorted_values, p.codes, side="left")
    ).astype(float)


def _ks(p: ColumnProfile, cur: np.ndarray) -> float:
    """Two-sample KS statistic, equal to ``scipy.stats.ks_2samp(ref, cur).statistic``.

    The statistic is the largest gap between two step-function CDFs. Between two consecutive
    window values the window CDF is flat and the reference CDF only rises, so the gap is largest
    at one end of the interval: at a window value (CDFs including it) or just below one (CDFs
    excluding it). Checking those 2m positions needs only binary searches on the precomputed
    sorted reference, where scipy re-sorts all 19,000 values on every call. Each candidate is
    the same ``count/n1 - count/n2`` pair scipy evaluates, so the result agrees to the last bit
    up to float noise.
    """
    m = len(cur)
    n1 = len(p.values)
    if p.kind == "categorical" or m == 0 or n1 == 0:
        return 0.0
    ref_right = np.searchsorted(p.values, cur, side="right") / n1
    cur_right = np.searchsorted(cur, cur, side="right") / m
    ref_left = np.searchsorted(p.values, cur, side="left") / n1
    cur_left = np.searchsorted(cur, cur, side="left") / m
    return float(max(np.abs(ref_right - cur_right).max(), np.abs(ref_left - cur_left).max()))


def _column_stats(p: ColumnProfile, x: np.ndarray) -> ColumnStats:
    n = len(x)
    cur = np.sort(x[~np.isnan(x)])  # non-null window values, ascending
    m = len(cur)
    discrete = p.kind != "continuous"
    has_ref = len(p.values) > 0

    null_delta = (n - m) / n - p.null_rate if n else 0.0

    out_of_range = unseen = log_ratio = 0.0
    code_set_changed = discrete and len(p.codes) > 0
    became_constant = False
    if m > 0:
        if has_ref:
            below = np.searchsorted(cur, p.minimum, side="left")
            above = m - np.searchsorted(cur, p.maximum, side="right")
            out_of_range = float((below + above) / m)
            median = cur[m // 2] if m % 2 else (cur[m // 2 - 1] + cur[m // 2]) / 2
            log_ratio = float(abs(np.log10((abs(median) + 1.0) / (abs(p.median) + 1.0))))
        if discrete:
            in_code = _code_counts(p, cur)
            unseen = float((m - in_code.sum()) / m)
            code_set_changed = not (bool((in_code > 0).all()) and unseen == 0.0)
        became_constant = bool(cur[0] == cur[-1] and p.n_distinct > 1)

    return ColumnStats(
        column=p.column,
        psi=_psi(p, cur, n),
        ks=_ks(p, cur),
        null_delta=float(null_delta),
        out_of_range=out_of_range,
        unseen_share=unseen,
        median_log_ratio=log_ratio,
        code_set_changed=bool(code_set_changed),
        became_constant=became_constant,
    )


def column_stats(ref: ReferenceProfile, current: pd.DataFrame) -> dict[str, ColumnStats]:
    """All eight signals for every profiled column of ``current``, in the profile's column order.

    The profile's order is ``FEATURE_COLS`` unless ``profile`` was given other columns. Extra
    columns in ``current`` are ignored; a missing profiled column raises ``KeyError``.
    """
    # One conversion for the whole frame is much cheaper than one per column.
    matrix = np.asfortranarray(current[list(ref.columns)].to_numpy(dtype=float, na_value=np.nan))
    return {c: _column_stats(ref.per_column[c], matrix[:, j]) for j, c in enumerate(ref.columns)}
