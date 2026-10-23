"""Cause ranker and the PSI baseline it is compared with.

Implements the scoring in PREREGISTRATION.md literally: each signal is turned into a robust
one-sided z against what no-fault windows look like, hypotheses (a fault class on a column)
take the maximum of the z-scores of their signals, and the incident threshold tau is fixed
from a second, separate set of no-fault windows.

The maths (``fit_null_model_from_stats``, ``robust_z``, ``rank``, ``psi_baseline``) works on
``ColumnStats`` only, so it is testable without any window machinery. The functions that draw
calibration windows import their helpers locally for the same reason.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mlops_pipeline.contracts import (
    FAULT_CLASSES,
    ColumnKind,
    ColumnStats,
    FaultClass,
    Hypothesis,
    Ranking,
    column_kind,
)
from mlops_pipeline.model.data import FEATURE_COLS

if TYPE_CHECKING:
    from mlops_pipeline.monitor.stats import ReferenceProfile

#: Signals that get a robust z (the boolean flags in ColumnStats are not scored).
SIGNALS: tuple[str, ...] = (
    "psi",
    "ks",
    "null_delta",
    "out_of_range",
    "unseen_share",
    "median_log_ratio",
)

MAD_SCALE = 1.4826  # makes the MAD comparable to a standard deviation for normal data
SCALE_FLOOR = 0.01  # one percentage point for the share signals
Z_CAP = 1_000.0
TAU_PERCENTILE = 99.0
BASELINE_PSI_THRESHOLD = 0.25

#: Which column kinds each hypothesis applies to, and which signals feed its evidence.
_APPLIES_TO: dict[FaultClass, frozenset[ColumnKind]] = {
    "covariate_shift": frozenset({"continuous", "discrete"}),
    "unit_break": frozenset({"continuous"}),
    "null_spike": frozenset({"continuous", "discrete", "categorical"}),
    "recode": frozenset({"categorical"}),
}
_EVIDENCE_SIGNALS: dict[FaultClass, tuple[str, ...]] = {
    "covariate_shift": ("ks", "psi"),
    "unit_break": ("out_of_range", "median_log_ratio"),
    "null_spike": ("null_delta",),
    "recode": ("psi", "unseen_share"),
}

_COLUMN_POSITION = {c: i for i, c in enumerate(FEATURE_COLS)}
#: Most specific explanation first (Amendment 1); covariate shift is the residual one.
_SPECIFICITY: dict[FaultClass, int] = {
    "null_spike": 0,
    "unit_break": 1,
    "recode": 2,
    "covariate_shift": 3,
}
assert set(_SPECIFICITY) == set(FAULT_CLASSES)


@dataclass(frozen=True)
class NullModel:
    """What each signal looks like on no-fault windows, plus the incident threshold."""

    medians: dict[tuple[str, str], float]  # (signal, column) -> median under no fault
    scales: dict[tuple[str, str], float]  # (signal, column) -> max(1.4826 * MAD, 0.01)
    tau: float

    def to_json(self, path: str | Path) -> None:
        payload = {
            "medians": {f"{s}|{c}": v for (s, c), v in self.medians.items()},
            "scales": {f"{s}|{c}": v for (s, c), v in self.scales.items()},
            "tau": self.tau,
        }
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> NullModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

        def split_keys(table: Mapping[str, float]) -> dict[tuple[str, str], float]:
            out: dict[tuple[str, str], float] = {}
            for key, value in table.items():
                signal, column = key.split("|", 1)
                out[(signal, column)] = float(value)
            return out

        return cls(
            medians=split_keys(payload["medians"]),
            scales=split_keys(payload["scales"]),
            tau=float(payload["tau"]),
        )


def hypotheses_for(column: str) -> list[FaultClass]:
    """Fault classes that can explain a problem in this column, in class order."""
    kind = column_kind(column)
    return [c for c in FAULT_CLASSES if kind in _APPLIES_TO[c]]


def robust_z(null: NullModel, stats: Mapping[str, ColumnStats]) -> dict[tuple[str, str], float]:
    """One-sided robust z per (signal, column): max(0, (value - median) / scale), capped.

    An undefined signal (NaN) carries no evidence, so it scores 0.
    """
    out: dict[tuple[str, str], float] = {}
    for column, cs in stats.items():
        for signal in SIGNALS:
            key = (signal, column)
            value = float(getattr(cs, signal))
            if math.isnan(value):
                out[key] = 0.0
                continue
            z = (value - null.medians[key]) / null.scales[key]
            out[key] = min(max(0.0, z), Z_CAP)
    return out


def _diagnosis_key(h: Hypothesis, tau: float) -> tuple[int, float, int]:
    """Order of the hypotheses within one column (Amendment 1).

    Hypotheses at or above tau come first, most specific class first; the rest follow by
    evidence. Null-rate and out-of-range shares sit near zero when nothing is wrong, so one
    crossing tau is specific; PSI and KS rise under every fault, which makes a shift the
    explanation of last resort rather than the default winner of a capped score.
    """
    if h.evidence >= tau:
        return (0, 0.0, _SPECIFICITY[h.cause])
    return (1, -h.evidence, _SPECIFICITY[h.cause])


def _hypotheses(
    z: Mapping[tuple[str, str], float], columns: Sequence[str], tau: float
) -> tuple[Hypothesis, ...]:
    by_column: dict[str, list[Hypothesis]] = {}
    for column in columns:
        for cause in hypotheses_for(column):
            used = {s: z[(s, column)] for s in _EVIDENCE_SIGNALS[cause]}
            by_column.setdefault(column, []).append(
                Hypothesis(cause, column, max(used.values()), used)
            )
    # Columns by their strongest hypothesis, ties by FEATURE_COLS position (unchanged by
    # Amendment 1); within a column, the diagnosis order above.
    ordered = sorted(
        by_column,
        key=lambda c: (-max(h.evidence for h in by_column[c]), _COLUMN_POSITION[c]),
    )
    return tuple(
        h
        for column in ordered
        for h in sorted(by_column[column], key=lambda h: _diagnosis_key(h, tau))
    )


def rank(null: NullModel, stats: Mapping[str, ColumnStats]) -> Ranking:
    """Rank every applicable (fault class, column) pair; incident iff max evidence >= tau."""
    hypotheses = _hypotheses(robust_z(null, stats), list(stats), null.tau)
    incident = bool(hypotheses) and _max_evidence(hypotheses) >= null.tau
    return Ranking("ranker", hypotheses, incident, null.tau)


def _max_evidence(hypotheses: Sequence[Hypothesis]) -> float:
    return max(h.evidence for h in hypotheses)


def psi_baseline(stats: Mapping[str, ColumnStats]) -> Ranking:
    """Rank columns by PSI, highest first; incident iff the largest PSI exceeds 0.25.

    An undefined (NaN) PSI ranks last. Each column is labelled ``covariate_shift``, which is
    what "PSI high on X" implies. Ties break by FEATURE_COLS position.
    """
    entries = [(column, float(cs.psi)) for column, cs in stats.items()]
    entries.sort(
        key=lambda e: (
            math.isnan(e[1]),
            -e[1] if not math.isnan(e[1]) else 0.0,
            _COLUMN_POSITION[e[0]],
        )
    )
    hypotheses = tuple(
        Hypothesis("covariate_shift", column, psi, {"psi": psi}) for column, psi in entries
    )
    defined = [psi for _, psi in entries if not math.isnan(psi)]
    incident = bool(defined) and max(defined) > BASELINE_PSI_THRESHOLD
    return Ranking("psi_baseline", hypotheses, incident, BASELINE_PSI_THRESHOLD)


def _median_and_scale(values: np.ndarray) -> tuple[float, float]:
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return 0.0, SCALE_FLOOR
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median, max(MAD_SCALE * mad, SCALE_FLOOR)


def fit_null_model_from_stats(
    first: Sequence[Mapping[str, ColumnStats]],
    second: Sequence[Mapping[str, ColumnStats]],
) -> NullModel:
    """Fit the null model from per-window column statistics (no windows needed).

    ``first`` fixes the median and MAD of each signal; ``second`` is a separate set of
    no-fault windows whose maximum evidence sets tau, so tau is not fitted on the same
    windows that define the scores. tau is ``np.percentile(max_evidences, 99)`` with numpy's
    default linear interpolation.
    """
    if not first or not second:
        raise ValueError("both calibration sets need at least one window")
    medians: dict[tuple[str, str], float] = {}
    scales: dict[tuple[str, str], float] = {}
    for column in first[0]:
        for signal in SIGNALS:
            values = np.array([getattr(w[column], signal) for w in first], dtype=float)
            medians[(signal, column)], scales[(signal, column)] = _median_and_scale(values)
    provisional = NullModel(medians, scales, tau=math.inf)
    max_evidences = [_max_evidence(rank(provisional, w).hypotheses) for w in second]
    tau = float(np.percentile(max_evidences, TAU_PERCENTILE))
    return dataclasses.replace(provisional, tau=tau)


def fit_null_model(
    ref: ReferenceProfile,
    pool: pd.DataFrame,
    *,
    window_size: int = 1000,
    progress: Callable[[int, int], None] | None = None,
) -> NullModel:
    """Fit the null model from 200 + 200 no-fault windows drawn from the calibration pool.

    The first seed range gives medians and MADs, the second gives tau. The imports are local
    so the pure maths above does not require the window machinery.
    """
    from mlops_pipeline.incident.faults import calibration_seeds, calibration_window
    from mlops_pipeline.monitor.stats import column_stats

    first_seeds, second_seeds = calibration_seeds()
    total = len(first_seeds) + len(second_seeds)
    done = 0
    sets: list[list[dict[str, ColumnStats]]] = []
    for seeds in (first_seeds, second_seeds):
        collected = []
        for seed in seeds:
            window = calibration_window(pool, seed, size=window_size)
            collected.append(column_stats(ref, window))
            done += 1
            if progress is not None:
                progress(done, total)
        sets.append(collected)
    return fit_null_model_from_stats(sets[0], sets[1])
