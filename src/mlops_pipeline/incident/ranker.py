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
#: Exploratory arm S1 (PREREGISTRATION.md, Amendment 2): the same evidence except that
#: ``unit_break`` reads ``out_of_range`` only. ``median_log_ratio`` is noisy on columns whose
#: median sits on a coarse grid, where its no-fault MAD collapses to the 0.01 floor. It never
#: feeds H1 or H2.
NO_MLR_EVIDENCE_SIGNALS: dict[FaultClass, tuple[str, ...]] = {
    **_EVIDENCE_SIGNALS,
    "unit_break": ("out_of_range",),
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
    """What each signal looks like on no-fault windows, plus the incident threshold.

    The scores are calibrated for windows of ``window_size`` rows: PSI, KS and the share
    signals all get noisier on a smaller window, so a smaller one reads as drift. Live callers
    use it as the minimum window they will judge. ``reference_sha256`` ties a saved model to
    the reference file it was fitted on. ``tau_no_mlr`` is the threshold of the exploratory
    variant (``NO_MLR_EVIDENCE_SIGNALS``); the medians and scales are shared because both
    variants read the same signals.
    """

    medians: dict[tuple[str, str], float]  # (signal, column) -> median under no fault
    scales: dict[tuple[str, str], float]  # (signal, column) -> max(1.4826 * MAD, 0.01)
    tau: float
    window_size: int = 1000
    reference_sha256: str | None = None
    tau_no_mlr: float | None = None

    def to_json(self, path: str | Path) -> None:
        payload = {
            "medians": {f"{s}|{c}": v for (s, c), v in self.medians.items()},
            "scales": {f"{s}|{c}": v for (s, c), v in self.scales.items()},
            "tau": self.tau,
            "window_size": self.window_size,
            "reference_sha256": self.reference_sha256,
            "tau_no_mlr": self.tau_no_mlr,
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
            window_size=int(payload.get("window_size", 1000)),
            reference_sha256=payload.get("reference_sha256"),
            tau_no_mlr=None if payload.get("tau_no_mlr") is None else float(payload["tau_no_mlr"]),
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
    z: Mapping[tuple[str, str], float],
    columns: Sequence[str],
    tau: float,
    evidence_signals: Mapping[FaultClass, tuple[str, ...]] = _EVIDENCE_SIGNALS,
) -> tuple[Hypothesis, ...]:
    by_column: dict[str, list[Hypothesis]] = {}
    for column in columns:
        for cause in hypotheses_for(column):
            used = {s: z[(s, column)] for s in evidence_signals[cause]}
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


def rank(
    null: NullModel,
    stats: Mapping[str, ColumnStats],
    *,
    evidence_signals: Mapping[FaultClass, tuple[str, ...]] | None = None,
    tau: float | None = None,
) -> Ranking:
    """Rank every applicable (fault class, column) pair; incident iff max evidence >= tau.

    With no keywords this is the frozen ranker. ``evidence_signals`` swaps which signals feed
    each class; the threshold depends on that choice, so pass the matching ``tau`` with it
    (``rank_no_mlr`` does).
    """
    signals = _EVIDENCE_SIGNALS if evidence_signals is None else evidence_signals
    threshold = null.tau if tau is None else tau
    hypotheses = _hypotheses(robust_z(null, stats), list(stats), threshold, signals)
    incident = bool(hypotheses) and _max_evidence(hypotheses) >= threshold
    return Ranking("ranker", hypotheses, incident, threshold)


def rank_no_mlr(null: NullModel, stats: Mapping[str, ColumnStats]) -> Ranking:
    """The exploratory variant: ``unit_break`` without ``median_log_ratio``, with its own tau."""
    if null.tau_no_mlr is None:
        raise ValueError("this null model has no tau for the variant; refit it")
    return rank(null, stats, evidence_signals=NO_MLR_EVIDENCE_SIGNALS, tau=null.tau_no_mlr)


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
    *,
    evidence_signals: Mapping[FaultClass, tuple[str, ...]] | None = None,
    variant_signals: Mapping[FaultClass, tuple[str, ...]] | None = None,
    window_size: int = 1000,
) -> NullModel:
    """Fit the null model from per-window column statistics (no windows needed).

    ``first`` fixes the median and MAD of each signal; ``second`` is a separate set of
    no-fault windows whose maximum evidence sets tau, so tau is not fitted on the same
    windows that define the scores. tau is ``np.percentile(max_evidences, 99)`` with numpy's
    default linear interpolation.

    ``evidence_signals`` (default: the frozen mapping) says which signals feed tau.
    ``variant_signals``, when given, also fits ``tau_no_mlr`` for that mapping from the same
    second set; medians and scales do not depend on the mapping, so one fit serves both.
    ``window_size`` records the length the calibration windows had.
    """
    if not first or not second:
        raise ValueError("both calibration sets need at least one window")
    medians: dict[tuple[str, str], float] = {}
    scales: dict[tuple[str, str], float] = {}
    for column in first[0]:
        for signal in SIGNALS:
            values = np.array([getattr(w[column], signal) for w in first], dtype=float)
            medians[(signal, column)], scales[(signal, column)] = _median_and_scale(values)
    provisional = NullModel(medians, scales, tau=math.inf, window_size=window_size)

    def fit_tau(signals: Mapping[FaultClass, tuple[str, ...]]) -> float:
        # The maximum evidence does not depend on tau; only the order inside a column does.
        peaks = [
            _max_evidence(_hypotheses(robust_z(provisional, w), list(w), math.inf, signals))
            for w in second
        ]
        return float(np.percentile(peaks, TAU_PERCENTILE))

    tau = fit_tau(_EVIDENCE_SIGNALS if evidence_signals is None else evidence_signals)
    tau_no_mlr = None if variant_signals is None else fit_tau(variant_signals)
    return dataclasses.replace(provisional, tau=tau, tau_no_mlr=tau_no_mlr)


def fit_null_model(
    ref: ReferenceProfile,
    pool: pd.DataFrame,
    *,
    window_size: int = 1000,
    evidence_signals: Mapping[FaultClass, tuple[str, ...]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> NullModel:
    """Fit the null model from 200 + 200 no-fault windows drawn from the calibration pool.

    The first seed range gives medians and MADs, the second gives tau, and from the same
    second set ``tau_no_mlr`` for the exploratory variant. The imports are local so the pure
    maths above does not require the window machinery.
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
    return fit_null_model_from_stats(
        sets[0],
        sets[1],
        evidence_signals=evidence_signals,
        variant_signals=NO_MLR_EVIDENCE_SIGNALS,
        window_size=window_size,
    )
