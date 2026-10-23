"""Shared types every module codes against.

Fixed before any module was written (see DESIGN.md). A change here is a dated amendment in
DESIGN.md, never a silent edit, because the evaluation compares outputs built on these shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mlops_pipeline.model.data import CATEGORICAL_COLS, FEATURE_COLS, PAY_COLS

FaultClass = Literal["covariate_shift", "unit_break", "null_spike", "recode"]
FAULT_CLASSES: tuple[FaultClass, ...] = ("covariate_shift", "unit_break", "null_spike", "recode")
CONTROL = "control"

ColumnKind = Literal["continuous", "discrete", "categorical"]
DISCRETE_COLS: tuple[str, ...] = tuple(PAY_COLS)
CONTINUOUS_COLS: tuple[str, ...] = tuple(
    c for c in FEATURE_COLS if c not in DISCRETE_COLS and c not in CATEGORICAL_COLS
)


def column_kind(column: str) -> ColumnKind:
    """Continuous amounts and age, discrete repayment-status codes, or nominal categories."""
    if column in CATEGORICAL_COLS:
        return "categorical"
    if column in DISCRETE_COLS:
        return "discrete"
    if column in CONTINUOUS_COLS:
        return "continuous"
    raise KeyError(f"{column!r} is not a model feature")


@dataclass(frozen=True)
class ColumnStats:
    """One column of a current window compared with the reference. Raw values, not scores.

    psi              quantile-binned PSI (vendored function, raw values incl. nulls) for
                     continuous/discrete; one-bin-per-reference-code PSI for categorical.
    ks               two-sample KS statistic on non-null values; 0.0 for categorical or when
                     the window has no non-null value.
    null_delta       null rate in the window minus null rate in the reference.
    out_of_range     share of non-null window values outside [reference min, reference max].
    unseen_share     share of non-null window values absent from the reference value set
                     (discrete and categorical only; 0.0 for continuous).
    median_log_ratio |log10((|median_cur| + 1) / (|median_ref| + 1))| on non-null values.
    code_set_changed discrete/categorical: the set of codes seen differs from the reference.
    became_constant  window has one distinct non-null value while the reference has more.
    """

    column: str
    psi: float
    ks: float
    null_delta: float
    out_of_range: float
    unseen_share: float
    median_log_ratio: float
    code_set_changed: bool
    became_constant: bool


@dataclass(frozen=True)
class Hypothesis:
    """One candidate cause: a fault class on a column, with the evidence that ranks it."""

    cause: FaultClass
    column: str
    evidence: float
    signals: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Ranking:
    """A responder's answer for one window: hypotheses sorted by evidence, highest first."""

    method: Literal["ranker", "psi_baseline"]
    hypotheses: tuple[Hypothesis, ...]
    incident: bool
    threshold: float

    def column_order(self) -> list[str]:
        """Columns in order of their first appearance in the hypothesis list."""
        seen: dict[str, None] = {}
        for h in self.hypotheses:
            seen.setdefault(h.column, None)
        return list(seen)

    @property
    def top(self) -> Hypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None
