"""Seeded fault injection: the scenarios the responder is scored on.

A scenario is a fault class, a severity index and a seed. Everything random about it comes from
one generator, ``default_rng([seed, class_index, severity_index])``, which first draws the
window rows from the pool and then (for the share-based faults) the affected rows. The same
scenario therefore always yields the same window, and the protocol in PREREGISTRATION.md can be
re-run exactly. Severity values and fault definitions are fixed there; this module only
implements them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlops_pipeline.contracts import CONTROL, FAULT_CLASSES, FaultClass
from mlops_pipeline.model.data import BILL_COLS

#: Severity 0 / 1 / 2 per class: k in units of reference sigma for covariate_shift, and the
#: share of affected rows for the other three. Fixed before any run (PREREGISTRATION.md).
SEVERITIES: dict[FaultClass, tuple[float, float, float]] = {
    "covariate_shift": (0.10, 0.25, 0.50),
    "unit_break": (0.05, 0.25, 1.00),
    "null_spike": (0.05, 0.25, 1.00),
    "recode": (0.05, 0.25, 1.00),
}

#: Position in the generator's seed sequence. Control is 4; the no-fault calibration windows
#: that fit the null model use 5, so they never share a stream with a scenario.
CLASS_INDEX: dict[str, int] = {
    "covariate_shift": 0,
    "unit_break": 1,
    "null_spike": 2,
    "recode": 3,
    CONTROL: 4,
}
CALIBRATION_CLASS_INDEX = 5

#: Grid spacing each shifted column is rounded back to (the data is recorded on these grids).
_GRID: dict[str, float] = {"AGE": 1.0, "LIMIT_BAL": 10_000.0}

EVAL_SEEDS = 20
EVAL_CONTROLS = 200
DEV_SEED_START = 10_000


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    fault: str  # a FaultClass or CONTROL
    severity: int  # 0..2; 0 for control
    seed: int
    columns: tuple[str, ...]  # affected columns; () for control


def affected_columns(fault: str, seed: int) -> tuple[str, ...]:
    """The columns a fault of this class touches for this seed."""
    if fault == "covariate_shift":
        return ("AGE",) if seed % 2 == 0 else ("LIMIT_BAL",)
    if fault == "unit_break":
        return tuple(BILL_COLS)
    if fault == "null_spike":
        return (f"PAY_{(seed % 6) + 1}",)
    if fault == "recode":
        return ("EDUCATION",)
    if fault == CONTROL:
        return ()
    raise ValueError(f"unknown fault class {fault!r}")


def make_scenario(fault: str, severity: int, seed: int) -> Scenario:
    """Build one scenario with its id and affected columns."""
    if fault == CONTROL:
        severity = 0
    elif fault not in FAULT_CLASSES:
        raise ValueError(f"unknown fault class {fault!r}")
    elif severity not in (0, 1, 2):
        raise ValueError(f"severity index must be 0, 1 or 2, got {severity}")
    return Scenario(
        scenario_id=f"{fault}-s{severity}-seed{seed:05d}",
        fault=fault,
        severity=severity,
        seed=seed,
        columns=affected_columns(fault, seed),
    )


def _grid(seed_start: int, n_seeds: int, n_controls: int) -> list[Scenario]:
    scenarios = [
        make_scenario(fault, severity, seed)
        for fault in FAULT_CLASSES
        for severity in range(3)
        for seed in range(seed_start, seed_start + n_seeds)
    ]
    scenarios += [
        make_scenario(CONTROL, 0, seed) for seed in range(seed_start, seed_start + n_controls)
    ]
    return scenarios


def eval_grid() -> list[Scenario]:
    """The frozen evaluation grid: 4 classes x 3 severities x seeds 0-19 plus 200 controls."""
    return _grid(0, EVAL_SEEDS, EVAL_CONTROLS)


def dev_grid(n_seeds: int = 2, n_controls: int = 10) -> list[Scenario]:
    """A small grid for development and tests. Seeds start at 10,000, never overlapping the
    evaluation seeds."""
    return _grid(DEV_SEED_START, n_seeds, n_controls)


def calibration_seeds() -> tuple[range, range]:
    """Seeds of the no-fault windows: (null-model fit, threshold fit). class_index 5."""
    return range(0, 200), range(1000, 1200)


def _rng(seed: int, class_index: int, severity: int) -> np.random.Generator:
    return np.random.default_rng([seed, class_index, severity])


def _draw_rows(pool: pd.DataFrame, rng: np.random.Generator, size: int) -> pd.DataFrame:
    """``size`` distinct rows of ``pool`` as a new frame with a fresh 0..size-1 index."""
    if size > len(pool):
        raise ValueError(f"window size {size} exceeds pool size {len(pool)}")
    rows = rng.choice(len(pool), size, replace=False)
    return pool.iloc[rows].reset_index(drop=True)


def draw_window(
    pool: pd.DataFrame, scenario: Scenario, reference: pd.DataFrame, *, size: int = 1000
) -> pd.DataFrame:
    """Draw the scenario's window from ``pool`` and apply its fault. The pool is not modified."""
    if scenario.fault not in CLASS_INDEX:
        raise ValueError(f"unknown fault class {scenario.fault!r}")
    rng = _rng(scenario.seed, CLASS_INDEX[scenario.fault], scenario.severity)
    window = _draw_rows(pool, rng, size)
    if scenario.fault == CONTROL:
        return window
    return apply_fault(window, scenario.fault, scenario.severity, scenario.columns, rng, reference)


def calibration_window(pool: pd.DataFrame, seed: int, *, size: int = 1000) -> pd.DataFrame:
    """A no-fault window for fitting the null model: rng([seed, 5, 0])."""
    return _draw_rows(pool, _rng(seed, CALIBRATION_CLASS_INDEX, 0), size)


def apply_fault(
    window: pd.DataFrame,
    fault: str,
    severity: int,
    columns: tuple[str, ...],
    rng: np.random.Generator,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy of ``window`` with the fault applied; the input is left untouched.

    Share-based faults pick ``round(share * len(window))`` rows without replacement (Python
    ``round``, half to even; with the preregistered shares and 1,000 rows every product is a
    whole number, so the rule never decides anything). Covariate shift draws no randomness: it
    moves every row.
    """
    if fault not in FAULT_CLASSES:
        raise ValueError(f"unknown fault class {fault!r}")
    level = SEVERITIES[fault][severity]
    out = window.copy()

    if fault == "covariate_shift":
        for col in columns:
            # sigma_ref is the sample standard deviation (pandas default, ddof=1) of the
            # reference column. x + k*sigma lands off-grid, so snap it back to the grid the
            # column is recorded on (np.rint: half to even).
            sigma = float(reference[col].std())
            step = _GRID[col]
            shifted = np.rint((out[col].to_numpy(dtype=float) + level * sigma) / step) * step
            out[col] = shifted.astype(out[col].dtype)
        return out

    n_hit = round(level * len(out))
    rows = rng.choice(len(out), n_hit, replace=False)

    if fault == "unit_break":
        # An upstream partition switches units: all six bill amounts on the same rows x1000.
        for col in columns:
            values = out[col].to_numpy(copy=True)
            values[rows] = values[rows] * 1000
            out[col] = values
    elif fault == "null_spike":
        for col in columns:
            values = out[col].to_numpy(dtype=float, copy=True)
            values[rows] = np.nan
            out[col] = values
    elif fault == "recode":
        for col in columns:
            values = out[col].to_numpy(copy=True)
            selected = values[rows]
            values[rows] = np.where(np.isin(selected, (1, 2)), 3 - selected, selected)
            out[col] = values
    return out
