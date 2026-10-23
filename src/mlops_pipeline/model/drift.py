# Vendored from github.com/Prithv122/fraud-calibrated @ ebf6770, src/fraud_calibrated/drift.py
# (MIT, same author). Imports rewritten to this package; logic unchanged.
"""Covariate-shift detection between two populations.

Most portfolio "drift" sections inject synthetic shift (add noise, shift a mean) and
then detect the thing they just added -- which proves the metric works, not that the
model degrades under a shift that could actually happen. This module instead uses
:func:`mlops_pipeline.model.data.stratum_split` to carve out a real subpopulation (younger
customers, or a specific education level) and measures both distribution shift (PSI)
and the resulting drop in model performance on that slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, *, n_bins: int = 10
) -> float:
    """PSI between two 1-D numeric samples, binned on the reference distribution's
    own quantiles so each reference bin starts with equal mass.

    Rule of thumb this project uses in the README: <0.1 no meaningful shift, 0.1-0.25
    moderate, >0.25 significant -- the thresholds Siddiqi (2006) popularised in credit
    scoring, cited because this is precisely that domain.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        # Degenerate reference distribution (e.g. near-constant column); PSI is
        # undefined rather than falsely zero or infinite.
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_frac = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_frac = np.clip(cur_counts / max(len(current), 1), 1e-6, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def feature_psi_table(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """PSI per numeric column, sorted worst-first."""
    rows = []
    for col in columns:
        psi = population_stability_index(reference[col].to_numpy(), current[col].to_numpy())
        rows.append({"feature": col, "psi": psi})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


@dataclass(frozen=True)
class PerformanceDrop:
    reference_auc: float
    current_auc: float
    reference_pr_auc: float
    current_pr_auc: float

    @property
    def auc_delta(self) -> float:
        return self.current_auc - self.reference_auc

    @property
    def pr_auc_delta(self) -> float:
        return self.current_pr_auc - self.reference_pr_auc


def performance_drop(
    y_reference: np.ndarray,
    prob_reference: np.ndarray,
    y_current: np.ndarray,
    prob_current: np.ndarray,
) -> PerformanceDrop:
    """AUC / PR-AUC on the reference slice vs. the shifted slice, same fixed model."""
    return PerformanceDrop(
        reference_auc=float(roc_auc_score(y_reference, prob_reference)),
        current_auc=float(roc_auc_score(y_current, prob_current)),
        reference_pr_auc=float(average_precision_score(y_reference, prob_reference)),
        current_pr_auc=float(average_precision_score(y_current, prob_current)),
    )
