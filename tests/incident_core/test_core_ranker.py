"""Ranker maths on hand-made ColumnStats: no windows, no reference profile."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlops_pipeline.contracts import CONTINUOUS_COLS, ColumnStats
from mlops_pipeline.incident.ranker import (
    SIGNALS,
    NullModel,
    fit_null_model_from_stats,
    hypotheses_for,
    psi_baseline,
    rank,
    robust_z,
)
from mlops_pipeline.model.data import BILL_COLS, FEATURE_COLS, PAY_COLS


def stats_for(column: str, **values: float) -> ColumnStats:
    base = {
        "psi": 0.0,
        "ks": 0.0,
        "null_delta": 0.0,
        "out_of_range": 0.0,
        "unseen_share": 0.0,
        "median_log_ratio": 0.0,
        "code_set_changed": False,
        "became_constant": False,
    }
    base.update(values)
    return ColumnStats(column=column, **base)


def window(**per_column: dict[str, float]) -> dict[str, ColumnStats]:
    """All 23 columns at zero except the overrides, e.g. window(AGE={"ks": 2.0})."""
    return {c: stats_for(c, **per_column.get(c, {})) for c in FEATURE_COLS}


def unit_null(tau: float = 1.0) -> NullModel:
    """Median 0 and scale 1 everywhere, so a z-score equals the raw signal value."""
    keys = [(s, c) for s in SIGNALS for c in FEATURE_COLS]
    return NullModel({k: 0.0 for k in keys}, {k: 1.0 for k in keys}, tau)


# ---------------------------------------------------------------- robust z


def test_robust_z_is_scaled_one_sided_and_capped() -> None:
    null = unit_null()
    medians = {**null.medians, ("psi", "AGE"): 0.1}
    scales = {**null.scales, ("psi", "AGE"): 0.5}
    null = NullModel(medians, scales, 1.0)

    z = robust_z(null, window(AGE={"psi": 0.6}))
    assert z[("psi", "AGE")] == pytest.approx(1.0)  # (0.6 - 0.1) / 0.5

    below = robust_z(null, window(AGE={"psi": 0.0}))
    assert below[("psi", "AGE")] == 0.0  # one-sided: below the median is not evidence

    huge = robust_z(null, window(AGE={"psi": 1e9}))
    assert huge[("psi", "AGE")] == 1000.0  # capped


def test_robust_z_nan_signal_carries_no_evidence() -> None:
    z = robust_z(unit_null(), window(AGE={"psi": math.nan}))
    assert z[("psi", "AGE")] == 0.0


def test_robust_z_covers_every_signal_and_column() -> None:
    z = robust_z(unit_null(), window())
    assert set(z) == {(s, c) for s in SIGNALS for c in FEATURE_COLS}


def test_fit_uses_mad_scaled_by_1_4826() -> None:
    # psi on AGE over five windows: median 0.3, |deviations| = .2 .1 0 .1 .2 -> MAD 0.1
    first = [window(AGE={"psi": v}) for v in (0.1, 0.2, 0.3, 0.4, 0.5)]
    null = fit_null_model_from_stats(first, [window()])
    assert null.medians[("psi", "AGE")] == pytest.approx(0.3)
    assert null.scales[("psi", "AGE")] == pytest.approx(1.4826 * 0.1)


def test_fit_scale_floor_for_signals_that_never_move() -> None:
    first = [window() for _ in range(5)]
    null = fit_null_model_from_stats(first, [window()])
    assert null.medians[("null_delta", "PAY_1")] == 0.0
    assert null.scales[("null_delta", "PAY_1")] == 0.01  # MAD 0 would divide by zero


def test_fit_scale_above_floor_is_not_floored() -> None:
    first = [window(AGE={"ks": v}) for v in (0.0, 0.2, 0.4)]  # median 0.2, MAD 0.2
    null = fit_null_model_from_stats(first, [window()])
    assert null.scales[("ks", "AGE")] == pytest.approx(1.4826 * 0.2)


def test_fit_requires_both_calibration_sets() -> None:
    with pytest.raises(ValueError):
        fit_null_model_from_stats([], [window()])
    with pytest.raises(ValueError):
        fit_null_model_from_stats([window()], [])


# ------------------------------------------------------------------- tau


def test_tau_is_99th_percentile_of_max_evidence_linear_interpolation() -> None:
    first = [window() for _ in range(3)]  # median 0, scale floor 0.01 for every signal
    # Second windows: ks on AGE = 0.01 * e, so the max evidence of window e is exactly e.
    second = [window(AGE={"ks": 0.01 * e}) for e in range(1, 101)]
    null = fit_null_model_from_stats(first, second)
    expected = float(np.percentile(np.arange(1, 101, dtype=float), 99))
    assert expected == pytest.approx(99.01)
    assert null.tau == pytest.approx(expected)


def test_tau_comes_from_second_set_only() -> None:
    first = [window(AGE={"ks": 0.05}) for _ in range(3)]  # moves the median, not tau's source
    second = [window(), window()]
    null = fit_null_model_from_stats(first, second)
    assert null.medians[("ks", "AGE")] == pytest.approx(0.05)
    assert null.tau == 0.0  # both second windows sit at the median: zero evidence everywhere


# --------------------------------------------------------- evidence table


def test_hypotheses_apply_to_the_right_column_kinds() -> None:
    assert hypotheses_for("AGE") == ["covariate_shift", "unit_break", "null_spike"]
    assert hypotheses_for("LIMIT_BAL") == ["covariate_shift", "unit_break", "null_spike"]
    assert hypotheses_for("PAY_3") == ["covariate_shift", "null_spike"]
    assert hypotheses_for("SEX") == ["null_spike", "recode"]
    assert hypotheses_for("EDUCATION") == ["null_spike", "recode"]
    assert hypotheses_for("MARRIAGE") == ["null_spike", "recode"]


def test_rank_enumerates_exactly_the_applicable_hypotheses() -> None:
    ranking = rank(unit_null(), window())
    pairs = {(h.cause, h.column) for h in ranking.hypotheses}
    assert len(ranking.hypotheses) == len(pairs) == 14 * 3 + 6 * 2 + 3 * 2
    assert ("recode", "AGE") not in pairs
    assert ("unit_break", "SEX") not in pairs
    assert ("unit_break", "PAY_1") not in pairs
    assert ("covariate_shift", "EDUCATION") not in pairs
    assert {c for cause, c in pairs if cause == "null_spike"} == set(FEATURE_COLS)
    assert {c for cause, c in pairs if cause == "unit_break"} == set(CONTINUOUS_COLS)


def test_evidence_is_max_of_the_signals_each_class_reads() -> None:
    # unit_break reads out_of_range and median_log_ratio, covariate_shift reads ks and psi.
    w = window(
        BILL_AMT3={"out_of_range": 0.4, "median_log_ratio": 3.0, "psi": 7.0},
        AGE={"ks": 2.0, "psi": 1.0},
        PAY_2={"null_delta": 0.5, "ks": 0.25},
        EDUCATION={"psi": 4.0, "unseen_share": 6.0, "ks": 99.0},
    )
    by_pair = {(h.cause, h.column): h.evidence for h in rank(unit_null(), w).hypotheses}
    assert by_pair[("unit_break", "BILL_AMT3")] == 3.0
    assert by_pair[("covariate_shift", "BILL_AMT3")] == 7.0
    assert by_pair[("covariate_shift", "AGE")] == 2.0
    assert by_pair[("null_spike", "PAY_2")] == 0.5
    assert by_pair[("covariate_shift", "PAY_2")] == 0.25
    assert by_pair[("recode", "EDUCATION")] == 6.0  # ks is not read by recode
    assert by_pair[("null_spike", "EDUCATION")] == 0.0


def test_hypothesis_signals_hold_the_z_scores_used() -> None:
    w = window(AGE={"ks": 2.0, "psi": 1.0})
    top = next(h for h in rank(unit_null(), w).hypotheses if h.cause == "covariate_shift")
    assert top.column == "AGE"
    assert top.signals == {"ks": 2.0, "psi": 1.0}


# ----------------------------------------------------------------- ties


def test_all_zero_evidence_orders_by_column_position_then_class() -> None:
    ranking = rank(unit_null(), window())
    first_three = [(h.column, h.cause) for h in ranking.hypotheses[:3]]
    assert first_three == [
        ("LIMIT_BAL", "covariate_shift"),
        ("LIMIT_BAL", "unit_break"),
        ("LIMIT_BAL", "null_spike"),
    ]
    last = ranking.hypotheses[-1]
    assert (last.column, last.cause) == ("MARRIAGE", "recode")


def test_ties_break_by_column_position_before_class() -> None:
    # PAY_1 covariate_shift and AGE null_spike both score 1.0: AGE comes first in FEATURE_COLS.
    w = window(PAY_1={"ks": 1.0}, AGE={"null_delta": 1.0})
    ranking = rank(unit_null(), w)
    assert [(h.column, h.cause) for h in ranking.hypotheses[:2]] == [
        ("AGE", "null_spike"),
        ("PAY_1", "covariate_shift"),
    ]


def test_ties_on_the_same_column_break_by_class_order() -> None:
    w = window(LIMIT_BAL={"ks": 2.0, "out_of_range": 2.0, "null_delta": 2.0})
    top3 = rank(unit_null(), w).hypotheses[:3]
    assert [h.cause for h in top3] == ["covariate_shift", "unit_break", "null_spike"]


def test_higher_evidence_beats_earlier_position() -> None:
    w = window(AGE={"ks": 1.0}, MARRIAGE={"unseen_share": 5.0})
    top = rank(unit_null(), w).top
    assert top is not None
    assert (top.column, top.cause) == ("MARRIAGE", "recode")


def test_column_order_lists_each_column_once_by_first_appearance() -> None:
    w = window(BILL_AMT3={"median_log_ratio": 9.0}, AGE={"ks": 4.0}, SEX={"unseen_share": 2.0})
    order = rank(unit_null(), w).column_order()
    assert order[:3] == ["BILL_AMT3", "AGE", "SEX"]
    assert sorted(order) == sorted(FEATURE_COLS)
    assert len(order) == len(set(order)) == 23
    # Remaining columns are all tied at zero, so they follow FEATURE_COLS order.
    rest = [c for c in order[3:]]
    assert rest == [c for c in FEATURE_COLS if c not in {"BILL_AMT3", "AGE", "SEX"}]


def test_incident_when_max_evidence_reaches_tau() -> None:
    null = unit_null(tau=3.0)
    assert rank(null, window(AGE={"ks": 3.0})).incident is True  # >= tau
    assert rank(null, window(AGE={"ks": 2.99})).incident is False
    ranking = rank(null, window())
    assert ranking.method == "ranker"
    assert ranking.threshold == 3.0


# --------------------------------------------------------------- baseline


def test_baseline_orders_by_psi_with_nan_last_and_position_ties() -> None:
    w = window(
        AGE={"psi": 0.1},
        LIMIT_BAL={"psi": math.nan},
        PAY_1={"psi": 0.3},
        SEX={"psi": 0.3},
        BILL_AMT2={"psi": 0.2},
    )
    ranking = psi_baseline(w)
    order = ranking.column_order()
    assert order[:4] == ["PAY_1", "SEX", "BILL_AMT2", "AGE"]  # tie: PAY_1 before SEX
    assert order[-1] == "LIMIT_BAL"  # undefined PSI ranks last
    assert len(order) == 23
    assert {h.cause for h in ranking.hypotheses} == {"covariate_shift"}
    assert ranking.method == "psi_baseline"


def test_baseline_incident_is_strictly_above_0_25() -> None:
    assert psi_baseline(window(AGE={"psi": 0.25})).incident is False
    assert psi_baseline(window(AGE={"psi": 0.2501})).incident is True
    assert psi_baseline(window(AGE={"psi": 0.1})).incident is False
    assert psi_baseline(window()).threshold == 0.25


def test_baseline_all_nan_psi_is_no_incident() -> None:
    w = {c: stats_for(c, psi=math.nan) for c in FEATURE_COLS}
    ranking = psi_baseline(w)
    assert ranking.incident is False
    assert ranking.column_order() == list(FEATURE_COLS)


def test_baseline_and_ranker_share_the_bill_columns_example() -> None:
    # A unit break moves all six BILL columns; the baseline's top column is one of them.
    w = window(**{c: {"psi": 1.0 + i / 10} for i, c in enumerate(BILL_COLS)})
    assert psi_baseline(w).column_order()[0] == "BILL_AMT6"
    assert set(PAY_COLS).isdisjoint(psi_baseline(w).column_order()[:6])


# ------------------------------------------------------------------- JSON


def test_null_model_json_round_trip(tmp_path) -> None:
    keys = [(s, c) for s in SIGNALS for c in FEATURE_COLS]
    null = NullModel(
        {k: 0.01 * i for i, k in enumerate(keys)},
        {k: 0.01 + i / 1000 for i, k in enumerate(keys)},
        7.25,
    )
    path = tmp_path / "null.json"
    null.to_json(path)
    assert '"psi|AGE"' in path.read_text(encoding="utf-8")
    assert NullModel.from_json(path) == null
