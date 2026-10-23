"""Ranker maths on hand-made ColumnStats: no windows, no reference profile."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from mlops_pipeline.contracts import CONTINUOUS_COLS, ColumnStats
from mlops_pipeline.incident.ranker import (
    _EVIDENCE_SIGNALS,
    NO_MLR_EVIDENCE_SIGNALS,
    SIGNALS,
    NullModel,
    fit_null_model_from_stats,
    hypotheses_for,
    psi_baseline,
    rank,
    rank_no_mlr,
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


def test_all_zero_evidence_orders_by_column_position_then_specificity() -> None:
    ranking = rank(unit_null(), window())
    first_three = [(h.column, h.cause) for h in ranking.hypotheses[:3]]
    assert first_three == [
        ("LIMIT_BAL", "null_spike"),
        ("LIMIT_BAL", "unit_break"),
        ("LIMIT_BAL", "covariate_shift"),
    ]
    last = ranking.hypotheses[-1]
    assert (last.column, last.cause) == ("MARRIAGE", "recode")


def test_ties_break_by_column_position_before_class() -> None:
    # PAY_1 covariate_shift and AGE null_spike both score 1.0: AGE comes first in FEATURE_COLS.
    w = window(PAY_1={"ks": 1.0}, AGE={"null_delta": 1.0})
    ranking = rank(unit_null(), w)
    assert ranking.column_order()[:2] == ["AGE", "PAY_1"]
    assert ranking.top is not None
    assert (ranking.top.column, ranking.top.cause) == ("AGE", "null_spike")


def test_ties_on_the_same_column_break_by_specificity() -> None:
    w = window(LIMIT_BAL={"ks": 2.0, "out_of_range": 2.0, "null_delta": 2.0})
    top3 = rank(unit_null(), w).hypotheses[:3]
    assert [h.cause for h in top3] == ["null_spike", "unit_break", "covariate_shift"]


# ------------------------------------------------- Amendment 1: diagnosis


def test_specific_class_above_tau_beats_a_larger_capped_shift_score() -> None:
    # A x1000 unit break: PSI saturates, out-of-range share is large but smaller in z.
    w = window(BILL_AMT1={"psi": 1000.0, "ks": 900.0, "out_of_range": 22.0})
    ranking = rank(unit_null(tau=10.0), w)
    assert ranking.top is not None
    assert (ranking.top.column, ranking.top.cause) == ("BILL_AMT1", "unit_break")
    assert ranking.incident


def test_below_tau_the_column_hypotheses_order_by_evidence() -> None:
    w = window(AGE={"ks": 5.0, "out_of_range": 2.0})
    ranking = rank(unit_null(tau=10.0), w)
    assert [h.cause for h in ranking.hypotheses[:2]] == ["covariate_shift", "unit_break"]
    assert not ranking.incident


def test_column_order_follows_each_columns_strongest_hypothesis() -> None:
    # The diagnosis reorders hypotheses inside a column, never the columns themselves.
    w = window(AGE={"ks": 50.0, "out_of_range": 12.0}, PAY_4={"null_delta": 40.0})
    ranking = rank(unit_null(tau=10.0), w)
    assert ranking.column_order()[:2] == ["AGE", "PAY_4"]
    assert [(h.column, h.cause) for h in ranking.hypotheses[:2]] == [
        ("AGE", "unit_break"),
        ("AGE", "covariate_shift"),
    ]


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


def test_null_model_json_round_trips_window_size_hash_and_variant_tau(tmp_path) -> None:
    keys = [(s, c) for s in SIGNALS for c in FEATURE_COLS]
    null = NullModel(
        {k: 0.0 for k in keys},
        {k: 1.0 for k in keys},
        7.25,
        window_size=250,
        reference_sha256="ab" * 32,
        tau_no_mlr=4.5,
    )
    path = tmp_path / "null.json"
    null.to_json(path)
    loaded = NullModel.from_json(path)
    assert loaded == null
    assert (loaded.window_size, loaded.reference_sha256, loaded.tau_no_mlr) == (250, "ab" * 32, 4.5)


def test_null_model_json_without_the_new_fields_gets_defaults(tmp_path) -> None:
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps({"medians": {"psi|AGE": 0.1}, "scales": {"psi|AGE": 0.5}, "tau": 3.0}),
        encoding="utf-8",
    )
    loaded = NullModel.from_json(path)
    assert loaded.window_size == 1000
    assert loaded.reference_sha256 is None and loaded.tau_no_mlr is None


def test_fit_records_the_window_size_it_was_fitted_at() -> None:
    null = fit_null_model_from_stats([window()], [window()], window_size=200)
    assert null.window_size == 200
    assert fit_null_model_from_stats([window()], [window()]).window_size == 1000


# ------------------------------------------------- exploratory variant (S1)


def _mixed_window() -> dict[str, ColumnStats]:
    return window(
        AGE={"ks": 0.8, "psi": 2.0, "out_of_range": 0.4},
        BILL_AMT1={"median_log_ratio": 3.0, "out_of_range": 0.2, "psi": 1.0},
        BILL_AMT2={"median_log_ratio": 9.0},
        PAY_1={"null_delta": 0.3, "psi": 5.0},
        EDUCATION={"unseen_share": 0.5, "psi": 0.2},
    )


def test_default_ranking_is_unchanged_by_passing_the_default_mapping() -> None:
    null = unit_null(tau=1.5)
    w = _mixed_window()
    assert rank(null, w) == rank(null, w, evidence_signals=_EVIDENCE_SIGNALS)
    assert rank(null, w) == rank(null, w, evidence_signals=dict(_EVIDENCE_SIGNALS), tau=null.tau)


def test_default_fit_is_unchanged_by_passing_the_default_mapping() -> None:
    first = [window(AGE={"psi": v}) for v in (0.1, 0.2, 0.3)]
    second = [
        window(AGE={"ks": v, "psi": 2 * v}, BILL_AMT1={"median_log_ratio": v}) for v in (1, 2, 5)
    ]
    plain = fit_null_model_from_stats(first, second)
    explicit = fit_null_model_from_stats(first, second, evidence_signals=_EVIDENCE_SIGNALS)
    assert plain == explicit
    assert plain.tau_no_mlr is None  # the variant is only fitted when asked for


def test_variant_changes_only_unit_break_evidence() -> None:
    assert NO_MLR_EVIDENCE_SIGNALS["unit_break"] == ("out_of_range",)
    for cause in ("covariate_shift", "null_spike", "recode"):
        assert NO_MLR_EVIDENCE_SIGNALS[cause] == _EVIDENCE_SIGNALS[cause]


def test_variant_ignores_median_log_ratio_for_unit_break() -> None:
    null = NullModel(**{**vars(unit_null()), "tau_no_mlr": 1.0})
    w = window(BILL_AMT2={"median_log_ratio": 9.0})
    frozen = {(h.cause, h.column): h.evidence for h in rank(null, w).hypotheses}
    variant = {(h.cause, h.column): h.evidence for h in rank_no_mlr(null, w).hypotheses}
    assert frozen[("unit_break", "BILL_AMT2")] == 9.0
    assert variant[("unit_break", "BILL_AMT2")] == 0.0
    # every other hypothesis is identical
    assert {k: v for k, v in frozen.items() if k[0] != "unit_break"} == {
        k: v for k, v in variant.items() if k[0] != "unit_break"
    }
    # and with an out_of_range signal the variant still sees the break
    w = window(BILL_AMT2={"median_log_ratio": 9.0, "out_of_range": 4.0})
    assert rank_no_mlr(null, w).top.cause == "unit_break"


def test_variant_uses_its_own_tau_for_the_incident_verdict() -> None:
    null = NullModel(**{**vars(unit_null(tau=5.0)), "tau_no_mlr": 2.0})
    w = window(AGE={"ks": 3.0})
    assert not rank(null, w).incident and rank_no_mlr(null, w).incident
    assert rank_no_mlr(null, w).threshold == 2.0


def test_rank_no_mlr_needs_a_fitted_variant_tau() -> None:
    with pytest.raises(ValueError):
        rank_no_mlr(unit_null(), window())


def test_variant_tau_comes_from_the_second_set_and_the_variant_mapping() -> None:
    first = [window() for _ in range(5)]  # medians 0, scale floor 0.01 everywhere
    # z = value / 0.01: median_log_ratio 0.5 -> 50, out_of_range 0.02 -> 2
    second = [window(BILL_AMT1={"median_log_ratio": 0.5, "out_of_range": 0.02}) for _ in range(10)]
    null = fit_null_model_from_stats(first, second, variant_signals=NO_MLR_EVIDENCE_SIGNALS)
    assert null.tau == pytest.approx(50.0)
    assert null.tau_no_mlr == pytest.approx(2.0)

    # the first set only moves the medians: with a median_log_ratio median of 0.5 that signal
    # scores 0 on the second set, so the frozen tau falls to the out_of_range score as well
    other_first = [window(BILL_AMT1={"median_log_ratio": 0.5}) for _ in range(5)]
    shifted = fit_null_model_from_stats(
        other_first, second, variant_signals=NO_MLR_EVIDENCE_SIGNALS
    )
    assert shifted.tau == pytest.approx(2.0)
    assert shifted.tau_no_mlr == pytest.approx(2.0)
