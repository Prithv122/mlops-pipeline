"""Scoring, statistics, summary and report on hand-made rankings and records."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from mlops_pipeline.contracts import CONTROL, Hypothesis, Ranking
from mlops_pipeline.evaluate import (
    RECORD_COLUMNS,
    paired_bootstrap,
    score_ranking,
    summarise,
    wilson,
    write_report,
)
from mlops_pipeline.model.data import BILL_COLS


def ranking(*hyps: tuple[str, str, float]) -> Ranking:
    return Ranking("ranker", tuple(Hypothesis(c, col, ev) for c, col, ev in hyps), True, 1.0)


# ---------------------------------------------------------------- scoring


def test_score_hit_on_any_of_the_six_bill_columns() -> None:
    r = ranking(("unit_break", "BILL_AMT5", 9.0), ("covariate_shift", "AGE", 3.0))
    s = score_ranking(r, "unit_break", BILL_COLS)
    assert s == {"top1": 1.0, "rr": 1.0, "cause_top1": 1.0}


def test_score_right_column_wrong_class_is_not_cause_top1() -> None:
    r = ranking(("covariate_shift", "BILL_AMT1", 9.0))
    s = score_ranking(r, "unit_break", BILL_COLS)
    assert s["top1"] == 1.0
    assert s["cause_top1"] == 0.0


def test_score_reciprocal_rank_counts_columns_not_hypotheses() -> None:
    # AGE appears twice, so BILL_AMT2 is the second distinct column, not the third.
    r = ranking(
        ("covariate_shift", "AGE", 9.0),
        ("null_spike", "AGE", 8.0),
        ("unit_break", "BILL_AMT2", 7.0),
    )
    s = score_ranking(r, "unit_break", BILL_COLS)
    assert s["top1"] == 0.0
    assert s["rr"] == pytest.approx(0.5)
    assert s["cause_top1"] == 0.0


def test_score_rank_three_and_miss() -> None:
    r = ranking(
        ("covariate_shift", "AGE", 9.0),
        ("covariate_shift", "LIMIT_BAL", 8.0),
        ("null_spike", "PAY_4", 7.0),
    )
    assert score_ranking(r, "null_spike", ["PAY_4"])["rr"] == pytest.approx(1 / 3)
    assert score_ranking(r, "null_spike", ["PAY_5"])["rr"] == 0.0


def test_score_single_affected_column_must_match_exactly() -> None:
    r = ranking(("covariate_shift", "LIMIT_BAL", 9.0))
    assert score_ranking(r, "covariate_shift", ["AGE"])["top1"] == 0.0
    assert score_ranking(r, "covariate_shift", ["LIMIT_BAL"])["cause_top1"] == 1.0


def test_score_control_is_not_applicable() -> None:
    s = score_ranking(ranking(("null_spike", "AGE", 1.0)), CONTROL, ())
    assert all(math.isnan(v) for v in s.values())


def test_score_empty_ranking() -> None:
    s = score_ranking(Ranking("ranker", (), False, 1.0), "recode", ["EDUCATION"])
    assert s == {"top1": 0.0, "rr": 0.0, "cause_top1": 0.0}


# ------------------------------------------------------------- statistics


def test_bootstrap_identical_arrays_gives_zero() -> None:
    a = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    assert paired_bootstrap(a, a, n=500) == (0.0, 0.0, 0.0)


def test_bootstrap_constant_difference_is_that_constant() -> None:
    a = np.full(30, 1.0)
    b = np.full(30, 0.75)
    mean, lo, hi = paired_bootstrap(a, b, n=500)
    assert mean == pytest.approx(0.25)
    assert lo == pytest.approx(0.25)
    assert hi == pytest.approx(0.25)


def test_bootstrap_is_seeded_and_interval_brackets_the_mean() -> None:
    rng = np.random.default_rng(123)
    a = rng.integers(0, 2, 120).astype(float)
    b = rng.integers(0, 2, 120).astype(float)
    first = paired_bootstrap(a, b, n=2000)
    assert first == paired_bootstrap(a, b, n=2000)
    mean, lo, hi = first
    assert lo <= mean <= hi
    assert mean == pytest.approx(float((a - b).mean()))


def test_bootstrap_matches_the_documented_procedure() -> None:
    a = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    diff = a - b
    idx = np.random.default_rng(0).integers(0, 6, size=(1000, 6))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    assert paired_bootstrap(a, b, n=1000, seed=0) == (float(diff.mean()), float(lo), float(hi))


def test_wilson_zero_of_200() -> None:
    lo, hi = wilson(0, 200)
    assert lo == 0.0
    assert hi == pytest.approx(0.0188, abs=5e-5)


def test_wilson_ten_of_one_hundred() -> None:
    lo, hi = wilson(10, 100)
    assert lo == pytest.approx(0.0552, abs=5e-4)
    assert hi == pytest.approx(0.1744, abs=5e-4)


def test_wilson_all_successes_and_no_trials() -> None:
    lo, hi = wilson(200, 200)
    assert hi == 1.0
    assert lo == pytest.approx(1 - 0.0188, abs=5e-4)
    assert all(math.isnan(v) for v in wilson(0, 0))


# ---------------------------------------------------------- summary/report


def rec(
    fault: str,
    severity: int,
    seed: int,
    *,
    r1: float,
    b1: float,
    r_inc: bool,
    b_inc: bool,
    cause: float | None = None,
) -> dict:
    """One hand-made record; fault rows carry scores, controls carry NaN."""
    is_control = fault == CONTROL
    nan = math.nan
    return {
        "scenario_id": f"{fault}-s{severity}-seed{seed:05d}",
        "fault": fault,
        "severity": severity,
        "seed": seed,
        "columns": "" if is_control else "AGE",
        "ranker_top1": nan if is_control else r1,
        "ranker_rr": nan if is_control else (1.0 if r1 else 0.5),
        "ranker_cause_top1": nan if is_control else (r1 if cause is None else cause),
        "ranker_incident": r_inc,
        "ranker_top_column": "AGE",
        "ranker_top_cause": "covariate_shift",
        "ranker_max_evidence": 5.0,
        "baseline_top1": nan if is_control else b1,
        "baseline_rr": nan if is_control else (1.0 if b1 else 0.25),
        "baseline_incident": b_inc,
        "baseline_top_column": "AGE",
        "baseline_max_psi": 0.3,
    }


def records_frame(
    fault_pairs: list[tuple[float, float]], control_incidents: int, n_controls: int
) -> pd.DataFrame:
    rows = [
        rec("covariate_shift", i % 3, 10_000 + i, r1=r, b1=b, r_inc=True, b_inc=bool(b))
        for i, (r, b) in enumerate(fault_pairs)
    ]
    rows += [
        rec(CONTROL, 0, 10_000 + i, r1=0, b1=0, r_inc=i < control_incidents, b_inc=False)
        for i in range(n_controls)
    ]
    return pd.DataFrame(rows, columns=list(RECORD_COLUMNS))


def test_summarise_basic_numbers() -> None:
    rows = [
        rec("covariate_shift", 0, 10_000, r1=1, b1=0, r_inc=True, b_inc=False),
        rec("covariate_shift", 0, 10_001, r1=1, b1=1, r_inc=True, b_inc=True),
        rec("unit_break", 2, 10_000, r1=0, b1=0, r_inc=False, b_inc=True, cause=0),
        rec("unit_break", 2, 10_001, r1=1, b1=0, r_inc=True, b_inc=False, cause=1),
        rec(CONTROL, 0, 10_000, r1=0, b1=0, r_inc=True, b_inc=False),
        rec(CONTROL, 0, 10_001, r1=0, b1=0, r_inc=False, b_inc=False),
    ]
    s = summarise(pd.DataFrame(rows))
    assert (s["n_scenarios"], s["n_faults"], s["n_controls"]) == (6, 4, 2)
    assert s["overall"]["ranker_top1"] == pytest.approx(0.75)
    assert s["overall"]["baseline_top1"] == pytest.approx(0.25)
    assert s["by_class"]["covariate_shift"]["ranker_top1"] == 1.0
    assert s["by_class"]["unit_break"]["ranker_top1"] == 0.5
    assert s["by_class"]["unit_break"]["ranker_detection"] == 0.5
    assert s["by_class"]["unit_break"]["baseline_detection"] == 0.5
    assert s["by_class"]["unit_break"]["ranker_cause_top1"] == 0.5
    assert s["by_severity"]["2"]["n"] == 2
    assert s["by_class_severity"]["covariate_shift|0"]["n"] == 2
    # The baseline's implied reading is "covariate shift": only the shift rows can count.
    assert s["overall"]["baseline_cause_top1"] == pytest.approx(0.25)
    assert s["by_class"]["unit_break"]["baseline_cause_top1"] == 0.0
    assert s["controls"]["ranker_false_incidents"] == 1
    assert s["controls"]["ranker_false_incident_rate"] == 0.5
    assert s["fault_seed_range"] == [10_000, 10_001]
    json.dumps(s, allow_nan=False)  # fully JSON-safe


def test_summarise_survives_missing_classes_and_no_controls() -> None:
    s = summarise(records_frame([(1, 0), (0, 0), (1, 1)], 0, 0))
    assert list(s["by_class"]) == ["covariate_shift"]
    assert s["controls"]["n"] == 0
    assert s["controls"]["ranker_false_incident_rate"] is None
    assert s["h2"]["passes"] is None
    assert s["h1"]["passes"] is False  # 3 scenarios cannot clear the interval rule
    json.dumps(s, allow_nan=False)


def test_summarise_with_only_controls() -> None:
    s = summarise(records_frame([], 1, 4))
    assert s["n_faults"] == 0
    assert s["h1"]["passes"] is None
    assert s["h2"]["passes"] is False  # 1/4 = 0.25 > 0.05
    assert s["by_class"] == {}


def test_h1_passes_when_difference_large_and_interval_excludes_zero(tmp_path) -> None:
    frame = records_frame([(1, 0)] * 40, 5, 200)
    s = summarise(frame)
    assert s["h1"]["passes"] is True
    assert s["h2"]["passes"] is True  # 5/200 = 0.025
    write_report(frame, s, tmp_path)
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "H1: PASS" in text
    assert "H2: PASS" in text


def test_h1_fails_when_difference_is_below_015_even_if_interval_excludes_zero() -> None:
    # 40 of 400 paired scenarios favour the ranker: diff 0.10, interval clear of 0.
    s = summarise(records_frame([(1, 0)] * 40 + [(0, 0)] * 360, 0, 0))
    assert s["h1"]["diff"] == pytest.approx(0.10)
    assert s["h1"]["lo"] > 0
    assert s["h1"]["passes"] is False


def test_h1_fails_when_interval_includes_zero_despite_large_difference() -> None:
    # 2 of 4 favour the ranker: diff 0.5, but a resample can hit 0 more than 2.5% of the time.
    s = summarise(records_frame([(1, 0), (1, 0), (0, 0), (0, 0)], 0, 0))
    assert s["h1"]["diff"] == pytest.approx(0.5)
    assert s["h1"]["lo"] == 0.0
    assert s["h1"]["passes"] is False


def test_h1_fails_on_a_tie(tmp_path) -> None:
    frame = records_frame([(1, 1), (0, 0), (1, 1), (0, 0)], 0, 10)
    s = summarise(frame)
    assert s["h1"]["diff"] == 0.0
    write_report(frame, s, tmp_path)
    assert "H1: FAIL" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")


def test_h2_boundary_is_inclusive() -> None:
    at = summarise(records_frame([(1, 0)], 10, 200))  # exactly 0.05
    over = summarise(records_frame([(1, 0)], 11, 200))
    assert at["h2"]["passes"] is True
    assert over["h2"]["passes"] is False


def test_report_files_and_contents(tmp_path) -> None:
    frame = records_frame([(1, 0)] * 6 + [(0, 1)] * 3, 11, 200)
    s = summarise(frame)
    out = tmp_path / "nested" / "out"
    write_report(frame, s, out)

    saved = pd.read_csv(out / "records.csv")
    assert list(saved.columns) == list(RECORD_COLUMNS)
    assert len(saved) == len(frame)
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == s

    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "H2: FAIL" in text  # 11/200 = 0.055
    assert "Scenarios evaluated: 209 (9 fault, 200 control)" in text
    assert "Fault seeds: 10000-10008" in text
    assert "Control seeds: 10000-10199" in text
    assert "11 of 200" in text
    assert "Wilson 95% interval" in text
    assert "### By fault class" in text
    assert "### By severity" in text
    assert "covariate_shift" in text


def test_record_columns_match_the_design() -> None:
    assert RECORD_COLUMNS == (
        "scenario_id", "fault", "severity", "seed", "columns", "ranker_top1", "ranker_rr",
        "ranker_cause_top1", "ranker_incident", "ranker_top_column", "ranker_top_cause",
        "ranker_max_evidence", "baseline_top1", "baseline_rr", "baseline_incident",
        "baseline_top_column", "baseline_max_psi",
    )  # fmt: skip
