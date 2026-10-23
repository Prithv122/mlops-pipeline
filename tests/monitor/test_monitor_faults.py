"""Fault injection: grids, determinism, and the signature each fault leaves in the statistics.

Every test that draws a window uses development seeds (10,000 and up); the evaluation seeds
are reserved for the one frozen run. The 3,000-row fixture is both the pool and the source of
the reference, so windows here hold 500 rows instead of 1,000.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlops_pipeline.contracts import CONTROL, FAULT_CLASSES
from mlops_pipeline.incident.faults import (
    CLASS_INDEX,
    SEVERITIES,
    apply_fault,
    calibration_seeds,
    calibration_window,
    dev_grid,
    draw_window,
    eval_grid,
    make_scenario,
)
from mlops_pipeline.model.data import BILL_COLS, FEATURE_COLS, split
from mlops_pipeline.monitor.stats import column_stats, profile

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "credit_sample.csv.gz"
SIZE = 500
SEED = 10_000


@pytest.fixture(scope="module")
def pool():
    return pd.read_csv(FIXTURE)


@pytest.fixture(scope="module")
def reference(pool):
    return split(pool, seed=0).train


@pytest.fixture(scope="module")
def ref_profile(reference):
    return profile(reference)


def _unfaulted(pool, scenario, size=SIZE):
    """The rows the scenario's generator draws first, before any fault touches them."""
    rng = np.random.default_rng([scenario.seed, CLASS_INDEX[scenario.fault], scenario.severity])
    return pool.iloc[rng.choice(len(pool), size, replace=False)].reset_index(drop=True)


# ----------------------------------------------------------------------------- grids


def test_eval_grid_counts_and_seed_ranges():
    grid = eval_grid()
    faults = [s for s in grid if s.fault != CONTROL]
    controls = [s for s in grid if s.fault == CONTROL]
    assert len(grid) == 440
    assert len(faults) == 240
    assert len(controls) == 200
    assert {s.seed for s in faults} == set(range(20))
    assert {s.seed for s in controls} == set(range(200))
    for fault in FAULT_CLASSES:
        for severity in range(3):
            assert sum(s.fault == fault and s.severity == severity for s in faults) == 20
    assert all(s.severity == 0 and s.columns == () for s in controls)
    assert len({s.scenario_id for s in grid}) == 440


def test_dev_grid_uses_only_development_seeds():
    grid = dev_grid()
    assert len(grid) == 4 * 3 * 2 + 10
    assert min(s.seed for s in grid) >= 10_000
    bigger = dev_grid(n_seeds=3, n_controls=4)
    assert len(bigger) == 4 * 3 * 3 + 4
    assert min(s.seed for s in bigger) >= 10_000


def test_scenario_ids_and_affected_columns():
    assert make_scenario("unit_break", 2, 7).scenario_id == "unit_break-s2-seed00007"
    assert make_scenario(CONTROL, 0, 199).scenario_id == "control-s0-seed00199"
    assert make_scenario("covariate_shift", 1, 4).columns == ("AGE",)
    assert make_scenario("covariate_shift", 1, 5).columns == ("LIMIT_BAL",)
    assert make_scenario("unit_break", 0, 3).columns == tuple(BILL_COLS)
    assert make_scenario("recode", 0, 3).columns == ("EDUCATION",)
    assert [make_scenario("null_spike", 0, s).columns[0] for s in range(7)] == [
        "PAY_1",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "PAY_1",
    ]


def test_preregistered_severities_and_class_order():
    assert SEVERITIES["covariate_shift"] == (0.10, 0.25, 0.50)
    for fault in ("unit_break", "null_spike", "recode"):
        assert SEVERITIES[fault] == (0.05, 0.25, 1.00)
    assert [CLASS_INDEX[c] for c in (*FAULT_CLASSES, CONTROL)] == [0, 1, 2, 3, 4]


def test_calibration_seed_ranges():
    fit, threshold = calibration_seeds()
    assert (fit.start, fit.stop) == (0, 200)
    assert (threshold.start, threshold.stop) == (1000, 1200)


# ------------------------------------------------------------------ determinism, copies


def test_same_scenario_gives_identical_window(pool, reference):
    sc = make_scenario("unit_break", 1, SEED)
    a = draw_window(pool, sc, reference, size=SIZE)
    b = draw_window(pool, sc, reference, size=SIZE)
    pd.testing.assert_frame_equal(a, b)


def test_different_severity_draws_different_rows(pool, reference):
    low = draw_window(pool, make_scenario("null_spike", 0, SEED), reference, size=SIZE)
    high = draw_window(pool, make_scenario("null_spike", 1, SEED), reference, size=SIZE)
    # compare a column the fault does not touch
    assert not low["LIMIT_BAL"].equals(high["LIMIT_BAL"])


def test_rows_are_drawn_before_the_fault(pool, reference):
    for fault in FAULT_CLASSES:
        sc = make_scenario(fault, 1, SEED)
        window = draw_window(pool, sc, reference, size=SIZE)
        clean = _unfaulted(pool, sc)
        untouched = [c for c in FEATURE_COLS if c not in sc.columns]
        pd.testing.assert_frame_equal(window[untouched], clean[untouched])


def test_window_rows_are_distinct_pool_rows_with_fresh_index(pool, reference):
    pool_with_id = pool.assign(row_id=np.arange(len(pool)))
    window = draw_window(pool_with_id, make_scenario(CONTROL, 0, SEED), reference, size=SIZE)
    assert list(window.index) == list(range(SIZE))
    assert window["row_id"].is_unique
    pd.testing.assert_frame_equal(
        window.drop(columns="row_id"),
        pool.iloc[window["row_id"].to_numpy()].reset_index(drop=True),
    )


def test_pool_and_reference_are_never_mutated(pool, reference):
    pool_before, ref_before = pool.copy(), reference.copy()
    for fault in FAULT_CLASSES:
        for severity in range(3):
            window = draw_window(pool, make_scenario(fault, severity, SEED), reference, size=SIZE)
            window.iloc[:, :] = 0  # writing to the result must not reach the pool either
    pd.testing.assert_frame_equal(pool, pool_before)
    pd.testing.assert_frame_equal(reference, ref_before)


def test_apply_fault_returns_a_copy(pool, reference):
    window = _unfaulted(pool, make_scenario("null_spike", 2, SEED))
    before = window.copy()
    rng = np.random.default_rng(SEED)
    for fault in FAULT_CLASSES:
        cols = make_scenario(fault, 2, SEED).columns
        apply_fault(window, fault, 2, cols, rng, reference)
    pd.testing.assert_frame_equal(window, before)


def test_control_window_is_the_plain_draw(pool, reference):
    sc = make_scenario(CONTROL, 0, SEED)
    pd.testing.assert_frame_equal(draw_window(pool, sc, reference, size=SIZE), _unfaulted(pool, sc))


def test_calibration_window_is_a_no_fault_draw_from_stream_five(pool):
    rng = np.random.default_rng([SEED, 5, 0])
    expected = pool.iloc[rng.choice(len(pool), SIZE, replace=False)].reset_index(drop=True)
    pd.testing.assert_frame_equal(calibration_window(pool, SEED, size=SIZE), expected)
    other = calibration_window(pool, SEED + 1, size=SIZE)
    assert not other["LIMIT_BAL"].equals(expected["LIMIT_BAL"])


# ---------------------------------------------------------------- fault signatures


def test_unit_break_flags_all_six_bill_columns(pool, reference, ref_profile):
    sc = make_scenario("unit_break", 2, SEED)
    window = draw_window(pool, sc, reference, size=SIZE)
    clean = _unfaulted(pool, sc)
    stats = column_stats(ref_profile, window)
    for col in BILL_COLS:
        assert stats[col].out_of_range > 0.7, col
        assert stats[col].median_log_ratio > 1.0, col
        np.testing.assert_array_equal(window[col], clean[col] * 1000)
    for col in FEATURE_COLS:
        if col not in BILL_COLS:
            assert stats[col].out_of_range < 0.02, col


def test_unit_break_changes_the_same_rows_in_every_bill_column(pool, reference):
    sc = make_scenario("unit_break", 1, SEED)
    window = draw_window(pool, sc, reference, size=SIZE)
    clean = _unfaulted(pool, sc)
    # a row whose six bills are all zero is unchanged by x1000, so count rows changed anywhere
    any_changed = np.logical_or.reduce([window[c] != clean[c] for c in BILL_COLS])
    hit = round(0.25 * SIZE)
    assert 0.5 * hit <= any_changed.sum() <= hit


@pytest.mark.parametrize("seed", [SEED, SEED + 1])
@pytest.mark.parametrize("severity", [0, 1, 2])
def test_null_spike_null_delta_equals_share_on_the_right_column_only(
    pool, reference, ref_profile, seed, severity
):
    sc = make_scenario("null_spike", severity, seed)
    window = draw_window(pool, sc, reference, size=SIZE)
    stats = column_stats(ref_profile, window)
    share = SEVERITIES["null_spike"][severity]
    target = f"PAY_{seed % 6 + 1}"
    assert sc.columns == (target,)
    assert stats[target].null_delta == pytest.approx(share)
    assert window[target].dtype == float
    for col in FEATURE_COLS:
        if col != target:
            assert stats[col].null_delta == 0.0, col


def test_recode_moves_education_psi_and_keeps_the_code_set(pool, reference, ref_profile):
    sc = make_scenario("recode", 2, SEED)
    window = draw_window(pool, sc, reference, size=SIZE)
    clean = _unfaulted(pool, sc)
    after = column_stats(ref_profile, window)["EDUCATION"]
    before = column_stats(ref_profile, clean)["EDUCATION"]
    assert after.psi > 0.05
    assert after.psi > 10 * before.psi
    assert after.code_set_changed is False
    assert after.unseen_share == 0.0
    # share 1.0: every 1 became 2 and every 2 became 1, everything else untouched
    swapped = clean["EDUCATION"].map({1: 2, 2: 1}).fillna(clean["EDUCATION"])
    np.testing.assert_array_equal(window["EDUCATION"], swapped)


def test_recode_partial_share_swaps_only_the_chosen_rows(pool, reference):
    sc = make_scenario("recode", 1, SEED)
    window = draw_window(pool, sc, reference, size=SIZE)
    clean = _unfaulted(pool, sc)
    changed = int((window["EDUCATION"] != clean["EDUCATION"]).sum())
    assert 0 < changed <= round(0.25 * SIZE)
    assert set(window["EDUCATION"]) <= set(clean["EDUCATION"])


@pytest.mark.parametrize(
    ("seed", "column", "step"), [(SEED, "AGE", 1), (SEED + 1, "LIMIT_BAL", 10_000)]
)
def test_covariate_shift_moves_the_mean_by_k_sigma_on_the_right_column(
    pool, reference, ref_profile, seed, column, step
):
    sc = make_scenario("covariate_shift", 2, seed)
    window = draw_window(pool, sc, reference, size=SIZE)
    clean = _unfaulted(pool, sc)
    assert sc.columns == (column,)
    k_sigma = SEVERITIES["covariate_shift"][2] * reference[column].std()  # ddof=1
    moved = window[column].mean() - clean[column].mean()
    assert abs(moved - k_sigma) <= step / 2
    assert (window[column] % step == 0).all()
    stats = column_stats(ref_profile, window)
    assert stats[column].psi > 0.5
    for col in FEATURE_COLS:
        if col != column:
            np.testing.assert_array_equal(window[col], clean[col])
            assert stats[col].psi < 0.1, col


def test_no_fault_window_stays_quiet(pool, reference, ref_profile):
    window = draw_window(pool, make_scenario(CONTROL, 0, SEED), reference, size=SIZE)
    for col, s in column_stats(ref_profile, window).items():
        assert s.psi < 0.1, col
        assert s.null_delta == 0.0
        assert s.out_of_range < 0.02, col


def test_unknown_fault_class_is_rejected(pool, reference):
    with pytest.raises(ValueError):
        make_scenario("gremlins", 0, SEED)
    with pytest.raises(ValueError):
        apply_fault(pool.head(10), "gremlins", 0, (), np.random.default_rng(0), reference)
