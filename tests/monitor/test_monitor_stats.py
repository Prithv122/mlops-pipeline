"""Monitoring statistics: agreement with the vendored PSI and with scipy, and each field by hand."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from mlops_pipeline.contracts import column_kind
from mlops_pipeline.model.data import CATEGORICAL_COLS, FEATURE_COLS, split
from mlops_pipeline.model.drift import population_stability_index
from mlops_pipeline.monitor.stats import column_stats, profile

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "credit_sample.csv.gz"
NON_CATEGORICAL = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]


@pytest.fixture(scope="module")
def parts():
    df = pd.read_csv(FIXTURE)
    s = split(df, seed=0)
    return s.train, s.test


@pytest.fixture(scope="module")
def ref_profile(parts):
    return profile(parts[0])


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def test_keys_follow_feature_cols_order(ref_profile, parts):
    stats = column_stats(ref_profile, parts[1])
    assert list(stats) == FEATURE_COLS
    assert all(stats[c].column == c for c in FEATURE_COLS)


@pytest.mark.parametrize("with_nan", [False, True])
def test_psi_equals_vendored_function_exactly(parts, ref_profile, with_nan):
    ref, cur = parts[0], parts[1].head(600).copy()
    cur = cur.astype(float)
    if with_nan:
        rng = np.random.default_rng(10_000)
        for col in NON_CATEGORICAL:
            cur.loc[rng.choice(len(cur), 90, replace=False), col] = np.nan
    stats = column_stats(ref_profile, cur)
    for col in NON_CATEGORICAL:
        expected = population_stability_index(ref[col].to_numpy(), cur[col].to_numpy())
        assert _same(stats[col].psi, expected), col
        assert math.isfinite(stats[col].psi)


def test_psi_nan_in_current_is_missing_mass():
    """A null falls in no bin, so a fully-null window has every bin fraction clipped to 1e-6."""
    ref = pd.DataFrame({"AGE": np.arange(1, 101)})
    cur = pd.DataFrame({"AGE": np.full(50, np.nan)})
    stats = column_stats(profile(ref, ["AGE"]), cur)
    expected = population_stability_index(ref["AGE"].to_numpy(), cur["AGE"].to_numpy())
    assert stats["AGE"].psi == expected


def test_categorical_psi_hand_computed():
    ref = pd.DataFrame({"EDUCATION": [1] * 50 + [2] * 30 + [3] * 20})
    cur = pd.DataFrame({"EDUCATION": [1] * 40 + [2] * 40 + [3] * 10 + [4] * 10})
    got = column_stats(profile(ref, ["EDUCATION"]), cur)["EDUCATION"].psi
    r = np.array([0.5, 0.3, 0.2, 1e-6])  # codes 1, 2, 3 and "any other value"
    c = np.array([0.4, 0.4, 0.1, 0.1])
    assert got == pytest.approx(float(np.sum((c - r) * np.log(c / r))), rel=1e-12)


def test_categorical_psi_counts_nulls_in_denominator_only():
    ref = pd.DataFrame({"MARRIAGE": [1] * 60 + [2] * 40})
    cur = pd.DataFrame({"MARRIAGE": [1] * 45 + [2] * 35 + [np.nan] * 20})
    got = column_stats(profile(ref, ["MARRIAGE"]), cur)["MARRIAGE"].psi
    r = np.array([0.6, 0.4, 1e-6])
    c = np.array([0.45, 0.35, 1e-6])  # nulls are in no bin; the "other" bin is empty
    assert got == pytest.approx(float(np.sum((c - r) * np.log(c / r))), rel=1e-12)


def test_ks_matches_scipy(parts, ref_profile):
    ref, cur = parts
    stats = column_stats(ref_profile, cur)
    for col in NON_CATEGORICAL:
        expected = ks_2samp(ref[col].to_numpy(float), cur[col].to_numpy(float)).statistic
        assert stats[col].ks == pytest.approx(expected, abs=1e-12), col
    for col in CATEGORICAL_COLS:
        assert stats[col].ks == 0.0


def test_ks_hand_computed():
    ref = pd.DataFrame({"AGE": [1, 2, 3, 4]})
    cur = pd.DataFrame({"AGE": [3, 4, 5, 6]})
    assert column_stats(profile(ref, ["AGE"]), cur)["AGE"].ks == pytest.approx(0.5)


def test_null_delta_against_reference_with_nulls():
    ref = pd.DataFrame({"BILL_AMT1": [np.nan] * 10 + list(range(90))})
    cur = pd.DataFrame({"BILL_AMT1": [np.nan] * 30 + list(range(70))})
    s = column_stats(profile(ref, ["BILL_AMT1"]), cur)["BILL_AMT1"]
    assert s.null_delta == pytest.approx(0.30 - 0.10)


def test_out_of_range_share_ignores_nulls():
    ref = pd.DataFrame({"BILL_AMT1": np.arange(100)})  # range 0..99
    cur = pd.DataFrame({"BILL_AMT1": [-1.0, 50.0, 100.0, 200.0, np.nan, np.nan]})
    s = column_stats(profile(ref, ["BILL_AMT1"]), cur)["BILL_AMT1"]
    assert s.out_of_range == pytest.approx(3 / 4)
    assert s.null_delta == pytest.approx(2 / 6)


def test_unseen_share_and_code_set_for_discrete():
    ref = pd.DataFrame({"PAY_1": [-1, 0, 1, 2] * 25})
    cur = pd.DataFrame({"PAY_1": [0.0, 0.0, 3.0, 9.0, np.nan]})
    s = column_stats(profile(ref, ["PAY_1"]), cur)["PAY_1"]
    assert s.unseen_share == pytest.approx(0.5)
    assert s.code_set_changed is True


def test_code_set_unchanged_when_same_codes_appear():
    ref = pd.DataFrame({"EDUCATION": [1, 2, 3, 4] * 25})
    cur = pd.DataFrame({"EDUCATION": [4, 3, 2, 1, 1, 2]})
    s = column_stats(profile(ref, ["EDUCATION"]), cur)["EDUCATION"]
    assert s.code_set_changed is False
    assert s.unseen_share == 0.0


def test_a_missing_reference_code_counts_as_a_changed_set():
    ref = pd.DataFrame({"EDUCATION": [1, 2, 3, 4] * 25})
    cur = pd.DataFrame({"EDUCATION": [1, 2, 3]})
    assert column_stats(profile(ref, ["EDUCATION"]), cur)["EDUCATION"].code_set_changed is True


def test_continuous_column_has_no_unseen_share_or_code_set():
    ref = pd.DataFrame({"AGE": np.arange(20, 70)})
    cur = pd.DataFrame({"AGE": [200.0, 300.0]})
    s = column_stats(profile(ref, ["AGE"]), cur)["AGE"]
    assert s.unseen_share == 0.0
    assert s.code_set_changed is False
    assert s.out_of_range == 1.0


def test_median_log_ratio():
    ref = pd.DataFrame({"BILL_AMT1": [99.0] * 11})  # |99| + 1 = 100
    cur = pd.DataFrame({"BILL_AMT1": [999.0] * 5 + [np.nan]})  # |999| + 1 = 1000
    s = column_stats(profile(ref, ["BILL_AMT1"]), cur)["BILL_AMT1"]
    assert s.median_log_ratio == pytest.approx(1.0)
    # symmetric in direction: a drop by the same factor gives the same value
    s2 = column_stats(profile(cur.dropna(), ["BILL_AMT1"]), ref)["BILL_AMT1"]
    assert s2.median_log_ratio == pytest.approx(1.0)


def test_median_uses_midpoint_for_even_counts():
    ref = pd.DataFrame({"BILL_AMT1": [0.0] * 4})
    cur = pd.DataFrame({"BILL_AMT1": [9.0, 19.0, 100.0, 200.0]})  # median 59.5 -> |.|+1 = 60.5
    s = column_stats(profile(ref, ["BILL_AMT1"]), cur)["BILL_AMT1"]
    assert s.median_log_ratio == pytest.approx(math.log10(60.5))


def test_became_constant():
    ref = pd.DataFrame({"LIMIT_BAL": [10_000, 50_000, 90_000, 150_000] * 5})
    varied = column_stats(
        profile(ref, ["LIMIT_BAL"]), pd.DataFrame({"LIMIT_BAL": [10_000, 50_000]})
    )
    constant = column_stats(profile(ref, ["LIMIT_BAL"]), pd.DataFrame({"LIMIT_BAL": [50_000] * 8}))
    assert varied["LIMIT_BAL"].became_constant is False
    assert constant["LIMIT_BAL"].became_constant is True
    # a reference that was constant already is not a change
    flat = pd.DataFrame({"LIMIT_BAL": [50_000] * 20})
    same = column_stats(profile(flat, ["LIMIT_BAL"]), pd.DataFrame({"LIMIT_BAL": [50_000] * 8}))
    assert same["LIMIT_BAL"].became_constant is False


def test_all_null_window_is_defined_everywhere(parts, ref_profile):
    cur = pd.DataFrame({c: np.full(40, np.nan) for c in FEATURE_COLS})
    for s in column_stats(ref_profile, cur).values():
        assert s.null_delta == pytest.approx(1.0)
        assert s.ks == 0.0
        assert s.out_of_range == 0.0
        assert s.unseen_share == 0.0
        assert s.median_log_ratio == 0.0
        assert s.became_constant is False
        assert math.isfinite(s.psi)


def test_empty_window_does_not_raise(parts, ref_profile):
    stats = column_stats(ref_profile, parts[1].iloc[0:0])
    assert all(math.isfinite(s.psi) and s.null_delta == 0.0 for s in stats.values())


def test_nullable_integer_columns_are_accepted(parts, ref_profile):
    cur = parts[1].head(200).astype("Int64")
    cur.loc[cur.index[:20], "AGE"] = pd.NA
    s = column_stats(ref_profile, cur)["AGE"]
    assert s.null_delta == pytest.approx(0.1)


def test_no_fault_window_gives_small_values(parts, ref_profile):
    stats = column_stats(ref_profile, parts[1])
    for col, s in stats.items():
        assert s.psi < 0.1, col
        assert s.ks < 0.15, col
        assert s.null_delta == 0.0
        assert s.out_of_range < 0.02, col
        assert s.median_log_ratio < 0.15, col
        assert s.unseen_share < 0.02, col


def test_column_kinds_cover_all_features():
    assert {column_kind(c) for c in FEATURE_COLS} == {"continuous", "discrete", "categorical"}


def test_column_stats_never_calls_scipy_per_column():
    """The fast path: KS comes from the presorted reference, not ``scipy.stats.ks_2samp``,
    which re-sorts 18,000 values per column and made one window cost ~400 ms. Checked
    structurally so a busy machine cannot fail it; the timing check below is opt-in."""
    from mlops_pipeline.monitor import stats

    assert "scipy" not in vars(stats)
    assert not any(getattr(v, "__module__", "").startswith("scipy") for v in vars(stats).values())


@pytest.mark.skipif(
    os.environ.get("MLOPS_TIMING_TESTS") != "1",
    reason="wall-clock check; opt in on a quiet machine",
)
def test_column_stats_speed_on_a_full_size_window(parts):
    """The evaluation makes about 840 calls; one call must stay far below 100 ms."""
    rng = np.random.default_rng(10_000)
    pool = pd.read_csv(FIXTURE)
    ref = pool.iloc[rng.integers(0, len(pool), 18_000)].reset_index(drop=True)
    cur = pool.iloc[rng.integers(0, len(pool), 1_000)].reset_index(drop=True)
    prof = profile(ref)
    column_stats(prof, cur)  # warm up
    start = time.perf_counter()
    for _ in range(5):
        column_stats(prof, cur)
    assert (time.perf_counter() - start) / 5 < 0.1
