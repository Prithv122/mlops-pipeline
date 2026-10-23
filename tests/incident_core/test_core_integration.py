"""End to end on the committed 3,000-row fixture, with the real statistics and fault code."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("mlops_pipeline.monitor.stats")
pytest.importorskip("mlops_pipeline.incident.faults")

from mlops_pipeline.evaluate import (
    RECORD_COLUMNS,
    fit_responder,
    run_evaluation,
    summarise,
    write_report,
)
from mlops_pipeline.incident.faults import dev_grid

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "credit_sample.csv.gz"
WINDOW = 200  # the fixture's valid and test pools hold 600 rows, below the usual 1,000


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


def test_run_evaluation_on_the_fixture_dev_grid(df: pd.DataFrame, tmp_path) -> None:
    scenarios = dev_grid(n_seeds=1, n_controls=2)
    assert all(s.seed >= 10_000 for s in scenarios)  # never the frozen evaluation seeds

    ticks: list[tuple[int, int]] = []
    records = run_evaluation(
        df, scenarios, window_size=WINDOW, progress=lambda done, total: ticks.append((done, total))
    )
    assert list(records.columns) == list(RECORD_COLUMNS)
    assert list(records["scenario_id"]) == [s.scenario_id for s in scenarios]
    assert ticks[-1] == (len(scenarios), len(scenarios))

    faults = records[records["fault"] != "control"]
    assert faults["ranker_top1"].isin([0.0, 1.0]).all()
    assert faults["baseline_top1"].isin([0.0, 1.0]).all()
    assert faults["ranker_rr"].between(0, 1).all()
    controls = records[records["fault"] == "control"]
    assert len(controls) == 2
    assert controls["ranker_top1"].isna().all()

    sens = [c for c in RECORD_COLUMNS if c.startswith("sens_")]
    assert len(sens) == 6 and list(records.columns[-6:]) == sens
    assert faults["sens_top1"].isin([0.0, 1.0]).all()
    assert controls["sens_top1"].isna().all()
    assert records["sens_incident"].isin([True, False]).all()

    summary = summarise(records)
    assert summary["sensitivity"] is not None
    assert summary["n_scenarios"] == len(scenarios)
    write_report(records, summary, tmp_path)
    assert (tmp_path / "REPORT.md").read_text(encoding="utf-8").startswith("# Incident")


def test_fit_responder_returns_reference_and_null_model(df: pd.DataFrame) -> None:
    ref, null = fit_responder(df, window_size=WINDOW)
    assert ref is not None
    assert null.tau >= 0.0
    assert null.window_size == WINDOW
    assert null.tau_no_mlr is not None and null.tau_no_mlr >= 0.0
    assert len(null.medians) == len(null.scales) == 6 * 23
