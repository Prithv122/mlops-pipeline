"""The Evidently wrapper writes a readable HTML report and a JSON twin. Slow; kept tiny."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mlops_pipeline.monitor.evidently_report import write_drift_report


def test_write_drift_report_creates_html_and_json(tmp_path):
    rng = np.random.default_rng(10_000)
    reference = pd.DataFrame(
        {"AGE": rng.integers(20, 70, 120), "LIMIT_BAL": rng.integers(1, 30, 120) * 10_000}
    )
    current = pd.DataFrame(
        {"AGE": rng.integers(30, 80, 80), "LIMIT_BAL": rng.integers(1, 30, 80) * 10_000}
    )
    current["p_default"] = 0.1  # a column the reference lacks is ignored

    html = write_drift_report(reference, current, tmp_path / "report")

    assert html.name == "drift_report.html"
    assert html.read_text(encoding="utf-8").lstrip().startswith("<")
    payload = json.loads((tmp_path / "report" / "drift_report.json").read_text(encoding="utf-8"))
    assert payload
