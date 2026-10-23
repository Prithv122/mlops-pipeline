"""Evidently data-drift report, written for a person to read.

This is a thin wrapper and nothing else depends on it: the responder's numbers come from
:mod:`mlops_pipeline.monitor.stats`, so a change in Evidently's defaults or API cannot change
an evaluation result. Evidently's own drift tests (Wasserstein, Jensen-Shannon, chi-square,
chosen per column type) are a second opinion on the same window, handy for browsing in a
browser and not used to score anything.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

HTML_NAME = "drift_report.html"
JSON_NAME = "drift_report.json"


def write_drift_report(reference: pd.DataFrame, current: pd.DataFrame, out_dir: str | Path) -> Path:
    """Run the drift preset on ``current`` against ``reference``; write HTML and JSON.

    Both files go into ``out_dir`` (created if missing). Returns the HTML path. Only the columns
    present in both frames are compared, so a window that also carries ``exposure`` or
    ``p_default`` does not break the report.
    """
    shared = [c for c in reference.columns if c in current.columns]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = Report([DataDriftPreset()]).run(current[shared], reference[shared])
    html_path = out / HTML_NAME
    snapshot.save_html(str(html_path))
    snapshot.save_json(str(out / JSON_NAME))
    return html_path
