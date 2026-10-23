from __future__ import annotations

import dataclasses

import pytest
from ops_fakes import T0, T1

from mlops_pipeline.contracts import Hypothesis, Ranking
from mlops_pipeline.incident.postmortem import BLIND_SPOT, LLMWriter, TemplateWriter
from mlops_pipeline.incident.webhook import IncidentAlert

ALERT = IncidentAlert(alert_id="drift-test-1", window_start=T0, window_end=T1)


def test_template_contains_top_cause_verdict_and_blind_spot(stats, ranking):
    text = TemplateWriter().write(ALERT, stats, ranking, 1000)
    assert "drift-test-1" in text
    assert "Rows analysed: 1000" in text
    assert "| # | Column | Diagnosed cause | Cause evidence | Column score |" in text
    assert "| 1 | BILL_AMT1 | unit_break | 412.5000 | 412.5000 |" in text
    assert "Verdict: incident" in text
    assert "Threshold: 12.5000" in text
    assert BLIND_SPOT in text
    assert "label" in BLIND_SPOT and "feature values only" in BLIND_SPOT


def test_template_lists_five_hypotheses_and_three_columns(stats, ranking):
    text = TemplateWriter().write(ALERT, stats, ranking, 10)
    assert text.count("\n| ") >= 5
    # one row per column: the second BILL_AMT1 hypothesis does not take a slot
    assert "| 4 | EDUCATION | recode |" in text
    assert "| 5 | AGE | covariate_shift |" in text
    assert "| 6 |" not in text
    # column order is BILL_AMT1, BILL_AMT2, PAY_1 -> raw signals for exactly those three
    for column in ("BILL_AMT1", "BILL_AMT2", "PAY_1"):
        assert f"### {column}" in text
    assert "### EDUCATION" not in text
    assert "| psi | 3.2000 |" in text


def test_table_shows_the_column_score_next_to_a_lower_diagnosed_cause(stats):
    # PAY_1 is ordered by its capped shift score (1000), but diagnosed as the null spike (15).
    ranking = Ranking(
        method="ranker",
        hypotheses=(
            Hypothesis("null_spike", "PAY_1", 15.0),
            Hypothesis("covariate_shift", "PAY_1", 1000.0),
            Hypothesis("unit_break", "BILL_AMT1", 40.0),
        ),
        incident=True,
        threshold=12.5,
    )
    text = TemplateWriter().write(ALERT, stats, ranking, 10)
    assert "| 1 | PAY_1 | null_spike | 15.0000 | 1000.0000 |" in text
    assert "| 2 | BILL_AMT1 | unit_break | 40.0000 | 40.0000 |" in text


def test_template_is_deterministic(stats, ranking):
    writer = TemplateWriter()
    assert writer.write(ALERT, stats, ranking, 10) == writer.write(ALERT, stats, ranking, 10)


def test_template_reports_no_incident(stats, ranking):
    quiet = dataclasses.replace(ranking, incident=False)
    text = TemplateWriter().write(ALERT, stats, quiet, 10)
    assert "Verdict: no incident" in text


def test_llm_writer_uses_the_callable_and_appends_measured_table_verbatim(stats, ranking):
    seen: list[str] = []

    def complete(prompt: str) -> str:
        seen.append(prompt)
        return "  The top hypothesis is a unit break.  "

    text = LLMWriter(complete).write(ALERT, stats, ranking, 1000)
    assert len(seen) == 1
    assert "The top hypothesis is a unit break." in text
    template = TemplateWriter().write(ALERT, stats, ranking, 1000)
    measured = template[template.index("## Verdict") :]
    assert text.endswith(measured)
    assert text.index("The top hypothesis") < text.index("| 1 | BILL_AMT1 | unit_break |")


def test_llm_prompt_has_the_ranking_but_no_raw_rows_and_forbids_extra_causes(stats, ranking):
    seen: list[str] = []
    LLMWriter(lambda p: seen.append(p) or "ok").write(ALERT, stats, ranking, 1000)
    prompt = seen[0]
    assert "BILL_AMT1" in prompt and "unit_break" in prompt and "412.5000" in prompt
    assert "Do not name any cause that is not in the ranking" in prompt
    assert "Rows analysed: 1000" in prompt
    # nothing row-shaped: no feature values, no per-row fields
    for forbidden in ("p_default", "exposure", "model_version"):
        assert forbidden not in prompt


def test_llm_writer_falls_back_on_exception(stats, ranking):
    def boom(prompt: str) -> str:
        raise RuntimeError("model unavailable")

    text = LLMWriter(boom).write(ALERT, stats, ranking, 10)
    assert text == TemplateWriter().write(ALERT, stats, ranking, 10)


@pytest.mark.parametrize("reply", ["", "   \n", None])
def test_llm_writer_falls_back_on_empty_output(stats, ranking, reply):
    text = LLMWriter(lambda p: reply).write(ALERT, stats, ranking, 10)
    assert text == TemplateWriter().write(ALERT, stats, ranking, 10)


def test_llm_writer_uses_the_given_fallback(stats, ranking):
    class Marker:
        def write(self, alert, stats, ranking, rows):
            return "fallback-text"

    def boom(prompt: str) -> str:
        raise ValueError

    assert LLMWriter(boom, fallback=Marker()).write(ALERT, stats, ranking, 10) == "fallback-text"
