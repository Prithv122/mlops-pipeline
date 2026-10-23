"""Evaluation of the incident responder, exactly as written in PREREGISTRATION.md.

``run_evaluation`` scores the ranker and the PSI baseline on the same windows,
``summarise`` computes every pre-registered metric, and ``write_report`` states the H1 and
H2 verdicts mechanically from the frozen rules. Nothing here tunes anything.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mlops_pipeline.contracts import CONTROL, FAULT_CLASSES, Ranking
from mlops_pipeline.model.data import Split, split

if TYPE_CHECKING:
    from mlops_pipeline.incident.faults import Scenario
    from mlops_pipeline.incident.ranker import NullModel
    from mlops_pipeline.monitor.stats import ReferenceProfile

Progress = Callable[[int, int], None]

RECORD_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "fault",
    "severity",
    "seed",
    "columns",
    "ranker_top1",
    "ranker_rr",
    "ranker_cause_top1",
    "ranker_incident",
    "ranker_top_column",
    "ranker_top_cause",
    "ranker_max_evidence",
    "baseline_top1",
    "baseline_rr",
    "baseline_incident",
    "baseline_top_column",
    "baseline_max_psi",
    # Exploratory arm S1 (PREREGISTRATION.md, Amendment 2), appended so nothing above moves.
    "sens_top1",
    "sens_rr",
    "sens_cause_top1",
    "sens_incident",
    "sens_top_column",
    "sens_top_cause",
)

H1_MIN_DIFF = 0.15
H2_MAX_RATE = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
_Z95 = 1.959963984540054


def stderr_progress(label: str, *, steps: int = 10) -> Progress:
    """A progress callback that prints to stderr about ``steps`` times per run."""

    def report(done: int, total: int) -> None:
        every = max(1, total // steps)
        if done == total or done % every == 0:
            print(f"{label}: {done}/{total}", file=sys.stderr, flush=True)

    return report


# --------------------------------------------------------------------------- scoring


def score_ranking(ranking: Ranking, fault: str, columns: Sequence[str]) -> dict[str, float]:
    """Column top-1, reciprocal rank and cause top-1 of one ranking for one scenario.

    Controls have no cause, so all three are NaN there. Column scores treat any affected
    column as a hit (the BILL fault affects six). Cause top-1 needs the right class on an
    affected column.
    """
    if fault == CONTROL or not columns:
        return {"top1": math.nan, "rr": math.nan, "cause_top1": math.nan}
    affected = set(columns)
    order = ranking.column_order()
    rr = 0.0
    for position, column in enumerate(order, start=1):
        if column in affected:
            rr = 1.0 / position
            break
    top = ranking.top
    return {
        "top1": float(bool(order) and order[0] in affected),
        "rr": rr,
        "cause_top1": float(top is not None and top.cause == fault and top.column in affected),
    }


def _max_evidence(ranking: Ranking) -> float:
    values = [h.evidence for h in ranking.hypotheses if not math.isnan(h.evidence)]
    return max(values) if values else math.nan


def _record(
    scenario: Scenario, ranker: Ranking, baseline: Ranking, variant: Ranking
) -> dict[str, Any]:
    r = score_ranking(ranker, scenario.fault, scenario.columns)
    b = score_ranking(baseline, scenario.fault, scenario.columns)
    v = score_ranking(variant, scenario.fault, scenario.columns)
    r_top, b_top, v_top = ranker.top, baseline.top, variant.top
    return {
        "scenario_id": scenario.scenario_id,
        "fault": scenario.fault,
        "severity": scenario.severity,
        "seed": scenario.seed,
        "columns": ";".join(scenario.columns),
        "ranker_top1": r["top1"],
        "ranker_rr": r["rr"],
        "ranker_cause_top1": r["cause_top1"],
        "ranker_incident": ranker.incident,
        "ranker_top_column": r_top.column if r_top else "",
        "ranker_top_cause": r_top.cause if r_top else "",
        "ranker_max_evidence": _max_evidence(ranker),
        "baseline_top1": b["top1"],
        "baseline_rr": b["rr"],
        "baseline_incident": baseline.incident,
        "baseline_top_column": b_top.column if b_top else "",
        "baseline_max_psi": _max_evidence(baseline),
        "sens_top1": v["top1"],
        "sens_rr": v["rr"],
        "sens_cause_top1": v["cause_top1"],
        "sens_incident": variant.incident,
        "sens_top_column": v_top.column if v_top else "",
        "sens_top_cause": v_top.cause if v_top else "",
    }


# ------------------------------------------------------------------------ evaluation


def _fit_on_split(
    parts: Split, window_size: int, progress: Progress | None
) -> tuple[ReferenceProfile, NullModel]:
    from mlops_pipeline.incident.ranker import fit_null_model
    from mlops_pipeline.monitor.stats import profile

    ref = profile(parts.train)
    return ref, fit_null_model(ref, parts.valid, window_size=window_size, progress=progress)


def fit_responder(
    df: pd.DataFrame, *, window_size: int = 1000, progress: Progress | None = None
) -> tuple[ReferenceProfile, NullModel]:
    """Reference profile (train split) and null model (valid split) for ``split(df, seed=0)``.

    The CLI and the API call this so they use the same reference and calibration as the
    evaluation.
    """
    return _fit_on_split(split(df, seed=0), window_size, progress)


def run_evaluation(
    df: pd.DataFrame,
    scenarios: Sequence[Scenario],
    *,
    window_size: int = 1000,
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Score the ranker, the baseline and the exploratory variant on one window per scenario.

    The variant (``sens_*`` columns) reads the same statistics; it never feeds H1 or H2.
    """
    from mlops_pipeline.incident.analyser import analyse
    from mlops_pipeline.incident.faults import draw_window
    from mlops_pipeline.incident.ranker import psi_baseline, rank_no_mlr

    parts = split(df, seed=0)
    ref, null = _fit_on_split(parts, window_size, progress)
    rows = []
    for i, scenario in enumerate(scenarios, start=1):
        window = draw_window(parts.test, scenario, parts.train, size=window_size)
        stats, ranker = analyse(ref, null, window)
        rows.append(_record(scenario, ranker, psi_baseline(stats), rank_no_mlr(null, stats)))
        if progress is not None:
            progress(i, len(scenarios))
    return pd.DataFrame(rows, columns=list(RECORD_COLUMNS))


# --------------------------------------------------------------------------- statistics


def paired_bootstrap(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    n: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Mean of a - b with a percentile 95% interval from resampling scenarios with replacement.

    Returns ``(mean_difference, lo, hi)``; the point estimate is the plain mean, not the
    bootstrap mean.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size == 0:
        return math.nan, math.nan, math.nan
    idx = np.random.default_rng(seed).integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% score interval for k successes in n trials."""
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    z2 = _Z95**2
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = _Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    # The score interval touches the boundary exactly when k is 0 or n; avoid float dust there.
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else math.nan


def _slice_metrics(faults: pd.DataFrame) -> dict[str, Any]:
    is_shift = faults["fault"] == "covariate_shift"
    return {
        "n": len(faults),
        "ranker_top1": _mean(faults["ranker_top1"]),
        "baseline_top1": _mean(faults["baseline_top1"]),
        "ranker_mrr": _mean(faults["ranker_rr"]),
        "baseline_mrr": _mean(faults["baseline_rr"]),
        "ranker_cause_top1": _mean(faults["ranker_cause_top1"]),
        # The baseline names no class; its implied reading is "covariate shift".
        "baseline_cause_top1": _mean(faults["baseline_top1"] * is_shift),
        "ranker_detection": _mean(faults["ranker_incident"].astype(float)),
        "baseline_detection": _mean(faults["baseline_incident"].astype(float)),
    }


def _seed_range(frame: pd.DataFrame) -> list[int] | None:
    return [int(frame["seed"].min()), int(frame["seed"].max())] if len(frame) else None


def _clean(value: Any) -> Any:
    """Replace NaN with None so the summary is valid JSON."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _sensitivity(df: pd.DataFrame) -> dict[str, Any] | None:
    """Exploratory arm S1: unit_break evidence without median_log_ratio.

    ``None`` for records written before the arm existed. Nothing here enters H1 or H2.
    """
    if "sens_top1" not in df.columns or df["sens_incident"].isna().any():
        return None
    faults = df[df["fault"] != CONTROL]
    controls = df[df["fault"] == CONTROL]

    def slice_metrics(part: pd.DataFrame) -> dict[str, Any]:
        return {
            "n": len(part),
            "sens_top1": _mean(part["sens_top1"]),
            "sens_mrr": _mean(part["sens_rr"]),
            "sens_cause_top1": _mean(part["sens_cause_top1"]),
            "sens_detection": _mean(part["sens_incident"].astype(float)),
            # The frozen ranker on the same rows, for reading the arm against it.
            "ranker_top1": _mean(part["ranker_top1"]),
            "ranker_cause_top1": _mean(part["ranker_cause_top1"]),
            "ranker_detection": _mean(part["ranker_incident"].astype(float)),
        }

    n_controls = len(controls)
    k = int(controls["sens_incident"].astype(bool).sum())
    lo_w, hi_w = wilson(k, n_controls)
    diff, lo, hi = paired_bootstrap(faults["sens_top1"], faults["baseline_top1"])
    return {
        "overall": slice_metrics(faults),
        "by_class": {
            c: slice_metrics(faults[faults["fault"] == c])
            for c in FAULT_CLASSES
            if (faults["fault"] == c).any()
        },
        "controls": {
            "n": n_controls,
            "false_incidents": k,
            "false_incident_rate": k / n_controls if n_controls else math.nan,
            "wilson_lo": lo_w,
            "wilson_hi": hi_w,
        },
        "vs_baseline": {"mean_diff": diff, "lo": lo, "hi": hi},
    }


def summarise(records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every pre-registered metric, overall and by class, severity and class x severity."""
    df = pd.DataFrame(records)
    faults = df[df["fault"] != CONTROL]
    controls = df[df["fault"] == CONTROL]

    by_class = {
        c: _slice_metrics(faults[faults["fault"] == c])
        for c in FAULT_CLASSES
        if (faults["fault"] == c).any()
    }
    by_severity = {
        str(s): _slice_metrics(faults[faults["severity"] == s])
        for s in sorted(faults["severity"].unique())
    }
    by_class_severity = {
        f"{c}|{s}": _slice_metrics(faults[(faults["fault"] == c) & (faults["severity"] == s)])
        for c in FAULT_CLASSES
        for s in sorted(faults["severity"].unique())
        if ((faults["fault"] == c) & (faults["severity"] == s)).any()
    }

    n_controls = len(controls)
    k_ranker = int(controls["ranker_incident"].astype(bool).sum())
    k_base = int(controls["baseline_incident"].astype(bool).sum())
    r_lo, r_hi = wilson(k_ranker, n_controls)
    b_lo, b_hi = wilson(k_base, n_controls)
    control_summary = {
        "n": n_controls,
        "ranker_false_incidents": k_ranker,
        "ranker_false_incident_rate": k_ranker / n_controls if n_controls else math.nan,
        "ranker_wilson_lo": r_lo,
        "ranker_wilson_hi": r_hi,
        "baseline_false_incidents": k_base,
        "baseline_false_incident_rate": k_base / n_controls if n_controls else math.nan,
        "baseline_wilson_lo": b_lo,
        "baseline_wilson_hi": b_hi,
    }

    diff, lo, hi = paired_bootstrap(faults["ranker_top1"], faults["baseline_top1"])
    bootstrap = {
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "mean_diff": diff,
        "lo": lo,
        "hi": hi,
    }
    h1_evaluable = len(faults) > 0
    h1 = {
        "min_diff": H1_MIN_DIFF,
        "diff": diff,
        "lo": lo,
        "hi": hi,
        "passes": (diff >= H1_MIN_DIFF and lo > 0) if h1_evaluable else None,
    }
    rate = control_summary["ranker_false_incident_rate"]
    h2 = {
        "max_rate": H2_MAX_RATE,
        "rate": rate,
        "passes": (rate <= H2_MAX_RATE) if n_controls else None,
    }

    summary = {
        "n_scenarios": len(df),
        "n_faults": len(faults),
        "n_controls": n_controls,
        "fault_seed_range": _seed_range(faults),
        "control_seed_range": _seed_range(controls),
        "overall": _slice_metrics(faults),
        "by_class": by_class,
        "by_severity": by_severity,
        "by_class_severity": by_class_severity,
        "controls": control_summary,
        "bootstrap": bootstrap,
        "h1": h1,
        "h2": h2,
        "sensitivity": _sensitivity(df),
    }
    return _clean(summary)


# ------------------------------------------------------------------------------ report


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _verdict(passes: bool | None) -> str:
    return "NOT EVALUABLE" if passes is None else ("PASS" if passes else "FAIL")


def _range_text(r: list[int] | None) -> str:
    return "none" if r is None else f"{r[0]}-{r[1]}"


_TABLE_HEADER = (
    "| {label} | n | ranker top-1 | baseline top-1 | ranker MRR | baseline MRR "
    "| ranker cause top-1 | baseline cause top-1 (shift reading) "
    "| ranker detection | baseline detection |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def _table(label: str, rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [_TABLE_HEADER.format(label=label)]
    for name, m in rows.items():
        cells = [
            name,
            str(m["n"]),
            _fmt(m["ranker_top1"]),
            _fmt(m["baseline_top1"]),
            _fmt(m["ranker_mrr"]),
            _fmt(m["baseline_mrr"]),
            _fmt(m["ranker_cause_top1"]),
            _fmt(m["baseline_cause_top1"]),
            _fmt(m["ranker_detection"]),
            _fmt(m["baseline_detection"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


_SENS_TITLE = (
    "Exploratory sensitivity (declared before the run, not a hypothesis test): "
    "unit_break without median_log_ratio"
)


def _sensitivity_lines(sens: Mapping[str, Any]) -> list[str]:
    ctl, vs = sens["controls"], sens["vs_baseline"]
    lines = [
        f"## {_SENS_TITLE}",
        "",
        "Same windows and signals as the ranker above, except that `unit_break` evidence is "
        "`out_of_range` only; the variant has its own tau, fitted the same way on the same "
        "calibration windows. Declared in `PREREGISTRATION.md` Amendment 2.",
        "",
        "| class | n | column top-1 | cause top-1 | detection "
        "| ranker cause top-1 | ranker detection |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = {"all": sens["overall"], **sens["by_class"]}
    for name, m in rows.items():
        cells = [
            name,
            str(m["n"]),
            _fmt(m["sens_top1"]),
            _fmt(m["sens_cause_top1"]),
            _fmt(m["sens_detection"]),
            _fmt(m["ranker_cause_top1"]),
            _fmt(m["ranker_detection"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        f"- Control false incidents: {ctl['false_incidents']} of {ctl['n']}, rate "
        f"{_fmt(ctl['false_incident_rate'])}, Wilson 95% interval {_fmt(ctl['wilson_lo'])} "
        f"to {_fmt(ctl['wilson_hi'])}.",
        f"- Column top-1 minus baseline column top-1 = {_fmt(vs['mean_diff'])} "
        f"(95% interval {_fmt(vs['lo'])} to {_fmt(vs['hi'])}).",
        "",
        "This arm does not change the H1/H2 verdicts above.",
        "",
    ]
    return lines


def _provenance_lines(prov: Mapping[str, Any]) -> list[str]:
    versions = prov.get("versions", {})
    lines = [
        "## Provenance",
        "",
        f"- Mode: {prov.get('mode')}",
        f"- Commit: {prov.get('commit') or 'none (not a git checkout)'}",
        f"- Tree clean: {prov.get('tree_clean')}; allow-dirty used: {prov.get('allow_dirty')}",
        f"- Data: `{prov.get('data')}`, SHA-256 `{prov.get('data_sha256')}`",
        f"- Window size: {prov.get('window_size')}",
        f"- Python: {prov.get('python')}",
        "- Libraries: " + ", ".join(f"{k} {v}" for k, v in versions.items()),
        f"- Started (UTC): {prov.get('started_utc')}; finished (UTC): {prov.get('finished_utc')}",
    ]
    if prov.get("allow_dirty") and not prov.get("tree_clean"):
        lines += [
            "",
            "WARNING: tree was not clean when this ran (the dirty-tree override was used), so "
            "the commit above does not fully describe the code that produced these numbers.",
        ]
    return [*lines, ""]


def _report_text(summary: Mapping[str, Any]) -> str:
    h1, h2, ctl = summary["h1"], summary["h2"], summary["controls"]
    lines = [
        "# Incident responder evaluation",
        "",
        "Windows are random draws from the held-out test split, not time slices. Rules are in "
        "`PREREGISTRATION.md`; the verdicts below are computed from them without judgement.",
        "",
        f"Scenarios evaluated: {summary['n_scenarios']} ({summary['n_faults']} fault, "
        f"{summary['n_controls']} control). Fault seeds: "
        f"{_range_text(summary['fault_seed_range'])}. Control seeds: "
        f"{_range_text(summary['control_seed_range'])}. Bootstrap: "
        f"{summary['bootstrap']['n_resamples']} resamples, seed {summary['bootstrap']['seed']}.",
        "",
        "## Verdicts",
        "",
        f"- H1: {_verdict(h1['passes'])}. Ranker column top-1 minus baseline column top-1 = "
        f"{_fmt(h1['diff'])} (95% interval {_fmt(h1['lo'])} to {_fmt(h1['hi'])}). "
        f"Rule: point estimate >= {h1['min_diff']:.2f} and interval lower bound > 0.",
        f"- H2: {_verdict(h2['passes'])}. Ranker false-incident rate on controls = "
        f"{_fmt(h2['rate'])}. Rule: <= {h2['max_rate']:.2f}.",
        "",
        "## Control windows (no fault)",
        "",
        f"- Ranker: {ctl['ranker_false_incidents']} of {ctl['n']} flagged as an incident, "
        f"rate {_fmt(ctl['ranker_false_incident_rate'])}, Wilson 95% interval "
        f"{_fmt(ctl['ranker_wilson_lo'])} to {_fmt(ctl['ranker_wilson_hi'])}.",
        f"- Baseline (max PSI > 0.25): {ctl['baseline_false_incidents']} of {ctl['n']}, "
        f"rate {_fmt(ctl['baseline_false_incident_rate'])}, Wilson 95% interval "
        f"{_fmt(ctl['baseline_wilson_lo'])} to {_fmt(ctl['baseline_wilson_hi'])}.",
        "",
        "## Fault scenarios",
        "",
        *_table("all", {"all": summary["overall"]}),
        "",
        "### By fault class",
        "",
        *_table("class", summary["by_class"]),
        "",
        "### By severity",
        "",
        *_table("severity", summary["by_severity"]),
        "",
        "### By class and severity",
        "",
        *_table("class and severity", summary["by_class_severity"]),
        "",
    ]
    if summary.get("sensitivity"):
        lines += _sensitivity_lines(summary["sensitivity"])
    if summary.get("provenance"):
        lines += _provenance_lines(summary["provenance"])
    return "\n".join(lines)


def write_report(records: pd.DataFrame, summary: Mapping[str, Any], out_dir: str | Path) -> None:
    """Write ``records.csv``, ``summary.json`` and ``REPORT.md`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "records.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(_report_text(summary), encoding="utf-8")
