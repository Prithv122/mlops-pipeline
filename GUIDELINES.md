# MLOps Pipeline + Incident Responder — D7

**Tier:** 3 · **Category:** Data Science · **Wave:** 4

The project's own engineering guide. Keep it short and current.

## What this is

A credit-default model that is trained, versioned, served and monitored, plus an incident
responder that ranks likely causes when drift fires. The responder's ranking is scored on
injected faults with known causes, against a "largest PSI feature" baseline.

## Stack

Python 3.12 · uv · LightGBM + isotonic calibration (vendored model code) · MLflow 3 registry
on a local SQLite backend · FastAPI + uvicorn · SQLite prediction log · Evidently (human-read
report) · DVC pipeline · pytest + ruff · Docker.

## Key documents

- `DESIGN.md` — module contracts. Build against it; change it only by a dated amendment.
- `PREREGISTRATION.md` — the evaluation protocol and decision rules, frozen before any run.
- `NOTES.md` — vendored-file origins, build log, decisions.

## Acceptance criteria

- [ ] `/predict` serves the registered champion; every call lands in the prediction log
- [ ] Retrain registers a challenger and promotes it only by the documented rule
- [ ] Responder evaluated once on the frozen grid; H1 and H2 verdicts reported as they came out
- [ ] Runs from a clean clone using only the documented steps
- [ ] Tests, lint and CI green; README results come from real runs

## Conventions

- `uv` for dependencies; `ruff check` + `ruff format`; `pytest`.
- Vendored files under `src/mlops_pipeline/model/` keep their logic unchanged.
- Evaluation seeds (0–19, controls 0–199) are run once, after the freeze. Development and
  tests use seeds from 10,000.
- Tests never call Slack, a language model or a remote MLflow server; those take injected
  callables or a local SQLite URI.
- No secrets in the repo. `.env.example` lists the variables the code reads.

## Project-specific notes

- Data: `data/credit_default.csv.gz` (UCI, CC BY 4.0), vendored; tests use
  `tests/fixtures/credit_sample.csv.gz` (3,000 rows).
- Local pytest on this machine may need `--basetemp` pointing at a writable folder.
