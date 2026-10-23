# Design and module contracts

Written before any module below existed. Each module is built against this file and
`src/mlops_pipeline/contracts.py`; a contract change is a dated amendment at the bottom.
The evaluation protocol lives in `PREREGISTRATION.md`.

## Flow

```
data cache ─► train (LightGBM + isotonic) ─► MLflow registry, alias "champion"
                                                  │
client ─► POST /predict ─► calibrated p_default + cost flag ─► prediction log (SQLite)
                                                  │
monitor: window of the log vs reference ─► stats ─► ranker ─► incident? ─► POST /incident/webhook
                                                                               │
              collector ─► analyser (stats) ─► ranker ─► postmortem ─► notifier (JSON / Slack payload)
```

Evaluation runs the same analyser and ranker in-process on windows with injected faults.

## Package layout

```
src/mlops_pipeline/
  contracts.py            shared types (ColumnStats, Hypothesis, Ranking, column kinds)
  model/                  vendored from fraud-calibrated @ ebf6770 (do not edit logic)
  train.py                fit + calibrate + metrics; CalibratedModel
  registry.py             MLflow logging, registration, aliases, champion loading, retrain decision
  predlog.py              SQLite prediction log
  serve/settings.py       Settings (pydantic-settings), get_settings()
  serve/routes.py         /health, /model, /predict
  serve/app.py            create_app()
  monitor/stats.py        ReferenceProfile, profile(), column_stats()
  monitor/evidently_report.py  write_drift_report()
  incident/faults.py      Scenario, eval_grid(), dev_grid(), draw_window(), apply_fault()
  incident/ranker.py      NullModel, rank(), psi_baseline()
  incident/analyser.py    analyse()
  incident/collector.py   collect()
  incident/postmortem.py  PostmortemWriter, TemplateWriter, LLMWriter
  incident/notifier.py    Notifier, JsonNotifier, SlackNotifier
  incident/webhook.py     router (POST /incident/webhook), IncidentAlert, Responder protocol
  monitor/run.py          monitor_once()
  evaluate.py             run_evaluation(), summarise(), paired_bootstrap(), write_report()
  cli.py                  train | serve | monitor | inject | eval-responder | retrain
```

## Contracts

### Model and registry (`train.py`, `registry.py`)

```python
@dataclass
class CalibratedModel:            # picklable
    booster: LGBMClassifier
    calibrator: IsotonicRegression
    def predict(self, frame: pd.DataFrame) -> np.ndarray   # calibrated P(default), FEATURE_COLS in any order, NaN allowed

@dataclass(frozen=True)
class TrainResult:
    model: CalibratedModel
    metrics: dict[str, float]     # test_auc, test_brier, test_ece, valid_auc
    n_train: int; n_valid: int; n_test: int

def train(df: pd.DataFrame, *, seed: int = 0) -> TrainResult
    # split(df, seed) -> fit_lightgbm(train) -> fit_isotonic on valid -> metrics on test

MODEL_NAME = "credit-default"; CHAMPION = "champion"

def log_and_register(result: TrainResult, *, tracking_uri: str, model_name: str = MODEL_NAME,
                     data_path: str, alias: str | None = CHAMPION) -> str      # returns version
def load_model(*, tracking_uri: str, model_name: str = MODEL_NAME, alias: str = CHAMPION) -> LoadedModel
@dataclass(frozen=True)
class LoadedModel: model: CalibratedModel; version: str; run_id: str; metrics: dict[str, float]

def retrain(df, *, tracking_uri, model_name=MODEL_NAME, seed=0) -> RetrainDecision
    # challenger registered without alias; both scored on the same test split;
    # promote (move "champion") iff challenger AUC >= champion AUC - 0.005
    #                           and challenger Brier <= champion Brier + 0.002
@dataclass(frozen=True)
class RetrainDecision: promoted: bool; challenger_version: str; champion_version: str | None; reason: str
```

MLflow backend is whatever `tracking_uri` says; default `sqlite:///mlflow.db`. Tests use a
`sqlite:///` file under pytest's tmp path. No MLflow server is needed.

### Prediction log (`predlog.py`)

SQLite table `predictions(id INTEGER PRIMARY KEY, ts TEXT, model_version TEXT, features TEXT,
exposure REAL, p_default REAL, flag INTEGER)`; `ts` is ISO 8601 UTC, `features` is a JSON
object of the 23 feature values (null allowed).

```python
class PredictionLog:
    def __init__(self, path: str | Path)        # creates the table if missing
    def append(self, *, ts: datetime, model_version: str, features: dict[str, float | None],
               exposure: float, p_default: float, flag: bool) -> None
    def window(self, start: datetime, end: datetime) -> pd.DataFrame
        # rows with start <= ts < end; columns FEATURE_COLS + exposure, p_default, flag, ts,
        # model_version; feature columns float with NaN for null; empty frame keeps the columns
```

### Serving (`serve/`)

- `Settings(BaseSettings)`, env prefix `MLOPS_`: `tracking_uri="sqlite:///mlflow.db"`,
  `model_name="credit-default"`, `model_alias="champion"`, `predlog_path="predlog.sqlite"`,
  `reference_path="data/credit_default.csv.gz"`, `postmortem_dir="postmortems"`,
  `slack_webhook_url: str | None = None`. `@lru_cache get_settings()`.
- `create_app(settings: Settings | None = None, *, model: LoadedModel | None = None,
  responder: Responder | None = None) -> FastAPI`. With `model=None` it loads the champion at
  startup (lifespan). Injected `model`/`responder` are for tests.
- `GET /health` → `{"status": "ok"}`. `GET /model` → `{name, alias, version, run_id, metrics}`.
- `POST /predict` body: the 23 feature fields (number or null) + `exposure` (≥ 0) →
  `{"p_default": float in [0,1], "flag": bool, "model_version": str}`. `flag` comes from the
  vendored `costs.decide` with the default `CostModel()`. Every call appends one log row.
- `POST /incident/webhook` (router from `incident/webhook.py`) body `{alert_id, window_start,
  window_end, source: "monitor"}` → `{alert_id, incident, rows, top: [{cause, column,
  evidence}] (up to 3), postmortem_path}`; 404-free: an empty window returns `incident: false`
  and `rows: 0`.

### Monitoring statistics (`monitor/stats.py`)

```python
@dataclass(frozen=True)
class ReferenceProfile: ...     # per column: values (non-null, sorted), min, max, median,
                                # null_rate, code set (discrete/categorical), raw values for PSI
def profile(reference: pd.DataFrame, columns: Sequence[str] = FEATURE_COLS) -> ReferenceProfile
def column_stats(ref: ReferenceProfile, current: pd.DataFrame) -> dict[str, ColumnStats]
    # keyed by column, in FEATURE_COLS order; definitions in contracts.ColumnStats.
    # psi for continuous/discrete MUST equal model.drift.population_stability_index(
    #     reference[col].to_numpy(), current[col].to_numpy()) exactly.
```

`monitor/evidently_report.py`: `write_drift_report(reference, current, out_dir) -> Path`
writes Evidently's data-drift report (HTML + JSON) for a human to read. It feeds nothing else.

### Faults and scenarios (`incident/faults.py`)

```python
SEVERITIES: dict[FaultClass, tuple[float, float, float]]   # from PREREGISTRATION.md
@dataclass(frozen=True)
class Scenario:
    scenario_id: str            # f"{fault}-s{severity}-seed{seed:05d}"
    fault: str                  # a FaultClass or CONTROL
    severity: int               # 0..2; 0 for control
    seed: int
    columns: tuple[str, ...]    # affected columns; () for control
def eval_grid() -> list[Scenario]          # 240 faults (seeds 0-19) + 200 controls (0-199)
def dev_grid(n_seeds: int = 2, n_controls: int = 10) -> list[Scenario]   # seeds from 10_000
def calibration_seeds() -> tuple[range, range]   # (range(0, 200), range(1000, 1200)), class_index 5
def draw_window(pool: pd.DataFrame, scenario: Scenario, reference: pd.DataFrame,
                *, size: int = 1000) -> pd.DataFrame
    # rng = default_rng([seed, class_index, severity]); rows drawn first, then the fault
def calibration_window(pool, seed, *, size=1000) -> pd.DataFrame   # rng([seed, 5, 0]), no fault
def apply_fault(window, fault, severity, columns, rng, reference) -> pd.DataFrame   # returns a copy
```

### Ranker (`incident/ranker.py`)

```python
@dataclass(frozen=True)
class NullModel:
    medians: dict[tuple[str, str], float]      # (signal, column) -> median under no fault
    scales:  dict[tuple[str, str], float]      # max(1.4826 * MAD, 0.01)
    tau: float
def fit_null_model(ref: ReferenceProfile, pool: pd.DataFrame) -> NullModel
    # 200 + 200 calibration windows, per PREREGISTRATION.md
def robust_z(null: NullModel, stats: dict[str, ColumnStats]) -> dict[tuple[str, str], float]
def rank(null: NullModel, stats: dict[str, ColumnStats]) -> Ranking       # method "ranker"
def psi_baseline(stats: dict[str, ColumnStats]) -> Ranking                 # method "psi_baseline"
```

Signals used for z: `psi, ks, null_delta, out_of_range, unseen_share, median_log_ratio`.
`NullModel` saves to and loads from JSON (`to_json(path)`, `from_json(path)`).

### Analyser and collector (`incident/analyser.py`, `incident/collector.py`)

```python
def analyse(ref: ReferenceProfile, null: NullModel, window: pd.DataFrame) -> tuple[dict[str, ColumnStats], Ranking]
def collect(log: PredictionLog, start: datetime, end: datetime) -> pd.DataFrame   # FEATURE_COLS only
```

### Responder, postmortem, notifier (`incident/webhook.py`, `postmortem.py`, `notifier.py`)

```python
class Responder(Protocol):
    def respond(self, window: pd.DataFrame) -> tuple[dict[str, ColumnStats], Ranking]: ...

class PostmortemWriter(Protocol):
    def write(self, alert: IncidentAlert, stats: dict[str, ColumnStats], ranking: Ranking,
              rows: int) -> str        # markdown
class TemplateWriter: ...             # deterministic markdown from the ranking and signals
class LLMWriter:                      # __init__(complete: Callable[[str], str], fallback=TemplateWriter())
    ...                               # prompt built only from the measured ranking; on any error -> fallback

class Notifier(Protocol):
    def notify(self, payload: dict) -> None
class JsonNotifier: ...               # one JSON line to a stream (stdout by default)
class SlackNotifier:                  # __init__(webhook_url, transport: Callable[[str, dict], None] | None)
    def build_payload(self, ...) -> dict      # Slack Block Kit JSON
```

No live Slack call or language-model call happens in tests or CI; both take injected callables.

### Monitor run (`monitor/run.py`)

```python
def monitor_once(log, ref, null, *, start, end, alert: Callable[[IncidentAlert], None],
                 report_dir: Path | None = None) -> Ranking | None
    # collect -> analyse; if ranking.incident: alert(IncidentAlert(...)); optional Evidently report
```

### Evaluation (`evaluate.py`)

```python
def run_evaluation(df: pd.DataFrame, scenarios: list[Scenario]) -> pd.DataFrame
    # split(df, seed=0); profile(train); fit_null_model(valid); per scenario: draw_window(test),
    # column_stats, rank, psi_baseline -> one row per scenario
def summarise(records) -> dict          # every metric in PREREGISTRATION.md, by class and severity
def paired_bootstrap(a, b, *, n=10_000, seed=0) -> tuple[float, float, float]   # mean diff, lo, hi
def wilson(k, n) -> tuple[float, float]
def write_report(records, summary, out_dir) -> None   # records.csv, summary.json, REPORT.md with H1/H2 verdicts
```

Record columns: `scenario_id, fault, severity, seed, columns, ranker_top1, ranker_rr,
ranker_cause_top1, ranker_incident, ranker_top_column, ranker_top_cause, ranker_max_evidence,
baseline_top1, baseline_rr, baseline_incident, baseline_top_column, baseline_max_psi`.

## Amendments

- **2026-10-23 (A1).** `Ranking.hypotheses` is grouped by column: columns ordered by their
  strongest hypothesis, and inside a column the diagnosis order of PREREGISTRATION.md
  Amendment 1. `Ranking.top` is therefore the diagnosed cause of the top column, not always
  the single highest evidence. Code that needs the maximum evidence takes it over all
  hypotheses.
- **2026-10-23.** Additions made while building, none changing a listed signature:
  `create_app(..., extra_routers=())`; app state keys `settings`, `predlog`, `responder`,
  `notifier`, `postmortem_writer`; a custom 422 handler that returns only `loc`, `msg`,
  `type`; `retrain(..., data_path="")`; `train.score_model`; `evaluate.fit_responder` and
  `window_size`/`progress` keywords; `monitor_once(..., reference_frame=None)` for the
  Evidently report; a `prepare` CLI command (writes the split for the DVC pipeline).
- **2026-10-23 (after the independent review).**
  - Settings defaults moved under `mlstore/` (`sqlite:///mlstore/mlflow.db`,
    `mlstore/predlog.sqlite`, `mlstore/postmortems`) and a local `.env` is read, blanks unset.
  - `NullModel` gains `window_size` (calibration size), `reference_sha256` and `tau_no_mlr`
    (the exploratory arm's threshold, PREREGISTRATION.md Amendment 2). A saved null model whose
    hash or size does not match is refitted.
  - Live verdicts need at least `window_size` rows. `monitor_window()` returns a
    `MonitorResult(status, rows, min_rows, ranking)` with status `analysed`, `empty` or
    `insufficient_rows`; `monitor_once()` keeps its signature and returns `None` for the last
    two. The webhook response gains `status` with the same values; only `analysed` responses
    are cached per alert id.
  - Alert ids are built from both window ends in UTC with microseconds.
  - Top-cause lists carry `column_score` (the column's strongest evidence) beside the
    diagnosed cause's evidence (`incident/display.py`).
  - `rank`/`fit_null_model` take an optional `evidence_signals` mapping; the default is the
    frozen table. `rank_no_mlr` is the exploratory arm. Records gain six `sens_*` columns after
    the listed ones.
  - `eval-responder --full` refuses unless window size 1,000, data SHA-256 equal to the vendored
    cache, clean tree, and no earlier `records.csv`; provenance is printed in REPORT.md.
