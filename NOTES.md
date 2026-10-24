# Build Notes — MLOps Pipeline + Incident Responder

Working notes: what broke, what was tried, why X over Y.

---

## Vendored code and data

The model code is copied, not installed, from `fraud-calibrated`
(https://github.com/Prithv122/fraud-calibrated, MIT, same author) at commit `ebf6770`.
Copying survives a later history rewrite of that repo and keeps the model readable here.
Each file carries a two-line header naming its origin; imports were rewritten from
`fraud_calibrated.` to `mlops_pipeline.model.`, nothing else changed.

| Here | Origin path at `ebf6770` | Origin `git hash-object` |
|---|---|---|
| `src/mlops_pipeline/model/data.py` | `src/fraud_calibrated/data.py` | `b4a35c1b2111c5a92b8fad971db1ab8163d5cb7a` |
| `src/mlops_pipeline/model/model.py` | `src/fraud_calibrated/model.py` | `44c3abf048fc60b58efb9aae7ee32c5afd70029b` |
| `src/mlops_pipeline/model/calibration.py` | `src/fraud_calibrated/calibration.py` | `9659302b00d01c71179b7f1c1ea859f3b6ae696b` |
| `src/mlops_pipeline/model/costs.py` | `src/fraud_calibrated/costs.py` | `35bcdfcc237bc63a58eda8bf72507e12b484b53c` |
| `src/mlops_pipeline/model/drift.py` | `src/fraud_calibrated/drift.py` | `e8733fe4712171cc3b17f4bb37b2a4f747cf1d1b` |
| `data/credit_default.csv.gz` | `data/interim/credit_default.csv.gz` | `e2b51e300cc08fc7f0bb219730ecb2189a36eee4` (byte-identical here) |

`tests/fixtures/credit_sample.csv.gz` is a 3,000-row subsample of the cache, stratified on
the target (seed 20260923, default rate 0.2213 vs 0.2212 in the full data), built by
`scripts/make_fixture.py`.

---

## Log

### 2026-10-23 — build day
- **Order of work.** Contracts (`DESIGN.md`, `contracts.py`) and `PREREGISTRATION.md` were
  committed before any module (`37e9524`). Four parts were then built in parallel against them
  (serving + registry, statistics + faults, ranker + evaluation, incident ops layer), each with
  its own tests, and integrated with an end-to-end test.
- **Vendoring checked by result.** Training the vendored model on the full cache through this
  repo's `train` command gives test AUC 0.7750, Brier 0.1339, ECE 0.0237: the same numbers the
  source project publishes.
- **Broke: the class diagnosis.** A development smoke (seeds 10,000–10,003) labelled every
  severity-1/2 unit break and every 100% null spike `covariate_shift`, because PSI and KS rise
  under every fault and hit the 1,000 cap. Fixed by Amendment 1 (specific class first inside a
  column); column results were byte-identical before and after, so H1 cannot have moved.
- **Wrong explanation, caught by review.** I wrote that shifts became `unit_break` through
  out-of-range values. The reviewer measured out-of-range at 0: it is LIMIT_BAL's median moving
  one 10,000 step with a MAD of 0. Erratum added to Amendment 1. Lesson: measure the mechanism
  before writing it down.
- **Heaped medians.** The same effect sets τ (Amendment 2 records it as a limitation and
  declares one exploratory arm without `median_log_ratio`; the primary method is unchanged).
- **Small live windows.** The null model is calibrated on 1,000-row windows; a 100-row window
  raised a false incident 100 times out of 100 in the review. Live verdicts now need at least
  the calibration size.
- **LightGBM threads.** On a loaded machine one fit with default threads took 714 s versus
  0.5 s single-threaded (OpenMP spin-waiting). Tests pin `n_jobs=1`; the vendored parameters
  are unchanged.
- **MLflow quirks.** The default artifact root is `./mlruns` relative to the process, so the
  experiment is created with an explicit location beside the SQLite file; MLflow's uv
  auto-detection copied `uv.lock` into every model version and is turned off.
- **`import conftest` across folders** resolved to the wrong folder's conftest in a full run;
  the ops fakes moved to `tests/ops/ops_fakes.py`.
- **Local CI replay #1** (hosted GitHub Actions unavailable on this account until about
  2026-11-02). Fresh clone of `4792c7a`, fresh `uv sync --all-groups --frozen`, then the
  workflow's steps in order: ruff check clean, ruff format clean (59 files), pytest 247 passed
  and 1 failed (91% coverage, 19 min on a heavily loaded machine), responder smoke on the
  fixture wrote 34 records, `dvc dag` lists the three stages. The failure was a wall-clock
  assertion (one `column_stats` call < 100 ms) that a loaded machine cannot promise; replaced by
  a structural check that the fast path never calls scipy, with the timing check opt-in
  (`MLOPS_TIMING_TESTS=1`, passes on its own). Differences from the hosted runner: Windows 11,
  not Ubuntu; `--basetemp` under the session scratch folder; same Python 3.12 and lockfile.

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| `uv add` the model repo by git SHA | Breaks if that repo's history is rewritten; hides the model code from a reader of this repo |

## Open questions

- [ ]

## Freeze

- **2026-10-23. Code and protocol frozen at `6362d6f`** (owner decision recorded in Amendment
  2: the frozen ranker is primary, S1 is exploratory only). The one-time evaluation runs from
  the commit that adds this entry, which changes nothing but this file:
  `uv run mlops-pipeline eval-responder --full --out results/eval`.

## The run

- **2026-10-23, `4d503e3`, 40 s.** H1 fail (+0.050, 0.008 to 0.092), H2 pass (1/200). Results
  committed as produced (`5fe047d`) before any write-up. Checked after the run, not before:
  18 of 19 ranker wins are severity-0 null spikes; all 7 losses and the one control false alarm
  are a spurious `unit_break` on a heaped-median column (LIMIT_BAL, PAY_AMT4/6, BILL_AMT2), the
  limitation Amendment 2 named. LIMIT_BAL shift diagnosis 1.0 / 0.5 / 0.0 by severity, as the
  erratum predicted. S1 (declared ablation): +0.092, still under the bar, so it would not have
  changed the verdict even if it had been allowed to.
- **Local CI replay #2 on the run commit `4d503e3`** (hosted Actions still unavailable): fresh
  clone, fresh `uv sync --all-groups --frozen`; ruff check and format clean; pytest 248 passed,
  1 skipped (the opt-in timing check), 91% coverage; fixture smoke wrote 34 records; `dvc dag`
  lists the three stages. Then the README steps in that clone with no `.env`: `train` registered
  version 1 (test AUC 0.7750), `serve` answered `/health`, `/model` and the webhook (`empty` for
  a window with no traffic). Same differences from the hosted runner as replay #1.

## Docker smoke check (2026-10-24)

- `docker build -t mlops-pipeline:check .` succeeded on Docker Desktop (Windows 11); image 3.07 GB.
- `docker run --rm -v mlops-check:/app/mlstore mlops-pipeline:check mlops-pipeline train` registered
  `credit-default` version 1 (test AUC 0.775, Brier 0.134, ECE 0.024).
- The image's default command (`serve` on 0.0.0.0:8000, no `--null-model`) started; `/health` returned
  ok, `/model` named version 1 under the `champion` alias, and two `/predict` calls on the first two
  dataset rows returned probabilities and flags; the prediction log in the volume held 2 rows.
- Not exercised in the container: the incident webhook, the responder's calibration, retraining.
  Container, volume and image removed afterwards. README's "not verified" list updated.
