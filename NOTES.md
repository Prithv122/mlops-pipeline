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

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| `uv add` the model repo by git SHA | Breaks if that repo's history is rewritten; hides the model code from a reader of this repo |

## Open questions

- [ ]
