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

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| `uv add` the model repo by git SHA | Breaks if that repo's history is rewritten; hides the model code from a reader of this repo |

## Open questions

- [ ]
