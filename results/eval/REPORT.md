# Incident responder evaluation

Windows are random draws from the held-out test split, not time slices. Rules are in `PREREGISTRATION.md`; the verdicts below are computed from them without judgement.

Scenarios evaluated: 440 (240 fault, 200 control). Fault seeds: 0-19. Control seeds: 0-199. Bootstrap: 10000 resamples, seed 0.

## Verdicts

- H1: FAIL. Ranker column top-1 minus baseline column top-1 = 0.050 (95% interval 0.008 to 0.092). Rule: point estimate >= 0.15 and interval lower bound > 0.
- H2: PASS. Ranker false-incident rate on controls = 0.005. Rule: <= 0.05.

## Control windows (no fault)

- Ranker: 1 of 200 flagged as an incident, rate 0.005, Wilson 95% interval 0.001 to 0.028.
- Baseline (max PSI > 0.25): 0 of 200, rate 0.000, Wilson 95% interval 0.000 to 0.019.

## Fault scenarios

| all | n | ranker top-1 | baseline top-1 | ranker MRR | baseline MRR | ranker cause top-1 | baseline cause top-1 (shift reading) | ranker detection | baseline detection |
|---|---|---|---|---|---|---|---|---|---|
| all | 240 | 0.792 | 0.742 | 0.825 | 0.770 | 0.729 | 0.242 | 0.579 | 0.371 |

### By fault class

| class | n | ranker top-1 | baseline top-1 | ranker MRR | baseline MRR | ranker cause top-1 | baseline cause top-1 (shift reading) | ranker detection | baseline detection |
|---|---|---|---|---|---|---|---|---|---|
| covariate_shift | 60 | 0.900 | 0.967 | 0.941 | 0.983 | 0.650 | 0.967 | 0.833 | 0.500 |
| unit_break | 60 | 0.967 | 1.000 | 0.983 | 1.000 | 0.967 | 0.000 | 0.700 | 0.650 |
| null_spike | 60 | 0.967 | 0.667 | 0.983 | 0.697 | 0.967 | 0.000 | 0.667 | 0.333 |
| recode | 60 | 0.333 | 0.333 | 0.394 | 0.400 | 0.333 | 0.000 | 0.117 | 0.000 |

### By severity

| severity | n | ranker top-1 | baseline top-1 | ranker MRR | baseline MRR | ranker cause top-1 | baseline cause top-1 (shift reading) | ranker detection | baseline detection |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 80 | 0.625 | 0.475 | 0.698 | 0.529 | 0.625 | 0.225 | 0.150 | 0.000 |
| 1 | 80 | 0.762 | 0.750 | 0.785 | 0.781 | 0.700 | 0.250 | 0.750 | 0.362 |
| 2 | 80 | 0.988 | 1.000 | 0.994 | 1.000 | 0.863 | 0.250 | 0.838 | 0.750 |

### By class and severity

| class and severity | n | ranker top-1 | baseline top-1 | ranker MRR | baseline MRR | ranker cause top-1 | baseline cause top-1 (shift reading) | ranker detection | baseline detection |
|---|---|---|---|---|---|---|---|---|---|
| covariate_shift|0 | 20 | 0.700 | 0.900 | 0.822 | 0.950 | 0.700 | 0.900 | 0.500 | 0.000 |
| covariate_shift|1 | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 0.500 |
| covariate_shift|2 | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 |
| unit_break|0 | 20 | 0.900 | 1.000 | 0.950 | 1.000 | 0.900 | 0.000 | 0.100 | 0.000 |
| unit_break|1 | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.950 |
| unit_break|2 | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| null_spike|0 | 20 | 0.900 | 0.000 | 0.950 | 0.092 | 0.900 | 0.000 | 0.000 | 0.000 |
| null_spike|1 | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| null_spike|2 | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| recode|0 | 20 | 0.000 | 0.000 | 0.069 | 0.075 | 0.000 | 0.000 | 0.000 | 0.000 |
| recode|1 | 20 | 0.050 | 0.000 | 0.139 | 0.125 | 0.050 | 0.000 | 0.000 | 0.000 |
| recode|2 | 20 | 0.950 | 1.000 | 0.975 | 1.000 | 0.950 | 0.000 | 0.350 | 0.000 |

## Exploratory sensitivity (declared before the run, not a hypothesis test): unit_break without median_log_ratio

Same windows and signals as the ranker above, except that `unit_break` evidence is `out_of_range` only; the variant has its own tau, fitted the same way on the same calibration windows. Declared in `PREREGISTRATION.md` Amendment 2.

| class | n | column top-1 | cause top-1 | detection | ranker cause top-1 | ranker detection |
|---|---|---|---|---|---|---|
| all | 240 | 0.833 | 0.829 | 0.808 | 0.729 | 0.579 |
| covariate_shift | 60 | 0.983 | 0.983 | 0.900 | 0.650 | 0.833 |
| unit_break | 60 | 1.000 | 0.983 | 1.000 | 0.967 | 0.700 |
| null_spike | 60 | 1.000 | 1.000 | 1.000 | 0.967 | 0.667 |
| recode | 60 | 0.350 | 0.350 | 0.333 | 0.333 | 0.117 |

- Control false incidents: 2 of 200, rate 0.010, Wilson 95% interval 0.003 to 0.036.
- Column top-1 minus baseline column top-1 = 0.092 (95% interval 0.058 to 0.129).

This arm does not change the H1/H2 verdicts above.

## Provenance

- Mode: full
- Commit: 4d503e3b0db34bce74a7cbf54f82ce4c2a2dfdb9
- Tree clean: True; allow-dirty used: False
- Data: `data/credit_default.csv.gz`, SHA-256 `c4fdd4c967cba262fd21be076cddece7ca18fd052eb19fb755b4938919c143f9`
- Window size: 1000
- Python: 3.12.13
- Libraries: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, mlops-pipeline 0.1.0
- Started (UTC): 2026-10-23T13:12:47+00:00; finished (UTC): 2026-10-23T13:13:27+00:00
