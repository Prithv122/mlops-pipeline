# Resume Bullets — MLOps Pipeline + Incident Responder

Form: **action → technical specifics → measured outcome.** Every number is in the README and
comes from a command in this repo.

---

## Bullets

- Built an end-to-end MLOps pipeline for a credit-default model (LightGBM + isotonic calibration): MLflow registry with a champion alias and a rule-based retrain gate, FastAPI serving with a SQLite prediction log, drift monitoring, DVC stages and a container image. The served model reproduces the source project's test AUC 0.7750 exactly.
- Designed an incident responder that ranks the likely cause of drift (8 statistics per column, robust z-scores calibrated on 200 no-fault windows). In a pre-registered test on 240 injected faults it named the fault type correctly 73% of the time and caught 5% null spikes that PSI missed (18 of 20 vs 0 of 20), with 1 false alarm in 200 clean windows.
- Reported the pre-registered column-finding test as a fail (+0.050 over "largest PSI" against a +0.15 bar), and traced every loss to one signal on heaped-median columns. The traced cause was declared before the run, and an ablation without that signal gives +0.092 and 83% correct fault types. It was reported as a finding, not promoted.

## Shorter variant (one line)

- MLOps pipeline (MLflow registry, FastAPI serving, drift monitoring, DVC) plus a drift-cause ranker scored once on 240 injected faults: 73% correct fault type and 18/20 null spikes caught vs 0/20 for PSI, at 0.5% false alarms.

## Which roles this supports

- [x] Data Scientist / ML (evaluation design, calibration, robust statistics, honest null result)
- [x] AI Engineer (LLM/NLP/CV): production ML serving and monitoring, not domain
- [x] Data Engineer (prediction log, DVC pipeline, registry, idempotent webhook)
- [ ] Data Analyst / Python Developer

## Keywords this project earns

MLflow model registry and aliases, champion/challenger promotion, FastAPI, pydantic-settings,
SQLite, data drift, PSI, Kolmogorov–Smirnov, robust z-score (median/MAD), Evidently, DVC,
Docker, LightGBM, isotonic calibration, fault injection, pre-registration, paired bootstrap,
Wilson interval, webhooks, idempotency, Slack Block Kit.
