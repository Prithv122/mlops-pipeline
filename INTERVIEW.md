# Interview Prep — MLOps Pipeline + Incident Responder

**Five questions, five answers.** An unanswered question means this project is not shipped.

---

### Q1. Walk me through the architecture in 90 seconds.

_A:_ The credit-default model from my earlier project is vendored in at a pinned commit:
LightGBM with native categoricals, isotonic calibration on a validation split, and an
expected-cost flag. `train` fits it, logs params and metrics to MLflow on a local SQLite
backend, registers a version and moves the `champion` alias. FastAPI loads the champion at
startup and serves `/predict`, which returns the calibrated probability, the cost flag and the
model version, and writes every request to a SQLite prediction log. A `monitor` job compares a
window of that log with the training data: eight statistics per column (PSI, KS, null-rate
change, out-of-range share and so on). If the responder says incident, it posts an alert to
`/incident/webhook`, which collects the same window, ranks causes, writes a markdown
postmortem and notifies (JSON line or Slack). `retrain` trains a challenger and only moves the
alias if it is no worse than the champion on the same test split. DVC wires
prepare → train → smoke evaluation, and a manual GitHub workflow runs retraining.

### Q2. Why vendor the model code instead of installing your other repo as a dependency?

_A:_ Two reasons. A git-SHA dependency breaks if that repo's history is ever rewritten, and I
have rewritten histories in this portfolio before. And the reader of this repo should see the
model it serves without opening a second project. Vendoring has a cost: drift between copies.
I paid for that with a record in NOTES.md of each file's origin path and `git hash-object`, a
header in each file, and a rule that their logic never changes here. I also checked it by
result: `train` on the full data reproduces the source project's test AUC 0.7750, Brier 0.1339
and ECE 0.0237 exactly.

### Q3. How do you know the cause ranker is any good? Couldn't you just tune it until it wins?

_A:_ That is exactly the risk, so the evaluation was written down and committed before any
ranker code existed: the fault classes, severities, seeds, window size, scoring maths, the
baseline, the metric and the pass rule (ranker beats "largest PSI" by at least 0.15 in column
top-1, with a paired bootstrap interval above zero, and no more than 5% false incidents on
no-fault windows). Development used separate seeds starting at 10,000; the evaluation seeds run
once, from a clean commit, and the CLI refuses to run them twice or on the wrong data. I made
two amendments before that run, both dated and both from development seeds. The first fixed
the fault-class diagnosis; I showed the column results were byte-identical before and after, so
it could not move the headline. The second recorded the development-seed H1 numbers I had
already seen, which were below the bar, and added one exploratory arm. That arm is reported
separately and never decides the verdict. The primary metric compares columns, not causes,
because the PSI baseline never names a fault type. Comparing causes would hand the ranker a
free win.

### Q4. What was the most interesting thing that went wrong?

_A:_ Two things, both about medians. First, my ranker labelled every big ×1000 unit break as
"covariate shift". PSI and KS rise under every fault and hit the score cap, so the generic
explanation won by default. The fix was to diagnose inside a column by specificity: if the
null rate or the out-of-range share crosses the incident line, that names the fault, because
those signals sit near zero when nothing is wrong. Second, I wrote down the wrong reason for a
side effect of that fix, and an independent review caught it. I claimed shifts were being
labelled unit breaks through out-of-range values. They measured those at zero. The real cause
is that LIMIT_BAL lives on a 10,000 grid: its median is the same value in most no-fault windows,
so its MAD is zero and the score scale falls to the floor. One grid step then scores like a
broken feed. The same heaping sets the incident threshold. The lesson I took: measure the
mechanism before you write it down, and robust statistics are not robust to discrete data.

### Q5. What did the evaluation show, and what would you do next?

_A:_ The pre-registered test failed, and I report it as a failure. On 240 injected faults the
ranker found the broken column 79.2% of the time against 74.2% for "largest PSI". That gain of
+0.050 has a 95% interval of 0.008 to 0.092, so it is real but a third of the 0.15 bar I set in
advance. The false-incident test passed: 1 of 200 clean windows. Where it did help is specific.
PSI is blind to a 5% null spike, because nulls fall out of every bin; the baseline found 0 of 20
and the ranker 18. It names the kind of fault correctly 73% of the time, and that is what tells
you whether to retrain or to fix a feed. All seven of its losses, and its one false alarm, came
from the heaped-median signal I had flagged before the run. Dropping that one signal, as an
exploratory ablation declared in advance, gives +0.092 column top-1 and 83% correct fault types.
I did not promote it, because I chose it after seeing development results. Next I would replace
the robust z on discrete columns with something that respects the grid, such as counting grid
steps or a chi-square on code frequencies, and pre-register that as a new study. Then I'd move
from random windows to time slices with late labels joined back, so the responder can see
performance as well as inputs.
