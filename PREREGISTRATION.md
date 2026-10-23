# Pre-registration — incident responder evaluation

Written and committed before any responder code existed and before any evaluation seed was
run. Changes after this commit are dated amendments at the bottom; the original text stays.

## Question

When one fault with a known cause is injected into a window of scoring traffic, how often does
the responder's cause ranker point at the right column first, compared with the naive "largest
PSI feature" baseline? Measured over seeded scenarios covering four fault classes, three
severities each, plus no-fault control windows.

## Data and windows

- Data: UCI Default of Credit Card Clients, the vendored cleaned cache (30,000 rows).
- Split: `split(df, seed=0)` from the vendored `data.py` (60/20/20 stratified).
  - **Reference** = the train split (18,000 rows). The responder compares every window with it.
  - **Calibration pool** = the valid split (6,000 rows). Used only to learn what no-fault
    windows look like (below). Never used for scenarios.
  - **Scenario pool** = the test split (6,000 rows). Every evaluation window is drawn from it.
- Window: **1,000 rows** drawn without replacement from the pool, then the fault (if any) is
  applied to that window only.
- The rows have no real timestamps, so a "window" is a random draw, not a time slice. The
  README says so next to every number.

## Faults (cause label = class + affected columns)

Each scenario has a fault class, a severity index 0/1/2 and a seed. All randomness for one
scenario comes from `numpy.random.default_rng([seed, class_index, severity_index])`, with
class_index 0..3 in the order below and 4 for control (severity_index 0). The same generator
draws the window rows first, then the affected rows.

| class | affected columns | severity 0 / 1 / 2 | how |
|---|---|---|---|
| `covariate_shift` | `AGE` for even seeds, `LIMIT_BAL` for odd seeds | k = 0.10 / 0.25 / 0.50 | every row: x + k·σ_ref, rounded to the column's grid (AGE: 1, LIMIT_BAL: 10,000) |
| `unit_break` | `BILL_AMT1` … `BILL_AMT6` (all six) | share = 0.05 / 0.25 / 1.00 | on that share of rows, all six columns ×1000 (an upstream partition switching units) |
| `null_spike` | `PAY_{(seed mod 6) + 1}` | share = 0.05 / 0.25 / 1.00 | that share of rows set to null in the one column |
| `recode` | `EDUCATION` | share = 0.05 / 0.25 / 1.00 | on that share of rows, codes 1 and 2 swapped (still in range) |

σ_ref is the column's standard deviation in the reference. Severity values were chosen before
any run on a fixed principle: the low level sits near the noise floor, the middle is a typical
partial incident, the high level is the obvious case. They were not tuned on any result.

Label-dependent faults (a shift in the default rate, concept drift) are **out of scope**:
labels arrive months late in credit, so a feature-only responder cannot see them. The README
lists this as a known blind spot.

## Evaluation grid

- Evaluation seeds **0–19** for each class × severity → 4 × 3 × 20 = **240 fault scenarios**.
- **200 control windows** (no fault), seeds 0–199.
- Development seeds start at **10,000**. Tests, smoke runs and any debugging use only those.
  The evaluation seeds are run **once**, after the code and this protocol are frozen and the
  freeze commit's hash is written into NOTES.md.

## Signals and scoring

Per column of a window (definitions in `contracts.ColumnStats`): PSI, KS, null-rate delta,
out-of-range share, unseen-code share, median log-ratio, code-set change, became-constant.

**No-fault calibration.** Draw **200 windows** of 1,000 rows from the calibration pool
(seeds 0–199 of a separate stream, class_index 5). For each signal and column, record the median
and the MAD of the signal over those windows. A window's robust score for that signal and column
is

    z = max(0, (value − median) / max(1.4826 · MAD, 0.01)), capped at 1,000

The floor of 0.01 (one percentage point for the share signals) stops a signal that never moves
under no fault (null delta, out-of-range share) from turning a single odd row into an
unbounded score.

**Evidence per hypothesis (fault class on a column).**

| hypothesis | columns it applies to | evidence |
|---|---|---|
| `covariate_shift` | continuous and discrete | max(z_ks, z_psi) |
| `unit_break` | continuous | max(z_out_of_range, z_median_log_ratio) |
| `null_spike` | all 23 | z_null_delta |
| `recode` | categorical | max(z_psi, z_unseen_share) |

Hypotheses are sorted by evidence, highest first; ties are broken by the column's position in
`FEATURE_COLS`, then by the class order above. The column ranking is the order in which columns
first appear in that list.

**Incident verdict (ranker).** `incident = max evidence ≥ τ`, where τ is the 99th percentile of
the maximum evidence over **200 further no-fault windows** from the calibration pool (seeds
1,000–1,199, class_index 5). τ is fixed once from those windows and never touched again.

**Baseline.** Rank the 23 columns by the `psi` signal, highest first (an undefined PSI ranks
last; ties by `FEATURE_COLS` position), each labelled `covariate_shift`, which is what a page
reading "PSI high on X" implies. `incident = max PSI > 0.25`, the rule of thumb the source
model's README uses.

## Metrics

- **Primary: column top-1** on the 240 fault scenarios: 1 if the first column in the ranking is
  one of the scenario's affected columns, else 0. Scored identically for ranker and baseline.
  The baseline does not name a fault class, so the comparison is made on columns.
- Column MRR: 1 / rank of the first affected column.
- Cause top-1 (ranker only): the top hypothesis has the right class **and** an affected column.
  The baseline's implied "covariate shift" reading is shown beside it for context, not tested.
- Detection: share of fault scenarios with `incident = true`, per class and severity.
- **Control false-incident rate**: share of the 200 controls with `incident = true`, with a
  Wilson 95% interval.
- Paired bootstrap over the 240 fault scenarios: 10,000 resamples, seed 0, percentile 95%
  interval of mean(ranker column top-1 − baseline column top-1).
- Every metric also broken down by class and by severity.

## Hypotheses and decision rules (owner-locked 2026-09-30)

- **H1.** Ranker column top-1 minus baseline column top-1 is **≥ 0.15** (point estimate) **and**
  the paired 95% interval excludes 0. Both must hold.
- **H2.** Ranker control false-incident rate **≤ 5%** (point estimate on the 200 controls).
- **Null result (allowed).** The ranker ties PSI: unit breaks and shifts both raise PSI, so the
  extra signals add nothing. The recode fault is expected to be hard for both methods. A null
  is reported as plainly as a pass.

## What is not evaluated

- Postmortem prose (template or language model). It is a formatted view of the measured
  ranking; its wording is not scored.
- Real time ordering, delayed labels, multiple simultaneous faults.

## Amendments

### Amendment 1 — 2026-10-23, before any evaluation seed was run: class diagnosis within a column

**What was seen.** A development smoke (seeds 10,000–10,003, 48 fault windows, 40 controls)
showed the literal rule labelling every severity-1 and severity-2 unit break, and every 100%
null spike, as `covariate_shift`. PSI and KS rise under every fault and reach the 1,000 cap,
so the shift hypothesis outscores (or ties, and wins the tie on class order) the specific
signal. The approved brief requires "×1000 → unit break, not shift".

**Change.** Only the order of hypotheses *inside* a column:

1. Columns are ordered exactly as before: by their strongest hypothesis's evidence, ties by
   `FEATURE_COLS` position. The column ranking, the incident verdict and τ are unchanged.
2. Within a column, hypotheses with evidence ≥ τ come first, most specific class first:
   `null_spike`, `unit_break`, `recode`, `covariate_shift`. Hypotheses below τ follow, by
   evidence, ties by the same specificity order.
3. `Ranking.hypotheses` is therefore grouped by column (column-rank order), not one global
   sort by evidence. The maximum evidence is computed over all hypotheses.

**Why this cannot move H1 or H2.** Column top-1, column MRR and both incident verdicts depend
only on the column order and the maximum evidence, which item 1 leaves untouched. On the
development smoke, `ranker_top1`, `ranker_rr`, `ranker_incident`, `ranker_top_column`,
`ranker_max_evidence` and every baseline column were identical before and after. Only the
secondary cause top-1 changes.

**Cost seen on the same development seeds, kept, not tuned.** Large covariate shifts push some
AGE or LIMIT_BAL values beyond the reference range, so out-of-range crosses τ and 3 of the 8
severity-1/2 shift windows are now labelled `unit_break`. Before the change, 8 of 8 unit
breaks at severity 1–2 and 4 of 4 full null spikes were mislabelled; after it, 0 of each.

**Erratum, 2026-10-23 (same day, before any evaluation seed).** The mechanism in the paragraph
above is wrong. An independent review measured `out_of_range` at exactly 0 in every shift
window it checked. The `unit_break` label comes from `median_log_ratio` on **LIMIT_BAL**: that
column's values sit on a 10,000 grid, its no-fault window median is 140,000 in 126 of the 200
calibration windows, so its MAD is 0 and the scale falls to the 0.01 floor. A +30,000 or
+60,000 shift moves the median one or more grid steps and scores z ≈ 6–18, above τ. Expect
most LIMIT_BAL (odd-seed) shifts at severity 1–2 to be labelled `unit_break`; AGE shifts are
labelled correctly. The rule itself is unchanged.

### Amendment 2 — 2026-10-23, before any evaluation seed was run: run guard and one exploratory arm

**What was seen (development seeds only).** Two development smokes have been run: seeds
10,000–10,003 (48 faults, 40 controls) gave ranker − baseline column top-1 = +0.083
(0.021 to 0.167); the reviewer's seeds 10,000–10,009 (120 faults, 100 controls) gave +0.067
(0.017 to 0.125), τ = 7.683, false incidents 1/100. Both are below the H1 bar. They are recorded
here so that nothing below can be read as tuning toward H1: **the primary method, H1 and H2 are
not changed.**

**Heaped medians (a limitation of the frozen method, stated before the run).** In about 160 of
the 200 τ-fitting windows the largest no-fault evidence is `unit_break` via
`median_log_ratio`, on columns whose medians move in steps (LIMIT_BAL 44, PAY_AMT4 33, PAY_AMT5
30). That sets τ ≈ 7.7, which keeps severity-0 null spikes and unit breaks below the incident
line, and the same noise can outrank a small genuine shift (seen once on development seed
10,004: AGE severity 0 lost column top-1 to LIMIT_BAL median noise, z 3.2 against 3.1).

**Exploratory arm S1, declared now.** The same grid is also scored by a variant in which
`unit_break` evidence is `out_of_range` only (no `median_log_ratio`), with its own τ fitted the
same way on the same calibration windows. Its column top-1, cause top-1, detection, control
false-incident rate and paired difference against the baseline are reported in a separate,
labelled section. **S1 never decides H1 or H2**, and it is reported whatever it shows.

**Run guard.** `eval-responder --full` refuses to run unless the window size is 1,000, the data
file's SHA-256 is the vendored cache's, the tree is clean, and the output folder has no earlier
`records.csv`. The report prints the commit, data hash and library versions.
