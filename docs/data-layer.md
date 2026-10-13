# Data layer — what's built, how to use it, what to check

**Owner:** Wai Chun · **Last updated:** 1 Oct 2026 (RQ1 sweep added)

> **Status note (13 Oct):** this page describes the **work-file** setup (`processed/` files,
> `manifest.json`, `import a2_common`). In the submitted **`main.ipynb`**, Section 1 builds
> the tables in memory and writes nothing to disk; see the README section "How the
> deliverable code works".
>
> The split table and results under "Numbers to expect" and "RQ1 result so far" were produced
> with **scikit-learn 1.7.2**, which gives a different split. With the agreed **1.9.1**, the
> day-24 split is 16,688 / 5,563 / 5,562 and the baseline macro-F1 is 0.3562. Use the 1.9.1
> numbers (`processed/splits_summary.csv`, `processed/lr_test_results.csv`) as the reference.

This covers the cleaning, aggregation and the shared split/preprocessing/evaluation code.
Everything here is **up for review** — if a decision looks wrong, say so. Several of them
are judgement calls and the report has to defend them, so it is better to argue now.

---

## Quick start

You do **not** need to download the 443 MB dataset or run the cleaning notebook.
The built tables are committed under `processed/`.

```python
import a2_common as A2

df = A2.load_table(24)                  # wide table at cutoff day 24
train, val, test = A2.make_splits(df)   # grouped by student, stratified, 60/20/20
cols = A2.feature_cols("engineered")    # or "baseline" / "behaviour"

pipe, scores = A2.score_model(YourModel(...), cols, train, val)
print(scores)   # accuracy, precision, recall, f1, macro_f1, roc_auc
```

Rules we need to keep:

- **Do not write your own split or preprocessing.** The rubric's Validation design
  criterion requires the same validation approach across every model compared.
  `a2_common` exists so that is automatic.
- **Tune on `val` only. Touch `test` once, at the end.**
- Run each model with `log_transform=True` and `False` — see "The log experiment" below.

---

## What's in `processed/`

| File | What |
|---|---|
| `wide_cutoff_000.csv` | One row per enrolment, features from day ≤ 0 |
| `wide_cutoff_024.csv` | **Main table.** Day ≤ 24 (75th pct of first assignment submission) |
| `wide_cutoff_060.csv` | Day ≤ 60 (second assignment submitted) |
| `wide_cutoff_120.csv` | Day ≤ 120 (course midpoint) |
| `manifest.json` | Target, group column, both feature sets, forbidden columns, skewed columns |
| `cutoff_summary.csv` | Sample counts and class balance per cutoff |
| `splits_summary.csv` | Train/val/test sizes and class balance |
| `evidence_1_hidden_missing.csv` | `?`-encoded missing values |
| `evidence_2_leakage.csv` | `date_unregistration` leakage cross-tab |
| `evidence_3_skewness.csv` | Skew before/after log per feature |
| `lr_tuning_grid.csv` | 28 Logistic Regression runs at cutoff 24 |
| `lr_test_results.csv` | Test-set numbers for baseline + LR at cutoff 24 |
| `rq1_cutoff_sweep.csv` | Baseline + LR across all four cutoffs |
| `rq1_summary.csv` | Per-cutoff summary with lift over each cutoff's own baseline |

All four tables have **identical columns** (61), so results are comparable across cutoffs.

---

## The target

`at_risk` — **1 = did not complete** (Fail or Withdrawn), **0 = completed** (Pass or Distinction).

Positive class is "did not complete" so that precision and recall read the way the
problem is framed: precision = of the students we flag, how many really are at risk;
recall = of the students really at risk, how many we catch.

Class balance is 44.68 % positive at cutoff 24 — mild imbalance, no resampling needed.

**Open to challenge:** merging Withdrawn with Fail treats "dropped out" and "failed the
exam" as the same outcome. They are arguably different phenomena. The alternatives were
(a) drop Withdrawn entirely, losing 31 % of rows including the most interventionable
group, or (b) predict Withdrawn vs rest, which changes the research question. We went
with the merge and will name the conflation as a limitation.

---

## Decisions that affect your models

### 1. Split is grouped by `id_student`, not random

3,538 students appear in more than one row (up to 5), because they take several modules
or retake one. Their outcomes are strongly correlated: among repeat students, those who
completed their first enrolment completed their second **55.9 %** of the time; those who
did not, **24.7 %** (overall base rate 47.2 %).

With a random split, a model can recognise a student it saw in training — region,
education, IMD band and credits barely change between their rows — and score well without
having learned anything generalisable. `make_splits` asserts no student crosses splits.

### 2. Features are truncated at a cutoff day

Withdrawn students stop producing clicks once they withdraw. Using the full course
window makes "few active days" look predictive of withdrawal when the causation runs the
other way. Every feature uses only `date <= cutoff`.

### 3. Risk set: students who had already withdrawn by the cutoff are excluded

Predicting at day *t* for someone who withdrew at day *t−5* is recording, not predicting.
Excluded counts: 3,097 (day 0) · 4,780 (day 24) · 6,232 (day 60) · 7,816 (day 120).
This is why *n* shrinks and class balance shifts as the cutoff moves — report *n* per cutoff.

### 4. `date_unregistration` is dropped

It is present for 99.08 % of Withdrawn rows and 0 % of Pass/Distinction rows — effectively
the label. It stays in the table only so the risk-set filter can use it. **It is in
`manifest.json["forbidden_columns"]`. Do not put it in a feature set.**

### 5. Missing values are `?` strings in the raw CSVs

`date_registration`, `score`, `week_from` and others parse as strings unless you pass
`na_values=['?']`. Already handled in the cleaning notebook; flagging it in case you load
a raw CSV yourself.

---

## Feature sets

| Set | Count | Contents |
|---|---|---|
| `baseline` | 14 | Background only: module, presentation, gender, region, disability, education (ordinal), IMD (ordinal), age band, previous attempts, credits, registration lead time, late-registration flag, course length |
| `behaviour` | 38 | Behaviour only: everything generated after enrolment (VLE clickstream + assignment submissions). Nothing known at registration time. |
| `engineered` | 52 | `baseline` + `behaviour` |

By construction `engineered == baseline + behaviour`, with no overlap — this is asserted
in `A2_01` so the three sets cannot silently drift apart. Call them with
`A2.feature_cols("baseline" | "behaviour" | "engineered")`; please do not hand-slice the
column list in your own notebook, or the three of us stop comparing the same thing.

`behaviour` exists for the ablation: `baseline` vs `engineered` answers RQ2 as stated,
and `behaviour` alone shows how much of the signal is carried by behaviour *without* any
demographic information — useful if you want to argue the model is not simply
reproducing demographic disparities.

Behavioural features: total clicks, active days, distinct resources, max clicks in a day,
mean clicks per active day, first-half vs second-half click counts and their ratio,
pre-start clicks, active-days ratio, 20 resource-type click proportions, assignments
submitted, mean/min score, weighted score, mean days early, banked count, and two
"had any activity" flags.

`highest_education` and `imd_band` are **ordinal-encoded, not one-hot** — they have a
natural order and one-hot would discard it. `imd_band` also had a formatting bug in the
raw data (`10-20` missing its `%`), fixed before encoding.

---

## The log experiment — please run both

Seven features are heavily right-skewed. `manifest.json["skewed_cols"]`:

```
vle_max_clicks_day, vle_trend_ratio, vle_clicks_h1,
vle_clicks_before_start, vle_total_clicks, vle_clicks_h2, asm_n_submitted
```

Selection rule was **not** "skew > 2". The `vle_prop_*` columns have skew up to 63 but
that is sparsity (most students never touch that resource type), and log barely moves it
(63.7 → 63.1). The rule used was: skew > 2 **and** log cuts |skew| by ≥ 50 % **and**
the feature is a magnitude, not a 0–1 proportion. Full table in `evidence_3_skewness.csv`.

`A2.score_model(..., log_transform=True)` applies `log1p` to exactly those columns.

**Why it matters per model family:**

| Family | Affected | Why |
|---|---|---|
| Decision Tree, Random Forest | No | Splits on `x > threshold`, so only ranks matter. `log` is monotonic — identical splits |
| Logistic Regression | Yes in theory | Weighted sum, extreme values can dominate coefficients |
| MLP | Most | Large inputs saturate activations / destabilise gradients |

**What we actually measured for Logistic Regression: essentially nothing.**
macro-F1 changed by −0.0007. Possible reasons: `StandardScaler` plus tuned L2 already
damp the extremes; `log` changes the functional form relative to the log-odds, not just
scale, and neither form fits much better here.

This is reported as-is, not smoothed over. The rubric rewards acknowledging evidence that
cuts against the expectation.

**MLP is the real test** — it has no default regularisation and is far more input-scale
sensitive. If log helps anywhere it should help there. Please report both numbers.

Sanity check that confirms the switch is wired correctly: on the `baseline` feature set
the difference is exactly 0.0000, because none of the skewed columns are background
features.

---

## Numbers to expect

If your run differs from these, something is off — tell me rather than working around it.
`SEED = 42` everywhere.

**Splits at cutoff 24** (n = 27,813):

| Split | n | Students | Positive % |
|---|---|---|---|
| train | 16,685 | 15,032 | 44.51 |
| validation | 5,553 | 5,024 | 44.12 |
| test | 5,575 | 5,012 | 45.76 |

**Test-set results:**

| Model | Features | Accuracy | Precision | Recall | macro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Majority-class baseline | — | 0.5424 | 0.0000 | 0.0000 | **0.3517** | 0.500 |
| Logistic Regression | background | 0.6210 | 0.6250 | 0.4292 | 0.6002 | 0.6616 |
| Logistic Regression | background + behaviour | 0.7220 | 0.7417 | 0.6021 | **0.7136** | 0.7905 |

Best `C` was 0.3 for `engineered`, 0.03 for `baseline` (searched 0.01 → 10).

**If your model scores above ~0.95 AUC, assume leakage before assuming success.**

---

## What I'd like you to check

1. **Does the baseline/engineered gap hold for your model?** For LR it is +0.11 macro-F1.
   If a tree model shows a much smaller gap, that is interesting and belongs in the report.
2. **Does `log_transform` help the MLP?** This is the open question above.
3. **Recall is low** — LR misses 40 % of at-risk students (1,015 false negatives vs 535
   false positives). In an intervention setting a miss costs more than a false alarm, so
   threshold tuning or `class_weight` is worth trying. If someone wants to own that
   experiment it would strengthen the Discussion.
4. **Feature engineering gaps.** The behavioural features are my guesses. If you think
   something obvious is missing — assignment *pass* rate rather than mean score, gaps
   between logins, weekday/weekend pattern — say so and I will add it to the builder.
5. **Anything in the "Decisions" section you disagree with.** Especially the
   Withdrawn/Fail merge and the choice of cutoff days.

---

## RQ1 result so far (Logistic Regression only)

All four cutoffs have now been swept with the majority baseline + LR
(`A2_03_rq1_cutoff_sweep.ipynb`, results in `processed/rq1_cutoff_sweep.csv` and
`rq1_summary.csv`, figure in `figures/fig_rq1_cutoff_sweep.png`).

| Cutoff | n | Positive % | Majority baseline | LR background | LR + behaviour | Behaviour gain |
|---|---|---|---|---|---|---|
| 0 | 29,496 | 47.84 | 0.3452 | 0.6107 | 0.6568 | +0.046 |
| 24 | 27,813 | 44.68 | 0.3517 | 0.6009 | 0.7136 | +0.113 |
| 60 | 26,361 | 41.64 | 0.3709 | 0.5819 | 0.7528 | +0.171 |
| 120 | 24,777 | 37.91 | 0.3847 | 0.5679 | 0.7911 | +0.223 |

(macro-F1, test set. Each cutoff is scored against **its own** baseline because *n* and
class balance shift as the risk set changes — raw scores are not comparable across cutoffs,
the lift is.)

**Two findings:**

1. A usable signal exists before the course starts (0.657 vs a 0.345 baseline at day 0),
   and keeps sharpening — it has not plateaued by the midpoint.
2. The value of behavioural data grows ~5x from day 0 to day 120, monotonically, while
   **background-only performance declines** (0.611 → 0.568, recall 0.546 → 0.284).

Proposed explanation for (2): withdrawal is front-loaded (30.7 % before day 0, 50.2 % by
day 28). The students most predictable from demographics are the ones who leave earliest,
so later cutoffs have already removed them from the risk set. The curves diverge because
the easy cases leave the sample, not because background information degrades.

### ⚠️ This needs your models to be trustworthy

Everything above is **one linear model**. The declining background curve could be a
property of the data or a property of Logistic Regression — there is no way to tell from
one model family.

**Please run your model across all four cutoffs, not just 24.** The loop is in
`A2_03_rq1_cutoff_sweep.ipynb` — swap the estimator and it works unchanged. If the same
divergence shows up in a tree model and the MLP, the claim is solid; if it does not, that
difference is itself a result worth reporting.

---

## Not done yet

- Tree models and MLP have not been run at any cutoff.
- No error analysis beyond the confusion matrix.
- Within-module comparison of repeat vs first-time students (controls for module difficulty).
- Sensitivity analysis restricted to post-start withdrawals.

---

## Reproducing from scratch

Only needed if you change the cleaning logic.

1. Download the dataset (link in README), unzip, put the seven CSVs directly in `data/`
2. Run `A2_01_clean_and_aggregate.ipynb` top to bottom — about 15 minutes, rebuilds `processed/`
3. Run `A2_02_baseline_and_lr.ipynb` — about 1 minute

Both were verified with Restart & Run All on 1 Oct 2026.

---

## Note for the merge into `main.ipynb`

`a2_common.py` is a separate file, but the submission is **one** `.ipynb`. When we merge,
its contents need to be pasted in as a single cell near the top and the `import a2_common`
lines dropped. Mechanical, but worth knowing now rather than on 15 October.
