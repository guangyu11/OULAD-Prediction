# COMP90049 Assignment 2

Predicting at-risk students from the Open University Learning Analytics dataset.

- **Report (Overleaf):** _paste link_
- **Dataset:** https://analyse.kmi.open.ac.uk/open_dataset
- **Paper:** Kuzilek, Hlosta & Zdrahal (2017), *Scientific Data* 4:170171

**RQ1** — How early in a course can at-risk students be identified?

**RQ2** — Can behavioural data predict the final outcome better than background data?

## Files

| File | What it is |
|---|---|
| `main.ipynb` | **The file we submit.** Runs top to bottom, produces every number in the report |
| `work_Guangyu.ipynb` | Guangyu's scratch space |
| `work_WaiChun.ipynb` | Wai Chun's scratch space |
| `work_Xinyan.ipynb` | Xinyan's scratch space |
| `docs/` | Dataset notes, exported results and figures |
| `data/` | The seven CSVs. Gitignored — never commit these |

## Structure of `main.ipynb`

The spec asks for one notebook covering data processing, models, visualisation and
evaluation. It reads the raw CSVs from `data/` and builds everything in memory — no
`processed/` files, no `import a2_common`.

```
0. Setup                         SEED, KEY, CUTOFF_DAYS, imports
1. Data Preprocessing & Feature Engineering
   1.1 Load raw CSVs
   1.2 Background features, target
   1.3 Click + assessment features per cutoff, risk set
   1.4 Feature sets (background / behaviour / all)
   1.5 Shared split, Pipeline, evaluate(), run loop   ← used by sections 2 and 3
2. Classical ML Models & Evaluation
   2.1 Majority baseline
   2.2 Logistic Regression   (tune C)
   2.3 Decision Tree         (tune max_depth)
   2.4 Random Forest         (tune min_samples_leaf)
3. Neural Network Model & Evaluation
   3.1 MLP                   (tune e.g. hidden size / alpha)
4. Results: Research Questions
   4.1 RQ1: all models × all cutoffs   → table + line figure
   4.2 RQ2: feature sets × models      → table + bar figure
   4.3 Error analysis                  → confusion matrix + specific cases
```

**Section 1.5 comes before both model sections**, so the classical models and the neural
network go through the same split, preprocessing and metrics. The rubric's validation
design criterion checks exactly this.

**Sections 2 and 3 produce results; section 4 reads them.** Every model runs over all
cutoffs × feature sets and appends its rows to one shared `results` table:

```python
results += run(model, param_grid, cutoffs=CUTOFF_DAYS, feature_sets=FEATURE_SETS)
```

Section 4 builds the RQ tables and figures from `results` without retraining anything.
Agree on the `run` signature before writing model code, so the sections merge without changes.

### Model choice and the bias–variance spectrum

The spec asks for algorithms that "cover some spectrum from high-bias/low-variance to
low-bias/high-variance". All four models below are taught in the subject, so each result
can be explained with lecture theory.

| Model | Lecture | Bias / variance | Why | Tuned (grid search on validation macro-F1) |
|---|---|---|---|---|
| Logistic Regression | Wk 4 | **High bias, low variance** | Linear decision boundary only; L2 regularisation further limits variance | `C` ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10} |
| Decision Tree | Wk 3 | **Low bias, high variance** | Can split until leaves are pure; lecture lists "prone to overfitting" | `max_depth` ∈ {2, 4, 6, 8, 10, 15, 20, None} |
| Random Forest | Wk 8 | **Low bias, reduced variance** | Bagging + random feature subsets average many trees, which cuts variance | `min_samples_leaf` ∈ {1, 5, 10, 20, 50} |
| MLP (neural network) | Wk 7–8 | Low bias, high variance | Hidden layers can approximate non-linear functions | e.g. hidden size / `alpha` |

**Why these three:**

- **LR and DT are the two ends of the spectrum**: a rigid linear boundary vs a flexible tree
  that can memorise the training set.
- **RF is the DT with its variance removed.** The DT → RF gap measures what bagging buys
  (Wk 8: "bagging reduces variance"). That is a direct, testable theory–practice link.
- **Each tuning knob moves its model along the spectrum.** Small `C` or small `max_depth`
  means more bias; large values mean more variance. Plotting train vs validation score
  against `max_depth` reproduces the lecture's underfitting/overfitting curve on our own data.

**Not chosen:**

- **Naive Bayes:** its feature-independence assumption is clearly broken here (total clicks,
  active days and click halves are strongly correlated), and `GaussianNB` has only
  `var_smoothing` to tune.
- **KNN:** distances are unreliable across ~60 mixed one-hot, ratio and count features
  (curse of dimensionality), and prediction is slow at ~20k training rows.
- **SVM:** with an RBF kernel it scales poorly to ~20k rows × 36 grid cells; linear SVM
  would duplicate LR's position on the spectrum.
- **Gradient boosting:** strong, but not taught (only AdaBoost is), so its results would be
  harder to explain with course theory.

## Setup

```bash
git clone <repo url>
cd comp90049-a2
pip install pandas numpy scikit-learn matplotlib seaborn jupyter nbstripout
nbstripout --install
```

Then download the dataset from the link above, unzip, and put the seven CSVs directly
in `data/`. Open your own `work_<name>.ipynb` and check you can load a CSV from `data/`.

`nbstripout` strips notebook outputs on commit. Without it, two people running the same
notebook conflicts on every cell. It is the one setup step worth not skipping.

## How we work

**Work in your own file.** `work_Guangyu`, `work_WaiChun` and `work_Xinyan.ipynb` belong
to one person each, so they never conflict. Commit to them as often as you like.

**`main.ipynb` is shared, so one person at a time.** Say so in the group chat before you
start editing it, and push as soon as you're done. When a piece of your work is finished,
paste it into the section it belongs to and strip out the exploratory parts.

**The setup cell is the source of truth.** `SEED`, `CUTOFF_DAYS` (the cutoffs RQ1 compares)
and `KEY` go at the top of `main.ipynb`. Copy that cell into your work file rather than writing your own.
If it changes, whoever changed it tells the group and everyone re-copies — otherwise we
end up with results computed under different settings and no way to tell.

**Pull before you start, push when you stop.** Every session.

**Commit your own work.** Don't send files to one person to push. The commit history is
the evidence for the Stage III peer evaluation.

Commit messages: short, say the area. `features: clicks by activity type`,
`eda: class balance plot`, `main: merge Wai Chun's tuning section`.

## Progress

Tick these off as they land.

- [ ] Data downloaded, row counts confirmed
- [ ] Unique students vs 32,593 rows — is the unit a student or a student-presentation?
- [ ] Target decided: four classes or binary at-risk
- [ ] Cutoff days to compare for RQ1 decided
- [ ] Split decided and justified
- [ ] Modelling table built
- [ ] Shared evaluation function works — same folds, same metrics, any model
- [ ] Baseline computed
- [ ] Four models trained at defaults
- [ ] All four tuned, search ranges logged
- [ ] RQ2: behavioural-only vs background-only feature sets compared
- [ ] Error analysis done
- [ ] Results frozen, everything exported to `docs/`
- [ ] Report drafted
- [ ] `main.ipynb` runs clean on a fresh kernel and matches the report

## Dates

| Date | What |
|---|---|
| Fri 3 Oct | Data checks done, target and cutoff decided |
| Tue 7 Oct | Shared evaluation function working |
| Tue 14 Oct | Results frozen |
| Fri 16 Oct | Report + notebook submitted — no extensions |
| Fri 23 Oct | Peer evaluation and self-reflection |

## Before submitting

Restart the kernel, run `main.ipynb` top to bottom, and check the numbers match the
report. A notebook that disagrees with the report can zero the report component.
