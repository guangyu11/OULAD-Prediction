# Meeting log

## Mon 12 Oct 2026

**Attendees:** Guangyu, Wai Chun, Xinyan
**Context:** results freeze is in about two days (14 Oct is a Wednesday, so confirm Tue 13 or Wed 14). Report and notebook are due Fri 16 Oct, 23:59.

Each item has a proposal. Fill in **Decision** and **Owner** during the meeting.

---

### 1. What data processing hands to the model sections (Guangyu, Xinyan)

**Proposal:** sections 2 and 3 receive exactly two things from section 1:

- `SPLITS`: `{cutoff: (train, val, test)}` for cutoffs 0 / 24 / 60 / 120, grouped by student, 60/20/20
- `FEATURE_SETS`: `background` (14 columns), `behaviour` (38, new), `all` (52)

**To confirm:** are the current `processed/` tables final? Any change to features after training means every model has to be rerun.

- **Decision:**
- **Owner:**

---

### 2. Section 1: Wai Chun's code consolidated, shared helpers reused by everyone

**Proposal:** Wai Chun consolidates the data-processing code into `work_WaiChun.ipynb`, which becomes Section 1 of `main.ipynb`:

```
1.1 Load raw CSVs
1.2 Background features, target
1.3 Click + assessment features per cutoff, risk set
1.4 Feature sets (background / behaviour / all)
1.5 Shared harness: make_splits, make_preprocessor, evaluate, score_model, run_experiment()
```

- **We reuse the shared helpers** in 1.5 (currently in `a2_common.py`) for every model. Nobody writes their own split, preprocessing or metrics, so every model goes through the same validation, which the rubric's validation-design criterion requires.
- English comments and output; no `manifest.json` / CSV hand-off layer.
- Reads the raw CSVs from `data/` and builds the tables in memory, as the spec requires (one notebook, data processing included).

**Risks:**

- **Consolidate, don't rewrite.** If the feature logic changes, every number changes, and there is little time to rerun before the freeze.
- **Freeze the helpers now.** A change to the split, preprocessing or metrics after models are trained means rerunning everything.

- **Decision:**
- **Owner:** Wai Chun (consolidation); everyone (reuse)
- **Due:**

---

### 3. Interfaces between our code

**Proposal:**

| Interface | Agreed form |
|---|---|
| Training loop | `run_experiment(name, model, param, values, cutoffs, feature_sets)` → `(rows, curve, best_pipe)` |
| Results columns | `model, cutoff, feature_set, best_param, accuracy, precision, recall, f1, macro_f1, roc_auc` |
| Feature-set names | `background`, `behaviour`, `all` (rename `baseline` → `background` to avoid confusion with the majority baseline) |
| Where `run_experiment()` lives | §1.5 (shared), so the MLP in Section 3 uses the same loop as the classic models |

Sections 2 and 3 are parallel: both take Section 1's output and produce results rows in the same format. Section 4 combines them.

- **Decision:**
- **Owner:**

---

### 4. Neural network: same validation as the classic models (Xinyan)

**Why it matters:**

- The spec requires at least one RQ to compare the ML models **with** the NN (missing it costs at least 5 marks).
- The rubric requires that **every** algorithm, the NN included, has at least one tuned hyperparameter with its method and range reported and the chosen value explained.
- All models must use the same validation approach, or the comparison isn't fair.

**The NN needs, like each classic model:**

- All 12 cells (4 cutoffs × 3 feature sets), each tuned on val → refit on train+val → tested once
- At least one tuned hyperparameter (e.g. `alpha` or hidden-layer size), with its grid
- A train + val tuning curve for cutoff 24 + all features (explains the chosen value; MLPs overfit easily, so it is a natural bias–variance point)

**Proposal, depending on the library:**

| NN library | Approach |
|---|---|
| scikit-learn `MLPClassifier` | Use the shared `run_experiment()` directly, e.g. `run_experiment("MLP", MLPClassifier(early_stopping=True, max_iter=500, random_state=SEED), "alpha", [1e-4, 1e-3, 1e-2, 1e-1])` |
| PyTorch / Keras | `run_experiment()` won't work (it relies on scikit-learn's `clone`, `set_params`, `predict_proba`). Write a separate loop that uses the same `SPLITS`, `FEATURE_SETS`, preprocessing, protocol and `evaluate`, and returns the same results columns |

**To confirm:** which library is the NN built with?

- **Decision:**
- **Owner:** Xinyan

---

### 5. Lead metrics

`A2.evaluate` returns **6** metrics: accuracy, precision, recall, F1, macro-F1, ROC-AUC. The rubric needs at least 3, each with a stated reason.

**Proposal:** compute all 6 (no extra cost), and feature these 3 in the report:

| Metric | Why |
|---|---|
| **Macro-F1** (lead) | Fair under class imbalance; the majority baseline scores only ~0.36 despite ~55% accuracy |
| **Recall** (at-risk class) | Missing an at-risk student costs more than a false alarm |
| **ROC-AUC** | Threshold-free, so models can be compared without fixing a cut-off |

Accuracy is reported only as context.

- **Decision:**

---

### Also worth settling if time allows

- **Freeze date:** Tue 13 or Wed 14 Oct
- **Package versions:** everyone uses scikit-learn **1.9.1** and pandas **3.0.6**. A different version produced a different split before (`rq1_cutoff_sweep.csv` doesn't match the current split)
- **Section 4 owner:** RQ tables and figures, error analysis

---

### Action items

| Action | Owner | Due |
|---|---|---|
| | | |
