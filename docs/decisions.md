# Decision log

Every choice that needs justifying in the report. Write the entry **the day you make
the decision**, while you still remember why — not in the final week from memory.

This file is the raw material for the Method section. A decision recorded with its
reasoning is a 2-mark sentence; the same decision reconstructed two weeks later is
"we joined the files on timestamps".

Format: what we decided, why, who, when.

---

## Dataset

**Decided:** OULAD (Open University Learning Analytics dataset)
**Why:** 32,593 rows guaranteed; peer-reviewed data descriptor in *Scientific Data*;
seven relational tables give real feature-engineering work; low risk compared with
alternatives whose row count was unverified.
**Who / when:** all three, 30 Sep

---

## Unit of analysis

**Decided:** _pending_
**The question:** `studentInfo` has 32,593 rows, but `id_student` repeats for anyone who
took more than one course. Is a row a student or a student-presentation?
**Why it matters:** if students repeat, a random split puts the same person on both
sides of it, which is leakage.
**Who / when:**

---

## Cutoff day

**Decided:** _pending — placeholder is day 30_
**The question:** features use only data from before this day of the presentation.
**Why it matters:** without a cutoff, assessment scores from late in the course nearly
determine `final_result` and the model is near-tautological. Too early and there is no
signal to learn from.
**What we checked:** how much click activity exists before the candidate day.
**Who / when:**

---

## Target definition

**Decided:** _pending_
**The question:** keep four classes (Distinction, Pass, Fail, Withdrawn) or collapse to
binary at-risk (Fail + Withdrawn)?
**Why it matters:** withdrawal is a decision, failure is an outcome — they are not the
same event. Collapsing them needs justifying.
**Who / when:**

---

## Early withdrawals

**Decided:** _pending_
**The question:** exclude students who withdrew before the cutoff?
**Why it matters:** they have no engagement signal and an already-settled outcome.
Excluding them is defensible but changes the class balance and what the model claims
to do. Count them either way.
**How many:** _fill in_
**Who / when:**

---

## Validation split

**Decided:** _pending_
**Options:** grouped by `id_student`; held-out presentations; stratified k-fold.
**Why it matters:** this is a whole rubric criterion, and the justification has to come
from a property of the data, not from convention.
**Who / when:**

---

## Metrics

**Decided:** _pending_
**Class balance observed:** _fill in_
**Why these:** _accuracy alone is misleading under imbalance — say what we used instead
and why_
**Who / when:**

---

## Hyperparameter searches

One row per model. All three columns go in the report.

| Model | Parameter | Range searched | Method | Winner |
|---|---|---|---|---|
| | | | | |

---

## Things we tried that did not work

Worth as much as the things that did — the rubric explicitly rewards showing an attempt
to overcome a limitation, even a failed one.

-
