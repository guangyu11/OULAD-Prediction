# COMP90049 Assignment 2

Predicting at-risk students from the Open University Learning Analytics dataset.

- **Report (Overleaf):** _paste link_
- **Dataset:** https://analyse.kmi.open.ac.uk/open_dataset
- **Paper:** Kuzilek, Hlosta & Zdrahal (2017), *Scientific Data* 4:170171

**RQ1** — Can students at risk of failing or withdrawing be identified from their first
weeks of engagement and enrolment data, and how do models across the bias–variance
spectrum compare?

**RQ2** — Which feature family carries the predictive signal (engagement behaviour,
early assessment performance, or static background), and does behavioural data add
anything beyond background alone?

## Files

| File | What it is |
|---|---|
| `main.ipynb` | **The file we submit.** Runs top to bottom, produces every number in the report |
| `work_A.ipynb` | A's scratch space — data and features |
| `work_B.ipynb` | B's scratch space — models and tuning |
| `work_C.ipynb` | C's scratch space — evaluation |
| `docs/` | Decision log, contribution log, dataset notes, exported results and figures |
| `data/` | The seven CSVs. Gitignored — never commit these |

## Setup

```bash
git clone <repo url>
cd comp90049-a2
pip install pandas numpy scikit-learn matplotlib seaborn jupyter nbstripout
nbstripout --install
```

Then download the dataset from the link above, unzip, and put the seven CSVs directly
in `data/`. Open `work_X.ipynb` and run the first two cells — if the row counts print,
you're set.

`nbstripout` strips notebook outputs on commit. Without it, two people running the same
notebook conflicts on every cell. It is the one setup step worth not skipping.

## How we work

**Work in your own file.** `work_A/B/C.ipynb` belong to one person each, so they never
conflict. Commit to them as often as you like.

**`main.ipynb` is shared, so one person at a time.** Say so in the group chat before you
start editing it, and push as soon as you're done. When a piece of your work is finished,
paste it into the section it belongs to and strip out the exploratory parts.

**The setup cell is the source of truth.** `SEED`, `CUTOFF_DAY` and `KEY` are defined at
the top of `main.ipynb`. Copy that cell into your work file rather than writing your own.
If it changes, whoever changed it tells the group and everyone re-copies — otherwise we
end up with results computed under different settings and no way to tell.

**Pull before you start, push when you stop.** Every session.

**Commit your own work.** Don't send files to one person to push. The commit history is
the evidence for the Stage III peer evaluation.

Commit messages: short, say the area. `features: clicks by activity type`,
`eda: class balance plot`, `main: merge B's tuning section`.

## Progress

Tick these off as they land.

- [ ] Data downloaded, row counts confirmed
- [ ] Unique students vs 32,593 rows — is the unit a student or a student-presentation?
- [ ] Target decided: four classes or binary at-risk
- [ ] Cutoff day decided and recorded in `docs/decisions.md`
- [ ] Split decided and justified
- [ ] Modelling table built
- [ ] Shared evaluation function works — same folds, same metrics, any model
- [ ] Baseline computed
- [ ] Four models trained at defaults
- [ ] All four tuned, search ranges logged
- [ ] RQ2 ablation run
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
