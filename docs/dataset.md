# OULAD — quick reference

Full field guide: https://claude.ai/artifact/XejJqRb86Qh619Sgoocx7x

## The seven tables

| File | Rows | What one row is |
|---|---|---|
| `courses.csv` | 22 | One course presentation, and how many days it ran |
| `studentInfo.csv` | 32,593 | One student in one course — background and `final_result` |
| `studentRegistration.csv` | 32,593 | When they enrolled, and when they left if they did |
| `vle.csv` | 6,364 | One piece of course material, and its `activity_type` |
| `studentVle.csv` | 10,655,280 | One student clicking one resource on one day |
| `assessments.csv` | 206 | One assignment: type, due day, weight |
| `studentAssessment.csv` | 173,912 | One student's mark on one assignment |

## Terms

**Module** — a course. Seven of them (AAA, BBB, …).

**Presentation** — one run of a course, like a semester offering. 22 of them. The code
is year plus a start-month letter counting A for January, so `2013J` started in October
2013 and `2014B` in February 2014. The data descriptor notes February and October runs
of the same module can differ structurally.

**VLE** — Virtual Learning Environment, the university's online course site. The
British term for an LMS. Since the OU teaches entirely at a distance, clicking is
effectively attending.

**IMD band** — Index of Multiple Deprivation, a UK measure of area disadvantage, given
as a decile. Has missing values.

## Things that will catch you out

**Dates are day numbers, not calendar dates.** Day 0 is the first day of that
presentation. Negative means before it started, so `date_registration = -90` is
enrolling three months early.

**The key is a triple.** `(code_module, code_presentation, id_student)`. `id_student`
alone is not unique — someone taking two courses appears twice. Merging on
`id_student` alone silently produces a cross-product.

**The column is `id_student`, not `student_id`.** Same for `id_site`, `id_assessment`.

**`studentAssessment` can't join directly.** It carries only `id_student` and
`id_assessment` — no module or presentation. Reach them through `assessments`.

**`date_unregistration` is null for anyone who finished.** Not missing data; it means
they did not withdraw.

**Final exam scores are mostly absent.** The descriptor says exams are marked and used
immediately. Do not treat the absence as a zero.

## The label

`final_result` in `studentInfo`: Distinction, Pass, Fail, Withdrawn.

## What each research question needs

**RQ1 — How early in a course can at-risk students be identified?**
Build the same features at several cutoff days and see how performance changes. At a
given cutoff, use only `studentVle` rows and `studentAssessment` submissions dated on or
before it. Students whose `date_unregistration` falls before the cutoff have already
withdrawn — decide whether to drop them, or the early cutoffs get an easy win.

**RQ2 — Can behavioural data predict the final outcome better than background data?**
- *Behavioural:* `studentVle` — clicks, active days, activity types.
- *Background:* `studentInfo` demographics (gender, region, education, IMD band, age,
  disability, previous attempts, credits) and `date_registration`.
- *Undecided:* assessment scores from `studentAssessment`. They are neither clicks nor
  background — agree as a group which side they go on, or keep them as a third set.

## Citation

Jakub Kuzilek, Martin Hlosta, and Zdenek Zdrahal. 2017. Open University Learning
Analytics dataset. *Scientific Data*, 4:170171.
