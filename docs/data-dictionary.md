# OULAD data dictionary

The [Open University Learning Analytics Dataset](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset)
(Kuzilek, Hlosta & Zdrahal, 2017) covers 7 modules run 22 times in 2013–2014.

## Tables

| File | Rows | One row is |
|---|---|---|
| `courses.csv` | 22 | One module presentation and its length in days |
| `studentInfo.csv` | 32,593 | One enrolment: background fields and `final_result` |
| `studentRegistration.csv` | 32,593 | When the student registered, and when they withdrew (if they did) |
| `vle.csv` | 6,364 | One online resource and its `activity_type` |
| `studentVle.csv` | 10,655,280 | One student's clicks on one resource on one day |
| `assessments.csv` | 206 | One assessment: type, due day, weight |
| `studentAssessment.csv` | 173,912 | One student's submission and score |

## Terms

- **Module:** a course, AAA to GGG.
- **Presentation:** one run of a module. `2013J` started in October 2013, `2014B` in February 2014.
- **Enrolment:** the triple `(code_module, code_presentation, id_student)`, the unit of prediction.
  `id_student` alone is not unique: 3,538 students have more than one enrolment.
- **VLE:** Virtual Learning Environment, the Open University's online course site.
- **IMD band:** Index of Multiple Deprivation decile of the student's area.

## Pitfalls handled in this project

| Pitfall | How it is handled |
|---|---|
| Missing values are the string `?`, so a default read reports 0% missing and loads numeric columns as text | `na_values=["?"]` on every read (`oulad.diagnostics.hidden_missing_values`) |
| `date_unregistration` is filled for 99.08% of withdrawals and almost nothing else, so it encodes the label | Never used as a feature; used only to build the risk set |
| Students with several enrolments have correlated outcomes | Splits are grouped by `id_student` |
| Dates are day numbers relative to the module start (day 0); negative means before the start | Cutoffs and the observation window are defined on this scale |
| `studentAssessment` has no module or presentation columns | Joined through `assessments` on `id_assessment` |
| The `10-20` IMD band is missing its `%` in 3,516 records | Normalised to `10-20%` before encoding |

## Label

`final_result` is Distinction, Pass, Fail or Withdrawn. This project predicts
`at_risk = 1` for Fail or Withdrawn, and `0` for Pass or Distinction.
