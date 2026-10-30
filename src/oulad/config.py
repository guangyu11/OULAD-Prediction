"""Project-wide constants."""

SEED = 42

# One row = one enrolment: a student taking one module in one presentation.
KEY = ["code_module", "code_presentation", "id_student"]

# Prediction points, in days relative to the module start (day 0).
CUTOFF_DAYS = [0, 24, 60, 120]

# First day of the VLE observation window (clicks before the module starts count).
VLE_WINDOW_START = -25

TARGET = "at_risk"            # 1 = Fail or Withdrawn, 0 = Pass or Distinction
GROUP_COL = "id_student"      # splits are grouped on this column

# Reference setting used for tuning curves and error analysis.
REFERENCE_CUTOFF = 24
REFERENCE_FEATURES = "all"

# Missing values in the raw CSVs are the string "?", not empty fields.
NA_VALUES = ["?"]
