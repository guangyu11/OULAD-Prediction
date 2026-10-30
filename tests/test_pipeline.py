"""Unit tests on small synthetic data; they do not need the OULAD download."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oulad.features import (BACKGROUND_CAT, BACKGROUND_NUM, FeatureBuilder,  # noqa: E402
                            feature_sets, vle_features)
from oulad.modelling import evaluate, make_split  # noqa: E402

TYPES = ["forumng", "quiz"]


def _enrolments(n=6):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "code_module": "AAA", "code_presentation": "2013J",
        "id_student": np.arange(n, dtype="int64"),
        "at_risk": [0, 1] * (n // 2),
        "date_unregistration": [np.nan] * (n - 1) + [5.0],   # last student withdraws on day 5
    })
    for col in BACKGROUND_CAT[2:]:
        df[col] = "x"
    for col in BACKGROUND_NUM:
        df[col] = rng.random(n)
    return df


def _clicks():
    return pd.DataFrame({
        "code_module": "AAA", "code_presentation": "2013J",
        "id_student": [0, 0, 1],
        "id_site": [1, 2, 1],
        "date": [-3, 20, 5],
        "sum_click": [4, 100, 2],
        "activity_type": pd.Categorical(["forumng", "quiz", "forumng"]),
    })


def _assessments():
    return pd.DataFrame({
        "code_module": "AAA", "code_presentation": "2013J",
        "id_student": [0, 1], "id_assessment": [10, 10],
        "date_submitted": [2, 30], "score": [80.0, 60.0], "weight": [10.0, 10.0],
        "is_banked": [0, 0], "days_early": [1, -2],
    })


def test_vle_features_ignore_clicks_after_cutoff():
    f = vle_features(_clicks(), cutoff=10, activity_types=TYPES)
    assert f.loc[("AAA", "2013J", 0), "vle_total_clicks"] == 4      # the day-20 click is excluded
    assert f.loc[("AAA", "2013J", 0), "vle_prop_quiz"] == 0.0


def test_risk_set_drops_students_who_already_withdrew():
    builder = FeatureBuilder(_enrolments(), _clicks(), _assessments(), TYPES)
    early, _ = builder.build(0)
    later, summary = builder.build(10)
    assert 5 in set(early["id_student"])
    assert 5 not in set(later["id_student"])
    assert summary["n_excluded"] == 1


def test_assessments_after_cutoff_are_not_counted():
    table, _ = FeatureBuilder(_enrolments(), _clicks(), _assessments(), TYPES).build(10)
    submitted = table.set_index("id_student")["asm_n_submitted"]
    assert submitted[0] == 1 and submitted[1] == 0               # student 1 submits on day 30


def test_feature_groups_are_disjoint_and_leak_free():
    table, _ = FeatureBuilder(_enrolments(), _clicks(), _assessments(), TYPES).build(10)
    sets = feature_sets(table)
    assert len(sets["background"]) == 14
    assert "date_unregistration" not in sets["all"]
    assert set(sets["all"]) == set(sets["background"]) | set(sets["behaviour"])


def test_split_keeps_each_student_in_one_set():
    rng = np.random.default_rng(1)
    students = np.repeat(np.arange(500), 2)                     # every student has two rows
    df = pd.DataFrame({"id_student": students, "at_risk": rng.integers(0, 2, len(students))})
    split = make_split(df)
    ids = [set(d["id_student"]) for d in split]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert len(split.test) == pytest.approx(0.2 * len(df), rel=0.1)


def test_majority_baseline_scores_zero_recall():
    y = np.array([0, 0, 0, 1, 1])
    scores = evaluate(y, np.zeros_like(y), np.full(len(y), 0.5))
    assert scores["recall"] == 0 and scores["roc_auc"] == 0.5
