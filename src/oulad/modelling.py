"""Splitting, preprocessing and evaluation shared by every model."""

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import GROUP_COL, SEED, TARGET
from .features import BACKGROUND_CAT


class Split(NamedTuple):
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def make_split(df, seed=SEED):
    """60/20/20 split, grouped by student and stratified on the target."""
    y, g = df[TARGET], df[GROUP_COL]

    # 1 of 5 folds = 20% test
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_pos, test_pos = next(outer.split(df, y, g))
    trainval = df.iloc[trainval_pos].reset_index(drop=True)
    test = df.iloc[test_pos].reset_index(drop=True)

    # 1 of the remaining 4 folds = 20% validation
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    tr_pos, val_pos = next(inner.split(trainval, trainval[TARGET], trainval[GROUP_COL]))
    train = trainval.iloc[tr_pos].reset_index(drop=True)
    val = trainval.iloc[val_pos].reset_index(drop=True)

    # No student may appear in more than one split.
    s_tr, s_va, s_te = (set(d[GROUP_COL]) for d in (train, val, test))
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
    return Split(train, val, test)


def describe_split(split):
    """Split sizes and class balance."""
    total = sum(len(d) for d in split)
    return pd.DataFrame([{
        "split": name,
        "n": len(d),
        "share_%": round(len(d) / total * 100, 1),
        "students": d[GROUP_COL].nunique(),
        "positive_%": round(d[TARGET].mean() * 100, 2),
    } for name, d in split._asdict().items()])


def _log1p_nonneg(X):
    """log(1 + x) with negatives clipped; a named function rather than a lambda so it pickles."""
    return np.log1p(np.clip(X, 0, None))


def make_preprocessor(cols, skewed_cols=()):
    """Impute, optionally log-transform skewed columns, scale numerics, one-hot categoricals.

    Fitted inside the model pipeline, so only training data ever informs it.
    """
    cat = [c for c in cols if c in BACKGROUND_CAT]
    num = [c for c in cols if c not in cat]
    skewed = [c for c in num if c in skewed_cols]
    plain = [c for c in num if c not in skewed]

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    skew_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("log", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    transformers = [("num", num_pipe, plain)]
    if skewed:
        transformers.append(("skew", skew_pipe, skewed))
    if cat:
        transformers.append(("cat", cat_pipe, cat))
    return ColumnTransformer(transformers, remainder="drop")


def evaluate(y_true, y_pred, y_proba=None):
    """Metrics for every model; the positive class is non-completion (at risk)."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    return {k: round(float(v), 4) for k, v in out.items()}


def confusion_frame(y_true, y_pred):
    """Confusion matrix with readable row and column labels."""
    return pd.DataFrame(confusion_matrix(y_true, y_pred),
                        index=["true: completed", "true: at risk"],
                        columns=["pred: completed", "pred: at risk"])
