"""COMP90049 A2 — shared data layer.

Every model is trained, split and scored through this module so the four
models are compared under an identical protocol.

    import a2_common as A2

    df = A2.load_table(24)
    train, val, test = A2.make_splits(df)
    cols = A2.feature_cols("engineered")        # or "baseline" / "behaviour"
    pipe, scores = A2.score_model(MyModel(), cols, train, val)

Tune on `val` only; score `test` once, at the end.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
RANDOM_SEED = SEED
DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

KEY = ["code_module", "code_presentation", "id_student"]
GROUP_COL = "id_student"
TARGET = "at_risk"
CUTOFF_DAYS = [0, 24, 60, 120]
MAIN_CUTOFF = 24
FEATURE_SET_NAMES = ("baseline", "behaviour", "engineered")

_manifest = None


def manifest():
    """Load and cache manifest.json."""
    global _manifest
    if _manifest is None:
        with open(PROCESSED_DIR / "manifest.json", encoding="utf-8") as fh:
            _manifest = json.load(fh)
    return _manifest


def feature_cols(which="engineered"):
    """Column names for one feature set; engineered = baseline + behaviour."""
    sets = manifest()["feature_sets"]
    if which == "behaviour" and "behaviour" not in sets:
        base = set(sets["baseline"])
        return [c for c in sets["engineered"] if c not in base]
    if which not in sets:
        raise KeyError(f"unknown feature set {which!r}; available: {', '.join(sets)}")
    return list(sets[which])


def load_table(cutoff=MAIN_CUTOFF):
    """Read the wide table for one cutoff day."""
    path = PROCESSED_DIR / f"wide_cutoff_{cutoff:03d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the aggregation notebook first")
    return pd.read_csv(path)


def make_splits(df, seed=SEED):
    """Split 60/20/20, grouped by student and stratified on the target."""
    y, g = df[TARGET], df[GROUP_COL]

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_pos, test_pos = next(outer.split(df, y, g))
    trainval = df.iloc[trainval_pos].reset_index(drop=True)
    test = df.iloc[test_pos].reset_index(drop=True)

    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    tr_pos, val_pos = next(inner.split(trainval, trainval[TARGET], trainval[GROUP_COL]))
    train = trainval.iloc[tr_pos].reset_index(drop=True)
    val = trainval.iloc[val_pos].reset_index(drop=True)

    # No student may appear in more than one split.
    s_tr, s_va, s_te = (set(d[GROUP_COL]) for d in (train, val, test))
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)

    return train, val, test


def describe_splits(train, val, test):
    """Split sizes and class balance, as reported in the Method section."""
    total = len(train) + len(val) + len(test)
    return pd.DataFrame([{
        "split": name,
        "n": len(d),
        "share_%": round(len(d) / total * 100, 1),
        "students": d[GROUP_COL].nunique(),
        "positive_%": round(d[TARGET].mean() * 100, 2),
    } for name, d in [("train", train), ("validation", val), ("test", test)]])


def _log1p_nonneg(X):
    """log(1 + x) with negatives clipped; named rather than a lambda so it pickles."""
    return np.log1p(np.clip(X, 0, None))


def make_preprocessor(cols, log_transform=False):
    """Impute, optionally log-transform skewed columns, scale numerics, one-hot categoricals."""
    mf = manifest()
    cat = [c for c in cols if c in mf["categorical_features"]]
    num = [c for c in cols if c not in cat]

    skewed = [c for c in num if c in mf.get("skewed_cols", [])] if log_transform else []
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
    """Metric set used for every model; the positive class is non-completion."""
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


def confusion(y_true, y_pred):
    """Confusion matrix with readable row and column labels."""
    return pd.DataFrame(confusion_matrix(y_true, y_pred),
                        index=["true: completed", "true: at risk"],
                        columns=["pred: completed", "pred: at risk"])


def score_model(model, cols, train, val, log_transform=False, fit_on=None):
    """Build the pipeline, fit it, and evaluate on `val`. Returns (pipeline, metrics)."""
    pipe = Pipeline([("prep", make_preprocessor(cols, log_transform)), ("clf", model)])
    fit_df = train if fit_on is None else fit_on
    pipe.fit(fit_df[cols], fit_df[TARGET])
    pred = pipe.predict(val[cols])
    proba = pipe.predict_proba(val[cols])[:, 1] if hasattr(pipe, "predict_proba") else None
    return pipe, evaluate(val[TARGET], pred, proba)
