"""The experiment grid: every model tuned and tested at every cutoff x feature group."""

from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .config import REFERENCE_CUTOFF, REFERENCE_FEATURES, TARGET
from .modelling import evaluate, make_preprocessor

RESULT_COLUMNS = ["model", "cutoff", "feature_set", "best_param",
                  "accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"]


@dataclass
class ExperimentResult:
    """Everything one model produced across the grid."""

    name: str
    param: str
    rows: list
    curve: dict = field(default_factory=dict)   # {value: validation macro-F1}, reference cell
    pipeline: Pipeline = None                   # fitted final model, reference cell

    @property
    def table(self):
        """Test scores, one row per cutoff x feature group."""
        return pd.DataFrame(self.rows, columns=RESULT_COLUMNS)

    @property
    def best_validation_score(self):
        return max(self.curve.values())


class ExperimentRunner:
    """Runs the same protocol for every model, so comparisons are fair.

    In each cutoff x feature-group cell:
    1. fit once per hyperparameter value on train, score macro-F1 on validation;
    2. keep the value with the highest validation macro-F1 (ties go to the first, simpler value);
    3. refit on train + validation;
    4. score the test set once.
    """

    def __init__(self, splits, feature_sets,
                 reference=(REFERENCE_CUTOFF, REFERENCE_FEATURES)):
        self.splits = splits                  # {cutoff: Split}
        self.feature_sets = feature_sets      # {name: [columns]}
        self.reference = reference

    @staticmethod
    def score(model, cols, fit_df, eval_df):
        """Fit a preprocessing + model pipeline and evaluate it. Returns (pipeline, metrics)."""
        pipe = Pipeline([("prep", make_preprocessor(cols)), ("clf", model)])
        pipe.fit(fit_df[cols], fit_df[TARGET])
        pred = pipe.predict(eval_df[cols])
        proba = pipe.predict_proba(eval_df[cols])[:, 1] if hasattr(pipe, "predict_proba") else None
        return pipe, evaluate(eval_df[TARGET], pred, proba)

    def run(self, name, model, param, values, feature_sets=None):
        """Tune `param` over `values` in every cell and return an ExperimentResult."""
        result = ExperimentResult(name=name, param=param, rows=[])
        for cut, split in self.splits.items():
            train_val = pd.concat([split.train, split.val], ignore_index=True)
            for fs, cols in (feature_sets or self.feature_sets).items():
                val_f1 = {}
                for v in values:
                    m = clone(model).set_params(**{param: v})
                    val_f1[v] = self.score(m, cols, split.train, split.val)[1]["macro_f1"]
                best = max(val_f1, key=val_f1.get)

                m = clone(model).set_params(**{param: best})
                pipe, scores = self.score(m, cols, train_val, split.test)
                result.rows.append({"model": name, "cutoff": cut, "feature_set": fs,
                                    "best_param": f"{param}={best}", **scores})
                if (cut, fs) == self.reference:
                    result.curve, result.pipeline = val_f1, pipe
        return result


def best_by_validation(results):
    """The result with the highest validation macro-F1 at the reference setting."""
    return max(results, key=lambda r: r.best_validation_score)
