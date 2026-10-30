"""End-to-end run: raw CSVs -> point-in-time tables -> experiment grid -> results."""

from dataclasses import dataclass

import pandas as pd

from .config import CUTOFF_DAYS, REFERENCE_CUTOFF, REFERENCE_FEATURES, TARGET
from .data import OULADData, build_enrolments, prepare_assessments, prepare_vle
from .experiment import ExperimentRunner, best_by_validation
from .features import FeatureBuilder, feature_sets
from .models import BASELINE, MODELS
from .modelling import make_split


@dataclass
class PipelineOutput:
    tables: dict          # {cutoff: wide table}
    summary: pd.DataFrame # rows, exclusions and at-risk share per cutoff
    feature_sets: dict
    splits: dict          # {cutoff: Split}
    baseline: object      # ExperimentResult
    results: list         # [ExperimentResult] for LR, DT, RF, MLP

    @property
    def scores(self):
        """All test scores (baseline + models) in one table."""
        return pd.concat([self.baseline.table] + [r.table for r in self.results], ignore_index=True)

    def result(self, name):
        return next(r for r in self.results if r.name == name)

    @property
    def best_classical(self):
        return best_by_validation([r for r in self.results if r.name != "MLP"])

    def reference_test(self):
        """Test set and feature columns of the reference setting (day 24, all features)."""
        return self.splits[REFERENCE_CUTOFF].test, self.feature_sets[REFERENCE_FEATURES]


def build_tables(data_dir="data", cutoffs=CUTOFF_DAYS):
    """Load the raw data and build one point-in-time table per cutoff."""
    data = OULADData.load(data_dir)
    enrolments = build_enrolments(data)
    clicks, activity_types = prepare_vle(data)
    asm = prepare_assessments(data)
    return FeatureBuilder(enrolments, clicks, asm, activity_types).build_all(cutoffs)


def run_pipeline(data_dir="data", cutoffs=CUTOFF_DAYS, models=MODELS, verbose=True):
    """Run the full study and return every table, split, model and score."""
    tables, summary = build_tables(data_dir, cutoffs)
    fsets = feature_sets(tables.get(REFERENCE_CUTOFF, tables[cutoffs[0]]))
    splits = {cut: make_split(tables[cut]) for cut in cutoffs}
    runner = ExperimentRunner(splits, fsets)

    baseline = runner.run(BASELINE.name, BASELINE.estimator, BASELINE.param, BASELINE.grid,
                          feature_sets={"—": fsets["background"]})
    results = []
    for spec in models:
        if verbose:
            print(f"Running {spec.name} ...", flush=True)
        results.append(runner.run(spec.name, spec.estimator, spec.param, spec.grid))
    return PipelineOutput(tables, summary, fsets, splits, baseline, results)


def recall_by_module(test, pred):
    """Per-module recall, next to the share of students with no assessment submitted yet."""
    ea = test[["code_module", "has_asm_activity", TARGET]].copy()
    ea["pred"] = pred
    return ea.groupby("code_module").apply(lambda g: pd.Series({
        "at_risk_n": int(g[TARGET].sum()),
        "recall": g.loc[g[TARGET] == 1, "pred"].mean(),
        "no_submission_%": (g["has_asm_activity"] == 0).mean() * 100,
    })).round(3)
