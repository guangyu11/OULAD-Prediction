"""Reproduce every result and figure from the raw data.

Usage (from the repository root, with the OULAD CSVs in data/):
    python scripts/run_experiments.py
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oulad import recall_by_module, run_pipeline  # noqa: E402
from oulad.config import TARGET  # noqa: E402
from oulad.plotting import (MODEL_LABELS, plot_confusion, plot_results_grid,  # noqa: E402
                            plot_tuning_curves)


def main():
    start = time.time()
    out = run_pipeline(ROOT / "data")
    results_dir, figures_dir = ROOT / "results", ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    scores = out.scores
    scores.to_csv(results_dir / "test_scores.csv", index=False)
    out.summary.to_csv(results_dir / "cutoff_summary.csv", index=False)

    plot_results_grid(scores).savefig(figures_dir / "results_by_cutoff.png", dpi=150)
    plot_tuning_curves(out.results).savefig(figures_dir / "tuning_curves.png", dpi=150)

    # Error analysis at the reference setting (day 24, all features).
    test, cols = out.reference_test()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, r in zip(axes, [out.best_classical, out.result("MLP")]):
        pred = (r.pipeline.predict_proba(test[cols])[:, 1] >= 0.5).astype(int)
        plot_confusion(confusion_matrix(test[TARGET], pred), MODEL_LABELS[r.name], ax=ax)
    fig.suptitle("Confusion matrices, test set, day 24, all features", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrices.png", dpi=150)

    mlp_pred = (out.result("MLP").pipeline.predict_proba(test[cols])[:, 1] >= 0.5).astype(int)
    by_module = recall_by_module(test, mlp_pred)
    by_module.to_csv(results_dir / "mlp_recall_by_module.csv")

    print(out.summary.to_string(index=False))
    print(scores.to_string(index=False))
    print(by_module.to_string())
    print(f"Best classical model (validation): {out.best_classical.name}")
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
