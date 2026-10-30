"""Figures for tuning, results and error analysis."""

import matplotlib.pyplot as plt

from .config import CUTOFF_DAYS

GROUP_COLOURS = {"background": "#2a78d6", "behaviour": "#eb6834", "all": "#1baf7a"}
METRIC_LABELS = {"macro_f1": "Macro-F1", "precision": "Precision (at risk)",
                 "recall": "Recall (at risk)", "roc_auc": "ROC-AUC"}
MODEL_LABELS = {"LogisticRegression": "Logistic Regression", "DecisionTree": "Decision Tree",
                "RandomForest": "Random Forest", "MLP": "MLP"}


def plot_tuning_curves(results):
    """Validation macro-F1 against each model's hyperparameter, on one shared y-axis.

    The shared scale is the point: it shows how sensitive each model is to tuning.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.2), sharey=True)
    for ax, r in zip(axes, results):
        labels = [str(v) for v in r.curve]
        best = max(r.curve, key=r.curve.get)
        ax.plot(labels, list(r.curve.values()), marker="o", color="#2a78d6")
        ax.axvline(list(r.curve).index(best), linestyle="--", color="grey",
                   label=f"chosen: {best}")
        ax.set_title(MODEL_LABELS.get(r.name, r.name), fontsize=10)
        ax.set_xlabel(r.param)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower center")
    axes[0].set_ylabel("Validation macro-F1")
    fig.suptitle("Tuning curves at day 24, all features", fontsize=11)
    fig.tight_layout()
    return fig


def plot_results_grid(table, metrics=("macro_f1", "precision", "recall")):
    """Test scores by cutoff day: one column per model, one row per metric, one line per feature group."""
    models = [m for m in MODEL_LABELS if m in set(table["model"])]
    baseline = table[table["model"] == "Baseline"].sort_values("cutoff")
    fig, axes = plt.subplots(len(metrics), len(models), figsize=(3.2 * len(models), 2.6 * len(metrics)),
                             sharex=True, sharey="row")
    for col, model in enumerate(models):
        d = table[table["model"] == model]
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            for fs, colour in GROUP_COLOURS.items():
                f = d[d["feature_set"] == fs].sort_values("cutoff")
                ax.plot(f["cutoff"], f[metric], marker="o", color=colour, label=fs)
            # The baseline never predicts "at risk", so it only has a meaningful macro-F1.
            if metric == "macro_f1":
                ax.plot(baseline["cutoff"], baseline[metric], "k--", linewidth=1, label="majority baseline")
            if row == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=10)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_xticks(CUTOFF_DAYS)
            ax.grid(alpha=0.3)
        axes[-1, col].set_xlabel("Cutoff day")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def plot_confusion(cm, title, ax=None):
    """Annotated 2x2 confusion matrix."""
    ax = ax or plt.subplots(figsize=(3.6, 3.2))[1]
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * .55 else "black")
    ax.set_xticks([0, 1], ["pred: completed", "pred: at risk"], fontsize=8)
    ax.set_yticks([0, 1], ["completed", "at risk"], fontsize=8)
    ax.set_title(title, fontsize=10)
    return ax
