"""The models compared, each with the one hyperparameter tuned on validation data."""

from dataclasses import dataclass

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import SEED


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param: str
    grid: tuple
    note: str = ""


BASELINE = ModelSpec(
    "Baseline", DummyClassifier(random_state=SEED), "strategy", ("most_frequent",),
    "Always predicts the majority class (not at risk); the floor every model must beat.")

MODELS = [
    ModelSpec("LogisticRegression",
              LogisticRegression(random_state=SEED, max_iter=2000),
              "C", (0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10),
              "High bias, low variance: a linear boundary with L2 regularisation."),
    ModelSpec("DecisionTree",
              DecisionTreeClassifier(random_state=SEED),
              "max_depth", (2, 4, 6, 8, 10, 15, 20, None),
              "Low bias, high variance: a single tree can memorise the training data."),
    ModelSpec("RandomForest",
              RandomForestClassifier(random_state=SEED, n_estimators=100, n_jobs=1),
              "min_samples_leaf", (1, 2, 5, 10, 20),
              "Low bias, reduced variance: bagging averages many decorrelated trees."),
    ModelSpec("MLP",
              MLPClassifier(alpha=1e-3, early_stopping=True, max_iter=400, random_state=SEED),
              "hidden_layer_sizes", ((16,), (32,), (64,), (128,), (64, 32)),
              "Most flexible family; held back by an L2 penalty and early stopping."),
]
