from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict[str, list[object]]


def build_model_specs(*, random_state: int = 42) -> list[ModelSpec]:

    logistic_regression = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )

    random_forest = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    xgboost = XGBClassifier(
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    return [
        ModelSpec(
            name="logistic_regression",
            estimator=logistic_regression,
            param_grid={
                "C":       [0.01, 0.1, 1, 10, 100],
                "solver":  ["lbfgs", "saga"],
            },
        ),
        ModelSpec(
            name="random_forest",
            estimator=random_forest,
            param_grid={
                "n_estimators":     [100, 200, 300],
                "max_depth":        [None, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
        ),
        ModelSpec(
            name="xgboost",
            estimator=xgboost,
            param_grid={
                "n_estimators":    [100, 200],
                "max_depth":       [3, 5, 7],
                "learning_rate":   [0.05, 0.1],
                "subsample":       [0.8, 1.0],
                "colsample_bytree": [0.8],
            },
        ),
    ]


def get_scoring_metric() -> str:
    return "f1_macro"