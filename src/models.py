from __future__ import annotations

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression


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
    return [
        ModelSpec(
            name="logistic_regression",
            estimator=logistic_regression,
            param_grid={
                "C":      [0.01, 0.1, 1, 10, 100],
                "solver": ["lbfgs", "saga"],
            },
        ),
    ]


def get_scoring_metric() -> str:
    return "f1_macro"