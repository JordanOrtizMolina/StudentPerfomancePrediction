from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ModelSpec:
	"""Container for an estimator and its search space."""

	name: str
	estimator: object
	param_grid: dict[str, list[object]]


def build_model_specs(*, random_state: int = 42) -> list[ModelSpec]:
	"""Return the model candidates used in training and comparison."""

	logistic_regression = LogisticRegression(
		max_iter=3000,
		solver="lbfgs",
	)

	random_forest = RandomForestClassifier(
		random_state=random_state,
		n_jobs=1,
	)

	return [
		ModelSpec(
			name="logistic_regression",
			estimator=logistic_regression,
			param_grid={
				"C": [0.5, 1.0, 5.0],
				"class_weight": [None, "balanced"],
			},
		),
		ModelSpec(
			name="random_forest",
			estimator=random_forest,
			param_grid={
				"n_estimators": [200],
				"max_depth": [None, 10],
				"min_samples_split": [2],
				"min_samples_leaf": [1, 2],
				"class_weight": [None, "balanced"],
			},
		),
	]


def get_scoring_metric() -> str:
	"""Primary selection metric for model comparison."""

	return "f1_macro"
