from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)


def classification_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float]:
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
		"recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
		"f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
		"precision_weighted": float(
			precision_score(y_true, y_pred, average="weighted", zero_division=0)
		),
		"recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
		"f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
	}


def confusion_matrix_frame(
	y_true: Iterable[str],
	y_pred: Iterable[str],
	labels: list[str],
) -> pd.DataFrame:
	matrix = confusion_matrix(y_true, y_pred, labels=labels)
	row_labels = [f"actual_{label}" for label in labels]
	column_labels = [f"pred_{label}" for label in labels]
	return pd.DataFrame(matrix, index=row_labels, columns=column_labels)


def feature_importance_frame(
	model: object,
	feature_names: Iterable[str],
) -> pd.DataFrame:
	names = list(feature_names)
	if hasattr(model, "feature_importances_"):
		values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
	elif hasattr(model, "coef_"):
		coefficients = np.asarray(getattr(model, "coef_"), dtype=float)
		if coefficients.ndim == 1:
			values = np.abs(coefficients)
		else:
			values = np.mean(np.abs(coefficients), axis=0)
	else:
		return pd.DataFrame(columns=["feature", "importance"])

	if values.shape[0] != len(names):
		raise ValueError(
			"La cantidad de importancias no coincide con el número de nombres de variables."
		)

	frame = pd.DataFrame({"feature": names, "importance": values})
	return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def evaluate_predictions(
	y_true: Iterable[str],
	y_pred: Iterable[str],
	labels: list[str],
) -> dict[str, object]:
	metrics = classification_metrics(y_true, y_pred)
	matrix = confusion_matrix_frame(y_true, y_pred, labels=labels)
	return {"metrics": metrics, "confusion_matrix": matrix}
