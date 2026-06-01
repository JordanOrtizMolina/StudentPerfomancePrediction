from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS


DEFAULT_MODEL_PATH = Path("model/best_model.pkl")
DEFAULT_PREPROCESSOR_PATH = Path("model/preprocessor.pkl")


def load_model_artifacts(
	model_path: str | Path = DEFAULT_MODEL_PATH,
	preprocessor_path: str | Path = DEFAULT_PREPROCESSOR_PATH,
) -> tuple[Any, Any]:
	"""Load the fitted estimator and preprocessor saved by the training script."""

	model_file = Path(model_path)
	preprocessor_file = Path(preprocessor_path)

	if not model_file.exists():
		raise FileNotFoundError(f"No se encontró el archivo del modelo: {model_file}")
	if not preprocessor_file.exists():
		raise FileNotFoundError(f"No se encontró el archivo de preprocesamiento: {preprocessor_file}")

	model = joblib.load(model_file)
	preprocessor = joblib.load(preprocessor_file)
	return model, preprocessor


def prepare_input_data(
	values: dict[str, Any],
	feature_columns: list[str] | tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
	"""Create a single-row DataFrame in the exact feature order expected by the model."""

	normalized_values = {str(column).strip(): value for column, value in values.items()}
	missing_columns = [column for column in feature_columns if column not in normalized_values]
	if missing_columns:
		raise ValueError(f"Faltan variables requeridas: {', '.join(missing_columns)}")

	return pd.DataFrame([{column: normalized_values[column] for column in feature_columns}])


def predict_student_status(
	model: Any,
	preprocessor: Any,
	input_data: pd.DataFrame | dict[str, Any],
	feature_columns: list[str] | tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[str, dict[str, float], float]:
	"""Transform the input data and generate the class prediction plus class probabilities."""

	if isinstance(input_data, pd.DataFrame):
		data_frame = input_data.copy()
	else:
		data_frame = prepare_input_data(input_data, feature_columns=feature_columns)

	transformed = preprocessor.transform(data_frame)
	prediction = str(model.predict(transformed)[0])

	probabilities = {prediction: 1.0}
	confidence = 1.0

	if hasattr(model, "predict_proba"):
		probabilities_array = model.predict_proba(transformed)[0]
		model_classes = [str(label) for label in getattr(model, "classes_", [prediction])]
		probabilities = {
			label: float(probability)
			for label, probability in zip(model_classes, probabilities_array)
		}
		confidence = float(probabilities.get(prediction, max(probabilities_array)))

	return prediction, probabilities, confidence
