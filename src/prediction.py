from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS


DEFAULT_MODEL_PATH        = Path("model/best_model.pkl")
DEFAULT_PREPROCESSOR_PATH = Path("model/preprocessor.pkl")


def load_model_artifacts(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    preprocessor_path: str | Path = DEFAULT_PREPROCESSOR_PATH,
) -> tuple[Any, Any]:
    model_file        = Path(model_path)
    preprocessor_file = Path(preprocessor_path)

    if not model_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {model_file}")
    if not preprocessor_file.exists():
        raise FileNotFoundError(f"No se encontró el preprocesador: {preprocessor_file}")

    return joblib.load(model_file), joblib.load(preprocessor_file)


def prepare_input_data(
    values: dict[str, Any],
    feature_columns: list[str] | tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    normalized_values = {str(k).strip(): v for k, v in values.items()}
    clean_features    = [str(c).strip() for c in feature_columns]

    missing = [c for c in clean_features if c not in normalized_values]
    if missing:
        raise ValueError(f"Faltan variables requeridas: {', '.join(missing)}")

    ordered_data = {
        orig_col: normalized_values[str(orig_col).strip()]
        for orig_col in feature_columns
    }
    return pd.DataFrame([ordered_data])


def predict_student_status(
    model: Any,
    preprocessor: Any,
    input_data: pd.DataFrame | dict[str, Any],
    feature_columns: list[str] | tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[str, dict[str, float], float]:
    if isinstance(input_data, pd.DataFrame):
        data_frame = input_data.copy()
    else:
        data_frame = prepare_input_data(input_data, feature_columns=feature_columns)

    transformed = preprocessor.transform(data_frame)
    prediction  = str(model.predict(transformed)[0])

    probabilities: dict[str, float] = {prediction: 1.0}
    confidence: float = 1.0

    if hasattr(model, "predict_proba"):
        proba_array   = model.predict_proba(transformed)[0]
        class_labels  = [str(c) for c in model.classes_]
        probabilities = {
            label: float(prob)
            for label, prob in zip(class_labels, proba_array)
        }
        confidence = float(probabilities.get(prediction, float(proba_array.max())))

    return prediction, probabilities, confidence