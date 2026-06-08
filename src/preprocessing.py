from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Target"

COLS_TO_DROP = [
    "Educational special needs",
    "Nacionality",
    "International",
    "Inflation rate",
    "GDP",
    "Unemployment rate",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (without evaluations)",
]

NUMERIC_FEATURES: list[str] = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]

CATEGORICAL_FEATURES: list[str] = [
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Marital status",
    "Displaced",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
]

FEATURE_COLUMNS: tuple[str, ...] = tuple(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def load_dataset(csv_path: str | Path, *, separator: str = ";") -> pd.DataFrame:
    return pd.read_csv(csv_path, sep=separator)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    cols_present = [c for c in COLS_TO_DROP if c in cleaned.columns]
    if cols_present:
        cleaned = cleaned.drop(columns=cols_present)

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype("string").str.strip()

    return cleaned


def split_features_target(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de entrada: {', '.join(missing)}")
    if target_column not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{target_column}'.")

    X = df.loc[:, list(feature_columns)].copy()
    y = df.loc[:, target_column].copy()
    return X, y


def prepare_training_frame(
    csv_path: str | Path,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    dataset = clean_dataset(load_dataset(csv_path))
    return split_features_target(
        dataset, feature_columns=feature_columns, target_column=target_column
    )


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor_lr(
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(numeric_features)),
            ("cat", _make_ohe(), list(categorical_features)),
        ],
        remainder="drop",
    )