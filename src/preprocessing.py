from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Target"

FEATURE_COLUMNS = [
	"Marital status",
	"Application mode",
	"Application order",
	"Course",
	"Daytime/evening attendance",
	"Previous qualification",
	"Previous qualification (grade)",
	"Mother's qualification",
	"Father's qualification",
	"Mother's occupation",
	"Father's occupation",
	"Admission grade",
	"Displaced",
	"Debtor",
	"Tuition fees up to date",
	"Gender",
	"Scholarship holder",
	"Age at enrollment",
	"Curricular units 1st sem (credited)",
	"Curricular units 1st sem (enrolled)",
	"Curricular units 1st sem (evaluations)",
	"Curricular units 1st sem (approved)",
	"Curricular units 1st sem (grade)",
	"Curricular units 1st sem (without evaluations)",
	"Curricular units 2nd sem (credited)",
	"Curricular units 2nd sem (enrolled)",
	"Curricular units 2nd sem (evaluations)",
	"Curricular units 2nd sem (approved)",
	"Curricular units 2nd sem (grade)",
	"Curricular units 2nd sem (without evaluations)",
	"Unemployment rate",
	"Inflation rate",
	"GDP",
]

CATEGORICAL_FEATURES = [
	"Marital status",
	"Application mode",
	"Course",
	"Daytime/evening attendance",
	"Previous qualification",
	"Mother's qualification",
	"Father's qualification",
	"Mother's occupation",
	"Father's occupation",
	"Displaced",
	"Debtor",
	"Tuition fees up to date",
	"Gender",
	"Scholarship holder",
]

NUMERIC_FEATURES = [
	column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES
]


def load_dataset(csv_path: str | Path, *, separator: str = ";") -> pd.DataFrame:
	"""Load the raw dataset used by the training pipeline."""

	return pd.read_csv(csv_path, sep=separator)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
	"""Normalize column names and remove obvious duplicates or empty markers."""

	cleaned = df.copy()
	cleaned.columns = [str(column).strip() for column in cleaned.columns]
	cleaned = cleaned.replace({"": pd.NA, " ": pd.NA, "?": pd.NA})
	cleaned = cleaned.drop_duplicates().reset_index(drop=True)

	if TARGET_COLUMN in cleaned.columns:
		cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype("string").str.strip()

	return cleaned


def split_features_target(
	df: pd.DataFrame,
	*,
	feature_columns: Sequence[str] = FEATURE_COLUMNS,
	target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
	"""Return the feature matrix and target series in the expected column order."""

	missing_features = [column for column in feature_columns if column not in df.columns]
	if missing_features:
		raise ValueError(f"Faltan columnas de entrada: {', '.join(missing_features)}")

	if target_column not in df.columns:
		raise ValueError(f"No se encontró la columna objetivo '{target_column}'.")

	X = df.loc[:, list(feature_columns)].copy()
	y = df.loc[:, target_column].copy()
	return X, y


def get_feature_groups(
	feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> tuple[list[str], list[str]]:
	"""Split a feature list into numeric and categorical groups."""

	numeric_features = [column for column in feature_columns if column in NUMERIC_FEATURES]
	categorical_features = [column for column in feature_columns if column in CATEGORICAL_FEATURES]
	return numeric_features, categorical_features


def make_one_hot_encoder() -> OneHotEncoder:
	"""Create a OneHotEncoder compatible with the installed scikit-learn version."""

	try:
		return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	except TypeError:
		return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
	feature_columns: Sequence[str] = FEATURE_COLUMNS,
	*,
	numeric_features: Sequence[str] | None = None,
	categorical_features: Sequence[str] | None = None,
) -> ColumnTransformer:
	"""Build the preprocessing transformer without fitting it yet."""

	if numeric_features is None or categorical_features is None:
		inferred_numeric, inferred_categorical = get_feature_groups(feature_columns)
		numeric_features = inferred_numeric if numeric_features is None else list(numeric_features)
		categorical_features = inferred_categorical if categorical_features is None else list(categorical_features)

	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)

	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("encoder", make_one_hot_encoder()),
		]
	)

	return ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, list(numeric_features)),
			("cat", categorical_pipeline, list(categorical_features)),
		],
		remainder="drop",
		verbose_feature_names_out=True,
	)


def prepare_training_frame(
	csv_path: str | Path,
	*,
	feature_columns: Sequence[str] = FEATURE_COLUMNS,
	target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
	"""Convenience helper to load, clean and split the dataset."""

	dataset = clean_dataset(load_dataset(csv_path))
	return split_features_target(dataset, feature_columns=feature_columns, target_column=target_column)
