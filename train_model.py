from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.evaluation import classification_metrics, confusion_matrix_frame, feature_importance_frame
from src.models import build_model_specs, get_scoring_metric
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor, prepare_training_frame


DATA_PATH = Path("data/data.csv")
MODEL_DIR = Path("model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the student performance models.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH, help="Ruta al dataset CSV.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción del conjunto de prueba.")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla para el particionado y modelos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directorio donde se guardarán los artefactos entrenados.",
    )
    return parser.parse_args()


def train_and_select_best_model(
    data_path: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: Path = MODEL_DIR,
) -> dict[str, object]:
    """Train the candidate models, compare them and persist the best artifact set."""

    X, y = prepare_training_frame(data_path, feature_columns=FEATURE_COLUMNS, target_column=TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(FEATURE_COLUMNS)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    transformed_feature_names = preprocessor.get_feature_names_out()

    labels = sorted(y.unique().tolist())
    scoring = get_scoring_metric()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    evaluation_rows: list[dict[str, object]] = []
    best_result: dict[str, object] | None = None

    for spec in build_model_specs(random_state=random_state):
        search = GridSearchCV(
            estimator=spec.estimator,
            param_grid=spec.param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train_transformed, y_train)

        test_predictions = search.best_estimator_.predict(X_test_transformed)
        test_metrics = classification_metrics(y_test, test_predictions)
        test_confusion_matrix = confusion_matrix_frame(y_test, test_predictions, labels=labels)
        importances = feature_importance_frame(search.best_estimator_, transformed_feature_names)

        row: dict[str, object] = {
            "model": spec.name,
            "best_params": search.best_params_,
            "cv_best_score": float(search.best_score_),
            **test_metrics,
            "confusion_matrix": test_confusion_matrix,
            "feature_importance": importances,
            "estimator": search.best_estimator_,
        }
        evaluation_rows.append(row)

        if best_result is None or test_metrics["f1_macro"] > best_result["metrics"]["f1_macro"]:
            best_result = {
                "model": spec.name,
                "estimator": search.best_estimator_,
                "metrics": test_metrics,
                "best_params": search.best_params_,
                "cv_best_score": float(search.best_score_),
            }

    if best_result is None:
        raise RuntimeError("No se pudo seleccionar un modelo ganador.")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pkl"
    preprocessor_path = output_dir / "preprocessor.pkl"
    joblib.dump(best_result["estimator"], best_model_path)
    joblib.dump(preprocessor, preprocessor_path)

    comparison_frame = pd.DataFrame(
        [
            {
                "model": row["model"],
                "cv_best_score": row["cv_best_score"],
                "accuracy": row["accuracy"],
                "precision_macro": row["precision_macro"],
                "recall_macro": row["recall_macro"],
                "f1_macro": row["f1_macro"],
            }
            for row in evaluation_rows
        ]
    ).sort_values("f1_macro", ascending=False)

    return {
        "comparison": comparison_frame,
        "evaluations": evaluation_rows,
        "best_model": best_result,
        "model_path": best_model_path,
        "preprocessor_path": preprocessor_path,
    }


def main() -> None:
    args = parse_args()
    results = train_and_select_best_model(
        args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )

    comparison = results["comparison"]
    best_model = results["best_model"]

    print("Comparación de modelos:")
    print(comparison.to_string(index=False))
    print()
    print(f"Mejor modelo: {best_model['model']}")
    print(f"Mejores hiperparámetros: {best_model['best_params']}")
    print(f"Ruta del modelo: {results['model_path']}")
    print(f"Ruta del preprocesador: {results['preprocessor_path']}")


if __name__ == "__main__":
    main()