from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report

from src.preprocessing import (
    TARGET_COLUMN,
    load_dataset, clean_dataset, build_preprocessor_lr
)
from src.models import build_model_specs
from src.evaluation import classification_metrics


DATA_PATH = Path("data/data.csv")
MODEL_DIR = Path("model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa el modelo de predicción de abandono escolar."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    return parser.parse_args()


def split_data(
    df: pd.DataFrame,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_select_best_model(
    data_path: Path,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    output_dir: Path = MODEL_DIR,
) -> dict:
    df = clean_dataset(load_dataset(data_path))
    print(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas tras limpieza.")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size=test_size, val_size=val_size, random_state=random_state
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    preprocessor = build_preprocessor_lr()
    X_train_lr = preprocessor.fit_transform(X_train)
    X_val_lr   = preprocessor.transform(X_val)
    X_test_lr  = preprocessor.transform(X_test)

    specs = build_model_specs(random_state=random_state)
    lr_spec = next(s for s in specs if s.name == "logistic_regression")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    print("\nAjustando Regresión Logística...")
    grid_lr = GridSearchCV(
        lr_spec.estimator,
        lr_spec.param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid_lr.fit(X_train_lr, y_train)
    lr_tuned = grid_lr.best_estimator_
    print(f"Mejores parámetros: {grid_lr.best_params_}")
    print(f"F1 macro CV: {grid_lr.best_score_:.4f}")

    y_pred_val   = lr_tuned.predict(X_val_lr)
    y_pred_train = lr_tuned.predict(X_train_lr)
    val_metrics  = classification_metrics(y_val, y_pred_val)

    f1_train = f1_score(y_train, y_pred_train, average="macro", zero_division=0)
    f1_val   = f1_score(y_val,   y_pred_val,   average="macro", zero_division=0)
    diff     = f1_train - f1_val
    estado   = "Severo" if diff > 0.15 else "Moderado" if diff > 0.05 else "Aceptable"

    print("\n=== Verificación de Overfitting ===")
    print(f"F1 Train: {f1_train:.4f}  F1 Val: {f1_val:.4f}  Diferencia: {diff:.4f}  Estado: {estado}")

    print("\n=== Evaluación Final sobre Conjunto de Prueba ===")
    y_pred_test  = lr_tuned.predict(X_test_lr)
    test_metrics = classification_metrics(y_test, y_pred_test)
    print(classification_report(y_test, y_pred_test, zero_division=0))
    print(f"F1 macro final: {test_metrics['f1_macro']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path   = output_dir / "best_model.pkl"
    preprocessor_path = output_dir / "preprocessor.pkl"

    joblib.dump(lr_tuned,    best_model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"\nModelo guardado en:        {best_model_path}")
    print(f"Preprocesador guardado en: {preprocessor_path}")

    return {
        "best_params":       grid_lr.best_params_,
        "val_metrics":       val_metrics,
        "test_metrics":      test_metrics,
        "model_path":        best_model_path,
        "preprocessor_path": preprocessor_path,
    }


def main() -> None:
    args = parse_args()
    results = train_and_select_best_model(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )

    print("\n=== Resumen Final ===")
    print(f"Mejores hiperparámetros: {results['best_params']}")
    print(f"F1 macro (validación):   {results['val_metrics']['f1_macro']:.4f}")
    print(f"F1 macro (prueba):       {results['test_metrics']['f1_macro']:.4f}")
    print(f"Ruta del modelo:         {results['model_path']}")
    print(f"Ruta del preprocesador:  {results['preprocessor_path']}")


if __name__ == "__main__":
    main()