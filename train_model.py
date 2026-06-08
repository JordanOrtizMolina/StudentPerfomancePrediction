from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, classification_report
)

from src.preprocessing import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
    load_dataset, clean_dataset, build_preprocessor_lr
)
from src.models import build_model_specs
from src.evaluation import classification_metrics


# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/data.csv")
MODEL_DIR = Path("model")


# ---------------------------------------------------------------------------
# Argumentos de línea de comandos
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa los modelos de predicción de abandono escolar."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="Proporción del conjunto de prueba (default 0.15).")
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="Proporción del conjunto de validación (default 0.15).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Preparación de datos
# ---------------------------------------------------------------------------
def load_and_clean(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep=";")
    df.columns = df.columns.str.strip()
    if 'COLS_ELIMINAR' in globals():
        df = df.drop(columns=COLS_ELIMINAR)
    return df


def split_data(
    df: pd.DataFrame,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    temp_size = test_size + val_size          # 0.30

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_size,
        random_state=random_state,
        stratify=y,
    )

    # 50 % de temp = val, 50 % = test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Función principal de entrenamiento
# ---------------------------------------------------------------------------
def train_and_select_best_model(
    data_path: Path,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    output_dir: Path = MODEL_DIR,
) -> dict:
    # ------------------------------------------------------------------
    # 1. Carga y limpieza
    # ------------------------------------------------------------------
    df = clean_dataset(load_dataset(data_path))
    print(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas tras limpieza.")

    # ------------------------------------------------------------------
    # 2. División 70 / 15 / 15
    # ------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size=test_size, val_size=val_size, random_state=random_state
    )
    print(
        f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}"
    )

    # ------------------------------------------------------------------
    # 3. Preprocesadores y transformaciones
    # ------------------------------------------------------------------
    preprocessor_lr   = build_preprocessor_lr()

    X_train_lr  = preprocessor_lr.fit_transform(X_train)
    X_val_lr    = preprocessor_lr.transform(X_val)
    X_test_lr   = preprocessor_lr.transform(X_test)

    le = None

    # ------------------------------------------------------------------
    # 4. Configuración de GridSearchCV
    # ------------------------------------------------------------------
    specs = build_model_specs(random_state=random_state)
    lr_spec = next(s for s in specs if s.name == "logistic_regression")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # ------------------------------------------------------------------
    # 5. Entrenamiento y ajuste — Regresión Logística
    # ------------------------------------------------------------------
    print("\n[1/1] Ajustando Regresión Logística...")
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
    print(f"  Mejores parámetros: {grid_lr.best_params_}")
    print(f"  F1 macro CV: {grid_lr.best_score_:.4f}")

    # ------------------------------------------------------------------
    # 8. Predicciones sobre validación con modelos tuned
    # ------------------------------------------------------------------
    y_pred_lr_val  = lr_tuned.predict(X_val_lr)
    metrics_lr_val  = classification_metrics(y_val, y_pred_lr_val)

    # ------------------------------------------------------------------
    # 9. Verificación de overfitting (train vs. val), igual que el notebook
    # ------------------------------------------------------------------
    y_pred_lr_train  = lr_tuned.predict(X_train_lr)

    def overfitting_state(diff: float) -> str:
        if diff > 0.15:
            return "Severo"
        if diff > 0.05:
            return "Moderado"
        return "Aceptable"

    print("\n=== Verificación de Overfitting ===")
    print(f"{'Modelo':<20} {'F1 Train':>10} {'F1 Val':>10} {'Diferencia':>12} {'Estado':>12}")
    print("-" * 67)
    for nombre, y_tr, y_pr_tr, y_v, y_pr_v in [
        ("Logistic Regression", y_train, y_pred_lr_train,  y_val, y_pred_lr_val),
    ]:
        f1_tr = f1_score(y_tr, y_pr_tr, average="macro", zero_division=0)
        f1_vl = f1_score(y_v, y_pr_v, average="macro", zero_division=0)
        diff  = f1_tr - f1_vl
        print(f"{nombre:<20} {f1_tr:>10.4f} {f1_vl:>10.4f} {diff:>12.4f} {overfitting_state(diff):>12}")

    # ------------------------------------------------------------------
    # 10. Tabla comparativa sobre validación
    # ------------------------------------------------------------------
    comparison = pd.DataFrame([
        {"model": "Logistic Regression", "cv_best_score": grid_lr.best_score_,  **metrics_lr_val},
    ])

    print("\n=== Comparación sobre validación ===")
    print(comparison.to_string(index=False))

    # ------------------------------------------------------------------
    # 11. Selección del mejor modelo por F1 macro sobre validación
    # ------------------------------------------------------------------
    best_name = "Logistic Regression"
    model_map = {
        "Logistic Regression": (lr_tuned,  preprocessor_lr,   grid_lr.best_params_,  metrics_lr_val),
    }
    best_estimator, best_preprocessor, best_params, best_val_metrics = model_map[best_name]

    print(f"\nModelo seleccionado: {best_name}")
    print(f"F1 macro en validación: {best_val_metrics['f1_macro']:.4f}")

    # ------------------------------------------------------------------
    # 12. Evaluación final sobre conjunto de prueba (una sola vez)
    # ------------------------------------------------------------------
    print("\n=== Evaluación Final sobre Conjunto de Prueba ===")
    X_test_final  = X_test_lr
    y_pred_test   = best_estimator.predict(X_test_final)

    test_metrics = classification_metrics(y_test, y_pred_test)
    print(classification_report(y_test, y_pred_test, zero_division=0))
    print(f"F1 macro final: {test_metrics['f1_macro']:.4f}")

    # ------------------------------------------------------------------
    # 13. Persistencia del mejor modelo y su preprocesador
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path      = output_dir / "best_model.pkl"
    preprocessor_path    = output_dir / "preprocessor.pkl"
    label_encoder_path   = output_dir / "label_encoder.pkl"

    joblib.dump(best_estimator,   best_model_path)
    joblib.dump(best_preprocessor, preprocessor_path)
    joblib.dump(le, label_encoder_path)   # necesario para decodificar predicciones de XGBoost

    print(f"\nModelo guardado en:       {best_model_path}")
    print(f"Preprocesador guardado en: {preprocessor_path}")
    print(f"LabelEncoder guardado en:  {label_encoder_path}")

    return {
        "comparison":       comparison,
        "best_model":       best_name,
        "best_params":      best_params,
        "val_metrics":      best_val_metrics,
        "test_metrics":     test_metrics,
        "model_path":       best_model_path,
        "preprocessor_path":preprocessor_path,
        "label_encoder_path": label_encoder_path,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
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
    print(f"Mejor modelo:            {results['best_model']}")
    print(f"Mejores hiperparámetros: {results['best_params']}")
    print(f"F1 macro (validación):   {results['val_metrics']['f1_macro']:.4f}")
    print(f"F1 macro (prueba):       {results['test_metrics']['f1_macro']:.4f}")
    print(f"Ruta del modelo:         {results['model_path']}")
    print(f"Ruta del preprocesador:  {results['preprocessor_path']}")


if __name__ == "__main__":
    main()