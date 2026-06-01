# Student Performance Prediction

Sistema de clasificación académica que predice si un estudiante va a desertar, mantenerse matriculado o graduarse, a partir de información personal, académica, familiar, financiera y macroeconómica. Desarrollado como proyecto final del curso de Inteligencia Artificial, 2026.

## Estructura del proyecto

```text
proyecto/
├── app.py                  # Interfaz Streamlit
├── train_model.py          # Script de entrenamiento
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Limpieza y preprocesador
│   ├── models.py           # Modelos y grillas de hiperparámetros
│   ├── evaluation.py       # Métricas y análisis
│   └── prediction.py       # Carga de artefactos y predicción
├── model/                  # .pkl generados al entrenar
├── data/
│   └── data.csv
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Entrenamiento

Carga `data/data.csv`, aplica el preprocesador, compara Regresión Logística y Random Forest con validación cruzada estratificada, y guarda el mejor modelo en `model/`.

```bash
python train_model.py
```

Parámetros opcionales:

```bash
python train_model.py --data-path data/data.csv --test-size 0.2 --random-state 42
```

## Aplicación

Una vez entrenado el modelo, levantá la interfaz con:

```bash
streamlit run app.py
```

La app permite ingresar los datos de un estudiante en un formulario de siete secciones y obtener la predicción en tiempo real junto con las probabilidades por clase.

## Módulos

`src/preprocessing.py` — carga, limpieza, separación de columnas numéricas y categóricas, y construcción del `ColumnTransformer` con pipelines de imputación, escalamiento y codificación.

`src/models.py` — definición de los modelos candidatos y sus grillas de hiperparámetros para `GridSearchCV`.

`src/evaluation.py` — cálculo de accuracy, precision, recall, F1, matriz de confusión e importancia de variables.

`src/prediction.py` — carga de artefactos `.pkl` y función de predicción reutilizada por `app.py`.

## Autores

Jordan Ortiz Molina · Yenifer Mata Flores · Deyaneira Altamirano Cordero
