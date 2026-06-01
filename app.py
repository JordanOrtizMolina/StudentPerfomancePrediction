from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "Student Performance Prediction System"
APP_SUBTITLE = "Sistema Inteligente para Prediccion del Desempeno Academico"
MODEL_PATH = Path("model/best_model.pkl")
PREPROCESSOR_PATH = Path("model/preprocessor.pkl")
CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]

FEATURE_COLUMNS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance\t",
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

SECTION_COUNTS = {
    "Informacion Personal": 4,
    "Informacion Academica de Ingreso": 7,
    "Informacion Familiar": 4,
    "Informacion Financiera": 3,
    "Rendimiento Primer Semestre": 6,
    "Rendimiento Segundo Semestre": 6,
    "Indicadores Economicos": 3,
}

RESULT_CONTENT = {
    "Dropout": {
        "risk": "High",
        "status": "Academic dropout risk",
        "accent": "#b42318",
        "description": (
            "The student presents characteristics associated with academic dropout. "
            "Early intervention and academic support are recommended."
        ),
    },
    "Enrolled": {
        "risk": "Medium",
        "status": "Active enrollment projection",
        "accent": "#b7791f",
        "description": (
            "The student is likely to remain enrolled, although continuous academic "
            "monitoring is recommended."
        ),
    },
    "Graduate": {
        "risk": "Low",
        "status": "Successful completion profile",
        "accent": "#1f7a4d",
        "description": (
            "The student presents characteristics associated with successful academic "
            "completion and graduation."
        ),
    },
}


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --navy-950: #071a2f;
                --navy-900: #0b2440;
                --navy-800: #12385f;
                --navy-700: #194b7d;
                --blue-100: #e7f0fb;
                --blue-050: #f4f8fd;
                --slate-900: #101828;
                --slate-700: #344054;
                --slate-600: #475467;
                --border: #d6e1ee;
                --surface: #ffffff;
            }

            .stApp {
                background: #f3f6fa;
                color: var(--slate-900);
            }

            .block-container {
                max-width: 1260px;
                padding: 1.6rem 2rem 2.5rem;
            }

            section[data-testid="stSidebar"] {
                background: var(--navy-950);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }

            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] span {
                color: #f8fbff;
            }

            section[data-testid="stSidebar"] div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.08);
                border-color: rgba(255, 255, 255, 0.15);
                box-shadow: none;
            }

            section[data-testid="stSidebar"] div[data-testid="stMetric"] label,
            section[data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                color: #ffffff;
            }

            .hero {
                background: var(--navy-900);
                border: 1px solid var(--navy-800);
                border-radius: 8px;
                padding: 1.4rem 1.5rem;
                margin-bottom: 1rem;
                color: #ffffff;
            }

            .hero h1 {
                margin: 0;
                color: #ffffff;
                font-size: 2.15rem;
                line-height: 1.16;
                letter-spacing: 0;
            }

            .hero p {
                margin: 0.45rem 0 0;
                color: #d9e8f8;
                font-size: 1.02rem;
                line-height: 1.5;
            }

            .dashboard-card,
            .section-card,
            .result-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 8px;
                box-shadow: 0 10px 24px rgba(16, 40, 71, 0.07);
            }

            .dashboard-card {
                padding: 1.05rem 1.1rem;
                min-height: 138px;
                border-top: 4px solid var(--navy-700);
            }

            .dashboard-card h3 {
                color: var(--navy-900);
                font-size: 1rem;
                margin: 0 0 0.45rem;
                letter-spacing: 0;
            }

            .dashboard-card p {
                color: var(--slate-600);
                line-height: 1.5;
                margin: 0;
            }

            .section-card {
                padding: 1rem 1.1rem;
                margin: 0.5rem 0 1rem;
                background: #fbfdff;
            }

            .section-card h3 {
                color: var(--navy-900);
                margin: 0 0 0.2rem;
                font-size: 1.05rem;
                letter-spacing: 0;
            }

            .section-card p {
                color: var(--slate-600);
                margin: 0;
                line-height: 1.45;
            }

            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 0.75rem 0.9rem;
                box-shadow: 0 8px 18px rgba(16, 40, 71, 0.06);
            }

            div[data-testid="stMetric"] label {
                color: var(--slate-600);
                font-weight: 650;
            }

            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                color: var(--navy-900);
                font-weight: 800;
            }

            div[data-testid="stTabs"] button {
                color: var(--navy-900);
                font-weight: 700;
            }

            div[data-baseweb="tab-highlight"] {
                background-color: var(--navy-700);
            }

            label, .stNumberInput label, .stSelectbox label {
                color: var(--slate-900) !important;
                font-weight: 650;
            }

            .stButton > button {
                background: var(--navy-700);
                border: 1px solid var(--navy-700);
                border-radius: 7px;
                color: #ffffff;
                font-weight: 800;
                min-height: 2.8rem;
                width: 100%;
            }

            .stButton > button:hover {
                background: var(--navy-800);
                border-color: var(--navy-800);
                color: #ffffff;
            }

            .result-card {
                padding: 1.2rem 1.25rem;
                border-left: 7px solid var(--navy-700);
                margin-top: 0.75rem;
            }

            .result-card h2 {
                color: var(--navy-900);
                margin: 0;
                font-size: 1.85rem;
                letter-spacing: 0;
            }

            .result-kicker {
                color: var(--slate-600);
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            .risk-pill {
                display: inline-block;
                background: var(--blue-100);
                border: 1px solid var(--border);
                border-radius: 6px;
                color: var(--navy-900);
                font-weight: 800;
                margin: 0.55rem 0 0.65rem;
                padding: 0.28rem 0.62rem;
            }

            .result-card p {
                color: var(--slate-700);
                line-height: 1.55;
                margin: 0;
            }

            .prob-row {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 0.65rem 0.8rem;
                margin-bottom: 0.55rem;
            }

            .prob-label {
                color: var(--navy-900);
                font-weight: 800;
                margin-bottom: 0.2rem;
            }

            .prob-value {
                color: var(--slate-600);
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[Any, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def int_input(label: str, value: int, min_value: int, max_value: int) -> int:
    return int(st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=1))


def float_input(label: str, value: float, min_value: float, max_value: float, step: float = 0.1) -> float:
    return float(st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step))


def yes_no_input(label: str, default: int = 0) -> int:
    selected = st.selectbox(
        label,
        options=[0, 1],
        index=1 if default else 0,
        format_func=lambda value: "Yes" if value == 1 else "No",
    )
    return int(selected)


def section_header(title: str, description: str, count: int) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            <p>{description} This section contains {count} model variables.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_input_form() -> dict[str, Any]:
    values: dict[str, Any] = {}

    tabs = st.tabs(
        [
            "Personal",
            "Ingreso",
            "Familiar",
            "Financiera",
            "1er semestre",
            "2do semestre",
            "Economicos",
        ]
    )

    with tabs[0]:
        section_header(
            "Informacion Personal",
            "Datos demograficos y condiciones generales del estudiante.",
            SECTION_COUNTS["Informacion Personal"],
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            values["Marital status"] = int_input("Marital status", 1, 1, 6)
        with col2:
            values["Gender"] = st.selectbox(
                "Gender",
                options=[0, 1],
                format_func=lambda value: "Female" if value == 0 else "Male",
            )
        with col3:
            values["Age at enrollment"] = int_input("Age at enrollment", 20, 15, 80)
        with col4:
            values["Displaced"] = yes_no_input("Displaced", 0)

    with tabs[1]:
        section_header(
            "Informacion Academica de Ingreso",
            "Variables relacionadas con el proceso de admision y antecedentes academicos.",
            SECTION_COUNTS["Informacion Academica de Ingreso"],
        )
        with st.expander("Application and admission profile", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                values["Application mode"] = int_input("Application mode", 17, 1, 99)
                values["Application order"] = int_input("Application order", 1, 0, 9)
                values["Course"] = int_input("Course", 9254, 1, 9999)
            with col2:
                values["Daytime/evening attendance\t"] = st.selectbox(
                    "Daytime/evening attendance",
                    options=[1, 0],
                    format_func=lambda value: "Daytime" if value == 1 else "Evening",
                )
                values["Previous qualification"] = int_input("Previous qualification", 1, 1, 99)
            with col3:
                values["Previous qualification (grade)"] = float_input(
                    "Previous qualification (grade)", 130.0, 0.0, 200.0
                )
                values["Admission grade"] = float_input("Admission grade", 130.0, 0.0, 200.0)

    with tabs[2]:
        section_header(
            "Informacion Familiar",
            "Nivel educativo y ocupacion de los padres como contexto socioacademico.",
            SECTION_COUNTS["Informacion Familiar"],
        )
        col1, col2 = st.columns(2)
        with col1:
            values["Mother's qualification"] = int_input("Mother's qualification", 19, 1, 99)
            values["Mother's occupation"] = int_input("Mother's occupation", 5, 0, 999)
        with col2:
            values["Father's qualification"] = int_input("Father's qualification", 12, 1, 99)
            values["Father's occupation"] = int_input("Father's occupation", 9, 0, 999)

    with tabs[3]:
        section_header(
            "Informacion Financiera",
            "Indicadores economicos individuales asociados a pagos y apoyo financiero.",
            SECTION_COUNTS["Informacion Financiera"],
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            values["Debtor"] = yes_no_input("Debtor", 0)
        with col2:
            values["Tuition fees up to date"] = yes_no_input("Tuition fees up to date", 1)
        with col3:
            values["Scholarship holder"] = yes_no_input("Scholarship holder", 0)

    with tabs[4]:
        section_header(
            "Rendimiento Primer Semestre",
            "Desempeno academico del primer semestre utilizado por el modelo.",
            SECTION_COUNTS["Rendimiento Primer Semestre"],
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            values["Curricular units 1st sem (credited)"] = int_input(
                "Curricular units 1st sem (credited)", 0, 0, 40
            )
            values["Curricular units 1st sem (enrolled)"] = int_input(
                "Curricular units 1st sem (enrolled)", 6, 0, 40
            )
        with col2:
            values["Curricular units 1st sem (evaluations)"] = int_input(
                "Curricular units 1st sem (evaluations)", 6, 0, 80
            )
            values["Curricular units 1st sem (approved)"] = int_input(
                "Curricular units 1st sem (approved)", 5, 0, 40
            )
        with col3:
            values["Curricular units 1st sem (grade)"] = float_input(
                "Curricular units 1st sem (grade)", 12.0, 0.0, 20.0
            )
            values["Curricular units 1st sem (without evaluations)"] = int_input(
                "Curricular units 1st sem (without evaluations)", 0, 0, 40
            )

    with tabs[5]:
        section_header(
            "Rendimiento Segundo Semestre",
            "Desempeno academico del segundo semestre utilizado por el modelo.",
            SECTION_COUNTS["Rendimiento Segundo Semestre"],
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            values["Curricular units 2nd sem (credited)"] = int_input(
                "Curricular units 2nd sem (credited)", 0, 0, 40
            )
            values["Curricular units 2nd sem (enrolled)"] = int_input(
                "Curricular units 2nd sem (enrolled)", 6, 0, 40
            )
        with col2:
            values["Curricular units 2nd sem (evaluations)"] = int_input(
                "Curricular units 2nd sem (evaluations)", 6, 0, 80
            )
            values["Curricular units 2nd sem (approved)"] = int_input(
                "Curricular units 2nd sem (approved)", 5, 0, 40
            )
        with col3:
            values["Curricular units 2nd sem (grade)"] = float_input(
                "Curricular units 2nd sem (grade)", 12.0, 0.0, 20.0
            )
            values["Curricular units 2nd sem (without evaluations)"] = int_input(
                "Curricular units 2nd sem (without evaluations)", 0, 0, 40
            )

    with tabs[6]:
        section_header(
            "Indicadores Economicos",
            "Contexto macroeconomico incluido en el entrenamiento del modelo.",
            SECTION_COUNTS["Indicadores Economicos"],
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            values["Unemployment rate"] = float_input("Unemployment rate", 11.0, 0.0, 30.0)
        with col2:
            values["Inflation rate"] = float_input("Inflation rate", 1.4, -10.0, 30.0)
        with col3:
            values["GDP"] = float_input("GDP", 1.0, -20.0, 20.0)

    return values


def prepare_input_data(values: dict[str, Any], preprocessor: Any) -> pd.DataFrame:
    expected_columns = list(getattr(preprocessor, "feature_names_in_", FEATURE_COLUMNS))
    missing_values = [column for column in expected_columns if column not in values]
    if missing_values:
        raise ValueError(f"Missing required variables: {', '.join(missing_values)}")

    input_df = pd.DataFrame([{column: values[column] for column in expected_columns}])
    if input_df.shape[1] != 33:
        raise ValueError(f"The model requires 33 variables, but {input_df.shape[1]} were provided.")

    return input_df


def predict_student_status(
    model: Any, preprocessor: Any, input_df: pd.DataFrame
) -> tuple[str, dict[str, float], float]:
    transformed_input = preprocessor.transform(input_df)
    prediction = str(model.predict(transformed_input)[0])

    probabilities = {class_name: 0.0 for class_name in CLASS_ORDER}
    if hasattr(model, "predict_proba"):
        raw_probabilities = model.predict_proba(transformed_input)[0]
        model_classes = [str(class_name) for class_name in getattr(model, "classes_", CLASS_ORDER)]
        probabilities.update(
            {
                class_name: float(probability)
                for class_name, probability in zip(model_classes, raw_probabilities)
            }
        )
        confidence = probabilities.get(prediction, float(np.max(raw_probabilities)))
    else:
        probabilities[prediction] = 1.0
        confidence = 1.0

    return prediction, probabilities, confidence


def display_dashboard_intro(model: Any) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
            <p>
                Dashboard institucional de Machine Learning para estimar el estado academico
                de un estudiante entre Dropout, Enrolled y Graduate.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="dashboard-card">
                <h3>Objetivo del proyecto</h3>
                <p>Apoyar la toma de decisiones academicas mediante predicciones
                basadas en datos personales, familiares, financieros y curriculares.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <h3>Modelo utilizado</h3>
                <p>{type(model).__name__} con preprocesamiento persistido para asegurar
                la misma estructura usada durante el entrenamiento.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="dashboard-card">
                <h3>Cobertura de variables</h3>
                <p>El formulario captura las 33 variables reales esperadas por el modelo,
                distribuidas en siete secciones de analisis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def display_sidebar(model: Any, preprocessor: Any) -> None:
    expected_count = len(getattr(preprocessor, "feature_names_in_", FEATURE_COLUMNS))
    st.sidebar.title("Academic ML Dashboard")
    st.sidebar.caption("Student Performance Prediction")
    st.sidebar.divider()
    st.sidebar.metric("Model", type(model).__name__)
    st.sidebar.metric("Input variables", f"{expected_count}/33")
    st.sidebar.metric("Classes", str(len(CLASS_ORDER)))
    st.sidebar.divider()
    st.sidebar.write("Prediction classes")
    for class_name in CLASS_ORDER:
        st.sidebar.write(f"- {class_name}")
    st.sidebar.divider()
    st.sidebar.write("Input sections")
    for section_name, count in SECTION_COUNTS.items():
        st.sidebar.write(f"- {section_name}: {count}")


def display_results(prediction: str, probabilities: dict[str, float], confidence: float) -> None:
    content = RESULT_CONTENT[prediction]
    st.subheader("Prediction Results")
    st.markdown(
        f"""
        <div class="result-card" style="border-left-color: {content["accent"]};">
            <div class="result-kicker">{content["status"]}</div>
            <h2>{prediction}</h2>
            <div class="risk-pill">Risk Level: {content["risk"]}</div>
            <p>{content["description"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Predicted class", prediction)
    metric_cols[1].metric("Confidence", f"{confidence * 100:.2f}%")
    metric_cols[2].metric("Variables used", "33")
    metric_cols[3].metric("Classification", "Multiclass")

    st.write("")
    st.subheader("Class Probabilities")
    for class_name in CLASS_ORDER:
        probability = probabilities.get(class_name, 0.0)
        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-label">{class_name}</div>
                <div class="prob-value">{probability * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(probability, 0.0), 1.0))


def display_analysis_panel(model: Any, preprocessor: Any) -> None:
    st.subheader("Analysis Panel")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model used", type(model).__name__)
    col2.metric("Execution date", datetime.now().strftime("%Y-%m-%d"))
    col3.metric("Variables used", str(len(getattr(preprocessor, "feature_names_in_", FEATURE_COLUMNS))))
    col4.metric("Problem type", "Multiclass Classification")


def validate_schema(preprocessor: Any) -> None:
    expected_columns = list(getattr(preprocessor, "feature_names_in_", FEATURE_COLUMNS))
    if len(expected_columns) != 33:
        raise ValueError(f"The loaded preprocessor expects {len(expected_columns)} variables, not 33.")

    missing_in_app = [column for column in expected_columns if column not in FEATURE_COLUMNS]
    extra_in_app = [column for column in FEATURE_COLUMNS if column not in expected_columns]
    if missing_in_app or extra_in_app:
        raise ValueError(
            "The application schema does not match the trained preprocessor. "
            f"Missing in app: {missing_in_app}. Extra in app: {extra_in_app}."
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    apply_custom_css()

    try:
        model, preprocessor = load_model()
        validate_schema(preprocessor)
    except Exception as exc:
        st.error("The trained model or preprocessing schema could not be loaded correctly.")
        with st.expander("Technical details"):
            st.exception(exc)
        return

    display_sidebar(model, preprocessor)
    display_dashboard_intro(model)

    st.write("")
    st.subheader("Student Input Variables")
    st.caption(
        "Complete each section. The prediction is generated from the same 33 variables used during training."
    )
    values = create_input_form()

    st.write("")
    if st.button("Generate Prediction", type="primary"):
        try:
            input_df = prepare_input_data(values, preprocessor)
            prediction, probabilities, confidence = predict_student_status(model, preprocessor, input_df)
            display_results(prediction, probabilities, confidence)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error("The prediction could not be generated. Please review the entered values.")
            with st.expander("Technical details"):
                st.exception(exc)
    else:
        st.info("Complete the grouped form and press Generate Prediction to calculate the academic status.")

    st.write("")
    display_analysis_panel(model, preprocessor)


if __name__ == "__main__":
    main()
