from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


TITULO_APP = "Sistema de Predicción del Desempeño Estudiantil"
SUBTITULO_APP = "Herramienta institucional de análisis predictivo académico"
RUTA_MODELO = Path("model/best_model.pkl")
RUTA_PREPROCESADOR = Path("model/preprocessor.pkl")
CLASES_MODELO = ["Dropout", "Enrolled", "Graduate"]

TRADUCCION_CLASES = {
    "Dropout": "Desertar",
    "Enrolled": "Continuar matriculado",
    "Graduate": "Graduarse",
}

COLUMNAS_MODELO = [
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

CONTEO_SECCIONES = {
    "Información personal": 4,
    "Información académica de ingreso": 7,
    "Información familiar": 4,
    "Información financiera": 3,
    "Rendimiento del primer semestre": 6,
    "Rendimiento del segundo semestre": 6,
    "Indicadores económicos": 3,
}

ESTADO_CIVIL = {
    "Soltero/a": 1,
    "Casado/a": 2,
    "Viudo/a": 3,
    "Divorciado/a": 4,
    "Unión de hecho": 5,
    "Separado/a legalmente": 6,
}

GENERO = {"Femenino": 0, "Masculino": 1}
SI_NO = {"No": 0, "Sí": 1}
TURNO = {"Diurno": 1, "Nocturno": 0}

MODALIDAD_SOLICITUD = {
    "Primera fase del contingente general": 1,
    "Ordenanza n.º 612/93": 2,
    "Primera fase del contingente especial de Azores": 5,
    "Titulares de otros cursos superiores": 7,
    "Ordenanza n.º 854-B/99": 10,
    "Estudiante internacional": 15,
    "Primera fase del contingente especial de Madeira": 16,
    "Segunda fase del contingente general": 17,
    "Tercera fase del contingente general": 18,
    "Ordenanza n.º 533-A/99, apartado b2": 26,
    "Ordenanza n.º 533-A/99, apartado b3": 27,
    "Mayores de 23 años": 39,
    "Transferencia": 42,
    "Cambio de carrera": 43,
    "Titulares de diploma de especialización tecnológica": 44,
    "Cambio de institución y carrera": 51,
    "Titulares de diploma de ciclo corto": 53,
    "Cambio de institución o carrera internacional": 57,
}

CARRERAS = {
    "Tecnologías de producción de biocombustibles": 33,
    "Diseño de animación y multimedia": 171,
    "Servicio social (nocturno)": 8014,
    "Agronomía": 9003,
    "Diseño de comunicación": 9070,
    "Enfermería veterinaria": 9085,
    "Ingeniería informática": 9119,
    "Equinicultura": 9130,
    "Gestión": 9147,
    "Servicio social": 9238,
    "Turismo": 9254,
    "Enfermería": 9500,
    "Higiene oral": 9556,
    "Gestión de publicidad y marketing": 9670,
    "Periodismo y comunicación": 9773,
    "Educación básica": 9853,
    "Gestión (nocturno)": 9991,
}

NIVELES_EDUCATIVOS = {
    "Educación secundaria": 1,
    "Educación superior, licenciatura": 2,
    "Educación superior, grado": 3,
    "Educación superior, maestría": 4,
    "Educación superior, doctorado": 5,
    "Frecuencia de educación superior": 6,
    "12.º año no completado": 9,
    "11.º año no completado": 10,
    "7.º año": 11,
    "Otro, 11.º año": 12,
    "10.º año": 14,
    "10.º año no completado": 15,
    "Curso general de comercio": 18,
    "Educación básica, tercer ciclo": 19,
    "Curso complementario de secundaria": 20,
    "Curso técnico profesional": 22,
    "Curso complementario no concluido": 25,
    "7.º año de escolaridad": 26,
    "Segundo ciclo del curso general de secundaria": 27,
    "9.º año no completado": 29,
    "8.º año": 30,
    "Curso administrativo y comercial": 31,
    "Curso complementario de contabilidad": 33,
    "Desconocido": 34,
    "No sabe leer ni escribir": 35,
    "Sabe leer sin completar 4.º año": 36,
    "Educación básica, primer ciclo": 37,
    "Educación básica, segundo ciclo": 38,
    "Curso de especialización tecnológica": 39,
    "Educación superior, grado de primer ciclo": 40,
    "Curso de estudios superiores especializados": 41,
    "Curso técnico superior profesional": 42,
    "Educación superior, maestría de segundo ciclo": 43,
    "Educación superior, doctorado de tercer ciclo": 44,
}

CALIFICACION_PREVIA = {
    etiqueta: codigo
    for etiqueta, codigo in NIVELES_EDUCATIVOS.items()
    if codigo in {1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43}
}

OCUPACIONES = {
    "Estudiante": 0,
    "Representante del poder legislativo o ejecutivo": 1,
    "Especialista de actividades intelectuales y científicas": 2,
    "Técnico o profesional de nivel intermedio": 3,
    "Personal administrativo": 4,
    "Trabajador de servicios personales, seguridad o ventas": 5,
    "Agricultor o trabajador calificado agropecuario": 6,
    "Trabajador calificado de industria, construcción o artesanía": 7,
    "Operador de instalaciones, maquinaria o ensamblaje": 8,
    "Trabajador no calificado": 9,
    "Profesión de fuerzas armadas": 10,
    "Otra situación laboral": 90,
    "Sin ocupación": 99,
    "Oficial de fuerzas armadas": 101,
    "Sargento de fuerzas armadas": 102,
    "Personal de otros rangos de fuerzas armadas": 103,
    "Director de servicios administrativos o comerciales": 112,
    "Director de hotelería, restauración, comercio u otros servicios": 114,
    "Especialista en ciencias físicas, matemáticas, ingeniería o tecnología": 121,
    "Profesional de salud": 122,
    "Profesor": 123,
    "Especialista en finanzas, contabilidad u organización administrativa": 124,
    "Especialista en tecnologías de la información": 125,
    "Profesional de ciencias jurídicas, sociales, artísticas o culturales": 131,
    "Técnico de ciencias e ingeniería de nivel intermedio": 132,
    "Técnico de tecnologías de la información y comunicación": 134,
    "Técnico de salud de nivel intermedio": 135,
    "Técnico de nivel intermedio legal, social, deportivo o cultural": 141,
    "Técnico de finanzas, administración o negocios": 143,
    "Técnico de servicios públicos, policía o fuerzas similares": 144,
    "Empleado de oficina, secretaría u operador de datos": 151,
    "Empleado de trato directo con clientes": 152,
    "Trabajador de datos, contabilidad, estadística o registro": 153,
    "Empleado administrativo de apoyo": 154,
    "Trabajador de servicios personales": 161,
    "Vendedor": 163,
    "Agricultor y trabajador calificado agrícola": 171,
    "Trabajador calificado de ganadería o pesca": 172,
    "Trabajador calificado forestal o de caza": 173,
    "Agricultor, ganadero, pescador o cazador de subsistencia": 174,
    "Trabajador calificado de construcción": 175,
    "Trabajador calificado de metalurgia o mecánica": 181,
    "Trabajador calificado de electricidad o electrónica": 182,
    "Trabajador de procesamiento de alimentos, madera, confección u otra industria": 183,
    "Operador de instalaciones fijas o maquinaria": 191,
    "Trabajador de ensamblaje": 192,
    "Conductor u operador de equipo móvil": 193,
    "Ayudante de preparación de comidas": 194,
    "Vendedor ambulante o trabajador de servicios callejeros": 195,
}

CONTENIDO_RESULTADOS = {
    "Dropout": {
        "riesgo": "Alto",
        "estado": "Riesgo de deserción académica",
        "color": "#9f2a2a",
        "descripcion": (
            "El estudiante presenta características asociadas con la deserción académica. "
            "Se recomienda intervención temprana, seguimiento académico y apoyo institucional."
        ),
    },
    "Enrolled": {
        "riesgo": "Medio",
        "estado": "Proyección de permanencia activa",
        "color": "#8a6417",
        "descripcion": (
            "El estudiante tiene alta probabilidad de mantenerse matriculado. "
            "Se recomienda monitoreo académico continuo para fortalecer su progreso."
        ),
    },
    "Graduate": {
        "riesgo": "Bajo",
        "estado": "Perfil asociado a graduación",
        "color": "#22634b",
        "descripcion": (
            "El estudiante presenta características asociadas con finalización académica exitosa. "
            "El perfil sugiere condiciones favorables para la graduación."
        ),
    },
}


def nombre_modelo_en_espanol(modelo: Any) -> str:
    nombres = {
        "LogisticRegression": "Regresión logística",
        "RandomForestClassifier": "Bosque aleatorio",
        "DecisionTreeClassifier": "Árbol de decisión",
        "KNeighborsClassifier": "Vecinos más cercanos",
        "SVC": "Máquina de vectores de soporte",
        "GaussianNB": "Clasificador bayesiano gaussiano",
        "GradientBoostingClassifier": "Potenciación por gradiente",
    }
    return nombres.get(type(modelo).__name__, "Modelo de clasificación entrenado")


def aplicar_estilos() -> None:
    st.markdown(
        """
        <style>
            :root {
                --azul-950: #071a2f;
                --azul-900: #0b2440;
                --azul-800: #12385f;
                --azul-700: #1c4f7f;
                --azul-100: #e7f0fb;
                --azul-050: #f4f8fd;
                --gris-900: #101828;
                --gris-700: #344054;
                --gris-600: #475467;
                --borde: #d6e1ee;
                --superficie: #ffffff;
            }

            .stApp {
                background: #f3f6fa;
                color: var(--gris-900);
            }

            header[data-testid="stHeader"],
            div[data-testid="stToolbar"],
            div[data-testid="stDecoration"],
            #MainMenu,
            footer {
                display: none;
                visibility: hidden;
                height: 0;
            }

            .block-container {
                max-width: 1260px;
                padding: 1rem 2rem 2.5rem;
            }

            .encabezado {
                background: var(--azul-900);
                border: 1px solid var(--azul-800);
                border-radius: 8px;
                color: #ffffff;
                margin-bottom: 1rem;
                padding: 1.4rem 1.5rem;
            }

            .encabezado h1 {
                color: #ffffff;
                font-size: 2.1rem;
                letter-spacing: 0;
                line-height: 1.16;
                margin: 0;
            }

            .encabezado p {
                color: #d9e8f8;
                font-size: 1.02rem;
                line-height: 1.5;
                margin: 0.45rem 0 0;
            }

            .tarjeta,
            .tarjeta-seccion,
            .tarjeta-resultado,
            .fila-probabilidad {
                background: var(--superficie);
                border: 1px solid var(--borde);
                border-radius: 8px;
                box-shadow: 0 10px 24px rgba(16, 40, 71, 0.07);
            }

            .tarjeta {
                border-top: 4px solid var(--azul-700);
                min-height: 142px;
                padding: 1.05rem 1.1rem;
            }

            .tarjeta h3,
            .tarjeta-seccion h3,
            .tarjeta-resultado h2 {
                color: var(--azul-900);
                letter-spacing: 0;
                margin: 0 0 0.45rem;
            }

            .tarjeta p,
            .tarjeta-seccion p,
            .tarjeta-resultado p {
                color: var(--gris-600);
                line-height: 1.5;
                margin: 0;
            }

            .tarjeta-seccion {
                background: #fbfdff;
                margin: 0.5rem 0 1rem;
                padding: 1rem 1.1rem;
            }

            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid var(--borde);
                border-radius: 8px;
                box-shadow: 0 8px 18px rgba(16, 40, 71, 0.06);
                padding: 0.75rem 0.9rem;
            }

            div[data-testid="stMetric"] label {
                color: var(--gris-600);
                font-weight: 650;
            }

            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                color: var(--azul-900);
                font-weight: 800;
            }

            div[data-testid="stTabs"] button {
                color: var(--azul-900);
                font-weight: 750;
            }

            div[data-baseweb="tab-highlight"] {
                background-color: var(--azul-700);
            }

            label, .stNumberInput label, .stSelectbox label {
                color: var(--gris-900) !important;
                font-weight: 650;
            }

            /* Ensure Streamlit buttons show white text across contexts */
            .stApp .stButton > button,
            .stApp .stButton button,
            .stButton > button {
                background: var(--azul-700);
                border: 1px solid var(--azul-700);
                border-radius: 7px;
                color: #ffffff !important;
                font-weight: 800;
                min-height: 2.8rem;
                width: 100%;
            }

            .stApp .stButton > button:hover,
            .stButton > button:hover {
                background: var(--azul-800);
                border-color: var(--azul-800);
                color: #ffffff !important;
            }

            .tarjeta-resultado {
                border-left: 7px solid var(--azul-700);
                margin-top: 0.75rem;
                padding: 1.2rem 1.25rem;
            }

            .etiqueta-resultado {
                color: var(--gris-600);
                font-weight: 750;
                margin-bottom: 0.25rem;
            }

            .nivel-riesgo {
                background: var(--azul-100);
                border: 1px solid var(--borde);
                border-radius: 6px;
                color: var(--azul-900);
                display: inline-block;
                font-weight: 800;
                margin: 0.55rem 0 0.65rem;
                padding: 0.28rem 0.62rem;
            }

            .fila-probabilidad {
                margin-bottom: 0.55rem;
                padding: 0.65rem 0.8rem;
            }

            .nombre-probabilidad {
                color: var(--azul-900);
                font-weight: 800;
                margin-bottom: 0.2rem;
            }

            .valor-probabilidad {
                color: var(--gris-600);
                font-weight: 750;
            }

            .barra-aplicacion {
                align-items: center;
                background: #ffffff;
                border: 1px solid var(--borde);
                border-radius: 8px;
                box-shadow: 0 8px 20px rgba(16, 40, 71, 0.06);
                display: flex;
                justify-content: space-between;
                margin-bottom: 1rem;
                padding: 0.8rem 1rem;
            }

            .marca-aplicacion {
                align-items: center;
                display: flex;
                gap: 0.75rem;
            }

            .marca-simbolo {
                align-items: center;
                background: var(--azul-900);
                border-radius: 8px;
                color: #ffffff;
                display: flex;
                font-weight: 900;
                height: 42px;
                justify-content: center;
                width: 42px;
            }

            .marca-texto {
                color: var(--azul-900);
                font-size: 1rem;
                font-weight: 850;
                line-height: 1.15;
            }

            .marca-subtexto {
                color: var(--gris-600);
                font-size: 0.82rem;
                font-weight: 650;
                margin-top: 0.12rem;
            }

            .chip-header {
                background: var(--azul-050);
                border: 1px solid var(--borde);
                border-radius: 6px;
                color: var(--azul-900);
                display: inline-block;
                font-size: 0.82rem;
                font-weight: 800;
                margin-left: 0.35rem;
                padding: 0.35rem 0.55rem;
            }

            .pantalla-inicio {
                background: var(--azul-900);
                border: 1px solid var(--azul-800);
                border-radius: 8px;
                color: #ffffff;
                padding: 2rem;
            }

            .pantalla-inicio h1 {
                color: #ffffff;
                font-size: 2.35rem;
                letter-spacing: 0;
                line-height: 1.12;
                margin: 0;
            }

            .pantalla-inicio p {
                color: #d9e8f8;
                font-size: 1.02rem;
                line-height: 1.58;
                margin: 0.75rem 0 0;
                max-width: 860px;
            }

            .pantalla-inicio-cta {
                margin-top: 1rem;
                text-align: center;
            }

            .pantalla-inicio-cta .stButton > button {
                max-width: 360px;
            }

            .bloque-inicio {
                background: #ffffff;
                border: 1px solid var(--borde);
                border-radius: 8px;
                box-shadow: 0 10px 24px rgba(16, 40, 71, 0.07);
                min-height: 170px;
                padding: 1.15rem;
            }

            .bloque-inicio h3 {
                color: var(--azul-900);
                margin: 0 0 0.45rem;
            }

            .bloque-inicio p {
                color: var(--gris-600);
                line-height: 1.5;
                margin: 0;
            }

            /* Footer: light variant (white background) with contrasted chips */
            .pie-aplicacion {
                background: var(--superficie);
                border: 1px solid var(--borde);
                border-radius: 8px;
                color: var(--gris-900);
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                align-items: center;
                margin-top: 1.4rem;
                padding: 0.9rem 1rem;
                box-shadow: 0 6px 18px rgba(16, 40, 71, 0.04);
            }

            .pie-aplicacion-contenido {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                align-items: center;
                gap: 0.75rem;
                width: 100%;
            }

            .pie-aplicacion-chip {
                background: var(--azul-050);
                border: 1px solid var(--borde);
                border-radius: 999px;
                color: var(--azul-900);
                display: inline-flex;
                font-size: 0.8rem;
                font-weight: 750;
                line-height: 1;
                padding: 0.32rem 0.65rem;
                white-space: nowrap;
            }

            .pie-aplicacion-titulo {
                color: var(--gris-900);
                font-size: 0.98rem;
                font-weight: 850;
                letter-spacing: 0.2px;
            }

            .pie-aplicacion-autor-chip {
                background: var(--azul-050);
                border-color: var(--borde);
                box-shadow: none;
                color: var(--azul-900);
            }

            .pie-aplicacion-autores {
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                gap: 0.35rem;
                margin-left: auto;
            }

            .pie-aplicacion-texto-footer {
                color: var(--gris-700);
                font-size: 0.82rem;
                font-weight: 650;
                letter-spacing: 0.1px;
                padding: 0.1rem 0;
            }

            .pie-aplicacion strong {
                color: var(--azul-900);
            }

            .pie-aplicacion span {
                color: var(--gris-700);
                font-size: 0.88rem;
                font-weight: 650;
            }

            @media (max-width: 760px) {
                .barra-aplicacion,
                .pie-aplicacion {
                    align-items: flex-start;
                    flex-direction: column;
                    gap: 0.75rem;
                }

                .pie-aplicacion-contenido {
                    width: 100%;
                }

                .chip-header {
                    margin: 0 0.25rem 0.25rem 0;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[Any, Any]:
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {RUTA_MODELO}")
    if not RUTA_PREPROCESADOR.exists():
        raise FileNotFoundError(f"No se encontró el archivo de preprocesamiento: {RUTA_PREPROCESADOR}")

    modelo = joblib.load(RUTA_MODELO)
    preprocesador = joblib.load(RUTA_PREPROCESADOR)
    return modelo, preprocesador


def selector_categoria(etiqueta: str, opciones: dict[str, int], ayuda: str, valor_inicial: str | None = None) -> int:
    etiquetas = list(opciones.keys())
    indice = etiquetas.index(valor_inicial) if valor_inicial in opciones else 0
    seleccion = st.selectbox(etiqueta, etiquetas, index=indice, help=ayuda)
    return opciones[seleccion]


def entrada_entera(etiqueta: str, valor: int, minimo: int, maximo: int, ayuda: str) -> int:
    return int(
        st.number_input(
            etiqueta,
            min_value=minimo,
            max_value=maximo,
            value=valor,
            step=1,
            help=ayuda,
        )
    )


def entrada_decimal(
    etiqueta: str,
    valor: float,
    minimo: float,
    maximo: float,
    ayuda: str,
    paso: float = 0.1,
) -> float:
    return float(
        st.number_input(
            etiqueta,
            min_value=minimo,
            max_value=maximo,
            value=valor,
            step=paso,
            help=ayuda,
        )
    )


def encabezado_seccion(titulo: str, descripcion: str, cantidad: int) -> None:
    return None


def create_input_form() -> tuple[dict[str, Any], bool]:
    valores: dict[str, Any] = {}
    generar_prediccion = False
    pestanas = st.tabs(
        [
            "Personal",
            "Ingreso",
            "Familiar",
            "Financiera",
            "Primer semestre",
            "Segundo semestre",
            "Economía",
        ]
    )

    with pestanas[0]:
        encabezado_seccion(
            "Información personal",
            "Datos demográficos y condiciones generales del estudiante.",
            CONTEO_SECCIONES["Información personal"],
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            valores["Marital status"] = selector_categoria(
                "Estado civil",
                ESTADO_CIVIL,
                "Situación civil actual del estudiante.",
                "Soltero/a",
            )
        with col2:
            valores["Gender"] = selector_categoria(
                "Género",
                GENERO,
                "Género registrado del estudiante.",
                "Femenino",
            )
        with col3:
            valores["Age at enrollment"] = entrada_entera(
                "Edad al momento de la matrícula",
                20,
                15,
                80,
                "Edad del estudiante cuando ingresó a la universidad.",
            )
        with col4:
            valores["Displaced"] = selector_categoria(
                "Desplazado o foráneo",
                SI_NO,
                "Indica si el estudiante vive fuera de su municipio de origen para estudiar.",
                "No",
            )

    with pestanas[1]:
        encabezado_seccion(
            "Información académica de ingreso",
            "Variables relacionadas con el proceso de admisión y los antecedentes académicos.",
            CONTEO_SECCIONES["Información académica de ingreso"],
        )
        with st.expander("Solicitud, carrera y admisión", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                valores["Application mode"] = selector_categoria(
                    "Modalidad de solicitud de ingreso",
                    MODALIDAD_SOLICITUD,
                    "Tipo de proceso mediante el cual el estudiante ingresó a la institución.",
                    "Segunda fase del contingente general",
                )
                valores["Application order"] = entrada_entera(
                    "Orden de preferencia de la solicitud",
                    1,
                    0,
                    9,
                    "Posición en la que el estudiante eligió esta carrera al aplicar.",
                )
                valores["Course"] = selector_categoria(
                    "Carrera",
                    CARRERAS,
                    "Carrera o programa académico seleccionado por el estudiante.",
                    "Turismo",
                )
            with col2:
                valores["Daytime/evening attendance\t"] = selector_categoria(
                    "Turno de asistencia",
                    TURNO,
                    "Horario principal de asistencia del estudiante.",
                    "Diurno",
                )
                valores["Previous qualification"] = selector_categoria(
                    "Nivel de calificación previa",
                    CALIFICACION_PREVIA,
                    "Nivel educativo alcanzado antes del ingreso a la universidad.",
                    "Educación secundaria",
                )
            with col3:
                valores["Previous qualification (grade)"] = entrada_decimal(
                    "Nota de calificación previa",
                    130.0,
                    0.0,
                    200.0,
                    "Calificación obtenida en la formación previa.",
                )
                valores["Admission grade"] = entrada_decimal(
                    "Nota de admisión",
                    130.0,
                    0.0,
                    200.0,
                    "Calificación obtenida en el proceso de admisión.",
                )

    with pestanas[2]:
        encabezado_seccion(
            "Información familiar",
            "Nivel educativo y ocupación de los padres como contexto socioacadémico.",
            CONTEO_SECCIONES["Información familiar"],
        )
        col1, col2 = st.columns(2)
        with col1:
            valores["Mother's qualification"] = selector_categoria(
                "Nivel educativo de la madre",
                NIVELES_EDUCATIVOS,
                "Categoría educativa registrada para la madre.",
                "Educación básica, tercer ciclo",
            )
            valores["Mother's occupation"] = selector_categoria(
                "Ocupación de la madre",
                OCUPACIONES,
                "Actividad laboral registrada para la madre.",
                "Trabajador de servicios personales, seguridad o ventas",
            )
        with col2:
            valores["Father's qualification"] = selector_categoria(
                "Nivel educativo del padre",
                NIVELES_EDUCATIVOS,
                "Categoría educativa registrada para el padre.",
                "Otro, 11.º año",
            )
            valores["Father's occupation"] = selector_categoria(
                "Ocupación del padre",
                OCUPACIONES,
                "Actividad laboral registrada para el padre.",
                "Trabajador no calificado",
            )

    with pestanas[3]:
        encabezado_seccion(
            "Información financiera",
            "Indicadores individuales asociados con pagos, deuda y apoyo financiero.",
            CONTEO_SECCIONES["Información financiera"],
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            valores["Debtor"] = selector_categoria(
                "Deudor",
                SI_NO,
                "Indica si existen deudas pendientes con la institución.",
                "No",
            )
        with col2:
            valores["Tuition fees up to date"] = selector_categoria(
                "Matrícula al día",
                SI_NO,
                "Indica si el estudiante está al día con sus pagos.",
                "Sí",
            )
        with col3:
            valores["Scholarship holder"] = selector_categoria(
                "Becado",
                SI_NO,
                "Indica si recibe una beca académica.",
                "No",
            )

    with pestanas[4]:
        encabezado_seccion(
            "Rendimiento del primer semestre",
            "Resultados académicos del primer semestre registrados en el modelo.",
            CONTEO_SECCIONES["Rendimiento del primer semestre"],
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            valores["Curricular units 1st sem (credited)"] = entrada_entera(
                "Unidades curriculares acreditadas",
                0,
                0,
                40,
                "Cantidad de unidades curriculares reconocidas o acreditadas en el primer semestre.",
            )
            valores["Curricular units 1st sem (enrolled)"] = entrada_entera(
                "Unidades curriculares matriculadas",
                6,
                0,
                40,
                "Cantidad de unidades curriculares inscritas durante el primer semestre.",
            )
        with col2:
            valores["Curricular units 1st sem (evaluations)"] = entrada_entera(
                "Evaluaciones realizadas",
                6,
                0,
                80,
                "Número de evaluaciones realizadas durante el primer semestre.",
            )
            valores["Curricular units 1st sem (approved)"] = entrada_entera(
                "Unidades curriculares aprobadas",
                5,
                0,
                40,
                "Cantidad de unidades curriculares aprobadas en el primer semestre.",
            )
        with col3:
            valores["Curricular units 1st sem (grade)"] = entrada_decimal(
                "Nota promedio del primer semestre",
                12.0,
                0.0,
                20.0,
                "Promedio académico obtenido durante el primer semestre.",
            )
            valores["Curricular units 1st sem (without evaluations)"] = entrada_entera(
                "Unidades curriculares sin evaluación",
                0,
                0,
                40,
                "Cantidad de unidades curriculares sin evaluación en el primer semestre.",
            )

    with pestanas[5]:
        encabezado_seccion(
            "Rendimiento del segundo semestre",
            "Resultados académicos del segundo semestre registrados en el modelo.",
            CONTEO_SECCIONES["Rendimiento del segundo semestre"],
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            valores["Curricular units 2nd sem (credited)"] = entrada_entera(
                "Unidades curriculares acreditadas",
                0,
                0,
                40,
                "Cantidad de unidades curriculares reconocidas o acreditadas en el segundo semestre.",
            )
            valores["Curricular units 2nd sem (enrolled)"] = entrada_entera(
                "Unidades curriculares matriculadas",
                6,
                0,
                40,
                "Cantidad de unidades curriculares inscritas durante el segundo semestre.",
            )
        with col2:
            valores["Curricular units 2nd sem (evaluations)"] = entrada_entera(
                "Evaluaciones realizadas",
                6,
                0,
                80,
                "Número de evaluaciones realizadas durante el segundo semestre.",
            )
            valores["Curricular units 2nd sem (approved)"] = entrada_entera(
                "Unidades curriculares aprobadas",
                5,
                0,
                40,
                "Cantidad de unidades curriculares aprobadas en el segundo semestre.",
            )
        with col3:
            valores["Curricular units 2nd sem (grade)"] = entrada_decimal(
                "Nota promedio del segundo semestre",
                12.0,
                0.0,
                20.0,
                "Promedio académico obtenido durante el segundo semestre.",
            )
            valores["Curricular units 2nd sem (without evaluations)"] = entrada_entera(
                "Unidades curriculares sin evaluación",
                0,
                0,
                40,
                "Cantidad de unidades curriculares sin evaluación en el segundo semestre.",
            )

    with pestanas[6]:
        encabezado_seccion(
            "Indicadores económicos",
            "Contexto macroeconómico del período de matrícula.",
            CONTEO_SECCIONES["Indicadores económicos"],
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            valores["Unemployment rate"] = entrada_decimal(
                "Tasa de desempleo",
                11.0,
                0.0,
                30.0,
                "Tasa de desempleo del país durante el período de matrícula.",
            )
        with col2:
            valores["Inflation rate"] = entrada_decimal(
                "Tasa de inflación",
                1.4,
                -10.0,
                30.0,
                "Tasa de inflación del país durante el período de matrícula.",
            )
        with col3:
            valores["GDP"] = entrada_decimal(
                "Producto Interno Bruto (PIB)",
                1.0,
                -20.0,
                20.0,
                "Indicador económico del crecimiento del país.",
            )

        st.write("")
        st.divider()
        generar_prediccion = st.button("Generar predicción", type="primary")

    return valores, generar_prediccion


def prepare_input_data(valores: dict[str, Any], preprocesador: Any) -> pd.DataFrame:
    columnas_esperadas = list(getattr(preprocesador, "feature_names_in_", COLUMNAS_MODELO))
    faltantes = [columna for columna in columnas_esperadas if columna not in valores]
    if faltantes:
        raise ValueError(f"Faltan variables requeridas: {', '.join(faltantes)}")

    datos = pd.DataFrame([{columna: valores[columna] for columna in columnas_esperadas}])
    if datos.shape[1] != 33:
        raise ValueError(f"El modelo requiere 33 variables, pero se recibieron {datos.shape[1]}.")

    return datos


def predict_student_status(
    modelo: Any, preprocesador: Any, datos: pd.DataFrame
) -> tuple[str, dict[str, float], float]:
    datos_transformados = preprocesador.transform(datos)
    prediccion = str(modelo.predict(datos_transformados)[0])

    probabilidades = {clase: 0.0 for clase in CLASES_MODELO}
    if hasattr(modelo, "predict_proba"):
        probabilidades_raw = modelo.predict_proba(datos_transformados)[0]
        clases_modelo = [str(clase) for clase in getattr(modelo, "classes_", CLASES_MODELO)]
        probabilidades.update(
            {
                clase: float(probabilidad)
                for clase, probabilidad in zip(clases_modelo, probabilidades_raw)
            }
        )
        confianza = probabilidades.get(prediccion, float(np.max(probabilidades_raw)))
    else:
        probabilidades[prediccion] = 1.0
        confianza = 1.0

    return prediccion, probabilidades, confianza


def mostrar_introduccion(modelo: Any) -> None:
    st.markdown(
        f"""
        <div class="encabezado">
            <h1>{TITULO_APP}</h1>
            <p>{SUBTITULO_APP}</p>
            <p>
                Plataforma universitaria para estimar si un estudiante podría desertar,
                mantenerse matriculado o graduarse, usando las 33 variables reales del modelo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="tarjeta">
                <h3>Objetivo del proyecto</h3>
                <p>Apoyar la toma de decisiones académicas mediante predicciones
                basadas en información personal, familiar, financiera y curricular.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="tarjeta">
                <h3>Modelo utilizado</h3>
                <p>{nombre_modelo_en_espanol(modelo)} con el mismo preprocesamiento utilizado
                durante el entrenamiento del proyecto.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="tarjeta">
                <h3>Cobertura del formulario</h3>
                <p>El formulario solicita exactamente las 33 variables entrenadas,
                organizadas en siete secciones para facilitar su uso.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def mostrar_header_aplicacion() -> None:
    st.markdown(
        """
        <div class="barra-aplicacion">
            <div class="marca-aplicacion">
                <div class="marca-simbolo">SP</div>
                <div>
                    <h1>Predicción de Rendimiento Académico</h1>
                </div>
            </div>
            <div>
                <span class="marca-subtexto"> Inteligencia Artificial, 2026</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mostrar_footer_aplicacion() -> None:
    st.markdown(
        """
        <div class="pie-aplicacion">
            <div class="pie-aplicacion-contenido">
                <span class="pie-aplicacion-texto-footer">Student Performance Prediction, Proyecto Inteligencia Artificial 2026                           </span>
                <div class="pie-aplicacion-autores">
                    <span class="pie-aplicacion-chip pie-aplicacion-autor-chip">Jordan Ortiz Molina</span>
                    <span class="pie-aplicacion-chip pie-aplicacion-autor-chip">Yenifer Mata Flores</span>
                    <span class="pie-aplicacion-chip pie-aplicacion-autor-chip">Deyaneira Altamirano Cordero</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mostrar_pantalla_inicio(modelo: Any, preprocesador: Any) -> None:
    mostrar_header_aplicacion()
    st.markdown(
        f"""
        <div class="tarjeta-seccion" style="margin-bottom:0.8rem; padding:0.8rem 1rem;">
            <p style="margin:0; color:var(--gris-600);">Este sistema permite estimar el estado académico probable de un estudiante: desertar, seguir matriculado o graduarse.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="bloque-inicio">
                <h3>Qué hace el sistema</h3>
                <p>Recibe información personal, académica, familiar, financiera y económica
                del estudiante para generar una predicción de desempeño académico.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="bloque-inicio">
                <h3>Para qué sirve</h3>
                <p>Su propósito es apoyar el análisis institucional y facilitar la identificación temprana de perfiles que podrían requerir acompañamiento académico</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="bloque-inicio">
                <h3>Cómo se entrenó</h3>
                <p>El modelo utiliza el dataset UCI de deserción y éxito académico, con
                {len(getattr(preprocesador, "feature_names_in_", COLUMNAS_MODELO))} variables y un modelo de {nombre_modelo_en_espanol(modelo).lower()}.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='pantalla-inicio-cta'>", unsafe_allow_html=True)
    if st.button("Ir al formulario de predicción", type="primary"):
        st.session_state["pantalla"] = "prediccion"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    mostrar_footer_aplicacion()


def display_results(prediccion: str, probabilidades: dict[str, float], confianza: float) -> None:
    contenido = CONTENIDO_RESULTADOS[prediccion]
    prediccion_es = TRADUCCION_CLASES[prediccion]

    st.subheader("Resultados de la predicción")
    st.markdown(
        f"""
        <div class="tarjeta-resultado" style="border-left-color: {contenido["color"]};">
            <div class="etiqueta-resultado">{contenido["estado"]}</div>
            <h2>{prediccion_es}</h2>
            <div class="nivel-riesgo">Nivel de riesgo: {contenido["riesgo"]}</div>
            <p>{contenido["descripcion"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    columnas = st.columns(2)
    columnas[0].metric("Clase predicha", prediccion_es)
    columnas[1].metric("Nivel de confianza", f"{confianza * 100:.2f}%")

    st.write("")
    st.subheader("Probabilidades por clase")
    for clase in CLASES_MODELO:
        probabilidad = probabilidades.get(clase, 0.0)
        st.markdown(
            f"""
            <div class="fila-probabilidad">
                <div class="nombre-probabilidad">{TRADUCCION_CLASES[clase]}</div>
                <div class="valor-probabilidad">{probabilidad * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(probabilidad, 0.0), 1.0))


def mostrar_panel_analisis(modelo: Any, preprocesador: Any) -> None:
    st.subheader("Panel de análisis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Modelo utilizado", nombre_modelo_en_espanol(modelo))
    col2.metric("Variables utilizadas", str(len(getattr(preprocesador, "feature_names_in_", COLUMNAS_MODELO))))
    col3.metric("Tipo de problema", "Clasificación multiclase")


def validar_esquema(preprocesador: Any) -> None:
    columnas_esperadas = list(getattr(preprocesador, "feature_names_in_", COLUMNAS_MODELO))
    if len(columnas_esperadas) != 33:
        raise ValueError(f"El preprocesador cargado espera {len(columnas_esperadas)} variables, no 33.")

    faltantes = [columna for columna in columnas_esperadas if columna not in COLUMNAS_MODELO]
    sobrantes = [columna for columna in COLUMNAS_MODELO if columna not in columnas_esperadas]
    if faltantes or sobrantes:
        raise ValueError(
            "El esquema de la aplicación no coincide con el preprocesador entrenado. "
            f"Faltantes: {faltantes}. Sobrantes: {sobrantes}."
        )


def main() -> None:
    st.set_page_config(
        page_title=TITULO_APP,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    aplicar_estilos()

    try:
        modelo, preprocesador = load_model()
        validar_esquema(preprocesador)
    except Exception as exc:
        st.error("No fue posible cargar correctamente el modelo entrenado o su esquema de datos.")
        st.error(str(exc))
        return

    if "pantalla" not in st.session_state:
        st.session_state["pantalla"] = "inicio"

    if st.session_state["pantalla"] == "inicio":
        mostrar_pantalla_inicio(modelo, preprocesador)
        return
    # Show header and provide navigation back to start
    mostrar_header_aplicacion()

    # Instruction: user must complete the form to generate a prediction
    st.markdown(
        """
        <div class="tarjeta-seccion" style="margin-bottom:0.8rem; padding:0.8rem 1rem;">
            <p style="margin:0; color:var(--gris-600);">Debe completar el formulario para generar la predicción.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    valores, generar_prediccion = create_input_form()

    st.write("")
    if generar_prediccion:
        try:
            datos = prepare_input_data(valores, preprocesador)
            prediccion, probabilidades, confianza = predict_student_status(modelo, preprocesador, datos)
            display_results(prediccion, probabilidades, confianza)
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("No fue posible generar la predicción. Revise los datos ingresados e inténtelo nuevamente.")

    volver_col, _ = st.columns([1, 4])
    with volver_col:
        if st.button("Volver al inicio"):
            st.session_state["pantalla"] = "inicio"
            st.rerun()

    st.write("")

    # Footer on form/results screen
    # mostrar_footer_aplicacion()


if __name__ == "__main__":
    main()
