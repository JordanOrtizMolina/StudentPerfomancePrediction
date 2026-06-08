from __future__ import annotations
from pathlib import Path
from typing import Any
import streamlit as st

from src.prediction import load_model_artifacts, predict_student_status, prepare_input_data
from src.preprocessing import FEATURE_COLUMNS


# ── Constantes ───────────────────────────────────────────────

TITULO_APP     = "Sistema de Predicción del Desempeño Estudiantil"
SUBTITULO_APP  = "Herramienta institucional de análisis predictivo académico"
RUTA_MODELO        = Path("model/best_model.pkl")
RUTA_PREPROCESADOR = Path("model/preprocessor.pkl")
RUTA_LABEL_ENCODER = Path("model/label_encoder.pkl")   # ← NUEVO: para XGBoost
CLASES_MODELO  = ["Dropout", "Enrolled", "Graduate"]
TOTAL_PASOS    = 6

TRADUCCION_CLASES = {
    "Dropout":  "Desertar",
    "Enrolled": "Continuar matriculado",
    "Graduate": "Graduarse",
}

TITULOS_PASOS = [
    "Información Personal",
    "Ingreso académico",
    "Información familiar",
    "Información financiera",
    "Primer semestre",
    "Segundo semestre",
]

COLUMNAS_MODELO = [
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
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]

CONTENIDO_RESULTADOS = {
    "Dropout": {
        "riesgo":    "Alto",
        "estado":    "Riesgo de deserción académica",
        "color":     "#9f2a2a",
        "descripcion": (
            "El estudiante presenta características asociadas con la deserción académica. "
            "Se recomienda intervención temprana, seguimiento académico y apoyo institucional."
        ),
    },
    "Enrolled": {
        "riesgo":    "Medio",
        "estado":    "Proyección de permanencia activa",
        "color":     "#8a6417",
        "descripcion": (
            "El estudiante tiene alta probabilidad de mantenerse matriculado. "
            "Se recomienda monitoreo académico continuo para fortalecer su progreso."
        ),
    },
    "Graduate": {
        "riesgo":    "Bajo",
        "estado":    "Perfil asociado a graduación",
        "color":     "#22634b",
        "descripcion": (
            "El estudiante presenta características asociadas con finalización académica exitosa. "
            "El perfil sugiere condiciones favorables para la graduación."
        ),
    },
}

# ── Catálogos ────────────────────────────────────────────────

ESTADO_CIVIL = {
    "Soltero/a": 1, "Casado/a": 2, "Viudo/a": 3,
    "Divorciado/a": 4, "Unión de hecho": 5, "Separado/a legalmente": 6,
}
GENERO = {"Femenino": 0, "Masculino": 1}
SI_NO  = {"No": 0, "Sí": 1}
TURNO  = {"Diurno": 1, "Nocturno": 0}

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


# ── Estilos ──────────────────────────────────────────────────

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
                --borde:    #d6e1ee;
                --superficie: #ffffff;
            }

            .stApp { background: #f3f6fa; color: var(--gris-900); }

            header[data-testid="stHeader"],
            div[data-testid="stToolbar"],
            div[data-testid="stDecoration"],
            #MainMenu, footer {
                display: none; visibility: hidden; height: 0;
            }

            .block-container { max-width: 1260px; padding: 1rem 2rem 2.5rem; }

            .barra-aplicacion {
                align-items: center;
                background: var(--azul-900);
                border: 1px solid var(--azul-800);
                border-radius: 8px;
                display: flex;
                justify-content: space-between;
                margin-bottom: 1rem;
                padding: 0.75rem 1rem;
            }
            .marca-aplicacion { align-items: center; display: flex; gap: 0.75rem; }
            .marca-simbolo {
                align-items: center; background: var(--azul-700); border-radius: 8px;
                color: #fff; display: flex; font-weight: 900;
                height: 40px; justify-content: center; width: 40px; flex-shrink: 0;
            }
            .marca-texto    { color: #fff; font-size: 1rem; font-weight: 850; line-height: 1.15; }
            .marca-subtexto { color: #9dc4e8; font-size: 0.82rem; font-weight: 600; margin-top: 0.1rem; }

            .tarjeta, .bloque-inicio {
                background: var(--superficie);
                border: 1px solid var(--borde);
                border-radius: 8px;
                box-shadow: 0 10px 24px rgba(16,40,71,0.07);
                padding: 1.05rem 1.1rem;
            }
            .tarjeta      { border-top: 4px solid var(--azul-700); min-height: 142px; }
            .bloque-inicio { min-height: 160px; }

            .tarjeta h3, .bloque-inicio h3 { color: var(--azul-900); margin: 0 0 0.45rem; }
            .tarjeta p,  .bloque-inicio p  { color: var(--gris-600); line-height: 1.5; margin: 0; }

            .aviso-seccion {
                background: #fbfdff;
                border: 1px solid var(--borde);
                border-radius: 8px;
                color: var(--gris-600);
                font-size: 0.9rem;
                line-height: 1.5;
                margin-bottom: 1rem;
                padding: 0.75rem 1rem;
            }

            .barra-progreso {
                align-items: center;
                background: var(--superficie);
                border: 1px solid var(--borde);
                border-radius: 8px;
                display: flex;
                gap: 0.35rem;
                margin-bottom: 1rem;
                padding: 0.65rem 1rem;
            }
            .paso-chip {
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 750;
                padding: 0.25rem 0.7rem;
                white-space: nowrap;
            }
            .paso-activo   { background: var(--azul-700); color: #fff; }
            .paso-completo { background: var(--azul-100); color: var(--azul-900); border: 1px solid var(--borde); }
            .paso-pendiente{ background: #f0f4f8; color: var(--gris-600); border: 1px solid var(--borde); }
            .paso-separador{ color: var(--gris-600); font-size: 0.75rem; }

            div[data-testid="stMetric"] {
                background: #fff; border: 1px solid var(--borde);
                border-radius: 8px; box-shadow: 0 8px 18px rgba(16,40,71,0.06);
                padding: 0.75rem 0.9rem;
            }
            div[data-testid="stMetric"] label                         { color: var(--gris-600); font-weight: 650; }
            div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--azul-900); font-weight: 800; }

            label, .stNumberInput label, .stSelectbox label {
                color: var(--gris-900) !important; font-weight: 650;
            }

            .stApp .stButton > button,
            .stButton > button {
                background: var(--azul-700);
                border: 1px solid var(--azul-700);
                border-radius: 7px;
                color: #ffffff !important;
                font-weight: 800;
                min-height: 2.6rem;
                width: 100%;
            }
            .stApp .stButton > button:hover,
            .stButton > button:hover {
                background: var(--azul-800); border-color: var(--azul-800); color: #fff !important;
            }

            .btn-secundario > button {
                background: var(--superficie) !important;
                border: 1px solid var(--borde) !important;
                color: var(--azul-900) !important;
                font-weight: 700 !important;
            }
            .btn-secundario > button:hover {
                background: var(--azul-050) !important;
                border-color: var(--azul-700) !important;
                color: var(--azul-900) !important;
            }

            .tarjeta-resultado {
                background: var(--superficie);
                border: 1px solid var(--borde);
                border-left: 7px solid var(--azul-700);
                border-radius: 8px;
                box-shadow: 0 10px 24px rgba(16,40,71,0.07);
                margin-top: 0.75rem;
                padding: 1.2rem 1.25rem;
            }
            .etiqueta-resultado { color: var(--gris-600); font-weight: 750; margin-bottom: 0.25rem; }
            .tarjeta-resultado h2 { color: var(--azul-900); margin: 0 0 0.25rem; }
            .nivel-riesgo {
                background: var(--azul-100); border: 1px solid var(--borde); border-radius: 6px;
                color: var(--azul-900); display: inline-block; font-weight: 800;
                margin: 0.45rem 0 0.65rem; padding: 0.25rem 0.6rem;
            }
            .tarjeta-resultado p { color: var(--gris-600); line-height: 1.5; margin: 0; }

            .fila-probabilidad {
                background: var(--superficie); border: 1px solid var(--borde);
                border-radius: 8px; margin-bottom: 0.55rem; padding: 0.65rem 0.8rem;
            }
            .nombre-probabilidad { color: var(--azul-900); font-weight: 800; margin-bottom: 0.2rem; }
            .valor-probabilidad  { color: var(--gris-600); font-weight: 750; }

            .pie-aplicacion {
                background: var(--superficie); border: 1px solid var(--borde); border-radius: 8px;
                display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center;
                margin-top: 1.4rem; padding: 0.9rem 1rem;
                box-shadow: 0 6px 18px rgba(16,40,71,0.04);
            }
            .pie-aplicacion-contenido {
                display: flex; flex-wrap: wrap; justify-content: space-between;
                align-items: center; gap: 0.75rem; width: 100%;
            }
            .pie-aplicacion-chip {
                background: var(--azul-050); border: 1px solid var(--borde); border-radius: 999px;
                color: var(--azul-900); display: inline-flex; font-size: 0.8rem;
                font-weight: 750; line-height: 1; padding: 0.32rem 0.65rem; white-space: nowrap;
            }
            .pie-aplicacion-texto { color: var(--gris-700); font-size: 0.82rem; font-weight: 650; }
            .pie-aplicacion-autores { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 0.35rem; margin-left: auto; }

            @media (max-width: 760px) {
                .barra-aplicacion { align-items: flex-start; flex-direction: column; gap: 0.75rem; }
                .barra-progreso   { flex-wrap: wrap; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Carga del modelo ─────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model() -> tuple[Any, Any]:
    return load_model_artifacts(RUTA_MODELO, RUTA_PREPROCESADOR)


def validar_esquema(preprocesador: Any) -> None:
    columnas_esperadas = list(getattr(preprocesador, "feature_names_in_", COLUMNAS_MODELO))
    if len(columnas_esperadas) != len(COLUMNAS_MODELO):
        raise ValueError(
            f"El preprocesador espera {len(columnas_esperadas)} variables; "
            f"la aplicación provee {len(COLUMNAS_MODELO)}."
        )
    faltantes = [c for c in columnas_esperadas if c not in COLUMNAS_MODELO]
    sobrantes  = [c for c in COLUMNAS_MODELO  if c not in columnas_esperadas]
    if faltantes or sobrantes:
        raise ValueError(
            f"Esquema no coincide. Faltantes: {faltantes}. Sobrantes: {sobrantes}."
        )


# ── Helpers de formulario ────────────────────────────────────

def selector(etiqueta: str, opciones: dict[str, int], ayuda: str, default: str | None = None) -> int:
    etiquetas = list(opciones.keys())
    idx = etiquetas.index(default) if default in opciones else 0
    return opciones[st.selectbox(etiqueta, etiquetas, index=idx, help=ayuda)]


def entero(etiqueta: str, valor: int, minimo: int, maximo: int, ayuda: str) -> int:
    return int(st.number_input(etiqueta, min_value=minimo, max_value=maximo, value=valor, step=1, help=ayuda))


def decimal(etiqueta: str, valor: float, minimo: float, maximo: float, ayuda: str, paso: float = 0.1) -> float:
    return float(st.number_input(etiqueta, min_value=minimo, max_value=maximo, value=valor, step=paso, help=ayuda))


# ── Secciones del formulario ─────────────────────────────────

def seccion_personal() -> dict[str, Any]:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ms = selector("Estado civil", ESTADO_CIVIL, "Situación civil actual.", "Soltero/a")
    with col2:
        ge = selector("Género", GENERO, "Género registrado.", "Femenino")
    with col3:
        ae = entero("Edad al matricularse", 20, 15, 80, "Edad al ingresar a la universidad.")
    with col4:
        di = selector("Desplazado / foráneo", SI_NO, "Vive fuera de su municipio de origen.", "No")
    return {
        "Marital status":    ms,
        "Gender":            ge,
        "Age at enrollment": ae,
        "Displaced":         di,
    }


def seccion_ingreso() -> dict[str, Any]:
    # Fila 1
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        
        da = selector("Turno", TURNO, "Horario de asistencia.", "Diurno")
    with col2:
        
        ag = decimal("Nota de admisión", 130.0, 0.0, 200.0, "Calificación en el proceso de admisión.")
    with col3:
        pg = decimal("Nota de calificación previa", 130.0, 0.0, 200.0, "Calificación en la formación previa.")
    with col4:
        ao = entero("Orden de preferencia", 1, 0, 9, "Posición de esta carrera en la solicitud.")

    # Fila 2
    col1, col2, col3 = st.columns(3)
    with col1:
        co = selector("Carrera", CARRERAS, "Programa académico seleccionado.", "Turismo")
        
        
    with col2:
        pq = selector("Calificación previa", CALIFICACION_PREVIA,
                      "Nivel educativo previo al ingreso.", "Educación secundaria")
         
    with col3:
        am = selector("Modalidad de solicitud", MODALIDAD_SOLICITUD,
                      "Proceso de admisión utilizado.", "Segunda fase del contingente general")
        

    return {
        "Application mode":                am,
        "Application order":               ao,
        "Course":                          co,
        "Daytime/evening attendance":      da,
        "Previous qualification":          pq,
        "Previous qualification (grade)":  pg,
        "Admission grade":                 ag,
    }


def seccion_familiar() -> dict[str, Any]:
    col1, col2 = st.columns(2)
    with col1:
        mq = selector("Nivel educativo de la madre", NIVELES_EDUCATIVOS,
                      "Categoría educativa de la madre.", "Educación básica, tercer ciclo")
        mo = selector("Ocupación de la madre", OCUPACIONES,
                      "Actividad laboral de la madre.", "Trabajador de servicios personales, seguridad o ventas")
    with col2:
        fq = selector("Nivel educativo del padre", NIVELES_EDUCATIVOS,
                      "Categoría educativa del padre.", "Otro, 11.º año")
        fo = selector("Ocupación del padre", OCUPACIONES,
                      "Actividad laboral del padre.", "Trabajador no calificado")
    return {
        "Mother's qualification": mq,
        "Mother's occupation":    mo,
        "Father's qualification": fq,
        "Father's occupation":    fo,
    }


def seccion_financiera() -> dict[str, Any]:
    col1, col2, col3 = st.columns(3)
    with col1:
        de = selector("Deudor", SI_NO, "Deudas pendientes con la institución.", "No")
    with col2:
        tf = selector("Matrícula al día", SI_NO, "Pagos de matrícula al corriente.", "Sí")
    with col3:
        sh = selector("Becado", SI_NO, "Recibe beca académica.", "No")
    return {
        "Debtor":                  de,
        "Tuition fees up to date": tf,
        "Scholarship holder":      sh,
    }


def seccion_semestre(n: int) -> dict[str, Any]:
    label = "1st" if n == 1 else "2nd"
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        en = entero("Unidades matriculadas", 6, 0, 40, f"Unidades inscritas en el {n}.er semestre.")
    with col2:
        ev = entero("Evaluaciones realizadas", 6, 0, 80, f"Evaluaciones en el {n}.er semestre.")
    with col3:
        ap = entero("Unidades aprobadas", 5, 0, 40, f"Unidades aprobadas en el {n}.er semestre.")
    with col4:
        gr = decimal("Nota promedio", 12.0, 0.0, 20.0, f"Promedio del {n}.er semestre.")
    return {
        f"Curricular units {label} sem (enrolled)":    en,
        f"Curricular units {label} sem (evaluations)": ev,
        f"Curricular units {label} sem (approved)":    ap,
        f"Curricular units {label} sem (grade)":       gr,
    }


SECCIONES: list[Any] = [
    seccion_personal,
    seccion_ingreso,
    seccion_familiar,
    seccion_financiera,
    lambda: seccion_semestre(1),
    lambda: seccion_semestre(2),
]


# ── Componentes de UI ────────────────────────────────────────

def nombre_modelo(modelo: Any) -> str:
    nombres = {
        "LogisticRegression":         "Regresión logística",
        "RandomForestClassifier":     "Bosque aleatorio",
        "XGBClassifier":              "XGBoost",
        "DecisionTreeClassifier":     "Árbol de decisión",
        "KNeighborsClassifier":       "Vecinos más cercanos",
        "SVC":                        "Máquina de vectores de soporte",
        "GradientBoostingClassifier": "Potenciación por gradiente",
    }
    return nombres.get(type(modelo).__name__, "Modelo de clasificación entrenado")


def mostrar_header(con_volver: bool = False) -> None:
    col_marca, col_btn = st.columns([5, 1]) if con_volver else (st.columns([1])[0], None)
    with col_marca:
        st.markdown(
            """
            <div class="barra-aplicacion">
                <div class="marca-aplicacion">
                    <div class="marca-simbolo">SP</div>
                    <div>
                        <div class="marca-texto">Predicción de Rendimiento Académico</div>
                        <div class="marca-subtexto">Inteligencia Artificial · 2026</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if con_volver and col_btn is not None:
        with col_btn:
            st.write("")
            st.markdown('<div class="btn-secundario">', unsafe_allow_html=True)
            if st.button("Volver a Inicio", key="btn_volver_header"):
                st.session_state["pantalla"] = "inicio"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def mostrar_barra_progreso(paso_actual: int) -> None:
    chips = ""
    for i, titulo in enumerate(TITULOS_PASOS):
        if i < paso_actual:
            cls = "paso-completo"
        elif i == paso_actual:
            cls = "paso-activo"
        else:
            cls = "paso-pendiente"
        chips += f'<span class="paso-chip {cls}">{i + 1}. {titulo}</span>'
        if i < len(TITULOS_PASOS) - 1:
            chips += '<span class="paso-separador">›</span>'
    st.markdown(f'<div class="barra-progreso">{chips}</div>', unsafe_allow_html=True)


#def mostrar_footer() -> None:
#    st.markdown(
#        """
#        <div class="pie-aplicacion">
#            <div class="pie-aplicacion-contenido">
#                <span class="pie-aplicacion-texto">Proyecto Inteligencia Artificial, Predicción de rendimiento académico</span>
#                <div class="pie-aplicacion-autores">
#                    <span class="pie-aplicacion-chip">Jordan Ortiz Molina</span>
#                    <span class="pie-aplicacion-chip">Yenifer Mata Flores</span>
#                    <span class="pie-aplicacion-chip">Deyaneira Altamirano Cordero</span>
#                </div>
#            </div>
#        </div>
#        """,
#        unsafe_allow_html=True,
#    )


def mostrar_resultados(prediccion: str, probabilidades: dict[str, float], confianza: float) -> None:
    contenido     = CONTENIDO_RESULTADOS[prediccion]
    prediccion_es = TRADUCCION_CLASES[prediccion]

    st.subheader("Resultados de la predicción")
    st.markdown(
        f"""
        <div class="tarjeta-resultado" style="border-left-color:{contenido['color']};">
            <div class="etiqueta-resultado">{contenido['estado']}</div>
            <h2>{prediccion_es}</h2>
            <div class="nivel-riesgo">Nivel de riesgo: {contenido['riesgo']}</div>
            <p>{contenido['descripcion']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2 = st.columns(2)
    col1.metric("Clase predicha",     prediccion_es)
    col2.metric("Nivel de confianza", f"{confianza * 100:.2f}%")

    st.write("")
    st.subheader("Probabilidades por clase")
    for clase in CLASES_MODELO:
        prob = probabilidades.get(clase, 0.0)
        st.markdown(
            f"""
            <div class="fila-probabilidad">
                <div class="nombre-probabilidad">{TRADUCCION_CLASES[clase]}</div>
                <div class="valor-probabilidad">{prob * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(prob, 0.0), 1.0))


# ── Pantallas ────────────────────────────────────────────────

def pantalla_inicio(modelo: Any, preprocesador: Any) -> None:
    mostrar_header(con_volver=False)
    st.markdown(
        """
        <div class="aviso-seccion">
            Este sistema permite predecir el estado académico de un estudiante univesitario, clasificando entre desertar,
            seguir matriculado o graduarse.
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="bloque-inicio">
                <h3>¿Qué hace el sistema?</h3>
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
                <h3>¿Para qué funciona?</h3>
                <p>Su propósito es apoyar el análisis institucional y facilitar la identificación temprana de perfiles que podrían requerir acompañamiento académico</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
             <div class="bloque-inicio">
                <h3>¿Cómo se entrenó?</h3>
                <p>Se utilizó el dataset UCI de deserción y éxito académico para entrenar un modelo mediante regresión logística.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")
    _, col_cta, _ = st.columns([2, 1, 2])
    with col_cta:
        if st.button("Ir al formulario →", type="primary"):
            st.session_state["pantalla"] = "formulario"
            st.session_state["paso"]     = 0
            st.session_state["valores"]  = {}
            st.rerun()
    #mostrar_footer()


def pantalla_formulario(modelo: Any, preprocesador: Any) -> None:
    if "paso"    not in st.session_state: st.session_state["paso"]    = 0
    if "valores" not in st.session_state: st.session_state["valores"] = {}

    paso = st.session_state["paso"]
    mostrar_header(con_volver=True)
    mostrar_barra_progreso(paso)

    st.subheader(f"Paso {paso + 1} de {TOTAL_PASOS} — {TITULOS_PASOS[paso]}")
    valores_paso = SECCIONES[paso]()

    st.write("")
    st.divider()

    es_ultimo = paso == TOTAL_PASOS - 1

    if paso == 0:
        _, col_sig, _ = st.columns([3, 1, 0.01])
    else:
        col_ant, _, col_sig = st.columns([1, 2, 1])
        with col_ant:
            st.markdown('<div class="btn-secundario">', unsafe_allow_html=True)
            if st.button("← Anterior", key="btn_anterior"):
                st.session_state["valores"].update(valores_paso)
                st.session_state["paso"] -= 1
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with col_sig:
        etiqueta_sig = "Generar predicción" if es_ultimo else "Siguiente →"
        if st.button(etiqueta_sig, type="primary", key="btn_siguiente"):
            st.session_state["valores"].update(valores_paso)
            if es_ultimo:
                st.session_state["pantalla"] = "resultado"
            else:
                st.session_state["paso"] += 1
            st.rerun()

    #mostrar_footer()


def pantalla_resultado(modelo: Any, preprocesador: Any) -> None:
    mostrar_header(con_volver=True)

    valores = st.session_state.get("valores", {})

    try:
        datos = prepare_input_data(valores, feature_columns=COLUMNAS_MODELO)
        prediccion, probabilidades, confianza = predict_student_status(
            modelo, preprocesador, datos,
            feature_columns=COLUMNAS_MODELO,
        )
        mostrar_resultados(prediccion, probabilidades, confianza)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"No fue posible generar la predicción: {exc}")
        return

    st.write("")
    _, col_nuevo, _ = st.columns([2, 1, 2])
    with col_nuevo:
        st.markdown('<div class="btn-secundario">', unsafe_allow_html=True)
        if st.button("Nueva predicción", key="btn_nuevo"):
            st.session_state["pantalla"] = "formulario"
            st.session_state["paso"]     = 0
            st.session_state["valores"]  = {}
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    valores = st.session_state.get("valores", {})

    #mostrar_footer()


# ── Main ─────────────────────────────────────────────────────

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
        st.error("No fue posible cargar el modelo o su esquema de datos.")
        st.error(str(exc))
        return

    if "pantalla" not in st.session_state:
        st.session_state["pantalla"] = "inicio"

    pantalla = st.session_state["pantalla"]

    if pantalla == "inicio":
        pantalla_inicio(modelo, preprocesador)
    elif pantalla == "formulario":
        pantalla_formulario(modelo, preprocesador)
    elif pantalla == "resultado":
        pantalla_resultado(modelo, preprocesador)
        


if __name__ == "__main__":
    main()