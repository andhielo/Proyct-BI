"""
PLATAFORMA INTEGRAL DE ANALITICA UNIVERSITARIA
BI, Big Data e IA Etica para la mejora de la Salud Mental en Universitarios de Lima Norte

Ejecutar localmente:
    pip install -r requirements.txt
    streamlit run app.py

Desplegar gratis en Streamlit Community Cloud:
    1. Sube esta carpeta a un repo de GitHub
    2. share.streamlit.io -> New app -> selecciona el repo -> main file: app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import io
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ============================================================
# CONFIGURACION GENERAL
# ============================================================
st.set_page_config(
    page_title="BI Salud Mental Lima Norte",
    page_icon="🧠",
    layout="wide",
)

DB_PATH = "/home/claude/proyecto_bi_salud_mental/data/warehouse.db"
RAW_CSV_DEFAULT = "/home/claude/proyecto_bi_salud_mental/data/encuestas_salud_mental_lima_norte.csv"

PASOS = [
    "Pantalla General",
    "1. Fuentes de Datos",
    "2. Staging Area",
    "3. Proceso ETL",
    "4. Data Warehouse",
    "5. Capa de IA Etica",
    "6. Capa Semantica & KPIs",
    "7. Visualizacion BI",
]

# ============================================================
# ESTADO DE SESION (simula el flujo persistente entre pasos)
# ============================================================
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_staging" not in st.session_state:
    st.session_state.df_staging = None
if "df_errores" not in st.session_state:
    st.session_state.df_errores = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "modelo_entrenado" not in st.session_state:
    st.session_state.modelo_entrenado = None
if "metricas_modelo" not in st.session_state:
    st.session_state.metricas_modelo = None
if "df_pred" not in st.session_state:
    st.session_state.df_pred = None

# ============================================================
# ESTILOS (paleta institucional simple)
# ============================================================
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1b2a4a 0%, #2b4a7a 100%);
    padding: 2rem 2.2rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.5rem;
}
.step-badge {
    display:inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight:600; margin-bottom:6px;
}
.kpi-card {
    background:#f7f9fc; border:1px solid #e3e8f0; border-radius:12px;
    padding: 1rem 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR - NAVEGACION (misma logica que la plataforma de referencia)
# ============================================================
with st.sidebar:
    st.markdown("### 🧠 MONITOREO DE\nSALUD MENTAL")
    st.caption("Universitarios de Lima Norte")
    st.markdown("---")
    paso = st.radio("**7 PASOS DEL FLUJO BI**", PASOS, label_visibility="visible")
    st.markdown("---")
    st.caption("Curso: Big Data / Business Intelligence")
    st.caption("Stack: Streamlit + SQLite + scikit-learn + Plotly")

# ============================================================
# PASO 0: PANTALLA GENERAL
# ============================================================
if paso == "Pantalla General":
    st.markdown("""
    <div class="main-header">
        <h1>PLATAFORMA INTEGRAL DE ANALITICA UNIVERSITARIA</h1>
        <h3 style="font-weight:400;">BI, Big Data e IA Etica para la mejora de la Salud Mental
        en Universitarios de Lima Norte</h3>
        <p>Arquitectura end-to-end: captura, staging, ETL, warehouse, IA etica y visualizacion ejecutiva.</p>
        <span style="background:#2ecc71; padding:4px 12px; border-radius:20px; font-size:0.8rem;">
        ● CSV + SQLite + Streamlit + Scikit-learn</span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    pasos_info = [
        ("Paso 1", "FUENTES DE DATOS", "Fuentes de Datos", "Captura de encuestas SISAP-U", "Registros crudos"),
        ("Paso 2", "STAGING AREA", "Staging Area", "Validacion y control de calidad", "Errores aislados"),
        ("Paso 3", "PROCESO ETL", "Proceso ETL", "Limpieza y transformacion", "Transformacion base"),
        ("Paso 4", "DATA WAREHOUSE", "Data Warehouse", "Carga en SQLite (modelo estrella)", "Modelo analitico"),
    ]
    for c, (badge, tag, titulo, desc, foot) in zip(cols, pasos_info):
        with c:
            st.markdown(f"**{badge}** &nbsp; `{tag}`")
            st.markdown(f"#### {titulo}")
            st.caption(desc)
            st.caption(f"🟢 {foot}")

    cols2 = st.columns(4)
    pasos_info2 = [
        ("Paso 5", "CAPA DE IA ETICA", "Capa de IA Etica", "Prediccion de riesgo psicoemocional", "Modelo con auditoria de sesgo"),
        ("Paso 6", "CAPA SEMANTICA & KPIS", "Capa Semantica & KPIs", "Indicadores de bienestar universitario", "Medidas de negocio"),
        ("Paso 7", "VISUALIZACION BI", "Visualizacion BI", "Dashboard ejecutivo final", "Panel ejecutivo"),
        ("", "", "", "", ""),
    ]
    for c, (badge, tag, titulo, desc, foot) in zip(cols2, pasos_info2):
        with c:
            if titulo:
                st.markdown(f"**{badge}** &nbsp; `{tag}`")
                st.markdown(f"#### {titulo}")
                st.caption(desc)
                st.caption(f"🟠 {foot}")

    st.markdown("---")
    st.markdown("""
    ##### Objetivo del proyecto
    Construir un pipeline de analitica de datos (Big Data / BI) que permita a las universidades
    de Lima Norte **detectar tempranamente patrones de riesgo psicoemocional** (estres, ansiedad,
    animo bajo) en su poblacion estudiantil, usando datos de encuestas de bienestar, e informar
    decisiones de bienestar universitario con **etica y transparencia algoritmica** (sin decisiones
    automatizadas sobre personas, solo agregados y alertas para equipos de bienestar).
    """)

# ============================================================
# PASO 1: FUENTES DE DATOS
# ============================================================
elif paso == "1. Fuentes de Datos":
    st.subheader("Fuentes de Datos")
    st.caption("Carga el archivo de encuestas de bienestar universitario (Excel o CSV)")

    archivo = st.file_uploader("Subir archivo", type=["csv", "xlsx", "xls"])

    usar_demo = st.checkbox("Usar dataset de ejemplo (encuestas_salud_mental_lima_norte.csv)", value=(archivo is None))

    df = None
    if archivo is not None:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)
    elif usar_demo:
        df = pd.read_csv(RAW_CSV_DEFAULT)

    if df is not None:
        st.session_state.df_raw = df
        st.success(f"Registros leidos: {len(df)} | Columnas: {df.shape[1]}")
        st.dataframe(df.head(20), use_container_width=True)
        st.info("➡ Continua a **Staging Area** para validar la calidad de estos datos.")
    else:
        st.warning("Sube un archivo o activa el dataset de ejemplo para continuar.")

# ============================================================
# PASO 2: STAGING AREA
# ============================================================
elif paso == "2. Staging Area":
    st.subheader("Staging Area")
    st.caption("Validacion y control de calidad de los datos crudos")

    if st.session_state.df_raw is None:
        st.warning("Primero carga datos en el Paso 1 (Fuentes de Datos).")
    else:
        df = st.session_state.df_raw.copy()

        # Reglas de validacion
        errores = []

        # 1. Nulos
        nulos = df.isna().sum()
        nulos = nulos[nulos > 0]

        # 2. Duplicados
        n_dup = df.duplicated(subset=["id_encuesta"]).sum()

        # 3. Edad fuera de rango (15-40 esperado en poblacion universitaria)
        edad_num = pd.to_numeric(df["edad"], errors="coerce")
        mask_edad_invalida = ~edad_num.between(15, 40)

        # 4. Horas de sueno fuera de rango fisiologico (0-14h)
        sueno_num = pd.to_numeric(df["horas_sueno_promedio"], errors="coerce")
        mask_sueno_invalido = ~sueno_num.between(0, 14) & sueno_num.notna()

        # 5. Fecha no parseable directamente
        fecha_parsed = pd.to_datetime(df["fecha_encuesta"], errors="coerce", format="mixed")
        mask_fecha_invalida = fecha_parsed.isna()

        mask_error = mask_edad_invalida.fillna(False) | mask_sueno_invalido.fillna(False)
        df_errores = df[mask_error | df.duplicated(subset=["id_encuesta"], keep=False)]
        df_ok = df[~mask_error]

        st.session_state.df_staging = df_ok
        st.session_state.df_errores = df_errores

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros totales", len(df))
        c2.metric("Duplicados detectados", int(n_dup))
        c3.metric("Edad fuera de rango", int(mask_edad_invalida.sum()))
        c4.metric("Horas sueno invalidas", int(mask_sueno_invalido.sum()))

        st.markdown("##### Valores nulos por columna")
        if len(nulos) > 0:
            st.bar_chart(nulos)
        else:
            st.caption("No se detectaron valores nulos.")

        st.markdown("##### Fechas con formato inconsistente (texto vs. datetime)")
        st.write(f"Se detectaron **{mask_fecha_invalida.sum()}** filas con formato de fecha no estandar "
                 f"(ej. `dd/mm/aaaa` en vez de ISO). Se corregiran en el Paso 3 (ETL).")

        st.markdown("##### Registros aislados en cuarentena (Paso 2 → no pasan al warehouse aun)")
        st.dataframe(df_errores.head(15), use_container_width=True)

        st.success(f"✅ {len(df_ok)} registros validados listos para el Proceso ETL.")

# ============================================================
# PASO 3: PROCESO ETL
# ============================================================
elif paso == "3. Proceso ETL":
    st.subheader("Proceso ETL")
    st.caption("Limpieza, estandarizacion y transformacion de variables")

    if st.session_state.df_staging is None:
        st.warning("Primero ejecuta el Paso 2 (Staging Area).")
    else:
        df = st.session_state.df_staging.copy()

        with st.expander("Ver transformaciones aplicadas", expanded=True):
            st.markdown("""
            1. **Estandarizacion de texto**: se normalizan mayusculas/espacios en `universidad`.
            2. **Parseo de fechas**: se homogenizan formatos mixtos (`dd/mm/aaaa` y datetime) a ISO.
            3. **Eliminacion de duplicados** por `id_encuesta`.
            4. **Imputacion de nulos**: variables numericas → mediana por facultad; categoricas → moda.
            5. **Ingenieria de variables**: se calcula el `indice_riesgo_psicoemocional` (0-10) combinando
               estres, ansiedad, animo (invertido) y apoyo social.
            6. **Categorizacion**: se etiqueta a cada estudiante en **Riesgo Bajo / Medio / Alto**.
            """)

        # 1. Texto
        df["universidad"] = df["universidad"].str.strip().str.title()

        # 2. Fechas
        df["fecha_encuesta"] = pd.to_datetime(df["fecha_encuesta"], errors="coerce", format="mixed")

        # 3. Duplicados
        antes = len(df)
        df = df.drop_duplicates(subset=["id_encuesta"])
        dedup = antes - len(df)

        # 4. Imputacion
        num_cols = ["horas_sueno_promedio", "apoyo_social", "promedio_academico", "calidad_alimentacion"]
        for col in num_cols:
            df[col] = df.groupby("facultad")[col].transform(lambda s: s.fillna(s.median()))
            df[col] = df[col].fillna(df[col].median())
        df["sexo"] = df["sexo"].fillna(df["sexo"].mode()[0])

        # 5. Indice de riesgo psicoemocional (0-10, mayor = mas riesgo)
        df["indice_riesgo_psicoemocional"] = (
            df["nivel_estres"] * 0.35
            + df["nivel_ansiedad"] * 0.35
            + (10 - df["nivel_animo"]) * 0.20
            + (10 - df["apoyo_social"]) * 0.10
        ).round(2)

        # 6. Categorizacion
        def categorizar(v):
            if v < 4.5:
                return "Bajo"
            elif v < 7:
                return "Medio"
            else:
                return "Alto"
        df["nivel_riesgo"] = df["indice_riesgo_psicoemocional"].apply(categorizar)

        st.session_state.df_clean = df

        c1, c2, c3 = st.columns(3)
        c1.metric("Registros tras limpieza", len(df))
        c2.metric("Duplicados eliminados", dedup)
        c3.metric("Variables nuevas creadas", 2)

        st.markdown("##### Muestra de datos transformados")
        st.dataframe(
            df[["id_encuesta", "universidad", "facultad", "fecha_encuesta",
                "indice_riesgo_psicoemocional", "nivel_riesgo"]].head(15),
            use_container_width=True,
        )

        st.markdown("##### Distribucion de niveles de riesgo (post-ETL)")
        fig = px.histogram(df, x="nivel_riesgo", color="nivel_riesgo",
                            category_orders={"nivel_riesgo": ["Bajo", "Medio", "Alto"]},
                            color_discrete_map={"Bajo": "#2ecc71", "Medio": "#f39c12", "Alto": "#e74c3c"})
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.success("✅ Datos listos para cargar en el Data Warehouse.")

# ============================================================
# PASO 4: DATA WAREHOUSE
# ============================================================
elif paso == "4. Data Warehouse":
    st.subheader("Data Warehouse (SQLite — modelo estrella)")

    if st.session_state.df_clean is None:
        st.warning("Primero ejecuta el Paso 3 (Proceso ETL).")
    else:
        df = st.session_state.df_clean.copy()

        st.markdown("""
        <div class="kpi-card">
        <b>ESTADO DEL DATA MART:</b> Listo para carga &nbsp;|&nbsp;
        Modelo logico: <b>Esquema Estrella</b> (FACT_SALUD_MENTAL + dimensiones)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Modelo logico — Esquema Estrella")
        st.code("""
DIM_ESTUDIANTE                    DIM_TIEMPO
----------------                  ----------------
PK id_estudiante                  PK id_tiempo
   universidad                       fecha
   facultad                          anio / mes / dia
   ciclo, edad, sexo

DIM_UNIVERSIDAD                              FACT_SALUD_MENTAL (tabla de hechos)
----------------                    ---->    ----------------------------------
PK id_universidad                            FK id_estudiante
   nombre_universidad                        FK id_tiempo
                                              FK id_universidad
                                              nivel_estres, nivel_ansiedad, nivel_animo
                                              indice_riesgo_psicoemocional
                                              nivel_riesgo
                                              promedio_academico
        """, language="text")

        col1, col2 = st.columns(2)
        with col1:
            reemplazar = st.checkbox("Reemplazar datos anteriores en el Warehouse", value=True)
        with col2:
            if st.button("💾 Guardar Data Warehouse (SQLite)", type="primary"):
                conn = sqlite3.connect(DB_PATH)

                dim_estudiante = df[["id_encuesta", "universidad", "facultad", "ciclo", "edad", "sexo"]].copy()
                dim_estudiante.rename(columns={"id_encuesta": "id_estudiante"}, inplace=True)

                dim_tiempo = df[["fecha_encuesta"]].drop_duplicates().reset_index(drop=True)
                dim_tiempo["id_tiempo"] = dim_tiempo.index + 1
                dim_tiempo["anio"] = pd.to_datetime(dim_tiempo["fecha_encuesta"]).dt.year
                dim_tiempo["mes"] = pd.to_datetime(dim_tiempo["fecha_encuesta"]).dt.month
                dim_tiempo["dia"] = pd.to_datetime(dim_tiempo["fecha_encuesta"]).dt.day

                fact = df.merge(dim_tiempo, on="fecha_encuesta", how="left")
                fact_cols = [
                    "id_encuesta", "id_tiempo", "universidad", "facultad",
                    "nivel_estres", "nivel_ansiedad", "nivel_animo", "apoyo_social",
                    "indice_riesgo_psicoemocional", "nivel_riesgo", "promedio_academico",
                    "horas_sueno_promedio", "uso_redes_sociales_horas", "actividad_fisica_horas_sem",
                ]
                fact_table = fact[fact_cols].rename(columns={"id_encuesta": "id_estudiante"})

                if_exists_mode = "replace" if reemplazar else "append"
                dim_estudiante.to_sql("dim_estudiante", conn, if_exists=if_exists_mode, index=False)
                dim_tiempo.to_sql("dim_tiempo", conn, if_exists=if_exists_mode, index=False)
                fact_table.to_sql("fact_salud_mental", conn, if_exists=if_exists_mode, index=False)
                conn.close()

                st.success(f"✅ Guardado en {DB_PATH}: {len(fact_table)} filas en fact_salud_mental")

        st.markdown("##### Vista previa de la tabla de hechos (fact_salud_mental)")
        st.dataframe(
            df[["id_encuesta", "universidad", "facultad", "nivel_estres", "nivel_ansiedad",
                "indice_riesgo_psicoemocional", "nivel_riesgo", "promedio_academico"]].head(10),
            use_container_width=True,
        )
        st.caption(f"Filas analiticas totales disponibles: **{len(df)}**")

# ============================================================
# PASO 5: CAPA DE IA ETICA
# ============================================================
elif paso == "5. Capa de IA Etica":
    st.subheader("Capa de IA Etica — Prediccion de Riesgo Psicoemocional")

    if st.session_state.df_clean is None:
        st.warning("Primero ejecuta el Paso 3 (Proceso ETL).")
    else:
        df = st.session_state.df_clean.copy()

        st.info("""
        **Principios de IA etica aplicados en esta capa:**
        Datos anonimizados (sin nombres/DNI) · el modelo **no toma decisiones automatizadas
        sobre personas** (solo genera alertas agregadas para el area de Bienestar Universitario) ·
        se audita el desempeño del modelo por **subgrupos** (sexo, universidad) para detectar sesgos ·
        se reportan las variables mas influyentes para mantener **transparencia algoritmica**.
        """)

        features = [
            "horas_sueno_promedio", "horas_trabajo_semanal", "horas_estudio_diario",
            "uso_redes_sociales_horas", "actividad_fisica_horas_sem", "tiempo_traslado_min",
            "apoyo_social", "calidad_alimentacion", "procrastinacion", "satisfaccion_vida", "ciclo", "edad",
        ]
        X = df[features].fillna(df[features].median())
        y = df["nivel_riesgo"]

        if st.button("🧠 Entrenar modelo (Random Forest)", type="primary"):
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X, y, df.index, test_size=0.25, random_state=42, stratify=y
            )
            modelo = RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42, class_weight="balanced")
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")

            st.session_state.modelo_entrenado = modelo
            st.session_state.metricas_modelo = {
                "acc": acc, "f1": f1, "y_test": y_test, "y_pred": y_pred,
                "idx_test": idx_test, "features": features,
            }

        if st.session_state.modelo_entrenado is not None:
            m = st.session_state.metricas_modelo
            c1, c2 = st.columns(2)
            c1.metric("Exactitud (accuracy)", f"{m['acc']*100:.1f}%")
            c2.metric("F1-score (macro)", f"{m['f1']*100:.1f}%")

            st.markdown("##### Importancia de variables (transparencia del modelo)")
            imp = pd.Series(st.session_state.modelo_entrenado.feature_importances_, index=m["features"])
            imp = imp.sort_values(ascending=True)
            fig_imp = px.bar(imp, orientation="h", labels={"value": "Importancia", "index": "Variable"})
            fig_imp.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("##### Matriz de confusion")
            labels = ["Bajo", "Medio", "Alto"]
            cm = confusion_matrix(m["y_test"], m["y_pred"], labels=labels)
            fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True,
                                labels=dict(x="Prediccion", y="Real", color="Casos"),
                                color_continuous_scale="Blues")
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("##### Auditoria de sesgo por subgrupo (sexo)")
            df_test = df.loc[m["idx_test"]].copy()
            df_test["pred"] = m["y_pred"]
            df_test["correcto"] = (df_test["pred"] == df_test["nivel_riesgo"]).astype(int)
            audit = df_test.groupby("sexo")["correcto"].mean().reset_index()
            audit.columns = ["sexo", "exactitud"]
            fig_audit = px.bar(audit, x="sexo", y="exactitud", color="sexo", range_y=[0, 1])
            fig_audit.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig_audit, use_container_width=True)
            st.caption("Se espera que la exactitud sea similar entre subgrupos; una brecha grande "
                       "indicaria sesgo del modelo hacia algun grupo y requeriria correccion.")

            st.session_state.df_pred = df

# ============================================================
# PASO 6: CAPA SEMANTICA & KPIs
# ============================================================
elif paso == "6. Capa Semantica & KPIs":
    st.subheader("Capa Semantica & KPIs — Indicadores de Bienestar Universitario")

    if st.session_state.df_clean is None:
        st.warning("Primero ejecuta el Paso 3 (Proceso ETL).")
    else:
        df = st.session_state.df_clean.copy()

        st.markdown("##### Definicion de indicadores de negocio (capa semantica)")
        st.markdown("""
        | KPI | Definicion | Uso |
        |---|---|---|
        | **% Riesgo Alto** | Estudiantes con `indice_riesgo_psicoemocional` ≥ 7 | Priorizar campañas de bienestar |
        | **Indice de sueño saludable** | % con ≥ 7h de sueño promedio | Salud fisica-mental |
        | **Correlacion Estres–Rendimiento** | Correlacion entre `nivel_estres` y `promedio_academico` | Argumentar inversion en bienestar |
        | **Cobertura de apoyo psicologico** | % que ha accedido a apoyo psicologico previo | Medir alcance de servicios |
        """)

        pct_alto = (df["nivel_riesgo"] == "Alto").mean() * 100
        pct_sueno_ok = (df["horas_sueno_promedio"] >= 7).mean() * 100
        corr = df["nivel_estres"].corr(df["promedio_academico"])
        cobertura = (df["apoyo_psicologico_previo"] == "Si").mean() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("% Riesgo Alto", f"{pct_alto:.1f}%")
        c2.metric("Sueño saludable (≥7h)", f"{pct_sueno_ok:.1f}%")
        c3.metric("Corr. Estres–Rendimiento", f"{corr:.2f}")
        c4.metric("Cobertura apoyo psicologico", f"{cobertura:.1f}%")

        st.markdown("##### Riesgo alto por universidad")
        tabla = df.groupby("universidad").agg(
            estudiantes=("id_encuesta", "count"),
            pct_riesgo_alto=("nivel_riesgo", lambda s: (s == "Alto").mean() * 100),
            promedio_academico=("promedio_academico", "mean"),
        ).reset_index().sort_values("pct_riesgo_alto", ascending=False)
        st.dataframe(tabla.style.format({"pct_riesgo_alto": "{:.1f}%", "promedio_academico": "{:.2f}"}),
                     use_container_width=True)

# ============================================================
# PASO 7: VISUALIZACION BI
# ============================================================
elif paso == "7. Visualizacion BI":
    st.subheader("Visualizacion BI — Dashboard Ejecutivo")

    if st.session_state.df_clean is None:
        st.warning("Primero ejecuta el Paso 3 (Proceso ETL).")
    else:
        df = st.session_state.df_clean.copy()

        with st.expander("Filtros", expanded=True):
            f1, f2, f3 = st.columns(3)
            univ_sel = f1.multiselect("Universidad", df["universidad"].unique(), default=list(df["universidad"].unique()))
            fac_sel = f2.multiselect("Facultad", df["facultad"].unique(), default=list(df["facultad"].unique()))
            riesgo_sel = f3.multiselect("Nivel de riesgo", ["Bajo", "Medio", "Alto"], default=["Bajo", "Medio", "Alto"])

        dff = df[df["universidad"].isin(univ_sel) & df["facultad"].isin(fac_sel) & df["nivel_riesgo"].isin(riesgo_sel)]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estudiantes analizados", len(dff))
        c2.metric("% Riesgo Alto", f"{(dff['nivel_riesgo']=='Alto').mean()*100:.1f}%")
        c3.metric("Estres promedio", f"{dff['nivel_estres'].mean():.1f}/10")
        c4.metric("Promedio academico", f"{dff['promedio_academico'].mean():.1f}/20")

        colA, colB = st.columns(2)
        with colA:
            fig1 = px.pie(dff, names="nivel_riesgo", title="Distribucion del nivel de riesgo",
                          color="nivel_riesgo",
                          color_discrete_map={"Bajo": "#2ecc71", "Medio": "#f39c12", "Alto": "#e74c3c"})
            st.plotly_chart(fig1, use_container_width=True)
        with colB:
            fig2 = px.bar(
                dff.groupby("facultad")["indice_riesgo_psicoemocional"].mean().sort_values().reset_index(),
                x="indice_riesgo_psicoemocional", y="facultad", orientation="h",
                title="Indice de riesgo promedio por facultad",
            )
            st.plotly_chart(fig2, use_container_width=True)

        colC, colD = st.columns(2)
        with colC:
            fig3 = px.scatter(dff, x="nivel_estres", y="promedio_academico", color="nivel_riesgo",
                               trendline="ols", title="Estres vs. Rendimiento academico",
                               color_discrete_map={"Bajo": "#2ecc71", "Medio": "#f39c12", "Alto": "#e74c3c"})
            st.plotly_chart(fig3, use_container_width=True)
        with colD:
            dff["mes"] = pd.to_datetime(dff["fecha_encuesta"]).dt.to_period("M").astype(str)
            tend = dff.groupby("mes")["indice_riesgo_psicoemocional"].mean().reset_index()
            fig4 = px.line(tend, x="mes", y="indice_riesgo_psicoemocional", markers=True,
                            title="Tendencia mensual del indice de riesgo")
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("##### Datos filtrados")
        st.dataframe(dff.head(50), use_container_width=True)