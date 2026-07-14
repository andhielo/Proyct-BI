"""
PLATAFORMA INTEGRAL DE ANALITICA UNIVERSITARIA
BI, Big Data e IA Etica para la mejora de la Salud Mental en Universitarios de Lima Norte

Version 2: TODO el pipeline se persiste en una base de datos SQL (SQLite) real:
raw_data -> staging_validos / staging_errores -> clean_data -> dim_*/fact_* (warehouse)
-> modelo (pickle) + predicciones_test -> KPIs y dashboard leen siempre desde la BD.

Esto significa que si recargas la pagina o el servidor se reinicia, el avance NO se pierde:
solo necesitas volver a pasar por cada paso (los botones ya usaran lo que esta guardado en BD).

Ejecutar localmente:
    pip install -r requirements.txt
    streamlit run app.py

Desplegar en Streamlit Community Cloud:
    1. Sube TODA esta carpeta (app.py, db_utils.py, requirements.txt, data/) a un repo de GitHub.
       IMPORTANTE: requirements.txt y app.py deben estar en la RAIZ del repo.
    2. share.streamlit.io -> New app -> selecciona el repo -> Main file path: app.py
"""

import json
import pandas as pd
import numpy as np
import streamlit as st

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import db_utils as dbu

# ============================================================
# CONFIGURACION GENERAL
# ============================================================
st.set_page_config(page_title="BI Salud Mental Lima Norte", page_icon="🧠", layout="wide")
dbu.init_db()

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

st.markdown("""
<style>
.main-header { background: linear-gradient(135deg, #1b2a4a 0%, #2b4a7a 100%);
    padding: 2rem 2.2rem; border-radius: 14px; color: white; margin-bottom: 1.5rem; }
.kpi-card { background:#f7f9fc; border:1px solid #e3e8f0; border-radius:12px; padding: 1rem 1.2rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🧠 MONITOREO DE\nSALUD MENTAL")
    st.caption("Universitarios de Lima Norte")
    st.markdown("---")
    paso = st.radio("**7 PASOS DEL FLUJO BI**", PASOS)
    st.markdown("---")
    st.caption("Datos persistidos en SQLite: `data/warehouse.db`")
    if st.button("🗑️ Reiniciar todo el pipeline"):
        dbu.reset_pipeline()
        st.success("Pipeline reiniciado. Vuelve al Paso 1.")
        st.rerun()

# ============================================================
# PASO 0
# ============================================================
if paso == "Pantalla General":
    st.markdown("""
    <div class="main-header">
        <h1>PLATAFORMA INTEGRAL DE ANALITICA UNIVERSITARIA</h1>
        <h3 style="font-weight:400;">BI, Big Data e IA Etica para la mejora de la Salud Mental
        en Universitarios de Lima Norte</h3>
        <p>Arquitectura end-to-end: captura, staging, ETL, warehouse, IA etica y visualizacion ejecutiva.</p>
        <span style="background:#2ecc71; padding:4px 12px; border-radius:20px; font-size:0.8rem;">
        ● CSV + SQLite (persistente) + Streamlit + Scikit-learn</span>
    </div>
    """, unsafe_allow_html=True)

    estado = [
        ("Paso 1", "Fuentes de Datos", dbu.table_exists("raw_data")),
        ("Paso 2", "Staging Area", dbu.table_exists("staging_validos")),
        ("Paso 3", "Proceso ETL", dbu.table_exists("clean_data")),
        ("Paso 4", "Data Warehouse", dbu.table_exists("fact_salud_mental")),
    ]
    cols = st.columns(4)
    for c, (badge, titulo, listo) in zip(cols, estado):
        with c:
            st.markdown(f"**{badge}**")
            st.markdown(f"#### {titulo}")
            st.caption("✅ Completado (en BD)" if listo else "⏳ Pendiente")

    estado2 = [
        ("Paso 5", "Capa de IA Etica", dbu.load_model("riesgo_rf")[0] is not None),
        ("Paso 6", "Capa Semantica & KPIs", dbu.table_exists("clean_data")),
        ("Paso 7", "Visualizacion BI", dbu.table_exists("clean_data")),
    ]
    cols2 = st.columns(4)
    for c, (badge, titulo, listo) in zip(cols2[:3], estado2):
        with c:
            st.markdown(f"**{badge}**")
            st.markdown(f"#### {titulo}")
            st.caption("✅ Completado (en BD)" if listo else "⏳ Pendiente")

    st.markdown("---")
    st.markdown("""
    ##### Objetivo del proyecto
    Construir un pipeline de analitica de datos (Big Data / BI) que permita a las universidades
    de Lima Norte **detectar tempranamente patrones de riesgo psicoemocional** en su poblacion
    estudiantil, usando datos de encuestas de bienestar, e informar decisiones con **etica y
    transparencia algoritmica** (sin decisiones automatizadas sobre personas).

    Todo el flujo (datos crudos, validados, limpios, modelo entrenado y predicciones) se guarda
    en una **base de datos SQL real** (`data/warehouse.db`), no solo en memoria: puedes cerrar la
    app y al volver a entrar, cada paso siguiente ya encontrara lo que el paso anterior guardo.
    """)

# ============================================================
# PASO 1: FUENTES DE DATOS
# ============================================================
elif paso == "1. Fuentes de Datos":
    st.subheader("Fuentes de Datos")
    st.caption("Carga el archivo de encuestas de bienestar universitario (Excel o CSV)")

    archivo = st.file_uploader("Subir archivo", type=["csv", "xlsx", "xls"])
    usar_demo = st.checkbox("Usar dataset de ejemplo (encuestas_salud_mental_lima_norte.csv)",
                             value=(archivo is None))

    df = None
    if archivo is not None:
        df = pd.read_csv(archivo) if archivo.name.endswith(".csv") else pd.read_excel(archivo)
    elif usar_demo:
        df = pd.read_csv(dbu.RAW_CSV_DEFAULT)

    if df is not None:
        st.success(f"Registros leidos: {len(df)} | Columnas: {df.shape[1]}")
        st.dataframe(df.head(20), use_container_width=True)
        if st.button("💾 Guardar en base de datos (tabla raw_data)", type="primary"):
            dbu.save_df(df, "raw_data")
            st.success("✅ Guardado en SQLite → tabla `raw_data`. Continua al Paso 2.")
    else:
        st.warning("Sube un archivo o activa el dataset de ejemplo para continuar.")

    if dbu.table_exists("raw_data"):
        st.info(f"ℹ️ Ya existe una tabla `raw_data` guardada con "
                f"{len(dbu.load_df('raw_data'))} filas. Puedes sobrescribirla arriba o avanzar al Paso 2.")

# ============================================================
# PASO 2: STAGING AREA
# ============================================================
elif paso == "2. Staging Area":
    st.subheader("Staging Area")
    st.caption("Validacion y control de calidad — lee `raw_data` de la BD y escribe `staging_validos` / `staging_errores`")

    df = dbu.load_df("raw_data")
    if df is None:
        st.warning("Primero guarda datos en el Paso 1 (Fuentes de Datos).")
    else:
        nulos = df.isna().sum()
        nulos = nulos[nulos > 0]
        n_dup = df.duplicated(subset=["id_encuesta"]).sum()
        edad_num = pd.to_numeric(df["edad"], errors="coerce")
        mask_edad_invalida = ~edad_num.between(15, 40)
        sueno_num = pd.to_numeric(df["horas_sueno_promedio"], errors="coerce")
        mask_sueno_invalido = ~sueno_num.between(0, 14) & sueno_num.notna()
        fecha_parsed = pd.to_datetime(df["fecha_encuesta"], errors="coerce", format="mixed")
        mask_fecha_invalida = fecha_parsed.isna()

        mask_error = mask_edad_invalida.fillna(False) | mask_sueno_invalido.fillna(False)
        df_errores = df[mask_error | df.duplicated(subset=["id_encuesta"], keep=False)]
        df_ok = df[~mask_error]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros totales", len(df))
        c2.metric("Duplicados detectados", int(n_dup))
        c3.metric("Edad fuera de rango", int(mask_edad_invalida.sum()))
        c4.metric("Horas sueno invalidas", int(mask_sueno_invalido.sum()))

        st.markdown("##### Valores nulos por columna")
        st.bar_chart(nulos) if len(nulos) > 0 else st.caption("No se detectaron valores nulos.")

        st.markdown(f"##### Fechas con formato inconsistente: **{mask_fecha_invalida.sum()}** filas (se corrigen en ETL)")

        st.markdown("##### Registros en cuarentena (no pasan al warehouse)")
        st.dataframe(df_errores.head(15), use_container_width=True)

        if st.button("💾 Guardar validados en BD (staging_validos / staging_errores)", type="primary"):
            dbu.save_df(df_ok, "staging_validos")
            dbu.save_df(df_errores, "staging_errores")
            st.success(f"✅ {len(df_ok)} registros guardados en `staging_validos`. Continua al Paso 3.")

# ============================================================
# PASO 3: PROCESO ETL
# ============================================================
elif paso == "3. Proceso ETL":
    st.subheader("Proceso ETL")
    st.caption("Lee `staging_validos` de la BD, transforma, y escribe `clean_data`")

    df = dbu.load_df("staging_validos")
    if df is None:
        st.warning("Primero ejecuta y guarda el Paso 2 (Staging Area).")
    else:
        with st.expander("Ver transformaciones aplicadas", expanded=True):
            st.markdown("""
            1. Estandarizacion de texto en `universidad`.
            2. Parseo de fechas mixtas a formato ISO.
            3. Eliminacion de duplicados por `id_encuesta`.
            4. Imputacion de nulos (mediana por facultad / moda).
            5. Calculo de `indice_riesgo_psicoemocional` (0-10).
            6. Categorizacion en Riesgo Bajo / Medio / Alto.
            """)

        df["universidad"] = df["universidad"].astype(str).str.strip().str.title()
        df["fecha_encuesta"] = pd.to_datetime(df["fecha_encuesta"], errors="coerce", format="mixed")
        antes = len(df)
        df = df.drop_duplicates(subset=["id_encuesta"])
        dedup = antes - len(df)

        num_cols = ["horas_sueno_promedio", "apoyo_social", "promedio_academico", "calidad_alimentacion"]
        for col in num_cols:
            df[col] = df.groupby("facultad")[col].transform(lambda s: s.fillna(s.median()))
            df[col] = df[col].fillna(df[col].median())
        df["sexo"] = df["sexo"].fillna(df["sexo"].mode()[0])

        df["indice_riesgo_psicoemocional"] = (
            df["nivel_estres"] * 0.35 + df["nivel_ansiedad"] * 0.35
            + (10 - df["nivel_animo"]) * 0.20 + (10 - df["apoyo_social"]) * 0.10
        ).round(2)

        def categorizar(v):
            if v < 4.5: return "Bajo"
            elif v < 7: return "Medio"
            else: return "Alto"
        df["nivel_riesgo"] = df["indice_riesgo_psicoemocional"].apply(categorizar)

        c1, c2, c3 = st.columns(3)
        c1.metric("Registros tras limpieza", len(df))
        c2.metric("Duplicados eliminados", dedup)
        c3.metric("Variables nuevas creadas", 2)

        st.dataframe(
            df[["id_encuesta", "universidad", "facultad", "fecha_encuesta",
                "indice_riesgo_psicoemocional", "nivel_riesgo"]].head(15),
            use_container_width=True,
        )

        fig = px.histogram(df, x="nivel_riesgo", color="nivel_riesgo",
                            category_orders={"nivel_riesgo": ["Bajo", "Medio", "Alto"]},
                            color_discrete_map={"Bajo": "#2ecc71", "Medio": "#f39c12", "Alto": "#e74c3c"})
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        if st.button("💾 Guardar datos limpios en BD (clean_data)", type="primary"):
            df["fecha_encuesta"] = df["fecha_encuesta"].astype(str)
            dbu.save_df(df, "clean_data")
            st.success(f"✅ Guardado en `clean_data`: {len(df)} filas. Continua al Paso 4.")

# ============================================================
# PASO 4: DATA WAREHOUSE
# ============================================================
elif paso == "4. Data Warehouse":
    st.subheader("Data Warehouse (SQLite — modelo estrella)")
    st.caption("Lee `clean_data` y construye dim_estudiante, dim_tiempo, fact_salud_mental")

    df = dbu.load_df("clean_data")
    if df is None:
        st.warning("Primero ejecuta y guarda el Paso 3 (Proceso ETL).")
    else:
        st.markdown("""
        <div class="kpi-card"><b>ESTADO:</b> Datos limpios listos para modelar como Esquema Estrella.</div>
        """, unsafe_allow_html=True)

        st.code("""
DIM_ESTUDIANTE            DIM_TIEMPO                    FACT_SALUD_MENTAL
----------------          ----------------              ----------------------------
PK id_estudiante          PK id_tiempo                  FK id_estudiante
   universidad               fecha_encuesta              FK id_tiempo
   facultad                  anio / mes / dia            nivel_estres, nivel_ansiedad
   ciclo, edad, sexo                                     indice_riesgo_psicoemocional
                                                          nivel_riesgo, promedio_academico
        """, language="text")

        if st.button("💾 Construir y guardar Data Warehouse", type="primary"):
            df["fecha_encuesta"] = pd.to_datetime(df["fecha_encuesta"], errors="coerce")

            dim_estudiante = df[["id_encuesta", "universidad", "facultad", "ciclo", "edad", "sexo"]].copy()
            dim_estudiante.rename(columns={"id_encuesta": "id_estudiante"}, inplace=True)

            dim_tiempo = df[["fecha_encuesta"]].drop_duplicates().reset_index(drop=True)
            dim_tiempo["id_tiempo"] = dim_tiempo.index + 1
            dim_tiempo["anio"] = dim_tiempo["fecha_encuesta"].dt.year
            dim_tiempo["mes"] = dim_tiempo["fecha_encuesta"].dt.month
            dim_tiempo["dia"] = dim_tiempo["fecha_encuesta"].dt.day

            fact = df.merge(dim_tiempo, on="fecha_encuesta", how="left")
            fact_cols = ["id_encuesta", "id_tiempo", "universidad", "facultad", "nivel_estres",
                         "nivel_ansiedad", "nivel_animo", "apoyo_social", "indice_riesgo_psicoemocional",
                         "nivel_riesgo", "promedio_academico", "horas_sueno_promedio",
                         "uso_redes_sociales_horas", "actividad_fisica_horas_sem"]
            fact_table = fact[fact_cols].rename(columns={"id_encuesta": "id_estudiante"})

            dim_estudiante["__k"] = 1  # evitar problemas de tipos mixtos al guardar
            dim_tiempo["fecha_encuesta"] = dim_tiempo["fecha_encuesta"].astype(str)

            dbu.save_df(dim_estudiante.drop(columns="__k"), "dim_estudiante")
            dbu.save_df(dim_tiempo, "dim_tiempo")
            dbu.save_df(fact_table, "fact_salud_mental")

            st.success(f"✅ Warehouse construido: {len(fact_table)} filas en `fact_salud_mental`.")

        if dbu.table_exists("fact_salud_mental"):
            ft = dbu.load_df("fact_salud_mental")
            st.markdown("##### Vista previa de `fact_salud_mental`")
            st.dataframe(ft.head(10), use_container_width=True)
            st.caption(f"Filas analiticas totales en el warehouse: **{len(ft)}**")

# ============================================================
# PASO 5: CAPA DE IA ETICA
# ============================================================
elif paso == "5. Capa de IA Etica":
    st.subheader("Capa de IA Etica — Prediccion de Riesgo Psicoemocional")

    df = dbu.load_df("clean_data")
    if df is None:
        st.warning("Primero ejecuta y guarda el Paso 3 (Proceso ETL).")
    else:
        st.info("""
        **Principios de IA etica aplicados:** datos anonimizados · sin decisiones automatizadas
        sobre personas (solo alertas agregadas para el area de Bienestar Universitario) ·
        auditoria de desempeño por subgrupos (sexo) para detectar sesgos ·
        transparencia mediante importancia de variables.
        """)

        features = ["horas_sueno_promedio", "horas_trabajo_semanal", "horas_estudio_diario",
                    "uso_redes_sociales_horas", "actividad_fisica_horas_sem", "tiempo_traslado_min",
                    "apoyo_social", "calidad_alimentacion", "procrastinacion", "satisfaccion_vida",
                    "ciclo", "edad"]
        X = df[features].fillna(df[features].median())
        y = df["nivel_riesgo"]

        if st.button("🧠 Entrenar y guardar modelo (Random Forest) en BD", type="primary"):
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X, y, df.index, test_size=0.25, random_state=42, stratify=y
            )
            modelo = RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42, class_weight="balanced")
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            metricas = {"acc": acc, "f1": f1, "features": features}
            dbu.save_model("riesgo_rf", modelo, json.dumps(metricas))

            df_test = df.loc[idx_test, ["id_encuesta", "sexo", "nivel_riesgo"]].copy()
            df_test["prediccion"] = y_pred
            dbu.save_df(df_test, "predicciones_test")

            st.success("✅ Modelo entrenado y guardado en BD (tabla `modelos`, blob pickled).")

        modelo, metricas_json = dbu.load_model("riesgo_rf")
        if modelo is not None:
            metricas = json.loads(metricas_json)
            c1, c2 = st.columns(2)
            c1.metric("Exactitud (accuracy)", f"{metricas['acc']*100:.1f}%")
            c2.metric("F1-score (macro)", f"{metricas['f1']*100:.1f}%")

            st.markdown("##### Importancia de variables")
            imp = pd.Series(modelo.feature_importances_, index=metricas["features"]).sort_values()
            fig_imp = px.bar(imp, orientation="h", labels={"value": "Importancia", "index": "Variable"})
            fig_imp.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig_imp, use_container_width=True)

            df_test = dbu.load_df("predicciones_test")
            if df_test is not None:
                labels = ["Bajo", "Medio", "Alto"]
                cm = confusion_matrix(df_test["nivel_riesgo"], df_test["prediccion"], labels=labels)
                st.markdown("##### Matriz de confusion")
                fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True,
                                    labels=dict(x="Prediccion", y="Real", color="Casos"),
                                    color_continuous_scale="Blues")
                st.plotly_chart(fig_cm, use_container_width=True)

                st.markdown("##### Auditoria de sesgo por subgrupo (sexo)")
                df_test["correcto"] = (df_test["prediccion"] == df_test["nivel_riesgo"]).astype(int)
                audit = df_test.groupby("sexo")["correcto"].mean().reset_index()
                audit.columns = ["sexo", "exactitud"]
                fig_audit = px.bar(audit, x="sexo", y="exactitud", color="sexo", range_y=[0, 1])
                fig_audit.update_layout(showlegend=False, height=320)
                st.plotly_chart(fig_audit, use_container_width=True)
                st.caption("Una brecha grande entre subgrupos indicaria sesgo del modelo.")

# ============================================================
# PASO 6: CAPA SEMANTICA & KPIs
# ============================================================
elif paso == "6. Capa Semantica & KPIs":
    st.subheader("Capa Semantica & KPIs")

    df = dbu.load_df("clean_data")
    if df is None:
        st.warning("Primero ejecuta y guarda el Paso 3 (Proceso ETL).")
    else:
        st.markdown("""
        | KPI | Definicion |
        |---|---|
        | **% Riesgo Alto** | Estudiantes con indice de riesgo ≥ 7 |
        | **Sueño saludable** | % con ≥ 7h de sueño promedio |
        | **Corr. Estres-Rendimiento** | Correlacion entre estres y promedio academico |
        | **Cobertura apoyo psicologico** | % que ha accedido a apoyo psicologico previo |
        """)

        pct_alto = (df["nivel_riesgo"] == "Alto").mean() * 100
        pct_sueno_ok = (df["horas_sueno_promedio"] >= 7).mean() * 100
        corr = df["nivel_estres"].corr(df["promedio_academico"])
        cobertura = (df["apoyo_psicologico_previo"] == "Si").mean() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("% Riesgo Alto", f"{pct_alto:.1f}%")
        c2.metric("Sueño saludable (≥7h)", f"{pct_sueno_ok:.1f}%")
        c3.metric("Corr. Estres-Rendimiento", f"{corr:.2f}")
        c4.metric("Cobertura apoyo psicologico", f"{cobertura:.1f}%")

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

    df = dbu.load_df("clean_data")
    if df is None:
        st.warning("Primero ejecuta y guarda el Paso 3 (Proceso ETL).")
    else:
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
                               title="Estres vs. Rendimiento academico",
                               color_discrete_map={"Bajo": "#2ecc71", "Medio": "#f39c12", "Alto": "#e74c3c"})
            st.plotly_chart(fig3, use_container_width=True)
        with colD:
            dff2 = dff.copy()
            dff2["mes"] = pd.to_datetime(dff2["fecha_encuesta"]).dt.to_period("M").astype(str)
            tend = dff2.groupby("mes")["indice_riesgo_psicoemocional"].mean().reset_index()
            fig4 = px.line(tend, x="mes", y="indice_riesgo_psicoemocional", markers=True,
                            title="Tendencia mensual del indice de riesgo")
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("##### Datos filtrados")
        st.dataframe(dff.head(50), use_container_width=True)
