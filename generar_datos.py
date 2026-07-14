"""
Generador de datos sinteticos: Encuestas de Salud Mental en Universitarios de Lima Norte
Curso: Big Data / BI
Autor: Plataforma Integral de Analitica Universitaria

Genera un CSV crudo (simulando exportacion de un formulario tipo Google Forms / Microsoft Forms)
con inconsistencias reales (nulos, duplicados, formatos mixtos) para que el pipeline ETL
tenga trabajo real que hacer en las etapas de Staging y ETL.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 3200  # numero de encuestas crudas (antes de limpieza)

UNIVERSIDADES = [
    "Universidad Cesar Vallejo - Lima Norte",
    "Universidad Privada del Norte",
    "Universidad de Ciencias y Humanidades",
    "Universidad Autonoma del Peru",
    "Universidad Nacional Federico Villarreal",
]
# pesos de matricula distintos por universidad (mas realista)
PESOS_UNIV = [0.32, 0.24, 0.16, 0.18, 0.10]

FACULTADES = [
    "Ingenieria", "Psicologia", "Ciencias de la Salud",
    "Administracion y Negocios", "Derecho", "Comunicaciones", "Educacion",
]

SEXO = ["Femenino", "Masculino", "Prefiero no decir"]

def fecha_aleatoria():
    inicio = datetime(2026, 1, 12)
    fin = datetime(2026, 6, 30)
    delta = fin - inicio
    dias = random.randint(0, delta.days)
    dt = inicio + timedelta(days=dias, hours=random.randint(7, 22), minutes=random.randint(0, 59))
    return dt

filas = []
for i in range(N):
    univ = np.random.choice(UNIVERSIDADES, p=PESOS_UNIV)
    fac = np.random.choice(FACULTADES)
    ciclo = np.random.randint(1, 11)
    edad = int(np.clip(np.random.normal(21, 2.3), 17, 32))
    sexo = np.random.choice(SEXO, p=[0.54, 0.44, 0.02])

    # Variables base correlacionadas para dar sentido estadistico real
    horas_sueno = np.clip(np.random.normal(6.1, 1.4), 2, 10)
    trabaja = np.random.choice(["Si", "No"], p=[0.34, 0.66])
    horas_trabajo = np.clip(np.random.normal(28, 8), 4, 48) if trabaja == "Si" else 0
    horas_estudio = np.clip(np.random.normal(3.2, 1.5), 0, 10)
    uso_redes = np.clip(np.random.normal(4.5, 2.0), 0, 12)
    act_fisica = np.clip(np.random.exponential(1.8), 0, 10)
    beca = np.random.choice(["Si", "No"], p=[0.22, 0.78])
    vive_familia = np.random.choice(["Si", "No"], p=[0.78, 0.22])
    traslado_min = int(np.clip(np.random.normal(55, 25), 5, 180))
    apoyo_psicologico_previo = np.random.choice(["Si", "No"], p=[0.14, 0.86])

    # Estres/ansiedad influenciados por sueno, trabajo, traslado, redes sociales
    base_estres = (
        4.5
        + (7 - horas_sueno) * 0.55
        + (horas_trabajo / 40) * 1.8
        + (traslado_min / 60) * 0.5
        + (uso_redes / 12) * 1.0
        - act_fisica * 0.25
        + np.random.normal(0, 1.3)
    )
    nivel_estres = int(np.clip(round(base_estres), 1, 10))

    base_ansiedad = base_estres * 0.85 + np.random.normal(0, 1.2)
    nivel_ansiedad = int(np.clip(round(base_ansiedad), 1, 10))

    base_animo = (
        6.0
        - (nivel_estres - 5) * 0.4
        + act_fisica * 0.3
        + (1 if vive_familia == "Si" else -0.6)
        + np.random.normal(0, 1.5)
    )
    nivel_animo = int(np.clip(round(base_animo), 1, 10))  # mas alto = mejor animo

    apoyo_social = int(np.clip(round(np.random.normal(6.2 if vive_familia == "Si" else 4.8, 1.8)), 1, 10))
    calidad_alimentacion = int(np.clip(round(np.random.normal(3.0, 1.0)), 1, 5))
    consumo_alcohol = np.random.choice(["Nunca", "Ocasional", "Frecuente"], p=[0.45, 0.44, 0.11])
    procrastinacion = int(np.clip(round(np.random.normal(5.5 + (nivel_estres - 5) * 0.3, 1.8)), 1, 10))
    satisfaccion_vida = int(np.clip(round(11 - nivel_estres * 0.6 + apoyo_social * 0.3 + np.random.normal(0, 1.2)), 1, 10))

    promedio_academico = np.clip(
        14.5 - (nivel_estres - 5) * 0.25 - (procrastinacion - 5) * 0.15 + horas_estudio * 0.2 + np.random.normal(0, 1.3),
        7, 20,
    )

    fila = {
        "id_encuesta": f"ENC-{10000+i}",
        "fecha_encuesta": fecha_aleatoria(),
        "universidad": univ,
        "facultad": fac,
        "ciclo": ciclo,
        "edad": edad,
        "sexo": sexo,
        "horas_sueno_promedio": round(horas_sueno, 1),
        "trabaja": trabaja,
        "horas_trabajo_semanal": round(horas_trabajo, 1),
        "horas_estudio_diario": round(horas_estudio, 1),
        "uso_redes_sociales_horas": round(uso_redes, 1),
        "actividad_fisica_horas_sem": round(act_fisica, 1),
        "beca": beca,
        "vive_con_familia": vive_familia,
        "tiempo_traslado_min": traslado_min,
        "apoyo_psicologico_previo": apoyo_psicologico_previo,
        "nivel_estres": nivel_estres,
        "nivel_ansiedad": nivel_ansiedad,
        "nivel_animo": nivel_animo,
        "apoyo_social": apoyo_social,
        "calidad_alimentacion": calidad_alimentacion,
        "consumo_alcohol": consumo_alcohol,
        "procrastinacion": procrastinacion,
        "satisfaccion_vida": satisfaccion_vida,
        "promedio_academico": round(promedio_academico, 2),
    }
    filas.append(fila)

df = pd.DataFrame(filas)

# ---- Inyectar "suciedad" real para que el pipeline ETL tenga sentido ----

# 1) Duplicados (~2%)
dup_idx = np.random.choice(df.index, size=int(N * 0.02), replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# 2) Nulos aleatorios en varias columnas (~3-6%)
for col in ["horas_sueno_promedio", "apoyo_social", "promedio_academico", "sexo", "calidad_alimentacion"]:
    idx = np.random.choice(df.index, size=int(len(df) * np.random.uniform(0.03, 0.06)), replace=False)
    df.loc[idx, col] = np.nan

# 3) Inconsistencias de formato de fecha (texto en vez de datetime) en algunas filas
idx_fecha_txt = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
df["fecha_encuesta"] = df["fecha_encuesta"].astype(object)
for i in idx_fecha_txt:
    d = df.loc[i, "fecha_encuesta"]
    df.at[i, "fecha_encuesta"] = d.strftime("%d/%m/%Y") if hasattr(d, "strftime") else d

# 4) Outliers imposibles (errores de digitacion)
idx_out = np.random.choice(df.index, size=15, replace=False)
df.loc[idx_out, "edad"] = np.random.choice([2, 5, 99, 150], size=15)

idx_out2 = np.random.choice(df.index, size=10, replace=False)
df.loc[idx_out2, "horas_sueno_promedio"] = np.random.choice([-1, 30, 45], size=10)

# 5) Inconsistencia de mayusculas/espacios en universidad (texto sucio)
idx_txt = np.random.choice(df.index, size=20, replace=False)
df.loc[idx_txt, "universidad"] = df.loc[idx_txt, "universidad"].str.upper() + "  "

df = df.sample(frac=1, random_state=7).reset_index(drop=True)

out_path = "/home/claude/proyecto_bi_salud_mental/data/encuestas_salud_mental_lima_norte.csv"
df.to_csv(out_path, index=False)
print(f"Generado: {out_path}")
print(f"Filas totales (con duplicados/nulos intencionales): {len(df)}")
print(df.head(3).to_string())
