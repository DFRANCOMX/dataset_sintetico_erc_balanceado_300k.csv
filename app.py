import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

model = xgb.Booster()
model.load_model("modelo_erc_xgb.json")

df_media = pd.read_csv("dataset_sintetico_erc_balanceado_300k.csv")
medias = df_media.mean()

st.title("Predicción de Enfermedad Renal Crónica (ERC)")
st.write("Ingrese los datos del paciente. Los campos vacíos se completarán automáticamente con valores promedio.")

variables = {
    "Sexo": (0, 1),  # 0 = Femenino, 1 = Masculino
    "Edad": (18, 100),
    "Glucosa": (50.0, 400.0),
    "Volumen_Orina_24h_ml": (300.0, 5000.0),
    "Creatinina_Orina_mg_dL": (20.0, 300.0),
    "Creatinina_Serica_mg_dL": (0.3, 15.0),
    "Urea": (5.0, 200.0),
    "BUN": (2.0, 100.0),
    "TFG": (5.0, 150.0),
    "HbA1c": (3.0, 15.0),
    "Proteinas_Orina_24h": (0.0, 10.0),
    "Sodio": (110.0, 160.0),
    "Potasio": (2.0, 8.0),
    "Calcio": (6.0, 15.0),
    "Microalbumina_24h": (0.0, 3000.0)
}

datos = {}

for var, (min_val, max_val) in variables.items():
    valor = st.number_input(
        f"{var} ({min_val} - {max_val})",
        min_value=min_val,
        max_value=max_val,
        value=None,
        step=0.1 if isinstance(min_val, float) else 1,
        format="%.2f" if isinstance(min_val, float) else "%d"
    )
    datos[var] = valor

if st.button("Predecir ERC"):
    for col in datos:
        if datos[col] is None:
            datos[col] = float(medias[col])

    input_df = pd.DataFrame([datos])
    dmat = xgb.DMatrix(input_df)

    probs = model.predict(dmat)
    pred = (probs >= 0.5).astype(int)[0]

    st.subheader("Resultados de la Predicción")
    st.write(f"**Probabilidad de ERC:** {probs[0]*100:.2f}%")
    st.write(f"**Clasificación:** {'Positivo' if pred == 1 else 'Negativo'}")
