import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import shap

@st.cache_resource
def cargar_modelo():
    model = xgb.Booster()
    model.load_model('modelo_erc_xgb.json')
    return model

model = cargar_modelo()

variables = {
    'Sexo': (0, 1),
    'Edad': (20, 100),
    'Glucosa': (50, 400),
    'Volumen_Orina_24h_ml': (300, 5000),
    'Creatinina_Orina_mg_dL': (10, 1000),
    'Creatinina_Serica_mg_dL': (0.3, 15.0),
    'Urea': (5, 150),
    'BUN': (3, 70),
    'TFG': (5, 130),
    'HbA1c': (4.0, 15.0),
    'Proteinas_Orina_24h': (0, 10),
    'Sodio': (120, 160),
    'Potasio': (2.5, 7.0),
    'Calcio': (7.0, 12.0),
    'Microalbumina_24h': (0, 500)
}

st.title("Predicción de Riesgo de Enfermedad Renal Crónica")
st.markdown("Ingresa los datos del paciente para predecir la probabilidad de ERC")

datos = {}
for var, (min_val, max_val) in variables.items():
    if isinstance(min_val, float) or isinstance(max_val, float):
        step = 0.1
    else:
        step = 1
    if var == 'Sexo':
        datos[var] = st.selectbox(f"{var} (0= Mujer, 1= Hombre)", options=[0, 1])
    else:
        datos[var] = st.slider(f"{var} ({min_val} - {max_val})", 
                               min_value=type(min_val)(min_val), 
                               max_value=type(max_val)(max_val), 
                               value=type(min_val)(min_val),
                               step=step)

df_paciente = pd.DataFrame([datos])

df_paciente = df_paciente[model.feature_names]

if st.button("Predecir"):
    dmat = xgb.DMatrix(df_paciente, feature_names=model.feature_names)
    probs = model.predict(dmat)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_paciente)

    clase_pred = np.argmax(probs[0])
    st.write(f"Clase predicha: {clase_pred} (0=nula, 1=baja, 2=alta, 3=falla renal)")
    st.write(f"Probabilidades: {probs[0]}")

    shap_vals_clase = shap_values[0, :, clase_pred]
    imp_df = pd.DataFrame({
        'Feature': df_paciente.columns,
        'SHAP value': shap_vals_clase
    }).sort_values(by='SHAP value', key=abs, ascending=False)

    st.write("Importancia de variables:")
    st.table(imp_df)
