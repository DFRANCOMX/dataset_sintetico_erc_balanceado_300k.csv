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
    'Glucosa': (50.0, 400.0),
    'Volumen_Orina_24h_ml': (300.0, 5000.0),
    'Creatinina_Orina_mg_dL': (10.0, 1000.0),
    'Creatinina_Serica_mg_dL': (0.3, 15.0),
    'Urea': (5.0, 150.0),
    'BUN': (3.0, 70.0),
    'TFG': (5.0, 130.0),
    'HbA1c': (4.0, 15.0),
    'Proteinas_Orina_24h': (0.0, 10.0),
    'Sodio': (120.0, 160.0),
    'Potasio': (2.5, 7.0),
    'Calcio': (7.0, 12.0),
    'Microalbumina_24h': (0.0, 500.0)
}

st.title("Predicción de Riesgo de Enfermedad Renal Crónica")
st.markdown("Ingresa los datos del paciente para predecir la probabilidad de ERC")

datos = {}
for var, (min_val, max_val) in variables.items():
    if var == 'Sexo':
        datos[var] = st.selectbox(f"{var} (0= Mujer, 1= Hombre)", options=[0,1])
    else:
        valor = st.text_input(f"{var} (rango {min_val} - {max_val})", "")
        if valor.strip() == "":
            datos[var] = np.nan
        else:
            try:
                val_float = float(valor)
                if val_float < min_val or val_float > max_val:
                    st.warning(f"⚠️ El valor de {var} está fuera del rango usual.")
                datos[var] = val_float
            except:
                st.error(f"⚠️ Entrada inválida para {var}, debe ser un número o vacío.")
                datos[var] = np.nan

df_paciente = pd.DataFrame([datos])

if st.button("Predecir"):
    df_paciente.fillna(df_paciente.mean(), inplace=True)
    
    dmat = xgb.DMatrix(df_paciente)
    probs = model.predict(dmat)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_paciente)

    probs_paciente = probs[0]
    clase_pred = np.argmax(probs_paciente)

    st.subheader("Resultados de la Predicción")
    st.write("Probabilidades por clase:")
    clases = ["nula", "baja", "alta", "falla renal"]
    for i, p in enumerate(probs_paciente):
        st.write(f"Clase {i} ({clases[i]}): {p:.4f} ({p*100:.2f}%)")

    st.write(f"**Clase predicha:** {clase_pred} ({clases[clase_pred]})")

    shap_vals_clase = shap_values[0, :, clase_pred]
    imp_df = pd.DataFrame({
        'Feature': df_paciente.columns,
        'SHAP value': shap_vals_clase
    }).sort_values(by='SHAP value', key=abs, ascending=False)

    st.write("Importancia de variables:")
    st.table(imp_df)
