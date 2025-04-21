# Inside streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Load model
model = joblib.load('models/final_water_quality_model.pkl')

# Load X_train for SHAP and LIME (must match training features)
X_train = pd.read_csv('data/X_train_for_explainers.csv')  # Ensure this file exists

st.title("💧 Water Potability Predictor with Explanations")

# User inputs
ph = st.slider('pH', 0.0, 14.0, 7.0)
hardness = st.number_input('Hardness', value=200)
solids = st.number_input('Solids (ppm)', value=10000)
chloramines = st.number_input('Chloramines (ppm)', value=7.0)
sulfate = st.number_input('Sulfate (ppm)', value=300)
conductivity = st.number_input('Conductivity (µS/cm)', value=500)
organic_carbon = st.number_input('Organic Carbon (ppm)', value=10.0)
trihalomethanes = st.number_input('Trihalomethanes (ppm)', value=50.0)
turbidity = st.number_input('Turbidity (NTU)', value=4.0)

# DataFrame for prediction
input_data = pd.DataFrame([{
    'ph': ph,
    'Hardness': hardness,
    'Solids': solids,
    'Chloramines': chloramines,
    'Sulfate': sulfate,
    'Conductivity': conductivity,
    'Organic_carbon': organic_carbon,
    'Trihalomethanes': trihalomethanes,
    'Turbidity': turbidity
}])

# Prediction button logic
if st.button("Predict Potability"):
    prediction = model.predict(input_data)[0]
    result = "Potable" if prediction == 1 else "Non-Potable"
    st.success(f"The water is predicted to be: **{result}**")

    # -------------------------------
    # ✅ SHAP EXPLANATION BLOCK HERE
    # -------------------------------
    st.subheader("🔍 SHAP Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(X_train)

    # Get values for the predicted class
    predicted_class = prediction
    values = shap_values[predicted_class]

    # Plot SHAP waterfall
    shap.initjs()
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots._waterfall.waterfall_legacy(
        explainer_shap.expected_value[predicted_class],
        values[0],
        input_data.iloc[0],
        feature_names=input_data.columns.tolist(),
        show=False
    )
    st.pyplot(fig)

    # -------------------------------
    # ✅ LIME EXPLANATION BLOCK
    # -------------------------------
    st.subheader("📊 LIME Explanation")
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=["Non-Potable", "Potable"],
        mode='classification'
    )

    exp = explainer_lime.explain_instance(
        data_row=input_data.values[0],
        predict_fn=model.predict_proba,
        num_features=5
    )
    fig_lime = exp.as_pyplot_figure()
    st.pyplot(fig_lime)
