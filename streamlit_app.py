import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model from the specified path
model = joblib.load('models/final_water_quality_model.pkl')

# Set the title of the dashboard
st.title('Water Potability Predictor')
st.write("Enter the water quality measurements below:")

# Create widgets for user input. Modify the default values as per your dataset's scale.
pH = st.slider('pH', min_value=0.0, max_value=14.0, value=7.0, step=0.1)
hardness = st.number_input('Hardness', value=200, step=1)
solids = st.number_input('Solids (ppm)', value=10000, step=100)
chloramines = st.number_input('Chloramines (ppm)', value=7.0, step=0.1)
sulfate = st.number_input('Sulfate (ppm)', value=300, step=1)
conductivity = st.number_input('Conductivity (µS/cm)', value=500, step=1)
organic_carbon = st.number_input('Organic Carbon (ppm)', value=10.0, step=0.1)
trihalomethanes = st.number_input('Trihalomethanes (ppm)', value=50.0, step=0.1)
turbidity = st.number_input('Turbidity (NTU)', value=4.0, step=0.1)

# Create a DataFrame from the user inputs that matches the model's expected input
input_data = pd.DataFrame({
    'ph': [pH],
    'Hardness': [hardness],
    'Solids': [solids],
    'Chloramines': [chloramines],
    'Sulfate': [sulfate],
    'Conductivity': [conductivity],
    'Organic_carbon': [organic_carbon],
    'Trihalomethanes': [trihalomethanes],
    'Turbidity': [turbidity]
})

# When the user clicks the Predict button, generate predictions using the model.
if st.button('Predict Potability'):
    prediction = model.predict(input_data)
    result = 'Potable' if prediction[0] == 1 else 'Non-Potable'
    st.success(f'The water is predicted to be: {result}')
