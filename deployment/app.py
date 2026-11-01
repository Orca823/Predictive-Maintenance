# Write the Streamlit interface that collects sensor values and displays a model prediction.
import os
import streamlit as st
import pandas as pd
import joblib

model_path = os.path.join(os.path.dirname(__file__), "engine_condition_model.joblib")
model = joblib.load(model_path)

st.title("Engine Predictive Maintenance App")
st.write("""
This application predicts the likelihood of engine issues based on its real-time sensor parameters.
""")

engine_rpm       = st.number_input("Engine RPM", 0, 3000, 700)
lub_oil_pressure = st.number_input("Lub Oil Pressure", 0.0, 10.0, 2.5, 0.1)
fuel_pressure    = st.number_input("Fuel Pressure", 0.0, 30.0, 11.8, 0.1)
coolant_pressure = st.number_input("Coolant Pressure", 0.0, 10.0, 3.2, 0.1)
lub_oil_temp     = st.number_input("Lub Oil Temperature", 50.0, 150.0, 84.1, 0.1)
coolant_temp     = st.number_input("Coolant Temperature", 50.0, 200.0, 81.6, 0.1)

input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp,
}])

if st.button("Predict Engine Condition"):
    estimator = model["model"] if isinstance(model, dict) else model
    prediction = estimator.predict(input_data)[0]
    result = "Engine Condition: Abnormal (1)" if prediction == 1 else "Engine Condition: Normal (0)"
    if prediction == 1:
        st.error(f"The model predicts: **{result}**")
    else:
        st.success(f"The model predicts: **{result}**")
