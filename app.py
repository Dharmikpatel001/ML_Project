import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("aqi_model.joblib")

st.set_page_config(page_title="AQI Predictor", layout="centered")

st.title("🌫 Air Quality Index (AQI) Predictor")
st.write("Enter air pollutant values to predict AQI")

# User Inputs
pm25 = st.number_input("PM2.5", min_value=0.0)
pm10 = st.number_input("PM10", min_value=0.0)
no2 = st.number_input("NO2", min_value=0.0)
co = st.number_input("CO", min_value=0.0)
if st.button("Predict AQI"):
    input_data = np.array([[pm25, pm10, no2, co]])
    prediction = model.predict(input_data)

    st.success(f"Predicted AQI: {int(prediction[0])}")

    # AQI Category
    if prediction <= 50:
        st.info("Good 🟢")
    elif prediction <= 100:
        st.warning("Moderate 🟡")
    elif prediction <= 200:
        st.warning("Poor 🟠")
    else:
        st.error("Severe 🔴")
