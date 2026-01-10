import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

import pickle


# Load trained model
model = pickle.load(open("aqi_model.pkl", "rb"))

st.set_page_config(page_title="AQI Predictor", layout="centered")

st.title("🌫 Air Quality Index (AQI) Predictor")
st.write("Enter air pollutant values to predict AQI")

# User Inputs
pm25_aqi = st.number_input("PM2.5 AQI (0–500)", min_value=0, max_value=500)
no2_aqi = st.number_input("NO2 AQI (0–500)", min_value=0, max_value=500)
co_aqi = st.number_input("CO AQI (0–500)", min_value=0, max_value=500)
ozone_aqi = st.number_input("Ozone AQI (0–500)", min_value=0, max_value=500)

if st.button("Predict AQI"):
    input_data = np.array([[pm25, pm10, no2, co]])
    prediction = model.predict(input_data)

    st.success(f"Predicted AQI: {int(prediction[0])}")

    # AQI Category
    if prediction <= 50:
        st.info("Good 🟢")
    elif prediction >=51 and prediction <=100:
        st.info("Satisfactory 🟡")
    elif prediction >= 101 and prediction <= 200:
        st.warning("Moderate 🟠")
    elif prediction >= 201 and prediction <= 300:
        st.warning("Poor 🔴")
    elif prediction >= 301 and prediction <= 400:
        st.warning("Very Poor 🟣")
    else:
        st.error("Severe ⚫")
