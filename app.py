import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("aqi_model.pkl", "rb"))

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
