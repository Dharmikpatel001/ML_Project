import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

import pickle

import requests

API_KEY = "H5xk0RZo4oX3kf7f5t8aJA8kVK6GV77bmxUSz561"

# Load trained model
model = pickle.load(open("aqi_model.pkl", "rb"))

st.set_page_config(page_title="AQI Predictor", layout="centered")

st.title("🌫 Air Quality Index (AQI) Predictor")
st.write("Enter air pollutant values to predict AQI")

# User Inputs
co_aqi = st.number_input("CO AQI (0–500)", min_value=0, max_value=500)
ozone_aqi = st.number_input("Ozone AQI (0–500)", min_value=0, max_value=500)
no2_aqi = st.number_input("NO2 AQI (0–500)", min_value=0, max_value=500)
pm25_aqi = st.number_input("PM2.5 AQI (0–500)", min_value=0, max_value=500)



if st.button("Predict AQI"):
    input_data = np.array([[co_aqi,ozone_aqi,no2_aqi, pm25_aqi]])
    prediction = model.predict(input_data)

    st.success(f"Predicted AQI: {int(prediction[0])}")

    # AQI Category
    if prediction <= 50:
        st.info("Good 🟢")
    elif prediction >=51 and prediction <=100:
        st.info("Moderate 🟡")
    elif prediction >= 101 and prediction <= 150:
        st.warning("Poor🟠")
    elif prediction >= 151 and prediction <= 200:
        st.warning("Unhealthy  🟤")
    elif prediction >= 201 and prediction <= 300:
        st.warning("Severe 🟣")
    else:
        st.error("Hazardous 🔴")


st.set_page_config(page_title="Live AQI Predictor", layout="centered")
st.title("🌫 Live AQI Predictor (API-Ninjas)")
st.write("Enter a city name to fetch live air quality data and predict AQI.")


# -------- City Input --------
city = st.text_input("City Name", "Delhi")

# -------- Helper Functions --------
def city_to_latlon(city):
    url = f"https://api.api-ninjas.com/v1/geocoding?city={city}"
    headers = {"X-Api-Key": API_KEY}
    res = requests.get(url, headers=headers).json()
    return res[0]["latitude"], res[0]["longitude"]

def fetch_aqi(lat, lon):
    url = f"https://api.api-ninjas.com/v1/airquality?lat={lat}&lon={lon}"
    headers = {"X-Api-Key": API_KEY}
    return requests.get(url, headers=headers).json()

# -------- Button Action --------
if st.button("Fetch & Predict AQI"):
    try:
        # Convert city → lat/lon
        lat, lon = city_to_latlon(city)

        # Fetch AQI data
        data = fetch_aqi(lat, lon)

        # Extract AQI sub-indices (MODEL INPUTS)
        co_aqi   = data["CO"]["aqi"]
        o3_aqi   = data["O3"]["aqi"]
        no2_aqi  = data["NO2"]["aqi"]
        pm25_aqi = data["PM2.5"]["aqi"]

        # Prepare input (ORDER MUST MATCH TRAINING)
        input_data = np.array([[co_aqi, o3_aqi, no2_aqi, pm25_aqi]])

        # Predict AQI
        predicted_aqi = model.predict(input_data)[0]
        real_aqi = data["overall_aqi"]

        # -------- Display --------
        st.success(f"🔮 Predicted AQI: {int(predicted_aqi)}")
        st.info(f"📡 Actual AQI (API): {real_aqi}")

        st.subheader("Pollutant AQI Values")
        st.write({
            "CO AQI": co_aqi,
            "Ozone AQI": o3_aqi,
            "NO₂ AQI": no2_aqi,
            "PM2.5 AQI": pm25_aqi
        })

    except Exception as e:
        st.error("Unable to fetch data. Please check city name or API key.")