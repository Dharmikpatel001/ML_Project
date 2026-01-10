import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AQI Prediction App", layout="wide")

st.title("AQI Prediction Using Machine Learning")

# -------------------- DATA UPLOAD --------------------
uploaded_file = st.file_uploader("Upload AQI CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------- TARGET SELECTION --------------------
    target_column = st.selectbox("Select Target Column (AQI)", df.columns)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # -------------------- TRAIN TEST SPLIT -------------------- 
    test_size = st.slider("Test Size (%)", 10, 40, 20)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42
    )

    # -------------------- MODEL SELECTION --------------------
    model_name = st.selectbox(
        "Select Model",
        ["Linear Regression", "Random Forest Regressor"]
    )

    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        n_estimators = st.slider("Number of Trees", 50, 300, 100)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

    # -------------------- TRAIN MODEL --------------------
    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # -------------------- METRICS --------------------
        st.subheader("Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", round(mean_absolute_error(y_test, y_pred), 2))
        col2.metric("MSE", round(mean_squared_error(y_test, y_pred), 2))
        col3.metric("R² Score", round(r2_score(y_test, y_pred), 2))

        # -------------------- PLOT --------------------
        st.subheader("Actual vs Predicted AQI")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test.values, label="Actual AQI")
        ax.plot(y_pred, label="Predicted AQI", alpha=0.7)
        ax.legend()
        ax.set_xlabel("Samples")
        ax.set_ylabel("AQI")

        st.pyplot(fig)

else:
    st.info("Upload a CSV file to get started")

