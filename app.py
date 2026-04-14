import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================
# TITLE
# ======================
st.title("📊 Gold Price Prediction App")

# ======================
# UPLOAD FILE
# ======================
file = st.file_uploader("Upload Gold Price CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # ======================
    # DATA CLEANING
    # ======================
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = df[col].astype(float)

    st.subheader("📌 Dataset Preview")
    st.write(df.head())

    # ======================
    # PLOT PRICE
    # ======================
    st.subheader("📈 Gold Price Trend")

    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Price'], label="Price", color="gold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # ======================
    # FEATURES & TARGET
    # ======================
    X = df[['Open', 'High', 'Low']]
    y = df['Price']

    # ======================
    # TRAIN TEST SPLIT
    # ======================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ======================
    # MODEL TRAINING
    # ======================
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ======================
    # PREDICTION
    # ======================
    y_pred = model.predict(X_test)

    # ======================
    # EVALUATION
    # ======================
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.write("MAE:", round(mae, 2))
    st.write("RMSE:", round(rmse, 2))
    st.write("R² Score:", round(r2, 4))

    # ======================
    # ACTUAL VS PREDICTED
    # ======================
    st.subheader("📉 Actual vs Predicted")

    fig2, ax2 = plt.subplots()
    ax2.plot(y_test.values[:50], label="Actual")
    ax2.plot(y_pred[:50], label="Predicted")
    ax2.legend()

    st.pyplot(fig2)

else:
    st.info("👆 Upload your Gold Price CSV file to start")
