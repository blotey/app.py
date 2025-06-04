import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Stock Forecast App", layout="wide")
st.title("📈 Stock Price Forecasting Dashboard")

uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    st.write("### 📄 Uploaded Data", df.tail())

    forecast_days = st.slider("Select Forecast Horizon (days)", 7, 60, 30)

    # --- SARIMA Forecast ---
    sarima_model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_result = sarima_model.fit(disp=False)
    sarima_forecast = sarima_result.get_forecast(steps=forecast_days)
    sarima_mean = sarima_forecast.predicted_mean
    sarima_conf = sarima_forecast.conf_int()
    forecast_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=forecast_days, freq='D')
    sarima_mean.index = forecast_dates
    sarima_conf.index = forecast_dates

    # --- Prophet Forecast ---
    prophet_df = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    prophet = Prophet()
    prophet.fit(prophet_df)
    future = prophet.make_future_dataframe(periods=forecast_days)
    forecast = prophet.predict(future)

    # --- LSTM Forecast ---
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    seq_len = 60
    X, y = create_sequences(scaled_data, seq_len)
    X_train, X_test = X[:-forecast_days], X[-forecast_days:]
    y_train, y_test = y[:-forecast_days], y[-forecast_days:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    lstm_dates = df.index[-forecast_days:]

    # --- Visualization Tabs ---
    tab1, tab2, tab3 = st.tabs(["📉 SARIMA", "📆 Prophet", "🧠 LSTM"])

    with tab1:
        st.subheader("SARIMA Forecast")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Close'], label="Historical")
        ax.plot(sarima_mean, label="Forecast", color='red')
        ax.fill_between(sarima_conf.index, sarima_conf.iloc[:, 0], sarima_conf.iloc[:, 1], color='pink', alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with tab2:
        st.subheader("Prophet Forecast")
        fig2 = prophet.plot(forecast)
        st.pyplot(fig2)

    with tab3:
        st.subheader("LSTM Forecast vs Actual")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(lstm_dates, y_test_actual, label="Actual")
        ax3.plot(lstm_dates, y_pred, label="Predicted", color='red')
        ax3.set_title("LSTM Prediction")
        ax3.legend()
        st.pyplot(fig3)
