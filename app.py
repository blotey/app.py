# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Stock Forecast App", layout="wide")
st.title("📈 Stock Price Forecasting Dashboard")

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --- Sidebar Inputs ---
st.sidebar.header("📁 Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=90, value=30)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    st.write("### 📄 Uploaded Data")
    st.line_chart(df['Close'])

    tab1, tab2, tab3 = st.tabs(["📉 SARIMA", "🔮 Prophet", "🧠 LSTM"])

    # --- SARIMA ---
    with tab1:
        st.subheader("SARIMA Forecast")
        train = df['Close'][:-forecast_days]
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
        results = model.fit(disp=False)
        pred = results.get_forecast(steps=forecast_days)
        sarima_mean = pred.predicted_mean
        sarima_conf = pred.conf_int()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Close'], label="Historical")
        ax.plot(sarima_mean.index, sarima_mean, label="Forecast", color='red')
        ax.fill_between(sarima_conf.index, sarima_conf.iloc[:, 0], sarima_conf.iloc[:, 1], color='pink', alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        test = df['Close'][-forecast_days:]
        sarima_rmse = np.sqrt(mean_squared_error(test, sarima_mean))
        sarima_mae = mean_absolute_error(test, sarima_mean)
        sarima_mape = calculate_mape(test, sarima_mean)
        st.write(f"**RMSE:** {sarima_rmse:.2f}")
        st.write(f"**MAE:** {sarima_mae:.2f}")
        st.write(f"**MAPE:** {sarima_mape:.2f}%")

    # --- Prophet ---
    with tab2:
        st.subheader("Prophet Forecast")
        prophet_df = df[['Close']].reset_index().rename(columns={"Date": "ds", "Close": "y"})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        fig2 = model.plot(forecast)
        st.pyplot(fig2)

        y_true = prophet_df['y'][-forecast_days:].values
        y_pred = forecast['yhat'][-forecast_days:].values[:len(y_true)]
        prophet_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        prophet_mae = mean_absolute_error(y_true, y_pred)
        prophet_mape = calculate_mape(y_true, y_pred)
        st.write(f"**RMSE:** {prophet_rmse:.2f}")
        st.write(f"**MAE:** {prophet_mae:.2f}")
        st.write(f"**MAPE:** {prophet_mape:.2f}%")

    # --- LSTM ---
    with tab3:
        st.subheader("LSTM Forecast")
        data = df[['Close']].values
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        look_back = forecast_days
        X, y = [], []
        for i in range(look_back, len(data_scaled)):
            X.append(data_scaled[i-look_back:i])
            y.append(data_scaled[i])
        X, y = np.array(X), np.array(y)

        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = Sequential([
            LSTM(50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        lstm_dates = df.index[-len(y_test_actual):]

        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(lstm_dates, y_test_actual, label="Actual")
        ax3.plot(lstm_dates, y_pred, label="Predicted", color='red')
        ax3.set_title("LSTM Prediction")
        ax3.legend()
        st.pyplot(fig3)

        lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        lstm_mae = mean_absolute_error(y_test_actual, y_pred)
        lstm_mape = calculate_mape(y_test_actual, y_pred)
        st.write(f"**RMSE:** {lstm_rmse:.2f}")
        st.write(f"**MAE:** {lstm_mae:.2f}")
        st.write(f"**MAPE:** {lstm_mape:.2f}%")

    # --- Comparison Table ---
    comparison_data = {
        "Model": ["SARIMA", "Prophet", "LSTM"],
        "RMSE": [sarima_rmse, prophet_rmse, lstm_rmse],
        "MAE": [sarima_mae, prophet_mae, lstm_mae],
        "MAPE (%)": [sarima_mape, prophet_mape, lstm_mape]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.write("### 📊 Model Comparison")
    st.dataframe(comparison_df)

else:
    st.info("📤 Please upload a CSV file with 'Date' and 'Close' columns from the sidebar.")
