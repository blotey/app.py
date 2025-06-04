import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

st.title("📈 Stock Price Forecasting with SARIMA")

uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    st.write("### Preview of Uploaded Data", df.tail())

    # Model fitting
    model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit(disp=False)

    # Forecasting
    forecast_steps = st.slider("Forecast Days", min_value=7, max_value=60, value=30)
    forecast = result.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Date index for forecast
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_mean.index = forecast_dates
    conf_int.index = forecast_dates

    # Plot
    st.write("### Forecast Plot")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'], label='Historical')
    ax.plot(forecast_mean, label='Forecast', color='red')
    ax.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    ax.set_title('SARIMA Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)
