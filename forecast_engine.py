# === ✅ FIXED forecast_engine.py ===
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# --- Prophet forecast ---
def run_prophet_forecast(df, days=30):
    df_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    changepoints = model.changepoints
    return forecast, model.changepoints

# --- ARIMA forecast ---
def run_arima_forecast(df, days=30):
    series = df['Close']
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    future_dates = pd.date_range(start=series.index[-1], periods=days+1, freq='B')[1:]
    return pd.DataFrame({'ds': future_dates, 'yhat': forecast})

# --- Evaluation (optional) ---
def evaluate_forecast(true, predicted):
    return mean_absolute_error(true, predicted)

# --- Manual test run ---
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    df = pd.read_csv('data/cleaned_google_stock.csv', parse_dates=['Date'], index_col='Date')

    forecast, model = run_prophet_forecast(df)
    model.plot(forecast)
    plt.title('Prophet Forecast')
    plt.show()

    forecast_arima = run_arima_forecast(df)
    forecast_arima.set_index('ds').plot(title='ARIMA Forecast')
    plt.show()
