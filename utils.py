import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

def detect_anomalies_zscore(df):
    rolling_mean = df['Close'].rolling(window=30).mean()
    rolling_std = df['Close'].rolling(window=30).std()
    df['z_score'] = (df['Close'] - rolling_mean) / rolling_std
    df['anomaly_zscore'] = df['z_score'].apply(lambda x: 1 if abs(x) > 3 else 0)
    return df

def detect_anomalies_isolation(df):
    features = df[['Close', 'Volume']].copy().fillna(method='bfill')
    iso = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly_iso'] = iso.fit_predict(features)
    df['anomaly_iso'] = df['anomaly_iso'].map({1: 0, -1: 1})
    return df

def detect_anomalies_lstm(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])
    
    def create_sequences(data, window=30):
        return np.array([data[i:i+window] for i in range(len(data) - window)])
    
    X = create_sequences(scaled)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X.shape[1], 1), return_sequences=False),
        RepeatVector(X.shape[1]),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(1))
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=10, batch_size=32, verbose=0)
    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=(1, 2))
    threshold = np.percentile(mse, 95)

    df['anomaly_lstm'] = 0
    df.iloc[30:, df.columns.get_loc('anomaly_lstm')] = (mse > threshold).astype(int)
    return df

def forecast_prophet(df):
    prophet_df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast, model.changepoints
