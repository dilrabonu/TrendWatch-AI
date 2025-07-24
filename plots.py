import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_anomalies(df, col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', mode='lines'))
    fig.add_trace(go.Scatter(x=df[df[col] == 1].index, y=df[df[col] == 1]['Close'],
                             mode='markers', name='Anomaly', marker=dict(color='red', size=8)))
    fig.update_layout(title=f"Anomaly Detection – {col}", template="plotly_white")
    return fig.show()

def plot_forecast(forecast, changepoints):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower', fill='tonexty'))
    fig.update_layout(title='Prophet Forecast with Confidence Intervals', template='plotly_white')
    fig.show()
