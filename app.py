import streamlit as st
from data_loader import load_data
from utils import detect_anomalies_zscore, detect_anomalies_isolation, detect_anomalies_lstm
from forecast_engine import run_prophet_forecast, run_arima_forecast
from plots import plot_anomalies, plot_forecast
from rag_assistant import ask_question
from portfolio_analyzer import analyze_portfolio_risk
from news_fetcher import get_latest_news
from sentiment_model import analyze_sentiment

query = "What happened to Google stock in 2023?"
file_path = "data/google_stock_news.txt"

answers = ask_question(file_path, query)

for i, answer in enumerate(answers, 1):
    print(f"Answer {i}:\n{answer}\n")

st.set_page_config(page_title="TradeWatch AI", layout="wide")
st.title("📈 TradeWatch AI – Stock Intelligence Agent")

# Tabs
tabs = st.tabs([
    "📊 Anomaly Detection",
    "🔮 Forecasting",
    "🧠 RAG Assistant",
    "📂 Portfolio Analyzer",
    "📰 News Feed",
    "📈 Sentiment Analyzer"
])

# === 1. ANOMALY DETECTION ===
with tabs[0]:
    st.subheader("📊 Anomaly Detection (Z-Score, Isolation Forest, LSTM Autoencoder)")
    upload = st.file_uploader("Upload your stock CSV (Date, Open, High, Low, Close, Volume)", type="csv")
    df = load_data(upload)

    model = st.selectbox("Choose Detection Method", ["Z-Score", "Isolation Forest", "LSTM Autoencoder"])
    if model == "Z-Score":
        df = detect_anomalies_zscore(df)
        plot_anomalies(df, "anomaly_zscore")
    elif model == "Isolation Forest":
        df = detect_anomalies_isolation(df)
        plot_anomalies(df, "anomaly_iso")
    elif model == "LSTM Autoencoder":
        df = detect_anomalies_lstm(df)
        plot_anomalies(df, "anomaly_lstm")

    st.download_button("💾 Download Results", df.to_csv(index=True), "Anomaly_Results.csv")

# === 2. FORECASTING ===
with tabs[1]:
    st.subheader("🔮 Forecasting (Prophet / ARIMA)")
    df = load_data(upload)
    method = st.radio("Choose Forecast Model", ["Prophet", "ARIMA"])

    if method == "Prophet":
        forecast, changepoints = run_prophet_forecast(df)
        plot_forecast(forecast, changepoints)
    else:
        forecast = run_arima_forecast(df)
        st.line_chart(forecast.set_index("ds")["yhat"])

# === 3. RAG FINANCIAL ASSISTANT ===
with tabs[2]:
    st.subheader("🧠 Ask Financial Questions (RAG+LLM)")
    question = st.text_input("Ask a question (e.g. 'What caused the dip in Jan 2021?')")
    if question:
        answer = ask_question(question)
        st.success(answer)

# === 4. PORTFOLIO RISK ANALYZER ===
with tabs[3]:
    st.subheader("📂 Portfolio Risk Analyzer (CSV)")
    portfolio_file = st.file_uploader("Upload Portfolio CSV (Asset, Weight, Return, Volatility)", type="csv", key="portfolio")
    if portfolio_file:
        risk_report = analyze_portfolio_risk(portfolio_file)
        st.write(risk_report)

# === 5. FINANCIAL NEWS ===
with tabs[4]:
    st.subheader("📰 Real-Time Financial News")
    news = get_latest_news()
    for article in news:
        st.markdown(f"**{article['title']}**")
        st.caption(article["source"])
        st.write(article["summary"])
        st.markdown(f"[Read more]({article['url']})")

# === 6. SENTIMENT ANALYSIS ===
with tabs[5]:
    st.subheader("📈 Sentiment Analysis")
    news_input = st.text_area("Enter financial news or text:")
    model_choice = st.radio("Choose Sentiment Model", ["VADER", "FinBERT"])
    if st.button("Analyze Sentiment"):
        sentiment = analyze_sentiment(news_input, model_choice)
        st.metric("Sentiment Score", sentiment["score"])
        st.write("Interpretation:", sentiment["label"])
