"""
Module to analyze uploaded stock portfolios:
- Calculates portfolio value, volatility
- Flags stocks with recent anomalies
- Useful for personalized financial risk detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_portfolio(file):
    df = pd.read_csv(file)
    assert {'Ticker', 'Qty', 'Entry_Price'}.issubset(df.columns), "CSV must contain Ticker, Qty, Entry_Price"
    return df


def merge_with_anomalies(portfolio_df, anomaly_df):
    latest_anomalies = anomaly_df.groupby('Ticker')['anomaly_zscore'].last()
    merged = portfolio_df.set_index('Ticker').join(latest_anomalies)
    merged['anomaly_flag'] = merged['anomaly_zscore'].fillna(0).astype(int)
    return merged.reset_index()


def compute_risk_metrics(prices_df):
    returns = prices_df.pivot(columns='Ticker', values='Close').pct_change().dropna()
    volatility = returns.std()
    correlation = returns.corr()
    return volatility, correlation


def plot_risk_heatmap(correlation_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Portfolio Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def calculate_portfolio_value(portfolio_df, current_prices):
    current_prices = current_prices.set_index('Ticker')['Close']
    portfolio_df['Current_Price'] = portfolio_df['Ticker'].map(current_prices)
    portfolio_df['Value'] = portfolio_df['Qty'] * portfolio_df['Current_Price']
    return portfolio_df, portfolio_df['Value'].sum()


# === ✅ New function: For Streamlit use ===
def analyze_portfolio_risk(file):
    try:
        df = pd.read_csv(file)
        if not {'Ticker', 'Qty', 'Entry_Price'}.issubset(df.columns):
            return "❌ CSV must include: Ticker, Qty, Entry_Price"

        avg_price = df['Entry_Price'].mean()
        total_qty = df['Qty'].sum()
        total_value = (df['Qty'] * df['Entry_Price']).sum()

        return {
            "📊 Avg Entry Price": round(avg_price, 2),
            "📦 Total Shares Held": int(total_qty),
            "💰 Total Entry Value": round(total_value, 2)
        }

    except Exception as e:
        return f"❌ Error analyzing portfolio: {e}"


# === CLI Test (Optional) ===
if __name__ == '__main__':
    portfolio = pd.DataFrame({
        'Ticker': ['GOOG', 'MSFT', 'TSLA'],
        'Qty': [10, 20, 5],
        'Entry_Price': [1200, 230, 700]
    })
    prices = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=3),
        'Ticker': ['GOOG', 'MSFT', 'TSLA'],
        'Close': [1340, 240, 690]
    })

    updated, total = calculate_portfolio_value(portfolio, prices)
    print(updated)
    print("Total Portfolio Value:", total)
