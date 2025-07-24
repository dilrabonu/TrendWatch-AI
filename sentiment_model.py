"""
Module to run sentiment analysis on financial headlines or user queries.
Supports VADER (lightweight) and FinBERT (contextual transformer).
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Load VADER analyzer (faster, less context-aware)
vader_analyzer = SentimentIntensityAnalyzer()

# Load FinBERT model (slower but better for financial text)
finbert_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

def analyze_with_vader(text):
    scores = vader_analyzer.polarity_scores(text)
    label = (
        "positive" if scores['compound'] > 0.05 else
        "negative" if scores['compound'] < -0.05 else
        "neutral"
    )
    return {
        "label": label,
        "score": scores["compound"]
    }

def analyze_with_finbert(text):
    results = finbert_pipeline(text)
    label = results[0]['label'].lower()
    return {
        "label": label,
        "score": results[0]['score']  # FinBERT score (confidence)
    }

def analyze_sentiment(text, model_choice="vader"):
    """
    Unified function used by Streamlit UI.
    Parameters:
        text (str): Input financial text
        model_choice (str): 'vader' or 'finbert'
    Returns:
        dict: { "label": "positive/neutral/negative", "score": float }
    """
    if model_choice.lower() == "finbert":
        return analyze_with_finbert(text)
    else:
        return analyze_with_vader(text)

def batch_sentiment(texts, method="vader"):
    """
    Analyze a list of texts using the selected method.
    """
    analyzer = analyze_with_vader if method == "vader" else analyze_with_finbert
    return [analyzer(text) for text in texts]


# === Test mode ===
if __name__ == '__main__':
    sample_headlines = [
        "Google shares fall after weaker-than-expected earnings",
        "Tesla delivers record vehicles in Q2",
        "AI optimism drives tech stock rally"
    ]

    print("VADER Sentiments:")
    for result in batch_sentiment(sample_headlines, method="vader"):
        print(result)

    print("\nFinBERT Sentiments:")
    for result in batch_sentiment(sample_headlines, method="finbert"):
        print(result)
