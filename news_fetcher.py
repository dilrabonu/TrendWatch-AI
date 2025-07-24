"""
Module to fetch real-time financial news headlines.
Includes:
- Yahoo Finance
- MarketWatch
- Keyword filtering
Returns:
    List of dictionaries with title, summary, source, and URL for use in Streamlit
"""

import requests
from bs4 import BeautifulSoup


def fetch_yahoo_finance_news(ticker="AAPL"):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        articles = soup.select("li.js-stream-content")

        news = []
        for a in articles[:10]:
            title_tag = a.find("h3")
            summary_tag = a.find("p")
            link_tag = a.find("a", href=True)

            if title_tag and link_tag:
                news.append({
                    "title": title_tag.get_text(strip=True),
                    "summary": summary_tag.get_text(strip=True) if summary_tag else "",
                    "source": "Yahoo Finance",
                    "url": "https://finance.yahoo.com" + link_tag["href"]
                })
        return news
    except Exception as e:
        print(f"[Yahoo Error] {e}")
        return []


def fetch_marketwatch_news():
    url = "https://www.marketwatch.com/latest-news"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        articles = soup.select("div.article__content")

        news = []
        for a in articles[:10]:
            title_tag = a.find("h3", class_="article__headline")
            summary_tag = a.find("p")
            link_tag = a.find("a", href=True)

            if title_tag and link_tag:
                news.append({
                    "title": title_tag.get_text(strip=True),
                    "summary": summary_tag.get_text(strip=True) if summary_tag else "",
                    "source": "MarketWatch",
                    "url": link_tag["href"]
                })
        return news
    except Exception as e:
        print(f"[MarketWatch Error] {e}")
        return []


def get_latest_news(ticker="GOOG"):
    yahoo = fetch_yahoo_finance_news(ticker)
    mw = fetch_marketwatch_news()
    return yahoo + mw


def filter_by_keywords(news, keywords):
    return [n for n in news if any(k.lower() in n['title'].lower() for k in keywords)]


if __name__ == '__main__':
    headlines = get_latest_news("GOOG")
    for h in headlines:
        print(f"\n📰 {h['title']}\n📌 {h['summary']}\n🔗 {h['url']}")
