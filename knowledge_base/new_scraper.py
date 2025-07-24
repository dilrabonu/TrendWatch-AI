import requests
from bs4 import BeautifulSoup
import os

def scrape_yahoo_news(ticker="GOOG", max_articles=20):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    articles = soup.find_all("li", class_="js-stream-content")
    news_list = []

    for i, article in enumerate(articles):
        if i >= max_articles:
            break
        title = article.find("h3")
        summary = article.find("p")
        if title and summary:
            news_list.append({
                "title": title.text.strip(),
                "summary": summary.text.strip()
            })

    return news_list

def save_to_txt(news, filename="goog_news.txt"):
    os.makedirs("knowledge_base", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for i, item in enumerate(news):
            f.write(f"📰 News {i+1}\n")
            f.write(f"Title: {item['title']}\n")
            f.write(f"Summary: {item['summary']}\n\n")


if __name__ == "__main__":
    print("📡 Scraping Yahoo Finance news for GOOG...")
    news = scrape_yahoo_news("GOOG", max_articles=20)
    save_to_txt(news)
    print("✅ News saved to knowledge_base/goog_news.txt")
