from newspaper import Article
from newspaper import build
import os

def scrape_google_news_articles(max_articles=10):
    print("🔍 Scraping Google-related news using newspaper3k...")
    paper = build("https://finance.yahoo.com/quote/GOOG/?p=GOOG&.tsrc=fin-srch", memoize_articles=False)
    articles = paper.articles[:max_articles]

    results = []

    for i, article in enumerate(articles):
        try:
            article.download()
            article.parse()
            if "Google" in article.text or "Alphabet" in article.text:
                results.append({
                    "title": article.title,
                    "summary": article.text[:400].replace("\n", " ") + "..."
                })
        except:
            continue

    return results

def save_to_txt(news, filename="knowledge_base/goog_news.txt"):
    os.makedirs("knowledge_base", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for i, item in enumerate(news):
            f.write(f"📰 News {i+1}\n")
            f.write(f"Title: {item['title']}\n")
            f.write(f"Summary: {item['summary']}\n\n")

if __name__ == "__main__":
    news = scrape_google_news_articles(max_articles=15)
    save_to_txt(news)
    print("✅ News successfully saved to knowledge_base/goog_news.txt")
