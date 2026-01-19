import feedparser
import json
import argparse
from pathlib import Path
from datetime import datetime
import ssl

# Fix for some SSL certificate errors
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

RSS_FEEDS = {
    "crypto": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss"
    ],
    "forex": [
        "https://www.investing.com/rss/news_1.rss", # General Forex
        "https://www.investing.com/rss/news_285.rss" # Commodities (Gold)
    ]
}

def fetch_news(category="all"):
    headlines = []
    
    target_feeds = []
    if category == "all":
        for cat in RSS_FEEDS:
            target_feeds.extend(RSS_FEEDS[cat])
    elif category in RSS_FEEDS:
        target_feeds = RSS_FEEDS[category]
        
    print(f"📡 Fetching news from {len(target_feeds)} feeds...")
    
    for url in target_feeds:
        try:
            feed = feedparser.parse(url)
            print(f"   - {feed.feed.get('title', url)}: Found {len(feed.entries)} items")
            
            for entry in feed.entries[:5]: # Top 5 per feed
                headlines.append({
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get('published', str(datetime.now())),
                    "source": feed.feed.get('title', 'Unknown')
                })
        except Exception as e:
            print(f"   ❌ Error fetching {url}: {e}")
            
    return headlines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="all", choices=["crypto", "forex", "all"])
    args = parser.parse_args()
    
    news_items = fetch_news(args.category)
    
    # Save to JSON
    output_file = ASSETS_DIR / "news_headlines.json"
    data = {
        "timestamp": datetime.now().isoformat(),
        "count": len(news_items),
        "headlines": news_items
    }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"\n✅ Saved {len(news_items)} headlines to {output_file}")
    
    # Preview
    print("\n📰 Latest Headlines:")
    for item in news_items[:5]:
        print(f" • {item['title']} ({item['source']})")

if __name__ == "__main__":
    main()
