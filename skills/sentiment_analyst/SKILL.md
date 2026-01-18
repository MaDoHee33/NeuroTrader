---
name: Sentiment Analyst Skill
description: Fetches financial news and performs sentiment analysis using Local LLM (Ollama) to generate Greed/Fear signals.
---

# Sentiment Analyst Skill

This skill provides "alternative data" to the trading system by analyzing unstructured text (news, tweets) to gauge market sentiment.

## Capabilities

### 1. Fetch News
Aggregates news headlines from RSS feeds (ForexFactory, CoinDesk, Yahoo Finance).
- **Script**: `scripts/fetch_news.py`
- **Usage**: `python skills/sentiment_analyst/scripts/fetch_news.py --target "gold, crypto"`
- **Output**: Saves `news_headlines.json` in `assets/`.

### 2. Analyze Sentiment (Ollama)
Sends recent headlines to a local LLM (Ollama) to score sentiment from -1 (Extreme Fear) to +1 (Extreme Greed).
- **Script**: `scripts/analyze_sentiment.py`
- **Usage**: `python skills/sentiment_analyst/scripts/analyze_sentiment.py --model llama3`
- **Output**: `sentiment_score.json` containing the latest scalar signal.

## Integration
- The `NeuroTrader` skill can read `sentiment_score.json` as an additional feature input (e.g., modifying position sizing or acting as a filter).

## Dependencies
- `feedparser`
- `requests`
- `ollama` (optional, for local inference)
