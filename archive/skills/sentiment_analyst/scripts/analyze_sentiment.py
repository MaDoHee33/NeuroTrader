import json
import argparse
import requests
import random
from pathlib import Path
from datetime import datetime

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
NEWS_FILE = ASSETS_DIR / "news_headlines.json"
SCORE_FILE = ASSETS_DIR / "sentiment_score.json"

OLLAMA_URL = "http://localhost:11434/api/generate"

def get_mock_sentiment():
    """Fallback if Ollama is offline."""
    return {
        "sentiment_score": round(random.uniform(-0.5, 0.5), 2),
        "reasoning": "Mock analysis: Market appears neutral to slightly volatile based on random noise."
    }

def call_ollama(headlines, model="llama3"):
    # Construct Prompt
    combined_text = "\n".join([f"- {h['title']}" for h in headlines[:20]]) # Limit to 20 to fit context
    
    prompt = f"""
    Analyze the following financial news headlines and determine the overall market sentiment for Gold (XAUUSD) and Crypto market.
    
    Headlines:
    {combined_text}
    
    Instructions:
    - determining if the news is Bullish (Positive) or Bearish (Negative).
    - Provide a score between -1.0 (Extreme Fear/Sell) and +1.0 (Extreme Greed/Buy).
    - 0.0 is Neutral.
    
    Output Format:
    You must output ONLY a valid JSON object in this format:
    {{
        "sentiment_score": float,
        "reasoning": "short summary string"
    }}
    Do not output any markdown or explanation outside the JSON.
    """
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json" # Force JSON mode if model supports it
    }
    
    try:
        print(f"🧠 Sending {len(headlines)} headlines to Ollama ({model})...")
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', '')
        
        # Parse JSON from response
        try:
            analysis = json.loads(response_text)
            return analysis
        except json.JSONDecodeError:
            print("⚠️ Failed to parse LLM JSON response. Text was:", response_text)
            return get_mock_sentiment()
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama connection failed: {e}")
        print("👉 switching to MOCK mode.")
        return get_mock_sentiment()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model name")
    args = parser.parse_args()
    
    if not NEWS_FILE.exists():
        print(f"❌ No news file found at {NEWS_FILE}. Run fetch_news.py first.")
        return
        
    with open(NEWS_FILE, "r") as f:
        data = json.load(f)
        headlines = data.get("headlines", [])
        
    if not headlines:
        print("⚠️ No headlines to analyze.")
        return
        
    # Analyze
    result = call_ollama(headlines, model=args.model)
    
    # Save Result
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "data": result
    }
    
    with open(SCORE_FILE, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\n✅ Sentiment Analysis Complete:")
    print(f"   Score: {result.get('sentiment_score')} (-1.0 to 1.0)")
    print(f"   Reason: {result.get('reasoning')}")

if __name__ == "__main__":
    main()
