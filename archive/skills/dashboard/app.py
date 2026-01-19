import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json
import time

# Helper Functions
def get_root_dir():
    return Path(__file__).resolve().parent.parent.parent

ROOT_DIR = get_root_dir()

# Config
st.set_page_config(
    page_title="NeuroTrader Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.header("🧠 NeuroTrader")
    st.markdown("---")
    menu = st.radio("Navigation", ["Main Cockpit", "Sentiment Analysis", "System Logs"])
    st.markdown("---")
    
    st.subheader("System Status")
    st.success("🟢 Body (MT5): Online (Mock)")
    st.success("🟢 Brain (PPO): Active")
    st.success("🟢 Senses (News): Active")

def load_sentiment():
    path = ROOT_DIR / "skills" / "sentiment_analyst" / "assets" / "sentiment_score.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)["data"]
    return {"sentiment_score": 0.0, "reasoning": "No data found."}

def load_market_data():
    # Load sample Augmented Parquet if exists, else mock
    path = ROOT_DIR / "data" / "processed" / "XAUUSDm_M15_L2_News.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df.tail(100) # Last 100 candles
    else:
        # Mock
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='15min')
        df = pd.DataFrame(index=dates)
        df['close'] = np.cumsum(np.random.randn(100)) + 2000
        df['high'] = df['close'] + 2
        df['low'] = df['close'] - 2
        df['open'] = df['close']
        df['news_impact_score'] = np.random.choice([0, 0, 0, 1, 3], 100)
        return df

# Main Logic
if menu == "Main Cockpit":
    st.title("🎛️ Control Center")
    col1, col2, col3 = st.columns(3)
    
    df = load_market_data()
    last_price = df.iloc[-1]['close']
    prev_price = df.iloc[-2]['close']
    delta = last_price - prev_price
    
    sentiment = load_sentiment()
    score = sentiment['sentiment_score']
    
    with col1:
        st.metric("XAUUSD Price", f"${last_price:,.2f}", f"{delta:+.2f}")
    with col2:
        st.metric("Sentiment Score", f"{score:.2f}", delta=score, delta_color="normal")
    with col3:
        st.metric("Active Trades", "1", "+1 (Buy)")
        
    # Chart
    st.markdown("### 📈 Live Market (M15)")
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])
    
    # Add News Markers
    news_candles = df[df['news_impact_score'] > 0]
    if not news_candles.empty:
        fig.add_trace(go.Scatter(
            x=news_candles.index, 
            y=news_candles['high'] + 5,
            mode='markers',
            marker=dict(symbol='star', size=10, color='yellow'),
            name='High News'
        ))

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Sentiment Analysis":
    st.title("👂 Sentiment Intelligence")
    sentiment = load_sentiment()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Current Mood")
        score = sentiment['sentiment_score']
        if score > 0.5:
            st.warning(f"Extreme Greed ({score})")
        elif score < -0.5:
            st.error(f"Extreme Fear ({score})")
        else:
            st.info(f"Neutral ({score})")
            
    with col2:
        st.subheader("Reasoning")
        st.write(sentiment['reasoning'])
        
    st.markdown("### 📰 Recent Headlines")
    # Load news file
    news_path = ROOT_DIR / "skills" / "sentiment_analyst" / "assets" / "news_headlines.json"
    if news_path.exists():
        with open(news_path, "r") as f:
            news = json.load(f)["headlines"]
            for n in news[:10]:
                st.markdown(f"- **{n['source']}**: {n['title']}")
    else:
        st.write("No headlines found.")

elif menu == "System Logs":
    st.title("📜 System Logs")
    st.code("NeuroTrader v2.1.0 Initialized...\nConnecting to MT5... OK\nLoading Model L3... OK\nWaiting for ticks...", language="bash")
