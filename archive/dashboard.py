import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime

# Add project root needed for imports if running from tools/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="NeuroTrader Dashboard", layout="wide")

st.title("🧠 NeuroTrader Live Dashboard")

# --- Database Connection ---
DB_PATH = "data/memory/neurotrader.db"
if not os.path.exists(DB_PATH):
    st.error(f"Database not found at {DB_PATH}. Run the agent first.")
    st.stop()

engine = create_engine(f'sqlite:///{DB_PATH}')

# --- Sidebar ---
st.sidebar.header("Status")
auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=True)
if auto_refresh:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5000, key="datarefresh")

# --- System Health ---
import json
STATUS_PATH = "data/status.json"
if os.path.exists(STATUS_PATH):
    try:
        with open(STATUS_PATH, 'r') as f:
            status = json.load(f)
            
        con_type = status.get('connection', 'UNKNOWN')
        is_shadow = status.get('shadow_mode', False)
        
        # Color coding
        color = "green" if con_type == "REAL" else "orange"
        mode_text = "👻 Shadow Mode" if is_shadow else "⚡ Live Execution"
        
        st.sidebar.markdown(f"### Connection: :{color}[{con_type}]")
        st.sidebar.markdown(f"**Mode**: {mode_text}")
        st.sidebar.caption(f"PID: {status.get('pid')} | Started: {status.get('start_time')}")
    except:
        st.sidebar.error("Error reading status file")
else:
    st.sidebar.warning("System Offline (No status file)")

# --- Load Data ---
try:
    trades_df = pd.read_sql("SELECT * FROM trade_history ORDER BY timestamp DESC LIMIT 200", engine)
except Exception as e:
    st.error(f"Error reading DB: {e}")
    st.stop()

# --- KPI Row ---
col1, col2, col3, col4 = st.columns(4)

total_trades = len(trades_df)
shadow_trades = len(trades_df[trades_df['is_shadow'] == 1])
real_trades = total_trades - shadow_trades

with col1:
    st.metric("Total Trades", total_trades)
with col2:
    st.metric("Shadow Trades", shadow_trades)
with col3:
    st.metric("Real Trades", real_trades)
with col4:
    last_active = trades_df['timestamp'].iloc[0] if not trades_df.empty else "N/A"
    st.metric("Last Activity", str(last_active))

# --- Charts ---
st.subheader("Recent Activity")
if not trades_df.empty:
    fig = px.scatter(trades_df, x="timestamp", y="price", color="action", 
                     symbol="is_shadow", hover_data=["symbol", "volume", "comment"],
                     title="Trade Executions")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(trades_df)
else:
    st.info("No trades recorded yet.")
