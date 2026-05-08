import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import psutil
import torch
import os
import time
import sys

# Ensure project root is in path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.gatekeeper import gatekeeper
from src.evaluate import evaluate as run_evaluation

st.set_page_config(page_title="FinTech RL Trading Dashboard", layout="wide")

st.title("FinTech RL Trading Dashboard")

# Sidebar for Ticker Selection
st.sidebar.header("Simulation Settings")
user_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., TSLA, NVDA, AAPL)", value=TICKER).upper().strip()

if st.sidebar.button("Run Simulation"):
    with st.spinner(f"Validating and running simulation for {user_ticker}..."):
        if gatekeeper.validate_ticker_exists(user_ticker):
            try:
                run_evaluation(ticker=user_ticker)
                st.sidebar.success(f"Simulation for {user_ticker} completed!")
                # Force refresh of results by continuing to render
            except Exception as e:
                st.sidebar.error(f"Error during simulation: {e}")
        else:
            st.sidebar.error(f"Ticker '{user_ticker}' is invalid or not found.")

st.subheader(f"Active Ticker: {user_ticker}")

# Sidebar for Telemetry
st.sidebar.markdown("---")
st.sidebar.header("System Telemetry")
cpu_usage = st.sidebar.empty()
mem_usage = st.sidebar.empty()
gpu_info = st.sidebar.empty()

def update_telemetry():
    cpu_usage.text(f"CPU Usage: {psutil.cpu_percent()}%")
    mem_usage.text(f"Memory Usage: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        gpu_info.text(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        gpu_info.text("GPU: N/A")

# Load evaluation results
RESULTS_CSV = os.path.join(LOGS_DIR, "eval_results.csv")

if os.path.exists(RESULTS_CSV):
    df = pd.read_csv(RESULTS_CSV)
    df['Date'] = pd.to_datetime(df['Date'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Historical Price & Actions")
        # Plot Candlesticks (Simplified as line chart if OHLC not fully available in results, 
        # but let's try to fetch full data for candlesticks)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], name='Price', line=dict(color='blue')))
        
        # Add Buy/Sell Markers
        buy_signals = df[df['Action'] == 1]
        sell_signals = df[df['Action'] == 2]
        
        fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Price'], mode='markers',
                                 marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy'))
        fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Price'], mode='markers',
                                 marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell'))
        
        fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
        
        st.header("Portfolio Value")
        fig_portfolio = go.Figure()
        fig_portfolio.add_trace(go.Scatter(x=df['Date'], y=df['PortfolioValue'], name='Portfolio', fill='tozeroy'))
        fig_portfolio.update_layout(height=400, xaxis_title="Date", yaxis_title="Balance ($)")
        st.plotly_chart(fig_portfolio, use_container_width=True)

    with col2:
        st.header("Performance Metrics")
        final_portfolio = df['PortfolioValue'].iloc[-1]
        initial_balance = 10000.0
        total_return = ((final_portfolio - initial_balance) / initial_balance) * 100
        
        st.metric("Final Portfolio", f"${final_portfolio:,.2f}", f"{total_return:.2f}%")
        
        avg_confidence = df['Confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.4f}")
        
        # Reward Curve from logs
        rewards_path = os.path.join(LOGS_DIR, "rewards.npy")
        if os.path.exists(rewards_path):
            rewards = np.load(rewards_path)
            st.header("Learning Curve")
            st.line_chart(rewards)

else:
    st.warning("Evaluation results not found. Please run the training and evaluation first.")

# Continuous telemetry update
while True:
    update_telemetry()
    time.sleep(1)
