import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import psutil
import torch
import os
import time
import sys

# Try to import streamlit_autorefresh for automatic telemetry updates
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# Try to import pynvml for real GPU metrics
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except:
    HAS_PYNVML = False

# Ensure project root is in path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.gatekeeper import gatekeeper
from src.evaluate import evaluate as run_evaluation

st.set_page_config(page_title="FinTech RL Trading Dashboard", layout="wide")

# Enable auto-refresh for telemetry updates (every 2 seconds)
if HAS_AUTOREFRESH:
    st_autorefresh(interval=2000, key="telemetry_refresh")

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
cpu_usage_placeholder = st.sidebar.empty()
mem_usage_placeholder = st.sidebar.empty()
gpu_name_placeholder = st.sidebar.empty()
gpu_util_placeholder = st.sidebar.empty()
gpu_mem_placeholder = st.sidebar.empty()
gpu_temp_placeholder = st.sidebar.empty()

def update_telemetry():
    # CPU and Memory metrics
    cpu_usage_placeholder.metric("CPU Usage", f"{psutil.cpu_percent():.1f}%")
    mem = psutil.virtual_memory()
    mem_usage_placeholder.metric("Memory Usage", f"{mem.percent:.1f}%", f"{mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB")

    # GPU metrics
    if torch.cuda.is_available():
        gpu_name_placeholder.text(f"GPU: {torch.cuda.get_device_name(0)}")

        if HAS_PYNVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                # GPU Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util_placeholder.metric("GPU Utilization", f"{util.gpu}%")

                # GPU Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used_mb = mem_info.used / (1024**2)
                mem_total_mb = mem_info.total / (1024**2)
                gpu_mem_placeholder.metric("GPU Memory", f"{mem_used_mb:.0f} MB / {mem_total_mb:.0f} MB", f"{(mem_info.used / mem_info.total * 100):.1f}%")

                # GPU Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_temp_placeholder.metric("GPU Temperature", f"{temp}°C")

            except Exception as e:
                gpu_util_placeholder.text(f"GPU Metrics: Error ({str(e)[:30]})")
        else:
            gpu_util_placeholder.text("GPU Metrics: pynvml not available")
            gpu_mem_placeholder.text("Install: pip install nvidia-ml-py3")
    else:
        gpu_name_placeholder.text("GPU: N/A (CPU mode)")
        gpu_util_placeholder.text("")
        gpu_mem_placeholder.text("")
        gpu_temp_placeholder.text("")

# Load evaluation results
RESULTS_CSV = os.path.join(LOGS_DIR, "eval_results.csv")

if os.path.exists(RESULTS_CSV):
    df = pd.read_csv(RESULTS_CSV)
    df['Date'] = pd.to_datetime(df['Date'])

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Historical Price & Actions")

        # Fetch OHLC data for candlestick chart
        try:
            # Get date range from evaluation results
            start_date = df['Date'].min().strftime('%Y-%m-%d')
            end_date = df['Date'].max().strftime('%Y-%m-%d')

            # Fetch full OHLC data via Gatekeeper
            ohlc_data = gatekeeper.fetch_stock_data(user_ticker, start_date, end_date)

            if ohlc_data is not None and not ohlc_data.empty:
                # Create candlestick chart
                fig = go.Figure()

                fig.add_trace(go.Candlestick(
                    x=ohlc_data.index,
                    open=ohlc_data['Open'],
                    high=ohlc_data['High'],
                    low=ohlc_data['Low'],
                    close=ohlc_data['Close'],
                    name='OHLC',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ))

                # Overlay Buy/Sell signals
                buy_signals = df[df['Action'] == 1]
                sell_signals = df[df['Action'] == 2]

                fig.add_trace(go.Scatter(
                    x=buy_signals['Date'],
                    y=buy_signals['Price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='lime', line=dict(color='darkgreen', width=1)),
                    name='Buy Signal'
                ))

                fig.add_trace(go.Scatter(
                    x=sell_signals['Date'],
                    y=sell_signals['Price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red', line=dict(color='darkred', width=1)),
                    name='Sell Signal'
                ))

                fig.update_layout(
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    xaxis_rangeslider_visible=False,  # Disable rangeslider for cleaner view
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                # Fallback to line chart if OHLC data unavailable
                st.warning("OHLC data unavailable, showing close price only")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], name='Close Price', line=dict(color='blue')))

                buy_signals = df[df['Action'] == 1]
                sell_signals = df[df['Action'] == 2]

                fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Price'], mode='markers',
                                         marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy'))
                fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Price'], mode='markers',
                                         marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell'))

                fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading OHLC data: {e}")
            # Ultra-fallback: show eval results only
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], name='Price', line=dict(color='blue')))
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

# Update telemetry once per render (autorefresh triggers re-render every 2s if available)
update_telemetry()

# If autorefresh is not available, provide manual refresh option
if not HAS_AUTOREFRESH:
    if st.sidebar.button("Refresh Telemetry"):
        st.rerun()
