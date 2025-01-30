import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimization", layout="wide")
st.title("📊 Portfolio Optimization (SLSQP, Monte Carlo, Consensus)")

# ✅ User Input for Stock Tickers
tickers = st.text_input("Enter stock tickers (comma separated):", "AAPL, MSFT, TSLA, GOOGL")
tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

# ✅ Risk-Free Rate Input
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (e.g., 0.05 for 5%)", value=0.05, step=0.01)

# ✅ Fetch Stock Data
def fetch_stock_data(tickers, start_date, end_date, retries=3):
    adj_close_df = pd.DataFrame()
    for ticker in tickers:
        for attempt in range(retries):
            try:
                stock = yf.Ticker(ticker)
                stock_data = stock.history(start=start_date, end=end_date)
                if "Adj Close" in stock_data.columns:
                    adj_close_df[ticker] = stock_data["Adj Close"]
                elif "Close" in stock_data.columns:
                    adj_close_df[ticker] = stock_data["Close"]
                else:
                    st.warning(f"⚠ No 'Adj Close' or 'Close' data for {ticker}. Skipping.")
                st.success(f"✅ Data downloaded for {ticker}")
                time.sleep(2)
                break
            except Exception as e:
                st.warning(f"Attempt {attempt+1}: Failed to fetch {ticker} - {e}")
                time.sleep(5)
    adj_close_df.dropna(axis=1, inplace=True)
    return adj_close_df

# ✅ Fetch Data When Button is Pressed
if st.sidebar.button("Optimize Portfolio"):
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    adj_close_df = fetch_stock_data(tickers, start_date, end_date)

    if adj_close_df.empty:
        st.error("⚠ No valid stock data available! Check tickers.")
    else:
        st.subheader("Stock Data (Adj Close / Close)")
        st.write(adj_close_df.head())

        # ✅ Compute Returns & Covariance Matrix
        log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
        mean_returns = log_returns.mean() * 252
        cov_matrix = log_returns.cov() * 252

        # ✅ Monte Carlo Simulation
        num_portfolios = 100000
        all_weights = np.zeros((num_portfolios, len(tickers)))
        results = np.zeros((3, num_portfolios))

        for i in range(num_portfolios):
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)  # Normalize
            all_weights[i, :] = weights
            port_return = np.sum(mean_returns * weights)
            port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe_ratio = (port_return - risk_free_rate) / port_volatility
            results[0, i] = port_return
            results[1, i] = port_volatility
            results[2, i] = sharpe_ratio

        max_sharpe_idx = np.argmax(results[2])
        mc_optimal_weights = all_weights[max_sharpe_idx, :]

        # ✅ Scipy SLSQP Optimization
        def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
            return -(np.sum(log_returns.mean() * weights) * 252 - risk_free_rate) / np.sqrt(weights.T @ cov_matrix @ weights)

        constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
        bounds = [(0, 0.5) for _ in range(len(tickers))]
        initial_weights = np.array([1 / len(tickers)] * len(tickers))

        optimized_results = minimize(
            neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate),
            method="SLSQP", constraints=constraints, bounds=bounds
        )
        slsqp_optimal_weights = optimized_results.x

        # ✅ Consensus Method (Average of Monte Carlo & SLSQP)
        consensus_weights = (mc_optimal_weights + slsqp_optimal_weights) / 2

        # ✅ Display Portfolio Weights
        st.subheader("Optimal Portfolio Weights")
        portfolio_results = pd.DataFrame({
            "Ticker": tickers,
            "Monte Carlo": mc_optimal_weights,
            "SLSQP": slsqp_optimal_weights,
            "Consensus": consensus_weights
        })
        st.write(portfolio_results)

        # ✅ Portfolio Performance Calculation
        def calc_portfolio_performance(weights):
            port_return = np.sum(mean_returns * weights)
            port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe_ratio = (port_return - risk_free_rate) / port_volatility
            return port_return, port_volatility, sharpe_ratio

        mc_return, mc_volatility, mc_sharpe = calc_portfolio_performance(mc_optimal_weights)
        slsqp_return, slsqp_volatility, slsqp_sharpe = calc_portfolio_performance(slsqp_optimal_weights)
        consensus_return, consensus_volatility, consensus_sharpe = calc_portfolio_performance(consensus_weights)

        performance_results = pd.DataFrame({
            "Method": ["Monte Carlo", "SLSQP", "Consensus"],
            "Expected Return": [mc_return, slsqp_return, consensus_return],
            "Volatility": [mc_volatility, slsqp_volatility, consensus_volatility],
            "Sharpe Ratio": [mc_sharpe, slsqp_sharpe, consensus_sharpe]
        })
        st.subheader("Portfolio Performance")
        st.write(performance_results)

        # ✅ Portfolio Performance Over Time (Cumulative Returns)
        st.subheader("📈 Portfolio Cumulative Returns Over Time")
        cumulative_returns = (log_returns @ consensus_weights).cumsum()
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, np.exp(cumulative_returns), label="Consensus Portfolio", linewidth=2)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.title("Portfolio Cumulative Performance Over Time")
        plt.legend()
        st.pyplot(plt)

        # ✅ Portfolio Weights Bar Chart
        st.subheader("📊 Portfolio Weights Distribution")
        plt.figure(figsize=(10, 6))
        plt.bar(tickers, consensus_weights, color="blue")
        plt.xlabel("Assets")
        plt.ylabel("Weight Allocation")
        plt.title("Optimal Portfolio Weights (Consensus Method)")
        st.pyplot(plt)

        # ✅ Download Portfolio Data
        csv = portfolio_results.to_csv(index=False)
        st.download_button("📥 Download Portfolio Weights", data=csv, file_name="portfolio_weights.csv", mime="text/csv")

        csv_perf = performance_results.to_csv(index=False)
        st.download_button("📥 Download Portfolio Performance", data=csv_perf, file_name="portfolio_performance.csv", mime="text/csv")
