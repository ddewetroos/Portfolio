import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog
from textblob import TextBlob
import requests

# News API Key (Register at https://newsapi.org/)
NEWS_API_KEY = "your_newsapi_key_here"

st.title("🚀 Enhanced Portfolio Optimization with Live Data & Sentiment Analysis")

# User Input: Stock Tickers
tickers = st.text_input("Enter stock tickers (comma separated):",
                        "INFY.NS, TCS.NS, TATAMOTORS.NS, MARUTI.NS, SUNPHARMA.NS, CIPLA.NS, ITC.NS, MARICO.NS, GOLDBEES.NS")
tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

# Function to get stock sectors dynamically from Yahoo Finance
def get_stock_sector(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('sector', 'Unknown')
    except:
        return 'Unknown'

# Fetch stock sectors dynamically
sector_map = {ticker: get_stock_sector(ticker) for ticker in tickers}
unique_sectors = set(sector_map.values())

# Sidebar: Sector Constraints
st.sidebar.header("Sector Constraints")
sector_constraints = {}
for sector in unique_sectors:
    min_val = st.sidebar.slider(f"Min allocation for {sector}", 0.0, 1.0, 0.05, 0.01)
    max_val = st.sidebar.slider(f"Max allocation for {sector}", min_val, 1.0, 0.5, 0.01)
    sector_constraints[sector] = (min_val, max_val)

# Time range for stock data
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)

# Download historical stock data
adj_close_df = pd.DataFrame()
for ticker in tickers:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        adj_close_df[ticker] = data['Adj Close']
    except:
        st.error(f"Error downloading data for {ticker}")

# Drop tickers with missing data
adj_close_df.dropna(axis=1, inplace=True)
tickers = adj_close_df.columns.tolist()

# Compute log returns
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
cov_matrix = log_returns.cov() * 252
mean_returns = log_returns.mean() * 252

# Risk-Free Rate
risk_free_rate = 0.071

### **🔹 Monte Carlo Optimization**
num_portfolios = 100000
results = np.zeros((3, num_portfolios))
all_weights = np.zeros((num_portfolios, len(log_returns.columns)))

for i in range(num_portfolios):
    weights = np.random.random(len(log_returns.columns))
    weights /= np.sum(weights)

    all_weights[i, :] = weights
    port_return = np.sum(mean_returns * weights)
    port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility

    results[0, i] = port_return
    results[1, i] = port_volatility
    results[2, i] = sharpe_ratio

max_sharpe_idx = np.argmax(results[2])
mc_optimal_weights = all_weights[max_sharpe_idx, :]

# Display Monte Carlo Results
st.subheader("📊 Monte Carlo Optimization Results")
st.write(f"**Expected Annual Return:** {results[0, max_sharpe_idx]:.4f}")
st.write(f"**Expected Volatility:** {results[1, max_sharpe_idx]:.4f}")
st.write(f"**Sharpe Ratio:** {results[2, max_sharpe_idx]:.4f}")

### **🔹 SLSQP Optimization**
def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -(np.sum(log_returns.mean() * weights) * 252 - risk_free_rate) / np.sqrt(weights.T @ cov_matrix @ weights)

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.5) for _ in range(len(log_returns.columns))]
initial_weights = np.array([1 / len(log_returns.columns)] * len(log_returns.columns))

optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate),
                             method='SLSQP', constraints=constraints, bounds=bounds)

slsqp_optimal_weights = optimized_results.x

# Display SLSQP Results
st.subheader("📊 SLSQP Optimization Results")
st.write(f"**Expected Annual Return:** {np.sum(mean_returns * slsqp_optimal_weights):.4f}")
st.write(f"**Expected Volatility:** {np.sqrt(slsqp_optimal_weights.T @ cov_matrix @ slsqp_optimal_weights):.4f}")

### **🔹 Linear Programming with Sector Constraints**
c = np.abs(slsqp_optimal_weights - mc_optimal_weights)
A_eq = np.ones((1, len(tickers)))
b_eq = np.array([1])
A_ub, b_ub, A_lb, b_lb = [], [], [], []

for sector, (min_val, max_val) in sector_constraints.items():
    sector_indices = [i for i, t in enumerate(tickers) if sector_map.get(t) == sector]
    if sector_indices:
        A_row = np.zeros(len(tickers))
        A_row[sector_indices] = 1
        A_ub.append(A_row)
        b_ub.append(max_val)
        A_lb.append(-A_row)
        b_lb.append(-min_val)

A_ub, b_ub = np.array(A_ub), np.array(b_ub)
A_lb, b_lb = np.array(A_lb), np.array(b_lb)
A_combined = np.vstack((A_ub, A_lb))
b_combined = np.hstack((b_ub, b_lb))
bounds = [(0, 0.5) for _ in range(len(tickers))]

lp_result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_combined, b_ub=b_combined, bounds=bounds, method="highs")
consensus_weights = lp_result.x if lp_result.success else None

# Display Consensus Results
st.subheader("📊 Consensus Portfolio Results")
if consensus_weights is not None:
    portfolio_df = pd.DataFrame({"Ticker": tickers, "Weight": consensus_weights})
    st.dataframe(portfolio_df)
else:
    st.warning("Linear programming failed.")

# **Live News Sentiment Analysis**
st.subheader("📰 Market Sentiment Analysis")
for ticker in tickers[:3]:
    response = requests.get(f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}")
    articles = response.json().get("articles", [])
    sentiments = [TextBlob(article["title"]).sentiment.polarity for article in articles if "title" in article]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    st.write(f"**{ticker} Sentiment:** {'📈 Positive' if avg_sentiment > 0 else '📉 Negative'} ({avg_sentiment:.2f})")

# **Auto-start Optimization**
if sector_constraints:
    st.rerun()
