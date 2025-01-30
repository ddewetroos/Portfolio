import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog

# Streamlit App Title
st.title("Portfolio Optimization with Sector Constraints")

# User Input: Enter Stock Tickers
tickers = st.text_input("Enter stock tickers (comma separated):", "INFY.NS, TCS.NS, TATAMOTORS.NS, MARUTI.NS, SUNPHARMA.NS, CIPLA.NS, ITC.NS, MARICO.NS, GOLDBEES.NS")
tickers = [ticker.strip() for ticker in tickers.split(",")]

# Define sector mapping
sector_map = {
    'INFY.NS': 'IT', 'TCS.NS': 'IT',
    'TATAMOTORS.NS': 'Auto', 'MARUTI.NS': 'Auto',
    'SUNPHARMA.NS': 'Pharma', 'CIPLA.NS': 'Pharma',
    'ITC.NS': 'Consumer', 'MARICO.NS': 'Consumer',
    'GOLDBEES.NS': 'Gold'
}

# Allow users to define sector constraints
st.sidebar.header("Sector Constraints")
sector_constraints = {}
for sector in set(sector_map.values()):
    min_val = st.sidebar.slider(f"Min allocation for {sector}", 0.0, 1.0, 0.05, 0.01)
    max_val = st.sidebar.slider(f"Max allocation for {sector}", min_val, 1.0, 0.5, 0.01)
    sector_constraints[sector] = (min_val, max_val)

# Time range
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)

# Download stock data
adj_close_df = pd.DataFrame()
for ticker in tickers:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        adj_close_df[ticker] = data['Adj Close']
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")

# Drop tickers with missing data
adj_close_df.dropna(axis=1, inplace=True)
tickers = adj_close_df.columns.tolist()

# Compute log returns
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
cov_matrix = log_returns.cov() * 252
mean_returns = log_returns.mean() * 252

# Risk-Free Rate (Indian 10-year Govt Bond Yield)
risk_free_rate = 0.071  # 7.1%

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

### **🔹 SLSQP Optimization**
def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -(np.sum(log_returns.mean() * weights) * 252 - risk_free_rate) / np.sqrt(weights.T @ cov_matrix @ weights)

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.5) for _ in range(len(log_returns.columns))]  # Max 50% per asset
initial_weights = np.array([1 / len(log_returns.columns)] * len(log_returns.columns))

optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate),
                             method='SLSQP', constraints=constraints, bounds=bounds)

slsqp_optimal_weights = optimized_results.x

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

A_ub = np.array(A_ub)
b_ub = np.array(b_ub)
A_lb = np.array(A_lb)
b_lb = np.array(b_lb)
A_combined = np.vstack((A_ub, A_lb))
b_combined = np.hstack((b_ub, b_lb))
bounds = [(0, 0.5) for _ in range(len(tickers))]

lp_result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_combined, b_ub=b_combined, bounds=bounds, method="highs")
consensus_weights = lp_result.x if lp_result.success else None

### **🔹 Display Portfolio Weights**
st.subheader("Portfolio Weights")
st.write("Monte Carlo Optimization:", dict(zip(tickers, mc_optimal_weights)))
st.write("SLSQP Optimization:", dict(zip(tickers, slsqp_optimal_weights)))

if consensus_weights is not None:
    st.write("Consensus Optimization (Sector Constrained):", dict(zip(tickers, consensus_weights)))
else:
    st.error("Linear programming failed to find a consensus solution.")

### **🔹 Visualization: Portfolio Weight Comparisons**
plt.figure(figsize=(12, 6))
bar_width = 0.3
x_indices = np.arange(len(log_returns.columns))

plt.bar(x_indices - bar_width, mc_optimal_weights, bar_width, label="Monte Carlo", alpha=0.7)
plt.bar(x_indices, slsqp_optimal_weights, bar_width, label="SLSQP", alpha=0.7)
if consensus_weights is not None:
    plt.bar(x_indices + bar_width, consensus_weights, bar_width, label="Consensus (Sector Constrained)", alpha=0.7)

plt.xlabel("Assets")
plt.ylabel("Portfolio Weights")
plt.title("Portfolio Weight Comparisons")
plt.xticks(ticks=x_indices, labels=log_returns.columns, rotation=45)
plt.legend()
st.pyplot(plt)
