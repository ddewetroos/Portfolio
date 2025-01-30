import streamlit as st
import yfinance as yf
import pandas as pd
import time

st.title("Portfolio Optimization Debug")

# ✅ User-defined tickers
tickers = ["AAPL", "MSFT", "TSLA", "GOOGL"]  # Example tickers

st.subheader("Yahoo Finance Debug Test")

# ✅ Debug Test: Fetch a single stock to check if yfinance works
try:
    test_stock = yf.Ticker("AAPL")
    test_data = test_stock.history(period="1mo")
    st.success("Yahoo Finance is working!")
    st.write(test_data.head())  # Display sample data
except Exception as e:
    st.error(f"Yahoo Finance Error: {e}")

# ✅ Fetch Individual Stocks One-by-One
def fetch_stock_data(tickers, start_date, end_date, retries=3):
    adj_close_df = pd.DataFrame()

    for ticker in tickers:
        for attempt in range(retries):
            try:
                stock = yf.Ticker(ticker)
                stock_data = stock.history(start=start_date, end=end_date)

                # ✅ Use "Adj Close" if available, otherwise fallback to "Close"
                if "Adj Close" in stock_data.columns:
                    adj_close_df[ticker] = stock_data["Adj Close"]
                elif "Close" in stock_data.columns:
                    adj_close_df[ticker] = stock_data["Close"]
                else:
                    st.warning(f"⚠ No 'Adj Close' or 'Close' data for {ticker}. Skipping.")

                st.success(f"✅ Data downloaded for {ticker}")
                time.sleep(2)  # ✅ Wait 2 seconds between requests to avoid rate limits
                break  # ✅ Exit retry loop on success

            except Exception as e:
                st.warning(f"Attempt {attempt+1}: Failed to fetch {ticker} - {e}")
                time.sleep(5)  # ✅ Wait 5 seconds before retrying

    adj_close_df.dropna(axis=1, inplace=True)  # ✅ Remove tickers with missing data
    return adj_close_df

# ✅ Set Date Range
start_date = "2022-01-01"
end_date = "2023-12-31"

# ✅ Fetch Data
adj_close_df = fetch_stock_data(tickers, start_date, end_date)

# ✅ Display Data if Available
if not adj_close_df.empty:
    st.subheader("Stock Data (Adj Close / Close)")
    st.write(adj_close_df.head())

    # ✅ Download Data as CSV
    csv = adj_close_df.to_csv(index=True)
    st.download_button("📥 Download Stock Data", data=csv, file_name="portfolio_data.csv", mime="text/csv")
else:
    st.error("⚠ No valid stock data available! Check tickers or API access.")
