import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

st.title("📈 Stock Data Fetcher with Moving Averages")

# ------------------ Cached Stock Data Fetcher ------------------
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1))

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    if data.empty:
        return pd.DataFrame()

    # Calculate percentage change
    data['Pct_Change'] = data['Close'].pct_change() * 100
    data['Prev_Pct_Change'] = data['Pct_Change'].shift(1).fillna(0)

    # ✅ Add Moving Averages
    data['MA3'] = data['Close'].rolling(window=3).mean()
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()

    # Prepare for display
    data.reset_index(inplace=True)
    data['DateOnly'] = data['Date'].dt.date
    return data

# ------------------ Streamlit UI ------------------
ticker = st.text_input("Enter stock ticker (e.g., SBUX)", "SBUX")
start_date = st.date_input("Start Date", date.today() - timedelta(days=30))
end_date = st.date_input("End Date", date.today())

if st.button("Fetch Stock Data"):
    with st.spinner(f"Fetching data for {ticker}..."):
        data = get_stock_data(ticker, start_date, end_date)

    if data.empty:
        st.warning("⚠️ No stock data found. Please check your ticker or date range.")
    else:
        st.success("✅ Data fetched successfully!")

        st.subheader("📊 Stock Data Preview")
        st.dataframe(data.tail(15))  # show last 15 rows for clarity

        # ------------------ Chart with MAs ------------------
        st.subheader("📈 Closing Price with Moving Averages")
        st.line_chart(
            data.set_index("Date")[["Close", "MA3", "MA5", "MA10"]],
            use_container_width=True
        )

        # ------------------ CSV Download ------------------
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"{ticker}_{start_date}_{end_date}.csv",
            mime="text/csv",
        )
