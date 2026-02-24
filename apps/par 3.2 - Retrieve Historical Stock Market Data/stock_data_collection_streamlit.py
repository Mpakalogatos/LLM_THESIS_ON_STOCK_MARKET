import streamlit as st
import yfinance as yf
from finvizfinance.quote import finvizfinance
import pandas as pd

# Function to get news data from Finviz
def get_news_data(ticker):
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()

    news_df['Title'] = news_df['Title'].str.lower()
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    news_df['DateOnly'] = news_df['Date'].dt.date
    return news_df

# Function to fetch historical stock data using yfinance
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Pct_Change'] = stock_data['Close'].pct_change() * 100
    return stock_data

# Streamlit UI section starts here
st.title("📈 Stock Data Viewer")

# Input box for user to enter stock ticker
ticker = st.text_input("Enter stock ticker (e.g., SBUX)", "SBUX")

# Button to trigger data fetching
if st.button("Get News and Stock Data"):
    try:
        st.info(f"Fetching news for {ticker}...")
        news_df = get_news_data(ticker)

        if news_df.empty:
            st.warning("No news data found.")
        else:
            st.subheader("📰 News Headlines")
            st.dataframe(news_df[['DateOnly', 'Title']])

            # Determine date range based on news
            start_date = news_df['DateOnly'].min().strftime('%Y-%m-%d')
            end_date = news_df['DateOnly'].max().strftime('%Y-%m-%d')

            st.info(f"Fetching stock data from {start_date} to {end_date}...")
            stock_data = get_stock_data(ticker, start_date, end_date)

            st.subheader("📊 Stock Prices and Daily Change")
            st.dataframe(stock_data[['Close', 'Pct_Change']].dropna())

            # Line chart of stock closing prices
            st.line_chart(stock_data['Close'])

    except Exception as e:
        st.error(f"Error: {e}")
