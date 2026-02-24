import streamlit as st
import pandas as pd
import altair as alt

# ------------------ Streamlit App ------------------

st.title("📈 Stock Sentiment Data Merge")

st.write("Upload your existing CSVs for stock data and news sentiment, and merge them on `DateOnly`.")

# Upload files
stock_file = st.file_uploader("📊 Upload Stock Data CSV", type="csv")
news_file = st.file_uploader("📰 Upload News Sentiment CSV", type="csv")

if stock_file and news_file:
    try:
        # --- Read CSV files ---
        stock_data = pd.read_csv(stock_file)
        news_data = pd.read_csv(news_file)

        # --- Ensure DateOnly is properly formatted ---
        stock_data['DateOnly'] = pd.to_datetime(stock_data['DateOnly'], errors='coerce').dt.date
        news_data['DateOnly'] = pd.to_datetime(news_data['DateOnly'], errors='coerce').dt.date

        # --- Prepare daily sentiment ---
        sentiment_col = None
        for col in ['Avg_Sentiment_Score', 'Sentiment_Score']:
            if col in news_data.columns:
                sentiment_col = col
                break

        if sentiment_col:
            daily_sentiment = (
                news_data.groupby('DateOnly')[sentiment_col].mean().reset_index()
                .rename(columns={sentiment_col: 'Avg_Sentiment_Score'})
            )
        else:
            st.warning("⚠️ News CSV must contain a 'Sentiment_Score' or 'Avg_Sentiment_Score' column.")
            daily_sentiment = pd.DataFrame()

        # --- Prepare stock data ---
        if 'Pct_Change' not in stock_data.columns and 'Close' in stock_data.columns:
            stock_data['Pct_Change'] = stock_data['Close'].pct_change() * 100

        stock_data['Prev_Pct_Change'] = stock_data['Pct_Change'].shift(1).fillna(0)

        if 'Volume' in stock_data.columns:
            stock_data['Volume'] = stock_data['Volume'] / 1_000_000  # scale volume to millions

        # --- Merge datasets ---
        if not daily_sentiment.empty:
            merge_cols = ['DateOnly', 'Pct_Change', 'Prev_Pct_Change']
            if 'Volume' in stock_data.columns:
                merge_cols.append('Volume')

            merged_data = pd.merge(
                daily_sentiment,
                stock_data[merge_cols],
                on='DateOnly',
                how='inner'
            )

            st.success("✅ Data merged successfully!")
            st.subheader("📊 Merged Data Preview")
            st.dataframe(merged_data.head())

            # --- Optional date range filter ---
            min_date, max_date = merged_data['DateOnly'].min(), merged_data['DateOnly'].max()
            date_range = st.date_input("📅 Select Date Range", [min_date, max_date])

            if len(date_range) == 2:
                start, end = date_range
                merged_data = merged_data[
                    (merged_data['DateOnly'] >= start) & (merged_data['DateOnly'] <= end)
                ]

            # --- Correlation metric ---
            correlation = merged_data['Avg_Sentiment_Score'].corr(merged_data['Pct_Change'])
            st.metric("📊 Correlation (Sentiment vs % Change)", f"{correlation:.2f}")

            # --- Visualization: Sentiment vs Stock % Change ---
            st.subheader("📈 Sentiment vs Stock % Change Over Time")

            chart = alt.Chart(merged_data).transform_fold(
                ['Avg_Sentiment_Score', 'Pct_Change'],
                as_=['Metric', 'Value']
            ).mark_line().encode(
                x='DateOnly:T',
                y='Value:Q',
                color='Metric:N'
            ).properties(title="Sentiment and Stock % Change")
            st.altair_chart(chart, use_container_width=True)

            # --- Download merged file ---
            st.download_button(
                "⬇️ Download Merged Data (CSV)",
                merged_data.to_csv(index=False).encode('utf-8'),
                file_name="merged_data.csv",
                mime='text/csv'
            )

        else:
            st.warning("⚠️ Daily sentiment data is empty — cannot merge.")

    except Exception as e:
        st.error(f"❌ Error processing files: {e}")
