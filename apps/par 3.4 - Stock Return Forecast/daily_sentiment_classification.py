import streamlit as st
import pandas as pd
import requests
import os

# ------------------ Sentiment Classification ------------------

def classify_sentiment(text, model='llama3', timeout=60):
    """Call Ollama locally to classify sentiment."""
    prompt = f"Classify the sentiment of this news headline as Positive, Negative, or Neutral:\n\n\"{text}\"\n\nSentiment:"
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout
        )
        if response.status_code == 200:
            result = response.json()["response"].strip().lower()
            if 'positive' in result:
                return "Positive"
            elif 'negative' in result:
                return "Negative"
            else:
                return "Neutral"
        else:
            return "Unknown"
    except requests.exceptions.Timeout:
        return "Timeout"
    except Exception:
        return "Error"

# ------------------ Streamlit App ------------------

st.title("📰 News Sentiment Classifier & Aggregator")

uploaded_file = st.file_uploader("Upload a news CSV file", type="csv")

if uploaded_file:
    news_df = pd.read_csv(uploaded_file)

    if 'Title' not in news_df.columns:
        st.error("CSV must have a 'Title' column.")
    else:
        # Ensure there's a Date column
        if 'Date' not in news_df.columns:
            st.error("CSV must have a 'Date' column.")
        else:
            news_df['Date'] = pd.to_datetime(news_df['Date'])
            news_df['DateOnly'] = news_df['Date'].dt.date

            if st.button("Classify Sentiment and Aggregate"):
                st.info("Classifying sentiment of news headlines...")
                sentiments = []
                progress = st.progress(0)
                for i, title in enumerate(news_df['Title']):
                    sentiment = classify_sentiment(title)
                    sentiments.append(sentiment)
                    progress.progress((i + 1) / len(news_df))

                news_df['Sentiment'] = sentiments
                # Map numeric sentiment score
                news_df['Sentiment_Score'] = news_df['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})

                # Aggregate by day
                daily_sentiment = news_df.groupby('DateOnly')['Sentiment_Score'].mean().reset_index()
                daily_sentiment.rename(columns={'Sentiment_Score': 'Avg_Sentiment_Score'}, inplace=True)
                daily_sentiment['Prev_Avg_Sentiment_Score'] = daily_sentiment['Avg_Sentiment_Score'].shift(1).fillna(0)

                os.makedirs("data_outputs", exist_ok=True)
                output_file = f"data_outputs/news_with_sentiment.csv"
                news_df.to_csv(output_file, index=False)
                daily_file = f"data_outputs/daily_sentiment.csv"
                daily_sentiment.to_csv(daily_file, index=False)

                st.success("✅ Sentiment classification and aggregation complete!")
                st.subheader("🗞 News with Sentiment")
                st.dataframe(news_df[['DateOnly', 'Title', 'Sentiment', 'Sentiment_Score']])

                st.subheader("📊 Daily Average Sentiment")
                st.dataframe(daily_sentiment)

                # Download buttons
                st.download_button(
                    "⬇️ Download News with Sentiment CSV",
                    news_df.to_csv(index=False).encode('utf-8'),
                    file_name="news_with_sentiment.csv",
                    mime='text/csv'
                )

                st.download_button(
                    "⬇️ Download Daily Sentiment CSV",
                    daily_sentiment.to_csv(index=False).encode('utf-8'),
                    file_name="daily_sentiment.csv",
                    mime='text/csv'
                )
