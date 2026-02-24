import streamlit as st               # For building the web app UI
import yfinance as yf               # To fetch historical stock data
from finvizfinance.quote import finvizfinance  # To get latest news headlines
import pandas as pd                 # For data manipulation
import requests                     # To send requests to the Ollama API

# ---------------------------------------
# Function to fetch news headlines using Finviz
# ---------------------------------------
def get_news_data(ticker):
    stock = finvizfinance(ticker)            # Create a Finviz object
    news_df = stock.ticker_news()            # Fetch news for the given stock ticker
    news_df['Title'] = news_df['Title'].str.lower()              # Convert titles to lowercase
    news_df['Date'] = pd.to_datetime(news_df['Date'])            # Convert to datetime format
    news_df['DateOnly'] = news_df['Date'].dt.date                # Extract just the date (not time)
    return news_df

# ---------------------------------------
# Function to classify sentiment using Ollama LLM and provide justification
# ---------------------------------------
def classify_sentiment_with_justification(text, model='llama3', timeout=300):
    # Construct a simple prompt to classify the sentiment and ask for justification
    prompt = f"Classify the sentiment of this news headline as Positive, Negative, or Neutral. Provide a brief justification for the classification:\n\n\"{text}\"\n\nSentiment and Justification:"
    try:
        # Send POST request to Ollama API running locally
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout  # Timeout to avoid infinite wait
        )
        # If successful, process the response
        if response.status_code == 200:
            result = response.json()["response"].strip()
            # Separate sentiment from justification (if present)
            sentiment, justification = result.split("\n", 1)
            sentiment = sentiment.strip().lower()
            if 'positive' in sentiment:
                sentiment = "Positive"
            elif 'negative' in sentiment:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            return sentiment, justification.strip()
        else:
            return "Unknown", "No justification available"
    except requests.exceptions.Timeout:
        return "Timeout", "No justification available"
    except Exception:
        return "Error", "No justification available"

# ---------------------------------------
# Streamlit UI Starts Here
# ---------------------------------------
st.title("📈 Stock News Sentiment Classifier")

# Input field for the stock ticker symbol (e.g., AAPL, TSLA, SBUX)
ticker = st.text_input("Enter stock ticker (e.g., SBUX)", "SBUX")

# When the button is clicked, the app fetches and displays data
if st.button("Get News and Sentiment Classification"):
    try:
        st.info(f"Fetching news for {ticker}...")

        # Fetch recent news headlines
        news_df = get_news_data(ticker)

        if news_df.empty:
            st.warning("No news data found.")
        else:
            # Limit to first 10 headlines to speed up classification
            st.info("Classifying sentiment using Ollama (first 10 headlines)...")
            news_df = news_df.head(10)

            # Display progress bar during sentiment classification
            progress = st.progress(0)
            sentiments = []
            justifications = []

            # Loop through each headline and classify sentiment
            for i, title in enumerate(news_df['Title']):
                sentiment, justification = classify_sentiment_with_justification(title)  # Call to Ollama with justification
                sentiments.append(sentiment)
                justifications.append(justification)
                progress.progress((i + 1) / len(news_df))  # Update progress bar

            # Add sentiment and justification results to the DataFrame
            news_df['Sentiment'] = sentiments
            news_df['Justification'] = justifications

            # Add emojis/colors for visual feedback
            def color_sentiment(sentiment):
                if sentiment == "Positive":
                    return "🟢 Positive"
                elif sentiment == "Negative":
                    return "🔴 Negative"
                elif sentiment == "Neutral":
                    return "⚪ Neutral"
                else:
                    return f"⚠️ {sentiment}"

            # Apply color-coded labels
            news_df['Sentiment_Colored'] = news_df['Sentiment'].apply(color_sentiment)

            # Display the final news DataFrame with colored sentiment
            st.subheader("📰 News Headlines with Sentiment")
            st.dataframe(news_df[['DateOnly', 'Title', 'Sentiment_Colored']].rename(
                columns={'DateOnly': 'Date', 'Sentiment_Colored': 'Sentiment'}
            ))

            # Display sentiment answers and justification
            st.subheader("💬 Sentiment Classification and Justification:")
            for idx, row in news_df.iterrows():
                st.markdown(f"**Headline**: {row['Title']}")
                st.markdown(f"**Sentiment**: {row['Sentiment']}")
                st.markdown(f"**Justification**: {row['Justification']}")
                st.markdown("---")

    except Exception as e:
        st.error(f"Error: {e}")  # Catch any exception and show error message
