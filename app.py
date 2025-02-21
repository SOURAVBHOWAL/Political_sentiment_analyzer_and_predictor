import tweepy

# Set up Twitter API credentials
API_KEY = "KXwSWftm5Z0O1ifm3fupkaSd4"
API_SECRET = "BNuIgSQcBb57HYempvoFySEzQIOBgiUonjqDtcHAYM6CwrW7EK"
ACCESS_TOKEN = "1797495074734948354-0YeL7eC27bo0GOnaAn7TDvQyzK9Xfw"
ACCESS_SECRET = "VMjlQXh9evdilSnWj1pxHTwFMALB3R4UOtL34k0es7yKd"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

def fetch_tweets(keyword, count=25):
    """
    Fetch tweets in real-time based on a keyword.
    """
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode="extended").items(count):
        tweets.append({"text": tweet.full_text, "user": tweet.user.screen_name, "created_at": tweet.created_at})
    return tweets

from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiments(tweets):
    """
    Analyze sentiment for a list of tweets.
    """
    results = []
    for tweet in tweets:
        sentiment = sentiment_pipeline(tweet["text"][:512])[0]  # Limit text to 512 characters
        results.append({
            "text": tweet["text"],
            "sentiment": sentiment["label"],
            "score": sentiment["score"],
            "user": tweet["user"],
            "created_at": tweet["created_at"]
        })
    return results


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Main Dashboard
st.title("Political Sentiment Tracker")

keyword = st.text_input("Enter a keyword (e.g., 'election'):", value="election")
num_tweets = st.slider("Number of tweets to analyze:", min_value=10, max_value=25, value=10)

if st.button("Fetch and Analyze"):
    st.write("Fetching tweets...")
    tweets = fetch_tweets(keyword, num_tweets)
    st.write("Analyzing sentiments...")
    analyzed_data = analyze_sentiments(tweets)

    # Convert to DataFrame
    df = pd.DataFrame(analyzed_data)

    # Display results
    st.subheader("Sentiment Analysis Results")
    st.write(df)

    # Sentiment distribution
    sentiment_counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax, color=["green", "red", "blue"])
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Plot timeline
    st.subheader("Sentiment Over Time")
    df["created_at"] = pd.to_datetime(df["created_at"])
    timeline_data = df.groupby([df["created_at"].dt.date, "sentiment"]).size().reset_index(name="count")
    timeline_chart = px.line(
        timeline_data, x="created_at", y="count", color="sentiment", title="Sentiment Over Time"
    )
    st.plotly_chart(timeline_chart)
