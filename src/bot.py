"""
This script runs the MLB Pitch Bot, which finds a surprising pitch
and posts it to Twitter.
"""

import os
import pandas as pd
import tweepy
from dotenv import load_dotenv

load_dotenv()

def post_tweet(tweet_text: str):
    """Posts a tweet to Twitter."""
    # Authenticate to Twitter
    # Make sure to set these environment variables in your .env file
    consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
    consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
    access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        print("Twitter credentials are not fully set in .env file. Skipping tweet.")
        return

    try:
        auth = tweepy.OAuth1UserHandler(
            consumer_key, consumer_secret, access_token, access_token_secret
        )
        api = tweepy.API(auth)
        api.update_status(tweet_text)
        print("Tweet posted successfully!")
    except tweepy.TweepyException as e:
        print(f"Error posting tweet: {e}")


def load_latest_dataset(data_dir: str = "data") -> pd.DataFrame | None:
    """Loads the most recent pitch feature dataset from the data directory."""
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("pitch_features_") and f.endswith(".csv")
    ]
    if not files:
        print("No dataset found.")
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading dataset: {latest_file}")
    return pd.read_csv(latest_file)


def find_surprising_pitch(df: pd.DataFrame) -> pd.Series:
    """
    Finds a "surprising" pitch from the dataset.

    A surprising pitch is defined as a pitch type thrown by a pitcher
    with a very low tendency for that pitch type.
    """
    # For now, let's define "surprising" as a pitch with the lowest tendency_pct
    # for the pitch type that was actually thrown.
    
    # Create a column with the tendency for the pitch that was thrown
    df["thrown_pitch_tendency"] = 0.0
    for pitch_type in df["pitch_type"].unique():
        if f"tendency_{pitch_type}_pct" in df.columns:
            mask = df["pitch_type"] == pitch_type
            df.loc[mask, "thrown_pitch_tendency"] = df.loc[
                mask, f"tendency_{pitch_type}_pct"
            ]

    # Find the pitch with the minimum tendency for the thrown pitch type
    # Also, filter for pitchers who have thrown a decent number of pitches
    surprising_pitch = df[df["tendency_total_pitches"] > 100].sort_values(
        by="thrown_pitch_tendency", ascending=True
    ).iloc[0]

    return surprising_pitch


def format_tweet(pitch_data: pd.Series) -> str:
    """Formats a tweet for a surprising pitch."""
    pitcher = pitch_data["pitcher"]
    batter = pitch_data["batter"]
    pitch_type = pitch_data["pitch_type"]
    tendency = pitch_data["thrown_pitch_tendency"]

    tweet = (
        f"⚾ Surprising Pitch! ⚾\n\n"
        f"Pitcher: {pitcher}\n"
        f"Batter: {batter}\n"
        f"Pitch Type: {pitch_type}\n\n"
        f"{pitcher} throws a {pitch_type} only {tendency:.1%} of the time!"
    )
    return tweet


def post_tweet(tweet_text: str):
    """Posts a tweet to Twitter."""
    try:
        api.update_status(tweet_text)
        print("Tweet posted successfully!")
    except tweepy.TweepyException as e:
        print(f"Error posting tweet: {e}")


if __name__ == "__main__":
    dataset = load_latest_dataset()
    if dataset is not None:
        surprising_pitch = find_surprising_pitch(dataset)
        tweet = format_tweet(surprising_pitch)
        print("\n--- TWEET ---\n")
        print(tweet)
        print("\n-------------\n")
        # Uncomment the line below to actually post the tweet
        # post_tweet(tweet)
