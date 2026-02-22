import os
import pandas as pd
import tweepy
from dotenv import load_dotenv

load_dotenv()

def get_twitter_conn_v2() -> tweepy.Client:
    """Get Twitter connection using API v2."""
    consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
    consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
    access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        return None

    return tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret
    )

def post_tweet(tweet_text: str):
    """Posts a tweet to Twitter using API v2."""
    client = get_twitter_conn_v2()
    if not client:
        print("Twitter credentials are not fully set in .env file. Skipping tweet.")
        print(f"DEBUG - Would have tweeted:\n{tweet_text}")
        return

    try:
        client.create_tweet(text=tweet_text)
        print("Tweet posted successfully!")
    except tweepy.TweepyException as e:
        print(f"Error posting tweet: {e}")


def load_latest_dataset(data_dir: str = "data") -> pd.DataFrame | None:
    """Loads the most recent pitch feature dataset from the data directory."""
    if not os.path.exists(data_dir):
        return None
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
    """
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


def format_tweet(pitcher: str, batter: str, pitch_type: str, surprisal: float, outcome: str) -> str:
    """Formats a tweet for an interesting pitch event."""
    if outcome == "strikeout":
        emoji = "🪓"
        header = "SURPRISING STRIKEOUT!"
    else:
        emoji = "💥"
        header = "SURPRISING HARD HIT!"

    tweet = (
        f"{emoji} {header} {emoji}\n\n"
        f"Pitcher: {pitcher}\n"
        f"Batter: {batter}\n"
        f"Pitch: {pitch_type}\n\n"
        f"Outcome: {outcome.replace('_', ' ').title()}"
    )
    if surprisal > 0:
        tweet += f"\nSurprisal: {surprisal:.2f} bits!"
    
    return tweet


if __name__ == "__main__":
    dataset = load_latest_dataset()
    if dataset is not None:
        surprising_pitch = find_surprising_pitch(dataset)
        # Dummy surprisal for formatting test
        tweet = format_tweet(
            surprising_pitch["pitcher"], 
            surprising_pitch["batter"], 
            surprising_pitch["pitch_type"], 
            4.5, 
            "strikeout"
        )
        print("\n--- TWEET PREVIEW ---\n")
        print(tweet)
        print("\n---------------------\n")
        # post_tweet(tweet)
