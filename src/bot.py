"""
Handles Twitter (X) API interaction and tweet formatting.
"""
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


def format_surprise_strikeout_tweet(
    pitcher: str, 
    batter: str, 
    pitch_type: str, 
    prob: float, 
    is_whiff: bool,
    inning_info: str,
    score_info: str,
    runners_info: str,
    outs: int,
    matchup_num: int,
    sequence: list[str],
    highlight_url: str = ""
) -> str:
    """
    Formats a detailed tweet for a 'Surprise Strikeout'.
    Template: <Pitcher> <freezes/whiffs> <batter> with <pitch> (% probability throwing it).
    """
    action = "whiffs" if is_whiff else "freezes"
    prob_pct = f"{prob * 100:.1f}%"
    
    header = f"🪓 {pitcher} {action} {batter} with a {pitch_type} ({prob_pct} prob)."
    
    context = (
        f"Context: {inning_info}, {score_info}, {runners_info}, {outs} Outs. "
        f"Matchup #{matchup_num}."
    )
    
    seq_str = "AB Sequence: " + ", ".join(sequence)
    
    tweet = f"{header}\n\n{context}\n{seq_str}"
    
    if highlight_url:
        tweet += f"\n\nWatch: {highlight_url}"
    
    return tweet


def format_tweet(pitcher: str, batter: str, pitch_type: str, surprisal: float, outcome: str) -> str:
    """
    Formats a tweet for an interesting pitch event.
    
    Args:
        pitcher: Full name of the pitcher.
        batter: Full name of the batter.
        pitch_type: Description of the pitch thrown.
        surprisal: Calculated surprisal in bits.
        outcome: Result of the play (e.g., 'strikeout').
    """
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

