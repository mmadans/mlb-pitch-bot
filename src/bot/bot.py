"""
Handles Twitter (X) API interaction and tweet formatting.
"""

import os
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
        access_token_secret=access_token_secret,
    )


def post_tweet(tweet_text: str, image_path: str = None):
    """Posts a tweet to Twitter using API v2, with optional image support."""
    client = get_twitter_conn_v2()
    if not client:
        print("Twitter credentials are not fully set in .env file. Skipping tweet.")
        print(f"DEBUG - Would have tweeted:\n{tweet_text}")
        if image_path:
            print(f"DEBUG - Would have attached image: {image_path}")
        return

    try:
        media_ids = None
        if image_path and os.path.exists(image_path):
            # API v1.1 is required for media upload
            consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
            consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
            access_token = os.getenv("TWITTER_ACCESS_TOKEN")
            access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

            auth = tweepy.OAuth1UserHandler(
                consumer_key, consumer_secret, access_token, access_token_secret
            )
            api = tweepy.API(auth)

            media = api.media_upload(image_path)
            media_ids = [media.media_id]
            print(f"Successfully uploaded image {image_path}")

        client.create_tweet(text=tweet_text, media_ids=media_ids)
        print("Tweet posted successfully!")
    except tweepy.TweepyException as e:
        print(f"Error posting tweet: {e}")


from src.constants import TEAM_HASHTAGS


def format_surprise_strikeout_tweet(
    pitcher: str,
    batter: str,
    pitch_type: str,
    pitch_family: str,
    prob: float,
    is_whiff: bool,
    narrative: str = "",
    away_team: str = "",
    home_team: str = "",
    pitcher_hand: str = "",
    batter_side: str = "",
) -> str:
    """
    Formats a concise tweet for a 'Surprise Strikeout' relying on the attached infographic for context.
    """
    action = "whiffs" if is_whiff else "freezes"
    if prob < 0.001:
        prob_pct = "<0.1%"
    else:
        prob_pct = f"{prob * 100:.1f}%"
    p_hand_str = f" ({pitcher_hand})" if pitcher_hand else ""
    b_side_str = f" ({batter_side})" if batter_side else ""

    # 1. Header with Narrative
    header_prefix = f"{narrative} " if narrative else ""
    header = f"{header_prefix}{pitcher}{p_hand_str} {action} {batter}{b_side_str} with a {pitch_type}.\n"
    header += (
        f"Prob: {prob_pct} of {pitch_family.replace('Breaking', 'Breaking Ball')}."
    )

    # 2. Add Hashtags
    hashtags = []
    if away_team in TEAM_HASHTAGS:
        hashtags.append(TEAM_HASHTAGS[away_team])
    if home_team in TEAM_HASHTAGS:
        hashtags.append(TEAM_HASHTAGS[home_team])

    parts = [header]
    if hashtags:
        parts.append(" ".join(hashtags))

    return "\n\n".join(parts)


def format_tweet(
    pitcher: str,
    batter: str,
    pitch_type: str,
    surprisal: float,
    outcome: str,
    away_team: str = "",
    home_team: str = "",
) -> str:
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

    hashtags = []
    if away_team in TEAM_HASHTAGS:
        hashtags.append(TEAM_HASHTAGS[away_team])
    if home_team in TEAM_HASHTAGS:
        hashtags.append(TEAM_HASHTAGS[home_team])
    if hashtags:
        tweet += "\n\n" + " ".join(hashtags)

    return tweet
