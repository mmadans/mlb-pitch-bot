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


PITCH_ABBR = {
    "Four-Seam Fastball": "FF",
    "Sinker": "SI",
    "Cutter": "FC",
    "Slider": "SL",
    "Sweeper": "ST",
    "Curveball": "CU",
    "Knuckle Curve": "KC",
    "Changeup": "CH",
    "Splitter": "FS",
    "Slurve": "SV",
    "Knuckleball": "KN",
    "Forkball": "FO",
    "Eephus": "EP",
    "Screwball": "SC",
    "Other": "OT",
    "Unknown": "UN"
}

def _get_pitch_abbr(full_name: str) -> str:
    """Returns a short abbreviation for a pitch type."""
    return PITCH_ABBR.get(full_name, full_name[:3].upper())

TEAM_HASHTAGS = {
    "ARI": "#DBacks", "ATL": "#BravesCountry", "BAL": "#Birdland", "BOS": "#DirtyWater",
    "CHC": "#YouHaveToSeeIt", "CWS": "#WhiteSox", "CIN": "#ATOBTTR", "CLE": "#ForTheLand",
    "COL": "#Rockies", "DET": "#RepDetroit", "HOU": "#NeverSettle", "KC": "#RaisedRoyal",
    "LAA": "#TheHaloWay", "LAD": "#LetsGoDodgers", "MIA": "#HomeOfBeisbol", "MIL": "#ThisIsMyCrew",
    "MIN": "#MNTwins", "NYM": "#LGM", "NYY": "#RepBX", "OAK": "#RootedInOakland",
    "PHI": "#RingTheBell", "PIT": "#LetsGoBucs", "SD": "#LetsGoPadres", "SF": "#SFGiants",
    "SEA": "#TrueToTheBlue", "STL": "#ForTheLou", "TB": "#RaysUp", "TEX": "#TexasRangers",
    "TOR": "#TOTHECORE", "WSH": "#Natitude"
}

def format_surprise_strikeout_tweet(
    pitcher: str, 
    batter: str, 
    pitch_type: str, 
    pitch_family: str,
    prob: float, 
    is_whiff: bool,
    inning_info: str,
    score_info: str,
    runners_info: str,
    outs: int,
    matchup_num: int,
    sequence: list[str],
    narrative: str = "",
    highlight_url: str = "",
    away_team: str = "",
    home_team: str = ""
) -> str:
    """
    Formats a detailed tweet for a 'Surprise Strikeout' with 280-char limit in mind.
    """
    action = "whiffs" if is_whiff else "freezes"
    prob_pct = f"{prob * 100:.1f}%"
    p_abbr = _get_pitch_abbr(pitch_type)
    
    # 1. Header with Narrative
    header_prefix = f"{narrative} " if narrative else ""
    header = f"{header_prefix}{pitcher} {action} {batter} with a {pitch_type}.\n"
    header += f"Prob: {prob_pct} of {pitch_family.replace('Breaking', 'Breaking Ball')}."
    
    # 2. Context (Expanded)
    context = (
        f"Inning: {inning_info}\n"
        f"Outs: {outs}\n"
        f"Score: {score_info}\n"
        f"Bases: {runners_info}"
    )
    
    # 3. Sequence (Abbreviations)
    def format_seq_item(s):
        # Expecting "Family (Description)"
        if "(" in s:
            fam, desc = s.split(" (")
            desc = desc.rstrip(")")
            return f"{_get_pitch_abbr(desc)}"
        return s[:2]

    seq_abbrs = [format_seq_item(s) for s in sequence]
    seq_str = "Sequence: " + " -> ".join(seq_abbrs)
    
    parts = [header, context, seq_str]
    
    # Add Hashtags
    hashtags = []
    if away_team in TEAM_HASHTAGS: hashtags.append(TEAM_HASHTAGS[away_team])
    if home_team in TEAM_HASHTAGS: hashtags.append(TEAM_HASHTAGS[home_team])
    if hashtags:
        parts.append(" ".join(hashtags))
        
    if highlight_url:
        parts.append(f"Video: {highlight_url}")
    
    tweet = "\n\n".join(parts)
    
    # Truncation safety
    if len(tweet) > 280:
        # If still over, drop the sequence
        parts = [header, context]
        if hashtags:
            parts.append(" ".join(hashtags))
        if highlight_url:
            parts.append(f"Video: {highlight_url}")
        tweet = "\n\n".join(parts)
        
    return tweet


def format_tweet(pitcher: str, batter: str, pitch_type: str, surprisal: float, outcome: str, away_team: str = "", home_team: str = "") -> str:
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
    if away_team in TEAM_HASHTAGS: hashtags.append(TEAM_HASHTAGS[away_team])
    if home_team in TEAM_HASHTAGS: hashtags.append(TEAM_HASHTAGS[home_team])
    if hashtags:
        tweet += "\n\n" + " ".join(hashtags)
    
    return tweet

