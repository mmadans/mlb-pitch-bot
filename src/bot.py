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


from src.utils import get_pitch_abbr

TEAM_HASHTAGS = {
    "ARI": "#DBacks", "ATL": "#ForTheA", "BAL": "#Birdland", "BOS": "#DirtyWater",
    "CHC": "#Cubs", "CWS": "#WhiteSox", "CIN": "#ATOBTTR", "CLE": "#GuarsBall",
    "COL": "#Rockies", "DET": "#RepDetroit", "HOU": "#ChaseTheFight", "KC": "#FountainsUp",
    "LAA": "#RepTheHalo", "LAD": "#Dodgers", "MIA": "#FightinFish", "MIL": "#ThisIsMyCrew",
    "MIN": "#MNTwins", "NYM": "#LGM", "NYY": "#RepBX", "OAK": "#RootedInOakland",
    "PHI": "#RingTheBell", "PIT": "#LetsGoBucs", "SD": "#ForTheFaithful", "SF": "#SFGiants",
    "SEA": "#TridentsUp", "STL": "#ForTheLou", "TB": "#RaysUp", "TEX": "#AllForTX",
    "TOR": "#BlueJays50", "WSH": "#Natitude"
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
    sequence: list,
    narrative: str = "",
    away_team: str = "",
    home_team: str = "",
    pitcher_hand: str = "",
    batter_side: str = ""
) -> str:
    """
    Formats a detailed tweet for a 'Surprise Strikeout' with 280-char limit in mind.
    """
    action = "whiffs" if is_whiff else "freezes"
    prob_pct = f"{prob * 100:.1f}%"
    p_hand_str = f" ({pitcher_hand})" if pitcher_hand else ""
    b_side_str = f" ({batter_side})" if batter_side else ""
    
    # 1. Header with Narrative
    header_prefix = f"{narrative} " if narrative else ""
    header = f"{header_prefix}{pitcher}{p_hand_str} {action} {batter}{b_side_str} with a {pitch_type}.\n"
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
        if isinstance(s, dict):
            desc = s.get("pitch_type_desc", s.get("name", "Unknown"))
            return get_pitch_abbr(desc)
            
        # Fallback for "Family (Description)" strings
        if isinstance(s, str) and "(" in s:
            parts = s.split(" (")
            if len(parts) > 1:
                desc = parts[1].rstrip(")")
                return f"{get_pitch_abbr(desc)}"
        return str(s)[:2]

    # Add Hashtags
    hashtags = []
    if away_team in TEAM_HASHTAGS: hashtags.append(TEAM_HASHTAGS[away_team])
    if home_team in TEAM_HASHTAGS: hashtags.append(TEAM_HASHTAGS[home_team])

    # 4. Dynamic Sequence Truncation
    def build_tweet(seq_list):
        seq_abbrs = [format_seq_item(s) for s in seq_list]
        seq_str = "Sequence: " + " -> ".join(seq_abbrs)
        parts = [header, context, seq_str]
        
        hashtags_str = " ".join(hashtags)
        if hashtags_str:
            parts.append(hashtags_str)
            
        return "\n\n".join(parts)

    current_sequence = list(sequence)
    tweet = build_tweet(current_sequence)
    
    while len(tweet) > 280 and len(current_sequence) > 0:
        current_sequence.pop(0)
        if not current_sequence:
            parts = [header, context]
            if hashtags:
                parts.append(" ".join(hashtags))
            tweet = "\n\n".join(parts)
            break
        tweet = build_tweet(current_sequence)
        
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

