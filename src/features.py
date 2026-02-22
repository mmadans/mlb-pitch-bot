"""
Feature engineering from raw MLB API game JSON.
Computes Pitcher's Tendency (last 5 pitches) and build_pitch_features
(count one-hot, is_leverage, previous pitch in AB).
"""

from collections import Counter

import pandas as pd


# Common pitch type groupings for tendency summary
FASTBALL_CODES = {"FF", "FT", "FC", "SI", "FS", "SF", "FA", "ST"}
BREAKING_CODES = {"SL", "CU", "KC", "SV", "CS", "CB", "GY"}
OFFSPEED_CODES = {"CH", "SC", "FO", "KN", "EP"}


def _classify_pitch_family(code: str | None) -> str:
    """Map pitch type code to family: Fastball, Breaking, Offspeed, or Other."""
    if not code:
        return "Other"
    code = code.upper()
    if code in FASTBALL_CODES:
        return "Fastball"
    if code in BREAKING_CODES:
        return "Breaking"
    if code in OFFSPEED_CODES:
        return "Offspeed"
    return "Other"


def _extract_pitches_in_order(game_json: dict) -> list[dict]:
    """
    Walk raw game JSON and yield pitch events in chronological order,
    each with play-level context (pitcher, batter, inning, count, etc.).
    """
    all_plays = game_json.get("liveData", {}).get("plays", {}).get("allPlays", [])
    pitches = []

    for play in all_plays:
        if "playEvents" not in play:
            continue
        matchup = play.get("matchup", {})
        batter = (matchup.get("batter") or {}).get("fullName", "")
        pitcher = (matchup.get("pitcher") or {}).get("fullName", "")
        about = play.get("about", {})
        inning = about.get("inning")
        half = about.get("halfInning", "")

        for event in play["playEvents"]:
            if not event.get("isPitch"):
                continue
            details = event.get("details", {})
            type_info = details.get("type", {})
            code = type_info.get("code")
            pitch_data = event.get("pitchData", {})
            count = event.get("count", {})

            pitch = {
                "batter": batter,
                "pitcher": pitcher,
                "pitch_type": code,
                "call": details.get("description"),
                "velocity": pitch_data.get("startSpeed"),
                "spin_rate": (pitch_data.get("breaks") or {}).get("spinRate"),
                "inning": inning,
                "half_inning": half,
                "balls": count.get("balls"),
                "strikes": count.get("strikes"),
                "outs": count.get("outs"),
            }
            pitches.append(pitch)

    return pitches


def extract_pitches_with_context(play_data: dict) -> list[dict]:
    """
    Extract pitch-level rows with score at time of pitch and previous pitch in AB.
    play_data: raw game JSON from MLB API (e.g. statsapi.get('game', {'gamePk': pk})).
    """
    all_plays = play_data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    pitches = []
    home_score, away_score = 0, 0

    for play in all_plays:
        if "playEvents" not in play:
            result = play.get("result", {})
            home_score = result.get("homeScore", home_score)
            away_score = result.get("awayScore", away_score)
            continue

        matchup = play.get("matchup", {})
        batter = (matchup.get("batter") or {}).get("fullName", "")
        pitcher = (matchup.get("pitcher") or {}).get("fullName", "")
        about = play.get("about", {})
        inning = about.get("inning") or 0
        half = about.get("halfInning", "")

        at_bat_index = about.get("atBatIndex")
        pitch_events = [e for e in play["playEvents"] if e.get("isPitch")]
        prev_pitch_type_in_ab = None

        for event in pitch_events:
            details = event.get("details", {})
            type_info = details.get("type", {})
            code = type_info.get("code")
            pitch_data = event.get("pitchData", {})
            count = event.get("count", {})

            pitch = {
                "at_bat_index": at_bat_index,
                "pitch_index": event.get("index"),
                "batter": batter,
                "pitcher": pitcher,
                "pitch_type": code,
                "call": details.get("description"),
                "velocity": pitch_data.get("startSpeed"),
                "spin_rate": (pitch_data.get("breaks") or {}).get("spinRate"),
                "inning": inning,
                "half_inning": half,
                "balls": count.get("balls"),
                "strikes": count.get("strikes"),
                "outs": count.get("outs"),
                "score_home": home_score,
                "score_away": away_score,
                "prev_pitch_type_in_ab": prev_pitch_type_in_ab,
            }
            pitches.append(pitch)
            prev_pitch_type_in_ab = (code or "UN").upper() if code else "UN"

        result = play.get("result", {})
        home_score = result.get("homeScore", home_score)
        away_score = result.get("awayScore", away_score)

    return pitches


# All (balls, strikes) count combinations for one-hot encoding
COUNT_LABELS = [
    "0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2",
    "3-0", "3-1", "3-2",
]


def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds contextual features to the pitch DataFrame.

    - One-hot encoding for current count (columns count_0_0, count_0_1, ... count_3_2)
    - is_leverage: 1 if inning > 7 and |home_score - away_score| <= 2, else 0
    """
    # Count string and one-hot
    df["balls"] = df["balls"].fillna(0).astype(int).clip(0, 3)
    df["strikes"] = df["strikes"].fillna(0).astype(int).clip(0, 2)
    df["count_str"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)

    for label in COUNT_LABELS:
        df[f"count_{label.replace('-', '_')}"] = (df["count_str"] == label).astype(int)

    # Leverage: inning > 7 and score diff <= 2
    score_home = df["score_home"].fillna(0).astype(int)
    score_away = df["score_away"].fillna(0).astype(int)
    df["score_diff"] = (score_home - score_away).abs()
    inning = df["inning"].fillna(0).astype(int)
    df["is_leverage"] = ((inning > 7) & (df["score_diff"] <= 2)).astype(int)

    # Drop helper columns used only for feature construction
    df = df.drop(columns=["count_str", "score_diff"], errors="ignore")

    return df


def add_pitcher_count_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds pitcher tendency features specific to each count (balls and strikes).
    For each (pitcher, balls, strikes), it calculates the frequency of each pitch type.
    """
    # Group by pitcher, balls, strikes, and pitch_type to get counts
    group_cols = ["pitcher", "balls", "strikes", "pitch_type"]
    counts = df.groupby(group_cols).size().unstack(fill_value=0)
    
    # Calculate percentages per (pitcher, balls, strikes)
    totals = counts.sum(axis=1)
    percentages = counts.div(totals, axis=0)
    
    # Rename columns to reflect they are count-specific tendencies
    percentages.columns = [
        f"tendency_count_{col}_pct" for col in percentages.columns
    ]
    
    # Merge back to original dataframe
    # Reset index for the join
    percentages = percentages.reset_index()
    df = df.merge(percentages, on=["pitcher", "balls", "strikes"], how="left")
    
    return df


def add_global_pitcher_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds global (regardless of count) pitcher tendency features.
    """
    # Calculate pitch type counts per pitcher
    pitch_counts = df.groupby(["pitcher", "pitch_type"]).size().unstack(fill_value=0)
    total_pitches = pitch_counts.sum(axis=1)
    pitch_percentages = pitch_counts.div(total_pitches, axis=0)

    # Rename columns
    pitch_percentages.columns = [
        f"tendency_global_{col}_pct" for col in pitch_percentages.columns
    ]
    pitch_percentages["tendency_total_pitches"] = total_pitches

    # Merge
    df = df.merge(pitch_percentages, on="pitcher", how="left")
    return df


if __name__ == "__main__":
    import statsapi
    import pandas as pd

    # Create a dummy dataframe to test the global tendency function
    dummy_data = {
        "pitcher": ["A", "A", "A", "B", "B", "B", "B"],
        "pitch_type": ["FF", "FF", "SL", "FF", "SL", "CU", "FF"],
        "batter": ["X", "Y", "X", "Y", "X", "Y", "X"],
    }
    dummy_df = pd.DataFrame(dummy_data)
    print("Original DataFrame:")
    print(dummy_df)

    tendency_df = add_global_pitcher_tendencies(dummy_df)
    print("\nDataFrame with Global Tendencies:")
    print(tendency_df.to_string())

    # Example: Pitcher A threw 3 pitches, 2 FF (66.7%) and 1 SL (33.3%)
    # Example: Pitcher B threw 4 pitches, 2 FF (50%), 1 SL (25%), 1 CU (25%)
