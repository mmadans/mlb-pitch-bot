"""
Feature engineering from raw MLB API game JSON.
Computes Pitcher's Situational tendencies (global and by count).
"""

from collections import Counter
import pandas as pd
from src.constants import FASTBALL_CODES, BREAKING_CODES, OFFSPEED_CODES


def _classify_pitch_family(code: str | None) -> str:
    """Map pitch type code to family: Fastball, Breaking, Offspeed, or Other."""
    if not isinstance(code, str) or not code:
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


def extract_pitches_with_context(play_data: dict, game_date: str | None = None) -> list[dict]:
    """
    Extract pitch-level rows with score at time of pitch and previous pitch in AB.
    play_data: raw game JSON from MLB API.
    game_date: optional date string (YYYY-MM-DD). If not provided, tries to extract from play_data.
    """
    all_plays = play_data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    pitches = []
    home_score, away_score = 0, 0
    pitcher_pitch_counts = {}
    matchup_tracker = {}

    for play in all_plays:
        if "playEvents" not in play:
            result = play.get("result", {})
            home_score = result.get("homeScore", home_score)
            away_score = result.get("awayScore", away_score)
            continue

        matchup = play.get("matchup", {})
        batter_info = matchup.get("batter") or {}
        batter_name = batter_info.get("fullName", "")
        batter_id = batter_info.get("id")
        
        pitcher_info = matchup.get("pitcher") or {}
        pitcher_name = pitcher_info.get("fullName", "")
        pitcher_id = pitcher_info.get("id")

        about = play.get("about", {})
        inning = about.get("inning") or 0
        half = about.get("halfInning", "")

        at_bat_index = about.get("atBatIndex")
        pitch_events = [e for e in play["playEvents"] if e.get("isPitch")]
        prev_pitch_type_in_ab = None
        prev_pitch_call_in_ab = None
        prev_pX_in_ab = 0.0
        prev_pZ_in_ab = 0.0
        
        if pitcher_id and batter_id:
            matchup_tracker[(pitcher_id, batter_id)] = matchup_tracker.get((pitcher_id, batter_id), 0) + 1
        times_faced = matchup_tracker.get((pitcher_id, batter_id), 1)
        
        # Streak tracking for the current at-bat
        breaking_streak = 0
        fastball_streak = 0
        offspeed_streak = 0

        if not game_date:
            game_date = play_data.get("gameData", {}).get("datetime", {}).get("officialDate", "")
        venue_id = play_data.get("gameData", {}).get("venue", {}).get("id")
        for event in pitch_events:
            details = event.get("details", {})
            type_info = details.get("type", {})
            code = type_info.get("code")
            pitch_family = _classify_pitch_family(code)
            
            pitch_data = event.get("pitchData", {})
            count = event.get("count", {})
            
            if pitcher_id:
                pitcher_pitch_counts[pitcher_id] = pitcher_pitch_counts.get(pitcher_id, 0) + 1
            pitch_count = pitcher_pitch_counts.get(pitcher_id, 1)

            pitch = {
                "at_bat_index": at_bat_index,
                "pitch_index": event.get("index"),
                "game_date": game_date,
                "batter": batter_name,
                "batter_id": batter_id,
                "pitcher": pitcher_name,
                "pitcher_id": pitcher_id,
                "pitch_type": code,
                "pitch_family": pitch_family,
                "pitcher_hand": matchup.get("pitchHand", {}).get("code"),
                "batter_side": matchup.get("batSide", {}).get("code"),
                "men_on_base": matchup.get("splits", {}).get("menOnBase"),
                "call": details.get("description"),
                "velocity": pitch_data.get("startSpeed"),
                "spin_rate": (pitch_data.get("breaks") or {}).get("spinRate"),
                "zone": pitch_data.get("zone"),
                "pX": (pitch_data.get("coordinates") or {}).get("pX"),
                "pZ": (pitch_data.get("coordinates") or {}).get("pZ"),
                "pitch_type_desc": type_info.get("description"),
                "inning": inning,
                "half_inning": half,
                "balls": count.get("balls"),
                "strikes": count.get("strikes"),
                "outs": count.get("outs"),
                "score_home": home_score,
                "score_away": away_score,
                "park_id": venue_id,
                "game_pk": play_data.get("gamePk"),
                "game_type": play_data.get("gameData", {}).get("game", {}).get("type"),
                "prev_pitch_type_in_ab": prev_pitch_type_in_ab,
                "prev_pitch_call": prev_pitch_call_in_ab,
                "prev_pX": prev_pX_in_ab,
                "prev_pZ": prev_pZ_in_ab,
                "pitch_count_in_game": pitch_count,
                "times_faced_today": times_faced,
                "breaking_streak": breaking_streak,
                "fastball_streak": fastball_streak,
                "offspeed_streak": offspeed_streak,
            }
            pitches.append(pitch)
            
            # Update streaks for the NEXT pitch
            if pitch_family == "Breaking":
                breaking_streak += 1
                fastball_streak = 0
                offspeed_streak = 0
            elif pitch_family == "Fastball":
                fastball_streak += 1
                breaking_streak = 0
                offspeed_streak = 0
            elif pitch_family == "Offspeed":
                offspeed_streak += 1
                breaking_streak = 0
                fastball_streak = 0
            else:
                # 'Other' resest all streaks
                breaking_streak = 0
                fastball_streak = 0
                offspeed_streak = 0

            prev_pitch_type_in_ab = (code or "UN").upper() if code else "UN"
            prev_pitch_call_in_ab = details.get("description")
            try:
                prev_pX_in_ab = float((pitch_data.get("coordinates") or {}).get("pX") or 0.0)
                prev_pZ_in_ab = float((pitch_data.get("coordinates") or {}).get("pZ") or 0.0)
            except (ValueError, TypeError):
                prev_pX_in_ab = 0.0
                prev_pZ_in_ab = 0.0

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
    
    # Run Differential: Pitching Team Score - Batting Team Score
    df["run_differential"] = 0
    is_top = (df["half_inning"].str.lower() == "top")
    # Top inning: Away bats, Home pitches
    df.loc[is_top, "run_differential"] = score_home[is_top] - score_away[is_top]
    # Bottom inning: Home bats, Away pitches
    df.loc[~is_top, "run_differential"] = score_away[~is_top] - score_home[~is_top]

    df["score_diff"] = (score_home - score_away).abs()
    inning = df["inning"].fillna(0).astype(int)
    df["is_leverage"] = ((inning > 7) & (df["score_diff"] <= 2)).astype(int)

    if "men_on_base" in df.columns and "outs" in df.columns:
        df["is_double_play_scenario"] = (((df["men_on_base"] == "Men_On") | (df["men_on_base"] == "Loaded")) & (df["outs"] < 2)).astype(int)
    else:
        df["is_double_play_scenario"] = 0

    if "prev_pitch_call" in df.columns:
        df["prev_pitch_was_whiff"] = df["prev_pitch_call"].str.lower().str.contains("swinging strike", na=False).astype(int)
        df["prev_pitch_was_foul"] = df["prev_pitch_call"].str.lower().str.contains("foul", na=False).astype(int)
    else:
        df["prev_pitch_was_whiff"] = 0
        df["prev_pitch_was_foul"] = 0

    # Drop helper columns used only for feature construction
    df = df.drop(columns=["count_str", "score_diff"], errors="ignore")

    return df


def add_pitcher_count_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds pitcher tendency features specific to each count (balls and strikes).
    For each (pitcher, balls, strikes), it calculates the frequency of each pitch type.
    """
    # Group by pitcher_id, balls, strikes, and pitch_family to get counts
    if 'pitch_family' not in df.columns:
        df['pitch_family'] = df['pitch_type'].apply(_classify_pitch_family)
    group_cols = ["pitcher_id", "balls", "strikes", "pitch_family"]
    counts = df.groupby(group_cols).size().unstack(fill_value=0)
    
    # Calculate percentages per (pitcher_id, balls, strikes) with Laplace Smoothing
    totals = counts.sum(axis=1)
    num_classes = counts.shape[1]
    smoothed_counts = counts + 1.0
    smoothed_totals = totals + num_classes
    percentages = smoothed_counts.div(smoothed_totals, axis=0)
    
    # Rename columns to reflect they are count-specific tendencies
    percentages.columns = [
        f"tendency_count_{col}_pct" for col in percentages.columns
    ]
    percentages["tendency_count_total_pitches"] = totals
    
    # Merge back to original dataframe
    # Reset index for the join
    percentages = percentages.reset_index()
    df = df.merge(percentages, on=["pitcher_id", "balls", "strikes"], how="left")
    
    return df


def add_batter_count_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds batter tendency features specific to each count (balls and strikes).
    For each (batter, balls, strikes), it calculates the frequency of each pitch family.
    Historically, this captures how the league typically scouts/attacks this batter.
    """
    if df.empty:
        return df
        
    # We want to use pitch families for more robust batter scouting
    df = df.copy()
    if 'pitch_family' not in df.columns:
        df['pitch_family'] = df['pitch_type'].apply(_classify_pitch_family)
        
    # Group by batter, balls, strikes, and pitch_family to get counts
    group_cols = ["batter_id", "balls", "strikes", "pitch_family"]
    counts = df.groupby(group_cols).size().unstack(fill_value=0)
    
    # Calculate percentages per (batter, balls, strikes) with Laplace Smoothing
    totals = counts.sum(axis=1)
    num_classes = counts.shape[1]
    smoothed_counts = counts + 1.0
    smoothed_totals = totals + num_classes
    percentages = smoothed_counts.div(smoothed_totals, axis=0)
    
    # Rename columns to reflect they are batter-count tendencies
    percentages.columns = [
        f"tendency_batter_count_{col}_pct" for col in percentages.columns
    ]
    percentages["tendency_batter_count_total_pitches"] = totals
    
    # Merge back to original dataframe
    percentages = percentages.reset_index()
    df = df.merge(percentages, on=["batter_id", "balls", "strikes"], how="left")
    
    return df


def add_league_count_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds league-wide pitch family frequencies for each count (balls and strikes).
    Provides a "baseline" of what is expected regardless of the specific pitcher.
    """
    if df.empty:
        return df
        
    df = df.copy()
    if 'pitch_family' not in df.columns:
        df['pitch_family'] = df['pitch_type'].apply(_classify_pitch_family)
        
    # Group by balls, strikes, and pitch_family to get league averages
    group_cols = ["balls", "strikes", "pitch_family"]
    counts = df.groupby(group_cols).size().unstack(fill_value=0)
    
    # Calculate percentages per count with Laplace Smoothing
    totals = counts.sum(axis=1)
    num_classes = counts.shape[1]
    smoothed_counts = counts + 1.0
    smoothed_totals = totals + num_classes
    percentages = smoothed_counts.div(smoothed_totals, axis=0)
    
    # Rename columns to reflect they are league-count tendencies
    percentages.columns = [
        f"tendency_league_count_{col}_pct" for col in percentages.columns
    ]
    percentages["tendency_league_count_total_pitches"] = totals
    
    # Merge back to original dataframe (join on count only)
    percentages = percentages.reset_index()
    df = df.merge(percentages, on=["balls", "strikes"], how="left")
    
    return df


def add_pitcher_out_pitch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the "Out Pitch" (highest K-rate) for each pitcher and adds it as a feature.
    """
    if df.empty:
        return df
        
    df = df.copy()
    if 'pitch_family' not in df.columns:
        df['pitch_family'] = df['pitch_type'].apply(_classify_pitch_family)
        
    # Analyze strikeouts by pitch family per pitcher
    # Assuming 'call' contains strikeout info if it's the 3rd strike
    # However, 'call' is usually 'called strike' or 'swinging strike'.
    # For a more robust 'Out Pitch', we should ideally look at the result of the play.
    # But since we're at pitch level, let's define it as "pitches that lead to a whiff on strike 2".
    df["is_whiff_strike_2"] = ((df["strikes"] == 2) & 
                               df["call"].str.lower().str.contains("swinging strike|foul tip", na=False)).astype(int)
    
    out_pitch_stats = df.groupby(["pitcher_id", "pitch_family"]).agg(
        total_pitches=("pitch_family", "count"),
        total_whiffs=("is_whiff_strike_2", "sum")
    ).reset_index()
    
    out_pitch_stats["whiff_rate"] = out_pitch_stats["total_whiffs"] / out_pitch_stats["total_pitches"]
    
    # Get the family with the highest wharf rate for each pitcher (primary out pitch)
    idx = out_pitch_stats.groupby("pitcher_id")["whiff_rate"].idxmax()
    primary_out_pitches = out_pitch_stats.loc[idx, ["pitcher_id", "pitch_family"]]
    primary_out_pitches.columns = ["pitcher_id", "primary_out_pitch"]
    
    df = df.merge(primary_out_pitches, on="pitcher_id", how="left")
    return df


def add_global_pitcher_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds global (regardless of count) pitcher tendency features.
    """
    # Calculate pitch family counts per pitcher_id with Laplace Smoothing
    if 'pitch_family' not in df.columns:
        df['pitch_family'] = df['pitch_type'].apply(_classify_pitch_family)
    pitch_counts = df.groupby(["pitcher_id", "pitch_family"]).size().unstack(fill_value=0)
    total_pitches = pitch_counts.sum(axis=1)
    
    num_classes = pitch_counts.shape[1]
    smoothed_counts = pitch_counts + 1.0
    smoothed_totals = total_pitches + num_classes
    pitch_percentages = smoothed_counts.div(smoothed_totals, axis=0)

    # Rename columns
    pitch_percentages.columns = [
        f"tendency_global_{col}_pct" for col in pitch_percentages.columns
    ]
    pitch_percentages["tendency_total_pitches"] = total_pitches

    # Merge
    pitch_percentages = pitch_percentages.reset_index()
    df = df.merge(pitch_percentages, on="pitcher_id", how="left")
    return df
def apply_baseline_to_df(df: pd.DataFrame, baseline: dict, is_train: bool = False) -> pd.DataFrame:
    """Applies baseline tendency dictionaries to a DataFrame securely, unifying offline and online features."""
    df = df.copy()
    
    # Global
    if baseline.get('global'):
        df_global = pd.DataFrame.from_dict(baseline['global'], orient='index')
        df_global.index.name = 'pitcher_id'
        df = df.merge(df_global, on='pitcher_id', how='left')
    
    # Count
    if baseline.get('count'):
        df_count = pd.DataFrame.from_dict(baseline['count'], orient='index')
        if not df_count.empty:
            df_count.index.names = ['pitcher_id', 'balls', 'strikes']
            df = df.merge(df_count, on=['pitcher_id', 'balls', 'strikes'], how='left')
    
    # Batter Count
    if baseline.get('batter_count'):
        df_bcount = pd.DataFrame.from_dict(baseline['batter_count'], orient='index')
        if not df_bcount.empty:
            df_bcount.index.names = ['batter_id', 'balls', 'strikes']
            df = df.merge(df_bcount, on=['batter_id', 'balls', 'strikes'], how='left')
    
    # League count
    if baseline.get('league_count'):
        df_lcount = pd.DataFrame.from_dict(baseline['league_count'], orient='index')
        if not df_lcount.empty:
            df_lcount.index.names = ['balls', 'strikes']
            df = df.merge(df_lcount, on=['balls', 'strikes'], how='left')
    
    # Out pitch
    if baseline.get('out_pitch'):
        out_pitch_s = pd.Series(baseline['out_pitch'], name='primary_out_pitch')
        df = df.merge(out_pitch_s, left_on='pitcher_id', right_index=True, how='left')
    
    if "primary_out_pitch" not in df.columns:
        df["primary_out_pitch"] = "Fastball"
    df["primary_out_pitch"] = df["primary_out_pitch"].fillna("Fastball")
    
    # Add contextual features to ensure pitch_family and other base dependencies exist
    df = add_contextual_features(df)
    if 'pitch_family' not in df.columns and 'pitch_type' in df.columns:
        df['pitch_family'] = df['pitch_type'].apply(_classify_pitch_family)

    # Fill other missing tendencies with 0.3333 (Laplace default instead of 0.0)
    tend_cols = [c for c in df.columns if c.startswith('tendency_') and c.endswith('_pct')]
    df[tend_cols] = df[tend_cols].fillna(0.3333)
    
    if is_train and "pitch_family" in df.columns:
        # Prevent Target Leakage using Leave-One-Out (LOO) calculation for train data
        for prefix in ["tendency_global", "tendency_count", "tendency_batter_count", "tendency_league_count"]:
            total_col = f"{prefix}_total_pitches"
            if total_col in df.columns:
                for fam in ["Fastball", "Breaking", "Offspeed"]:
                    pct_col = f"{prefix}_{fam}_pct"
                    if pct_col in df.columns:
                        # Inverse calculation for Laplace smoothing:
                        # pct = (actual_count + 1) / (total + num_classes)
                        num_classes = 3
                        # We use rounding to combat floating point precision issues
                        actual_count = (df[pct_col] * (df[total_col] + num_classes) - 1.0).round()
                        
                        # Subtract 1 if this row's family matches the column's family
                        is_match = (df["pitch_family"] == fam).astype(int)
                        new_count = actual_count - is_match
                        new_total = df[total_col] - 1
                        
                        # Clip to prevent negative counts in edge cases
                        new_count = new_count.clip(lower=0)
                        new_total = new_total.clip(lower=0)
                        
                        # Re-apply Laplace Smoothing
                        new_pct = (new_count + 1.0) / (new_total + num_classes)
                        df[pct_col] = new_pct.fillna(0.3333)
    
    # Add platoon advantage
    if 'pitcher_hand' in df.columns and 'batter_side' in df.columns:
        df['is_platoon_advantage'] = ((df['pitcher_hand'] == df['batter_side']) & (df['batter_side'] != 'S')).astype(int)
    
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
