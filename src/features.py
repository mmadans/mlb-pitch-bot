"""
Feature engineering from raw MLB API game JSON.
Computes Pitcher's Situational tendencies (global and by count).
"""

from collections import Counter
import pandas as pd


from src.api_extractors import _classify_pitch_family


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
