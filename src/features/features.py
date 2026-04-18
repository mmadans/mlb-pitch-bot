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
        prev_call = df["prev_pitch_call"].fillna("").astype(str).str.lower()
        df["prev_pitch_was_whiff"] = prev_call.str.contains("swinging strike", na=False).astype(int)
        df["prev_pitch_was_foul"] = prev_call.str.contains("foul", na=False).astype(int)
    else:
        df["prev_pitch_was_whiff"] = 0
        df["prev_pitch_was_foul"] = 0

    # Streak × count advantage interactions
    # count_adv is positive when batter-favorable (more balls than strikes)
    if all(c in df.columns for c in ["fastball_streak", "breaking_streak", "offspeed_streak"]):
        count_adv = df["balls"] - df["strikes"]
        df["fastball_streak_x_count_adv"] = df["fastball_streak"] * count_adv
        df["breaking_streak_x_count_adv"] = df["breaking_streak"] * count_adv
        df["offspeed_streak_x_count_adv"] = df["offspeed_streak"] * count_adv
    else:
        df["fastball_streak_x_count_adv"] = 0
        df["breaking_streak_x_count_adv"] = 0
        df["offspeed_streak_x_count_adv"] = 0

    # Drop helper columns used only for feature construction
    df = df.drop(columns=["count_str", "score_diff"], errors="ignore")

    return df


def _compute_tendencies(df: pd.DataFrame, group_cols: list, prefix: str) -> pd.DataFrame:
    """
    Shared helper: Laplace-smoothed pitch-family frequency for any grouping.

    Groups df by group_cols + pitch_family, computes smoothed percentages,
    renames columns to f"{prefix}_{family}_pct", adds a total-pitches column,
    and left-merges the result back onto df.
    """
    if df.empty:
        return df
    if 'pitch_family' not in df.columns:
        df['pitch_family'] = df['pitch_type'].apply(_classify_pitch_family)

    counts = df.groupby(group_cols + ["pitch_family"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    smoothed = (counts + 1.0).div(totals + counts.shape[1], axis=0)
    smoothed.columns = [f"{prefix}_{col}_pct" for col in smoothed.columns]
    smoothed[f"{prefix}_total_pitches"] = totals

    return df.merge(smoothed.reset_index(), on=group_cols, how="left")


def add_pitcher_count_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """Pitch-family frequencies per (pitcher, balls, strikes) with Laplace smoothing."""
    return _compute_tendencies(df, ["pitcher_id", "balls", "strikes"], "tendency_count")


def add_batter_count_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """Pitch-family frequencies per (batter, balls, strikes) — how pitchers attack this batter."""
    return _compute_tendencies(df, ["batter_id", "balls", "strikes"], "tendency_batter_count")


def add_league_count_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """League-wide pitch-family frequencies per count — baseline regardless of pitcher."""
    return _compute_tendencies(df, ["balls", "strikes"], "tendency_league_count")


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

    # Require at least 30 pitches of a family before trusting its whiff rate.
    # For pitchers with no qualifying family, fall back to their most-thrown family.
    MIN_OUT_PITCH_PITCHES = 30

    def _pick_out_pitch(group):
        qualifying = group[group["total_pitches"] >= MIN_OUT_PITCH_PITCHES]
        if not qualifying.empty:
            return qualifying.loc[qualifying["whiff_rate"].idxmax(), "pitch_family"]
        return group.loc[group["total_pitches"].idxmax(), "pitch_family"]

    pitcher_out = (
        out_pitch_stats.groupby("pitcher_id")
        .apply(_pick_out_pitch)
        .reset_index()
    )
    pitcher_out.columns = ["pitcher_id", "primary_out_pitch"]
    primary_out_pitches = pitcher_out
    
    df = df.merge(primary_out_pitches, on="pitcher_id", how="left")
    return df


def add_global_pitcher_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """Global (count-agnostic) pitch-family frequencies per pitcher with Laplace smoothing."""
    # Uses a non-standard total_pitches column name for historical compatibility
    if df.empty:
        return df
    if 'pitch_family' not in df.columns:
        df['pitch_family'] = df['pitch_type'].apply(_classify_pitch_family)

    counts = df.groupby(["pitcher_id", "pitch_family"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    smoothed = (counts + 1.0).div(totals + counts.shape[1], axis=0)
    smoothed.columns = [f"tendency_global_{col}_pct" for col in smoothed.columns]
    smoothed["tendency_total_pitches"] = totals  # note: no prefix — consumed by live_game_tracker

    return df.merge(smoothed.reset_index(), on="pitcher_id", how="left")
