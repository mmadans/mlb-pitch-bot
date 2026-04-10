import pandas as pd
import joblib
import os
from src.features import (
    add_global_pitcher_tendencies, 
    add_pitcher_count_tendencies,
    add_batter_count_tendencies,
    add_league_count_tendencies,
    add_pitcher_out_pitch
)
from src.constants import BASELINE_PATH, DATABASE_PATH
from src.database import query_all_pitches

def build_baseline(df: pd.DataFrame, output_path: str = BASELINE_PATH):
    """
    Builds a baseline tendency mapping from a DataFrame.
    """
    print("Calculating tendencies from data...")
    # These functions add columns to the DF. We want to extract just the mapping.
    df = add_global_pitcher_tendencies(df)
    df = add_pitcher_count_tendencies(df)
    df = add_batter_count_tendencies(df)
    df = add_league_count_tendencies(df)
    df_full = add_pitcher_out_pitch(df)
    
    # Extract unique mappings
    # Global: pitcher_id -> {tendency_global_...}
    global_cols = [c for c in df_full.columns if c.startswith("tendency_global_") or c == "tendency_total_pitches"]
    baseline_global = df_full[["pitcher_id"] + global_cols].groupby("pitcher_id").first().to_dict("index")
    
    # Count: (pitcher_id, balls, strikes) -> {tendency_count_...}
    count_cols = [c for c in df_full.columns if c.startswith("tendency_count_")]
    baseline_count = df_full[["pitcher_id", "balls", "strikes"] + count_cols].groupby(["pitcher_id", "balls", "strikes"]).first().to_dict("index")

    # Batter Count: (batter_id, balls, strikes) -> {tendency_batter_count_...}
    batter_count_cols = [c for c in df_full.columns if c.startswith("tendency_batter_count_")]
    baseline_batter_count = df_full[["batter_id", "balls", "strikes"] + batter_count_cols].dropna(subset=["batter_id"]).groupby(["batter_id", "balls", "strikes"]).first().to_dict("index")
    
    # League Count: (balls, strikes) -> {tendency_league_count_...}
    league_count_cols = [c for c in df_full.columns if c.startswith("tendency_league_count_")]
    baseline_league_count = df_full[["balls", "strikes"] + league_count_cols].groupby(["balls", "strikes"]).first().to_dict("index")

    # Out Pitch: pitcher_id -> primary_out_pitch
    baseline_out_pitch = df_full[["pitcher_id", "primary_out_pitch"]].dropna().groupby("pitcher_id").first()["primary_out_pitch"].to_dict()

    baseline = {
        "global": baseline_global,
        "count": baseline_count,
        "batter_count": baseline_batter_count,
        "league_count": baseline_league_count,
        "out_pitch": baseline_out_pitch,
        "feature_cols": list(df_full.columns) # To know what columns to expect
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(baseline, output_path)
    print(f"Baseline saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=str, default=DATABASE_PATH, help="Path to SQLite database")
    args = p.parse_args()
    
    if not os.path.exists(args.db):
        print(f"Database not found at {args.db}. Run dataset_generator first.")
        exit(1)

    print(f"Loading data from database: {args.db}")
    df = query_all_pitches()
    
    if df.empty:
        print("Database is empty. Run dataset_generator first.")
        exit(1)
        
    build_baseline(df)
