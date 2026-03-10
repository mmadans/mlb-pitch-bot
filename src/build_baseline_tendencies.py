import pandas as pd
import joblib
import os
from src.features import add_global_pitcher_tendencies, add_pitcher_count_tendencies
from src.constants import BASELINE_PATH, DATABASE_PATH
from src.database import query_all_pitches

def build_baseline(df: pd.DataFrame, output_path: str = BASELINE_PATH):
    """
    Builds a baseline tendency mapping from a DataFrame.
    """
    print("Calculating tendencies from data...")
    # These functions add columns to the DF. We want to extract just the mapping.
    df_global = add_global_pitcher_tendencies(df)
    df_full = add_pitcher_count_tendencies(df_global)
    
    # Extract unique mappings
    # Global: pitcher -> {tendency_global_...}
    global_cols = [c for c in df_full.columns if c.startswith("tendency_global_") or c == "tendency_total_pitches"]
    baseline_global = df_full[["pitcher"] + global_cols].groupby("pitcher").first().to_dict("index")
    
    # Count: (pitcher, balls, strikes) -> {tendency_count_...}
    count_cols = [c for c in df_full.columns if c.startswith("tendency_count_")]
    baseline_count = df_full[["pitcher", "balls", "strikes"] + count_cols].groupby(["pitcher", "balls", "strikes"]).first().to_dict("index")

    # Batter Count: (batter_id, balls, strikes) -> {tendency_batter_count_...}
    batter_count_cols = [c for c in df_full.columns if c.startswith("tendency_batter_count_")]
    baseline_batter_count = df_full[["batter_id", "balls", "strikes"] + batter_count_cols].dropna(subset=["batter_id"]).groupby(["batter_id", "balls", "strikes"]).first().to_dict("index")
    
    baseline = {
        "global": baseline_global,
        "count": baseline_count,
        "batter_count": baseline_batter_count,
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
