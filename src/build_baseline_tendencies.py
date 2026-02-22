import pandas as pd
import joblib
import os
from src.features import add_global_pitcher_tendencies, add_pitcher_count_tendencies
from src.constants import BASELINE_PATH

def build_baseline(csv_path: str, output_path: str = BASELINE_PATH):
    """
    Reads a historical CSV and builds a baseline tendency mapping.
    """
    print(f"Reading historical data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Calculating tendencies...")
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
    
    baseline = {
        "global": baseline_global,
        "count": baseline_count,
        "feature_cols": list(df_full.columns) # To know what columns to expect
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(baseline, output_path)
    print(f"Baseline saved to {output_path}")

if __name__ == "__main__":
    import argparse
    import glob
    
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=None, help="Path to historical CSV. If None, finds latest in data/")
    args = p.parse_args()
    
    csv_file = args.csv
    if not csv_file:
        files = glob.glob("data/pitch_features_*.csv")
        if not files:
            print("No CSV found in data/. Please run dataset_generator first.")
            exit(1)
        csv_file = max(files)
        
    build_baseline(csv_file)
