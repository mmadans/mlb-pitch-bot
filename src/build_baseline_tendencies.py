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
    p.add_argument("--csv", type=str, default=None, help="Path to historical CSV. If None, uses DB or finds latest in data/")
    p.add_argument("--no-db", action="store_false", dest="use_db", help="Do not use the SQLite database as source")
    p.set_defaults(use_db=True)
    args = p.parse_args()
    
    df = None
    if args.use_db and os.path.exists(DATABASE_PATH):
        print(f"Loading data from database: {DATABASE_PATH}")
        df = query_all_pitches()
    
    if df is None or df.empty:
        csv_file = args.csv
        if not csv_file:
            files = glob.glob("data/raw_pitches_*.csv") or glob.glob("data/pitch_features_*.csv")
            if not files:
                print("No source found (DB or CSV). Please run dataset_generator first.")
                exit(1)
            csv_file = max(files)
        print(f"Loading data from CSV: {csv_file}")
        df = pd.read_csv(csv_file)
        
    build_baseline(df)
