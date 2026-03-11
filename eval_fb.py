import pandas as pd
import joblib
import sys
from pathlib import Path
from collections import Counter

# Add src to path if needed (though running from root should work)
sys.path.append(str(Path(__file__).resolve().parent))

from src.constants import (
    DATABASE_PATH, SCALER_PATH, FEATURE_COLS_PATH
)
from src.database import query_all_pitches
from src import train_model

def mock_get_batter_features(df, use_api=False):
    # Ignore use_api completely, load from joblib
    try:
        return joblib.load("models/batter_features.joblib")
    except Exception:
        return pd.DataFrame()

# Patch the function
train_model.get_batter_features = mock_get_batter_features

def main():
    print("Loading data...")
    df = query_all_pitches()
    
    # Filter for 2 strikes
    df_2s = df[df["strikes"] == 2].copy()
    print(f"Loaded {len(df_2s)} pitches with 2 strikes.")
    
    if len(df_2s) == 0:
        return
        
    print("Preparing features...")
    # This will now use the mocked get_batter_features and load the cached file
    X, y_encoded, le, prev_le, _, p_le, b_le, mob_le, out_pitch_le, _, weights = train_model.prepare_target_and_features(df_2s, include_batter_stats=True)
    
    MID_MODEL_PATH = "models/pitch_classifier_mid.pkl"
    print(f"Loading MIDDLE model from {MID_MODEL_PATH}...")
    model = joblib.load(MID_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected_cols = joblib.load(FEATURE_COLS_PATH)
    
    print("Aligning columns...")
    # Add missing columns with 0
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
            
    # Keep only expected columns in correct order
    X = X[expected_cols]
    
    print("Scaling features...")
    X_scaled = scaler.transform(X)
    
    print("Predicting labels with MIDDLE MODEL...")
    preds = model.predict(X_scaled)
    pred_labels = le.inverse_transform(preds)
    
    # Getting actual labels just for comparison if we want
    actual_labels = le.inverse_transform(y_encoded)
    
    print(f"\n=======================")
    print(f"Total 2-strike pitches evaluated: {len(preds)}")
    
    print("\nPredicted Pitch Group Frequencies (2 Strikes) - MIDDLE MODEL:")
    counts = pd.Series(pred_labels).value_counts()
    for label, count in counts.items():
        pct = count / len(preds) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
        
    print("\nActual Pitch Group Frequencies (2 Strikes):")
    actual_counts = pd.Series(actual_labels).value_counts()
    for label, count in actual_counts.items():
        pct = count / len(preds) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    print(f"=======================\n")

if __name__ == "__main__":
    main()
