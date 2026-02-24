"""
Train XGBoost pitch-type classifier using data from SQLite database.
"""
import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.constants import (
    DATABASE_PATH, MODEL_PATH, ENCODER_PATH, PREV_ENCODER_PATH,
    P_HAND_ENCODER_PATH, B_SIDE_ENCODER_PATH, MOB_ENCODER_PATH,
    SCALER_PATH, FEATURE_COLS_PATH
)
from src.database import query_all_pitches
from src.dataset_generator import add_features
from src.batter_tendency_processing import get_batter_features


from src.features import _classify_pitch_family

def prepare_target_and_features(df: pd.DataFrame, include_batter_stats: bool = True):
    """
    Cleans and encodes raw pitch data for training.
    Predicts pitch families (Fastball, Breaking, Offspeed) instead of individual codes.
    """
    # Ensure features (tendencies) are calculated if not present
    if "tendency_total_pitches" not in df.columns:
        print("    Calculating situational tendencies for training...")
        df = add_features(df)

    # Drop rows without a pitch type
    df = df.dropna(subset=["pitch_type"]).copy()
    
    # Map pitch types to families
    print("    Mapping pitch types to families...")
    df["pitch_family"] = df["pitch_type"].apply(_classify_pitch_family)
    
    # Filter out rare 'Other' pitches to prevent stratification errors
    df = df[df["pitch_family"] != "Other"].copy()
    
    # We want to predict families
    y = df["pitch_family"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    count_cols = [c for c in df.columns if c.startswith("count_") and "-" not in c]
    numeric = ["inning", "balls", "strikes", "outs", "is_leverage", "score_home", "score_away"]
    tendency_cols = [
        c for c in df.columns
        if (c.startswith("tendency_global_") or c.startswith("tendency_count_")) and (c.endswith("_pct") or c == "tendency_total_pitches")
    ]
    feature_cols = count_cols + [c for c in numeric if c in df.columns] + tendency_cols

    # Previous pitch family encoding
    print("    Encoding categorical features...")
    
    # Handedness and Men On Base
    p_hand_enc = LabelEncoder()
    df["pitcher_hand_enc"] = p_hand_enc.fit_transform(df["pitcher_hand"].fillna("R"))
    
    b_side_enc = LabelEncoder()
    df["batter_side_enc"] = b_side_enc.fit_transform(df["batter_side"].fillna("R"))
    
    mob_enc = LabelEncoder()
    df["men_on_base_enc"] = mob_enc.fit_transform(df["men_on_base"].fillna("Empty"))
    
    df["prev_pitch_family"] = df["prev_pitch_type_in_ab"].apply(_classify_pitch_family)
    prev_encoder = LabelEncoder()
    df["prev_pitch_family_enc"] = prev_encoder.fit_transform(df["prev_pitch_family"])
    
    new_cats = ["pitcher_hand_enc", "batter_side_enc", "men_on_base_enc", "prev_pitch_family_enc"]
    feature_cols = new_cats + [c for c in feature_cols if c in df.columns]

    # Batter Tendencies Integration
    batter_df = pd.DataFrame()
    if include_batter_stats and "batter_id" in df.columns:
        print("    Calculating and merging batter tendencies...")
        batter_df = get_batter_features(df, use_api=True)
        df = df.merge(batter_df, on="batter_id", how="left")
        
        # Add batter columns to feature list
        batter_cols = [c for c in batter_df.columns if c != "batter_id"]
        feature_cols += batter_cols

    X = df[feature_cols].fillna(0)
    
    # Store Leverage weights (2 strikes or 3 balls)
    # Give 2.0x weight to high-advantage/disadvantage situations
    df["sample_weight"] = 1.0
    leveraged_mask = (df["strikes"] == 2) | (df["balls"] == 3)
    df.loc[leveraged_mask, "sample_weight"] = 2.0
    weights = df["sample_weight"]

    return X, y_encoded, label_encoder, prev_encoder, feature_cols, p_hand_enc, b_side_enc, mob_enc, batter_df, weights


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(DATABASE_PATH):
        print(f"Database not found at {DATABASE_PATH}. Run dataset_generator first.")
        return

    print(f"Loading data from database: {DATABASE_PATH}")
    df = query_all_pitches()
    
    if df.empty:
        print("Database is empty. Run dataset_generator first.")
        return
    print(f"Database contains {len(df)} total pitches.")
    
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
        max_date = df["game_date"].max()
        start_date = max_date - pd.Timedelta(days=60)
        print(f"Filtering dataset to 60-day temporal window: {start_date.date()} to {max_date.date()}")
        df = df[(df["game_date"] > start_date) & (df["game_date"] <= max_date)].copy()
    
    print(f"Training on {len(df)} pitches from recent 60 days.")

    X, y, le, prev_le, feature_cols, p_le, b_le, mob_le, batter_df, weights = prepare_target_and_features(df)
    print(f"Features: {len(feature_cols)} columns.")
    print(f"Classes: {le.classes_.tolist()}.")

    # Split with weights
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (Crucial for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train Logistic Regression with Hyperparameter Tuning
    print("Training Logistic Regression model with tuned hyperparameters...")
    model = LogisticRegression(
        max_iter=1000,
        C=0.1,  # Stronger regularization
        class_weight='balanced',  # Crucial for maximizing balanced_accuracy on non-Fastballs
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train_scaled, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test_scaled)
    acc = (y_pred == y_test).mean()
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\nOverall Test Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f} (Better for class imbalance)\n")
    
    print("Detailed Classification Report (Checking if we over-predict Fastballs):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nText-based Confusion Matrix (Rows=Actual, Cols=Predicted):")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    joblib.dump(prev_le, PREV_ENCODER_PATH)
    joblib.dump(p_le, P_HAND_ENCODER_PATH)
    joblib.dump(b_le, B_SIDE_ENCODER_PATH)
    joblib.dump(mob_le, MOB_ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    
    # Save batter features for inference
    if not batter_df.empty:
        joblib.dump(batter_df, root / "models" / "batter_features.joblib")
    
    print(f"Saved artifacts to {models_dir}")

    # Specific analysis for high-leverage situations
    X_test_lev = X_test[w_test > 1.0]
    y_test_lev = y_test[w_test > 1.0]
    if not X_test_lev.empty:
        X_test_lev_scaled = scaler.transform(X_test_lev)
        y_pred_lev = model.predict(X_test_lev_scaled)
        print("\n--- PERFORMANCE ON HIGH-LEVERAGE COUNTS (2 Strikes or 3 Balls) ---")
        print(classification_report(y_test_lev, y_pred_lev, target_names=le.classes_))


if __name__ == "__main__":
    main()
