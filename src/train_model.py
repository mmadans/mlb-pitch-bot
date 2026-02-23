"""
Train XGBoost pitch-type classifier using data from SQLite or CSV.
"""
import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb

from src.constants import (
    DATABASE_PATH, MODEL_PATH, ENCODER_PATH, PREV_ENCODER_PATH,
    P_HAND_ENCODER_PATH, B_SIDE_ENCODER_PATH, MOB_ENCODER_PATH, FEATURE_COLS_PATH
)
from src.database import query_all_pitches
from src.dataset_generator import add_features


from src.features import _classify_pitch_family

def prepare_target_and_features(df: pd.DataFrame):
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

    X = df[feature_cols].fillna(0)
    # Return encoders for inference (packaging them into a dict for now if needed, 
    # but the function signature currently returns them individually. I'll stick to the pattern but it's getting crowded).
    # Actually, I should probably return a dict or a structured object.
    # For now, I'll just return the updated list of values.
    return X, y_encoded, label_encoder, prev_encoder, feature_cols, p_hand_enc, b_side_enc, mob_enc


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
    
    print(f"Loaded {len(df)} pitches.")

    X, y, le, prev_le, feature_cols, p_le, b_le, mob_le = prepare_target_and_features(df)
    print(f"Features: {len(feature_cols)} columns.")
    print(f"Classes: {le.classes_.tolist()}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"\nOverall Test Accuracy: {acc:.4f}\n")
    
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    joblib.dump(prev_le, PREV_ENCODER_PATH)
    joblib.dump(p_le, P_HAND_ENCODER_PATH)
    joblib.dump(b_le, B_SIDE_ENCODER_PATH)
    joblib.dump(mob_le, MOB_ENCODER_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    
    print(f"Saved artifacts to {models_dir}")


if __name__ == "__main__":
    main()
