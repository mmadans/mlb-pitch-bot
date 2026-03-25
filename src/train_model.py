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
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from src.constants import (
    DATABASE_PATH, MODEL_PATH, ENCODER_PATH, PREV_ENCODER_PATH,
    P_HAND_ENCODER_PATH, B_SIDE_ENCODER_PATH, MOB_ENCODER_PATH,
    OUT_PITCH_ENCODER_PATH, PARK_ENCODER_PATH, FEATURE_COLS_PATH,
    PREV_CALL_ENCODER_PATH
)
from src.database import query_all_pitches
from src.dataset_generator import add_features
from src.batter_tendency_processing import get_batter_features



from src.features import _classify_pitch_family, add_contextual_features
from src.build_baseline_tendencies import build_baseline

def apply_baseline_to_df(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    '''Applies baseline tendency dictionaries to a DataFrame securely (no leakage).'''
    df = df.copy()
    
    # Global
    df_global = pd.DataFrame.from_dict(baseline['global'], orient='index')
    df_global.index.name = 'pitcher'
    df = df.merge(df_global, on='pitcher', how='left')
    
    # Count
    df_count = pd.DataFrame.from_dict(baseline['count'], orient='index')
    if not df_count.empty:
        df_count.index.names = ['pitcher', 'balls', 'strikes']
        df = df.merge(df_count, on=['pitcher', 'balls', 'strikes'], how='left')
    
    # Batter Count
    df_bcount = pd.DataFrame.from_dict(baseline['batter_count'], orient='index')
    if not df_bcount.empty:
        df_bcount.index.names = ['batter_id', 'balls', 'strikes']
        df = df.merge(df_bcount, on=['batter_id', 'balls', 'strikes'], how='left')
    
    # League count
    df_lcount = pd.DataFrame.from_dict(baseline['league_count'], orient='index')
    if not df_lcount.empty:
        df_lcount.index.names = ['balls', 'strikes']
        df = df.merge(df_lcount, on=['balls', 'strikes'], how='left')
    
    # out pitch
    out_pitch_s = pd.Series(baseline['out_pitch'], name='primary_out_pitch')
    df = df.merge(out_pitch_s, left_on='pitcher_id', right_index=True, how='left')
    df["primary_out_pitch"] = df["primary_out_pitch"].fillna("Fastball")
    
    # Fill other missing tendencies with 0
    tend_cols = [c for c in df.columns if c.startswith('tendency_')]
    df[tend_cols] = df[tend_cols].fillna(0.0)
    
    # Add contextual features
    df = add_contextual_features(df)
    
    # Add platoon advantage (Pitcher hand == Batter side, AND Batter side != 'S')
    df['is_platoon_advantage'] = ((df['pitcher_hand'] == df['batter_side']) & (df['batter_side'] != 'S')).astype(int)
    
    return df


def prepare_target_and_features(df: pd.DataFrame, include_batter_stats: bool = True):
    """
    Cleans and encodes raw pitch data for training.
    Predicts pitch families (Fastball, Breaking, Offspeed) instead of individual codes.
    """


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
    numeric = ["balls", "strikes", "outs", "is_leverage", "is_platoon_advantage", "run_differential"]
    tendency_cols = [
        c for c in df.columns
        if (c.startswith("tendency_global_") or c.startswith("tendency_count_") or c.startswith("tendency_batter_count_") or c.startswith("tendency_league_count_")) and (c.endswith("_pct") or c == "tendency_total_pitches")
    ]
    feature_cols = count_cols + [c for c in numeric if c in df.columns] + tendency_cols

    # Previous pitch family encoding
    print("    Encoding categorical features...")
    
    # Handedness, Men On Base, and Out Pitch
    p_hand_enc = LabelEncoder()
    df["pitcher_hand_enc"] = p_hand_enc.fit_transform(df["pitcher_hand"].fillna("R"))
    
    b_side_enc = LabelEncoder()
    df["batter_side_enc"] = b_side_enc.fit_transform(df["batter_side"].fillna("R"))
    
    mob_enc = LabelEncoder()
    df["men_on_base_enc"] = mob_enc.fit_transform(df["men_on_base"].fillna("Empty"))
    
    out_pitch_enc = LabelEncoder()
    df["primary_out_pitch_enc"] = out_pitch_enc.fit_transform(df["primary_out_pitch"].fillna("Fastball"))

    park_enc = LabelEncoder()
    df["park_id_enc"] = park_enc.fit_transform(df["park_id"].fillna(0).astype(str))

    df["prev_pitch_family"] = df["prev_pitch_type_in_ab"].apply(_classify_pitch_family)
    prev_encoder = LabelEncoder()
    df["prev_pitch_family_enc"] = prev_encoder.fit_transform(df["prev_pitch_family"])
    
    call_encoder = LabelEncoder()
    df["prev_pitch_call_enc"] = call_encoder.fit_transform(df["prev_pitch_call"].fillna("None"))
    
    new_cats = ["pitcher_hand_enc", "batter_side_enc", "men_on_base_enc", "prev_pitch_family_enc", "primary_out_pitch_enc", "park_id_enc", "prev_pitch_call_enc"]
    feature_cols = new_cats + [c for c in feature_cols if c in df.columns]

    # Batter Tendencies Integration
    batter_df = pd.DataFrame()
    if include_batter_stats and "batter_id" in df.columns:
        print("    Calculating and merging batter tendencies...")
        batter_df = get_batter_features(df, use_api=True)
        df = df.merge(batter_df, on="batter_id", how="left")
        
        # Add batter columns to feature list (Filtering out low-importance stats)
        exclude_stats = ["chase_rate", "whiff_rate", "k_pct"]
        batter_cols = [c for c in batter_df.columns if c != "batter_id" and c not in exclude_stats]
        feature_cols += batter_cols

    X = df[feature_cols].fillna(0)
    
    # RE-ENCODE y here to ensure it perfectly matches the final df/X length
    # (The merge might have changed row count or alignment if not 1:1)
    y_final = label_encoder.transform(df["pitch_family"])
    
    # Store Leverage weights (2 strikes or 3 balls)
    # Give 2.0x weight to high-advantage/disadvantage situations
    df["sample_weight"] = 1.0
    leveraged_mask = (df["strikes"] == 2) | (df["balls"] == 3)
    df.loc[leveraged_mask, "sample_weight"] = 2.0
    weights = df["sample_weight"]

    return X, y_final, label_encoder, prev_encoder, call_encoder, feature_cols, p_hand_enc, b_side_enc, mob_enc, out_pitch_enc, park_enc, batter_df, weights


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
    
    print(f"Dataset contains {len(df)} pitches. Splitting chronologically to prevent data leakage...")
    df = df.sort_values(by=["game_date", "at_bat_index", "pitch_index"]).reset_index(drop=True)
    
    # 80/20 Chronological Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train on {len(train_df)} pitches, Test on {len(test_df)}")
    
    baseline_path = str(root / "models/baseline_tendencies.pkl")
    print("Building baseline tendencies from Training Set only...")
    build_baseline(train_df, output_path=baseline_path)
    baseline = joblib.load(baseline_path)
    
    print("Applying baseline tendencies securely to train and test sets...")
    train_df = apply_baseline_to_df(train_df, baseline)
    test_df = apply_baseline_to_df(test_df, baseline)
    
    # Combine just for categorical label encoding consistency
    # We will manually split them back using their lengths
    df_combined = pd.concat([train_df, test_df], ignore_index=True)
    
    X, y, le, prev_le, call_le, feature_cols, p_le, b_le, mob_le, out_pitch_le, park_le, batter_df, weights = prepare_target_and_features(df_combined)
    
    # Since we dropped rows in prepare_target_and_features (e.g. 'Other' pitch family),
    # we can't just use split_idx. We need a mask or array slicing based on actual train_df rows.
    # Fortunately, train_test_split natively supports shuffle=False for sequential splitting!
    # By passing the processed X, y, weights to it, we avoid any shape mismatches.
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, shuffle=False
    )

    print(f"Features: {len(feature_cols)} columns.")
    print(f"Classes: {le.classes_.tolist()}.")

    # Initialize and train Calibrated XGBoost
    print("Training XGBoost Classifier model with Isotonic Calibration...")
    base_model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1
    )
    
    model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=3)
    # XGBoost handles internal scaling natively, so no scaler needed
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    acc = (y_pred == y_test).mean()
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import label_binarize
    import numpy as np
    y_test_bin = label_binarize(y_test, classes=model.classes_)
    rmse = np.sqrt(mean_squared_error(y_test_bin, y_prob))
    brier = mean_squared_error(y_test_bin, y_prob)

    print(f"\nOverall Test Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f} (Better for class imbalance)")
    print(f"Probability RMSE: {rmse:.4f}")
    print(f"Brier Score (MSE): {brier:.4f}\n")
    
    print("Detailed Classification Report (Checking if we over-predict Fastballs):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nText-based Confusion Matrix (Rows=Actual, Cols=Predicted):")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    joblib.dump(prev_le, PREV_ENCODER_PATH)
    joblib.dump(call_le, PREV_CALL_ENCODER_PATH)
    joblib.dump(p_le, P_HAND_ENCODER_PATH)
    joblib.dump(b_le, B_SIDE_ENCODER_PATH)
    joblib.dump(mob_le, MOB_ENCODER_PATH)
    joblib.dump(out_pitch_le, OUT_PITCH_ENCODER_PATH)
    joblib.dump(park_le, PARK_ENCODER_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    
    # Save batter features for inference
    if not batter_df.empty:
        joblib.dump(batter_df, root / "models" / "batter_features.joblib")
    
    print(f"Saved artifacts to {models_dir}")

    # Specific analysis for high-leverage situations
    X_test_lev = X_test[w_test > 1.0]
    y_test_lev = y_test[w_test > 1.0]
    if not X_test_lev.empty:
        y_pred_lev = model.predict(X_test_lev)
        print("\n--- PERFORMANCE ON HIGH-LEVERAGE COUNTS (2 Strikes or 3 Balls) ---")
        print(classification_report(y_test_lev, y_pred_lev, target_names=le.classes_))


if __name__ == "__main__":
    main()
