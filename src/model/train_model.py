"""
Train XGBoost pitch-type classifier using data from SQLite database.
"""
import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from dotenv import load_dotenv
from src.constants import (
    DATABASE_PATH, MODEL_PATH, TARGET_ENCODER_PATH, CATEGORICAL_ENCODER_PATH, FEATURE_COLS_PATH
)
from src.database import query_all_pitches
from src.dataset_generator import add_features
from src.batter_tendency_processing import get_batter_features

load_dotenv()



from src.api_extractors import _classify_pitch_family
from src.baseline_manager import apply_baseline_to_df
from src.build_baseline_tendencies import build_baseline


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

    # Exclude all redundant count-encoding features:
    # - Count one-hots: were 71% of importance, drowning pitcher-specific signals
    # - Raw balls/strikes: create tree splits that override pitcher tendency features
    # - League count tendencies: redundant with pitcher count tendency; their presence
    #   causes the model to learn "at 2 strikes, league throws little offspeed" and
    #   ignore that individual pitchers deviate (e.g. Matz at 22% OS vs league 12%)
    # - Delta features: also become redundant when league features are removed
    #
    # The pitcher's count tendency (tendency_count_*_pct) already encodes BOTH the
    # current count AND the pitcher's identity. The model should use it as the primary
    # signal and only adjust based on situational context (leverage, matchup, streaks).
    numeric = [
        "outs", "is_leverage", "is_platoon_advantage", "run_differential",
        "breaking_streak", "fastball_streak", "offspeed_streak",
        "fastball_streak_x_count_adv", "breaking_streak_x_count_adv", "offspeed_streak_x_count_adv",
        "pitch_count_in_game", "times_faced_today", "prev_pX", "prev_pZ",
        "is_double_play_scenario", "prev_pitch_was_whiff", "prev_pitch_was_foul"
    ]
    tendency_cols = [
        c for c in df.columns
        if (c.startswith("tendency_global_") or c.startswith("tendency_count_") or c.startswith("tendency_batter_count_")) and
           (c.endswith("_pct") or c.endswith("_Fastball") or c.endswith("_Breaking") or c.endswith("_Offspeed") or c == "tendency_total_pitches")
    ]
    feature_cols = [c for c in numeric if c in df.columns] + tendency_cols


    print("    Encoding categorical features...")
    cat_cols = ["pitcher_hand", "batter_side", "men_on_base", "primary_out_pitch", "park_id", "prev_pitch_family", "prev_pitch_call"]
    
    df["park_id"] = df["park_id"].fillna(0).astype(str)
    df["prev_pitch_family"] = df["prev_pitch_type_in_ab"].apply(_classify_pitch_family)
    
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)
        
    categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoded_vals = categorical_encoder.fit_transform(df[cat_cols])
    
    new_cats = []
    for i, col in enumerate(cat_cols):
        enc_col = f"{col}_enc"
        df[enc_col] = encoded_vals[:, i]
        new_cats.append(enc_col)
        
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
    
    # Leverage weights: 2x for high-stakes counts (2 strikes or 3 balls)
    df["sample_weight"] = 1.0
    leveraged_mask = (df["strikes"] == 2) | (df["balls"] == 3)
    df.loc[leveraged_mask, "sample_weight"] = 2.0

    # Class-balance weights: multiply by inverse class frequency so that
    # minority classes (Offspeed ~14%, Breaking ~31%) get proportionally
    # more weight relative to the Fastball majority (~55%).
    # Formula: total / (n_classes * class_count) — same as sklearn "balanced".
    n_classes = df["pitch_family"].nunique()
    class_counts = df["pitch_family"].value_counts()
    for family, count in class_counts.items():
        balance_w = len(df) / (n_classes * count)
        df.loc[df["pitch_family"] == family, "sample_weight"] *= balance_w

    weights = df["sample_weight"]

    return X, y_final, label_encoder, categorical_encoder, feature_cols, batter_df, weights


def main(tune: bool = False) -> None:
    import wandb

    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    try:
        wandb_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "mlb-pitch-bot"),
            job_type="training",
            config={
                "n_estimators": 150,
                "max_depth": 5,
                "learning_rate": 0.1,
                "objective": "multi:softprob",
                "tune": tune,
                "class_balancing": "inverse_frequency",
                "leverage_weight": 2.0,
                "train_test_split": "chronological_80_20",
            },
        )
    except Exception as e:
        print(f"W&B init failed (training will continue): {e}")

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
        min_date = df["game_date"].min()
        db_span_days = (max_date - min_date).days

        if db_span_days > 60:
            # Use all data in the DB — caller is responsible for loading the desired window
            print(f"Using full database range: {min_date.date()} to {max_date.date()} ({db_span_days} days)")
        else:
            # DB is a narrow window (e.g. live season top-up); apply 60-day cap
            start_date = max_date - pd.Timedelta(days=60)
            print(f"Filtering dataset to 60-day temporal window: {start_date.date()} to {max_date.date()}")
            df = df[(df["game_date"] > start_date) & (df["game_date"] <= max_date)].copy()
    
    # Filter for Regular and Postseason ONLY (Ignore Spring Training 'S' and All-Star 'A')
    if "game_type" in df.columns:
        valid_types = ["R", "P", "W"]
        before_count = len(df)
        df = df[df["game_type"].isin(valid_types)].copy()
        if len(df) < before_count:
            print(f"    Filtered out {before_count - len(df)} spring training/exhibition pitches.")
    
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
    
    # We must ensure pitch_family is available for LOO encoding
    if "pitch_family" not in train_df.columns:
        train_df["pitch_family"] = train_df["pitch_type"].apply(_classify_pitch_family)
        
    train_df = apply_baseline_to_df(train_df, baseline, is_train=True)
    test_df = apply_baseline_to_df(test_df, baseline, is_train=False)
    
    # Combine just for categorical label encoding consistency
    # We will manually split them back using their lengths
    df_combined = pd.concat([train_df, test_df], ignore_index=True)
    
    X, y, le, cat_enc, feature_cols, batter_df, weights = prepare_target_and_features(df_combined)
    
    # Since we dropped rows in prepare_target_and_features (e.g. 'Other' pitch family),
    # we can't just use split_idx. We need a mask or array slicing based on actual train_df rows.
    # Fortunately, train_test_split natively supports shuffle=False for sequential splitting!
    # By passing the processed X, y, weights to it, we avoid any shape mismatches.
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, shuffle=False
    )

    print(f"Features: {len(feature_cols)} columns.")
    print(f"Classes: {le.classes_.tolist()}.")

    # Initialize and train XGBoost
    print("Training XGBoost Classifier model...")
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

    if tune:
        print("Running GridSearchCV for hyperparameter tuning...")
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='neg_log_loss', n_jobs=-1)
        grid_search.fit(X_train, y_train, sample_weight=w_train)
        print(f"Best parameters found: {grid_search.best_params_}")
        best_estimator = grid_search.best_estimator_
    else:
        best_estimator = base_model

    print("Fitting model...")
    model = best_estimator
    if not tune:
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
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nText-based Confusion Matrix (Rows=Actual, Cols=Predicted):")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df)

    if wandb_run is not None:
        try:
            wandb_metrics = {
                "accuracy":          acc,
                "balanced_accuracy": bal_acc,
                "brier_score":       brier,
                "prob_rmse":         rmse,
                "train_pitches":     len(X_train),
                "test_pitches":      len(X_test),
                "n_features":        len(feature_cols),
            }
            for cls in le.classes_:
                cls_key = cls.lower()
                wandb_metrics[f"precision_{cls_key}"] = report_dict[cls]["precision"]
                wandb_metrics[f"recall_{cls_key}"]    = report_dict[cls]["recall"]
                wandb_metrics[f"f1_{cls_key}"]        = report_dict[cls]["f1-score"]
            if tune and hasattr(best_estimator, "best_params_"):
                wandb_run.config.update(grid_search.best_params_)
            wandb_run.log(wandb_metrics)
            wandb_run.log({"confusion_matrix": wandb.Table(dataframe=cm_df.reset_index().rename(columns={"index": "actual"}))})
        except Exception as e:
            print(f"W&B metric logging failed: {e}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, TARGET_ENCODER_PATH)
    joblib.dump(cat_enc, CATEGORICAL_ENCODER_PATH)
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
        lev_report_dict = classification_report(y_test_lev, y_pred_lev, target_names=le.classes_, output_dict=True)
        print(classification_report(y_test_lev, y_pred_lev, target_names=le.classes_))
        if wandb_run is not None:
            try:
                for cls in le.classes_:
                    cls_key = cls.lower()
                    wandb_run.log({
                        f"leverage_recall_{cls_key}":    lev_report_dict[cls]["recall"],
                        f"leverage_precision_{cls_key}": lev_report_dict[cls]["precision"],
                    })
            except Exception as e:
                print(f"W&B leverage logging failed: {e}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning grid search")
    args = parser.parse_args()
    main(tune=args.tune)
