"""
Train Alternative Pitch Classifier without class-weight balancing
to see if it predicts fastballs closer to the true baseline rate.
"""
import os
import joblib
import pandas as pd
import sys
from pathlib import Path

# Add src to path if needed (though running from root should work)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.constants import (
    DATABASE_PATH, SCALER_PATH, FEATURE_COLS_PATH
)
from src.database import query_all_pitches
from src import train_model

def mock_get_batter_features(df, use_api=False):
    # Ignore use_api completely, load from joblib
    try:
        return joblib.load(Path(__file__).resolve().parent.parent / "models/batter_features.joblib")
    except Exception:
        return pd.DataFrame()

# Patch the function
train_model.get_batter_features = mock_get_batter_features

def main() -> None:
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    ALT_MODEL_PATH = str(models_dir / 'pitch_classifier_alt.pkl')

    if not os.path.exists(DATABASE_PATH):
        print(f"Database not found at {DATABASE_PATH}. Run dataset_generator first.")
        return

    print(f"Loading data from database: {DATABASE_PATH}")
    df = query_all_pitches()
    
    if df.empty:
        print("Database is empty. Run dataset_generator first.")
        return
        
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
        max_date = df["game_date"].max()
        start_date = max_date - pd.Timedelta(days=60)
        print(f"Filtering dataset to 60-day temporal window: {start_date.date()} to {max_date.date()}")
        df = df[(df["game_date"] > start_date) & (df["game_date"] <= max_date)].copy()
    
    print(f"Training on {len(df)} pitches from recent 60 days.")

    X, y, le, prev_le, feature_cols, p_le, b_le, mob_le, out_pitch_le, batter_df, weights = train_model.prepare_target_and_features(df, include_batter_stats=True) # CHANGED TO TRUE to use the mocked fn properly
    
    # Fill any missing columns
    X = X.fillna(0)

    # Split with weights
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train Logistic Regression WITHOUT class_weight='balanced'
    print("Training alternative Logistic Regression model (Unbalanced)...")
    model = LogisticRegression(
        max_iter=1000,
        C=0.1,  
        class_weight=None,  # CHANGED: Allow natural class frequencies
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train_scaled, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test_scaled)
    acc = (y_pred == y_test).mean()
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\nOverall Test Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}\n")
    
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nText-based Confusion Matrix (Rows=Actual, Cols=Predicted):")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df)

    joblib.dump(model, ALT_MODEL_PATH)
    print(f"Saved alternative model to {ALT_MODEL_PATH}")

if __name__ == "__main__":
    main()
