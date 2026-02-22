"""
Train XGBoost pitch-type classifier on data/ CSV and save model + encoder.
Run with: uv run python -m src.train_model
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb


def get_data_path() -> Path:
    """Return path to most recent pitch_features_*.csv in data/."""
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"No data directory: {data_dir}. Run dataset_generator first.")
    csvs = list(data_dir.glob("pitch_features_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No pitch_features_*.csv in {data_dir}. Run dataset_generator first.")
    return max(csvs, key=lambda p: p.stat().st_mtime)


def prepare_target_and_features(df: pd.DataFrame):
    """Mirror notebook: prepare y (pitch_type) and X with same feature columns."""
    df = df.dropna(subset=["pitch_type"]).copy()
    df["pitch_type"] = df["pitch_type"].astype(str).str.upper()

    MIN_SAMPLES = 20
    counts = df["pitch_type"].value_counts()
    rare = counts[counts < MIN_SAMPLES].index.tolist()
    if rare:
        df.loc[df["pitch_type"].isin(rare), "pitch_type"] = "Other"

    y = df["pitch_type"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    count_cols = [c for c in df.columns if c.startswith("count_") and "-" not in c]
    numeric = ["inning", "balls", "strikes", "outs", "is_leverage", "score_home", "score_away"]
    tendency_cols = [
        c for c in df.columns
        if (c.startswith("tendency_global_") or c.startswith("tendency_count_")) and (c.endswith("_pct") or c == "tendency_total_pitches")
    ]
    feature_cols = count_cols + [c for c in numeric if c in df.columns] + tendency_cols

    prev = df["prev_pitch_type_in_ab"].fillna("None").astype(str).str.upper()
    prev_encoder = LabelEncoder()
    df["prev_pitch_type_in_ab_enc"] = prev_encoder.fit_transform(prev)
    feature_cols = ["prev_pitch_type_in_ab_enc"] + [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    return X, y_encoded, label_encoder, prev_encoder, feature_cols


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    path = get_data_path()
    print(f"Loading {path.name}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} pitches.")

    X, y, label_encoder, prev_encoder, feature_cols = prepare_target_and_features(df)
    print(f"Features: {len(feature_cols)} columns.")
    print(f"Classes: {label_encoder.classes_.tolist()}.")

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
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    model_path = models_dir / "pitch_classifier.pkl"
    encoder_path = models_dir / "encoder.pkl"
    prev_encoder_path = models_dir / "prev_pitch_encoder.pkl"
    feature_cols_path = models_dir / "feature_cols.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(prev_encoder, prev_encoder_path)
    joblib.dump(feature_cols, feature_cols_path)
    
    print(f"Saved model to {model_path}")
    print(f"Saved encoder to {encoder_path}")
    print(f"Saved prev pitch encoder to {prev_encoder_path}")
    print(f"Saved feature columns to {feature_cols_path}")


if __name__ == "__main__":
    main()
