import pandas as pd
import joblib
import numpy as np

from src.constants import (
    MODEL_PATH, TARGET_ENCODER_PATH, CATEGORICAL_ENCODER_PATH, FEATURE_COLS_PATH,  
    BATTER_FEATURES_PATH
)

from src.api_extractors import _classify_pitch_family

class PitchPredictor:
    """
    Wrapper for the pitch classification model to handle preprocessing and prediction.
    """
    def __init__(self, model_path=MODEL_PATH, target_encoder_path=TARGET_ENCODER_PATH, 
                 categorical_encoder_path=CATEGORICAL_ENCODER_PATH, feature_cols_path=FEATURE_COLS_PATH):
        """Loads the model and associated encoders from disk."""
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(target_encoder_path)
        self.categorical_encoder = joblib.load(categorical_encoder_path)
        self.feature_cols = joblib.load(feature_cols_path)
        try:
            self.batter_features = joblib.load(BATTER_FEATURES_PATH)
        except Exception:
            self.batter_features = pd.DataFrame()

    def predict_probabilities(self, features_df):
        """
        Prepares features and calculates probabilities for each pitch family.
        
        Args:
            features_df: A DataFrame containing a single row of pitch context.
        Returns:
            A dictionary mapping pitch families to their predicted probabilities.
        """
        # categorical encoding
        cat_cols = ["pitcher_hand", "batter_side", "men_on_base", "primary_out_pitch", "park_id", "prev_pitch_family", "prev_pitch_call"]
        
        # Hydrate synthetic columns derived from primary string columns
        if 'prev_pitch_type_in_ab' in features_df.columns:
            features_df['prev_pitch_family'] = features_df['prev_pitch_type_in_ab'].apply(_classify_pitch_family)
            
        for col in cat_cols:
            if col not in features_df.columns:
                features_df[col] = "Unknown"
            features_df[col] = features_df[col].fillna("Unknown").astype(str)
            
        encoded_vals = self.categorical_encoder.transform(features_df[cat_cols])
        
        for i, col in enumerate(cat_cols):
            features_df[f"{col}_enc"] = encoded_vals[:, i]

        # Batter lookup
        if not self.batter_features.empty and 'batter_id' in features_df.columns:
            bid = features_df['batter_id'].iloc[0]
            # Find the row in batter_features for this ID
            b_stats = self.batter_features[self.batter_features['batter_id'] == bid]
            if not b_stats.empty:
                for col in b_stats.columns:
                    if col != 'batter_id':
                        features_df[col] = b_stats[col].iloc[0]

        # Add platoon advantage (switch hitters always bat from the advantaged side)
        if 'pitcher_hand' in features_df.columns and 'batter_side' in features_df.columns:
            same_hand = (features_df['pitcher_hand'] == features_df['batter_side']) & (features_df['batter_side'] != 'S')
            switch_hitter = features_df['batter_side'] == 'S'
            features_df['is_platoon_advantage'] = (same_hand | switch_hitter).astype(int)

        # Streak × count advantage interaction features
        if all(c in features_df.columns for c in ["fastball_streak", "breaking_streak", "offspeed_streak", "balls", "strikes"]):
            count_adv = features_df["balls"].fillna(0).astype(int) - features_df["strikes"].fillna(0).astype(int)
            features_df["fastball_streak_x_count_adv"] = features_df["fastball_streak"] * count_adv
            features_df["breaking_streak_x_count_adv"] = features_df["breaking_streak"] * count_adv
            features_df["offspeed_streak_x_count_adv"] = features_df["offspeed_streak"] * count_adv

        # Delta tendency features: pitcher vs league average at this count
        for fam in ["Fastball", "Breaking", "Offspeed"]:
            count_col = f"tendency_count_{fam}_pct"
            league_col = f"tendency_league_count_{fam}_pct"
            global_col = f"tendency_global_{fam}_pct"
            if count_col in features_df.columns and league_col in features_df.columns:
                features_df[f"delta_count_{fam}"] = features_df[count_col] - features_df[league_col]
            if global_col in features_df.columns and league_col in features_df.columns:
                features_df[f"delta_global_{fam}"] = features_df[global_col] - features_df[league_col]

        # Ensure all columns the model expects are present
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0

        model_input = features_df[self.feature_cols]
        
        # Predict probabilities
        raw_probs = self.model.predict_proba(model_input)[0]
        
        # Floor each class at 1% — no pitch family should ever be essentially impossible.
        # Sub-1% outputs almost always indicate a feature hydration problem, not a real prediction.
        # Iterative floor: repeat until all classes satisfy the floor after renormalization.
        MIN_PROB = 0.01
        probs = raw_probs / raw_probs.sum()
        for _ in range(5):  # converges in 1-2 iterations in practice
            if probs.min() >= MIN_PROB:
                break
            probs = np.maximum(probs, MIN_PROB)
            probs = probs / probs.sum()
        
        # Map probabilities to pitch names
        pitch_names = self.encoder.classes_
        return dict(zip(pitch_names, probs))

    def calculate_surprisal(self, actual_pitch, probs):
        """
        Calculates the surprisal of the actual pitch in bits.
        Formula: -log2(P(pitch))
        """
        p = probs.get(actual_pitch, 0.001) # Avoid log(0)
        return -np.log2(p)
    def hydrate_and_predict(self, inference_row, baseline):
        """
        Takes a raw pitch DataFrame row, applies the hierarchical baseline fallbacks
        using the centralized baseline_manager, and returns
        (probabilities, surprisal, actual_family, hydrated_row).
        hydrated_row contains all tendency columns needed for visualization.
        """
        from src.baseline_manager import apply_baseline_to_df
        from src.api_extractors import _classify_pitch_family

        # Apply strict fallbacks using the dedicated scaler
        hydrated_row = apply_baseline_to_df(inference_row.copy(), baseline, is_train=False)
        
        # Ensure missing features expected by model are zeroed out (e.g. absent columns)
        for col in self.feature_cols:
            if col not in hydrated_row.columns or pd.isna(hydrated_row[col].iloc[0]):
                hydrated_row[col] = 0.0
                
        probabilities = self.predict_probabilities(hydrated_row)
        
        actual = 'Other'
        if 'pitch_type' in hydrated_row.columns:
            actual = _classify_pitch_family(hydrated_row['pitch_type'].iloc[0])
            
        surprisal = self.calculate_surprisal(actual, probabilities) if actual != 'Other' else 0.0
        return probabilities, surprisal, actual, hydrated_row
