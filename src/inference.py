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

        # Add platoon advantage
        if 'pitcher_hand' in features_df.columns and 'batter_side' in features_df.columns:
            features_df['is_platoon_advantage'] = ((features_df['pitcher_hand'] == features_df['batter_side']) & (features_df['batter_side'] != 'S')).astype(int)

        # Ensure all columns the model expects are present
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0

        model_input = features_df[self.feature_cols]
        
        # Predict probabilities
        raw_probs = self.model.predict_proba(model_input)[0]
        
        # Apply a floor of 1e-4 to ensure no class is exactly 0.0
        # This prevents the '0.0% chance' bug in tweets
        epsilon = 0.0001
        probs = (raw_probs + epsilon)
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
        using the centralized baseline_manager, and returns (probabilities, surprisal, actual_family).
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
        return probabilities, surprisal, actual
