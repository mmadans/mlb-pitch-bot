import xgboost as xgb
import pandas as pd
import joblib
import numpy as np

from src.constants import (
    MODEL_PATH, ENCODER_PATH, PREV_ENCODER_PATH, FEATURE_COLS_PATH,
    P_HAND_ENCODER_PATH, B_SIDE_ENCODER_PATH, MOB_ENCODER_PATH
)

from src.features import _classify_pitch_family

class PitchPredictor:
    """
    Wrapper for the XGBoost model to handle preprocessing and prediction.
    """
    def __init__(self, model_path=MODEL_PATH, encoder_path=ENCODER_PATH, 
                 prev_encoder_path=PREV_ENCODER_PATH, feature_cols_path=FEATURE_COLS_PATH,
                 p_hand_path=P_HAND_ENCODER_PATH, b_side_path=B_SIDE_ENCODER_PATH,
                 mob_path=MOB_ENCODER_PATH):
        """Loads the model and associated encoders from disk."""
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.prev_encoder = joblib.load(prev_encoder_path)
        self.feature_cols = joblib.load(feature_cols_path)
        self.p_hand_encoder = joblib.load(p_hand_path)
        self.b_side_encoder = joblib.load(b_side_path)
        self.mob_encoder = joblib.load(mob_path)

    def predict_probabilities(self, features_df):
        """
        Prepares features and calculates probabilities for each pitch family.
        
        Args:
            features_df: A DataFrame containing a single row of pitch context.
        Returns:
            A dictionary mapping pitch families to their predicted probabilities.
        """
        # categorical encoding
        if 'prev_pitch_type_in_ab' in features_df.columns:
            raw_prev = features_df['prev_pitch_type_in_ab'].iloc[0]
            family_prev = _classify_pitch_family(raw_prev)
            labels = list(self.prev_encoder.classes_)
            val = self.prev_encoder.transform([family_prev])[0] if family_prev in labels else 0
            features_df['prev_pitch_family_enc'] = val

        if 'pitcher_hand' in features_df.columns:
            val = features_df['pitcher_hand'].iloc[0] or "R"
            labels = list(self.p_hand_encoder.classes_)
            features_df['pitcher_hand_enc'] = self.p_hand_encoder.transform([val])[0] if val in labels else 0

        if 'batter_side' in features_df.columns:
            val = features_df['batter_side'].iloc[0] or "R"
            labels = list(self.b_side_encoder.classes_)
            features_df['batter_side_enc'] = self.b_side_encoder.transform([val])[0] if val in labels else 0

        if 'men_on_base' in features_df.columns:
            val = features_df['men_on_base'].iloc[0] or "Empty"
            labels = list(self.mob_encoder.classes_)
            features_df['men_on_base_enc'] = self.mob_encoder.transform([val])[0] if val in labels else 0

        # Ensure all columns the model expects are present
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0

        # XGBClassifier.predict_proba returns probability for each class
        # Ensure we only pass the columns the model was trained on
        model_input = features_df[self.feature_cols]
        probs = self.model.predict_proba(model_input)
        
        # Map probabilities to pitch names
        pitch_names = self.encoder.classes_
        return dict(zip(pitch_names, probs[0]))

    def calculate_surprisal(self, actual_pitch, probs):
        """
        Calculates the surprisal of the actual pitch in bits.
        Formula: -log2(P(pitch))
        """
        p = probs.get(actual_pitch, 0.001) # Avoid log(0)
        return -np.log2(p)