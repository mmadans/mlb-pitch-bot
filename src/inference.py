import xgboost as xgb
import pandas as pd
import joblib
import numpy as np

class PitchPredictor:
    def __init__(self, model_path='models/pitch_classifier.pkl', encoder_path='models/encoder.pkl', prev_encoder_path='models/prev_pitch_encoder.pkl', feature_cols_path='models/feature_cols.pkl'):
        # Train model saves an XGBClassifier object via joblib
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.prev_encoder = joblib.load(prev_encoder_path)
        self.feature_cols = joblib.load(feature_cols_path)

    def predict_probabilities(self, features_df):
        # Handle the categorical encoding for previous pitch
        if 'prev_pitch_type_in_ab' in features_df.columns:
            prev = features_df['prev_pitch_type_in_ab'].fillna("None").astype(str).str.upper()
            # We need to handle unseen labels gracefully
            # A simple way is to map unknown to the 'None' or 'UN' class index if it exists
            labels = list(self.prev_encoder.classes_)
            features_df['prev_pitch_type_in_ab_enc'] = prev.apply(
                lambda x: self.prev_encoder.transform([x])[0] if x in labels else self.prev_encoder.transform(["NONE"])[0] if "NONE" in labels else 0
            )

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
        p = probs.get(actual_pitch, 0.001) # Avoid log(0)
        return -np.log2(p)