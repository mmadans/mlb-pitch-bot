import xgboost as xgb
import pandas as pd
import joblib
import numpy as np

class PitchPredictor:
    def __init__(self, model_path='models/pitch_model.json', encoder_path='models/encoder.pkl'):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.encoder = joblib.load(encoder_path)

    def predict_probabilities(self, features_df):
        # XGBoost DMatrix is faster for single-row inference
        dmatrix = xgb.DMatrix(features_df)
        probs = self.model.predict(dmatrix) # Returns array of probabilities
        
        # Map probabilities to pitch names
        pitch_names = self.encoder.classes_
        return dict(zip(pitch_names, probs[0]))

    def calculate_surprisal(self, actual_pitch, probs):
        p = probs.get(actual_pitch, 0.001) # Avoid log(0)
        return -np.log2(p)