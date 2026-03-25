import os
from pathlib import Path

# Project root setup
ROOT = Path(__file__).resolve().parent.parent

# Common pitch type groupings
FASTBALL_CODES = {"FF", "FT", "FC", "SI", "FS", "SF", "FA", "ST"}
BREAKING_CODES = {"SL", "CU", "KC", "SV", "CS", "CB", "GY"}
OFFSPEED_CODES = {"CH", "SC", "FO", "KN", "EP"}

# Thresholds for surprisal and outcomes
SURPRISAL_THRESHOLD = 2.5
BARREL_EV_THRESHOLD = 98.0
POLLING_INTERVAL_SECONDS = 30

# Model paths (Absolute for cross-directory compatibility)
MODEL_PATH = str(ROOT / 'models/pitch_classifier.pkl')
ENCODER_PATH = str(ROOT / 'models/encoder.pkl')
PREV_ENCODER_PATH = str(ROOT / 'models/prev_pitch_encoder.pkl')
PREV_CALL_ENCODER_PATH = str(ROOT / 'models/prev_call_encoder.pkl')
P_HAND_ENCODER_PATH = str(ROOT / 'models/p_hand_encoder.pkl')
B_SIDE_ENCODER_PATH = str(ROOT / 'models/b_side_encoder.pkl')
MOB_ENCODER_PATH = str(ROOT / 'models/mob_encoder.pkl')
OUT_PITCH_ENCODER_PATH = str(ROOT / 'models/out_pitch_encoder.pkl')
PARK_ENCODER_PATH = str(ROOT / 'models/park_encoder.pkl')
SCALER_PATH = str(ROOT / 'models/scaler.pkl')
FEATURE_COLS_PATH = str(ROOT / 'models/feature_cols.pkl')
BATTER_FEATURES_PATH = str(ROOT / 'models/batter_features.joblib')
BASELINE_PATH = str(ROOT / 'models/baseline_tendencies.pkl')
DATABASE_PATH = str(ROOT / 'data/pitches.db')
