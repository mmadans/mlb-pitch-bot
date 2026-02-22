"""
Shared constants for the MLB Pitch Bot.
"""

# Common pitch type groupings
FASTBALL_CODES = {"FF", "FT", "FC", "SI", "FS", "SF", "FA", "ST"}
BREAKING_CODES = {"SL", "CU", "KC", "SV", "CS", "CB", "GY"}
OFFSPEED_CODES = {"CH", "SC", "FO", "KN", "EP"}

# Thresholds for surprisal and outcomes
SURPRISAL_THRESHOLD = 2.5
BARREL_EV_THRESHOLD = 98.0
POLLING_INTERVAL_SECONDS = 30

# Model paths
MODEL_PATH = 'models/pitch_classifier.pkl'
ENCODER_PATH = 'models/encoder.pkl'
PREV_ENCODER_PATH = 'models/prev_pitch_encoder.pkl'
FEATURE_COLS_PATH = 'models/feature_cols.pkl'
BASELINE_PATH = 'models/baseline_tendencies.pkl'
