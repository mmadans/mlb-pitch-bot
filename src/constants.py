from pathlib import Path

# Project root setup
ROOT = Path(__file__).resolve().parent.parent

# Common pitch type groupings
FASTBALL_CODES = {"FF", "FT", "FC", "SI", "FA"}
BREAKING_CODES = {"SL", "CU", "KC", "SV", "CS", "CB", "GY", "ST"}
OFFSPEED_CODES = {"CH", "SC", "FO", "KN", "EP", "FS", "SF"}

PITCH_COLORS = {
    "Fastball": "#D22D49",
    "Breaking": "#00A1DE",
    "Offspeed": "#1D8A58",
    "Other": "#93A1A1",
}

TEAM_HASHTAGS = {
    "AZ": "#DBacks",
    "ATH": "#Athletics",
    "ATL": "#BravesCountry",
    "BAL": "#Birdland",
    "BOS": "#DirtyWater",
    "CHC": "#Cubs",
    "CWS": "#WhiteSox",
    "CIN": "#ATOBTTR",
    "CLE": "#GuardsBall",
    "COL": "#Rockies",
    "DET": "#RepDetroit",
    "HOU": "#ChaseTheFight",
    "KC": "#FountainsUp",
    "LAA": "#RepTheHalo",
    "LAD": "#Dodgers",
    "MIA": "#FightinFish",
    "MIL": "#ThisIsMyCrew",
    "MIN": "#MNTwins",
    "NYM": "#LGM",
    "NYY": "#RepBX",
    "PHI": "#RingTheBell",
    "PIT": "#LetsGoBucs",
    "SD": "#ForTheFaithful",
    "SF": "#SFGiants",
    "SEA": "#TridentsUp",
    "STL": "#ForTheLou",
    "TB": "#RaysUp",
    "TEX": "#AllForTX",
    "TOR": "#BlueJays50",
    "WSH": "#Natitude",
}

# Thresholds for surprisal and outcomes
SURPRISAL_THRESHOLD = 2.5
BARREL_EV_THRESHOLD = 98.0
POLLING_INTERVAL_SECONDS = 30

# Inference
TENDENCY_BLEND_PRIOR_N = 30  # sample-size prior for pitcher count-tendency blending
MIN_PITCH_PROB = (
    0.01  # probability floor — no pitch family is ever essentially impossible
)

# Monitoring / calibration
N_CAL_BINS = 5  # calibration bins (wide enough to have data on a single game day)

# Empirical MLB pitch-family distribution; used as fallback when pitcher history is missing
LEAGUE_PRIORS = {"Fastball": 0.55, "Breaking": 0.25, "Offspeed": 0.20}

# Model paths (Absolute for cross-directory compatibility)
MODEL_PATH = str(ROOT / "models/pitch_classifier.pkl")
TARGET_ENCODER_PATH = str(ROOT / "models/target_encoder.pkl")
CATEGORICAL_ENCODER_PATH = str(ROOT / "models/categorical_encoder.pkl")
FEATURE_COLS_PATH = str(ROOT / "models/feature_cols.pkl")
BATTER_FEATURES_PATH = str(ROOT / "models/batter_features.joblib")
BASELINE_PATH = str(ROOT / "models/baseline_tendencies.pkl")
DATABASE_PATH = str(ROOT / "data/pitches.db")
