"""Generate sample_1.png, sample_2.png, sample_3.png in output/."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bot.visualization import generate_pitch_infographic

def _seq(entries):
    return [
        {"pX": x, "pZ": z, "pitch_family": fam, "pitch_type_desc": desc, "call": call}
        for x, z, fam, desc, call in entries
    ]

# ── Sample 1: Gerrit Cole, 2-strike count, predicted Fastball, threw Fastball ──
sample_1_data = {
    "pitcher": "Gerrit Cole", "pitcher_hand": "R",
    "batter": "Rafael Devers", "batter_side": "L",
    "inning": 7, "half_inning": "top",
    "balls": 1, "strikes": 2, "outs": 1,
    "away_team": "NYY", "home_team": "BOS",
    "score_away": 3, "score_home": 2,
    "men_on_base": "First",
    "pitch_type": "FF", "pitch_type_desc": "Four-Seam Fastball",
    "pitch_family": "Fastball",
    "velocity": 97.4, "spin_rate": 2520,
    "pX": 0.35, "pZ": 3.10,
    "call": "Swinging Strike",
    "primary_out_pitch": "Fastball",
    "is_platoon_advantage": 0,
    "fastball_streak": 2,
    "breaking_streak": 0,
    "offspeed_streak": 0,
    "tendency_global_Fastball_pct": 0.58,
    "tendency_count_Fastball_pct": 0.71,
    "tendency_batter_count_Fastball_pct": 0.65,
}
sample_1_probs = {"Fastball": 0.72, "Breaking": 0.19, "Offspeed": 0.09}
sample_1_seq = _seq([
    (-0.40, 2.80, "Fastball",  "Four-Seam Fastball", "Ball"),
    ( 0.10, 3.20, "Breaking",  "Slider",             "Foul"),
    ( 0.55, 2.60, "Fastball",  "Four-Seam Fastball", "Foul"),
    (-0.20, 3.40, "Offspeed",  "Changeup",           "Ball"),
])

# ── Sample 2: Corbin Burnes, 3-ball count, predicted Breaking, threw Slider ──
sample_2_data = {
    "pitcher": "Corbin Burnes", "pitcher_hand": "R",
    "batter": "Freddie Freeman", "batter_side": "L",
    "inning": 4, "half_inning": "bottom",
    "balls": 3, "strikes": 1, "outs": 2,
    "away_team": "MIL", "home_team": "LAD",
    "score_away": 1, "score_home": 4,
    "men_on_base": "Second_Third",
    "pitch_type": "SL", "pitch_type_desc": "Slider",
    "pitch_family": "Breaking",
    "velocity": 87.1, "spin_rate": 2740,
    "pX": -0.55, "pZ": 1.80,
    "call": "Called Strike",
    "primary_out_pitch": "Breaking",
    "is_platoon_advantage": 1,
    "fastball_streak": 0,
    "breaking_streak": 0,
    "offspeed_streak": 0,
    "tendency_global_Breaking_pct": 0.34,
    "tendency_count_Breaking_pct": 0.29,
    "tendency_batter_count_Breaking_pct": 0.22,
}
sample_2_probs = {"Fastball": 0.31, "Breaking": 0.48, "Offspeed": 0.21}
sample_2_seq = _seq([
    ( 0.20, 3.30, "Fastball", "Sinker",   "Ball"),
    (-0.30, 2.50, "Breaking", "Slider",   "Foul"),
    ( 0.60, 3.50, "Fastball", "Cutter",   "Ball"),
    ( 0.15, 2.90, "Offspeed", "Splitter", "Ball"),
])

# ── Sample 3: Zack Wheeler, first pitch, predicted Fastball, threw Sinker (surprise) ──
sample_3_data = {
    "pitcher": "Zack Wheeler", "pitcher_hand": "R",
    "batter": "Yordan Alvarez", "batter_side": "L",
    "inning": 1, "half_inning": "top",
    "balls": 0, "strikes": 0, "outs": 0,
    "away_team": "HOU", "home_team": "PHI",
    "score_away": 0, "score_home": 0,
    "men_on_base": "Empty",
    "pitch_type": "SI", "pitch_type_desc": "Sinker",
    "pitch_family": "Fastball",
    "velocity": 96.2, "spin_rate": 2180,
    "pX": 0.50, "pZ": 2.20,
    "call": "In play, out(s)",
    "primary_out_pitch": "Fastball",
    "is_platoon_advantage": 0,
    "fastball_streak": 0,
    "breaking_streak": 0,
    "offspeed_streak": 0,
    "tendency_global_Fastball_pct": 0.52,
    "tendency_count_Fastball_pct": 0.60,
    "tendency_batter_count_Fastball_pct": 0.55,
}
sample_3_probs = {"Fastball": 0.61, "Breaking": 0.25, "Offspeed": 0.14}
sample_3_seq = []  # first at-bat, no prior pitches

samples = [
    (sample_1_data, sample_1_probs, 1.8, sample_1_seq, "output/sample_1.png"),
    (sample_2_data, sample_2_probs, 2.3, sample_2_seq, "output/sample_2.png"),
    (sample_3_data, sample_3_probs, 0.9, sample_3_seq, "output/sample_3.png"),
]

for pitch_data, probs, surprisal, sequence, path in samples:
    out = generate_pitch_infographic(pitch_data, probs, surprisal, sequence, path)
    print(f"Saved: {out}")
