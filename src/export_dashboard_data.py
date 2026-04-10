
import pandas as pd
import json
import os
import numpy as np
import joblib
import statsapi
from pathlib import Path
from src.inference import PitchPredictor
from src.constants import BASELINE_PATH, MODEL_PATH, TARGET_ENCODER_PATH, BARREL_EV_THRESHOLD
from src.database import query_all_pitches, get_recent_live_predictions
from src.api_extractors import _classify_pitch_family

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data/pitches.db"
OUTPUT_DIR = ROOT / "dashboard/public/data"

def _clean_json_data(obj):
    """Recursively convert numpy types and handle NaNs for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _clean_json_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_json_data(i) for i in obj]
    elif isinstance(obj, (float, np.float32, np.float64)):
        if pd.isna(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, (int, np.int32, np.int64)):
        return int(obj)
    elif pd.isna(obj):
        return None
    return obj

def _get_game_metadata(date_str):
    """Fetches schedule for a specific date to get team names and venue details."""
    try:
        schedule = statsapi.schedule(date=date_str)
        metadata = {}
        for game in schedule:
            game_id = str(game.get('game_id'))
            metadata[game_id] = {
                "matchup": f"{game.get('away_name')} @ {game.get('home_name')}",
                "venue_name": game.get('venue_name', 'Unknown Venue'),
                "away_abbr": game.get('away_id'), # Fallback to ID if abbr missing
                "home_abbr": game.get('home_id')
            }
        return metadata
    except Exception as e:
        print(f"Error fetching metadata for {date_str}: {e}")
        return {}

PITCH_NAMES = {
    "CH": "Changeup",
    "CU": "Curveball",
    "FC": "Cutter",
    "EP": "Eephus",
    "FO": "Forkball",
    "FF": "Four-Seam Fastball",
    "KN": "Knuckleball",
    "KC": "Knuckle Curve",
    "SC": "Screwball",
    "SI": "Sinker",
    "SL": "Slider",
    "SV": "Slurve",
    "FS": "Splitter",
    "ST": "Sweeper",
    "CS": "Slow Curve",
    "CB": "Curveball",
    "FA": "Fastball",
    "FT": "Two-Seam Fastball",
    "GY": "Gyroball"
}

def _get_explorer_data(df, predictor, baseline, pitcher_mixes=None, batter_mixes=None):
    """Builds a hierarchical structure for the Matchup Explorer."""
    
    def _extract_mix(row_df, prefix):
        """Helper to extract tendency percentages from a row and format as a dict."""
        families = ["Fastball", "Breaking", "Offspeed", "Other"]
        mix = {}
        for fam in families:
            col = f"{prefix}_{fam}_pct"
            if col in row_df.columns:
                val = row_df[col].iloc[0]
                mix[fam] = float(val) if not pd.isna(val) else 0.0
        
        # Also extract total sample size for this specific count
        total_col = f"{prefix}_total_pitches"
        if total_col in row_df.columns:
            mix["total"] = int(row_df[total_col].iloc[0])
        return mix

    if df.empty or not predictor or not baseline:
        return {}

    # Filter for last 5 unique dates to keep data small but relevant
    recent_dates = sorted(df['game_date'].unique(), reverse=True)[:5]
    rdf = df[df['game_date'].isin(recent_dates)].copy()
    
    # Pre-calculate count one-hots for efficiency
    for b in range(4):
        for s in range(3):
            rdf[f"count_{b}_{s}"] = ((rdf['balls'] == b) & (rdf['strikes'] == s)).astype(int)
    
    if 'is_leverage' not in rdf.columns:
        rdf['is_leverage'] = ((rdf['inning'] >= 7) & (rdf['score_home'].sub(rdf['score_away']).abs() <= 2)).astype(int)

    explorer = {}
    
    # Group by date
    for date, date_df in rdf.groupby('game_date'):
        date_str = str(date)
        explorer[date_str] = {"games": {}}
        
        # Fetch metadata for this date's games
        game_metadata = _get_game_metadata(date_str)
        
        # Group by game
        for game_pk, game_df in date_df.groupby('game_pk'):
            pk_str = str(game_pk)
            meta = game_metadata.get(pk_str, {})
            
            explorer[date_str]["games"][pk_str] = {
                "id": pk_str,
                "matchup": meta.get("matchup", "Unknown Matchup"),
                "venue": meta.get("venue_name", "Unknown Venue"),
                "innings": {}
            }
            
            # Group by inning
            for inning, inning_df in game_df.groupby('inning'):
                inn_str = str(inning)
                explorer[date_str]["games"][pk_str]["innings"][inn_str] = {"top": [], "bottom": []}
                
                # Group by half-inning
                for half, half_df in inning_df.groupby('half_inning'):
                    half_key = str(half).lower()
                    if half_key not in ["top", "bottom"]: continue
                    
                    # Group by at-bat
                    for ab_index, ab_df in half_df.groupby('at_bat_index'):
                        # Sort by pitch index
                        ab_df = ab_df.sort_values('pitch_index')
                        rep_ab = ab_df.iloc[0]
                        
                        pitcher_name = str(rep_ab.get('pitcher', 'Unknown'))
                        batter_name = str(rep_ab.get('batter', 'Unknown'))
                        
                        ab_data = {
                            "ab_index": int(ab_index),
                            "pitcher": pitcher_name,
                            "batter": batter_name,
                            "pitcher_hand": str(rep_ab.get('pitcher_hand', 'Unknown')),
                            "batter_side": str(rep_ab.get('batter_side', 'Unknown')),
                            "outs": int(rep_ab.get('outs', 0)),
                            "men_on_base": str(rep_ab.get('men_on_base', 'Empty')),
                            "pitcher_mix": pitcher_mixes.get(pitcher_name, {}) if pitcher_mixes else {},
                            "batter_mix": batter_mixes.get(batter_name, {}) if batter_mixes else {},
                            "pitches": []
                        }
                        
                        # Process each pitch
                        for _, row in ab_df.iterrows():
                            # Inference logic
                            inference_row = pd.DataFrame([row])
                            try:
                                actual_p_type = str(row.get('pitch_type', 'UNK'))
                                if actual_p_type == 'nan' or not actual_p_type: actual_p_type = 'UNK'
                                
                                probs, surprisal, actual_fam = predictor.hydrate_and_predict(inference_row, baseline)
                                balls = int(row.get('balls', 0))
                                strikes = int(row.get('strikes', 0))
                                
                                # Best prediction
                                pred_fam = max(probs.items(), key=lambda x: x[1])[0]
                                
                                pitch_data = {
                                    "p_type": PITCH_NAMES.get(actual_p_type, actual_p_type),
                                    "p_code": actual_p_type,
                                    "fam": actual_fam,
                                    "speed": float(row.get('release_speed', 0.0)) if not pd.isna(row.get('release_speed')) else 0.0,
                                    "count": f"{balls}-{strikes}",
                                    "prediction": pred_fam,
                                    "prob": float(probs.get(actual_fam, 0.0)),
                                    "surprisal": float(surprisal) if not np.isinf(surprisal) else 10.0,
                                    "outcome": str(row.get('call', 'Unknown')),
                                    "pitcher_count_mix": _extract_mix(inference_row, "tendency_count"),
                                    "batter_count_mix": _extract_mix(inference_row, "tendency_batter_count"),
                                    "league_count_mix": _extract_mix(inference_row, "tendency_league_count")
                                }
                                ab_data["pitches"].append(pitch_data)
                            except:
                                continue
                        
                        explorer[date_str]["games"][pk_str]["innings"][inn_str][half_key].append(ab_data)
                        
    return explorer

def export_dashboard_data():
    """Extracts interesting stats from sqlite and saves as JSON for the React app."""
    print("Exporting data for Web Dashboard...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH}")
        return
        
    df = query_all_pitches()
    
    if df.empty:
        print("Database is empty.")
        return
        
    print(f"Loaded {len(df)} total pitches.")
    
    # Initialize Predictor and Baseline
    try:
        predictor = PitchPredictor(model_path=MODEL_PATH, target_encoder_path=TARGET_ENCODER_PATH)
        baseline = joblib.load(BASELINE_PATH)
        print("Model and Baseline loaded for inference.")
    except Exception as e:
        print(f"Error loading model/baseline: {e}. Falling back to mock data if necessary.")
        predictor = None
        baseline = None

    # Map database columns to expected names if necessary
    # Based on research: velocity is used instead of release_speed
    if 'velocity' in df.columns and 'release_speed' not in df.columns:
        df['release_speed'] = df['velocity']
    
    # 1. Summary Stats
    total_pitches = len(df)
    total_games = df['game_pk'].nunique() if 'game_pk' in df.columns else 0
    
    # Detection for historical data where 'events' might be missing
    # A strikeout happens on a strike with 2 strikes already in the count
    def is_strikeout(row):
        if 'events' in row and row['events'] == 'strikeout':
            return True
        # Fallback: Call contains 'Strike' and strikes == 2
        return row.get('strikes') == 2 and 'Strike' in str(row.get('call', ''))

    df['is_strikeout'] = df.apply(is_strikeout, axis=1)
    k_count = df['is_strikeout'].sum()
    
    # 2. Pitch Group Distribution
    df['pitch_family'] = df['pitch_type'].apply(_classify_pitch_family)
    family_counts = df[df['pitch_family'] != "Unknown"]['pitch_family'].value_counts().to_dict()
    
    # 3. Top Pitchers by Usage
    top_pitchers_series = df['pitcher'].value_counts().head(10)
    
    # 4. Actual Surprisal Data
    surprises = []
    
    # Filter for interesting outcomes
    # Hard hits: Launch speed >= threshold
    if 'launch_speed' in df.columns:
        hard_hit_mask = df['launch_speed'] >= BARREL_EV_THRESHOLD
    else:
        hard_hit_mask = pd.Series([False] * len(df))
        
    interesting_plays = df[df['is_strikeout'] | hard_hit_mask].copy()
    
    if not interesting_plays.empty and predictor and baseline:
        print(f"Running inference on {len(interesting_plays)} interesting plays...")
        
        # Pre-calculate count one-hots for the whole subset to optimize
        for b in range(4):
            for s in range(3):
                col = f"count_{b}_{s}"
                interesting_plays[col] = ((interesting_plays['balls'] == b) & (interesting_plays['strikes'] == s)).astype(int)
        
        # Add leverage indicator if missing
        if 'is_leverage' not in interesting_plays.columns:
            interesting_plays['is_leverage'] = ((interesting_plays['inning'] >= 7) & (interesting_plays['score_home'].sub(interesting_plays['score_away']).abs() <= 2)).astype(int)

        for idx, row in interesting_plays.iterrows():
            # Prep row for inference
            inference_row = pd.DataFrame([row])
            
            try:
                balls = row.get('balls', 0)
                strikes = row.get('strikes', 0)
                probabilities, surprisal, actual_pitch_family = predictor.hydrate_and_predict(inference_row, baseline)
                prob = probabilities.get(actual_pitch_family, 0.001)
                
                # Filter out infinity surprisal
                if surprisal == float('inf') or pd.isna(surprisal):
                    surprisal = 10.0 # High fallback
                
                # Ensure prob is never NaN
                if pd.isna(prob):
                    prob = 0.001

                outcome = "Strikeout" if row.get('is_strikeout') else "Hard Hit"
                
                speed = row.get('release_speed', 0.0)
                if pd.isna(speed):
                    speed = 0.0
                speed = float(speed)

                surprises.append({
                    "id": str(idx),
                    "pitcher": str(row.get('player_name', 'Unknown')),
                    "batter": str(row.get('batter_name', 'Unknown')),
                    "pitch_type": str(row.get('pitch_type', 'UNK')),
                    "pitch_family": _classify_pitch_family(row.get('pitch_type')),
                    "speed": speed,
                    "count": f"{int(row.get('balls', 0))}-{int(row.get('strikes', 0))}",
                    "inning": f"{row.get('inning_topbot', 'top').lower()} {int(row.get('inning', 1))}",
                    "prob": float(prob),
                    "surprisal": float(surprisal),
                    "outcome": str(outcome),
                    "launch_speed": float(row.get('launch_speed', 0)) if 'launch_speed' in row and not pd.isna(row.get('launch_speed')) else None
                })
            except Exception as e:
                # Log error for this specific row but continue
                continue
    
    # Sort and take top 50
    surprises = sorted(surprises, key=lambda x: x['surprisal'], reverse=True)[:50]
    
    # Final cleanup of surprises to ensure no numpy types
    for s in surprises:
        s['prob'] = float(s['prob'])
        s['surprisal'] = float(s['surprisal'])
        if s['launch_speed'] is not None:
            s['launch_speed'] = float(s['launch_speed'])
    
    # 5. Pitcher Spotlight Data
    pitcher_spotlight = []
    for p_name in top_pitchers_series.index[:5]:
        p_df = df[df['pitcher'] == p_name]
        p_dist = p_df['pitch_family'].value_counts().to_dict()
        
        # Safe aggregation for speed
        if not p_df.empty and 'release_speed' in p_df.columns:
            avg_speed_series = p_df.groupby('pitch_family')['release_speed'].mean()
            avg_speeds = {str(k): round(float(v), 1) for k, v in avg_speed_series.to_dict().items() if not pd.isna(v)}
        else:
            avg_speeds = {}
        
        pitcher_spotlight.append({
            "name": str(p_name),
            "total_pitches": int(len(p_df)),
            "distribution": {str(k): int(v) for k, v in p_dist.items()},
            "avg_speeds": avg_speeds
        })

    # Calculate global mixes for all pitchers and batters in the dataset
    pitcher_mixes = {}
    if 'pitcher' in df.columns and 'pitch_family' in df.columns:
        p_groups = df[df['pitch_family'] != "Unknown"].groupby(['pitcher', 'pitch_family']).size().unstack(fill_value=0)
        pitcher_mixes = {k: {fk: int(fv) for fk, fv in v.items()} for k, v in p_groups.to_dict(orient='index').items()}
        
    batter_mixes = {}
    # Sometimes it's 'batter' sometimes 'batter_name' in SQL based on generation
    batter_col = 'batter' if 'batter' in df.columns else 'batter_name'
    if batter_col in df.columns and 'pitch_family' in df.columns:
        b_groups = df[df['pitch_family'] != "Unknown"].groupby([batter_col, 'pitch_family']).size().unstack(fill_value=0)
        batter_mixes = {k: {fk: int(fv) for fk, fv in v.items()} for k, v in b_groups.to_dict(orient='index').items()}

    # 6. Matchup Explorer Data
    explorer_data = _get_explorer_data(df, predictor, baseline, pitcher_mixes, batter_mixes)

    # 7. Live Metrics History
    live_metrics = []
    try:
        live_df = get_recent_live_predictions(days=7)
        
        if not live_df.empty:
            # Coerce float columns – some older rows may store bytes from legacy schema
            for fc in ['surprisal', 'prob_fastball', 'prob_breaking', 'prob_offspeed']:
                live_df[fc] = pd.to_numeric(live_df[fc], errors='coerce')
            # Drop inf / NaN
            live_df = live_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['surprisal', 'prob_fastball', 'prob_breaking', 'prob_offspeed', 'actual_pitch_family'])
            live_df['date'] = pd.to_datetime(live_df['timestamp']).dt.date
            
            for d, group in live_df.groupby('date'):
                avg_surprisal = group['surprisal'].mean()
                
                actual_map = {'Fastball': 0, 'Breaking': 1, 'Offspeed': 2}
                valid_group = group[group['actual_pitch_family'].isin(actual_map.keys())]
                
                brier = 0.0
                if not valid_group.empty:
                    actual_idx = valid_group['actual_pitch_family'].map(actual_map).values
                    y_true = np.zeros((len(actual_idx), 3))
                    y_true[np.arange(len(actual_idx)), actual_idx] = 1.0
                    
                    valid_preds = valid_group[['prob_fastball', 'prob_breaking', 'prob_offspeed']].values
                    brier = np.mean(np.sum((valid_preds - y_true)**2, axis=1)) # Multi-class Brier score
                    
                live_metrics.append({
                    "date": str(d),
                    "avg_surprisal": float(avg_surprisal),
                    "brier_score": float(brier),
                    "sample_size": int(len(group))
                })
        live_metrics = sorted(live_metrics, key=lambda x: x['date'])
    except Exception as e:
        print(f"Error generating live metrics: {e}")

    # Compile the final data object
    dashboard_data = {
        "last_updated": pd.Timestamp.now().isoformat(),
        "summary": {
            "total_pitches": int(total_pitches),
            "total_games": int(total_games),
            "total_strikeouts": int(k_count)
        },
        "pitch_distribution": {str(k): int(v) for k, v in family_counts.items()},
        "top_pitchers": [{"name": str(k), "count": int(v)} for k, v in top_pitchers_series.items()],
        "top_surprises": surprises,
        "pitcher_spotlight": pitcher_spotlight,
        "explorer": explorer_data,
        "live_metrics": live_metrics
    }
    
    # Save to the React public folder so it can be fetched client-side
    # Clean the entire object before export
    cleaned_dashboard_data = _clean_json_data(dashboard_data)
    
    # Save as JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "dashboard_data.json", "w") as f:
        json.dump(cleaned_dashboard_data, f, indent=2)
    
    print(f"\nDashboard data exported successfully to {OUTPUT_DIR / 'dashboard_data.json'}")

if __name__ == "__main__":
    export_dashboard_data()
