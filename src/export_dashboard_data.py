import sqlite3
import pandas as pd
import json
import os
import numpy as np
import joblib
import statsapi
from pathlib import Path
from src.inference import PitchPredictor
from src.constants import BASELINE_PATH, MODEL_PATH, ENCODER_PATH, BARREL_EV_THRESHOLD
from src.features import _classify_pitch_family

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

def _get_explorer_data(df, predictor, baseline):
    """Builds a hierarchical structure for the Matchup Explorer."""
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
                        
                        ab_data = {
                            "ab_index": int(ab_index),
                            "pitcher": str(rep_ab.get('pitcher', 'Unknown')),
                            "batter": str(rep_ab.get('batter', 'Unknown')),
                            "pitches": []
                        }
                        
                        # Process each pitch
                        for _, row in ab_df.iterrows():
                            # Inference logic
                            inference_row = pd.DataFrame([row])
                            pitcher = row.get('pitcher')
                            balls = int(row.get('balls', 0))
                            strikes = int(row.get('strikes', 0))
                            
                            # Hydrate from baseline
                            for level in ['global', 'count']:
                                key = pitcher if level == 'global' else (pitcher, balls, strikes)
                                stats = baseline.get(level, {}).get(key, {})
                                for col, val in stats.items():
                                    inference_row[col] = val
                            
                            # Batter/League
                            b_id = row.get('batter_id')
                            if b_id:
                                b_stats = baseline.get('batter_count', {}).get((b_id, balls, strikes), {})
                                for col, val in b_stats.items():
                                    inference_row[col] = val
                            
                            l_stats = baseline.get('league_count', {}).get((balls, strikes), {})
                            for col, val in l_stats.items():
                                inference_row[col] = val

                            # Fill missing
                            for col in predictor.feature_cols:
                                if col not in inference_row.columns or pd.isna(inference_row[col].iloc[0]):
                                    inference_row[col] = 0.0

                            try:
                                actual_p_type = str(row.get('pitch_type', 'UNK'))
                                if actual_p_type == 'nan' or not actual_p_type: actual_p_type = 'UNK'
                                
                                actual_fam = _classify_pitch_family(actual_p_type)
                                probs = predictor.predict_probabilities(inference_row)
                                
                                # Best prediction
                                pred_fam = max(probs.items(), key=lambda x: x[1])[0]
                                surprisal = predictor.calculate_surprisal(actual_fam, probs)
                                
                                pitch_data = {
                                    "p_type": actual_p_type,
                                    "fam": actual_fam,
                                    "speed": float(row.get('release_speed', 0.0)) if not pd.isna(row.get('release_speed')) else 0.0,
                                    "count": f"{balls}-{strikes}",
                                    "prediction": pred_fam,
                                    "prob": float(probs.get(actual_fam, 0.0)),
                                    "surprisal": float(surprisal) if not np.isinf(surprisal) else 10.0
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
        
    conn = sqlite3.connect(DB_PATH)
    
    # Load recent pitches
    df = pd.read_sql("SELECT * FROM pitches", conn)
    conn.close()
    
    if df.empty:
        print("Database is empty.")
        return
        
    print(f"Loaded {len(df)} total pitches.")
    
    # Initialize Predictor and Baseline
    try:
        predictor = PitchPredictor(model_path=MODEL_PATH, encoder_path=ENCODER_PATH)
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
            
            # Simple hydration for export (baseline stats)
            pitcher = row.get('pitcher')
            balls = row.get('balls', 0)
            strikes = row.get('strikes', 0)
            
            # Global tendencies
            p_global = baseline['global'].get(pitcher, {})
            for col, val in p_global.items():
                inference_row[col] = val
                
            # Count tendencies
            p_count = baseline['count'].get((pitcher, balls, strikes), {})
            for col, val in p_count.items():
                inference_row[col] = val
            
            # Batter/League tendencies if available in baseline
            batter_id = row.get('batter_id')
            if batter_id:
                b_count = baseline.get('batter_count', {}).get((batter_id, balls, strikes), {})
                for col, val in b_count.items():
                    inference_row[col] = val
            
            l_count = baseline.get('league_count', {}).get((balls, strikes), {})
            for col, val in l_count.items():
                inference_row[col] = val

            # Fill all expected model columns with 0 if missing/NaN
            for col in predictor.feature_cols:
                if col not in inference_row.columns or pd.isna(inference_row[col].iloc[0]):
                    inference_row[col] = 0.0

            # Predict (predict_probabilities will handle the remaining _enc columns internal to its logic)
            try:
                actual_pitch_family = _classify_pitch_family(row.get('pitch_type'))
                probabilities = predictor.predict_probabilities(inference_row)
                surprisal = predictor.calculate_surprisal(actual_pitch_family, probabilities)
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
                    "pitch_type": p_type,
                    "pitch_family": _classify_pitch_family(p_type),
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

    # 6. Matchup Explorer Data
    explorer_data = _get_explorer_data(df, predictor, baseline)

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
        "explorer": explorer_data
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
