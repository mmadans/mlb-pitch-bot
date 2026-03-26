"""
Monitors live MLB games to identify surprising pitches with specific outcomes.
Analyzes only:
1. Strikeouts
2. Barreled (Hard Hit) balls that do not result in an out.
"""
from datetime import datetime, timedelta
import time
import statsapi
import pandas as pd
import os
import joblib
from dotenv import load_dotenv

from src.features import (
    extract_pitches_with_context, 
    add_contextual_features, 
    add_global_pitcher_tendencies,
    _classify_pitch_family
)
from src.inference import PitchPredictor
from src.bot import post_tweet, format_tweet, format_surprise_strikeout_tweet
from src.database import create_live_predictions_table, insert_live_prediction
from src.constants import (
    POLLING_INTERVAL_SECONDS,
    SURPRISAL_THRESHOLD,
    BARREL_EV_THRESHOLD,
    MODEL_PATH,
    ENCODER_PATH,
    BASELINE_PATH
)

load_dotenv()

# --- State ---
processed_pitches = set()
matchup_tracker = {} # (game_pk, pitcher_id, batter_id) -> count
pending_tweets = [] # list of dicts: {'post_at': timestamp, 'text': string, 'game_pk': int, 'play_id': str}
baseline = None # Global baseline to be loaded in main()

def get_live_game_pks():
    """Fetches gamePks for games that are currently live."""
    try:
        schedule = statsapi.schedule() # Use today's date by default
        live_games = [
            game["game_id"]
            for game in schedule
            if game["status"] in ("In Progress", "Live")
        ]
        return live_games
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return []


def process_new_pitch(pitch_id: tuple, game_data: dict, predictor: PitchPredictor):
    """Processes a pitch if it matches outcome criteria."""
    game_pk, at_bat_index, pitch_index = pitch_id
    
    # 1. Identify the play and event
    all_plays = game_data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    current_play = next((p for p in all_plays if p.get('about', {}).get('atBatIndex') == at_bat_index), None)
    if not current_play: return

    # Check if this pitch ended the play
    play_events = current_play.get('playEvents', [])
    if not play_events: return
    last_pitch_event = next((e for e in reversed(play_events) if e.get('isPitch')), None)
    
    if not last_pitch_event or last_pitch_event.get('index') != pitch_index:
        return

    # Check outcomes
    result = current_play.get("result", {})
    event_type = result.get("eventType", "").lower()
    hit_data = current_play.get("hitData", {})
    launch_speed = hit_data.get("launchSpeed", 0)
    is_out = result.get("isOut", False)

    outcome_str = None
    if event_type == "strikeout":
        outcome_str = "strikeout"
    elif launch_speed >= BARREL_EV_THRESHOLD and not is_out:
        outcome_str = f"hard_hit_{int(launch_speed)}mph"

    if not outcome_str:
        return

    print(f"Analyzing interesting outcome: {outcome_str} in game {game_pk}")

    # 2. Extract features and run inference
    try:
        pitch_rows = extract_pitches_with_context(game_data)
        df = pd.DataFrame(pitch_rows)
        
        # Add basic contextual features (inning, count one-hot, etc.)
        df = add_contextual_features(df)
        
        # Hydrate with Baseline Tendencies
        # This replaces the need to calculate them from the single-game DF
        row = df[(df['at_bat_index'] == at_bat_index) & (df['pitch_index'] == pitch_index)].copy()
        if row.empty: return
        
        pitcher = row['pitcher'].values[0]
        balls = row['balls'].values[0]
        strikes = row['strikes'].values[0]
        
        # Hydrate tendencies if baseline is ready
        if baseline:
            # Global tendencies
            p_global = baseline['global'].get(pitcher, {})
            for col, val in p_global.items():
                row[col] = val
                
            # Count tendencies
            p_count = baseline['count'].get((pitcher, balls, strikes), {})
            for col, val in p_count.items():
                row[col] = val
            
            # Batter Count tendencies
            batter_id = row['batter_id'].values[0]
            b_count = baseline.get('batter_count', {}).get((batter_id, balls, strikes), {})
            for col, val in b_count.items():
                row[col] = val

            # League Count tendencies
            l_count = baseline.get('league_count', {}).get((balls, strikes), {})
            for col, val in l_count.items():
                row[col] = val

            # Out Pitch
            pitcher_id = row['pitcher_id'].values[0]
            row['primary_out_pitch'] = baseline.get('out_pitch', {}).get(pitcher_id, "Fastball")

            # Park ID
            venue_id = game_data.get('gameData', {}).get('venue', {}).get('id', 0)
            row['park_id'] = venue_id
        else:
            print("  Warning: Baseline tendencies not loaded. Expect surprises in predictions.")

        # 2. Run inference
        actual_pitch_code = row['pitch_type'].values[0]
        actual_pitch_family = _classify_pitch_family(actual_pitch_code)
        
        # predict_probabilities now handles its own feature alignment and encoding
        probabilities = predictor.predict_probabilities(row)
        surprisal = predictor.calculate_surprisal(actual_pitch_family, probabilities)
        
        # Log every single live prediction to monitor model calibration over time
        try:
            game_pk_val = game_pk
            pitcher_id_val = int(row['pitcher_id'].values[0])
            batter_id_val = int(row['batter_id'].values[0])
            play_id_val = last_pitch_event.get('playId') or current_play.get('playId', '')
            insert_live_prediction(
                game_pk=game_pk_val, play_id=play_id_val, 
                pitcher_id=pitcher_id_val, batter_id=batter_id_val, 
                actual_pitch_family=actual_pitch_family,
                probs=probabilities, surprisal=surprisal
            )
        except Exception as e:
            print(f"  Warning: Failed to log live prediction DB entry: {e}")

        # Determine Narrative based on situational context and tendencies
        expected_prob = probabilities.get(actual_pitch_family, 0)
        is_surprise_pitch = expected_prob < 0.15
        
        # Check pitch description to distinguish swinging from looking
        pitch_description = last_pitch_event.get('details', {}).get('description', '').lower()
        is_looking = "called strike" in pitch_description
        is_swinging = "swinging strike" in pitch_description or "foul tip" in pitch_description
        
        print(f"  Pitch: {actual_pitch_code}, Prob: {expected_prob:.2f}, Surprisal: {surprisal:.2f}, Desc: {pitch_description}")
        
        # --- Narrative Selection Logic ---
        tweet_logic = False
        narrative = ""
        
        if event_type == "strikeout":
            # 1. Unexpected pitch leads to strikeout (Frozen or Fooled)
            if is_surprise_pitch:
                if is_looking:
                    narrative = "🥶 Frozen!"
                else:
                    narrative = "🔀 Fooled him!"
                tweet_logic = True
            # 2. Dominance: Predictive fastball but still gets the whiff
            elif expected_prob > 0.8 and actual_pitch_family == "Fastball" and is_swinging:
                narrative = "😤 Pure Dominance."
                tweet_logic = True
            # 3. General high-surprisal strikeout fallback
            elif surprisal > SURPRISAL_THRESHOLD:
                narrative = "🤯 Unbelievable K!"
                tweet_logic = True
        elif launch_speed >= BARREL_EV_THRESHOLD and not is_out:
            # 3. Hitter hits a pitch they were statistically expecting
            if expected_prob > 0.4:
                narrative = "🎯 Sitting on it!"
                tweet_logic = True
            # 4. Hitter punishes an unconventional pitch
            elif surprisal > SURPRISAL_THRESHOLD:
                narrative = "💥 Punished!"
                tweet_logic = True

        if tweet_logic and event_type == "strikeout":
            # 3. Handle Context for the new Surprise Strikeout format
            pitcher_name = current_play.get('matchup', {}).get('pitcher', {}).get('fullName')
            batter_name = current_play.get('matchup', {}).get('batter', {}).get('fullName')
            actual_pitch_desc = last_pitch_event.get('details', {}).get('type', {}).get('description', actual_pitch_code)
            
            inning = row['inning'].values[0]
            half = row['half_inning'].values[0]
            inning_info = f"{'Top' if half == 'top' else 'Bottom'} {inning}"
            
            h_score = row['score_home'].values[0]
            a_score = row['score_away'].values[0]
            a_team = game_data.get('gameData', {}).get('teams', {}).get('away', {}).get('abbreviation', 'AWY')
            h_team = game_data.get('gameData', {}).get('teams', {}).get('home', {}).get('abbreviation', 'HOM')
            score_info = f"{a_team} {a_score}, {h_team} {h_score}"
            
            mob = row['men_on_base'].values[0] or "Empty"
            runners_info = mob.replace('_', ' ')
            
            outs = row['outs'].values[0]
            
            # Matchup tracking
            m_key = (game_pk, pitcher_id, batter_id)
            matchup_num = matchup_tracker.get(m_key, 0) + 1
            matchup_tracker[m_key] = matchup_num
            
            # Sequence: Map each pitch to "Family (Description)"
            # Limit to last 4 pitches and shorten descriptions to save character space
            sequence = []
            for e in play_events:
                if e.get('isPitch'):
                    p_code = e.get('details', {}).get('type', {}).get('code', 'UN')
                    p_desc = e.get('details', {}).get('type', {}).get('description', 'Unknown')
                    
                    # Shorten common long descriptions
                    p_desc = p_desc.replace('Four-Seam Fastball', '4-Seam') \
                                 .replace('Fastball', 'FB') \
                                 .replace('Changeup', 'CH') \
                                 .replace('Curveball', 'CU') \
                                 .replace('Slider', 'SL') \
                                 .replace('Sinker', 'SI') \
                                 .replace('Sweeper', 'ST') \
                                 .replace('Knuckle Curve', 'KC') \
                                 .replace('Splitter', 'FS') \
                                 .replace('Cutter', 'FC')
                                 
                    p_family = _classify_pitch_family(p_code)
                    sequence.append(f"{p_family} ({p_desc})")
            
            sequence = sequence[-4:] # Only show the last 4 pitches to avoid 280-char limit
            
            # Use playId from the pitch event if available, fallback to play level
            precise_play_id = last_pitch_event.get('playId') or current_play.get('playId')
            
            # Prepare tweet text without video link first
            tweet_text = format_surprise_strikeout_tweet(
                pitcher_name, batter_name, actual_pitch_desc, actual_pitch_family, expected_prob, is_swinging,
                inning_info, score_info, runners_info, outs, matchup_num, sequence,
                narrative=narrative, away_team=a_team, home_team=h_team
            )
            
            print("\n" + "="*50)
            print("🚀 [POSTING LIVE TWEET]")
            print(tweet_text)
            print("="*50 + "\n")
            post_tweet(tweet_text) # Enable for production

        elif tweet_logic: # Fallback for non-strikeout surprises (hard hits)
            pitcher_name = current_play.get('matchup', {}).get('pitcher', {}).get('fullName')
            batter_name = current_play.get('matchup', {}).get('batter', {}).get('fullName')
            actual_pitch_desc = last_pitch_event.get('details', {}).get('type', {}).get('description', actual_pitch_code)
            a_team = game_data.get('gameData', {}).get('teams', {}).get('away', {}).get('abbreviation', 'AWY')
            h_team = game_data.get('gameData', {}).get('teams', {}).get('home', {}).get('abbreviation', 'HOM')
            
            tweet_text = format_tweet(pitcher_name, batter_name, actual_pitch_desc, surprisal, outcome_str, away_team=a_team, home_team=h_team)
            tweet_text = f"{narrative}\n\n{tweet_text}"
            print(f"🚀 [QUEUED HARD HIT]: {narrative}")
            # Log it but maybe don't queue for video if we don't have video logic for hard hits yet
        else:
            print(f"  Skipping: Not a significant surprise/outcome combo.")

    except Exception as e:
        print(f"Error in model inference: {e}")
        import traceback
        traceback.print_exc()




def main():
    global baseline
    print("Starting MLB Live Game Tracker (Simulation Mode)...")
    processed_pitches.clear()
    create_live_predictions_table()
    try:
        baseline = joblib.load(BASELINE_PATH)
        # Ensure model paths are correct
        predictor = PitchPredictor(model_path=MODEL_PATH, encoder_path=ENCODER_PATH)
        print("Model and Baseline loaded.")
    except Exception as e:
        print(f"Error loading model: {e}. (Have you run src/train_model.py?)")
        # predictor = None # Allow running for debug, but here we probably want to stop
        return

    # --- Startup Skip Logic ---
    # To prevent spamming tweets for plays that already happened, 
    # we mark all existing pitches in live games as 'processed' upon launch.
    print("Initializing: Skipping already completed pitches in live games...")
    live_game_pks = get_live_game_pks()
    for game_pk in live_game_pks:
        try:
            game_data = statsapi.get("game", {"gamePk": game_pk})
            all_plays = game_data.get("liveData", {}).get("plays", {}).get("allPlays", [])
            for play in all_plays:
                at_bat_idx = play.get("about", {}).get("atBatIndex")
                for event in play.get("playEvents", []):
                    if event.get("isPitch"):
                        processed_pitches.add((game_pk, at_bat_idx, event.get("index")))
        except Exception as e:
            print(f"Error during initialization for game {game_pk}: {e}")
    
    print(f"Initialization complete. Monitoring {len(live_game_pks)} games.")

    while True:
        # Removed pending tweets check Queue

        live_game_pks = get_live_game_pks()
        if not live_game_pks:
            print(f"Waiting for live games... (Interval: {POLLING_INTERVAL_SECONDS}s)")
        else:
            for game_pk in live_game_pks:
                try:
                    game_data = statsapi.get("game", {"gamePk": game_pk})
                    all_plays = game_data.get("liveData", {}).get("plays", {}).get("allPlays", [])
                    for play in all_plays:
                        at_bat_idx = play.get("about", {}).get("atBatIndex")
                        for event in play.get("playEvents", []):
                            if not event.get("isPitch"): continue
                            pitch_idx = event.get("index")
                            pitch_id = (game_pk, at_bat_idx, pitch_idx)
                            if pitch_id not in processed_pitches:
                                processed_pitches.add(pitch_id)
                                process_new_pitch(pitch_id, game_data, predictor)
                except Exception as e:
                    print(f"Error in game {game_pk}: {e}")

        time.sleep(POLLING_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
