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
    apply_baseline_to_df,
    _classify_pitch_family
)
from src.inference import PitchPredictor
from src.bot import post_tweet, format_tweet, format_surprise_strikeout_tweet
from src.database import create_live_predictions_table, insert_live_prediction
from src.visualization import generate_pitch_infographic
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

def evaluate_pitch_narrative(event_type: str, is_surprise: bool, is_looking: bool, is_swinging: bool, 
                             expected_prob: float, actual_family: str, surprisal: float) -> tuple[bool, str]:
    """Determines if a pitch warrants a tweet and selects the appropriate narrative prefix."""
    tweet_logic = False
    narrative = ""
    
    if event_type == "strikeout":
        if is_surprise:
            narrative = "🥶 Frozen!" if is_looking else "🔀 Fooled him!"
            tweet_logic = True
        elif expected_prob > 0.8 and actual_family == "Fastball" and is_swinging:
            narrative = "😤 Pure Dominance."
            tweet_logic = True
        elif surprisal > SURPRISAL_THRESHOLD:
            narrative = "🤯 Unbelievable K!"
            tweet_logic = True
            
    elif event_type.startswith("hard_hit"):
        if expected_prob > 0.4:
            narrative = "🎯 Sitting on it!"
            tweet_logic = True
        elif surprisal > SURPRISAL_THRESHOLD:
            narrative = "💥 Punished!"
            tweet_logic = True
            
    return tweet_logic, narrative

def build_pitch_sequence(play_events: list) -> list:
    """Builds the sequence visualization dictionary from raw MLB API play events."""
    sequence = []
    for e in play_events:
        if e.get('isPitch'):
            p_code = e.get('details', {}).get('type', {}).get('code', 'UN')
            p_desc = e.get('details', {}).get('type', {}).get('description', 'Unknown')
            e_pitch_data = e.get('pitchData', {})
            sequence.append({
                "pitch_type_code": p_code,
                "pitch_type_desc": p_desc,
                "pitch_family": _classify_pitch_family(p_code),
                "call": e.get('details', {}).get('description', ''),
                "pX": (e_pitch_data.get('coordinates') or {}).get('pX'),
                "pZ": (e_pitch_data.get('coordinates') or {}).get('pZ'),
                "pitch_number": e.get('index')
            })
    return sequence

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
        
        # Identify the exact pitch
        row = df[(df['at_bat_index'] == at_bat_index) & (df['pitch_index'] == pitch_index)].copy()
        if row.empty: return
        
        # Add context required before feature hydration
        venue_id = game_data.get('gameData', {}).get('venue', {}).get('id', 0)
        row['park_id'] = venue_id
        
        # Hydrate tendencies if baseline is ready
        if baseline:
            row = apply_baseline_to_df(row, baseline, is_train=False)
        else:
            print("  Warning: Baseline tendencies not loaded. Expect surprises in predictions.")
            row = add_contextual_features(row)

        # 2. Run inference
        actual_pitch_code = row['pitch_type'].values[0]
        actual_pitch_family = _classify_pitch_family(actual_pitch_code)
        
        # predict_probabilities now handles its own feature alignment and encoding
        probabilities = predictor.predict_probabilities(row)
        surprisal = predictor.calculate_surprisal(actual_pitch_family, probabilities)
        
        pitcher_id = int(row['pitcher_id'].values[0])
        batter_id = int(row['batter_id'].values[0])
        
        # Log every single live prediction to monitor model calibration over time
        try:
            play_id_val = last_pitch_event.get('playId') or current_play.get('playId', '')
            insert_live_prediction(
                game_pk=game_pk, play_id=play_id_val, 
                pitcher_id=pitcher_id, batter_id=batter_id, 
                actual_pitch_family=actual_pitch_family,
                probs=probabilities, surprisal=surprisal
            )
        except Exception as e:
            print(f"  Warning: Failed to log live prediction DB entry: {e}")

        # Determine Narrative based on situational context and tendencies
        expected_prob = probabilities.get(actual_pitch_family, 0)
        
        # Squelch tweets for 'Other' pitch families (not handled by model)
        if actual_pitch_family == "Other":
            print(f"  Skipping: Pitch classified as 'Other' ({actual_pitch_code}). Model cannot provide reliable probability.")
            return

        is_surprise_pitch = expected_prob < 0.15
        
        # Check pitch description to distinguish swinging from looking
        pitch_description = last_pitch_event.get('details', {}).get('description', '').lower()
        is_looking = "called strike" in pitch_description
        is_swinging = "swinging strike" in pitch_description or "foul tip" in pitch_description
        
        print(f"  Pitch: {actual_pitch_code}, Prob: {expected_prob:.2f}, Surprisal: {surprisal:.2f}, Desc: {pitch_description}")
        
        # --- Narrative Selection Logic ---
        tweet_logic, narrative = evaluate_pitch_narrative(
            event_type=outcome_str if outcome_str.startswith("hard_hit") else event_type, 
            is_surprise=is_surprise_pitch, 
            is_looking=is_looking, 
            is_swinging=is_swinging, 
            expected_prob=expected_prob, 
            actual_family=actual_pitch_family, 
            surprisal=surprisal
        )

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
            
            sequence = build_pitch_sequence(play_events)
            
            # We no longer handle truncation logic, the infographic handles context entirely!
            p_hand = row['pitcher_hand'].values[0] if 'pitcher_hand' in row.columns else ""
            b_side = row['batter_side'].values[0] if 'batter_side' in row.columns else ""

            # Prepare tweet text
            tweet_text = format_surprise_strikeout_tweet(
                pitcher_name, batter_name, actual_pitch_desc, actual_pitch_family, expected_prob, is_swinging,
                narrative=narrative, away_team=a_team, home_team=h_team,
                pitcher_hand=p_hand, batter_side=b_side
            )
            
            # Generate the infographic image
            os.makedirs("output", exist_ok=True)
            image_path = f"output/live_tweet_{game_pk}_{at_bat_index}.png"
            print(f"  Generating infographic: {image_path}")
            
            pitch_data_dict = row.to_dict('records')[0]
            generate_pitch_infographic(
                pitch_data=pitch_data_dict,
                probs=probabilities,
                surprisal=surprisal,
                sequence=sequence,
                output_path=image_path
            )
            
            print("\n" + "="*50)
            print("🚀 [POSTING LIVE TWEET]")
            print(tweet_text)
            print("="*50 + "\n")
            post_tweet(tweet_text, image_path=image_path) # Enable for production

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
