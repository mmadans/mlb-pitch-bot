"""
Monitors live MLB games to identify surprising pitches with specific outcomes.
Analyzes only:
1. Strikeouts
2. Barreled (Hard Hit) balls that do not result in an out.
"""
import time
import statsapi
import pandas as pd
import os
import joblib
from dotenv import load_dotenv

from src.features import (
    extract_pitches_with_context, 
    add_contextual_features, 
    add_global_pitcher_tendencies
)
from src.inference import PitchPredictor
from src.bot import post_tweet, format_tweet
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


def get_video_link(game_pk: int, play_id: str) -> str | None:
    """
    Fetches the official MLB video link for a specific play.
    Note: Highlights usually take 2-5 minutes to appear after the play.
    """
    try:
        content = statsapi.get("game_content", {"gamePk": game_pk})
        highlights = content.get("highlights", {}).get("highlights", {}).get("items", [])
        
        # Look for a highlight that matches our playId
        for item in highlights:
            if item.get("playId") == play_id:
                # Find the best quality mp4
                playbacks = item.get("playbacks", [])
                # Prefer 720p or 1080p mp4
                best_url = None
                for pb in playbacks:
                    if "mp4" in pb.get("url", ""):
                        url = pb.get("url")
                        if "1200K" in url or "2500K" in url: # Decent qualities
                            best_url = url
                return best_url or (playbacks[0].get("url") if playbacks else None)
    except Exception as e:
        print(f"Error fetching video: {e}")
    return None

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
        else:
            print("  Warning: Baseline tendencies not loaded. Expect surprises in predictions.")

        # 2. Run inference
        actual_pitch_code = row['pitch_type'].values[0]
        # predict_probabilities now handles its own feature alignment and encoding
        probabilities = predictor.predict_probabilities(row)
        surprisal = predictor.calculate_surprisal(actual_pitch_code, probabilities)

        # Determine Narrative based on situational context and tendencies
        expected_prob = probabilities.get(actual_pitch_code, 0)
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
                    narrative = "🥶 Frozen! Caught him looking with a pitch he wasn't expecting."
                else:
                    narrative = "🔀 Fooled him! Went against tendency to get the swinging K."
                tweet_logic = True
            # 2. General high-surprisal strikeout fallback
            elif surprisal > SURPRISAL_THRESHOLD:
                narrative = "🤯 Unbelievable K! High-surprisal pitch."
                tweet_logic = True
        elif launch_speed >= BARREL_EV_THRESHOLD and not is_out:
            # 3. Hitter hits a pitch they were statistically expecting
            if expected_prob > 0.4:
                narrative = "🎯 Sitting on it! Hitter was waiting for that specific tendency."
                tweet_logic = True
            # 4. Hitter punishes an unconventional pitch
            elif surprisal > SURPRISAL_THRESHOLD:
                narrative = "💥 Punished! Unconventional pitch didn't work."
                tweet_logic = True

        if tweet_logic:
            pitcher_name = current_play.get('matchup', {}).get('pitcher', {}).get('fullName')
            batter_name = current_play.get('matchup', {}).get('batter', {}).get('fullName')
            actual_pitch_desc = last_pitch_event.get('details', {}).get('type', {}).get('description', actual_pitch_code)
            
            video_url = None
            play_id = current_play.get('playId')
            if play_id:
                time.sleep(2)
                video_url = get_video_link(game_pk, play_id)

            tweet_text = format_tweet(pitcher_name, batter_name, actual_pitch_desc, surprisal, outcome_str)
            tweet_text = f"{narrative}\n\n{tweet_text}"
            
            if video_url:
                tweet_text += f"\n\n📺 Video: {video_url}"
            
            print("\n" + "="*50)
            print("🚀 [SIMULATED TWEET]")
            print(tweet_text)
            print("="*50 + "\n")
            # post_tweet(tweet_text) # Disabled for simulation
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
    try:
        baseline = joblib.load(BASELINE_PATH)
        # Ensure model paths are correct
        predictor = PitchPredictor(model_path=MODEL_PATH, encoder_path=ENCODER_PATH)
        print("Model and Baseline loaded.")
    except Exception as e:
        print(f"Error loading model: {e}. (Have you run src/train_model.py?)")
        # predictor = None # Allow running for debug, but here we probably want to stop
        return

    while True:
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
