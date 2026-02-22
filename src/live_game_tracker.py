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
from dotenv import load_dotenv

from src.features import extract_pitches_with_context, add_contextual_features, add_global_pitcher_tendencies
from src.inference import PitchPredictor
from src.bot import post_tweet, format_tweet

load_dotenv()

# --- Constants ---
POLLING_INTERVAL_SECONDS = 30
SURPRISAL_THRESHOLD = 2.5 # Adjusted for SOs/Barrels
BARREL_EV_THRESHOLD = 98.0

# --- State ---
processed_pitches = set()

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

def prepare_features_for_model(df: pd.DataFrame, predictor: PitchPredictor) -> pd.DataFrame:
    """Prepares a DataFrame row for the XGBoost model."""
    # Ensure all columns the model expects are present
    for col in predictor.model.feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[predictor.model.feature_names]

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

    # 2. Extract basic info and tweet directly
    try:
        pitcher_name = current_play.get('matchup', {}).get('pitcher', {}).get('fullName')
        batter_name = current_play.get('matchup', {}).get('batter', {}).get('fullName')
        actual_pitch_type = last_pitch_event.get('details', {}).get('type', {}).get('description', 'Pitch')
        
        # Try to get playId for video lookup
        play_id = current_play.get('playId')
        video_url = None
        if play_id:
            # We wait a few seconds as the content API might be slightly behind the live data
            time.sleep(2) 
            video_url = get_video_link(game_pk, play_id)

        tweet_text = format_tweet(pitcher_name, batter_name, actual_pitch_type, 0.0, outcome_str)
        
        if video_url:
            tweet_text += f"\n\n📺 Video: {video_url}"
        
        post_tweet(tweet_text)

    except Exception as e:
        print(f"Error posting tweet: {e}")

def main():
    print("Starting MLB Live Game Tracker (Outcome-Focused Build)...")
    try:
        # Ensure model paths are correct
        predictor = PitchPredictor(model_path='models/pitch_classifier.pkl', encoder_path='models/encoder.pkl')
        print("Model loaded.")
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
