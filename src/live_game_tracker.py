"""
Monitors live MLB games to identify surprising pitches or hard-hit balls
on expected pitches, laying the foundation for a Twitter bot.
"""
import time
import statsapi
import pandas as pd

from src.features import build_pitch_features, add_pitcher_tendency
from src.inference import PitchPredictor

# --- Constants ---
# How often to poll the MLB API for game updates.
POLLING_INTERVAL_SECONDS = 20
# Surprisal threshold for identifying an "unexpected" pitch.
SURPRISAL_THRESHOLD = 3.5
# Exit velocity threshold (mph) for a "hard-hit" ball.
HARD_HIT_THRESHOLD = 100.0
# Surprisal threshold for a pitch being "expected" by the batter.
EXPECTED_PITCH_THRESHOLD = 1.0


# --- State ---
# A set to store unique identifiers of pitches we've already processed.
# The identifier is a tuple: (game_id, at_bat_index, pitch_index)
processed_pitches = set()


def get_live_game_pks():
    """
    Fetches the schedule for the current day and returns a list of gamePks
    for games that are currently live ("In Progress" or "Live").
    """
    try:
        # You can add a date string to statsapi.schedule() for testing specific days.
        schedule = statsapi.schedule()
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
    """
    Takes a DataFrame of pitch features and prepares it for the model by:
    1. Dropping columns the model wasn't trained on.
    2. Adding any missing feature columns (with a value of 0).
    3. Reordering columns to match the model's expected input.
    """
    # Drop columns that are not features for the model
    cols_to_drop = [
        'batter', 'pitcher', 'pitch_type', 'call', 'half_inning',
        'score_home', 'score_away', 'prev_pitch_type_in_ab'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Ensure all columns the model expects are present
    for col in predictor.model.feature_names:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match the model's training order
    return df[predictor.model.feature_names]


def check_for_tweetable_event(play: dict, pitch_event: dict, surprisal: float, predictor: PitchPredictor):
    """
    Checks if a pitch event is "tweetable" based on surprisal and hit data.
    """
    # Case 1: Pitcher fools the batter with an unexpected pitch.
    if surprisal > SURPRISAL_THRESHOLD:
        pitch_details = pitch_event.get('details', {})
        pitch_type = pitch_details.get('type', {}).get('description')
        batter_name = play.get('matchup', {}).get('batter', {}).get('fullName')
        pitcher_name = play.get('matchdcup', {}).get('pitcher', {}).get('fullName')
        
        message = (
            f"🤯 SURPRISING PITCH! {pitcher_name} fools {batter_name} with a {pitch_type}."
            f"\nPitch surprisal: {surprisal:.2f} bits."
        )
        print(f"TWEET: {message}")
        # send_tweet(message) # Placeholder for actual tweet call

    # Case 2: Batter correctly guesses the pitch and hits it hard.
    play_result = play.get('result', {})
    if play.get('about', {}).get('isComplete') and play_result.get('eventType'):
        hit_data = play.get('hitData', {})
        launch_speed = hit_data.get('launchSpeed')

        if launch_speed and launch_speed > HARD_HIT_THRESHOLD and surprisal < EXPECTED_PITCH_THRESHOLD:
            batter_name = play.get('matchup', {}).get('batter', {}).get('fullName')
            pitcher_name = play.get('matchup', {}).get('pitcher', {}).get('fullName')
            
            message = (
                f"💥 LOCKED IN! {batter_name} was all over that pitch from {pitcher_name}."
                f"\n{launch_speed} mph exit velocity on a pitch with only {surprisal:.2f} bits of surprisal."
            )
            print(f"TWEET: {message}")
            # send_tweet(message) # Placeholder for actual tweet call


def process_new_pitch(pitch_id: tuple, game_data: dict, predictor: PitchPredictor):
    """
    Processes a single new pitch event from start to finish.
    """
    game_pk, at_bat_index, pitch_index = pitch_id
    print(f"New pitch detected in game {game_pk} (At-bat: {at_bat_index}, Pitch: {pitch_index})")

    # 1. Generate base features and tendency features from the whole game context.
    features_df = build_pitch_features(game_data)
    tendency_list = add_pitcher_tendency(game_data)

    if features_df.empty or not tendency_list:
        print("Feature generation failed or returned empty data. Skipping.")
        return

    # 2. Combine feature sets.
    tendency_df = pd.DataFrame(tendency_list)
    if len(features_df) == len(tendency_df):
        tendency_cols = [
            "pitcher_tendency_primary", "pitcher_tendency_primary_pct",
            "pitcher_tendency_fastball_pct", "pitcher_tendency_breaking_pct",
            "pitcher_tendency_offspeed_pct", "pitcher_tendency_pitches_used",
        ]
        for col in tendency_cols:
            if col in tendency_df:
                features_df[col] = tendency_df[col]
    else:
        print("Warning: Mismatch in feature row counts. Tendency features may be incorrect.")

    # 3. Isolate the features for just the new pitch (the last row).
    latest_pitch_features = features_df.iloc[-1:].copy()
    actual_pitch_type = latest_pitch_features['pitch_type'].values[0]

    # 4. Prepare the feature row for the model.
    model_input_df = prepare_features_for_model(latest_pitch_features, predictor)

    # 5. Run inference to get pitch probabilities.
    try:
        probabilities = predictor.predict_probabilities(model_input_df)
        surprisal = predictor.calculate_surprisal(actual_pitch_type, probabilities)

        print(f"  Actual: {actual_pitch_type}, Surprisal: {surprisal:.2f}")

        # 6. Check if this event is interesting enough to tweet about.
        all_plays = game_data.get("liveData", {}).get("plays", {}).get("allPlays", [])
        current_play = next((p for p in all_plays if p.get('about', {}).get('atBatIndex') == at_bat_index), None)
        current_pitch_event = next((e for e in current_play['playEvents'] if e.get('isPitch') and e.get('index') == pitch_index), None)
        
        if current_play and current_pitch_event:
            check_for_tweetable_event(current_play, current_pitch_event, surprisal, predictor)

    except Exception as e:
        print(f"Error during prediction or event checking: {e}")
        print("Features used for prediction:")
        print(model_input_df.to_string())


def main():
    """ Main monitoring loop. """
    print("Starting MLB Live Game Tracker...")
    try:
        predictor = PitchPredictor()
        print("PitchPredictor model loaded successfully.")
    except Exception as e:
        print(f"Fatal: Could not load the prediction model. {e}")
        return

    while True:
        live_game_pks = get_live_game_pks()

        if not live_game_pks:
            print(f"No live games found. Waiting for {POLLING_INTERVAL_SECONDS}s...")
        else:
            print(f"Found {len(live_game_pks)} live game(s): {live_game_pks}")

        for game_pk in live_game_pks:
            try:
                game_data = statsapi.get("game", {"gamePk": game_pk})
                all_plays = game_data.get("liveData", {}).get("plays", {}).get("allPlays", [])

                for play in all_plays:
                    at_bat_index = play.get("about", {}).get("atBatIndex")
                    if "playEvents" not in play:
                        continue

                    for event in play["playEvents"]:
                        if not event.get("isPitch"):
                            continue

                        pitch_index = event.get("index")
                        pitch_id = (game_pk, at_bat_index, pitch_index)

                        if pitch_id not in processed_pitches:
                            processed_pitches.add(pitch_id)
                            process_new_pitch(pitch_id, game_data, predictor)

            except Exception as e:
                print(f"Error processing game {game_pk}: {e}")

        time.sleep(POLLING_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
