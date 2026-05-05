"""
Monitors live MLB games to identify surprising pitches with specific outcomes.
Analyzes only:
1. Strikeouts
2. Barreled (Hard Hit) balls that do not result in an out.
"""

from datetime import datetime
import json
import time
import logging
import statsapi
import pandas as pd
import os
import joblib
from dotenv import load_dotenv

log = logging.getLogger(__name__)

from src.data.api_extractors import extract_pitches_with_context, _classify_pitch_family
from src.features.features import add_contextual_features
from src.model.inference import PitchPredictor
from src.bot.bot import post_tweet, format_surprise_strikeout_tweet
from src.data.database import create_live_predictions_table, insert_live_prediction
from src.bot.visualization import generate_pitch_infographic
from src.constants import (
    ROOT,
    POLLING_INTERVAL_SECONDS,
    SURPRISAL_THRESHOLD,
    BARREL_EV_THRESHOLD,
    MODEL_PATH,
    TARGET_ENCODER_PATH,
    CATEGORICAL_ENCODER_PATH,
    BASELINE_PATH,
)

_PROCESSED_CACHE_PATH = str(ROOT / "data/processed_pitches_cache.json")

load_dotenv()

import wandb

# Minimum number of historical pitches a pitcher must have in the baseline before
# we trust the model's output. Below this threshold the global tendency features
# are too noisy and the model falls back to league-average behaviour, which
# produces misleading predictions (e.g. 99 % Fastball for a breaking-ball pitcher).
MIN_PITCHER_SAMPLE = 75

# --- State ---
processed_pitches = set()
matchup_tracker = {}  # (game_pk, pitcher_id, batter_id) -> count
pending_tweets = []  # list of dicts: {'post_at': timestamp, 'text': string, 'game_pk': int, 'play_id': str}
baseline = None  # Global baseline to be loaded in main()

# --- Session counters (reset each run in main()) ---
_session_predictions = 0
_session_tweets = 0
_session_games: set = set()
_pitch_type_counts: dict = {"Fastball": 0, "Breaking": 0, "Offspeed": 0}
_error_counts: dict = {"Fastball": 0, "Breaking": 0, "Offspeed": 0}


def _load_processed_cache() -> set:
    """Return today's processed pitch IDs persisted from a previous run."""
    try:
        with open(_PROCESSED_CACHE_PATH) as f:
            data = json.load(f)
        if data.get("date") != datetime.now().date().isoformat():
            return set()  # stale — new day, start fresh
        return {tuple(p) for p in data.get("pitches", [])}
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return set()


def _save_processed_cache():
    """Persist today's processed pitch IDs so restarts don't re-tweet old pitches."""
    try:
        os.makedirs(os.path.dirname(_PROCESSED_CACHE_PATH), exist_ok=True)
        with open(_PROCESSED_CACHE_PATH, "w") as f:
            json.dump(
                {
                    "date": datetime.now().date().isoformat(),
                    "pitches": [list(p) for p in processed_pitches],
                },
                f,
            )
    except Exception as e:
        log.warning("Failed to save processed pitches cache: %s", e)


def get_live_game_pks():
    """Fetches gamePks for games that are currently live."""
    try:
        schedule = statsapi.schedule()  # Use today's date by default
        live_games = [
            game["game_id"]
            for game in schedule
            if game["status"] in ("In Progress", "Live")
        ]
        return live_games
    except Exception as e:
        log.error("Error fetching schedule: %s", e)
        return []


def evaluate_pitch_narrative(
    event_type: str,
    is_surprise: bool,
    is_looking: bool,
    is_swinging: bool,
    expected_prob: float,
    actual_family: str,
    surprisal: float,
) -> tuple[bool, str]:
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
        if e.get("isPitch"):
            p_code = e.get("details", {}).get("type", {}).get("code", "UN")
            p_desc = e.get("details", {}).get("type", {}).get("description", "Unknown")
            e_pitch_data = e.get("pitchData", {})
            sequence.append(
                {
                    "pitch_type_code": p_code,
                    "pitch_type_desc": p_desc,
                    "pitch_family": _classify_pitch_family(p_code),
                    "call": e.get("details", {}).get("description", ""),
                    "pX": (e_pitch_data.get("coordinates") or {}).get("pX"),
                    "pZ": (e_pitch_data.get("coordinates") or {}).get("pZ"),
                    "pitch_number": e.get("index"),
                }
            )
    return sequence


def _identify_outcome(game_data: dict, at_bat_index: int, pitch_index: int):
    """Returns (current_play, last_pitch_event, play_events, event_type, outcome_str) or None."""
    all_plays = game_data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    current_play = next(
        (p for p in all_plays if p.get("about", {}).get("atBatIndex") == at_bat_index),
        None,
    )
    if not current_play:
        return None
    play_events = current_play.get("playEvents", [])
    last_pitch_event = next(
        (e for e in reversed(play_events) if e.get("isPitch")), None
    )
    if not last_pitch_event or last_pitch_event.get("index") != pitch_index:
        return None
    result = current_play.get("result", {})
    event_type = result.get("eventType", "").lower()
    hit_data = current_play.get("hitData", {})
    launch_speed = hit_data.get("launchSpeed", 0)
    outcome_str = None
    if event_type == "strikeout":
        outcome_str = "strikeout"
    elif launch_speed >= BARREL_EV_THRESHOLD and not result.get("isOut", False):
        outcome_str = f"hard_hit_{int(launch_speed)}mph"
    if not outcome_str:
        return None
    return current_play, last_pitch_event, play_events, event_type, outcome_str


def _run_inference(row, game_data: dict, predictor: PitchPredictor):
    """Hydrates the pitch row and runs model inference. Returns (probs, surprisal, family, hydrated_row)."""
    actual_pitch_code = row["pitch_type"].values[0]
    if baseline:
        return predictor.hydrate_and_predict(row, baseline)
    log.warning("Baseline tendencies not loaded — predictions will be low-quality.")
    row = add_contextual_features(row)
    actual_pitch_family = _classify_pitch_family(actual_pitch_code)
    probabilities = predictor.predict_probabilities(row)
    surprisal = predictor.calculate_surprisal(actual_pitch_family, probabilities)
    return probabilities, surprisal, actual_pitch_family, row


def _get_sample_sizes(hydrated_row) -> tuple:
    """Returns (pitcher_sample_n, count_sample_n) integers or None if missing."""

    def _safe_int(col):
        if col not in hydrated_row.columns:
            return None
        val = hydrated_row[col].iloc[0]
        return int(val) if pd.notna(val) else None

    return _safe_int("tendency_total_pitches"), _safe_int(
        "tendency_count_total_pitches"
    )


def _log_prediction_to_db(
    game_pk,
    current_play,
    last_pitch_event,
    pitcher_id,
    batter_id,
    actual_pitch_family,
    probabilities,
    surprisal,
    pitcher_sample_n,
    count_sample_n,
    balls_val,
    strikes_val,
):
    try:
        play_id_val = last_pitch_event.get("playId") or current_play.get("playId", "")
        insert_live_prediction(
            game_pk=game_pk,
            play_id=play_id_val,
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            actual_pitch_family=actual_pitch_family,
            probs=probabilities,
            surprisal=surprisal,
            pitcher_sample_n=pitcher_sample_n,
            count_sample_n=count_sample_n,
            balls=balls_val,
            strikes=strikes_val,
        )
    except Exception as e:
        log.warning("Failed to log live prediction to DB: %s", e)


def _log_prediction_to_wandb(
    game_pk,
    actual_pitch_family,
    probabilities,
    surprisal,
    pitcher_sample_n,
    count_sample_n,
    outcome_str,
):
    global \
        _session_predictions, \
        _session_games, \
        _pitch_type_counts, \
        _error_counts, \
        _session_tweets
    try:
        predicted_family = max(probabilities, key=probabilities.get)
        is_correct = int(predicted_family == actual_pitch_family)
        _session_predictions += 1
        _session_games.add(game_pk)
        if actual_pitch_family in _pitch_type_counts:
            _pitch_type_counts[actual_pitch_family] += 1
        if not is_correct and actual_pitch_family in _error_counts:
            _error_counts[actual_pitch_family] += 1
        wandb.log(
            {
                "surprisal": surprisal if surprisal != float("inf") else None,
                "prob_fastball": probabilities.get("Fastball", 0),
                "prob_breaking": probabilities.get("Breaking", 0),
                "prob_offspeed": probabilities.get("Offspeed", 0),
                "actual_family": actual_pitch_family,
                "predicted_family": predicted_family,
                "correct": is_correct,
                "error": 1 - is_correct,
                "cumulative_predictions": _session_predictions,
                "cumulative_tweets": _session_tweets,
                "pitcher_sample_n": pitcher_sample_n,
                "count_sample_n": count_sample_n,
                "outcome": outcome_str,
                "game_pk": game_pk,
            }
        )
    except Exception as e:
        log.warning("W&B log failed: %s", e)


def _post_strikeout_tweet(
    game_pk,
    at_bat_index,
    pitcher_id,
    batter_id,
    current_play,
    last_pitch_event,
    play_events,
    row,
    hydrated_row,
    game_data,
    probabilities,
    surprisal,
    actual_pitch_family,
    actual_pitch_code,
    expected_prob,
    is_swinging,
    narrative,
):
    global _session_tweets
    pitcher_name = current_play.get("matchup", {}).get("pitcher", {}).get("fullName")
    batter_name = current_play.get("matchup", {}).get("batter", {}).get("fullName")
    actual_pitch_desc = (
        last_pitch_event.get("details", {})
        .get("type", {})
        .get("description", actual_pitch_code)
    )
    a_team = (
        game_data.get("gameData", {})
        .get("teams", {})
        .get("away", {})
        .get("abbreviation", "AWY")
    )
    h_team = (
        game_data.get("gameData", {})
        .get("teams", {})
        .get("home", {})
        .get("abbreviation", "HOM")
    )

    m_key = (game_pk, pitcher_id, batter_id)
    matchup_tracker[m_key] = matchup_tracker.get(m_key, 0) + 1

    p_hand = row["pitcher_hand"].values[0] if "pitcher_hand" in row.columns else ""
    b_side = row["batter_side"].values[0] if "batter_side" in row.columns else ""
    tweet_text = format_surprise_strikeout_tweet(
        pitcher_name,
        batter_name,
        actual_pitch_desc,
        actual_pitch_family,
        expected_prob,
        is_swinging,
        narrative=narrative,
        away_team=a_team,
        home_team=h_team,
        pitcher_hand=p_hand,
        batter_side=b_side,
    )

    os.makedirs("output", exist_ok=True)
    image_path = f"output/live_tweet_{game_pk}_{at_bat_index}.png"
    print(f"  Generating infographic: {image_path}")
    pitch_data_dict = hydrated_row.to_dict("records")[0]
    pitch_data_dict["away_team"] = a_team
    pitch_data_dict["home_team"] = h_team
    generate_pitch_infographic(
        pitch_data=pitch_data_dict,
        probs=probabilities,
        surprisal=surprisal,
        sequence=build_pitch_sequence(play_events),
        output_path=image_path,
    )

    print("\n" + "=" * 50)
    print("🚀 [POSTING LIVE TWEET]")
    print(tweet_text)
    print("=" * 50 + "\n")
    post_tweet(tweet_text, image_path=image_path)
    if os.path.exists(image_path):
        os.remove(image_path)
    _session_tweets += 1


def process_new_pitch(pitch_id: tuple, game_data: dict, predictor: PitchPredictor):
    """Processes a pitch if it matches outcome criteria."""
    game_pk, at_bat_index, pitch_index = pitch_id

    outcome = _identify_outcome(game_data, at_bat_index, pitch_index)
    if outcome is None:
        return
    current_play, last_pitch_event, play_events, event_type, outcome_str = outcome
    print(f"Analyzing interesting outcome: {outcome_str} in game {game_pk}")

    try:
        pitch_rows = extract_pitches_with_context(game_data)
        row = pd.DataFrame(pitch_rows)
        row = row[
            (row["at_bat_index"] == at_bat_index) & (row["pitch_index"] == pitch_index)
        ].copy()
        if row.empty:
            return
        row["park_id"] = game_data.get("gameData", {}).get("venue", {}).get("id", 0)
        actual_pitch_code = row["pitch_type"].values[0]

        probabilities, surprisal, actual_pitch_family, hydrated_row = _run_inference(
            row, game_data, predictor
        )

        pitcher_id = int(row["pitcher_id"].values[0])
        batter_id = int(row["batter_id"].values[0])
        balls_val = (
            int(row["balls"].values[0])
            if "balls" in row.columns and pd.notna(row["balls"].values[0])
            else None
        )
        strikes_val = (
            int(row["strikes"].values[0])
            if "strikes" in row.columns and pd.notna(row["strikes"].values[0])
            else None
        )

        pitcher_sample_n, count_sample_n = _get_sample_sizes(hydrated_row)
        if pitcher_sample_n is None or pitcher_sample_n < MIN_PITCHER_SAMPLE:
            print(
                f"  Skipping pitcher {pitcher_id}: insufficient sample ({pitcher_sample_n}, need {MIN_PITCHER_SAMPLE})."
            )
            return

        _log_prediction_to_db(
            game_pk,
            current_play,
            last_pitch_event,
            pitcher_id,
            batter_id,
            actual_pitch_family,
            probabilities,
            surprisal,
            pitcher_sample_n,
            count_sample_n,
            balls_val,
            strikes_val,
        )
        _log_prediction_to_wandb(
            game_pk,
            actual_pitch_family,
            probabilities,
            surprisal,
            pitcher_sample_n,
            count_sample_n,
            outcome_str,
        )

        if actual_pitch_family == "Other":
            print(
                f"  Skipping: 'Other' pitch ({actual_pitch_code}) — model has no reliable probability."
            )
            return

        expected_prob = probabilities.get(actual_pitch_family, 0)
        pitch_description = (
            last_pitch_event.get("details", {}).get("description", "").lower()
        )
        is_looking = "called strike" in pitch_description
        is_swinging = (
            "swinging strike" in pitch_description or "foul tip" in pitch_description
        )
        print(
            f"  Pitch: {actual_pitch_code}, Prob: {expected_prob:.2f}, Surprisal: {surprisal:.2f}"
        )

        tweet_logic, narrative = evaluate_pitch_narrative(
            event_type=outcome_str
            if outcome_str.startswith("hard_hit")
            else event_type,
            is_surprise=expected_prob < 0.15,
            is_looking=is_looking,
            is_swinging=is_swinging,
            expected_prob=expected_prob,
            actual_family=actual_pitch_family,
            surprisal=surprisal,
        )

        if tweet_logic and event_type == "strikeout":
            _post_strikeout_tweet(
                game_pk,
                at_bat_index,
                pitcher_id,
                batter_id,
                current_play,
                last_pitch_event,
                play_events,
                row,
                hydrated_row,
                game_data,
                probabilities,
                surprisal,
                actual_pitch_family,
                actual_pitch_code,
                expected_prob,
                is_swinging,
                narrative,
            )
        elif tweet_logic:
            print(
                f"  Hard-hit narrative generated but posting not yet implemented: {narrative}"
            )
        else:
            print("  Skipping: Not a significant surprise/outcome combo.")

    except Exception:
        log.exception("Error in model inference for pitch %s", pitch_id)


def main():
    global \
        baseline, \
        _session_predictions, \
        _session_tweets, \
        _session_games, \
        _pitch_type_counts, \
        _error_counts
    print("Starting MLB Live Game Tracker (Simulation Mode)...")
    processed_pitches.clear()
    processed_pitches.update(_load_processed_cache())
    print(f"Restored {len(processed_pitches)} processed pitches from cache.")
    _session_predictions = 0
    _session_tweets = 0
    _session_games = set()
    _pitch_type_counts = {"Fastball": 0, "Breaking": 0, "Offspeed": 0}
    _error_counts = {"Fastball": 0, "Breaking": 0, "Offspeed": 0}
    create_live_predictions_table()

    # Initialise W&B run for today's game session
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "mlb-pitch-bot"),
        name=f"live-{datetime.now().strftime('%Y-%m-%d')}",
        job_type="live-tracking",
        config={
            "surprisal_threshold": SURPRISAL_THRESHOLD,
            "min_pitcher_sample": MIN_PITCHER_SAMPLE,
            "polling_interval_s": POLLING_INTERVAL_SECONDS,
        },
        resume="allow",  # safe to restart mid-game without creating duplicate runs
    )

    try:
        baseline = joblib.load(BASELINE_PATH)
        # Ensure model paths are correct
        predictor = PitchPredictor(
            model_path=MODEL_PATH,
            target_encoder_path=TARGET_ENCODER_PATH,
            categorical_encoder_path=CATEGORICAL_ENCODER_PATH,
        )
        print("Model and Baseline loaded.")
    except Exception as e:
        log.error("Error loading model: %s. (Have you run src/train_model.py?)", e)
        wandb.finish(exit_code=1)
        return

    # --- Startup Skip Logic ---
    # To prevent spamming tweets for plays that already happened,
    # we mark all existing pitches in live games as 'processed' upon launch.
    print("Initializing: Skipping already completed pitches in live games...")
    live_game_pks = get_live_game_pks()
    for game_pk in live_game_pks:
        try:
            game_data = statsapi.get("game", {"gamePk": game_pk})
            if not game_data:
                log.warning("Startup skip: empty response for game %s", game_pk)
                continue
            all_plays = (
                game_data.get("liveData", {}).get("plays", {}).get("allPlays", [])
            )
            for play in all_plays:
                at_bat_idx = play.get("about", {}).get("atBatIndex")
                for event in play.get("playEvents", []):
                    if event.get("isPitch"):
                        processed_pitches.add((game_pk, at_bat_idx, event.get("index")))
        except Exception as e:
            log.error("Error during initialization for game %s: %s", game_pk, e)

    print(f"Initialization complete. Monitoring {len(live_game_pks)} games.")

    try:
        while True:
            live_game_pks = get_live_game_pks()
            if not live_game_pks:
                print(
                    f"Waiting for live games... (Interval: {POLLING_INTERVAL_SECONDS}s)"
                )
            else:
                for game_pk in live_game_pks:
                    try:
                        game_data = statsapi.get("game", {"gamePk": game_pk})
                        all_plays = (
                            game_data.get("liveData", {})
                            .get("plays", {})
                            .get("allPlays", [])
                        )
                        for play in all_plays:
                            at_bat_idx = play.get("about", {}).get("atBatIndex")
                            for event in play.get("playEvents", []):
                                if not event.get("isPitch"):
                                    continue
                                pitch_idx = event.get("index")
                                pitch_id = (game_pk, at_bat_idx, pitch_idx)
                                if pitch_id not in processed_pitches:
                                    processed_pitches.add(pitch_id)
                                    process_new_pitch(pitch_id, game_data, predictor)
                    except Exception as e:
                        log.error("Error polling game %s: %s", game_pk, e)

            _save_processed_cache()
            time.sleep(POLLING_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nTracker stopped.")
    finally:
        try:
            tweet_pct = (
                _session_tweets / _session_predictions if _session_predictions else 0.0
            )
            wandb.summary.update(
                {
                    "total_games": len(_session_games),
                    "total_predictions": _session_predictions,
                    "total_tweets": _session_tweets,
                    "tweet_pct": tweet_pct,
                }
            )

            # Pitch type distribution table (for pie chart)
            pitch_dist = wandb.Table(
                columns=["pitch_family", "count"],
                data=[[fam, cnt] for fam, cnt in _pitch_type_counts.items()],
            )
            # Error distribution table (for pie chart)
            error_dist = wandb.Table(
                columns=["pitch_family", "errors"],
                data=[[fam, cnt] for fam, cnt in _error_counts.items()],
            )
            wandb.log(
                {
                    "pitch_type_distribution": pitch_dist,
                    "error_distribution": error_dist,
                }
            )
        except Exception as e:
            log.warning("W&B session summary failed: %s", e)
        wandb.finish()


if __name__ == "__main__":
    main()
